/* Qwen3.5 CUDA kernels. sm_70+ (Tesla V100). fp32 accumulate. No bf16, no
   cp.async, no ldmatrix (that is sm_75). Prefill GEMM (T>1) can use Volta
   WMMA 16x16x16. Decode GEMV (T==1) has a bandwidth path: 128-bit loads, dp4a against an
   activation quantized to q8. Tall matrices use four rows per warp; short
   ones use gemv2r1_* (one row per warp). The original one-warp GEMV stays
   compiled for ODINFER_CUDA_GEMV=old. g_decode_pos is the token index when
   a kernel is launched with pos=-1 (CUDA graph replay).
   NVRTC does not ship mma.h or cuda_fp16.h, so WMMA and dp4a are inline
   PTX. NVRTC is given --gpu-architecture=compute_XX (CUDA 13 NVRTC
   defaults to sm_75 and will not JIT on sm_70).

   Quant layouts match packages/ggml/quant.odin. GEMV is one warp per output
   row. MMQ reuses the same accumulator over a tile of up to 8 tokens and
   writes token-major [T, rows]. */

/* Half conversion is inline so NVRTC does not need cuda_fp16.h. The pip
   wheel (nvidia-cuda-nvrtc) does not ship that header, and CUDA 12.9's
   libnvrtc-builtins does not register it as a builtin include. Round-to-
   nearest-even matches CUDA __float2half and Odin's f16 cast. */

/* Engine decode stores the token position here and passes pos=-1 so a CUDA
   graph can replay every token. A non-negative argument is the position. */
__device__ int g_decode_pos;

__device__ __forceinline__ int live_pos(int pos) {
    return pos >= 0 ? pos : g_decode_pos;
}

__device__ __forceinline__ float ld_f16(const unsigned char* p) {
    unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    unsigned sign = ((unsigned)u >> 15) & 1u;
    unsigned exp = ((unsigned)u >> 10) & 0x1fu;
    unsigned mant = (unsigned)u & 0x3ffu;
    unsigned out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            out = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        out = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return __uint_as_float(out);
}

__device__ __forceinline__ void st_f16(unsigned char* p, float v) {
    unsigned x = __float_as_uint(v);
    unsigned sign = (x >> 16) & 0x8000u;
    unsigned exp_f = (x >> 23) & 0xffu;
    unsigned mant = x & 0x7fffffu;
    unsigned short h;
    if (exp_f == 0xffu) {
        h = (unsigned short)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    } else if (exp_f == 0) {
        h = (unsigned short)sign;
    } else {
        int exp = (int)exp_f - 127 + 15;
        if (exp >= 31) {
            h = (unsigned short)(sign | 0x7c00u);
        } else if (exp <= 0) {
            if (exp < -10) {
                h = (unsigned short)sign;
            } else {
                mant |= 0x800000u;
                int shift = 14 - exp;
                unsigned t = mant >> shift;
                unsigned rem = mant & ((1u << shift) - 1u);
                unsigned halfway = 1u << (shift - 1);
                if (rem > halfway || (rem == halfway && (t & 1u))) t++;
                h = (unsigned short)(sign | t);
            }
        } else {
            unsigned t = ((unsigned)exp << 10) | (mant >> 13);
            unsigned rem = mant & 0x1fffu;
            if (rem > 0x1000u || (rem == 0x1000u && (t & 1u))) t++;
            if (t >= 0x7c00u) h = (unsigned short)(sign | 0x7c00u);
            else h = (unsigned short)(sign | t);
        }
    }
    p[0] = (unsigned char)(h & 0xff);
    p[1] = (unsigned char)(h >> 8);
}

__device__ __forceinline__ float ld_f32(const unsigned char* p) {
    unsigned u = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
    return __uint_as_float(u);
}

__device__ __forceinline__ unsigned ld_u32(const unsigned char* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

/* blockDim.x must be 128. Returns the sum to every thread. */
__device__ float block_sum_128(float v) {
    __shared__ float sm[128];
    int tid = threadIdx.x;
    sm[tid] = v;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float r = sm[0];
    __syncthreads();
    return r;
}

__device__ float block_max_128(float v) {
    __shared__ float sm[128];
    int tid = threadIdx.x;
    sm[tid] = v;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    float r = sm[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoid_stable(float x) {
    if (x >= 0.f) {
        float z = expf(-x);
        return 1.f / (1.f + z);
    }
    float z = expf(x);
    return z / (1.f + z);
}

__device__ __forceinline__ float softplus_stable(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return 0.f;
    return logf(1.f + expf(x));
}

__device__ void scale_min_k4(int j, const unsigned char* q, int* sc, int* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* ---- per-kind accumulators. lane is 0..31. acc[0..nt) is this lane's partial. ---- */

__device__ void acc_f32(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    const unsigned char* wr = wb + (size_t)row * (size_t)n * 4;
    for (int j = lane; j < n; j += 32) {
        float wv = ld_f32(wr + (size_t)j * 4);
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + j];
    }
}

__device__ void acc_f16(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    const unsigned char* wr = wb + (size_t)row * (size_t)n * 2;
    for (int j = lane; j < n; j += 32) {
        float wv = ld_f16(wr + (size_t)j * 2);
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + j];
    }
}

__device__ void acc_q8_0(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 34;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 34;
        float d = ld_f16(blk);
        float wv = d * (float)(signed char)blk[2 + lane];
        int k = (b << 5) + lane;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q4_0(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 18;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 18;
        float d = ld_f16(blk);
        unsigned char q = blk[2 + j];
        int qv = hi ? (int)(q >> 4) - 8 : (int)(q & 0xF) - 8;
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q4_1(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 20;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 20;
        float d = ld_f16(blk);
        float m = ld_f16(blk + 2);
        unsigned char q = blk[4 + j];
        int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv + m;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q5_0(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 22;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 22;
        float d = ld_f16(blk);
        unsigned qh = ld_u32(blk + 2);
        unsigned char qs = blk[6 + j];
        int qv;
        if (hi == 0) {
            unsigned xh = ((qh >> j) << 4) & 0x10;
            qv = (int)((qs & 0xF) | xh) - 16;
        } else {
            unsigned xh = (qh >> (j + 12)) & 0x10;
            qv = (int)((qs >> 4) | xh) - 16;
        }
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q5_1(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 24;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 24;
        float d = ld_f16(blk);
        float m = ld_f16(blk + 2);
        unsigned qh = ld_u32(blk + 4);
        unsigned char qs = blk[8 + j];
        int qv;
        if (hi == 0) {
            unsigned xh = ((qh >> j) << 4) & 0x10;
            qv = (int)((qs & 0xF) | xh);
        } else {
            unsigned xh = (qh >> (j + 12)) & 0x10;
            qv = (int)((qs >> 4) | xh);
        }
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv + m;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q4_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 8;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 144;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 144;
        float d = ld_f16(blk);
        float dmin = ld_f16(blk + 2);
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;
        for (int p = 0; p < 4; ++p) {
            int sc0, m0, sc1, m1;
            scale_min_k4(p * 2, scales, &sc0, &m0);
            scale_min_k4(p * 2 + 1, scales, &sc1, &m1);
            float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
            float d2 = d * (float)sc1, mm2 = dmin * (float)m1;
            unsigned char q = qs[p * 32 + lane];
            int klo = b * 256 + p * 64 + lane;
            int khi = klo + 32;
            float wl = (float)(q & 0xF) * d1 - mm1;
            float wh = (float)(q >> 4) * d2 - mm2;
            for (int a = 0; a < nt; ++a) {
                const float* xa = x + (size_t)a * (size_t)n;
                acc[a] += wl * xa[klo] + wh * xa[khi];
            }
        }
    }
}

__device__ void acc_q6_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 8;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 210;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const unsigned char* sc = blk + 192;
        float d = ld_f16(blk + 208);
        for (int half = 0; half < 2; ++half) {
            int l = lane;
            int is = l >> 4;
            int qloff = half * 64;
            int qhoff = half * 32;
            int scoff = half * 8;
            int yoff = half * 128;
            unsigned char qh1 = qh[qhoff + l];
            int q1 = (int)((ql[qloff + l] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[qloff + l + 32] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[qloff + l] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[qloff + l + 32] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;
            float s0 = d * (float)(signed char)sc[scoff + is + 0];
            float s1 = d * (float)(signed char)sc[scoff + is + 2];
            float s2 = d * (float)(signed char)sc[scoff + is + 4];
            float s3 = d * (float)(signed char)sc[scoff + is + 6];
            int k0 = b * 256 + yoff + l;
            for (int a = 0; a < nt; ++a) {
                const float* xa = x + (size_t)a * (size_t)n;
                acc[a] += s0 * (float)q1 * xa[k0]
                        + s1 * (float)q2 * xa[k0 + 32]
                        + s2 * (float)q3 * xa[k0 + 64]
                        + s3 * (float)q4 * xa[k0 + 96];
            }
        }
    }
}

__device__ void dequant_q2k(const unsigned char* blk, float* dst) {
    float d = ld_f16(blk + 80);
    float dmin = ld_f16(blk + 82);
    int is = 0, qoff = 0, yoff = 0;
    for (int half = 0; half < 2; ++half) {
        int shift = 0;
        for (int s = 0; s < 4; ++s) {
            unsigned char sc = blk[is++];
            float dl = d * (float)(sc & 0xF);
            float ml = dmin * (float)(sc >> 4);
            for (int l = 0; l < 16; ++l)
                dst[yoff + l] = dl * (float)((blk[16 + qoff + l] >> shift) & 3) - ml;
            sc = blk[is++];
            dl = d * (float)(sc & 0xF);
            ml = dmin * (float)(sc >> 4);
            for (int l = 0; l < 16; ++l)
                dst[yoff + 16 + l] = dl * (float)((blk[16 + qoff + 16 + l] >> shift) & 3) - ml;
            shift += 2;
            yoff += 32;
        }
        qoff += 32;
    }
}

__device__ void dequant_q3k(const unsigned char* blk, float* dst) {
    const unsigned KMASK1 = 0x03030303u;
    const unsigned KMASK2 = 0x0f0f0f0fu;
    const unsigned char* hm = blk;
    const unsigned char* qs = blk + 32;
    const unsigned char* scales = blk + 96;
    float d = ld_f16(blk + 108);
    unsigned aux0 = ld_u32(scales);
    unsigned aux1 = ld_u32(scales + 4);
    unsigned tmp = ld_u32(scales + 8);
    unsigned aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    unsigned aux3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux0 = (aux0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    unsigned aux[4];
    aux[0] = aux0; aux[1] = aux1; aux[2] = aux2; aux[3] = aux3;
    int yoff = 0, qoff = 0, is = 0;
    unsigned m = 1;
    for (int half = 0; half < 2; ++half) {
        int shift = 0;
        for (int s = 0; s < 4; ++s) {
            int sc = (int)(signed char)((aux[is / 4] >> (8 * (is & 3))) & 0xff);
            is += 1;
            float dl = d * (float)(sc - 32);
            for (int l = 0; l < 16; ++l) {
                int q = (int)((qs[qoff + l] >> shift) & 3) - ((hm[l] & m) == 0 ? 4 : 0);
                dst[yoff + l] = dl * (float)q;
            }
            sc = (int)(signed char)((aux[is / 4] >> (8 * (is & 3))) & 0xff);
            is += 1;
            dl = d * (float)(sc - 32);
            for (int l = 0; l < 16; ++l) {
                int q = (int)((qs[qoff + 16 + l] >> shift) & 3) - ((hm[l + 16] & m) == 0 ? 4 : 0);
                dst[yoff + 16 + l] = dl * (float)q;
            }
            shift += 2;
            m <<= 1;
            yoff += 32;
        }
        qoff += 32;
    }
}

__device__ void dequant_q5k(const unsigned char* blk, float* dst) {
    float d = ld_f16(blk);
    float dmin = ld_f16(blk + 2);
    const unsigned char* scales = blk + 4;
    const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    int yoff = 0, is = 0, qloff = 0;
    unsigned u1 = 1, u2 = 2;
    for (int p = 0; p < 4; ++p) {
        int sc0, m0, sc1, m1;
        scale_min_k4(is, scales, &sc0, &m0);
        scale_min_k4(is + 1, scales, &sc1, &m1);
        float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
        float d2 = d * (float)sc1, mm2 = dmin * (float)m1;
        for (int l = 0; l < 32; ++l) {
            int v = (int)(ql[qloff + l] & 0xF) + ((qh[l] & u1) == 0 ? 0 : 16);
            dst[yoff + l] = d1 * (float)v - mm1;
        }
        for (int l = 0; l < 32; ++l) {
            int v = (int)(ql[qloff + l] >> 4) + ((qh[l] & u2) == 0 ? 0 : 16);
            dst[yoff + 32 + l] = d2 * (float)v - mm2;
        }
        qloff += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        yoff += 64;
    }
}

__device__ void dequant_q8k(const unsigned char* blk, float* dst) {
    float d = ld_f32(blk);
    for (int j = 0; j < 256; ++j) dst[j] = d * (float)(signed char)blk[4 + j];
}

__device__ void acc_dequant(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc, int block_elems, int block_bytes, int kind) {
    __shared__ float w[256];
    int nb = n / block_elems;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * (size_t)block_bytes;
    for (int b = 0; b < nb; ++b) {
        if (lane == 0) {
            const unsigned char* blk = wr + (size_t)b * (size_t)block_bytes;
            if (kind == 0) dequant_q2k(blk, w);
            else if (kind == 1) dequant_q3k(blk, w);
            else if (kind == 2) dequant_q5k(blk, w);
            else dequant_q8k(blk, w);
        }
        __syncthreads();
        for (int j = lane; j < block_elems; j += 32) {
            float wv = w[j];
            int k = b * block_elems + j;
            for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
        }
        __syncthreads();
    }
}

__device__ void acc_q2_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 84, 0);
}
__device__ void acc_q3_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 110, 1);
}
__device__ void acc_q5_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 176, 2);
}
__device__ void acc_q8_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 292, 3);
}

#define DEF_ROW(tag) \
extern "C" __global__ void gemv_##tag(const unsigned char* wb, const float* x, float* out, int n, int d) { \
    int row = (int)blockIdx.x; \
    int lane = (int)threadIdx.x; \
    if (row >= d) return; \
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
    acc_##tag(wb, row, x, n, 1, lane, acc); \
    float s = warp_sum(acc[0]); \
    if (lane == 0) out[row] = s; \
} \
extern "C" __global__ void mmq_##tag(const unsigned char* wb, const float* x, float* out, int n, int d, int T) { \
    int row = (int)blockIdx.x; \
    int lane = (int)threadIdx.x; \
    if (row >= d) return; \
    for (int t0 = 0; t0 < T; t0 += 8) { \
        int nt = T - t0; \
        if (nt > 8) nt = 8; \
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
        acc_##tag(wb, row, x + (size_t)t0 * (size_t)n, n, nt, lane, acc); \
        float red0 = warp_sum(acc[0]); \
        float red1 = warp_sum(acc[1]); \
        float red2 = warp_sum(acc[2]); \
        float red3 = warp_sum(acc[3]); \
        float red4 = warp_sum(acc[4]); \
        float red5 = warp_sum(acc[5]); \
        float red6 = warp_sum(acc[6]); \
        float red7 = warp_sum(acc[7]); \
        if (lane == 0) { \
            float red[8] = {red0, red1, red2, red3, red4, red5, red6, red7}; \
            for (int a = 0; a < nt; ++a) out[((size_t)t0 + a) * (size_t)d + row] = red[a]; \
        } \
    } \
}

DEF_ROW(f32)
DEF_ROW(f16)
DEF_ROW(q8_0)
DEF_ROW(q4_0)
DEF_ROW(q4_1)
DEF_ROW(q5_0)
DEF_ROW(q5_1)
DEF_ROW(q4_k)
DEF_ROW(q6_k)
DEF_ROW(q2_k)
DEF_ROW(q3_k)
DEF_ROW(q5_k)
DEF_ROW(q8_k)

/* ---- elementwise / attention. block 128 where a reduction is used. ---- */

extern "C" __global__ void rmsnorm_k(const float* inp, float* outp, const float* w, int size, float eps) {
    int grp = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* xi = inp + (size_t)grp * (size_t)size;
    float* oi = outp + (size_t)grp * (size_t)size;
    float ss = 0.f;
    for (int j = tid; j < size; j += 128) ss += xi[j] * xi[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)size + eps);
    for (int j = tid; j < size; j += 128) oi[j] = w[j] * (inv * xi[j]);
}

extern "C" __global__ void qnorm_extract(const float* qproj, float* q, const float* qn, int n, float eps) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* src = qproj + (size_t)h * (size_t)(2 * n);
    float* dst = q + (size_t)h * (size_t)n;
    float ss = 0.f;
    for (int j = tid; j < n; j += 128) ss += src[j] * src[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)n + eps);
    for (int j = tid; j < n; j += 128) dst[j] = qn[j] * (inv * src[j]);
}

extern "C" __global__ void rope_partial(float* vec, int head_dim, int pos, float rope_freq, int rotary_dim, int n_heads) {
    int half = rotary_dim >> 1;
    int total = n_heads * half;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= total) return;
    int h = gid / half;
    int i = gid - h * half;
    float* q = vec + (size_t)h * (size_t)head_dim;
    float freq = 1.f / powf(rope_freq, (float)i / (float)half);
    float ang = (float)live_pos(pos) * freq;
    float c = cosf(ang), s = sinf(ang);
    float x0 = q[i], y0 = q[i + half];
    q[i] = x0 * c - y0 * s;
    q[i + half] = x0 * s + y0 * c;
}

extern "C" __global__ void attn_gate(float* xb3, const float* qproj, int head_dim, int n_heads) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = n_heads * head_dim;
    if (gid >= n) return;
    int h = gid / head_dim;
    int i = gid - h * head_dim;
    float g = qproj[(size_t)h * (size_t)(2 * head_dim) + head_dim + i];
    xb3[(size_t)h * (size_t)head_dim + i] *= sigmoid_stable(g);
}

extern "C" __global__ void swiglu(float* hb, const float* hb2, int n) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) return;
    float v = hb[gid];
    hb[gid] = (v * sigmoid_stable(v)) * hb2[gid];
}

extern "C" __global__ void residual(float* x, const float* y, int n) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) return;
    x[gid] += y[gid];
}

extern "C" __global__ void axpy_k(float* y, const float* x, float a, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] += a * x[i];
}

/* f16 KV: head-major [kvh, seq_len, head_dim], little-endian half. */
extern "C" __global__ void store_kv_f16(const float* k, const float* v, unsigned char* kc, unsigned char* vc,
                                        int head_dim, int seq_len, int pos, int n_kv) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = n_kv * head_dim;
    if (gid >= n) return;
    int kvh = gid / head_dim;
    int i = gid - kvh * head_dim;
    size_t dst = (((size_t)kvh * (size_t)seq_len + (size_t)live_pos(pos)) * (size_t)head_dim + (size_t)i) * 2;
    st_f16(kc + dst, k[gid]);
    st_f16(vc + dst, v[gid]);
}

extern "C" __global__ void attention_f16(const float* q, const unsigned char* kc, const unsigned char* vc, float* xb3, float* scores,
                                         int head_dim, int seq_len, int kv_mul, int pos) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int kvh = h / kv_mul;
    const float* qh = q + (size_t)h * (size_t)head_dim;
    const unsigned char* kbase = kc + (size_t)kvh * (size_t)seq_len * (size_t)head_dim * 2;
    const unsigned char* vbase = vc + (size_t)kvh * (size_t)seq_len * (size_t)head_dim * 2;
    float scale = rsqrtf((float)head_dim);
    int npos = live_pos(pos) + 1;
    float* sc = scores + (size_t)h * (size_t)seq_len;
    for (int t = tid; t < npos; t += 128) {
        const unsigned char* kk = kbase + (size_t)t * (size_t)head_dim * 2;
        float s = 0.f;
        for (int i = 0; i < head_dim; ++i) s += qh[i] * ld_f16(kk + (size_t)i * 2);
        sc[t] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;
    for (int t = tid; t < npos; t += 128) lm = fmaxf(lm, sc[t]);
    float m = block_max_128(lm);
    float ls = 0.f;
    for (int t = tid; t < npos; t += 128) {
        float e = expf(sc[t] - m);
        sc[t] = e;
        ls += e;
    }
    float sum = block_sum_128(ls);
    float inv = 1.f / sum;
    for (int i = tid; i < head_dim; i += 128) {
        float acc = 0.f;
        for (int t = 0; t < npos; ++t) {
            const unsigned char* vv = vbase + ((size_t)t * (size_t)head_dim + (size_t)i) * 2;
            acc += sc[t] * ld_f16(vv);
        }
        xb3[(size_t)h * (size_t)head_dim + i] = acc * inv;
    }
}

extern "C" __global__ void store_kv_q8(const float* k, const float* v, unsigned char* kc, unsigned char* vc,
                                       int nblocks, int pos_bytes, int pos) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= nblocks) return;
    const float* k32 = k + (size_t)gid * 32;
    const float* v32 = v + (size_t)gid * 32;
    float kamax = 0.f, vamax = 0.f;
    for (int j = 0; j < 32; ++j) {
        kamax = fmaxf(kamax, fabsf(k32[j]));
        vamax = fmaxf(vamax, fabsf(v32[j]));
    }
    float kd = kamax / 127.f, vd = vamax / 127.f;
    float kid = kd != 0.f ? 1.f / kd : 0.f;
    float vid = vd != 0.f ? 1.f / vd : 0.f;
    int pslot = live_pos(pos);
    unsigned char* kb = kc + (size_t)pslot * (size_t)pos_bytes + (size_t)gid * 34;
    unsigned char* vb = vc + (size_t)pslot * (size_t)pos_bytes + (size_t)gid * 34;
    st_f16(kb, kd);
    st_f16(vb, vd);
    for (int j = 0; j < 32; ++j) {
        kb[2 + j] = (unsigned char)(signed char)(int)roundf(k32[j] * kid);
        vb[2 + j] = (unsigned char)(signed char)(int)roundf(v32[j] * vid);
    }
}

extern "C" __global__ void attention_q8(const float* q, const unsigned char* kc, const unsigned char* vc, float* xb3, float* scores,
                                        int head_dim, int pos_bytes, int kv_mul, int pos, int seq_len) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int nb = head_dim >> 5;
    int kvh = h / kv_mul;
    int hb = kvh * nb;
    const float* qh = q + (size_t)h * (size_t)head_dim;
    float scale = rsqrtf((float)head_dim);
    int npos = live_pos(pos) + 1;
    float* sc = scores + (size_t)h * (size_t)seq_len;
    for (int t = tid; t < npos; t += 128) {
        const unsigned char* kb = kc + (size_t)t * (size_t)pos_bytes + (size_t)hb * 34;
        float s = 0.f;
        for (int b = 0; b < nb; ++b) {
            const unsigned char* bb = kb + (size_t)b * 34;
            float d = ld_f16(bb);
            for (int j = 0; j < 32; ++j) s += qh[b * 32 + j] * (d * (float)(signed char)bb[2 + j]);
        }
        sc[t] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;
    for (int t = tid; t < npos; t += 128) lm = fmaxf(lm, sc[t]);
    float m = block_max_128(lm);
    float ls = 0.f;
    for (int t = tid; t < npos; t += 128) {
        float e = expf(sc[t] - m);
        sc[t] = e;
        ls += e;
    }
    float sum = block_sum_128(ls);
    float inv = 1.f / sum;
    for (int i = tid; i < head_dim; i += 128) {
        int elem = kvh * head_dim + i;
        int bb = elem >> 5;
        int jj = elem - (bb << 5);
        float acc = 0.f;
        for (int t = 0; t < npos; ++t) {
            const unsigned char* vb = vc + (size_t)t * (size_t)pos_bytes + (size_t)bb * 34;
            float d = ld_f16(vb);
            acc += sc[t] * (d * (float)(signed char)vb[2 + jj]);
        }
        xb3[(size_t)h * (size_t)head_dim + i] = acc * inv;
    }
}

extern "C" __global__ void store_kv_q8v(const float* k, const float* v, unsigned char* kc, unsigned char* vc,
                                        int nblocks, int k_pos_bytes, int v_pos_bytes, int pos) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= nblocks) return;
    const float* k32 = k + (size_t)gid * 32;
    const float* v32 = v + (size_t)gid * 32;
    float kamax = 0.f;
    for (int j = 0; j < 32; ++j) kamax = fmaxf(kamax, fabsf(k32[j]));
    float kd = kamax / 127.f;
    float kid = kd != 0.f ? 1.f / kd : 0.f;
    int pslot = live_pos(pos);
    unsigned char* kb = kc + (size_t)pslot * (size_t)k_pos_bytes + (size_t)gid * 34;
    st_f16(kb, kd);
    for (int j = 0; j < 32; ++j) kb[2 + j] = (unsigned char)(signed char)(int)roundf(k32[j] * kid);

    float vamax = fabsf(v32[0]);
    int jmax = 0;
    for (int j = 1; j < 32; ++j) {
        float a = fabsf(v32[j]);
        if (a > vamax) { vamax = a; jmax = j; }
    }
    float amax2 = 0.f;
    for (int j = 0; j < 32; ++j) if (j != jmax) amax2 = fmaxf(amax2, fabsf(v32[j]));
    float vd = amax2 / 127.f;
    float vid = vd != 0.f ? 1.f / vd : 0.f;
    unsigned char* vb = vc + (size_t)pslot * (size_t)v_pos_bytes + (size_t)gid * 40;
    for (int j = 0; j < 40; ++j) vb[j] = 0;
    st_f16(vb, vd);
    vb[2] = (unsigned char)jmax;
    st_f16(vb + 4, v32[jmax]);
    for (int j = 0; j < 32; ++j) {
        if (j != jmax) vb[6 + j] = (unsigned char)(signed char)(int)roundf(v32[j] * vid);
    }
}

extern "C" __global__ void attention_q8v(const float* q, const unsigned char* kc, const unsigned char* vc, float* xb3, float* scores,
                                         int head_dim, int k_pos_bytes, int v_pos_bytes, int kv_mul, int pos, int seq_len) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int nb = head_dim >> 5;
    int kvh = h / kv_mul;
    int hb = kvh * nb;
    const float* qh = q + (size_t)h * (size_t)head_dim;
    float scale = rsqrtf((float)head_dim);
    int npos = live_pos(pos) + 1;
    float* sc = scores + (size_t)h * (size_t)seq_len;
    for (int t = tid; t < npos; t += 128) {
        const unsigned char* kb = kc + (size_t)t * (size_t)k_pos_bytes + (size_t)hb * 34;
        float s = 0.f;
        for (int b = 0; b < nb; ++b) {
            const unsigned char* bb = kb + (size_t)b * 34;
            float d = ld_f16(bb);
            for (int j = 0; j < 32; ++j) s += qh[b * 32 + j] * (d * (float)(signed char)bb[2 + j]);
        }
        sc[t] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;
    for (int t = tid; t < npos; t += 128) lm = fmaxf(lm, sc[t]);
    float m = block_max_128(lm);
    float ls = 0.f;
    for (int t = tid; t < npos; t += 128) {
        float e = expf(sc[t] - m);
        sc[t] = e;
        ls += e;
    }
    float sum = block_sum_128(ls);
    float inv = 1.f / sum;
    for (int i = tid; i < head_dim; i += 128) {
        int elem = kvh * head_dim + i;
        int bb = elem >> 5;
        int jj = elem - (bb << 5);
        float acc = 0.f;
        for (int t = 0; t < npos; ++t) {
            const unsigned char* vb = vc + (size_t)t * (size_t)v_pos_bytes + (size_t)bb * 40;
            float d = ld_f16(vb);
            int idx = vb[2];
            float ov = ld_f16(vb + 4);
            float vv = (jj == idx) ? ov : d * (float)(signed char)vb[6 + jj];
            acc += sc[t] * vv;
        }
        xb3[(size_t)h * (size_t)head_dim + i] = acc * inv;
    }
}

extern "C" __global__ void conv1d_step(float* qkv_out, const float* qkv_in, float* conv_state, const float* weight,
                                       int conv_dim, int kernel) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= conv_dim) return;
    int km1 = kernel - 1;
    float acc = weight[c * kernel + km1] * qkv_in[c];
    for (int i = 0; i < km1; ++i) acc += weight[c * kernel + i] * conv_state[c * km1 + i];
    qkv_out[c] = acc * sigmoid_stable(acc);
    for (int i = 0; i + 1 < km1; ++i) conv_state[c * km1 + i] = conv_state[c * km1 + i + 1];
    if (km1 > 0) conv_state[c * km1 + km1 - 1] = qkv_in[c];
}

extern "C" __global__ void l2norm_scale(float* q_out, float* k_out, const float* q_in, const float* k_in,
                                        int head_k_dim, float eps, float scale) {
    int kh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int n = head_k_dim;
    const float* qi = q_in + (size_t)kh * (size_t)n;
    const float* ki = k_in + (size_t)kh * (size_t)n;
    float* qo = q_out + (size_t)kh * (size_t)n;
    float* ko = k_out + (size_t)kh * (size_t)n;
    float sq = 0.f, sk = 0.f;
    for (int j = tid; j < n; j += 128) {
        sq += qi[j] * qi[j];
        sk += ki[j] * ki[j];
    }
    sq = block_sum_128(sq);
    sk = block_sum_128(sk);
    float iq = 1.f / fmaxf(sqrtf(sq), eps);
    float ik = 1.f / fmaxf(sqrtf(sk), eps);
    for (int j = tid; j < n; j += 128) {
        qo[j] = qi[j] * iq * scale;
        ko[j] = ki[j] * ik;
    }
}

extern "C" __global__ void delta_recurrent(float* out, float* state_all, const float* q_lin, const float* k_lin,
                                           const float* v_vec, const float* b_vec, const float* a_vec,
                                           const float* a_decay, const float* dt_bias,
                                           int head_k_dim, int head_v_dim, int n_k_heads) {
    int vh = (int)blockIdx.x;
    int hkd = head_k_dim, hvd = head_v_dim;
    int kh = vh % n_k_heads;
    float* state = state_all + (size_t)vh * (size_t)hkd * (size_t)hvd;
    const float* qt = q_lin + (size_t)kh * (size_t)hkd;
    const float* kt = k_lin + (size_t)kh * (size_t)hkd;
    const float* vt = v_vec + (size_t)vh * (size_t)hvd;
    float beta = sigmoid_stable(b_vec[vh]);
    float g_decay = expf(a_decay[vh] * softplus_stable(a_vec[vh] + dt_bias[vh]));
    for (int j = (int)threadIdx.x; j < hvd; j += (int)blockDim.x) {
        for (int i = 0; i < hkd; ++i) state[(size_t)i * (size_t)hvd + j] *= g_decay;
        float kv_mem = 0.f;
        for (int i = 0; i < hkd; ++i) kv_mem += state[(size_t)i * (size_t)hvd + j] * kt[i];
        float delta = (vt[j] - kv_mem) * beta;
        for (int i = 0; i < hkd; ++i) state[(size_t)i * (size_t)hvd + j] += kt[i] * delta;
        float o = 0.f;
        for (int i = 0; i < hkd; ++i) o += state[(size_t)i * (size_t)hvd + j] * qt[i];
        out[(size_t)vh * (size_t)hvd + j] = o;
    }
}

extern "C" __global__ void rmsnorm_gated(float* out, const float* inp, const float* weight, const float* gate, int size, float eps) {
    int vh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* xi = inp + (size_t)vh * (size_t)size;
    float* oi = out + (size_t)vh * (size_t)size;
    const float* zi = gate + (size_t)vh * (size_t)size;
    float ss = 0.f;
    for (int j = tid; j < size; j += 128) ss += xi[j] * xi[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)size + eps);
    for (int j = tid; j < size; j += 128) {
        float normed = weight[j] * (inv * xi[j]);
        float gz = zi[j] * sigmoid_stable(zi[j]);
        oi[j] = normed * gz;
    }
}

/* n <= 256. One thread. Matches moe_ffn: softmax, top-k, clamp, expert_w_scale. */
extern "C" __global__ void moe_softmax_topk(const float* logits, int* sel, float* weights, int n, int k, float clamp_v, float scale) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s[256];
    int picked[256];
    float m = logits[0];
    for (int i = 1; i < n; ++i) m = fmaxf(m, logits[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        s[i] = expf(logits[i] - m);
        sum += s[i];
        picked[i] = 0;
    }
    float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) s[i] *= inv;
    for (int t = 0; t < k; ++t) {
        int best = 0;
        float bv = -1e30f;
        for (int j = 0; j < n; ++j) {
            if (!picked[j] && s[j] > bv) { bv = s[j]; best = j; }
        }
        sel[t] = best;
        picked[best] = 1;
        weights[t] = s[best];
    }
    float wsum = 0.f;
    for (int t = 0; t < k; ++t) wsum += weights[t];
    if (wsum < clamp_v) wsum = clamp_v;
    for (int t = 0; t < k; ++t) {
        weights[t] /= wsum;
        if (scale != 1.f && scale != 0.f) weights[t] *= scale;
    }
}

/* ---- Volta tensor-core GEMM -------------------------------------------------
   Y[T, d] = X[T, n] * W[d, n], fp32 accumulate. W is dequantized into fp16
   tiles in shared memory, X is narrowed to fp16, then one warp issues
   wmma 16x16x16. A block owns 64 tokens by 64 rows. T==1 never launches this.
   The hardware op under the WMMA is HMMA.884; the PTX we emit is the sm_70
   WMMA form, which NVRTC 12.9 accepts for compute_70. */

__device__ __forceinline__ unsigned short f32_to_f16(float v) {
    unsigned short h;
    st_f16((unsigned char*)&h, v);
    return h;
}

__device__ __forceinline__ void wmma_load_a(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3,
                                            unsigned& a4, unsigned& a5, unsigned& a6, unsigned& a7,
                                            const unsigned short* p, int stride) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4), "=r"(a5), "=r"(a6), "=r"(a7)
        : "l"(p), "r"(stride));
}

__device__ __forceinline__ void wmma_load_b(unsigned& b0, unsigned& b1, unsigned& b2, unsigned& b3,
                                            unsigned& b4, unsigned& b5, unsigned& b6, unsigned& b7,
                                            const unsigned short* p, int stride) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;\n"
        : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3), "=r"(b4), "=r"(b5), "=r"(b6), "=r"(b7)
        : "l"(p), "r"(stride));
}

__device__ __forceinline__ void wmma_mma(float& c0, float& c1, float& c2, float& c3,
                                         float& c4, float& c5, float& c6, float& c7,
                                         unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned a4, unsigned a5, unsigned a6, unsigned a7,
                                         unsigned b0, unsigned b1, unsigned b2, unsigned b3,
                                         unsigned b4, unsigned b5, unsigned b6, unsigned b7) {
    asm volatile(
        "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3), "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7));
}

__device__ __forceinline__ void wmma_store(float* p, int stride,
                                           float c0, float c1, float c2, float c3,
                                           float c4, float c5, float c6, float c7) {
    asm volatile(
        "wmma.store.d.sync.aligned.row.m16n16k16.f32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8}, %9;\n"
        :: "l"(p), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(stride));
}

__device__ float w_f32(const unsigned char* wb, int row, int k, int n) {
    return ld_f32(wb + ((size_t)row * (size_t)n + (size_t)k) * 4);
}
__device__ float w_f16(const unsigned char* wb, int row, int k, int n) {
    return ld_f16(wb + ((size_t)row * (size_t)n + (size_t)k) * 2);
}
__device__ float w_q8_0(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 34;
    return ld_f16(blk) * (float)(signed char)blk[2 + (k & 31)];
}
__device__ float w_q4_0(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 18;
    unsigned char q = blk[2 + j];
    int qv = hi ? (int)(q >> 4) - 8 : (int)(q & 0xF) - 8;
    return ld_f16(blk) * (float)qv;
}
__device__ float w_q4_1(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 20;
    unsigned char q = blk[4 + j];
    int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
    return ld_f16(blk) * (float)qv + ld_f16(blk + 2);
}
__device__ float w_q5_0(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 22;
    unsigned qh = ld_u32(blk + 2);
    unsigned char qs = blk[6 + j];
    int qv;
    if (hi == 0) {
        unsigned xh = ((qh >> j) << 4) & 0x10u;
        qv = (int)((qs & 0xF) | xh) - 16;
    } else {
        unsigned xh = (qh >> (j + 12)) & 0x10u;
        qv = (int)((qs >> 4) | xh) - 16;
    }
    return ld_f16(blk) * (float)qv;
}
__device__ float w_q5_1(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 24;
    unsigned qh = ld_u32(blk + 4);
    unsigned char qs = blk[8 + j];
    int qv;
    if (hi == 0) {
        unsigned xh = ((qh >> j) << 4) & 0x10u;
        qv = (int)((qs & 0xF) | xh);
    } else {
        unsigned xh = (qh >> (j + 12)) & 0x10u;
        qv = (int)((qs >> 4) | xh);
    }
    return ld_f16(blk) * (float)qv + ld_f16(blk + 2);
}
__device__ float w_q4_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 144;
    int inner = k & 255;
    int p = inner >> 6;
    int within = inner & 63;
    int hi = within >> 5;
    int lane = within & 31;
    int sc, m;
    scale_min_k4(p * 2 + hi, blk + 4, &sc, &m);
    unsigned char q = blk[16 + p * 32 + lane];
    int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
    float d = ld_f16(blk);
    float dmin = ld_f16(blk + 2);
    return (float)qv * (d * (float)sc) - (dmin * (float)m);
}
__device__ float w_q6_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 210;
    int inner = k & 255;
    int half = inner >> 7;
    int rem = inner & 127;
    int group = rem >> 5;
    int l = rem & 31;
    const unsigned char* ql = blk;
    const unsigned char* qh = blk + 128;
    const unsigned char* scb = blk + 192;
    int qloff = half * 64;
    int qhoff = half * 32;
    int scoff = half * 8;
    unsigned char qh1 = qh[qhoff + l];
    int qv, sc_i;
    int is = l >> 4;
    if (group == 0) {
        qv = (int)((ql[qloff + l] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
        sc_i = scoff + is;
    } else if (group == 1) {
        qv = (int)((ql[qloff + l + 32] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
        sc_i = scoff + is + 2;
    } else if (group == 2) {
        qv = (int)((ql[qloff + l] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
        sc_i = scoff + is + 4;
    } else {
        qv = (int)((ql[qloff + l + 32] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;
        sc_i = scoff + is + 6;
    }
    float d = ld_f16(blk + 208);
    return d * (float)(signed char)scb[sc_i] * (float)qv;
}
__device__ float w_q8_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 292;
    return ld_f32(blk) * (float)(signed char)blk[4 + (k & 255)];
}
__device__ float w_q2_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 84;
    int inner = k & 255;
    int group = inner >> 5;
    int half = group >> 2;
    int s = group & 3;
    int shift = s * 2;
    int qoff = half * 32;
    int within = inner & 31;
    int l = within & 15;
    int hi = within >> 4;
    unsigned char sc = blk[group * 2 + hi];
    float d = ld_f16(blk + 80);
    float dmin = ld_f16(blk + 82);
    float dl = d * (float)(sc & 0xF);
    float ml = dmin * (float)(sc >> 4);
    unsigned char byte = blk[16 + qoff + hi * 16 + l];
    return dl * (float)((byte >> shift) & 3) - ml;
}
__device__ float w_q3_k(const unsigned char* wb, int row, int k, int n) {
    const unsigned KMASK1 = 0x03030303u;
    const unsigned KMASK2 = 0x0f0f0f0fu;
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 110;
    int inner = k & 255;
    unsigned aux0 = ld_u32(blk + 96);
    unsigned aux1 = ld_u32(blk + 100);
    unsigned tmp = ld_u32(blk + 104);
    unsigned aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    unsigned aux3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux0 = (aux0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    unsigned aux[4];
    aux[0] = aux0; aux[1] = aux1; aux[2] = aux2; aux[3] = aux3;
    int group = inner >> 5;
    int half = group >> 2;
    int s = group & 3;
    int shift = s * 2;
    int qoff = half * 32;
    int within = inner & 31;
    int l = within & 15;
    int hi = within >> 4;
    int is = group * 2 + hi;
    int sc = (int)(signed char)((aux[is / 4] >> (8 * (is & 3))) & 0xff);
    float dl = ld_f16(blk + 108) * (float)(sc - 32);
    unsigned mbit = 1u << group;
    const unsigned char* hm = blk + (hi ? 16 : 0);
    const unsigned char* qs = blk + 32;
    int q = (int)((qs[qoff + hi * 16 + l] >> shift) & 3) - ((hm[l] & mbit) == 0 ? 4 : 0);
    return dl * (float)q;
}
__device__ float w_q5_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 176;
    int inner = k & 255;
    int p = inner >> 6;
    int within = inner & 63;
    int hi = within >> 5;
    int l = within & 31;
    int sc, m;
    scale_min_k4(p * 2 + hi, blk + 4, &sc, &m);
    float d = ld_f16(blk);
    float dmin = ld_f16(blk + 2);
    const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    unsigned u = 1u << (2 * p + hi);
    int nib = hi ? (int)(ql[p * 32 + l] >> 4) : (int)(ql[p * 32 + l] & 0xF);
    int v = nib + ((qh[l] & u) ? 16 : 0);
    return (d * (float)sc) * (float)v - (dmin * (float)m);
}

template<int KIND>
__device__ float tc_w(const unsigned char* wb, int row, int k, int n) {
    if (KIND == 0) return w_f32(wb, row, k, n);
    if (KIND == 1) return w_f16(wb, row, k, n);
    if (KIND == 2) return w_q4_0(wb, row, k, n);
    if (KIND == 3) return w_q4_1(wb, row, k, n);
    if (KIND == 4) return w_q5_0(wb, row, k, n);
    if (KIND == 5) return w_q5_1(wb, row, k, n);
    if (KIND == 6) return w_q8_0(wb, row, k, n);
    if (KIND == 7) return w_q2_k(wb, row, k, n);
    if (KIND == 8) return w_q3_k(wb, row, k, n);
    if (KIND == 9) return w_q4_k(wb, row, k, n);
    if (KIND == 10) return w_q5_k(wb, row, k, n);
    if (KIND == 11) return w_q6_k(wb, row, k, n);
    return w_q8_k(wb, row, k, n);
}

/* KIND matches the order in bind_kernels. Block: 512 threads, 16 warps.
   grid: (ceil(d/64), ceil(T/64)). */
template<int KIND>
__device__ void tc_run(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 32;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int t0 = (int)blockIdx.y * BM;
    const int n0 = (int)blockIdx.x * BN;
    if (t0 >= T || n0 >= d) return;

    __shared__ __align__(16) unsigned char sm[16 * 256 * 4];
    unsigned short* Xs = (unsigned short*)sm;
    unsigned short* Ws = (unsigned short*)(sm + BM * BK * 2);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        for (int i = tid; i < BM * BK; i += 512) {
            int t = i / BK;
            int kk = i - t * BK;
            float v = 0.f;
            if (t0 + t < T && k0 + kk < n) v = x[((size_t)t0 + t) * (size_t)n + (size_t)k0 + kk];
            Xs[i] = f32_to_f16(v);
        }
        for (int i = tid; i < BN * BK; i += 512) {
            int row = i / BK;
            int kk = i - row * BK;
            float v = 0.f;
            if (n0 + row < d && k0 + kk < n) v = tc_w<KIND>(wb, n0 + row, k0 + kk, n);
            Ws[i] = f32_to_f16(v);
        }
        __syncthreads();
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned a0, a1, a2, a3, a4, a5, a6, a7;
            unsigned b0, b1, b2, b3, b4, b5, b6, b7;
            wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, Xs + (wm * 16) * BK + kk, BK);
            wmma_load_b(b0, b1, b2, b3, b4, b5, b6, b7, Ws + (wn * 16) * BK + kk, BK);
            wmma_mma(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
        }
        __syncthreads();
    }
    float* tile = ((float*)sm) + warp * 256;
    wmma_store(tile, 16, c0, c1, c2, c3, c4, c5, c6, c7);
    __syncthreads();
    for (int idx = tid; idx < 16 * 256; idx += 512) {
        int w = idx >> 8;
        int e = idx & 255;
        int i = e >> 4;
        int j = e & 15;
        int tw = w >> 2;
        int nw = w & 3;
        int t = t0 + tw * 16 + i;
        int row = n0 + nw * 16 + j;
        if (t < T && row < d) out[(size_t)t * (size_t)d + row] = ((float*)sm)[idx];
    }
}

#define DEF_TC(tag, kind) \
extern "C" __global__ void tc_##tag(const unsigned char* wb, const float* x, float* out, int n, int d, int T) { \
    tc_run<kind>(wb, x, out, n, d, T); \
}

DEF_TC(f32, 0)
DEF_TC(f16, 1)
DEF_TC(q4_0, 2)
DEF_TC(q4_1, 3)
DEF_TC(q5_0, 4)
DEF_TC(q5_1, 5)
DEF_TC(q8_0, 6)
DEF_TC(q2_k, 7)
DEF_TC(q3_k, 8)
DEF_TC(q4_k, 9)
DEF_TC(q5_k, 10)
DEF_TC(q6_k, 11)
DEF_TC(q8_k, 12)

#undef DEF_TC

/* ---- bandwidth GEMV (T == 1) ------------------------------------------------
   Four rows per warp, eight threads per row. Each of those eight threads
   issues one ld.global.v4 (16 bytes) so a row's quant payload is one
   coalesced 128-byte read. The activation was quantized to q8 (block 32)
   once per vector; the dot is dp4a.s32.s32. K-quant rows require n % 256
   == 0. Q8_0 / Q5_0 require n % 32 == 0. No per-element tail branch.

   Lane split: sub = lane >> 3 (row within the four), t = lane & 7.
   Block: (warps * 32) threads, grid ceil(rows / (warps * 4)). */

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ int pack_lo_nib(unsigned v) {
    return (int)(v & 0x0f0f0f0fu);
}

__device__ __forceinline__ int pack_hi_nib(unsigned v) {
    return (int)((v >> 4) & 0x0f0f0f0fu);
}

__device__ __forceinline__ unsigned ld_u32_any(const unsigned char* p) {
    unsigned long long a = (unsigned long long)p;
    if ((a & 3u) == 0u) return *(const unsigned*)p;
    unsigned short a0 = *(const unsigned short*)p;
    unsigned short a1 = *(const unsigned short*)(p + 2);
    return (unsigned)a0 | ((unsigned)a1 << 16);
}

__device__ __forceinline__ uint4 ld_u4_any(const unsigned char* p) {
    unsigned long long a = (unsigned long long)p;
    if ((a & 15u) == 0u) return *(const uint4*)p;
    uint4 v;
    v.x = ld_u32_any(p);
    v.y = ld_u32_any(p + 4);
    v.z = ld_u32_any(p + 8);
    v.w = ld_u32_any(p + 12);
    return v;
}

/* 16 bytes at p, using only aligned ld.global.v4. Misaligned p (Q6 blocks are
   210 bytes) is a 2-byte shift of two aligned windows. The caller must pad the
   allocation by 32 bytes so the second window stays in range. */
__device__ __forceinline__ uint4 ld_u4_aligned(const unsigned char* p) {
    unsigned long long a = (unsigned long long)p;
    unsigned mis = (unsigned)(a & 15ull);
    if (mis == 0u) return *(const uint4*)p;
    const uint4* base = (const uint4*)(a & ~15ull);
    uint4 lo = base[0];
    uint4 hi = base[1];
    unsigned long long L0 = (unsigned long long)lo.x | ((unsigned long long)lo.y << 32);
    unsigned long long L1 = (unsigned long long)lo.z | ((unsigned long long)lo.w << 32);
    unsigned long long H0 = (unsigned long long)hi.x | ((unsigned long long)hi.y << 32);
    unsigned long long H1 = (unsigned long long)hi.z | ((unsigned long long)hi.w << 32);
    unsigned long long A, B;
    if (mis < 8u) {
        unsigned sh = mis * 8u;
        A = (L0 >> sh) | (L1 << (64u - sh));
        B = (L1 >> sh) | (H0 << (64u - sh));
    } else if (mis == 8u) {
        A = L1;
        B = H0;
    } else {
        unsigned sh = (mis - 8u) * 8u;
        A = (L1 >> sh) | (H0 << (64u - sh));
        B = (H0 >> sh) | (H1 << (64u - sh));
    }
    uint4 o;
    o.x = (unsigned)A;
    o.y = (unsigned)(A >> 32);
    o.z = (unsigned)B;
    o.w = (unsigned)(B >> 32);
    return o;
}

/* Subtract 32 from each byte. No vsub4: NVRTC 12.9 emits
   vsub4.s32.s32.s32, and the sm_70 PTX JIT rejects that opcode
   (CUDA_ERROR_INVALID_PTX), which rejects the whole module. */
__device__ __forceinline__ int sub_bias32_bytes(unsigned x) {
    unsigned b0 = (x & 0xffu) - 32u;
    unsigned b1 = ((x >> 8) & 0xffu) - 32u;
    unsigned b2 = ((x >> 16) & 0xffu) - 32u;
    unsigned b3 = ((x >> 24) & 0xffu) - 32u;
    return (int)((b0 & 0xffu) | ((b1 & 0xffu) << 8) | ((b2 & 0xffu) << 16) | ((b3 & 0xffu) << 24));
}

/* Four Q6 codes from one ql word and the matching qh word. hi_pair 0 is
   q1/q3 (qh bits 0 and 4); hi_pair 1 is q2/q4 (qh bits 2 and 6). */
__device__ __forceinline__ void q6_pack(unsigned ql, unsigned qh, int hi_pair, int* qa, int* qb) {
    unsigned lo = ql & 0x0f0f0f0fu;
    unsigned hi = (ql >> 4) & 0x0f0f0f0fu;
    unsigned hs = hi_pair ? 2u : 0u;
    unsigned hb = hi_pair ? 6u : 4u;
    *qa = sub_bias32_bytes(lo | (((qh >> hs) & 0x03030303u) << 4));
    *qb = sub_bias32_bytes(hi | (((qh >> hb) & 0x03030303u) << 4));
}

__device__ __forceinline__ unsigned hdr_scale_byte(uint4 hdr, int i) {
    unsigned w = (i < 4) ? hdr.y : (i < 8) ? hdr.z : hdr.w;
    return (w >> ((i & 3) * 8)) & 0xffu;
}

__device__ __forceinline__ void scale_min_hdr(int j, uint4 hdr, int* sc, int* m) {
    if (j < 4) {
        *sc = (int)(hdr_scale_byte(hdr, j) & 63u);
        *m = (int)(hdr_scale_byte(hdr, j + 4) & 63u);
    } else {
        unsigned qj4 = hdr_scale_byte(hdr, j + 4);
        unsigned qjm = hdr_scale_byte(hdr, j - 4);
        unsigned qj = hdr_scale_byte(hdr, j);
        *sc = (int)((qj4 & 0xFu) | ((qjm >> 6) << 4));
        *m = (int)((qj4 >> 4) | ((qj >> 6) << 4));
    }
}

__device__ __forceinline__ int scv_i8(uint4 scv, int idx) {
    unsigned w = idx < 4 ? scv.x : idx < 8 ? scv.y : idx < 12 ? scv.z : scv.w;
    return (int)(signed char)((w >> ((idx & 3) * 8)) & 0xffu);
}

__device__ __forceinline__ void store_row(float* out, int row, int d, int writer, float acc, int accumulate) {
    if (writer && row < d) {
        if (accumulate) out[row] += acc;
        else out[row] = acc;
    }
}

__device__ __forceinline__ int dot_nib(uint4 q, uint4 x, int hi) {
    unsigned qw[4] = {q.x, q.y, q.z, q.w};
    unsigned xw[4] = {x.x, x.y, x.z, x.w};
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int packed = hi ? pack_hi_nib(qw[i]) : pack_lo_nib(qw[i]);
        acc = dp4a_s32(packed, (int)xw[i], acc);
    }
    return acc;
}

__device__ __forceinline__ float f16_bits(unsigned bits) {
    unsigned char p[2];
    p[0] = (unsigned char)(bits & 0xff);
    p[1] = (unsigned char)((bits >> 8) & 0xff);
    return ld_f16(p);
}

__device__ __forceinline__ uint4 shfl_u4(uint4 v, int src) {
    uint4 o;
    o.x = (unsigned)__shfl_sync(0xffffffff, (int)v.x, src);
    o.y = (unsigned)__shfl_sync(0xffffffff, (int)v.y, src);
    o.z = (unsigned)__shfl_sync(0xffffffff, (int)v.z, src);
    o.w = (unsigned)__shfl_sync(0xffffffff, (int)v.w, src);
    return o;
}

__device__ __forceinline__ float warp8_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

/* x[n] -> q8 groups of 32. scale[g], sum16[g] = (sum of first 16, sum of last 16)
   of the original floats, so the Q4_K / Q5_K min term stays exact. */
extern "C" __global__ void quant_q8(const float* x, signed char* qs, float* scale, float2* sum16, int n) {
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ng = n >> 5;
    if (g >= ng) return;
    const float* xi = x + (size_t)g * 32;
    float amax = 0.f, s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float a = xi[j];
        float b = xi[j + 16];
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        s0 += a;
        s1 += b;
    }
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    scale[g] = d;
    sum16[g] = make_float2(s0, s1);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        int q = (int)rintf(xi[j] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qs[(size_t)g * 32 + j] = (signed char)q;
    }
}

/* RMSNorm one vector and quantize it. One block, 128 threads. n % 32 == 0, n <= 32768. */
extern "C" __global__ void rmsnorm_quant_q8(const float* inp, float* out, const float* weight,
                                            signed char* qs, float* scale, float2* sum16, int n, float eps) {
    int tid = (int)threadIdx.x;
    float ss = 0.f;
    for (int i = tid; i < n; i += 128) {
        float v = inp[i];
        ss += v * v;
    }
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)n + eps);
    for (int i = tid; i < n; i += 128) out[i] = weight[i] * (inp[i] * inv);
    __syncthreads();
    int ng = n >> 5;
    for (int g = tid; g < ng; g += 128) {
        const float* xi = out + (size_t)g * 32;
        float amax = 0.f, s0 = 0.f, s1 = 0.f;
        for (int j = 0; j < 16; ++j) {
            float a = xi[j];
            float b = xi[j + 16];
            amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
            s0 += a;
            s1 += b;
        }
        float d = amax / 127.f;
        float id = d > 0.f ? 1.f / d : 0.f;
        scale[g] = d;
        sum16[g] = make_float2(s0, s1);
        for (int j = 0; j < 32; ++j) {
            int q = (int)rintf(xi[j] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[(size_t)g * 32 + j] = (signed char)q;
        }
    }
}

extern "C" __global__ void gemv2_q4_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int p = t >> 1;
    int half = t & 1;
    int nb = n >> 8;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        uint4 hdr = make_uint4(0, 0, 0, 0);
        uint4 qv = make_uint4(0, 0, 0, 0);
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 144;
            if (t == 0) hdr = *(const uint4*)blk;
            qv = *(const uint4*)(blk + 16 + t * 16);
        }
        int src = lane & ~7;
        hdr = shfl_u4(hdr, src);
        if (row < d) {
            float dW = f16_bits(hdr.x & 0xffff);
            float dmin = f16_bits(hdr.x >> 16);
            int sc0, m0, sc1, m1;
            scale_min_hdr(p * 2, hdr, &sc0, &m0);
            scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);
            int group = b * 8 + p * 2;
            int off = half * 16;
            uint4 xlo = *(const uint4*)(xq + (size_t)group * 32 + off);
            uint4 xhi = *(const uint4*)(xq + (size_t)(group + 1) * 32 + off);
            int dot_lo = dot_nib(qv, xlo, 0);
            int dot_hi = dot_nib(qv, xhi, 1);
            float2 slo = sum16[group];
            float2 shi = sum16[group + 1];
            float sum_lo = half ? slo.y : slo.x;
            float sum_hi = half ? shi.y : shi.x;
            acc += (dW * (float)sc0) * xd[group] * (float)dot_lo - (dmin * (float)m0) * sum_lo;
            acc += (dW * (float)sc1) * xd[group + 1] * (float)dot_hi - (dmin * (float)m1) * sum_hi;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

extern "C" __global__ void gemv2_q6_k(const unsigned char* wb, const signed char* xq, const float* xd,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int half = t >> 2;
    int piece = t & 3;
    int l0 = (piece & 1) * 16;
    int hi_pair = piece >> 1;
    int nb = n >> 8;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        uint4 ql = make_uint4(0, 0, 0, 0);
        uint4 qh = make_uint4(0, 0, 0, 0);
        uint4 scv = make_uint4(0, 0, 0, 0);
        float dW = 0.f;
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 210;
            ql = ld_u4_aligned(blk + t * 16);
            qh = ld_u4_aligned(blk + 128 + half * 32 + l0);
            if (t == 0) {
                scv = ld_u4_aligned(blk + 192);
                dW = ld_f16(blk + 208);
            }
        }
        int src = lane & ~7;
        scv = shfl_u4(scv, src);
        dW = __shfl_sync(0xffffffff, dW, src);
        if (row < d) {
            int scoff = half * 8;
            int is = l0 >> 4;
            int s_a = scoff + is + (hi_pair ? 2 : 0);
            int s_b = scoff + is + (hi_pair ? 6 : 4);
            float sa = dW * (float)scv_i8(scv, s_a);
            float sb = dW * (float)scv_i8(scv, s_b);
            int group0 = b * 8 + half * 4 + (hi_pair ? 1 : 0);
            int off = l0;
            uint4 xa = *(const uint4*)(xq + (size_t)group0 * 32 + off);
            uint4 xb = *(const uint4*)(xq + (size_t)(group0 + 2) * 32 + off);
            unsigned qw[4] = {ql.x, ql.y, ql.z, ql.w};
            unsigned hw[4] = {qh.x, qh.y, qh.z, qh.w};
            unsigned aw[4] = {xa.x, xa.y, xa.z, xa.w};
            unsigned bw[4] = {xb.x, xb.y, xb.z, xb.w};
            int dot_a = 0, dot_b = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int qa, qb;
                q6_pack(qw[i], hw[i], hi_pair, &qa, &qb);
                dot_a = dp4a_s32(qa, (int)aw[i], dot_a);
                dot_b = dp4a_s32(qb, (int)bw[i], dot_b);
            }
            acc += sa * xd[group0] * (float)dot_a;
            acc += sb * xd[group0 + 2] * (float)dot_b;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

extern "C" __global__ void gemv2_q5_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int p = t >> 1;
    int half = t & 1;
    int nb = n >> 8;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        uint4 hdr = make_uint4(0, 0, 0, 0);
        uint4 qv = make_uint4(0, 0, 0, 0);
        uint4 hv = make_uint4(0, 0, 0, 0);
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 176;
            if (t == 0) hdr = *(const uint4*)blk;
            qv = *(const uint4*)(blk + 48 + t * 16);
            hv = *(const uint4*)(blk + 16 + half * 16);
        }
        int src = lane & ~7;
        hdr = shfl_u4(hdr, src);
        if (row < d) {
            float dW = f16_bits(hdr.x & 0xffff);
            float dmin = f16_bits(hdr.x >> 16);
            int sc0, m0, sc1, m1;
            scale_min_hdr(p * 2, hdr, &sc0, &m0);
            scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);
            unsigned bit_lo = 1u << (2 * p);
            unsigned bit_hi = bit_lo << 1;
            int group = b * 8 + p * 2;
            int off = half * 16;
            uint4 xlo = *(const uint4*)(xq + (size_t)group * 32 + off);
            uint4 xhi = *(const uint4*)(xq + (size_t)(group + 1) * 32 + off);
            unsigned qw[4] = {qv.x, qv.y, qv.z, qv.w};
            unsigned hw[4] = {hv.x, hv.y, hv.z, hv.w};
            unsigned al[4] = {xlo.x, xlo.y, xlo.z, xlo.w};
            unsigned ah[4] = {xhi.x, xhi.y, xhi.z, xhi.w};
            int dot_lo = 0, dot_hi = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int plo = 0, phi = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    unsigned byte = (qw[i] >> (8 * k)) & 0xff;
                    unsigned hb = (hw[i] >> (8 * k)) & 0xff;
                    int lo = (int)(byte & 0xf) + ((hb & bit_lo) ? 16 : 0);
                    int hi = (int)(byte >> 4) + ((hb & bit_hi) ? 16 : 0);
                    plo |= (lo & 0xff) << (8 * k);
                    phi |= (hi & 0xff) << (8 * k);
                }
                dot_lo = dp4a_s32(plo, (int)al[i], dot_lo);
                dot_hi = dp4a_s32(phi, (int)ah[i], dot_hi);
            }
            float2 slo = sum16[group];
            float2 shi = sum16[group + 1];
            float sum_lo = half ? slo.y : slo.x;
            float sum_hi = half ? shi.y : shi.x;
            acc += (dW * (float)sc0) * xd[group] * (float)dot_lo - (dmin * (float)m0) * sum_lo;
            acc += (dW * (float)sc1) * xd[group + 1] * (float)dot_hi - (dmin * (float)m1) * sum_hi;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

extern "C" __global__ void gemv2_q8_0(const unsigned char* wb, const signed char* xq, const float* xd,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int nb = n >> 5;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 34;
            float dW = ld_f16(blk);
            unsigned w = ld_u32_any(blk + 2 + t * 4);
            unsigned xv = *(const unsigned*)(xq + (size_t)b * 32 + t * 4);
            int dot = dp4a_s32((int)w, (int)xv, 0);
            acc += dW * xd[b] * (float)dot;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

extern "C" __global__ void gemv2_q5_0(const unsigned char* wb, const signed char* xq, const float* xd,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int nb = n >> 5;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        uint4 qv = make_uint4(0, 0, 0, 0);
        unsigned qh = 0;
        float dW = 0.f;
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 22;
            dW = ld_f16(blk);
            qh = ld_u32_any(blk + 2);
            if (t == 0) qv = ld_u4_any(blk + 6);
        }
        int src = lane & ~7;
        qv = shfl_u4(qv, src);
        if (row < d) {
            unsigned qw[4] = {qv.x, qv.y, qv.z, qv.w};
            int base = (t & 3) * 4;
            int hi = t >> 2;
            unsigned xv = *(const unsigned*)(xq + (size_t)b * 32 + t * 4);
            int packed = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int j = base + k;
                unsigned byte = (qw[j >> 2] >> (8 * (j & 3))) & 0xff;
                int nib = hi ? (int)(byte >> 4) : (int)(byte & 0xf);
                int elem = hi ? (j + 16) : j;
                int qv_i = (nib | (((qh >> elem) & 1) << 4)) - 16;
                packed |= (qv_i & 0xff) << (8 * k);
            }
            int dot = dp4a_s32(packed, (int)xv, 0);
            acc += dW * xd[b] * (float)dot;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

/* One output row per warp. Used when the matrix is short (wo, w2, 9B wq):
   four rows per warp leaves the V100 under-occupied. One lane owns one
   element of each 32-wide group, so the loads are coalesced. The Q4 min
   term is the full 32-element sum and is subtracted once, on lane 0. */
extern "C" __global__ void gemv2r1_q4_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                        float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * ((int)blockDim.x >> 5) + warp;
    int nb = n >> 8;
    float acc = 0.f;
    if (row < d) {
        for (int b = 0; b < nb; ++b) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 144;
            uint4 hdr = make_uint4(0, 0, 0, 0);
            if (lane == 0) hdr = *(const uint4*)blk;
            hdr = shfl_u4(hdr, 0);
            float dW = f16_bits(hdr.x & 0xffff);
            float dmin = f16_bits(hdr.x >> 16);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int sc0, m0, sc1, m1;
                scale_min_hdr(p * 2, hdr, &sc0, &m0);
                scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);
                unsigned char q = blk[16 + p * 32 + lane];
                int g0 = b * 8 + p * 2;
                int xlo = (int)xq[(size_t)g0 * 32 + lane];
                int xhi = (int)xq[(size_t)(g0 + 1) * 32 + lane];
                acc += (dW * (float)sc0) * xd[g0] * (float)(q & 0xf) * (float)xlo;
                acc += (dW * (float)sc1) * xd[g0 + 1] * (float)(q >> 4) * (float)xhi;
                if (lane == 0) {
                    float2 s0 = sum16[g0];
                    float2 s1 = sum16[g0 + 1];
                    acc -= (dmin * (float)m0) * (s0.x + s0.y);
                    acc -= (dmin * (float)m1) * (s1.x + s1.y);
                }
            }
        }
    }
    acc = warp_sum(acc);
    store_row(out, row, d, lane == 0, acc, accumulate);
}

extern "C" __global__ void gemv2r1_q6_k(const unsigned char* wb, const signed char* xq, const float* xd,
                                        float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * ((int)blockDim.x >> 5) + warp;
    int nb = n >> 8;
    float acc = 0.f;
    if (row < d) {
        for (int b = 0; b < nb; ++b) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 210;
            uint4 scv = make_uint4(0, 0, 0, 0);
            float dW = 0.f;
            if (lane == 0) {
                scv = ld_u4_aligned(blk + 192);
                dW = ld_f16(blk + 208);
            }
            scv = shfl_u4(scv, 0);
            dW = __shfl_sync(0xffffffff, dW, 0);
            const unsigned char* ql = blk;
            const unsigned char* qh = blk + 128;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int is = lane >> 4;
                int qloff = half * 64;
                int qhoff = half * 32;
                int scoff = half * 8;
                int xelem = lane;
                unsigned char qhb = qh[qhoff + lane];
                int q1 = (int)((ql[qloff + lane] & 0xF) | (((qhb >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[qloff + lane + 32] & 0xF) | (((qhb >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[qloff + lane] >> 4) | (((qhb >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[qloff + lane + 32] >> 4) | (((qhb >> 6) & 3) << 4)) - 32;
                float s0 = dW * (float)scv_i8(scv, scoff + is + 0);
                float s1 = dW * (float)scv_i8(scv, scoff + is + 2);
                float s2 = dW * (float)scv_i8(scv, scoff + is + 4);
                float s3 = dW * (float)scv_i8(scv, scoff + is + 6);
                int g = b * 8 + half * 4;
                acc += s0 * xd[g + 0] * (float)q1 * (float)(int)xq[(size_t)(g + 0) * 32 + xelem];
                acc += s1 * xd[g + 1] * (float)q2 * (float)(int)xq[(size_t)(g + 1) * 32 + xelem];
                acc += s2 * xd[g + 2] * (float)q3 * (float)(int)xq[(size_t)(g + 2) * 32 + xelem];
                acc += s3 * xd[g + 3] * (float)q4 * (float)(int)xq[(size_t)(g + 3) * 32 + xelem];
            }
        }
    }
    acc = warp_sum(acc);
    store_row(out, row, d, lane == 0, acc, accumulate);
}
