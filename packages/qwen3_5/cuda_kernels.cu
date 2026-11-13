/* Qwen3.5 CUDA kernels. sm_70+ (Tesla V100). fp32 accumulate. No bf16, no
   cp.async, no ldmatrix (that is sm_75). Prefill GEMM (T>1) can use Volta
   WMMA 16x16x16: wmma.mma.sync.aligned.m16n16k16, fp16 operands, fp32
   accumulate. NVRTC does not ship mma.h, so the WMMA ops are inline PTX.
   Decode stays on GEMV. NVRTC is given an explicit
   --gpu-architecture=compute_XX (CUDA 13 NVRTC defaults to sm_75 and will
   not JIT on sm_70).

   Quant layouts match packages/ggml/quant.odin. GEMV is one warp per output
   row. MMQ reuses the same accumulator over a tile of up to 8 tokens and
   writes token-major [T, rows]. */

/* Half conversion is inline so NVRTC does not need cuda_fp16.h. The pip
   wheel (nvidia-cuda-nvrtc) does not ship that header, and CUDA 12.9's
   libnvrtc-builtins does not register it as a builtin include. Round-to-
   nearest-even matches CUDA __float2half and Odin's f16 cast. */

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
    float ang = (float)pos * freq;
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
    size_t dst = (((size_t)kvh * (size_t)seq_len + (size_t)pos) * (size_t)head_dim + (size_t)i) * 2;
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
    int npos = pos + 1;
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
    unsigned char* kb = kc + (size_t)pos * (size_t)pos_bytes + (size_t)gid * 34;
    unsigned char* vb = vc + (size_t)pos * (size_t)pos_bytes + (size_t)gid * 34;
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
    int npos = pos + 1;
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
    unsigned char* kb = kc + (size_t)pos * (size_t)k_pos_bytes + (size_t)gid * 34;
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
    unsigned char* vb = vc + (size_t)pos * (size_t)v_pos_bytes + (size_t)gid * 40;
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
    int npos = pos + 1;
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
