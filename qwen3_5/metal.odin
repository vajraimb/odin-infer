#+build darwin

/* Metal GPU forward pass for Qwen3.5 hybrid (full + linear attention).
   Reuses the GEMV / rmsnorm / attention / swiglu / residual kernels from the
   Qwen3 engine and adds: partial-rotary MRoPE, attention output gate, conv1d
   step, l2norm+scale, gated-delta recurrent step, and gated RMSNorm. */

package qwen3_5

import ggml "ggml:ggml"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:math"
import "core:os"
import MTL "vendor:darwin/Metal"

PAGE_SIZE :: 16384
GEMV_ROWS :: 8
GEMV_TG :: 256
MAX_BATCH_T :: 512 // max tokens per batched-prefill chunk (Stage 1a)

MSL_SRC := `
#include <metal_stdlib>
using namespace metal;

struct Dims { uint n; uint d; };
constant uint GEMV_ROWS = 8;
constant uint GEMV_SG   = 32;

inline void get_scale_min_k4(int j, device const uchar *q, thread uchar &d, thread uchar &m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j + 0] >> 6) << 4); }
}

kernel void gemv_f32(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { device const float *w = (device const float*)wb + (ulong)row * dim.n;
        for (uint j = lane; j < dim.n; j += GEMV_SG) sum += w[j] * x[j]; }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_f16(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { device const half *w = (device const half*)wb + (ulong)row * dim.n;
        for (uint j = lane; j < dim.n; j += GEMV_SG) sum += (float)w[j] * x[j]; }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_q8_0(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nb = dim.n / 32; device const uchar *wr = wb + (ulong)row * nb * 34;
        for (uint b = lane; b < nb; b += GEMV_SG) { device const uchar *blk = wr + b * 34;
            float dd = (float)(*(device const half*)blk); device const char *qs = (device const char*)(blk + 2);
            uint base = b * 32; float4 acc = float4(0.0);
            for (uint j = 0; j < 32; j += 4) { float4 xv = float4(x[base+j],x[base+j+1],x[base+j+2],x[base+j+3]);
                float4 qv = float4((float)qs[j],(float)qs[j+1],(float)qs[j+2],(float)qs[j+3]); acc += qv * dd * xv; }
            sum += acc.x + acc.y + acc.z + acc.w; } }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_q4_0(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nb = dim.n / 32; device const uchar *wr = wb + (ulong)row * nb * 18;
        for (uint b = lane; b < nb; b += GEMV_SG) { device const uchar *blk = wr + b * 18;
            float dd = (float)(*(device const half*)blk); device const uchar *qs = blk + 2; uint base = b * 32;
            for (uint j = 0; j < 16; ++j) { float w0 = (float)((int)(qs[j] & 0xF) - 8); float w1 = (float)((int)(qs[j] >> 4) - 8);
                sum += w0 * dd * x[base + j]; sum += w1 * dd * x[base + j + 16]; } } }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_q4_1(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nb = dim.n / 32; device const uchar *wr = wb + (ulong)row * nb * 20;
        for (uint b = lane; b < nb; b += GEMV_SG) { device const uchar *blk = wr + b * 20;
            float dd = (float)(*(device const half*)blk); float mm = (float)(*(device const half*)(blk + 2));
            device const uchar *qs = blk + 4; uint base = b * 32;
            for (uint j = 0; j < 16; ++j) { sum += ((float)(qs[j] & 0xF) * dd + mm) * x[base + j];
                sum += ((float)(qs[j] >> 4) * dd + mm) * x[base + j + 16]; } } }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_q5_0(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nb = dim.n / 32; device const uchar *wr = wb + (ulong)row * nb * 22;
        for (uint b = lane; b < nb; b += GEMV_SG) { device const uchar *blk = wr + b * 22;
            float dd = (float)(*(device const half*)blk); uint qh = (uint)blk[2]|((uint)blk[3]<<8)|((uint)blk[4]<<16)|((uint)blk[5]<<24);
            device const uchar *qs = blk + 6; uint base = b * 32;
            for (uint j = 0; j < 16; ++j) { uchar xh0 = (uchar)((qh >> j) << 4) & 0x10; uchar xh1 = (uchar)(qh >> (j + 12)) & 0x10;
                float w0 = (float)((int)((qs[j] & 0xF) | xh0) - 16); float w1 = (float)((int)((qs[j] >> 4) | xh1) - 16);
                sum += w0 * dd * x[base + j]; sum += w1 * dd * x[base + j + 16]; } } }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_q5_1(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nb = dim.n / 32; device const uchar *wr = wb + (ulong)row * nb * 24;
        for (uint b = lane; b < nb; b += GEMV_SG) { device const uchar *blk = wr + b * 24;
            float dd = (float)(*(device const half*)blk); float mm = (float)(*(device const half*)(blk + 2));
            uint qh = (uint)blk[4]|((uint)blk[5]<<8)|((uint)blk[6]<<16)|((uint)blk[7]<<24); device const uchar *qs = blk + 8; uint base = b * 32;
            for (uint j = 0; j < 16; ++j) { uchar xh0 = (uchar)((qh >> j) << 4) & 0x10; uchar xh1 = (uchar)(qh >> (j + 12)) & 0x10;
                float w0 = (float)((qs[j] & 0xF) | xh0); float w1 = (float)((qs[j] >> 4) | xh1);
                sum += (w0 * dd + mm) * x[base + j]; sum += (w1 * dd + mm) * x[base + j + 16]; } } }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_q4_k(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 144;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 144;
            float dall = (float)(*(device const half*)blk); float dmin = (float)(*(device const half*)(blk + 2));
            device const uchar *scales = blk + 4; device const uchar *qs = blk + 16; uint ybase = sb * 256; int is = 0; uint qoff = 0;
            for (uint j = 0; j < 256; j += 64) { uchar sc, m; get_scale_min_k4(is + 0, scales, sc, m); float d1 = dall * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, scales, sc, m); float d2 = dall * sc, m2 = dmin * m;
                for (uint l = 0; l < 32; ++l) { sum += (d1 * (float)(qs[qoff + l] & 0xF) - m1) * x[ybase + j + l];
                    sum += (d2 * (float)(qs[qoff + l] >> 4) - m2) * x[ybase + j + 32 + l]; }
                qoff += 32; is += 2; } } }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}
inline float dot_q6_superblock(device const uchar *blk, device const float *x, uint ybase) {
    device const uchar *ql = blk; device const uchar *qh = blk + 128; device const char *sc = (device const char*)(blk + 192);
    float dall = (float)(*(device const half*)(blk + 208)); float sum = 0.0;
    for (int hf = 0; hf < 2; ++hf) { uint qlo = hf * 64, qho = hf * 32, sco = hf * 8, yo = ybase + hf * 128;
        float ds0=dall*(float)sc[sco+0],ds1=dall*(float)sc[sco+1],ds2=dall*(float)sc[sco+2],ds3=dall*(float)sc[sco+3];
        float ds4=dall*(float)sc[sco+4],ds5=dall*(float)sc[sco+5],ds6=dall*(float)sc[sco+6],ds7=dall*(float)sc[sco+7];
        for (int l = 0; l < 16; l += 4) {
            float4 w1=float4((float)((int)((ql[qlo+l+0]&0xF)|(((qh[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+1]&0xF)|(((qh[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+2]&0xF)|(((qh[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+3]&0xF)|(((qh[qho+l+3]>>0)&3)<<4))-32));
            float4 w2=float4((float)((int)((ql[qlo+l+32]&0xF)|(((qh[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+33]&0xF)|(((qh[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+34]&0xF)|(((qh[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+35]&0xF)|(((qh[qho+l+3]>>2)&3)<<4))-32));
            float4 w3=float4((float)((int)((ql[qlo+l+0]>>4)|(((qh[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+1]>>4)|(((qh[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+2]>>4)|(((qh[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+3]>>4)|(((qh[qho+l+3]>>4)&3)<<4))-32));
            float4 w4=float4((float)((int)((ql[qlo+l+32]>>4)|(((qh[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+33]>>4)|(((qh[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+34]>>4)|(((qh[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+35]>>4)|(((qh[qho+l+3]>>6)&3)<<4))-32));
            float4 xv1=float4(x[yo+l],x[yo+l+1],x[yo+l+2],x[yo+l+3]); float4 xv2=float4(x[yo+l+32],x[yo+l+33],x[yo+l+34],x[yo+l+35]);
            float4 xv3=float4(x[yo+l+64],x[yo+l+65],x[yo+l+66],x[yo+l+67]); float4 xv4=float4(x[yo+l+96],x[yo+l+97],x[yo+l+98],x[yo+l+99]);
            sum += ds0*dot(w1,xv1)+ds2*dot(w2,xv2)+ds4*dot(w3,xv3)+ds6*dot(w4,xv4);
        }
        for (int l = 16; l < 32; l += 4) {
            float4 w1=float4((float)((int)((ql[qlo+l+0]&0xF)|(((qh[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+1]&0xF)|(((qh[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+2]&0xF)|(((qh[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+3]&0xF)|(((qh[qho+l+3]>>0)&3)<<4))-32));
            float4 w2=float4((float)((int)((ql[qlo+l+32]&0xF)|(((qh[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+33]&0xF)|(((qh[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+34]&0xF)|(((qh[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+35]&0xF)|(((qh[qho+l+3]>>2)&3)<<4))-32));
            float4 w3=float4((float)((int)((ql[qlo+l+0]>>4)|(((qh[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+1]>>4)|(((qh[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+2]>>4)|(((qh[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+3]>>4)|(((qh[qho+l+3]>>4)&3)<<4))-32));
            float4 w4=float4((float)((int)((ql[qlo+l+32]>>4)|(((qh[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+33]>>4)|(((qh[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+34]>>4)|(((qh[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+35]>>4)|(((qh[qho+l+3]>>6)&3)<<4))-32));
            float4 xv1=float4(x[yo+l],x[yo+l+1],x[yo+l+2],x[yo+l+3]); float4 xv2=float4(x[yo+l+32],x[yo+l+33],x[yo+l+34],x[yo+l+35]);
            float4 xv3=float4(x[yo+l+64],x[yo+l+65],x[yo+l+66],x[yo+l+67]); float4 xv4=float4(x[yo+l+96],x[yo+l+97],x[yo+l+98],x[yo+l+99]);
            sum += ds1*dot(w1,xv1)+ds3*dot(w2,xv2)+ds5*dot(w3,xv3)+ds7*dot(w4,xv4);
        }
    }
    return sum;
}
kernel void gemv_q6_k(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 210;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) sum += dot_q6_superblock(wr + sb * 210, x, sb * 256); }
    sum = simd_sum(sum); if (lane == 0 && row < dim.d) out[row] = sum;
}

// ---------- elementwise / attention ----------
struct NormP { uint size; float eps; };
struct RopeP { uint head_dim; uint pos; float rope_freq; uint rotary_dim; };
struct AttnP { uint head_dim; uint seq_len; uint kv_mul; uint pos; };
struct StoreP { uint head_dim; uint seq_len; uint pos; };

kernel void rmsnorm(device const float *inp [[buffer(0)]], device float *outp [[buffer(1)]],
    device const float *w [[buffer(2)]], constant NormP &P [[buffer(3)]],
    uint grp [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    device const float *xi = inp + (ulong)grp * P.size; device float *oi = outp + (ulong)grp * P.size;
    float ss = 0.0; for (uint j = tid; j < P.size; j += 32) ss += xi[j] * xi[j];
    ss = simd_sum(ss); ss = ss / (float)P.size + P.eps; ss = rsqrt(ss);
    for (uint j = tid; j < P.size; j += 32) oi[j] = w[j] * (ss * xi[j]);
}

// Partial-rotary MRoPE: rotate only the first rotary_dim/2 pairs per head.
kernel void rope_partial(device float *vec [[buffer(0)]], constant RopeP &P [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    uint half_r = P.rotary_dim / 2; uint h = gid / half_r; uint i = gid % half_r;
    device float *q = vec + (ulong)h * P.head_dim;
    float freq = 1.0 / pow(P.rope_freq, (float)i / (float)half_r);
    float c = cos((float)P.pos * freq), s = sin((float)P.pos * freq);
    float x0 = q[i], y0 = q[i + half_r];
    q[i] = x0 * c - y0 * s; q[i + half_r] = x0 * s + y0 * c;
}

// Extract q (first head_dim of each head_dim*2 block of qproj) and per-head RMSNorm it.
kernel void qnorm_extract(device const float *qproj [[buffer(0)]], device float *q [[buffer(1)]],
    device const float *qn [[buffer(2)]], constant NormP &P [[buffer(3)]],
    uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint n = P.size; device const float *src = qproj + (ulong)h * (2 * n); device float *dst = q + (ulong)h * n;
    float ss = 0.0; for (uint j = tid; j < n; j += 32) ss += src[j] * src[j];
    ss = simd_sum(ss); ss = ss / (float)n + P.eps; ss = rsqrt(ss);
    for (uint j = tid; j < n; j += 32) dst[j] = qn[j] * (ss * src[j]);
}

kernel void store_kv(device const float *ktmp [[buffer(0)]], device const float *vtmp [[buffer(1)]],
    device half *kc [[buffer(2)]], device half *vc [[buffer(3)]], constant StoreP &P [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint kvh = gid / P.head_dim, i = gid % P.head_dim;
    ulong dst = ((ulong)kvh * P.seq_len + P.pos) * P.head_dim + i;
    kc[dst] = (half)ktmp[gid]; vc[dst] = (half)vtmp[gid];
}

kernel void attention(device const float *q [[buffer(0)]], device const half *kc [[buffer(1)]],
    device const half *vc [[buffer(2)]], device float *xb3 [[buffer(3)]], constant AttnP &P [[buffer(4)]],
    threadgroup float *scores [[threadgroup(0)]], uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    const uint NT = 128; threadgroup float red[NT];
    device const float *qh = q + (ulong)h * P.head_dim; uint kvh = h / P.kv_mul;
    device const half *kbase = kc + (ulong)kvh * P.seq_len * P.head_dim;
    device const half *vbase = vc + (ulong)kvh * P.seq_len * P.head_dim;
    float scale = rsqrt((float)P.head_dim); uint n = P.pos + 1;
    for (uint t = tid; t < n; t += NT) { device const half *k = kbase + (ulong)t * P.head_dim;
        float s = 0.0; for (uint i = 0; i < P.head_dim; ++i) s += qh[i] * (float)k[i]; scores[t] = s * scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lm = -INFINITY; for (uint t = tid; t < n; t += NT) lm = max(lm, scores[t]); red[tid] = lm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = NT / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = max(red[tid], red[tid + s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float m = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls = 0.0; for (uint t = tid; t < n; t += NT) { float e = exp(scores[t] - m); scores[t] = e; ls += e; }
    red[tid] = ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = NT / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = red[tid] + red[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0 / red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < P.head_dim; i += NT) { float acc = 0.0;
        for (uint t = 0; t < n; ++t) acc += scores[t] * (float)vbase[(ulong)t * P.head_dim + i];
        xb3[(ulong)h * P.head_dim + i] = acc * inv; }
}

// Multiply attention output by sigmoid(gate), where gate is the second half of each head's qproj block.
kernel void attn_gate(device float *xb3 [[buffer(0)]], device const float *qproj [[buffer(1)]],
    constant uint &head_dim [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint n = head_dim; uint h = gid / n, i = gid % n;
    float g = qproj[(ulong)h * (2 * n) + n + i];
    g = 1.0 / (1.0 + exp(-g));
    xb3[(ulong)h * n + i] *= g;
}

kernel void swiglu(device float *hb [[buffer(0)]], device const float *hb2 [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    float v = hb[gid]; v *= 1.0 / (1.0 + exp(-v)); hb[gid] = v * hb2[gid];
}
kernel void residual(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { x[gid] += y[gid]; }

// ---------- linear-attention (gated delta net) kernels ----------
// Depthwise causal conv1d single-token update + silu. weight = [conv_dim, kernel], oldest tap first.
kernel void conv1d_step_silu(device float *qkv_out [[buffer(0)]], device const float *qkv_in [[buffer(1)]],
    device float *conv_state [[buffer(2)]], device const float *weight [[buffer(3)]],
    constant uint2 &P [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    uint k = P.x, km1 = k - 1;
    float acc = weight[c * k + km1] * qkv_in[c];
    for (uint i = 0; i < km1; ++i) acc += weight[c * k + i] * conv_state[c * km1 + i];
    qkv_out[c] = acc / (1.0 + exp(-acc));
    for (uint i = 0; i + 1 < km1; ++i) conv_state[c * km1 + i] = conv_state[c * km1 + i + 1];
    conv_state[c * km1 + km1 - 1] = qkv_in[c];
}

// Batched depthwise causal conv1d + silu over T tokens (Stage 2 step 1). Each
// (t,c) is independent. Left-pads with conv_state (the prefix's last km1 inputs)
// so token t only sees inputs [t-km1, t] -- no future. Matches conv1d_step_silu
// exactly: out[t,c] = silu(sum_i w[c*k+i] * src(t-km1+i, c)), src(j,c)=history if j<0.
// P = (conv_dim, T, kernel, km1). Does NOT modify conv_state (read-only here).
kernel void conv1d_batch_silu(
    device const float * x [[buffer(0)]], device float * out [[buffer(1)]],
    device const float * conv_state [[buffer(2)]], device const float * weight [[buffer(3)]],
    constant uint4 & P [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint conv_dim = P.x, T = P.y, k = P.z, km1 = P.w;
    if (gid >= T * conv_dim) return;
    uint t = gid / conv_dim, c = gid - t * conv_dim;
    device const float * wc = weight + c * k;
    // Match conv1d_step_silu's summation order EXACTLY (current tap first, then
    // history oldest->newest) so the batched output is bit-identical to per-token.
    float acc = wc[km1] * x[t * conv_dim + c];          // i = km1 (current input)
    for (uint i = 0; i < km1; i++) {                      // i = 0..km1-1 (history, oldest->newest)
        int si = int(t) - int(km1) + int(i);
        float xv = (si < 0) ? conv_state[(uint)c * km1 + uint(si + int(km1))] : x[uint(si) * conv_dim + c];
        acc += wc[i] * xv;
    }
    out[gid] = acc / (1.0f + exp(-acc));
}
// After a batched conv chunk, slide conv_state to hold the chunk's last km1
// inputs (next chunk / prefix-cache resumes correctly). T >= km1 assumed (true:
// chunks are multiples of 8, km1=3). P = (conv_dim, T, kernel, km1).
kernel void conv1d_update_state(
    device const float * x [[buffer(0)]], device float * conv_state [[buffer(1)]],
    constant uint4 & P [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint conv_dim = P.x, T = P.y, km1 = P.w;
    if (gid >= conv_dim * km1) return;
    uint c = gid / km1, i = gid - c * km1;
    conv_state[gid] = x[(T - km1 + i) * conv_dim + c];
}

struct L2P { uint head_k_dim; float eps; float scale; };
// Per key-head: l2norm q and k, scale q by 1/sqrt(head_k_dim). q_in/k_in -> q_out/k_out.
kernel void l2norm_scale(device float *q_out [[buffer(0)]], device float *k_out [[buffer(1)]],
    device const float *q_in [[buffer(2)]], device const float *k_in [[buffer(3)]], constant L2P &P [[buffer(4)]],
    uint kh [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint n = P.head_k_dim;
    device const float *qi = q_in + (ulong)kh * n; device const float *ki = k_in + (ulong)kh * n;
    device float *qo = q_out + (ulong)kh * n; device float *ko = k_out + (ulong)kh * n;
    float sq = 0.0, sk = 0.0;
    for (uint j = tid; j < n; j += 32) { sq += qi[j] * qi[j]; sk += ki[j] * ki[j]; }
    sq = simd_sum(sq); sk = simd_sum(sk);
    float iq = 1.0 / max(sqrt(sq), P.eps), ik = 1.0 / max(sqrt(sk), P.eps);
    for (uint j = tid; j < n; j += 32) { qo[j] = qi[j] * iq * P.scale; ko[j] = ki[j] * ik; }
}

struct DeltaP { uint head_k_dim; uint head_v_dim; uint n_k_heads; };
// Gated delta-rule recurrent step. One threadgroup per value-head, one thread per output column j.
// V-heads are stored in ggml "tiled" order in the GGUF, so v-head vh maps to
// k-head (vh % n_k_heads). Each thread owns column j of state[i*hvd+j] across
// all i, so phases are column-local (no barriers).
kernel void delta_recurrent(device float *out [[buffer(0)]], device float *state_all [[buffer(1)]],
    device const float *q_lin [[buffer(2)]], device const float *k_lin [[buffer(3)]],
    device const float *v_vec [[buffer(4)]], device const float *b_vec [[buffer(5)]],
    device const float *a_vec [[buffer(6)]], device const float *a_decay [[buffer(7)]],
    device const float *dt_bias [[buffer(8)]], constant DeltaP &P [[buffer(9)]],
    uint vh [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]]) {
    uint hkd = P.head_k_dim, hvd = P.head_v_dim;
    uint kh = vh % P.n_k_heads;
    device float *state = state_all + (ulong)vh * hkd * hvd;
    device const float *qt = q_lin + (ulong)kh * hkd;
    device const float *kt = k_lin + (ulong)kh * hkd;
    device const float *vt = v_vec + (ulong)vh * hvd;
    // per-head scalars (identical across the threadgroup)
    float bv = b_vec[vh], av = a_vec[vh], a_dec = a_decay[vh], dtb = dt_bias[vh];
    float beta = 1.0 / (1.0 + exp(-bv));
    float sp = av + dtb;
    float softplus = sp > 20.0 ? sp : (sp < -20.0 ? 0.0 : log(1.0 + exp(sp)));
    float g_decay = exp(a_dec * softplus); // a_dec = -exp(A_log)
    if (j >= hvd) return;
    // decay
    for (uint i = 0; i < hkd; ++i) state[(ulong)i * hvd + j] *= g_decay;
    // kv_mem + delta
    float kv_mem = 0.0; for (uint i = 0; i < hkd; ++i) kv_mem += state[(ulong)i * hvd + j] * kt[i];
    float delta = (vt[j] - kv_mem) * beta;
    // state update
    for (uint i = 0; i < hkd; ++i) state[(ulong)i * hvd + j] += kt[i] * delta;
    // output readout
    float o = 0.0; for (uint i = 0; i < hkd; ++i) o += state[(ulong)i * hvd + j] * qt[i];
    out[(ulong)vh * hvd + j] = o;
}

// RMSNormGated: rmsnorm then * silu(z), per value-head (head_v_dim).
kernel void rmsnorm_gated(device float *out [[buffer(0)]], device const float *inp [[buffer(1)]],
    device const float *weight [[buffer(2)]], device const float *gate [[buffer(3)]], constant NormP &P [[buffer(4)]],
    uint vh [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint n = P.size;
    device const float *xi = inp + (ulong)vh * n; device float *oi = out + (ulong)vh * n;
    device const float *zi = gate + (ulong)vh * n;
    float ss = 0.0; for (uint j = tid; j < n; j += 32) ss += xi[j] * xi[j];
    ss = simd_sum(ss); ss = ss / (float)n + P.eps; ss = rsqrt(ss);
    for (uint j = tid; j < n; j += 32) {
        float normed = weight[j] * (ss * xi[j]);
        float gz = zi[j] * (1.0 / (1.0 + exp(-zi[j]))); // silu(z) = z*sigmoid(z)
        oi[j] = normed * gz;
    }
}

// ---------- Stage 2 batched linear-attention kernels ----------
// Batched l2norm+scale over [T tokens x n_kh key-heads]. q_in/k_in are the
// per-token conv output [T x conv_dim]; q slice = [0,key_dim), k = [key_dim,2*key_dim).
// Dispatch (T*n_kh) threadgroups of 32 threads. q_out/k_out are [T x key_dim].
struct L2B { uint head_k_dim; uint key_dim; uint conv_dim; float eps; float scale; };
kernel void l2norm_scale_batch(
    device float * q_out [[buffer(0)]], device float * k_out [[buffer(1)]],
    device const float * qkv [[buffer(2)]], constant L2B & P [[buffer(3)]],
    uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint hkd = P.head_k_dim, key_dim = P.key_dim, conv_dim = P.conv_dim;
    uint n_kh = key_dim / hkd;
    uint kh = g % n_kh;
    uint tok = g / n_kh;
    device const float *qi = qkv + (ulong)tok * conv_dim + (ulong)kh * hkd;
    device const float *ki = qkv + (ulong)tok * conv_dim + (ulong)key_dim + (ulong)kh * hkd;
    device float *qo = q_out + (ulong)tok * key_dim + (ulong)kh * hkd;
    device float *ko = k_out + (ulong)tok * key_dim + (ulong)kh * hkd;
    float sq = 0.0, sk = 0.0;
    for (uint j = tid; j < hkd; j += 32) { sq += qi[j]*qi[j]; sk += ki[j]*ki[j]; }
    sq = simd_sum(sq); sk = simd_sum(sk);
    float iq = 1.0/max(sqrt(sq), P.eps)*P.scale, ik = 1.0/max(sqrt(sk), P.eps);
    for (uint j = tid; j < hkd; j += 32) { qo[j] = qi[j]*iq; ko[j] = ki[j]*ik; }
}

// Batched RMSNormGated over [T tokens x n_vh value-heads]. in/gate are [T x value_dim].
kernel void rmsnorm_gated_batch(
    device float * out [[buffer(0)]], device const float * inp [[buffer(1)]],
    device const float * weight [[buffer(2)]], device const float * gate [[buffer(3)]],
    constant NormP & P [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint n = P.size;
    device const float *xi = inp + (ulong)g * n; device float *oi = out + (ulong)g * n;
    device const float *zi = gate + (ulong)g * n;
    float ss = 0.0; for (uint j = tid; j < n; j += 32) ss += xi[j]*xi[j];
    ss = simd_sum(ss); ss = ss/(float)n + P.eps; ss = rsqrt(ss);
    for (uint j = tid; j < n; j += 32) {
        float normed = weight[j]*(ss*xi[j]);
        float gz = zi[j]*(1.0/(1.0+exp(-zi[j])));
        oi[j] = normed*gz;
    }
}

// Engine chunked gated-delta. One threadgroup per value-head (vh = grid x); TG = hvd
// threads (one per v-col). Processes one chunk of C tokens starting at chunk_start.
// q/k read strided from qlin_t/klin_t [T x key_dim] (k-head = vh % n_kh); v from
// qkv2 [T x conv_dim] (v slice = 2*key_dim + vh*hvd); beta/g computed per token from
// bt/at [T x n_vh] + a_decay/dt_bias [n_vh]. State S in rec_states [n_linear x n_vh x
// hkd x hvd]. Writes outs to lout [T x value_dim] and the updated state. Matches the
// validated delta_harness chunked_delta math (forward-sub, no inverse).
struct ChunkP { uint hkd; uint hvd; uint C; uint key_dim; uint conv_dim; uint value_dim; uint n_vh; uint n_kh; uint chunk_start; uint slot; };
kernel void chunked_delta(
    device const float * qlin [[buffer(0)]], device const float * klin [[buffer(1)]],
    device const float * qkv2 [[buffer(2)]], device const float * bt [[buffer(3)]],
    device const float * at [[buffer(4)]], device const float * a_decay [[buffer(5)]],
    device const float * dt_bias [[buffer(6)]], device float * rec_states [[buffer(7)]],
    device float * lout [[buffer(8)]], constant ChunkP & P [[buffer(9)]],
    uint vh [[threadgroup_position_in_grid]], uint vc [[thread_position_in_threadgroup]]) {
    const uint hkd=P.hkd, hvd=P.hvd, C=P.C, key_dim=P.key_dim, conv_dim=P.conv_dim;
    const uint value_dim=P.value_dim, n_vh=P.n_vh, kh = vh % P.n_kh, cs = P.chunk_start;
    device float * S = rec_states + (ulong)P.slot * n_vh * hkd * hvd + (ulong)vh * hkd * hvd;
    threadgroup float KKT[256]; threadgroup float KQT[256];
    for (uint idx = vc; idx < C*C; idx += hvd) {
        uint t = idx/C, i = idx - t*C;
        device const float * ki = klin + (ulong)(cs+i)*key_dim + (ulong)kh*hkd;
        device const float * kt = klin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        device const float * qt = qlin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        float dk=0, dq=0;
        for (uint a=0;a<hkd;a++){ dk+=ki[a]*kt[a]; dq+=ki[a]*qt[a]; }
        KKT[idx]=dk; KQT[idx]=dq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (vc >= hvd) return;
    float beta[16], cp[16];
    float a_dec = a_decay[vh], dtb = dt_bias[vh];
    for (uint t=0;t<C;t++) {
        float bv = bt[(ulong)(cs+t)*n_vh + vh];
        float av = at[(ulong)(cs+t)*n_vh + vh];
        beta[t] = 1.0f/(1.0f+exp(-bv));
        float sp = av + dtb;
        float softplus = sp > 20.0f ? sp : (sp < -20.0f ? 0.0f : log(1.0f+exp(sp)));
        cp[t] = (t==0) ? exp(a_dec*softplus) : cp[t-1]*exp(a_dec*softplus);
    }
    float delta[16];
    for (uint t=0;t<C;t++) {
        device const float * kt = klin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        float sik=0; for (uint a=0;a<hkd;a++) sik += S[a*hvd+vc]*kt[a];
        float vval = qkv2[(ulong)(cs+t)*conv_dim + 2ull*key_dim + (ulong)vh*hvd + vc];
        float d = beta[t]*vval - beta[t]*cp[t]*sik;
        for (uint i=0;i<t;i++){ float L = beta[t]*(cp[t]/cp[i])*KKT[t*C+i]; d -= L*delta[i]; }
        delta[t]=d;
        device const float * qt = qlin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        float siq=0; for (uint a=0;a<hkd;a++) siq += S[a*hvd+vc]*qt[a];
        float o = cp[t]*siq;
        for (uint i=0;i<=t;i++){ float pfac = (i<t)?cp[t]/cp[i]:1.0f; o += pfac*KQT[t*C+i]*delta[i]; }
        lout[(ulong)(cs+t)*value_dim + (ulong)vh*hvd + vc] = o;
    }
    float cpC = cp[C-1];
    for (uint a=0;a<hkd;a++){
        float s = cpC*S[a*hvd+vc];
        for (uint i=0;i<C;i++) s += (cpC/cp[i])*klin[(ulong)(cs+i)*key_dim+(ulong)kh*hkd+a]*delta[i];
        S[a*hvd+vc] = s;
    }
}

// ---------- batched prefill: Q4_K MMQ + helpers (BATCHED_PREFILL Stage 1a) ----------
// Q4_K dequant matches ggml dequant_q4_k (ggml/quant.odin) exactly.
inline void k4_get_scale_min(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
inline float k4_dequant_elem(device const uchar * blk, int e) {
    float d = (float)(*(device const half *)(blk));
    float dmin = (float)(*(device const half *)(blk + 2));
    device const uchar * scales = blk + 4;
    device const uchar * qs = blk + 16;
    int sup = e >> 6, ws = e & 63, hf = ws >> 5, l = ws & 31;
    int is = sup * 2 + hf, qoff = sup * 32;
    uchar sc, m; k4_get_scale_min(is, scales, sc, m);
    uchar nib = (qs[qoff + l] >> (4 * hf)) & 0xF;
    return d * (float)sc * (float)nib - dmin * (float)m;
}
struct GemmDims { uint M; uint N; uint K; };
// C[M,N] = A[M,K](Q4_K) @ B[K,N](half). A row r at A + r*(K/256)*144. M,N,K % 8, K % 256.
// C[M,N] = A[M,K](Q4_K) @ B[K,N](half). Tuned: 4 simdgroups/tg, BM=32 x BN=8 tile,
// sa[32x32] dequantized-once staging shared across M-rows, and the dequant computes
// one Q4_K scale per 8-element chunk (a chunk's 8 cols share one super-block half).
// M must be a multiple of 32, N of 8, K of 256. Bit-identical to the per-element form.
kernel void gemm_q4k_f32(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 8;
    const ulong row_bytes = (ulong)(K / 256) * 144;
    threadgroup half sa[32 * 32];
    threadgroup half sb[32 * 8];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tt = 0; tt < 1; tt++) {
            uint row = tiitg >> 2, col_base = (tiitg & 3) * 8;
            uint e_base = k + col_base;
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 144;
            int ew = int(e_base & 255), sup = ew >> 6, hf = (ew >> 5) & 1, is = sup * 2 + hf;
            float d = (float)(*(device const half *)(blk));
            float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar * scales = blk + 4;
            device const uchar * qs = blk + 16;
            uchar sc, m; k4_get_scale_min(is, scales, sc, m);
            float dl = d * (float)sc, ml = dmin * (float)m;
            int qoff = sup * 32, shift = 4 * hf;
            for (uint j = 0; j < 8; j++) {
                uchar nib = (qs[qoff + col_base + j] >> shift) & 0xF;
                sa[row * 32 + col_base + j] = (half)(dl * (float)nib - ml);
            }
        }
        for (uint t = tiitg; t < 32 * 8; t += 128) {
            uint kr = t >> 3, nc = t & 7;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 32 + ik * 8, 32);
            simdgroup_load(b_tile, sb + ik * 64, 8);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0, N);
}

// Q6_K MMQ (matches ggml dequant_q6_k). Q6_K block = 210 bytes: ql[128],qh[64],sc[16](int8),d(half).
inline float k6_dequant_elem(device const uchar * blk, int e) {
    float d = (float)(*(device const half *)(blk + 208));
    device const uchar * ql = blk;
    device const uchar * qh = blk + 128;
    device const char  * sc = (device const char *)(blk + 192);
    int hf = e >> 7, within = e & 127, l = within & 31, sub = within >> 5, is = l >> 4;
    int qloff = hf * 64, qhoff = hf * 32, scoff = hf * 8;
    int q, scidx;
    if (sub == 0)      { q = ((ql[qloff + l] & 0xF)      | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32; scidx = scoff + is + 0; }
    else if (sub == 1) { q = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32; scidx = scoff + is + 2; }
    else if (sub == 2) { q = ((ql[qloff + l] >> 4)       | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32; scidx = scoff + is + 4; }
    else               { q = ((ql[qloff + l + 32] >> 4)  | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32; scidx = scoff + is + 6; }
    return d * (float)sc[scidx] * (float)q;
}
// C[M,N] = A[M,K](Q6_K) @ B[K,N](half). Tuned (v3): 4 simdgroups/tg, BM=32 x BN=8,
// sa[32x32] staging, one Q6_K scale per 8-element chunk. Bit-identical to per-element form.
kernel void gemm_q6k_f32(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 8;
    const ulong row_bytes = (ulong)(K / 256) * 210;
    threadgroup half sa[32 * 32];
    threadgroup half sb[32 * 8];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tt = 0; tt < 1; tt++) {
            uint row = tiitg >> 2, col_base = (tiitg & 3) * 8;
            uint e_base = k + col_base;
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 210;
            int ew = int(e_base & 255);
            int hf = ew >> 7, sub = (ew >> 5) & 3, is = (ew >> 4) & 1;
            float d = (float)(*(device const half *)(blk + 208));
            device const uchar * ql = blk;
            device const uchar * qh = blk + 128;
            device const char  * sc = (device const char *)(blk + 192);
            float dl = d * (float)sc[hf * 8 + is + 2 * sub];
            int qloff = hf * 64, qhoff = hf * 32;
            for (uint j = 0; j < 8; j++) {
                int l = int(col_base) + int(j);
                int q;
                if (sub == 0)      q = ((ql[qloff + l]      & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32;
                else if (sub == 1) q = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32;
                else if (sub == 2) q = ((ql[qloff + l] >> 4)       | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32;
                else               q = ((ql[qloff + l + 32] >> 4)  | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32;
                sa[row * 32 + col_base + j] = (half)(dl * (float)q);
            }
        }
        for (uint t = tiitg; t < 32 * 8; t += 128) {
            uint kr = t >> 3, nc = t & 7;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 32 + ik * 8, 32);
            simdgroup_load(b_tile, sb + ik * 64, 8);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0, N);
}

// elementwise f32 -> f16 cast (activations must be half for simdgroup_load B)
kernel void cast_f32_f16(device const float * src [[buffer(0)]], device half * dst [[buffer(1)]],
    constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = (half)src[gid];
}
// elementwise f32 copy (load/store a token's residual between batch and scratch)
kernel void copy_f32(device const float * src [[buffer(0)]], device float * dst [[buffer(1)]],
    constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = src[gid];
}
// token-major [T x dim] f32  ->  feature-major [dim x T] f16  (MLP activation staging)
//   dst[i*T + t] = (half) src[t*dim + i];   td = (T, dim)
kernel void transpose_to_f16(device const float * src [[buffer(0)]], device half * dst [[buffer(1)]],
    constant uint2 &td [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint T = td.x, dim = td.y;
    if (gid >= T * dim) return;
    uint t = gid / dim, i = gid % dim;
    dst[i * T + t] = (half)src[gid];
}
// feature-major [dim x T] f32  ->  token-major [T x dim] f32  (w2 output back to residual stream)
//   dst[t*dim + i] = src[i*T + t];   td = (dim, T)
kernel void transpose_f32(device const float * src [[buffer(0)]], device float * dst [[buffer(1)]],
    constant uint2 &td [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint dim = td.x, T = td.y;
    if (gid >= dim * T) return;
    uint i = gid / T, t = gid % T;
    dst[t * dim + i] = src[gid];
}
`

// ---------- metal state ----------
@(private = "file")
m_device: ^MTL.Device
@(private = "file")
m_queue: ^MTL.CommandQueue
@(private = "file")
g_fast_math: bool    // shader fast-math (env QFASTMATH=0 to disable for precision debugging)
@(private = "file")
g_timing: bool       // print per-prefill GPU ms (env QTIMING=1)
@(private = "file")
g_prof_nolin: bool   // ablation: skip linear-layer per-token stateful loop (env QPROF_NOLIN=1; output garbage, GPU time valid)
@(private = "file")
g_prof_nofull: bool  // ablation: skip full-attention per-token block (env QPROF_NOFULL=1)
@(private = "file")
m_weights: ^MTL.Buffer
@(private = "file")
m_mmap_base: uintptr
@(private = "file")
metal_enabled: bool

@(private = "file")
m_pso_f32, m_pso_f16, m_pso_q8_0, m_pso_q4_0, m_pso_q4_1, m_pso_q5_0, m_pso_q5_1, m_pso_q4_k, m_pso_q6_k: ^MTL.ComputePipelineState
@(private = "file")
m_pso_rmsnorm, m_pso_rope, m_pso_qnorm, m_pso_store_kv, m_pso_attn, m_pso_attn_gate, m_pso_swiglu, m_pso_residual: ^MTL.ComputePipelineState
@(private = "file")
m_pso_conv, m_pso_l2norm, m_pso_delta, m_pso_rmsgated: ^MTL.ComputePipelineState
@(private = "file")
m_pso_q4k, m_pso_q6k, m_pso_cast, m_pso_copy, m_pso_tr_h16, m_pso_tr_f32: ^MTL.ComputePipelineState // Stage 1a
@(private = "file")
m_pso_conv_batch, m_pso_conv_update: ^MTL.ComputePipelineState // Stage 2 batched conv1d
@(private = "file")
m_pso_l2norm_batch, m_pso_rmsgated_batch, m_pso_chunked_delta: ^MTL.ComputePipelineState // Stage 2 chunked delta

// activation buffers
@(private = "file")
m_b_x, m_b_xb, m_b_xb2, m_b_qproj, m_b_q, m_b_xb3: ^MTL.Buffer
@(private = "file")
m_b_hb, m_b_hb2, m_b_ktmp, m_b_vtmp, m_b_logits: ^MTL.Buffer
@(private = "file")
m_b_kc, m_b_vc: ^MTL.Buffer
@(private = "file")
m_b_qkv, m_b_qkv2, m_b_z, m_b_b, m_b_a, m_b_qlin, m_b_klin, m_b_lout: ^MTL.Buffer
@(private = "file")
m_b_conv_states, m_b_rec_states: ^MTL.Buffer

// batched-prefill activation buffers ([MAX_BATCH_T x dim/hidden_dim])
@(private = "file")
m_batch_dim, m_batch_hidden, m_batch_max_t: int
@(private = "file")
m_b_bx, m_b_bxb, m_b_bxbh, m_b_bxout: ^MTL.Buffer // [T*dim] x(resid), rmsnorm(x), half(x), w2-out
@(private = "file")
m_b_bhb, m_b_bhb2, m_b_bhb2h: ^MTL.Buffer        // [T*hidden] w1-out, w3/swiglu-out, half(swiglu-out)
// Stage 1b: batched attention projections (per-token stateful ops stay per-token)
@(private = "file")
m_b_proj_half: ^MTL.Buffer                         // [dim x T] half transpose scratch (B for MMQ)
@(private = "file")
m_b_proj_outf: ^MTL.Buffer                         // [max_out x T] f32 MMQ output (feature-major) scratch
@(private = "file")
m_b_qproj_t, m_b_qkv_t: ^MTL.Buffer               // [T x 8192] wq / in_qkv output (token-major)
m_b_qkv2_t: ^MTL.Buffer                            // [T x conv_dim] batched conv1d output (Stage 2)
m_b_qlin_t, m_b_klin_t: ^MTL.Buffer                // [T x key_dim] batched l2norm'd q/k (Stage 2 chunked delta)
@(private = "file")
m_b_kt, m_b_vt: ^MTL.Buffer                        // [T x kv_dim] wk / wv output
@(private = "file")
m_b_zt, m_b_loutt: ^MTL.Buffer                     // [T x value_dim] in_z / linear-out
@(private = "file")
m_b_bt, m_b_at: ^MTL.Buffer                        // [T x n_vh] in_b / in_a
@(private = "file")
m_b_xb3t: ^MTL.Buffer                              // [T x att_head_dim] wo input
@(private = "file")
m_b_xb2t: ^MTL.Buffer                              // [T x dim] post-proj output (residual add)

metal_ready :: proc() -> bool { return metal_enabled }

// Print the quant type of each projection tensor for the first linear and first
// full-attention layer. Makes non-Q4_K tensors (e.g. Q6_K ffn_down/wv/in_qkv in
// Q4_K_M) visible at startup so the batched MMQ dispatch (Q4_K vs Q6_K) is
// obviously covered. Cheap regression guard against the Q6_K-as-Q4_K class of bug.
@(private = "file")
kstr :: proc(k: ggml.GGML_Type) -> string {
	r: string
	#partial switch k {
	case .Q4_K: r = "Q4_K"
	case .Q6_K: r = "Q6_K"
	case .F16: r = "F16"
	case .F32: r = "F32"
	case .Q8_0: r = "Q8_0"
	case: r = "?"
	}
	return r
}

@(private = "file")
print_quant_summary :: proc(t: ^Transformer) {
	w := &t.weights
	for l in 0 ..< t.config.n_layers {
		lw := &w.layers[l]
		if lw.layer_type == .Linear_Attention && t.config.n_linear > 0 {
			la := &lw.linear
			fmt.printf("metal: quant (linear)  w1=%s w3=%s w2=%s  qkv=%s z=%s b=%s a=%s out=%s\n",
				kstr(lw.w1.kind), kstr(lw.w3.kind), kstr(lw.w2.kind),
				kstr(la.in_qkv.kind), kstr(la.in_z.kind), kstr(la.in_b.kind),
				kstr(la.in_a.kind), kstr(la.out.kind))
			break
		}
	}
	for l in 0 ..< t.config.n_layers {
		lw := &w.layers[l]
		if lw.layer_type == .Full_Attention {
			fa := &lw.full
			fmt.printf("metal: quant (full)    w1=%s w3=%s w2=%s  wq=%s wk=%s wv=%s wo=%s\n",
				kstr(lw.w1.kind), kstr(lw.w3.kind), kstr(lw.w2.kind),
				kstr(fa.wq.kind), kstr(fa.wk.kind), kstr(fa.wv.kind), kstr(fa.wo.kind))
			break
		}
	}
}

// Zero the per-request-evolving GPU state (conv_state, recurrent state, KV
// cache) so a fresh prompt can be processed from pos 0. Shared Metal buffers
// are CPU-writable; called between requests (after the previous cmd buffer
// committed), so no race.
metal_reset_state :: proc() {
	if !metal_enabled do return
	for b in ([]^MTL.Buffer{m_b_conv_states, m_b_rec_states, m_b_kc, m_b_vc}) {
		s := b->contentsAsSlice([]u8)
		for i in 0 ..< len(s) {
			s[i] = 0
		}
	}
}

@(private = "file")
make_pso :: proc(lib: ^MTL.Library, name: string) -> ^MTL.ComputePipelineState {
	ns_name := NS.String.alloc()->initWithOdinString(name)
	defer ns_name->release()
	fn := lib->newFunctionWithName(ns_name)
	if fn == nil { fmt.eprintfln("metal: kernel '%s' not found", name); os.exit(1) }
	defer fn->release()
	pso, err := m_device->newComputePipelineStateWithFunction(fn)
	if err != nil { fmt.eprintfln("metal: pipeline '%s' failed: %s", name, err->localizedDescription()->odinString()); os.exit(1) }
	return pso
}

@(private = "file")
new_shared :: proc(n_floats: int) -> ^MTL.Buffer {
	return m_device->newBufferWithLength(NS.UInteger(n_floats * size_of(f32)), MTL.ResourceStorageModeShared)
}

@(private = "file")
new_shared_bytes :: proc(n_bytes: int) -> ^MTL.Buffer {
	return m_device->newBufferWithLength(NS.UInteger(n_bytes), MTL.ResourceStorageModeShared)
}

metal_init :: proc(t: ^Transformer) -> bool {
	g := &t.gguf
	c := t.config

	m_device = MTL.CreateSystemDefaultDevice()
	if m_device == nil { fmt.eprintln("metal: no Metal device available"); return false }
	fmt.printf("metal: %s\n", m_device->name()->odinString())
	m_queue = m_device->newCommandQueue()

	// profiling/debug toggles (env). QFASTMATH=0 -> precise shaders (for f32 drift
	// debugging, e.g. the Stage 2 chunked-delta C=16 case). QTIMING=1 -> per-prefill
	// GPU ms. QPROF_NOLIN=1 -> ablation: skip linear-layer per-token stateful loop.
	g_fast_math = os.get_env("QFASTMATH", context.temp_allocator) != "0"
	g_timing = os.get_env("QTIMING", context.temp_allocator) == "1"
	g_prof_nolin = os.get_env("QPROF_NOLIN", context.temp_allocator) == "1"
	g_prof_nofull = os.get_env("QPROF_NOFULL", context.temp_allocator) == "1"
	fmt.printfln("metal: shader fastMath={} (QFASTMATH=0 to disable)", g_fast_math)
	fmt.printfln("metal: maxThreadgroupMemoryLength={} bytes", m_device->maxThreadgroupMemoryLength())

	src := NS.String.alloc()->initWithOdinString(MSL_SRC)
	defer src->release()
	opts := MTL.CompileOptions.alloc()->init()
	defer opts->release()
	opts->setFastMathEnabled(g_fast_math)
	lib, err := m_device->newLibraryWithSource(src, opts)
	if err != nil {
		fmt.eprintfln("metal: shader compile failed: %s", err->localizedDescription()->odinString())
		return false
	}
	defer lib->release()

	m_pso_f32 = make_pso(lib, "gemv_f32"); m_pso_f16 = make_pso(lib, "gemv_f16")
	m_pso_q8_0 = make_pso(lib, "gemv_q8_0"); m_pso_q4_0 = make_pso(lib, "gemv_q4_0")
	m_pso_q4_1 = make_pso(lib, "gemv_q4_1"); m_pso_q5_0 = make_pso(lib, "gemv_q5_0")
	m_pso_q5_1 = make_pso(lib, "gemv_q5_1"); m_pso_q4_k = make_pso(lib, "gemv_q4_k")
	m_pso_q6_k = make_pso(lib, "gemv_q6_k")
	m_pso_rmsnorm = make_pso(lib, "rmsnorm"); m_pso_rope = make_pso(lib, "rope_partial")
	m_pso_qnorm = make_pso(lib, "qnorm_extract"); m_pso_store_kv = make_pso(lib, "store_kv")
	m_pso_attn = make_pso(lib, "attention"); m_pso_attn_gate = make_pso(lib, "attn_gate")
	m_pso_swiglu = make_pso(lib, "swiglu"); m_pso_residual = make_pso(lib, "residual")
	m_pso_conv = make_pso(lib, "conv1d_step_silu"); m_pso_l2norm = make_pso(lib, "l2norm_scale")
	m_pso_delta = make_pso(lib, "delta_recurrent"); m_pso_rmsgated = make_pso(lib, "rmsnorm_gated")
	m_pso_q4k = make_pso(lib, "gemm_q4k_f32"); m_pso_q6k = make_pso(lib, "gemm_q6k_f32")
	m_pso_cast = make_pso(lib, "cast_f32_f16"); m_pso_copy = make_pso(lib, "copy_f32")
	m_pso_tr_h16 = make_pso(lib, "transpose_to_f16"); m_pso_tr_f32 = make_pso(lib, "transpose_f32")
	m_pso_conv_batch = make_pso(lib, "conv1d_batch_silu"); m_pso_conv_update = make_pso(lib, "conv1d_update_state")
	m_pso_l2norm_batch = make_pso(lib, "l2norm_scale_batch")
	m_pso_rmsgated_batch = make_pso(lib, "rmsnorm_gated_batch")
	m_pso_chunked_delta = make_pso(lib, "chunked_delta")

	base := raw_data(g.mmap)
	m_mmap_base = uintptr(base)
	whole := ([^]u8)(base)[:ggml.align_up(len(g.mmap), PAGE_SIZE)]
	m_weights = m_device->newBufferWithBytesNoCopy(whole, MTL.ResourceStorageModeShared, nil)
	if m_weights == nil { fmt.eprintln("metal: failed to create no-copy weight buffer"); return false }

	att_head_dim := c.n_heads * c.head_dim
	kv_dim := c.n_kv_heads * c.head_dim
	conv_dim := c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
	key_dim := c.lin_n_k_heads * c.lin_head_k_dim
	value_dim := c.lin_n_v_heads * c.lin_head_v_dim

	m_b_x = new_shared(c.dim); m_b_xb = new_shared(c.dim); m_b_xb2 = new_shared(c.dim)
	m_b_qproj = new_shared(c.n_heads * c.head_dim * 2)
	m_b_q = new_shared(att_head_dim); m_b_xb3 = new_shared(att_head_dim)
	m_b_hb = new_shared(c.hidden_dim); m_b_hb2 = new_shared(c.hidden_dim)
	m_b_ktmp = new_shared(kv_dim); m_b_vtmp = new_shared(kv_dim)
	m_b_logits = new_shared(c.vocab_size)
	kv_elems := c.n_full * c.seq_len * kv_dim
	m_b_kc = new_shared_bytes(kv_elems * 2); m_b_vc = new_shared_bytes(kv_elems * 2)
	m_b_qkv = new_shared(conv_dim); m_b_qkv2 = new_shared(conv_dim)
	m_b_z = new_shared(value_dim); m_b_b = new_shared(c.lin_n_v_heads); m_b_a = new_shared(c.lin_n_v_heads)
	m_b_qlin = new_shared(key_dim); m_b_klin = new_shared(key_dim); m_b_lout = new_shared(value_dim)
	m_b_conv_states = new_shared_bytes(c.n_linear * conv_dim * (c.lin_conv_kernel - 1) * 4)
	m_b_rec_states = new_shared_bytes(c.n_linear * c.lin_n_v_heads * c.lin_head_k_dim * c.lin_head_v_dim * 4)
	// conv_state (no history) and recurrent_state (initial state) must start at
	// zero; Metal newBufferWithLength has undefined contents, unlike CPU make([]f32).
	zero_f32_buffer(m_b_conv_states)
	zero_f32_buffer(m_b_rec_states)

	// batched-prefill buffers (Stage 1a: MLP projections). MAX_BATCH_T tokens
	// at once; longer prompts are chunked by the caller. hidden_dim/dim are both
	// multiples of 256 (Q4_K block) and 8 (simdgroup tile).
	m_batch_dim = c.dim
	m_batch_hidden = c.hidden_dim
	m_batch_max_t = MAX_BATCH_T
	mt := MAX_BATCH_T
	m_b_bx = new_shared_bytes(mt * c.dim * 4)
	m_b_bxb = new_shared_bytes(mt * c.dim * 4)
	m_b_bxbh = new_shared_bytes(mt * c.dim * 2)   // half
	m_b_bxout = new_shared_bytes(mt * c.dim * 4)
	m_b_bhb = new_shared_bytes(mt * c.hidden_dim * 4)
	m_b_bhb2 = new_shared_bytes(mt * c.hidden_dim * 4)
	m_b_bhb2h = new_shared_bytes(mt * c.hidden_dim * 2) // half

	// Stage 1b batched attention-projection buffers
	qproj_dim := c.n_heads * c.head_dim * 2 // wq output (q+gate interleaved)
	proj_max_out := max(max(qproj_dim, conv_dim), max(c.dim, value_dim))
	m_b_proj_half = new_shared_bytes(c.dim * mt * 2)              // half [dim x T]
	m_b_proj_outf = new_shared_bytes(proj_max_out * mt * 4)       // f32 [max_out x T]
	m_b_qproj_t = new_shared_bytes(qproj_dim * mt * 4)
	m_b_qkv_t = new_shared_bytes(conv_dim * mt * 4)
	m_b_qkv2_t = new_shared_bytes(conv_dim * mt * 4) // Stage 2 batched conv1d output
	m_b_qlin_t = new_shared_bytes(key_dim * mt * 4)
	m_b_klin_t = new_shared_bytes(key_dim * mt * 4)
	m_b_kt = new_shared_bytes(kv_dim * mt * 4)
	m_b_vt = new_shared_bytes(kv_dim * mt * 4)
	m_b_zt = new_shared_bytes(value_dim * mt * 4)
	m_b_loutt = new_shared_bytes(value_dim * mt * 4)
	m_b_bt = new_shared_bytes(c.lin_n_v_heads * mt * 4)
	m_b_at = new_shared_bytes(c.lin_n_v_heads * mt * 4)
	m_b_xb3t = new_shared_bytes(att_head_dim * mt * 4)
	m_b_xb2t = new_shared_bytes(c.dim * mt * 4)

	print_quant_summary(t)

	metal_enabled = true
	return true
}

@(private = "file")
zero_f32_buffer :: proc(b: ^MTL.Buffer) {
	s := b->contentsAsSlice([]f32)
	for i in 0 ..< len(s) {
		s[i] = 0
	}
}

metal_destroy :: proc() {
	if !metal_enabled do return
	for b in ([]^MTL.Buffer{
		m_b_x, m_b_xb, m_b_xb2, m_b_qproj, m_b_q, m_b_xb3, m_b_hb, m_b_hb2,
		m_b_ktmp, m_b_vtmp, m_b_logits, m_b_kc, m_b_vc,
		m_b_qkv, m_b_qkv2, m_b_z, m_b_b, m_b_a, m_b_qlin, m_b_klin, m_b_lout,
		m_b_conv_states, m_b_rec_states,
		m_b_bx, m_b_bxb, m_b_bxbh, m_b_bxout, m_b_bhb, m_b_bhb2, m_b_bhb2h,
		m_b_proj_half, m_b_proj_outf, m_b_qproj_t, m_b_qkv_t, m_b_qkv2_t, m_b_qlin_t, m_b_klin_t, m_b_kt, m_b_vt,
		m_b_zt, m_b_loutt, m_b_bt, m_b_at, m_b_xb3t, m_b_xb2t,
	}) {
		b->release()
	}
	m_weights->release(); m_queue->release(); m_device->release()
	metal_enabled = false
}

@(private = "file")
pso_for :: proc(k: ggml.GGML_Type) -> ^MTL.ComputePipelineState {
	#partial switch k {
	case .F32: return m_pso_f32
	case .F16: return m_pso_f16
	case .Q8_0: return m_pso_q8_0
	case .Q4_0: return m_pso_q4_0
	case .Q4_1: return m_pso_q4_1
	case .Q5_0: return m_pso_q5_0
	case .Q5_1: return m_pso_q5_1
	case .Q4_K: return m_pso_q4_k
	case .Q6_K: return m_pso_q6_k
	}
	return nil
}

@(private = "file")
bytes_of :: proc(p: rawptr, n: int) -> []u8 { return ([^]u8)(p)[:n] }

@(private = "file")
woff :: proc(p: rawptr) -> NS.UInteger { return NS.UInteger(uintptr(p) - m_mmap_base) }

@(private = "file")
enc_gemv :: proc(enc: ^MTL.ComputeCommandEncoder, kind: ggml.GGML_Type, w_off: NS.UInteger,
	xb: ^MTL.Buffer, x_off: NS.UInteger, ob: ^MTL.Buffer, o_off: NS.UInteger, n, d: int) {
	enc->setComputePipelineState(pso_for(kind))
	enc->setBuffer(m_weights, w_off, 0); enc->setBuffer(xb, x_off, 1); enc->setBuffer(ob, o_off, 2)
	dims := [2]u32{u32(n), u32(d)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	tg_x := (d + GEMV_ROWS - 1) / GEMV_ROWS
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{GEMV_TG, 1, 1})
}

@(private = "file")
enc_rmsnorm :: proc(enc: ^MTL.ComputeCommandEncoder, inp, outp: ^MTL.Buffer, w: rawptr, size, count: int, eps: f32) {
	enc->setComputePipelineState(m_pso_rmsnorm)
	enc->setBuffer(inp, 0, 0); enc->setBuffer(outp, 0, 1); enc->setBuffer(m_weights, woff(w), 2)
	P := struct { size: u32, eps: f32 }{u32(size), eps}
	enc->setBytes(bytes_of(&P, size_of(P)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(count), 1, 1}, MTL.Size{32, 1, 1})
}

@(private = "file")
enc_elementwise :: proc(enc: ^MTL.ComputeCommandEncoder, pso: ^MTL.ComputePipelineState, a, b: ^MTL.Buffer, count: int) {
	enc->setComputePipelineState(pso); enc->setBuffer(a, 0, 0); enc->setBuffer(b, 0, 1)
	tg := min(count, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(count), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

// ---- batched-prefill encoder helpers (Stage 1a) ----
GemmDims :: struct { M, N, K: u32 }

@(private = "file")
enc_cast :: proc(enc: ^MTL.ComputeCommandEncoder, src, dst: ^MTL.Buffer, n: int) {
	enc->setComputePipelineState(m_pso_cast); enc->setBuffer(src, 0, 0); enc->setBuffer(dst, 0, 1)
	nu := u32(n); enc->setBytes(bytes_of(&nu, size_of(nu)), 2)
	tg := min(n, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(n), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

@(private = "file")
enc_copy :: proc(enc: ^MTL.ComputeCommandEncoder, src: ^MTL.Buffer, src_off: NS.UInteger, dst: ^MTL.Buffer, dst_off: NS.UInteger, n: int) {
	enc->setComputePipelineState(m_pso_copy); enc->setBuffer(src, src_off, 0); enc->setBuffer(dst, dst_off, 1)
	nu := u32(n); enc->setBytes(bytes_of(&nu, size_of(nu)), 2)
	tg := min(n, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(n), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

// Q4_K MMQ: C[M,N] = A[M,K] @ B[K,N]. A = Q4_K weights (mmap), B = half activations.
// Tuned kernel: 4 simdgroups (128 threads) compute a 32x8 output tile -> grid y = M/32.
@(private = "file")
enc_q4k :: proc(enc: ^MTL.ComputeCommandEncoder, w: rawptr, B: ^MTL.Buffer, B_off: NS.UInteger, C: ^MTL.Buffer, C_off: NS.UInteger, M, N, K: int) {
	enc->setComputePipelineState(m_pso_q4k)
	enc->setBuffer(m_weights, woff(w), 0); enc->setBuffer(B, B_off, 1); enc->setBuffer(C, C_off, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / 8), NS.Integer(M / 32), 1}, MTL.Size{128, 1, 1})
}

// Q6_K MMQ variant (Q4_K_M stores ffn_down/output/qkv as Q6_K). Tuned (4 sg/tg, M/32).
@(private = "file")
enc_q6k :: proc(enc: ^MTL.ComputeCommandEncoder, w: rawptr, B: ^MTL.Buffer, B_off: NS.UInteger, C: ^MTL.Buffer, C_off: NS.UInteger, M, N, K: int) {
	enc->setComputePipelineState(m_pso_q6k)
	enc->setBuffer(m_weights, woff(w), 0); enc->setBuffer(B, B_off, 1); enc->setBuffer(C, C_off, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / 8), NS.Integer(M / 32), 1}, MTL.Size{128, 1, 1})
}

// dispatch batched GEMM by weight quant kind (Q4_K or Q6_K).
@(private = "file")
enc_mm :: proc(enc: ^MTL.ComputeCommandEncoder, kind: ggml.GGML_Type, w: rawptr, B: ^MTL.Buffer, B_off: NS.UInteger, C: ^MTL.Buffer, C_off: NS.UInteger, M, N, K: int) {
	#partial switch kind {
	case .Q4_K: enc_q4k(enc, w, B, B_off, C, C_off, M, N, K)
	case .Q6_K: enc_q6k(enc, w, B, B_off, C, C_off, M, N, K)
	case:
		fmt.eprintf("enc_mm: unsupported batched quant type %v (need Q4_K/Q6_K)\n", kind)
		os.exit(1)
	}
}

// Batched projection: out_tok[T x M_out] = W[M_out x K_in] @ in_tok[T x K_in].
// in_tok and out_tok are token-major f32. Internally transposes to the feature-
// major half layout the MMQ needs, then back. Uses shared scratch buffers.
@(private = "file")
enc_proj_fwd :: proc(enc: ^MTL.ComputeCommandEncoder, kind: ggml.GGML_Type, w: rawptr,
	in_tok: ^MTL.Buffer, M_out, T, K_in: int, out_tok: ^MTL.Buffer) {
	enc_tr_h16(enc, in_tok, m_b_proj_half, T, K_in)                              // [T x K] -> [K x T] half
	enc_mm(enc, kind, w, m_b_proj_half, 0, m_b_proj_outf, 0, M_out, T, K_in)     // -> [M_out x T] feat-major
	enc_tr_f32(enc, m_b_proj_outf, out_tok, M_out, T)                            // -> [T x M_out] token-major
}

// Batched depthwise causal conv1d + silu (Stage 2). x/out token-major [T x conv_dim].
@(private = "file")
enc_conv1d_batch :: proc(enc: ^MTL.ComputeCommandEncoder, x, out: ^MTL.Buffer, conv_state: ^MTL.Buffer, conv_state_off: NS.UInteger, weight: rawptr, conv_dim, T, kernel: int) {
	enc->setComputePipelineState(m_pso_conv_batch)
	enc->setBuffer(x, 0, 0); enc->setBuffer(out, 0, 1)
	enc->setBuffer(conv_state, conv_state_off, 2); enc->setBuffer(m_weights, woff(weight), 3)
	P := [4]u32{u32(conv_dim), u32(T), u32(kernel), u32(kernel - 1)}
	enc->setBytes(bytes_of(&P, size_of(P)), 4)
	n := T * conv_dim; tg := min(n, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(n), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

// Slide conv_state to the chunk's last km1 inputs.
@(private = "file")
enc_conv1d_update :: proc(enc: ^MTL.ComputeCommandEncoder, x, conv_state: ^MTL.Buffer, conv_state_off: NS.UInteger, conv_dim, T, kernel: int) {
	enc->setComputePipelineState(m_pso_conv_update)
	enc->setBuffer(x, 0, 0); enc->setBuffer(conv_state, conv_state_off, 1)
	P := [4]u32{u32(conv_dim), u32(T), u32(kernel), u32(kernel - 1)}
	enc->setBytes(bytes_of(&P, size_of(P)), 2)
	n := conv_dim * (kernel - 1); tg := min(n, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(n), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

// Stage 2: batched l2norm+scale over [T x n_kh] (q,k slices of conv output)
@(private = "file")
enc_l2norm_batch :: proc(enc: ^MTL.ComputeCommandEncoder, qkv, q_out, k_out: ^MTL.Buffer, hkd, key_dim, conv_dim, T, n_kh: int, scale: f32) {
	enc->setComputePipelineState(m_pso_l2norm_batch)
	enc->setBuffer(q_out, 0, 0); enc->setBuffer(k_out, 0, 1); enc->setBuffer(qkv, 0, 2)
	P := struct { hkd: u32, key_dim: u32, conv_dim: u32, eps: f32, scale: f32 }{u32(hkd), u32(key_dim), u32(conv_dim), 1e-6, scale}
	enc->setBytes(bytes_of(&P, size_of(P)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(T * n_kh), 1, 1}, MTL.Size{32, 1, 1})
}

// Stage 2: batched gated-rmsnorm over [T x n_vh]
@(private = "file")
enc_rmsgated_batch :: proc(enc: ^MTL.ComputeCommandEncoder, outp, inp: ^MTL.Buffer, weight: rawptr, gate: ^MTL.Buffer, head_v_dim, n_vh, T: int, eps: f32) {
	enc->setComputePipelineState(m_pso_rmsgated_batch)
	enc->setBuffer(outp, 0, 0); enc->setBuffer(inp, 0, 1); enc->setBuffer(m_weights, woff(weight), 2); enc->setBuffer(gate, 0, 3)
	P := struct { size: u32, eps: f32 }{u32(head_v_dim), eps}
	enc->setBytes(bytes_of(&P, size_of(P)), 4)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(T * n_vh), 1, 1}, MTL.Size{32, 1, 1})
}

// Stage 2: chunked gated-delta for one chunk of C tokens (grid = n_vh value-heads)
@(private = "file")
enc_chunked_delta :: proc(enc: ^MTL.ComputeCommandEncoder,
	qlin, klin, qkv2, bt, at: ^MTL.Buffer, a_decay, dt_bias: rawptr, rec_states, lout: ^MTL.Buffer,
	hkd, hvd, C, key_dim, conv_dim, value_dim, n_vh, n_kh, chunk_start, slot: int) {
	enc->setComputePipelineState(m_pso_chunked_delta)
	enc->setBuffer(qlin, 0, 0); enc->setBuffer(klin, 0, 1); enc->setBuffer(qkv2, 0, 2)
	enc->setBuffer(bt, 0, 3); enc->setBuffer(at, 0, 4)
	enc->setBuffer(m_weights, woff(a_decay), 5); enc->setBuffer(m_weights, woff(dt_bias), 6)
	enc->setBuffer(rec_states, 0, 7); enc->setBuffer(lout, 0, 8)
	P := struct {
		hkd: u32, hvd: u32, C: u32, key_dim: u32, conv_dim: u32, value_dim: u32, n_vh: u32, n_kh: u32, chunk_start: u32, slot: u32,
	}{u32(hkd), u32(hvd), u32(C), u32(key_dim), u32(conv_dim), u32(value_dim), u32(n_vh), u32(n_kh), u32(chunk_start), u32(slot)}
	enc->setBytes(bytes_of(&P, size_of(P)), 9)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_vh), 1, 1}, MTL.Size{NS.Integer(hvd), 1, 1})
}

@(private = "file")
enc_tr_h16 :: proc(enc: ^MTL.ComputeCommandEncoder, src, dst: ^MTL.Buffer, T, dim: int) {
	enc->setComputePipelineState(m_pso_tr_h16); enc->setBuffer(src, 0, 0); enc->setBuffer(dst, 0, 1)
	td := [2]u32{u32(T), u32(dim)}
	enc->setBytes(bytes_of(&td, size_of(td)), 2)
	n := T * dim; tg := min(n, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(n), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

@(private = "file")
enc_tr_f32 :: proc(enc: ^MTL.ComputeCommandEncoder, src, dst: ^MTL.Buffer, dim, T: int) {
	enc->setComputePipelineState(m_pso_tr_f32); enc->setBuffer(src, 0, 0); enc->setBuffer(dst, 0, 1)
	td := [2]u32{u32(dim), u32(T)}
	enc->setBytes(bytes_of(&td, size_of(td)), 2)
	n := dim * T; tg := min(n, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(n), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

forward_gpu :: proc(transformer: ^Transformer, token: int, pos: int) -> []f32 {
	NS.scoped_autoreleasepool()
	p := &transformer.config
	w := &transformer.weights

	if pos < 0 || pos >= p.seq_len { fmt.eprintf("forward_gpu: pos=%d out of range\n", pos); os.exit(1) }
	if token < 0 || token >= p.vocab_size { fmt.eprintf("forward_gpu: token=%d out of range\n", token); os.exit(1) }

	dim := p.dim; hidden_dim := p.hidden_dim; head_dim := p.head_dim
	n_heads := p.n_heads; n_kv_heads := p.n_kv_heads
	kv_dim := n_kv_heads * head_dim; kv_mul := n_heads / n_kv_heads
	att_head_dim := n_heads * head_dim; seq_len := p.seq_len; eps := p.rms_eps
	rotary_dim := p.rotary_dim
	key_dim := p.lin_n_k_heads * p.lin_head_k_dim
	value_dim := p.lin_n_v_heads * p.lin_head_v_dim
	conv_dim := key_dim * 2 + value_dim
	kernel := p.lin_conv_kernel
	head_k_dim := p.lin_head_k_dim; head_v_dim := p.lin_head_v_dim
	n_vh := p.lin_n_v_heads; n_kh := p.lin_n_k_heads; kh_per_vh := n_vh / n_kh
	l2_scale := 1.0 / math.sqrt_f32(f32(head_k_dim))

	// embedding (CPU into shared x)
	x_slice := m_b_x->contentsAsSlice([]f32)
	get_embedding_row(&w.token_embedding, token, dim, x_slice[:dim])

	cmd := m_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	kv_layer_bytes := NS.UInteger(seq_len * kv_dim * 2)
	conv_state_stride := conv_dim * (kernel - 1)
	rec_state_stride := n_vh * head_k_dim * head_v_dim

	for l in 0 ..< p.n_layers {
		lw := &w.layers[l]
		// shared: input norm + MLP (post_attention_norm)
		enc_rmsnorm(enc, m_b_x, m_b_xb, raw_data(lw.attn_norm), dim, 1, eps)

		switch lw.layer_type {
		case .Full_Attention:
			fa := &lw.full
			slot := transformer.full_slot[l]
			kv_loff := NS.UInteger(slot) * kv_layer_bytes
			enc_gemv(enc, fa.wq.kind, woff(raw_data(fa.wq.data)), m_b_xb, 0, m_b_qproj, 0, dim, n_heads * head_dim * 2)
			enc_gemv(enc, fa.wk.kind, woff(raw_data(fa.wk.data)), m_b_xb, 0, m_b_ktmp, 0, dim, kv_dim)
			enc_gemv(enc, fa.wv.kind, woff(raw_data(fa.wv.data)), m_b_xb, 0, m_b_vtmp, 0, dim, kv_dim)
			// extract + norm q (per head, strided in qproj); norm k (contiguous per kv head)
			enc->setComputePipelineState(m_pso_qnorm)
			enc->setBuffer(m_b_qproj, 0, 0); enc->setBuffer(m_b_q, 0, 1); enc->setBuffer(m_weights, woff(raw_data(fa.q_norm)), 2)
			qnp := struct { size: u32, eps: f32 }{u32(head_dim), eps}
			enc->setBytes(bytes_of(&qnp, size_of(qnp)), 3)
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_heads), 1, 1}, MTL.Size{32, 1, 1})
			enc_rmsnorm(enc, m_b_ktmp, m_b_ktmp, raw_data(fa.k_norm), head_dim, n_kv_heads, eps)
			// partial-rotary MRoPE on q and k
			rope_p := struct { head_dim: u32, pos: u32, rope_freq: f32, rotary_dim: u32 }{u32(head_dim), u32(pos), p.rope_theta, u32(rotary_dim)}
			half_r := rotary_dim / 2
			enc->setComputePipelineState(m_pso_rope); enc->setBuffer(m_b_q, 0, 0)
			enc->setBytes(bytes_of(&rope_p, size_of(rope_p)), 1)
			enc->dispatchThreads(MTL.Size{NS.Integer(n_heads * half_r), 1, 1}, MTL.Size{NS.Integer(min(half_r, 64)), 1, 1})
			enc->setComputePipelineState(m_pso_rope); enc->setBuffer(m_b_ktmp, 0, 0)
			enc->setBytes(bytes_of(&rope_p, size_of(rope_p)), 1)
			enc->dispatchThreads(MTL.Size{NS.Integer(n_kv_heads * half_r), 1, 1}, MTL.Size{NS.Integer(min(half_r, 64)), 1, 1})
			// store K/V (f16), attention, output gate
			enc->setComputePipelineState(m_pso_store_kv)
			enc->setBuffer(m_b_ktmp, 0, 0); enc->setBuffer(m_b_vtmp, 0, 1)
			enc->setBuffer(m_b_kc, kv_loff, 2); enc->setBuffer(m_b_vc, kv_loff, 3)
			store_p := struct { head_dim: u32, seq_len: u32, pos: u32 }{u32(head_dim), u32(seq_len), u32(pos)}
			enc->setBytes(bytes_of(&store_p, size_of(store_p)), 4)
			enc->dispatchThreads(MTL.Size{NS.Integer(kv_dim), 1, 1}, MTL.Size{NS.Integer(min(kv_dim, 256)), 1, 1})
			enc->setComputePipelineState(m_pso_attn)
			enc->setBuffer(m_b_q, 0, 0); enc->setBuffer(m_b_kc, kv_loff, 1); enc->setBuffer(m_b_vc, kv_loff, 2); enc->setBuffer(m_b_xb3, 0, 3)
			attn_p := struct { head_dim: u32, seq_len: u32, kv_mul: u32, pos: u32 }{u32(head_dim), u32(seq_len), u32(kv_mul), u32(pos)}
			enc->setBytes(bytes_of(&attn_p, size_of(attn_p)), 4)
			enc->setThreadgroupMemoryLength(NS.UInteger((pos + 1) * size_of(f32)), 0)
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_heads), 1, 1}, MTL.Size{128, 1, 1})
			enc->setComputePipelineState(m_pso_attn_gate)
			enc->setBuffer(m_b_xb3, 0, 0); enc->setBuffer(m_b_qproj, 0, 1)
			hd_u32 := u32(head_dim)
			enc->setBytes(bytes_of(&hd_u32, size_of(hd_u32)), 2)
			enc->dispatchThreads(MTL.Size{NS.Integer(att_head_dim), 1, 1}, MTL.Size{NS.Integer(min(att_head_dim, 256)), 1, 1})
			enc_gemv(enc, fa.wo.kind, woff(raw_data(fa.wo.data)), m_b_xb3, 0, m_b_xb2, 0, att_head_dim, dim)
			enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb2, dim)

		case .Linear_Attention:
			la := &lw.linear
			slot := transformer.lin_slot[l]
			conv_off := NS.UInteger(slot * conv_state_stride * 4)
			rec_off := NS.UInteger(slot * rec_state_stride * 4)
			enc_gemv(enc, la.in_qkv.kind, woff(raw_data(la.in_qkv.data)), m_b_xb, 0, m_b_qkv, 0, dim, conv_dim)
			enc_gemv(enc, la.in_z.kind, woff(raw_data(la.in_z.data)), m_b_xb, 0, m_b_z, 0, dim, value_dim)
			enc_gemv(enc, la.in_b.kind, woff(raw_data(la.in_b.data)), m_b_xb, 0, m_b_b, 0, dim, n_vh)
			enc_gemv(enc, la.in_a.kind, woff(raw_data(la.in_a.data)), m_b_xb, 0, m_b_a, 0, dim, n_vh)
			// conv1d step + silu (weight is F32 in mmap)
			enc->setComputePipelineState(m_pso_conv)
			enc->setBuffer(m_b_qkv2, 0, 0); enc->setBuffer(m_b_qkv, 0, 1)
			enc->setBuffer(m_b_conv_states, conv_off, 2); enc->setBuffer(m_weights, woff(raw_data(la.conv.data)), 3)
			cp := [2]u32{u32(kernel), 0}
			enc->setBytes(bytes_of(&cp, size_of(cp)), 4)
			enc->dispatchThreads(MTL.Size{NS.Integer(conv_dim), 1, 1}, MTL.Size{NS.Integer(min(conv_dim, 256)), 1, 1})
			// l2norm q,k (q_raw at qkv2[0:key_dim], k_raw at qkv2[key_dim:2*key_dim])
			enc->setComputePipelineState(m_pso_l2norm)
			enc->setBuffer(m_b_qlin, 0, 0); enc->setBuffer(m_b_klin, 0, 1)
			enc->setBuffer(m_b_qkv2, 0, 2); enc->setBuffer(m_b_qkv2, NS.UInteger(key_dim * 4), 3)
			l2p := struct { head_k_dim: u32, eps: f32, scale: f32 }{u32(head_k_dim), 1e-6, l2_scale}
			enc->setBytes(bytes_of(&l2p, size_of(l2p)), 4)
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_kh), 1, 1}, MTL.Size{32, 1, 1})
			// delta recurrent (v = qkv2[2*key_dim:])
			enc->setComputePipelineState(m_pso_delta)
			enc->setBuffer(m_b_lout, 0, 0); enc->setBuffer(m_b_rec_states, rec_off, 1)
			enc->setBuffer(m_b_qlin, 0, 2); enc->setBuffer(m_b_klin, 0, 3)
			enc->setBuffer(m_b_qkv2, NS.UInteger(2 * key_dim * 4), 4)
			enc->setBuffer(m_b_b, 0, 5); enc->setBuffer(m_b_a, 0, 6)
			enc->setBuffer(m_weights, woff(raw_data(la.a_decay)), 7)
			enc->setBuffer(m_weights, woff(raw_data(la.dt_bias)), 8)
			dp := struct { head_k_dim: u32, head_v_dim: u32, n_k_heads: u32 }{u32(head_k_dim), u32(head_v_dim), u32(n_kh)}
			enc->setBytes(bytes_of(&dp, size_of(dp)), 9)
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_vh), 1, 1}, MTL.Size{NS.Integer(head_v_dim), 1, 1})
			// gated rmsnorm (out = rmsnorm(lout) * silu(z))
			enc->setComputePipelineState(m_pso_rmsgated)
			enc->setBuffer(m_b_lout, 0, 0); enc->setBuffer(m_b_lout, 0, 1)
			enc->setBuffer(m_weights, woff(raw_data(la.norm_w)), 2); enc->setBuffer(m_b_z, 0, 3)
			ngp := struct { size: u32, eps: f32 }{u32(head_v_dim), eps}
			enc->setBytes(bytes_of(&ngp, size_of(ngp)), 4)
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_vh), 1, 1}, MTL.Size{32, 1, 1})
			enc_gemv(enc, la.out.kind, woff(raw_data(la.out.data)), m_b_lout, 0, m_b_xb2, 0, value_dim, dim)
			enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb2, dim)
		}

		// shared MLP (post_attention_norm)
		enc_rmsnorm(enc, m_b_x, m_b_xb, raw_data(lw.ffn_norm), dim, 1, eps)
		enc_gemv(enc, lw.w1.kind, woff(raw_data(lw.w1.data)), m_b_xb, 0, m_b_hb, 0, dim, hidden_dim)
		enc_gemv(enc, lw.w3.kind, woff(raw_data(lw.w3.data)), m_b_xb, 0, m_b_hb2, 0, dim, hidden_dim)
		enc_elementwise(enc, m_pso_swiglu, m_b_hb, m_b_hb2, hidden_dim)
		enc_gemv(enc, lw.w2.kind, woff(raw_data(lw.w2.data)), m_b_hb, 0, m_b_xb, 0, hidden_dim, dim)
		enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb, dim)
	}

	enc_rmsnorm(enc, m_b_x, m_b_x, raw_data(w.output_norm), dim, 1, eps)
	enc_gemv(enc, w.output.kind, woff(raw_data(w.output.data)), m_b_x, 0, m_b_logits, 0, dim, p.vocab_size)

	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
	return m_b_logits->contentsAsSlice([]f32)[:p.vocab_size]
}

// Batched prefill (BATCHED_PREFILL Stage 1a): process T tokens at once, batching
// the MLP projections (w1/w3/w2) via Q4_K MMQ. Attention (full + linear) still
// runs per-token on single-token scratch (its projections are GEMV; batching them
// is Stage 1b). T must be a multiple of 8 and <= MAX_BATCH_T (caller chunks; the
// trailing <8 remainder goes through forward_gpu). Returns the LAST token's logits.
forward_gpu_batch :: proc(transformer: ^Transformer, tokens: []int, pos_start: int) -> []f32 {
	NS.scoped_autoreleasepool()
	p := &transformer.config
	w := &transformer.weights

	T := len(tokens)
	if T == 0 { fmt.eprintln("forward_gpu_batch: empty"); os.exit(1) }
	if T > MAX_BATCH_T || (T % 8) != 0 {
		fmt.eprintf("forward_gpu_batch: T=%d must be in (0,%d] and %%8==0\n", T, MAX_BATCH_T)
		os.exit(1)
	}

	dim := p.dim; hidden_dim := p.hidden_dim; head_dim := p.head_dim
	n_heads := p.n_heads; n_kv_heads := p.n_kv_heads
	kv_dim := n_kv_heads * head_dim; kv_mul := n_heads / n_kv_heads
	att_head_dim := n_heads * head_dim; seq_len := p.seq_len; eps := p.rms_eps
	rotary_dim := p.rotary_dim
	key_dim := p.lin_n_k_heads * p.lin_head_k_dim
	value_dim := p.lin_n_v_heads * p.lin_head_v_dim
	conv_dim := key_dim * 2 + value_dim
	kernel := p.lin_conv_kernel
	head_k_dim := p.lin_head_k_dim; head_v_dim := p.lin_head_v_dim
	n_vh := p.lin_n_v_heads; n_kh := p.lin_n_k_heads
	l2_scale := 1.0 / math.sqrt_f32(f32(head_k_dim))

	// CPU: token embeddings into batch residual (token-major [T x dim])
	bx := m_b_bx->contentsAsSlice([]f32)
	for t in 0 ..< T {
		get_embedding_row(&w.token_embedding, tokens[t], dim, bx[t * dim : (t + 1) * dim])
	}

	cmd := m_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	kv_layer_bytes := NS.UInteger(seq_len * kv_dim * 2)
	conv_state_stride := conv_dim * (kernel - 1)
	rec_state_stride := n_vh * head_k_dim * head_v_dim
	dim_bytes := NS.UInteger(dim * 4)

	for l in 0 ..< p.n_layers {
		lw := &w.layers[l]

		// ---- batched attn_norm + attention projections (Stage 1b) ----
		enc_rmsnorm(enc, m_b_bx, m_b_bxb, raw_data(lw.attn_norm), dim, T, eps) // [T x dim]

		switch lw.layer_type {
		case .Full_Attention:
			fa := &lw.full
			qproj_dim := n_heads * head_dim * 2
			enc_proj_fwd(enc, fa.wq.kind, raw_data(fa.wq.data), m_b_bxb, qproj_dim, T, dim, m_b_qproj_t)
			enc_proj_fwd(enc, fa.wk.kind, raw_data(fa.wk.data), m_b_bxb, kv_dim, T, dim, m_b_kt)
			enc_proj_fwd(enc, fa.wv.kind, raw_data(fa.wv.data), m_b_bxb, kv_dim, T, dim, m_b_vt)
			// per-token stateful: qnorm / rope / store_kv / attention / gate
			slot := transformer.full_slot[l]
			kv_loff := NS.UInteger(slot) * kv_layer_bytes
			qproj_b := NS.UInteger(qproj_dim * 4); kv_b := NS.UInteger(kv_dim * 4); xb3_b := NS.UInteger(att_head_dim * 4)
			if !g_prof_nofull { // QPROF_NOFULL ablation
			for t in 0 ..< T {
				pos := pos_start + t
				oq := NS.UInteger(t) * qproj_b; okv := NS.UInteger(t) * kv_b; oxb3 := NS.UInteger(t) * xb3_b
				enc_copy(enc, m_b_qproj_t, oq, m_b_qproj, 0, qproj_dim)
				enc_copy(enc, m_b_kt, okv, m_b_ktmp, 0, kv_dim)
				enc_copy(enc, m_b_vt, okv, m_b_vtmp, 0, kv_dim)
				enc->setComputePipelineState(m_pso_qnorm)
				enc->setBuffer(m_b_qproj, 0, 0); enc->setBuffer(m_b_q, 0, 1); enc->setBuffer(m_weights, woff(raw_data(fa.q_norm)), 2)
				qnp := struct { size: u32, eps: f32 }{u32(head_dim), eps}
				enc->setBytes(bytes_of(&qnp, size_of(qnp)), 3)
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_heads), 1, 1}, MTL.Size{32, 1, 1})
				enc_rmsnorm(enc, m_b_ktmp, m_b_ktmp, raw_data(fa.k_norm), head_dim, n_kv_heads, eps)
				rope_p := struct { head_dim: u32, pos: u32, rope_freq: f32, rotary_dim: u32 }{u32(head_dim), u32(pos), p.rope_theta, u32(rotary_dim)}
				half_r := rotary_dim / 2
				enc->setComputePipelineState(m_pso_rope); enc->setBuffer(m_b_q, 0, 0)
				enc->setBytes(bytes_of(&rope_p, size_of(rope_p)), 1)
				enc->dispatchThreads(MTL.Size{NS.Integer(n_heads * half_r), 1, 1}, MTL.Size{NS.Integer(min(half_r, 64)), 1, 1})
				enc->setComputePipelineState(m_pso_rope); enc->setBuffer(m_b_ktmp, 0, 0)
				enc->setBytes(bytes_of(&rope_p, size_of(rope_p)), 1)
				enc->dispatchThreads(MTL.Size{NS.Integer(n_kv_heads * half_r), 1, 1}, MTL.Size{NS.Integer(min(half_r, 64)), 1, 1})
				enc->setComputePipelineState(m_pso_store_kv)
				enc->setBuffer(m_b_ktmp, 0, 0); enc->setBuffer(m_b_vtmp, 0, 1)
				enc->setBuffer(m_b_kc, kv_loff, 2); enc->setBuffer(m_b_vc, kv_loff, 3)
				store_p := struct { head_dim: u32, seq_len: u32, pos: u32 }{u32(head_dim), u32(seq_len), u32(pos)}
				enc->setBytes(bytes_of(&store_p, size_of(store_p)), 4)
				enc->dispatchThreads(MTL.Size{NS.Integer(kv_dim), 1, 1}, MTL.Size{NS.Integer(min(kv_dim, 256)), 1, 1})
				enc->setComputePipelineState(m_pso_attn)
				enc->setBuffer(m_b_q, 0, 0); enc->setBuffer(m_b_kc, kv_loff, 1); enc->setBuffer(m_b_vc, kv_loff, 2); enc->setBuffer(m_b_xb3, 0, 3)
				attn_p := struct { head_dim: u32, seq_len: u32, kv_mul: u32, pos: u32 }{u32(head_dim), u32(seq_len), u32(kv_mul), u32(pos)}
				enc->setBytes(bytes_of(&attn_p, size_of(attn_p)), 4)
				enc->setThreadgroupMemoryLength(NS.UInteger((pos + 1) * size_of(f32)), 0)
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_heads), 1, 1}, MTL.Size{128, 1, 1})
				enc->setComputePipelineState(m_pso_attn_gate)
				enc->setBuffer(m_b_xb3, 0, 0); enc->setBuffer(m_b_qproj, 0, 1)
				hd_u32 := u32(head_dim)
				enc->setBytes(bytes_of(&hd_u32, size_of(hd_u32)), 2)
				enc->dispatchThreads(MTL.Size{NS.Integer(att_head_dim), 1, 1}, MTL.Size{NS.Integer(min(att_head_dim, 256)), 1, 1})
				enc_copy(enc, m_b_xb3, 0, m_b_xb3t, oxb3, att_head_dim)
			}
			} // end QPROF_NOFULL guard
			// batched wo -> residual
			enc_proj_fwd(enc, fa.wo.kind, raw_data(fa.wo.data), m_b_xb3t, dim, T, att_head_dim, m_b_xb2t)
			enc_elementwise(enc, m_pso_residual, m_b_bx, m_b_xb2t, T * dim)

		case .Linear_Attention:
			la := &lw.linear
			enc_proj_fwd(enc, la.in_qkv.kind, raw_data(la.in_qkv.data), m_b_bxb, conv_dim, T, dim, m_b_qkv_t)
			enc_proj_fwd(enc, la.in_z.kind, raw_data(la.in_z.data), m_b_bxb, value_dim, T, dim, m_b_zt)
			enc_proj_fwd(enc, la.in_b.kind, raw_data(la.in_b.data), m_b_bxb, n_vh, T, dim, m_b_bt)
			enc_proj_fwd(enc, la.in_a.kind, raw_data(la.in_a.data), m_b_bxb, n_vh, T, dim, m_b_at)
			// batched conv1d (Stage 2): all T tokens at once, causal left-pad from conv_state
			slot := transformer.lin_slot[l]
			conv_off := NS.UInteger(slot * conv_state_stride * 4)
			rec_off := NS.UInteger(slot * rec_state_stride * 4)
			enc_conv1d_batch(enc, m_b_qkv_t, m_b_qkv2_t, m_b_conv_states, conv_off, raw_data(la.conv.data), conv_dim, T, kernel)
			// NOTE: a chunked_delta integration (Stage 2) was validated in isolation
			// (delta_harness) but the v1 engine wiring neither sped up prefill nor stayed
			// bit-identical, and C=16 had an unresolved failure. It is reverted to the
			// per-token delta loop below until the fused (S-in-threadgroup) version lands.
			qkv_b := NS.UInteger(conv_dim * 4); z_b := NS.UInteger(value_dim * 4); ba_b := NS.UInteger(n_vh * 4); lout_b := NS.UInteger(value_dim * 4)
			if !g_prof_nolin { // QPROF_NOLIN ablation: skip to measure linear per-token stateful share
			for t in 0 ..< T {
				oqkv := NS.UInteger(t) * qkv_b; oz := NS.UInteger(t) * z_b; oba := NS.UInteger(t) * ba_b; olout := NS.UInteger(t) * lout_b
				enc_copy(enc, m_b_qkv2_t, oqkv, m_b_qkv2, 0, conv_dim)
				enc_copy(enc, m_b_zt, oz, m_b_z, 0, value_dim)
				enc_copy(enc, m_b_bt, oba, m_b_b, 0, n_vh)
				enc_copy(enc, m_b_at, oba, m_b_a, 0, n_vh)
				enc->setComputePipelineState(m_pso_l2norm)
				enc->setBuffer(m_b_qlin, 0, 0); enc->setBuffer(m_b_klin, 0, 1)
				enc->setBuffer(m_b_qkv2, 0, 2); enc->setBuffer(m_b_qkv2, NS.UInteger(key_dim * 4), 3)
				l2p := struct { head_k_dim: u32, eps: f32, scale: f32 }{u32(head_k_dim), 1e-6, l2_scale}
				enc->setBytes(bytes_of(&l2p, size_of(l2p)), 4)
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_kh), 1, 1}, MTL.Size{32, 1, 1})
				enc->setComputePipelineState(m_pso_delta)
				enc->setBuffer(m_b_lout, 0, 0); enc->setBuffer(m_b_rec_states, rec_off, 1)
				enc->setBuffer(m_b_qlin, 0, 2); enc->setBuffer(m_b_klin, 0, 3)
				enc->setBuffer(m_b_qkv2, NS.UInteger(2 * key_dim * 4), 4)
				enc->setBuffer(m_b_b, 0, 5); enc->setBuffer(m_b_a, 0, 6)
				enc->setBuffer(m_weights, woff(raw_data(la.a_decay)), 7)
				enc->setBuffer(m_weights, woff(raw_data(la.dt_bias)), 8)
				dp := struct { head_k_dim: u32, head_v_dim: u32, n_k_heads: u32 }{u32(head_k_dim), u32(head_v_dim), u32(n_kh)}
				enc->setBytes(bytes_of(&dp, size_of(dp)), 9)
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_vh), 1, 1}, MTL.Size{NS.Integer(head_v_dim), 1, 1})
				enc->setComputePipelineState(m_pso_rmsgated)
				enc->setBuffer(m_b_lout, 0, 0); enc->setBuffer(m_b_lout, 0, 1)
				enc->setBuffer(m_weights, woff(raw_data(la.norm_w)), 2); enc->setBuffer(m_b_z, 0, 3)
				ngp := struct { size: u32, eps: f32 }{u32(head_v_dim), eps}
				enc->setBytes(bytes_of(&ngp, size_of(ngp)), 4)
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_vh), 1, 1}, MTL.Size{32, 1, 1})
				enc_copy(enc, m_b_lout, 0, m_b_loutt, olout, value_dim)
			}
			} // end QPROF_NOLIN guard
			enc_conv1d_update(enc, m_b_qkv_t, m_b_conv_states, conv_off, conv_dim, T, kernel)
			enc_proj_fwd(enc, la.out.kind, raw_data(la.out.data), m_b_loutt, dim, T, value_dim, m_b_xb2t)
			enc_elementwise(enc, m_pso_residual, m_b_bx, m_b_xb2t, T * dim)
		}

		// ---- batched MLP over all T tokens (Stage 1a) ----
		enc_rmsnorm(enc, m_b_bx, m_b_bxb, raw_data(lw.ffn_norm), dim, T, eps)            // [T x dim] token-major
		enc_tr_h16(enc, m_b_bxb, m_b_bxbh, T, dim)                                      // -> [dim x T] half
		enc_mm(enc, lw.w1.kind, raw_data(lw.w1.data), m_b_bxbh, 0, m_b_bhb, 0, hidden_dim, T, dim)   // [hidden x T]
		enc_mm(enc, lw.w3.kind, raw_data(lw.w3.data), m_b_bxbh, 0, m_b_bhb2, 0, hidden_dim, T, dim)  // [hidden x T]
		enc_elementwise(enc, m_pso_swiglu, m_b_bhb, m_b_bhb2, T * hidden_dim)           // hb = silu(hb)*hb2
		enc_cast(enc, m_b_bhb, m_b_bhb2h, T * hidden_dim)                               // [hidden x T] f32 -> half
		enc_mm(enc, lw.w2.kind, raw_data(lw.w2.data), m_b_bhb2h, 0, m_b_bxout, 0, dim, T, hidden_dim) // [dim x T]
		enc_tr_f32(enc, m_b_bxout, m_b_bxb, dim, T)                                     // -> [T x dim] token-major
		enc_elementwise(enc, m_pso_residual, m_b_bx, m_b_bxb, T * dim)                  // batch_x += mlp_out
	}

	// final norm + output projection: LAST token only
	last_off := NS.UInteger(T - 1) * dim_bytes
	enc_copy(enc, m_b_bx, last_off, m_b_x, 0, dim)
	enc_rmsnorm(enc, m_b_x, m_b_x, raw_data(w.output_norm), dim, 1, eps)
	enc_gemv(enc, w.output.kind, woff(raw_data(w.output.data)), m_b_x, 0, m_b_logits, 0, dim, p.vocab_size)

	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
	if g_timing {
		gpu_ms := f64(cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0
		tag := ""
		if g_prof_nolin { tag = " [QPROF_NOLIN]" }
		if g_prof_nofull { tag = " [QPROF_NOFULL]" }
		fmt.eprintfln("[timing] forward_gpu_batch T={} GPU={:.2f} ms ({:.2f} ms/tok){}", len(tokens), gpu_ms, gpu_ms / f64(len(tokens)), tag)
	}
	return m_b_logits->contentsAsSlice([]f32)[:p.vocab_size]
}
