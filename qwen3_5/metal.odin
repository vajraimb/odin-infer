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
`

// ---------- metal state ----------
@(private = "file")
m_device: ^MTL.Device
@(private = "file")
m_queue: ^MTL.CommandQueue
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

metal_ready :: proc() -> bool { return metal_enabled }

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

	src := NS.String.alloc()->initWithOdinString(MSL_SRC)
	defer src->release()
	lib, err := m_device->newLibraryWithSource(src, nil)
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
