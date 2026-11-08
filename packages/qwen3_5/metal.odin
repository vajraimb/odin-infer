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
import "core:mem"
import "core:strings"
import "core:os"
import "core:strconv"
import "core:time"
import MTL "vendor:darwin/Metal"

PAGE_SIZE :: 16384
GEMV_ROWS :: 32 // rows per TG; more rows/TG = fewer TG launches (output layer has 248k rows)
GEMV_TG :: 256
MAX_BATCH_T :: 512 // max tokens per batched-prefill chunk (Stage 1a)

MSL_SRC := `
#include <metal_stdlib>
using namespace metal;

struct Dims { uint n; uint d; };
constant uint GEMV_ROWS   = 32;  // must match the Odin-side GEMV_ROWS (dispatch shape)
constant uint GEMV_SG     = 8;   // lanes per row; rows = GEMV_TG / GEMV_SG

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
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_f16(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { device const half *w = (device const half*)wb + (ulong)row * dim.n;
        for (uint j = lane; j < dim.n; j += GEMV_SG) sum += (float)w[j] * x[j]; }
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
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
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
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
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
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
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
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
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
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
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}
kernel void gemv_q4_k(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    // v2: float4 dot() accumulation with two chains (lo/hi nibble groups) per row.
    // Same products as v1; only the addition grouping changed -> 8x shorter
    // dependency chains. Row mapping unchanged (1 simdgroup per row).
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 144;
        float slo = 0.0, shi = 0.0;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 144;
            float dall = (float)(*(device const half*)blk); float dmin = (float)(*(device const half*)(blk + 2));
            device const uchar *scales = blk + 4; device const uchar *qs = blk + 16; uint ybase = sb * 256; int is = 0; uint qoff = 0;
            for (uint j = 0; j < 256; j += 64) { uchar sc, m; get_scale_min_k4(is + 0, scales, sc, m); float d1 = dall * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, scales, sc, m); float d2 = dall * sc, m2 = dmin * m;
                for (uint l = 0; l < 32; l += 4) {
                    float4 wl = float4((float)(qs[qoff+l]&0xF),(float)(qs[qoff+l+1]&0xF),(float)(qs[qoff+l+2]&0xF),(float)(qs[qoff+l+3]&0xF));
                    float4 wh = float4((float)(qs[qoff+l]>>4),(float)(qs[qoff+l+1]>>4),(float)(qs[qoff+l+2]>>4),(float)(qs[qoff+l+3]>>4));
                    float4 xl = float4(x[ybase+j+l],x[ybase+j+l+1],x[ybase+j+l+2],x[ybase+j+l+3]);
                    float4 xh = float4(x[ybase+j+32+l],x[ybase+j+32+l+1],x[ybase+j+32+l+2],x[ybase+j+32+l+3]);
                    slo += dot(wl * d1 - m1, xl);
                    shi += dot(wh * d2 - m2, xh);
                }
                qoff += 32; is += 2; } }
        sum = slo + shi; }
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}
// ============ norm-fused GEMV (QGEMVNORM=1) ============
// Each dispatch recomputes rms(x) cooperatively (256 threads) and folds the
// norm weight into the activation: y[j] = x[j] * wn[j] * rms. Eliminates the
// 64 standalone 32-thread rmsnorm dispatches per token and the xb round trip.
kernel void gemv_q4_k_norm(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    device const float *wn [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float red[256];
    float ss = 0.0;
    for (uint j = tid; j < dim.n; j += 256) { float v = x[j]; ss += v * v; }
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sh = 128; sh > 0; sh >>= 1) {
        if (tid < sh) { red[tid] += red[tid + sh]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(red[0] / (float)dim.n + eps);

    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 144;
        float slo = 0.0, shi = 0.0;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 144;
            float dall = (float)(*(device const half*)blk); float dmin = (float)(*(device const half*)(blk + 2));
            device const uchar *scales = blk + 4; device const uchar *qs = blk + 16; uint ybase = sb * 256; int is = 0; uint qoff = 0;
            for (uint j = 0; j < 256; j += 64) { uchar sc, m; get_scale_min_k4(is + 0, scales, sc, m); float d1 = dall * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, scales, sc, m); float d2 = dall * sc, m2 = dmin * m;
                for (uint l = 0; l < 32; l += 4) {
                    float4 wl = float4((float)(qs[qoff+l]&0xF),(float)(qs[qoff+l+1]&0xF),(float)(qs[qoff+l+2]&0xF),(float)(qs[qoff+l+3]&0xF));
                    float4 wh = float4((float)(qs[qoff+l]>>4),(float)(qs[qoff+l+1]>>4),(float)(qs[qoff+l+2]>>4),(float)(qs[qoff+l+3]>>4));
                    float4 xl = float4(x[ybase+j+l]*wn[ybase+j+l]*rms, x[ybase+j+l+1]*wn[ybase+j+l+1]*rms, x[ybase+j+l+2]*wn[ybase+j+l+2]*rms, x[ybase+j+l+3]*wn[ybase+j+l+3]*rms);
                    float4 xh = float4(x[ybase+j+32+l]*wn[ybase+j+32+l]*rms, x[ybase+j+32+l+1]*wn[ybase+j+32+l+1]*rms, x[ybase+j+32+l+2]*wn[ybase+j+32+l+2]*rms, x[ybase+j+32+l+3]*wn[ybase+j+32+l+3]*rms);
                    slo += dot(wl * d1 - m1, xl);
                    shi += dot(wh * d2 - m2, xh);
                }
                qoff += 32; is += 2; } }
        sum = slo + shi; }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

inline float2 dot_q6_superblock2n(device const uchar *blk, device const float *x, device const float *wn, float rms, uint ybase) {
    device const uchar *ql = blk; device const uchar *qh = blk + 128; device const char *sc = (device const char*)(blk + 192);
    float dall = (float)(*(device const half*)(blk + 208)); float s0 = 0.0, s1 = 0.0;
    for (int hf = 0; hf < 2; ++hf) { uint qlo = hf * 64, qho = hf * 32, sco = hf * 8, yo = ybase + hf * 128;
        float ds0=dall*(float)sc[sco+0],ds1=dall*(float)sc[sco+1],ds2=dall*(float)sc[sco+2],ds3=dall*(float)sc[sco+3];
        float ds4=dall*(float)sc[sco+4],ds5=dall*(float)sc[sco+5],ds6=dall*(float)sc[sco+6],ds7=dall*(float)sc[sco+7];
        for (int l = 0; l < 16; l += 4) {
            float4 w1=float4((float)((int)((ql[qlo+l+0]&0xF)|(((qh[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+1]&0xF)|(((qh[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+2]&0xF)|(((qh[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+3]&0xF)|(((qh[qho+l+3]>>0)&3)<<4))-32));
            float4 w2=float4((float)((int)((ql[qlo+l+32]&0xF)|(((qh[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+33]&0xF)|(((qh[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+34]&0xF)|(((qh[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+35]&0xF)|(((qh[qho+l+3]>>2)&3)<<4))-32));
            float4 w3=float4((float)((int)((ql[qlo+l+0]>>4)|(((qh[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+1]>>4)|(((qh[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+2]>>4)|(((qh[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+3]>>4)|(((qh[qho+l+3]>>4)&3)<<4))-32));
            float4 w4=float4((float)((int)((ql[qlo+l+32]>>4)|(((qh[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+33]>>4)|(((qh[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+34]>>4)|(((qh[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+35]>>4)|(((qh[qho+l+3]>>6)&3)<<4))-32));
            float4 xv1=float4(x[yo+l]*wn[yo+l]*rms,x[yo+l+1]*wn[yo+l+1]*rms,x[yo+l+2]*wn[yo+l+2]*rms,x[yo+l+3]*wn[yo+l+3]*rms);
            float4 xv2=float4(x[yo+l+32]*wn[yo+l+32]*rms,x[yo+l+33]*wn[yo+l+33]*rms,x[yo+l+34]*wn[yo+l+34]*rms,x[yo+l+35]*wn[yo+l+35]*rms);
            float4 xv3=float4(x[yo+l+64]*wn[yo+l+64]*rms,x[yo+l+65]*wn[yo+l+65]*rms,x[yo+l+66]*wn[yo+l+66]*rms,x[yo+l+67]*wn[yo+l+67]*rms);
            float4 xv4=float4(x[yo+l+96]*wn[yo+l+96]*rms,x[yo+l+97]*wn[yo+l+97]*rms,x[yo+l+98]*wn[yo+l+98]*rms,x[yo+l+99]*wn[yo+l+99]*rms);
            s0 += ds0*dot(w1,xv1)+ds2*dot(w2,xv2)+ds4*dot(w3,xv3)+ds6*dot(w4,xv4);
        }
        for (int l = 16; l < 32; l += 4) {
            float4 w1=float4((float)((int)((ql[qlo+l+0]&0xF)|(((qh[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+1]&0xF)|(((qh[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+2]&0xF)|(((qh[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+3]&0xF)|(((qh[qho+l+3]>>0)&3)<<4))-32));
            float4 w2=float4((float)((int)((ql[qlo+l+32]&0xF)|(((qh[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+33]&0xF)|(((qh[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+34]&0xF)|(((qh[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+35]&0xF)|(((qh[qho+l+3]>>2)&3)<<4))-32));
            float4 w3=float4((float)((int)((ql[qlo+l+0]>>4)|(((qh[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+1]>>4)|(((qh[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+2]>>4)|(((qh[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+3]>>4)|(((qh[qho+l+3]>>4)&3)<<4))-32));
            float4 w4=float4((float)((int)((ql[qlo+l+32]>>4)|(((qh[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+33]>>4)|(((qh[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+34]>>4)|(((qh[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+35]>>4)|(((qh[qho+l+3]>>6)&3)<<4))-32));
            float4 xv1=float4(x[yo+l]*wn[yo+l]*rms,x[yo+l+1]*wn[yo+l+1]*rms,x[yo+l+2]*wn[yo+l+2]*rms,x[yo+l+3]*wn[yo+l+3]*rms);
            float4 xv2=float4(x[yo+l+32]*wn[yo+l+32]*rms,x[yo+l+33]*wn[yo+l+33]*rms,x[yo+l+34]*wn[yo+l+34]*rms,x[yo+l+35]*wn[yo+l+35]*rms);
            float4 xv3=float4(x[yo+l+64]*wn[yo+l+64]*rms,x[yo+l+65]*wn[yo+l+65]*rms,x[yo+l+66]*wn[yo+l+66]*rms,x[yo+l+67]*wn[yo+l+67]*rms);
            float4 xv4=float4(x[yo+l+96]*wn[yo+l+96]*rms,x[yo+l+97]*wn[yo+l+97]*rms,x[yo+l+98]*wn[yo+l+98]*rms,x[yo+l+99]*wn[yo+l+99]*rms);
            s1 += ds1*dot(w1,xv1)+ds3*dot(w2,xv2)+ds5*dot(w3,xv3)+ds7*dot(w4,xv4);
        }
    }
    return float2(s0, s1);
}

kernel void gemv_q6_k_norm(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    device const float *wn [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float red[256];
    float ss = 0.0;
    for (uint j = tid; j < dim.n; j += 256) { float v = x[j]; ss += v * v; }
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sh = 128; sh > 0; sh >>= 1) {
        if (tid < sh) { red[tid] += red[tid + sh]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(red[0] / (float)dim.n + eps);

    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 210;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) {
            float2 acc = dot_q6_superblock2n(wr + sb * 210, x, wn, rms, sb * 256);
            sum += acc.x + acc.y; } }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

inline float2 dot_q6_superblock2(device const uchar *blk, device const float *x, uint ybase) {
    device const uchar *ql = blk; device const uchar *qh = blk + 128; device const char *sc = (device const char*)(blk + 192);
    float dall = (float)(*(device const half*)(blk + 208)); float s0 = 0.0, s1 = 0.0;
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
            s0 += ds0*dot(w1,xv1)+ds2*dot(w2,xv2)+ds4*dot(w3,xv3)+ds6*dot(w4,xv4);
        }
        for (int l = 16; l < 32; l += 4) {
            float4 w1=float4((float)((int)((ql[qlo+l+0]&0xF)|(((qh[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+1]&0xF)|(((qh[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+2]&0xF)|(((qh[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql[qlo+l+3]&0xF)|(((qh[qho+l+3]>>0)&3)<<4))-32));
            float4 w2=float4((float)((int)((ql[qlo+l+32]&0xF)|(((qh[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+33]&0xF)|(((qh[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+34]&0xF)|(((qh[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql[qlo+l+35]&0xF)|(((qh[qho+l+3]>>2)&3)<<4))-32));
            float4 w3=float4((float)((int)((ql[qlo+l+0]>>4)|(((qh[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+1]>>4)|(((qh[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+2]>>4)|(((qh[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql[qlo+l+3]>>4)|(((qh[qho+l+3]>>4)&3)<<4))-32));
            float4 w4=float4((float)((int)((ql[qlo+l+32]>>4)|(((qh[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+33]>>4)|(((qh[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+34]>>4)|(((qh[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql[qlo+l+35]>>4)|(((qh[qho+l+3]>>6)&3)<<4))-32));
            float4 xv1=float4(x[yo+l],x[yo+l+1],x[yo+l+2],x[yo+l+3]); float4 xv2=float4(x[yo+l+32],x[yo+l+33],x[yo+l+34],x[yo+l+35]);
            float4 xv3=float4(x[yo+l+64],x[yo+l+65],x[yo+l+66],x[yo+l+67]); float4 xv4=float4(x[yo+l+96],x[yo+l+97],x[yo+l+98],x[yo+l+99]);
            s1 += ds1*dot(w1,xv1)+ds3*dot(w2,xv2)+ds5*dot(w3,xv3)+ds7*dot(w4,xv4);
        }
    }
    return float2(s0, s1);
}
kernel void gemv_q6_k(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 210;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) {
            float2 acc = dot_q6_superblock2(wr + sb * 210, x, sb * 256);
            sum += acc.x + acc.y; } }
    // 16-lane rows share a simdgroup, so reduce within each 16-lane half only
    // (simd_sum would mix the two rows sharing this simdgroup).
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

// ============ GEMV v4: 2 rows per lane, 8-lane cooperation (QGEMV2R=1) ============
// 64 rows per TG (256 threads / 8 lanes per row-pair * 2 rows). Same chunk
// dequant math as the v2 kernels; each lane accumulates both rows of its pair.
kernel void gemv_q4_k_2r(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint grp = tid / 8, lane = tid % 8;
    uint row0 = tg * 64 + grp * 2;
    if (row0 >= dim.d) return;
    uint nsb = dim.n / 256;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 144;
    device const uchar *wr1 = (row0 + 1 < dim.d) ? wr0 + nsb * 144 : wr0;
    float s0l = 0.0, s0h = 0.0, s1l = 0.0, s1h = 0.0;
    for (uint sb = lane; sb < nsb; sb += 8) {
        uint ybase = sb * 256;
        device const uchar *blk0 = wr0 + sb * 144;
        device const uchar *blk1 = wr1 + sb * 144;
        float dall0 = (float)(*(device const half*)blk0); float dmin0 = (float)(*(device const half*)(blk0 + 2));
        float dall1 = (float)(*(device const half*)blk1); float dmin1 = (float)(*(device const half*)(blk1 + 2));
        device const uchar *scales0 = blk0 + 4; device const uchar *qs0 = blk0 + 16;
        device const uchar *scales1 = blk1 + 4; device const uchar *qs1 = blk1 + 16;
        int is = 0; uint qoff = 0;
        for (uint j = 0; j < 256; j += 64) {
            uchar sc, m;
            get_scale_min_k4(is + 0, scales0, sc, m); float d01 = dall0 * sc, m01 = dmin0 * m;
            get_scale_min_k4(is + 1, scales0, sc, m); float d02 = dall0 * sc, m02 = dmin0 * m;
            get_scale_min_k4(is + 0, scales1, sc, m); float d11 = dall1 * sc, m11 = dmin1 * m;
            get_scale_min_k4(is + 1, scales1, sc, m); float d12 = dall1 * sc, m12 = dmin1 * m;
            for (uint l = 0; l < 32; l += 4) {
                float4 xl = float4(x[ybase+j+l],x[ybase+j+l+1],x[ybase+j+l+2],x[ybase+j+l+3]);
                float4 xh = float4(x[ybase+j+32+l],x[ybase+j+32+l+1],x[ybase+j+32+l+2],x[ybase+j+32+l+3]);
                float4 w0l = float4((float)(qs0[qoff+l]&0xF),(float)(qs0[qoff+l+1]&0xF),(float)(qs0[qoff+l+2]&0xF),(float)(qs0[qoff+l+3]&0xF));
                float4 w0h = float4((float)(qs0[qoff+l]>>4),(float)(qs0[qoff+l+1]>>4),(float)(qs0[qoff+l+2]>>4),(float)(qs0[qoff+l+3]>>4));
                float4 w1l = float4((float)(qs1[qoff+l]&0xF),(float)(qs1[qoff+l+1]&0xF),(float)(qs1[qoff+l+2]&0xF),(float)(qs1[qoff+l+3]&0xF));
                float4 w1h = float4((float)(qs1[qoff+l]>>4),(float)(qs1[qoff+l+1]>>4),(float)(qs1[qoff+l+2]>>4),(float)(qs1[qoff+l+3]>>4));
                s0l += dot(w0l * d01 - m01, xl); s0h += dot(w0h * d02 - m02, xh);
                s1l += dot(w1l * d11 - m11, xl); s1h += dot(w1h * d12 - m12, xh);
            }
            qoff += 32; is += 2;
        }
    }
    float t0 = s0l + s0h, t1 = s1l + s1h;
    t0 += simd_shuffle_xor(t0, 4); t0 += simd_shuffle_xor(t0, 2); t0 += simd_shuffle_xor(t0, 1);
    t1 += simd_shuffle_xor(t1, 4); t1 += simd_shuffle_xor(t1, 2); t1 += simd_shuffle_xor(t1, 1);
    if (lane == 0) {
        if (row0 < dim.d) out[row0] = t0;
        if (row0 + 1 < dim.d) out[row0 + 1] = t1;
    }
}

kernel void gemv_q6_k_2r(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint grp = tid / 8, lane = tid % 8;
    uint row0 = tg * 64 + grp * 2;
    if (row0 >= dim.d) return;
    uint nsb = dim.n / 256;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 210;
    device const uchar *wr1 = (row0 + 1 < dim.d) ? wr0 + nsb * 210 : wr0;
    float s0 = 0.0, s1 = 0.0;
    for (uint sb = lane; sb < nsb; sb += 8) {
        uint ybase = sb * 256;
        float2 a0 = dot_q6_superblock2(wr0 + sb * 210, x, ybase);
        float2 a1 = dot_q6_superblock2(wr1 + sb * 210, x, ybase);
        s0 += a0.x + a0.y;
        s1 += a1.x + a1.y;
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) {
        if (row0 < dim.d) out[row0] = s0;
        if (row0 + 1 < dim.d) out[row0 + 1] = s1;
    }
}

// ============ GEMV v3 (llama.cpp kernel_mul_mv_ext_q4x4 structure) ============
// Row-split GEMV: each row is computed by NXPSG threads of a simdgroup, each
// thread dequantizes 16-element chunks into float4x4 registers and dots them
// against float4 activations; R1PTG rows per thread; reduced with shuffle_down.
constant uint GEMV3_NXPSG = 8;    // threads cooperating on one row
constant uint GEMV3_R1PTG = 2;    // rows per thread
constant uint GEMV3_ROWS  = 64;   // rows per TG = 8 simdgroups * (32/8) * 2

inline void deq_q4_k_chunk16(device const uchar *blk, uint il, thread float4 reg[4]) {
    device const uchar *qs = blk + 16;
    device const uchar *scales = blk + 4;
    float dall = (float)(*(device const half*)blk);
    float dmin = (float)(*(device const half*)(blk + 2));
    uchar sc, m; get_scale_min_k4((int)(il >> 1), scales, sc, m);
    float d = dall * sc, mn = dmin * m;
    uint qo = (il >> 2) * 32 + (il & 1) * 16;
    if ((il & 3) < 2) {
        for (uint j = 0; j < 16; j += 4) {
            reg[j/4] = float4((float)(qs[qo+j]&0xF)*d - mn, (float)(qs[qo+j+1]&0xF)*d - mn,
                              (float)(qs[qo+j+2]&0xF)*d - mn, (float)(qs[qo+j+3]&0xF)*d - mn);
        }
    } else {
        for (uint j = 0; j < 16; j += 4) {
            reg[j/4] = float4((float)(qs[qo+j]>>4)*d - mn, (float)(qs[qo+j+1]>>4)*d - mn,
                              (float)(qs[qo+j+2]>>4)*d - mn, (float)(qs[qo+j+3]>>4)*d - mn);
        }
    }
}

inline void deq_q6_k_chunk16(device const uchar *blk, uint il, thread float4 reg[4]) {
    // 16-element chunk c of half hf (il = 8*hf + c). Element r = 16c+j of the half:
    //   ql byte = (r < 64) ? r : r-64, nibble = low (r<64) / high (r>=64)
    //   qh byte = 32*hf + 16*(c&1) + j, 2-bit field shift = (c & 6)
    //   value = ((nibble | (qh_bits << 4)) - 32) * dall * scales[8*hf + c]
    device const uchar *ql = blk;
    device const uchar *qh = blk + 128;
    device const char *scales = (device const char *)(blk + 192);
    float dall = (float)(*(device const half*)(blk + 208));
    uint hf = il >> 3, c = il & 7;
    float dsc = dall * (float)scales[8*hf + c];
    uint qhshift = (c & 6);
    int qloff = 64 * (int)hf + 16 * (int)c - ((c >= 4) ? 64 : 0);
    uint qhoff = 32 * hf + 16 * (c & 1);
    bool hi = c >= 4;
    for (uint j = 0; j < 16; j += 4) {
        float4 hb = float4((float)((qh[qhoff+j] >> qhshift) & 3), (float)((qh[qhoff+j+1] >> qhshift) & 3),
                           (float)((qh[qhoff+j+2] >> qhshift) & 3), (float)((qh[qhoff+j+3] >> qhshift) & 3));
        float4 lb;
        if (!hi) {
            lb = float4((float)(ql[qloff+j]&0xF), (float)(ql[qloff+j+1]&0xF), (float)(ql[qloff+j+2]&0xF), (float)(ql[qloff+j+3]&0xF));
        } else {
            lb = float4((float)(ql[qloff+j]>>4), (float)(ql[qloff+j+1]>>4), (float)(ql[qloff+j+2]>>4), (float)(ql[qloff+j+3]>>4));
        }
        reg[j/4] = (lb + hb * 16.0f - 32.0f) * dsc;
    }
}

kernel void gemv_q4_k_v3(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint tx = lane % GEMV3_NXPSG, ty = lane / GEMV3_NXPSG;
    uint row0 = tg * GEMV3_ROWS + sg * ((32 / GEMV3_NXPSG) * GEMV3_R1PTG) + ty * GEMV3_R1PTG;
    if (row0 >= dim.d) return;
    uint nb = dim.n / 256;
    device const uchar *wr0 = wb + (ulong)row0 * nb * 144;
    device const uchar *wr1 = (row0 + 1 < dim.d) ? wr0 + nb * 144 : wr0;
    device const float4 *y4 = (device const float4 *)x + tx * 4;
    float sum0 = 0.0, sum1 = 0.0;
    for (uint ich = tx; 16*ich < dim.n; ich += GEMV3_NXPSG) {
        uint b = ich >> 4, il = ich & 15;
        float4 r0[4], r1[4];
        deq_q4_k_chunk16(wr0 + b * 144, il, r0);
        deq_q4_k_chunk16(wr1 + b * 144, il, r1);
        float4 y0 = y4[0], y1 = y4[1], y2 = y4[2], y3 = y4[3];
        sum0 += dot(r0[0],y0) + dot(r0[1],y1) + dot(r0[2],y2) + dot(r0[3],y3);
        sum1 += dot(r1[0],y0) + dot(r1[1],y1) + dot(r1[2],y2) + dot(r1[3],y3);
        y4 += 32;   // NXPSG * 4 float4 per step
    }
    sum0 += simd_shuffle_down(sum0, 4);
    sum0 += simd_shuffle_down(sum0, 2);
    sum0 += simd_shuffle_down(sum0, 1);
    sum1 += simd_shuffle_down(sum1, 4);
    sum1 += simd_shuffle_down(sum1, 2);
    sum1 += simd_shuffle_down(sum1, 1);
    if (tx == 0) {
        if (row0 < dim.d) out[row0] = sum0;
        if (row0 + 1 < dim.d) out[row0 + 1] = sum1;
    }
}

kernel void gemv_q6_k_v3(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint tx = lane % GEMV3_NXPSG, ty = lane / GEMV3_NXPSG;
    uint row0 = tg * GEMV3_ROWS + sg * ((32 / GEMV3_NXPSG) * GEMV3_R1PTG) + ty * GEMV3_R1PTG;
    if (row0 >= dim.d) return;
    uint nb = dim.n / 256;
    device const uchar *wr0 = wb + (ulong)row0 * nb * 210;
    device const uchar *wr1 = (row0 + 1 < dim.d) ? wr0 + nb * 210 : wr0;
    device const float4 *y4 = (device const float4 *)x + tx * 4;
    float sum0 = 0.0, sum1 = 0.0;
    for (uint ich = tx; 16*ich < dim.n; ich += GEMV3_NXPSG) {
        uint b = ich >> 4, il = ich & 15;
        float4 r0[4], r1[4];
        deq_q6_k_chunk16(wr0 + b * 210, il, r0);
        deq_q6_k_chunk16(wr1 + b * 210, il, r1);
        float4 y0 = y4[0], y1 = y4[1], y2 = y4[2], y3 = y4[3];
        sum0 += dot(r0[0],y0) + dot(r0[1],y1) + dot(r0[2],y2) + dot(r0[3],y3);
        sum1 += dot(r1[0],y0) + dot(r1[1],y1) + dot(r1[2],y2) + dot(r1[3],y3);
        y4 += 32;
    }
    sum0 += simd_shuffle_down(sum0, 4);
    sum0 += simd_shuffle_down(sum0, 2);
    sum0 += simd_shuffle_down(sum0, 1);
    sum1 += simd_shuffle_down(sum1, 4);
    sum1 += simd_shuffle_down(sum1, 2);
    sum1 += simd_shuffle_down(sum1, 1);
    if (tx == 0) {
        if (row0 < dim.d) out[row0] = sum0;
        if (row0 + 1 < dim.d) out[row0 + 1] = sum1;
    }
}

kernel void gemv_q4_k_o(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint NXPSG = 32; const uint R1 = 4; const uint ROWS_TG = 32;
    uint tx = lane % NXPSG, ty = lane / NXPSG;
    uint row0 = tg * ROWS_TG + sg * (1 * R1) + ty * R1;
    if (row0 >= dim.d) return;
    uint nb = dim.n / 256;
    device const uchar *wr0 = wb + (ulong)row0 * nb * 144;
    device const uchar *wr1 = (row0 + 1 < dim.d) ? wr0 + (ulong)1 * nb * 144 : wr0;
    device const uchar *wr2 = (row0 + 2 < dim.d) ? wr0 + (ulong)2 * nb * 144 : wr0;
    device const uchar *wr3 = (row0 + 3 < dim.d) ? wr0 + (ulong)3 * nb * 144 : wr0;
    device const float4 *y4 = (device const float4 *)x + tx * 4;
    float sum0 = 0.0;
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    for (uint ich = tx; 16*ich < dim.n; ich += NXPSG) {
        uint b = ich >> 4, il = ich & 15;
        float4 r0[4], r1[4], r2[4], r3[4];
        deq_q4_k_chunk16(wr0 + b * 144, il, r0);
        deq_q4_k_chunk16(wr1 + b * 144, il, r1);
        deq_q4_k_chunk16(wr2 + b * 144, il, r2);
        deq_q4_k_chunk16(wr3 + b * 144, il, r3);
        float4 y0 = y4[0], y1 = y4[1], y2 = y4[2], y3 = y4[3];
        sum0 += dot(r0[0],y0) + dot(r0[1],y1) + dot(r0[2],y2) + dot(r0[3],y3);
        sum1 += dot(r1[0],y0) + dot(r1[1],y1) + dot(r1[2],y2) + dot(r1[3],y3);
        sum2 += dot(r2[0],y0) + dot(r2[1],y1) + dot(r2[2],y2) + dot(r2[3],y3);
        sum3 += dot(r3[0],y0) + dot(r3[1],y1) + dot(r3[2],y2) + dot(r3[3],y3);
        y4 += NXPSG * 4;
    }
    sum0 += simd_shuffle_xor(sum0, 16);
    sum0 += simd_shuffle_xor(sum0, 8);
    sum0 += simd_shuffle_xor(sum0, 4);
    sum0 += simd_shuffle_xor(sum0, 2);
    sum0 += simd_shuffle_xor(sum0, 1);
    sum1 += simd_shuffle_xor(sum1, 16);
    sum1 += simd_shuffle_xor(sum1, 8);
    sum1 += simd_shuffle_xor(sum1, 4);
    sum1 += simd_shuffle_xor(sum1, 2);
    sum1 += simd_shuffle_xor(sum1, 1);
    sum2 += simd_shuffle_xor(sum2, 16);
    sum2 += simd_shuffle_xor(sum2, 8);
    sum2 += simd_shuffle_xor(sum2, 4);
    sum2 += simd_shuffle_xor(sum2, 2);
    sum2 += simd_shuffle_xor(sum2, 1);
    sum3 += simd_shuffle_xor(sum3, 16);
    sum3 += simd_shuffle_xor(sum3, 8);
    sum3 += simd_shuffle_xor(sum3, 4);
    sum3 += simd_shuffle_xor(sum3, 2);
    sum3 += simd_shuffle_xor(sum3, 1);
    if (tx == 0) {
        if (row0 + 0 < dim.d) out[row0 + 0] = sum0;
        if (row0 + 1 < dim.d) out[row0 + 1] = sum1;
        if (row0 + 2 < dim.d) out[row0 + 2] = sum2;
        if (row0 + 3 < dim.d) out[row0 + 3] = sum3;
    }
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

// bm=64 bk=64 variants (autotune winner, gemm_harness/gemm_autotune.odin):
// 8 simdgroups (256 threads), sa[64x64], sb[64x8], k-step 64. Requires M%64==0.
// NB: qs index uses the within-32-group column l32 = e_base&31 (== col_base only
// when k%32==0; v3's BK=32 made them coincide, BK=64 must not assume it).
kernel void gemm_q4k_f32_b64(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 64, c0 = tgpig.x * 8;
    const ulong row_bytes = (ulong)(K / 256) * 144;
    threadgroup half sa[64 * 64];
    threadgroup half sb[64 * 8];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / 8; row < 64; row += 32) {       // TPR=8, RSTEP=256/8=32
            const uint col_base = (tiitg % 8) * 8;
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
            int l32 = ew & 31;
            for (uint j = 0; j < 8; j++) {
                uchar nib = (qs[qoff + l32 + int(j)] >> shift) & 0xF;
                sa[row * 64 + col_base + j] = (half)(dl * (float)nib - ml);
            }
        }
        for (uint t = tiitg; t < 64 * 8; t += 256) {
            uint kr = t >> 3, nc = t & 7;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 64 + ik * 8, 64);
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

// ============ MoE (qwen35moe) kernels ============
// F32 GEMV: router gate_inp and the shared-expert projections (uploaded F32).
// Same 8-lane production shape as gemv_q4_k.
kernel void moe_gemv_f32(device const float *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) {
        device const float *wr = wb + (ulong)row * dim.n;
        for (uint j = lane; j < dim.n; j += GEMV_SG) sum += wr[j] * x[j];
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

// Q8_0 GEMV: 34-byte blocks (f16 scale d + 32 int8), used by ffn_down_exps.
kernel void moe_gemv_q80(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nb = dim.n / 32; device const uchar *wr = wb + (ulong)row * nb * 34;
        for (uint b = lane; b < nb; b += GEMV_SG) { device const uchar *blk = wr + b * 34;
            float d = (float)(*(device const half*)blk);
            device const char *qs = (device const char*)(blk + 2);
            uint y = b * 32;
            for (uint j = 0; j < 32; j += 8) {
                sum += d * ((float)qs[j]   * x[y+j]   + (float)qs[j+1] * x[y+j+1] +
                            (float)qs[j+2] * x[y+j+2] + (float)qs[j+3] * x[y+j+3] +
                            (float)qs[j+4] * x[y+j+4] + (float)qs[j+5] * x[y+j+5] +
                            (float)qs[j+6] * x[y+j+6] + (float)qs[j+7] * x[y+j+7]);
            }
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

// Q5_0 GEMV: 22-byte blocks (f16 d, 4B high bits, 16B nibbles), 32 elems.
kernel void moe_gemv_q50(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    if (row < dim.d) { uint nb = dim.n / 32; device const uchar *wr = wb + (ulong)row * nb * 22;
        for (uint b = lane; b < nb; b += GEMV_SG) { device const uchar *blk = wr + b * 22;
            float d = (float)(*(device const half*)blk);
            uint qh = (uint)blk[2] | ((uint)blk[3] << 8) | ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
            device const uchar *qs = blk + 6;
            uint y = b * 32;
            for (uint j = 0; j < 16; j += 4) {
                // high bits: element j (j<16) uses qh bit j; element j+16 uses
                // qh bit j+16 (the CPU dequant masks 0x10 after shifting by j+12)
                float x0 = (float)(((int)(qs[j] & 0xF) | (int)((qh >> j) & 1u) * 16) - 16);
                float x1 = (float)(((int)(qs[j] >> 4) | (int)((qh >> (j + 16)) & 1u) * 16) - 16);
                float x2 = (float)(((int)(qs[j+1] & 0xF) | (int)((qh >> (j + 1)) & 1u) * 16) - 16);
                float x3 = (float)(((int)(qs[j+1] >> 4) | (int)((qh >> (j + 17)) & 1u) * 16) - 16);
                float x4 = (float)(((int)(qs[j+2] & 0xF) | (int)((qh >> (j + 2)) & 1u) * 16) - 16);
                float x5 = (float)(((int)(qs[j+2] >> 4) | (int)((qh >> (j + 18)) & 1u) * 16) - 16);
                float x6 = (float)(((int)(qs[j+3] & 0xF) | (int)((qh >> (j + 3)) & 1u) * 16) - 16);
                float x7 = (float)(((int)(qs[j+3] >> 4) | (int)((qh >> (j + 19)) & 1u) * 16) - 16);
                sum += d * (x0 * x[y+j] + x1 * x[y+j+16] + x2 * x[y+j+1] + x3 * x[y+j+17] +
                            x4 * x[y+j+2] + x5 * x[y+j+18] + x6 * x[y+j+3] + x7 * x[y+j+19]);
            }
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

// MoE router tail: softmax over n_expert logits, top-k selection, weight
// renormalization (divisor clamped to the F16 min, llama.cpp norm_w). Single
// thread — n_expert <= 256, k <= 8 makes this trivial.
kernel void moe_softmax_topk(device const float *logits [[buffer(0)]],
    device int *sel [[buffer(1)]], device float *weights [[buffer(2)]],
    constant uint2 &P [[buffer(3)]]) {
    uint n = P.x, k = P.y;
    threadgroup float probs[256];
    threadgroup bool picked[256];
    float m = -1e30;
    for (uint j = 0; j < n; j++) m = max(m, logits[j]);
    float sum = 0.0;
    for (uint j = 0; j < n; j++) { probs[j] = exp(logits[j] - m); sum += probs[j]; }
    for (uint j = 0; j < n; j++) { probs[j] /= sum; picked[j] = false; }
    for (uint i = 0; i < k; i++) {
        int best = -1; float bv = -1e30;
        for (uint j = 0; j < n; j++) { if (!picked[j] && probs[j] > bv) { bv = probs[j]; best = (int)j; } }
        sel[i] = best; weights[i] = probs[best]; picked[best] = true;
    }
    float wsum = 0.0;
    for (uint i = 0; i < k; i++) wsum += weights[i];
    if (wsum < 6.103515625e-5) wsum = 6.103515625e-5;
    for (uint i = 0; i < k; i++) weights[i] /= wsum;
}

// Weighted sum of the selected experts' down outputs plus the gated shared
// expert: out[i] = sum_k weights[k] * down[k*dim + i] + has_shexp * gate * shexp[i].
// The shared-expert gate is sigmoided here (llama.cpp applies sigmoid to the
// shared expert gate before multiplying the output).
kernel void moe_combine(device const float *down [[buffer(0)]],
    device float *out [[buffer(1)]], device const float *weights [[buffer(2)]],
    device const float *shexp [[buffer(3)]],
    constant uint3 &P [[buffer(4)]], constant float &shexp_gate [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    float acc = 0.0;
    for (uint k = 0; k < P.x; k++) acc += weights[k] * down[k * P.z + gid];
    if (P.y) { float g = 1.0 / (1.0 + exp(-shexp_gate)); acc += g * shexp[gid]; }
    out[gid] = acc;
}

// Batched F32 GEMV for the MoE router: x is [T x dim] token-major, w is
// [d x dim] row-major, out is [T x d] token-major. One thread per (row, token).
kernel void gemv_f32_b(device const float *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint3 &dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint r = gid / dim.z; uint t = gid % dim.z;   // dim.z = T
    if (r >= dim.d) return;
    device const float *wr = wb + (ulong)r * dim.n;
    device const float *xt = x + (ulong)t * dim.n;
    float sum = 0.0;
    for (uint j = 0; j < dim.n; j++) sum += wr[j] * xt[j];
    out[(ulong)t * dim.d + r] = sum;
}

// Batched MoE router tail: one thread per token column. Same math as
// moe_softmax_topk (softmax -> top-k -> renormalize with F16-min clamp).
kernel void moe_softmax_topk_b(device const float *logits [[buffer(0)]],
    device int *sel [[buffer(1)]], device float *weights [[buffer(2)]],
    constant uint3 &P [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    uint n = P.x, k = P.y, T = P.z;
    if (t >= T) return;
    threadgroup float probs[256];
    threadgroup bool picked[256];
    device const float *lg = logits + (ulong)t * n;
    float m = -1e30;
    for (uint j = 0; j < n; j++) m = max(m, lg[j]);
    float sum = 0.0;
    for (uint j = 0; j < n; j++) { probs[j] = exp(lg[j] - m); sum += probs[j]; }
    for (uint j = 0; j < n; j++) { probs[j] /= sum; picked[j] = false; }
    for (uint i = 0; i < k; i++) {
        int best = -1; float bv = -1e30;
        for (uint j = 0; j < n; j++) { if (!picked[j] && probs[j] > bv) { bv = probs[j]; best = (int)j; } }
        sel[(ulong)t * k + i] = best; weights[(ulong)t * k + i] = probs[best]; picked[best] = true;
    }
    float wsum = 0.0;
    for (uint i = 0; i < k; i++) wsum += weights[(ulong)t * k + i];
    if (wsum < 6.103515625e-5) wsum = 6.103515625e-5;
    for (uint i = 0; i < k; i++) weights[(ulong)t * k + i] /= wsum;
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
g_gemv3: bool       // use row-split v3 GEMV kernels (env QGEMV3=1)
g_gemv2r: bool      // use 2-rows-per-lane GEMV kernels (env QGEMV2R=1)
g_normfuse: bool    // fuse rmsnorm into GEMV dispatches (env QGEMVNORM=1)

// ---- first-run GEMV kernel autotuning (ODIN_TUNE=0 disables) ----
Tune_Cand :: enum { Prod, R2, V3, O }
Tune_Entry :: struct { kind: ggml.GGML_Type, d, n: int, cand: Tune_Cand }
g_tune: [dynamic]Tune_Entry
g_tune_enabled: bool
@(private = "file")
m_weights: ^MTL.Buffer
@(private = "file")
m_mmap_base: uintptr
@(private = "file")
metal_enabled: bool

@(private = "file")
m_pso_f32, m_pso_f16, m_pso_q8_0, m_pso_q4_0, m_pso_q4_1, m_pso_q5_0, m_pso_q5_1, m_pso_q4_k, m_pso_q6_k: ^MTL.ComputePipelineState
	m_pso_q4_k3, m_pso_q6_k3: ^MTL.ComputePipelineState // row-split v3 GEMV kernels (QGEMV3=1)
	m_pso_q4_k2r, m_pso_q6_k2r: ^MTL.ComputePipelineState // 2 rows/lane GEMV kernels (QGEMV2R=1)
	m_pso_q4_knorm, m_pso_q6_knorm: ^MTL.ComputePipelineState // norm-fused GEMV (QGEMVNORM=1)
	m_pso_q4_k_o: ^MTL.ComputePipelineState           // output-layer GEMV (32 lanes/row x 4 rows/thread, llama.cpp default shape)
@(private = "file")
m_pso_rmsnorm, m_pso_rope, m_pso_qnorm, m_pso_store_kv, m_pso_attn, m_pso_attn_gate, m_pso_swiglu, m_pso_residual: ^MTL.ComputePipelineState
	m_pso_gemv_f32, m_pso_gemv_q80, m_pso_gemv_q50, m_pso_moe_topk, m_pso_moe_combine: ^MTL.ComputePipelineState
	m_pso_gemv_f32_b, m_pso_moe_topk_b: ^MTL.ComputePipelineState
@(private = "file")
m_pso_conv, m_pso_l2norm, m_pso_delta, m_pso_rmsgated: ^MTL.ComputePipelineState
@(private = "file")
m_pso_q4k, m_pso_q6k, m_pso_cast, m_pso_copy, m_pso_tr_h16, m_pso_tr_f32: ^MTL.ComputePipelineState // Stage 1a
@(private = "file")
m_pso_q4k_b64: ^MTL.ComputePipelineState // Stage 1a autotuned bm=64 bk=64 (M%64==0; Q4K only, Q6K regresses)
@(private = "file")
g_b64_seen: bool // one-shot dispatch confirmation (debug)
@(private = "file")
m_pso_conv_batch, m_pso_conv_update: ^MTL.ComputePipelineState // Stage 2 batched conv1d
@(private = "file")
m_pso_l2norm_batch, m_pso_rmsgated_batch, m_pso_chunked_delta: ^MTL.ComputePipelineState // Stage 2 chunked delta

// activation buffers
@(private = "file")
m_b_x, m_b_xb, m_b_xb2, m_b_qproj, m_b_q, m_b_xb3: ^MTL.Buffer
@(private = "file")
m_b_hb, m_b_hb2, m_b_ktmp, m_b_vtmp, m_b_logits: ^MTL.Buffer
m_b_verify_logits: ^MTL.Buffer // [T × vocab] for batch verification (speculative decoding)
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
	// MoE (qwen35moe) buffers
	m_b_router:      ^MTL.Buffer // [n_layers * n_expert * dim] f32 router weights
	m_b_shexp:       ^MTL.Buffer // per layer: up|gate|down|gate_inp f32 (shared expert)
	m_b_moe_logits:  ^MTL.Buffer // [n_expert] f32 router logits
	m_b_moe_sel:     ^MTL.Buffer // [n_expert_used] i32 selected experts
	m_b_moe_w:       ^MTL.Buffer // [n_expert_used] f32 normalized weights
	m_b_moe_shexp_out: ^MTL.Buffer // [dim] f32 shared expert output
	m_b_moe_shexp_gate: ^MTL.Buffer // [1] f32 shared expert gate
	m_moe_hbuf:      [dynamic]^MTL.Buffer // per-expert silu(gate)*up results [n_ff_exp]
	m_moe_gbuf:      [dynamic]^MTL.Buffer // per-expert gate GEMV outputs [n_ff_exp]
	m_b_moe_down_all: ^MTL.Buffer // [used * dim] f32 expert down outputs (combine stride)
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
			fmt.eprintf("metal: quant (linear)  w1=%s w3=%s w2=%s  qkv=%s z=%s b=%s a=%s out=%s\n",
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
			fmt.eprintf("metal: quant (full)    w1=%s w3=%s w2=%s  wq=%s wk=%s wv=%s wo=%s\n",
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
	// Use mem.zero_slice (SIMD) instead of byte-by-byte loop — at max_ctx=20480
	// the KV cache alone is 1.3 GB; the loop version was taking 4-5 seconds.
	for b in ([]^MTL.Buffer{m_b_conv_states, m_b_rec_states, m_b_kc, m_b_vc}) {
		s := b->contentsAsSlice([]u8)
		mem.zero_slice(s)
	}
}

// KV persist: Metal stores K/V as f16 head-major [n_full, n_kv_heads, seq_len,
// head_dim]. The on-disk canonical format is f32 layer-major [n_full, seq_len,
// kv_dim] (= [n_full, seq_len, n_kv_heads, head_dim] flattened). We do the
// conversion in both directions here. conv_states and recurrent_states are
// already f32 standard layout on both backends — straight copy.

metal_save_kv :: proc(t: ^Transformer, path: string, n_valid_pos: int) -> bool {
	if !metal_enabled { fmt.eprintln("kv: metal not enabled"); return false }
	c := &t.config
	if n_valid_pos > c.seq_len {
		fmt.eprintf("kv: n_valid_pos %d > seq_len %d\n", n_valid_pos, c.seq_len)
		return false
	}
	h := build_header(t, n_valid_pos)
	f, err := os.open(path, os.O_CREATE | os.O_WRONLY | os.O_TRUNC)
	if err != os.ERROR_NONE {
		fmt.eprintf("kv: cannot open %s for write (%v)\n", path, err)
		return false
	}
	defer os.close(f)

	write_header(f, h)

	kv_dim := c.n_kv_heads * c.head_dim
	kc_bytes := m_b_kc->contentsAsSlice([]u8)
	vc_bytes := m_b_vc->contentsAsSlice([]u8)

	// Per-layer temp buffer of [n_valid_pos * kv_dim] f32
	tmp_size := n_valid_pos * kv_dim
	tmp: []f32 = make([]f32, tmp_size)
	defer delete(tmp)

	for l in 0 ..< c.n_full {
		// Metal per-layer offset: l * (seq_len * kv_dim * 2) bytes
		layer_byte_off := u64(l) * u64(c.seq_len) * u64(kv_dim) * 2
		// Convert head-major f16 → layer-major f32
		for h_idx in 0 ..< c.n_kv_heads {
			for p in 0 ..< n_valid_pos {
				for d in 0 ..< c.head_dim {
					m_off := int(layer_byte_off) +
						(h_idx * c.seq_len * c.head_dim +
						 p * c.head_dim +
						 d) * 2
					f16_raw := u16(kc_bytes[m_off]) | (u16(kc_bytes[m_off + 1]) << 8)
					cpu_off := p * kv_dim + h_idx * c.head_dim + d
					tmp[cpu_off] = f32(transmute(f16)(f16_raw))
				}
			}
		}
		if !write_f32_slice(f, tmp) {
			fmt.eprintf("kv: K write failed at layer %d\n", l); return false
		}
		// V: same layout
		for h_idx in 0 ..< c.n_kv_heads {
			for p in 0 ..< n_valid_pos {
				for d in 0 ..< c.head_dim {
					m_off := int(layer_byte_off) +
						(h_idx * c.seq_len * c.head_dim +
						 p * c.head_dim +
						 d) * 2
					f16_raw := u16(vc_bytes[m_off]) | (u16(vc_bytes[m_off + 1]) << 8)
					cpu_off := p * kv_dim + h_idx * c.head_dim + d
					tmp[cpu_off] = f32(transmute(f16)(f16_raw))
				}
			}
		}
		if !write_f32_slice(f, tmp) {
			fmt.eprintf("kv: V write failed at layer %d\n", l); return false
		}
	}

	// conv_states, recurrent_states: f32 standard layout, straight write
	conv_slice := (cast([^]f32)raw_data(m_b_conv_states->contentsAsSlice([]u8)))[:h.n_linear_layers * h.conv_dim * (h.lin_conv_kernel - 1)]
	if !write_f32_slice(f, conv_slice) {
		fmt.eprintln("kv: conv_states write failed"); return false
	}
	rec_n := h.n_linear_layers * h.lin_n_v_heads * h.lin_head_k_dim * h.lin_head_v_dim
	rec_slice := (cast([^]f32)raw_data(m_b_rec_states->contentsAsSlice([]u8)))[:rec_n]
	if !write_f32_slice(f, rec_slice) {
		fmt.eprintln("kv: recurrent_states write failed"); return false
	}

	if !append_crc32(path) {
		fmt.eprintln("kv: crc append failed"); return false
	}
	fmt.eprintfln("kv: saved (Metal) %s (%d valid pos, %d full L, kv_dim=%d)",
		path, n_valid_pos, c.n_full, kv_dim)
	return true
}

metal_load_kv :: proc(t: ^Transformer, path: string) -> (int, bool) {
	if !metal_enabled { fmt.eprintln("kv: metal not enabled"); return 0, false }
	f, err := os.open(path, os.O_RDONLY)
	if err != os.ERROR_NONE {
		fmt.eprintf("kv: cannot open %s (%v)\n", path, err)
		return 0, false
	}
	defer os.close(f)

	h, ok := read_header(f)
	if !ok { fmt.eprintln("kv: header truncated"); return 0, false }
	if h.magic != KV_MAGIC {
		fmt.eprintf("kv: bad magic 0x%08x\n", h.magic); return 0, false
	}
	if h.version != KV_VERSION {
		fmt.eprintf("kv: version mismatch (file=%d, supported=%d)\n", h.version, KV_VERSION); return 0, false
	}
	if h.arch != KV_ARCH_QWEN3_5 {
		fmt.eprintf("kv: arch mismatch (file=%d, expected qwen3_5)\n", h.arch); return 0, false
	}
	expected_fp := model_fingerprint(t)
	if h.fingerprint != expected_fp {
		fmt.eprintf("kv: fingerprint mismatch (file=0x%016x, model=0x%016x)\n",
			h.fingerprint, expected_fp)
		return 0, false
	}
	c := &t.config
	if int(h.seq_len) != c.seq_len {
		fmt.eprintf("kv: seq_len mismatch (file=%d, engine=%d) — use same -c\n",
			h.seq_len, c.seq_len)
		return 0, false
	}
	if !verify_crc32(path) {
		fmt.eprintln("kv: CRC mismatch — file corrupted"); return 0, false
	}

	// Zero the K/V GPU buffers first (everything beyond n_valid_pos is stale).
	kc_bytes := m_b_kc->contentsAsSlice([]u8)
	vc_bytes := m_b_vc->contentsAsSlice([]u8)
	for i in 0 ..< len(kc_bytes) { kc_bytes[i] = 0 }
	for i in 0 ..< len(vc_bytes) { vc_bytes[i] = 0 }

	kv_dim := c.n_kv_heads * c.head_dim
	tmp_size := int(h.n_valid_pos) * kv_dim
	tmp: []f32 = make([]f32, tmp_size)
	defer delete(tmp)

	for l in 0 ..< c.n_full {
		layer_byte_off := u64(l) * u64(c.seq_len) * u64(kv_dim) * 2
		// K
		if !read_f32_into(f, tmp) {
			fmt.eprintf("kv: K read failed at layer %d\n", l); return 0, false
		}
		for h_idx in 0 ..< c.n_kv_heads {
			for p in 0 ..< int(h.n_valid_pos) {
				for d in 0 ..< c.head_dim {
					m_off := int(layer_byte_off) +
						(h_idx * c.seq_len * c.head_dim +
						 p * c.head_dim +
						 d) * 2
					cpu_off := p * kv_dim + h_idx * c.head_dim + d
					f16_raw := transmute(u16)(f16(tmp[cpu_off]))
					kc_bytes[m_off]     = u8(f16_raw & 0xff)
					kc_bytes[m_off + 1] = u8((f16_raw >> 8) & 0xff)
				}
			}
		}
		// V
		if !read_f32_into(f, tmp) {
			fmt.eprintf("kv: V read failed at layer %d\n", l); return 0, false
		}
		for h_idx in 0 ..< c.n_kv_heads {
			for p in 0 ..< int(h.n_valid_pos) {
				for d in 0 ..< c.head_dim {
					m_off := int(layer_byte_off) +
						(h_idx * c.seq_len * c.head_dim +
						 p * c.head_dim +
						 d) * 2
					cpu_off := p * kv_dim + h_idx * c.head_dim + d
					f16_raw := transmute(u16)(f16(tmp[cpu_off]))
					vc_bytes[m_off]     = u8(f16_raw & 0xff)
					vc_bytes[m_off + 1] = u8((f16_raw >> 8) & 0xff)
				}
			}
		}
	}

	// conv_states, recurrent_states: f32 standard layout, straight read into GPU buffer
	conv_dst := (cast([^]f32)raw_data(m_b_conv_states->contentsAsSlice([]u8)))[:h.n_linear_layers * h.conv_dim * (h.lin_conv_kernel - 1)]
	if !read_f32_into(f, conv_dst) {
		fmt.eprintln("kv: conv_states read failed"); return 0, false
	}
	rec_n := h.n_linear_layers * h.lin_n_v_heads * h.lin_head_k_dim * h.lin_head_v_dim
	rec_dst := (cast([^]f32)raw_data(m_b_rec_states->contentsAsSlice([]u8)))[:rec_n]
	if !read_f32_into(f, rec_dst) {
		fmt.eprintln("kv: recurrent_states read failed"); return 0, false
	}

	fmt.eprintfln("kv: loaded (Metal) %s (%d valid pos)", path, h.n_valid_pos)
	return int(h.n_valid_pos), true
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

metal_init :: proc(t: ^Transformer, model_path: string) -> bool {
	g := &t.gguf
	c := t.config

	m_device = MTL.CreateSystemDefaultDevice()
	if m_device == nil { fmt.eprintln("metal: no Metal device available"); return false }
	fmt.eprintf("metal: %s\n", m_device->name()->odinString())
	m_queue = m_device->newCommandQueue()

	// profiling/debug toggles (env). QFASTMATH=0 -> precise shaders (for f32 drift
	// debugging, e.g. the Stage 2 chunked-delta C=16 case). QTIMING=1 -> per-prefill
	// GPU ms. QPROF_NOLIN=1 -> ablation: skip linear-layer per-token stateful loop.
	g_fast_math = os.get_env("QFASTMATH", context.temp_allocator) != "0"
	g_timing = os.get_env("QTIMING", context.temp_allocator) == "1"
	g_prof_nolin = os.get_env("QPROF_NOLIN", context.temp_allocator) == "1"
	g_prof_nofull = os.get_env("QPROF_NOFULL", context.temp_allocator) == "1"
	g_gemv3 = os.get_env("QGEMV3", context.temp_allocator) == "1"
	g_gemv2r = os.get_env("QGEMV2R", context.temp_allocator) == "1"
	g_normfuse = os.get_env("QGEMVNORM", context.temp_allocator) == "1"
	g_tune_enabled = os.get_env("ODIN_TUNE", context.temp_allocator) != "0"
	fmt.eprintfln("metal: shader fastMath={} (QFASTMATH=0 to disable)", g_fast_math)
	fmt.eprintfln("metal: maxThreadgroupMemoryLength={} bytes", m_device->maxThreadgroupMemoryLength())

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
	m_pso_q4_k3 = make_pso(lib, "gemv_q4_k_v3"); m_pso_q6_k3 = make_pso(lib, "gemv_q6_k_v3")
	m_pso_q4_k2r = make_pso(lib, "gemv_q4_k_2r"); m_pso_q6_k2r = make_pso(lib, "gemv_q6_k_2r")
	m_pso_q4_knorm = make_pso(lib, "gemv_q4_k_norm"); m_pso_q6_knorm = make_pso(lib, "gemv_q6_k_norm")
	m_pso_q4_k_o = make_pso(lib, "gemv_q4_k_o")
	m_pso_rmsnorm = make_pso(lib, "rmsnorm"); m_pso_rope = make_pso(lib, "rope_partial")
	m_pso_qnorm = make_pso(lib, "qnorm_extract"); m_pso_store_kv = make_pso(lib, "store_kv")
	m_pso_attn = make_pso(lib, "attention"); m_pso_attn_gate = make_pso(lib, "attn_gate")
	m_pso_swiglu = make_pso(lib, "swiglu"); m_pso_residual = make_pso(lib, "residual")
	m_pso_gemv_f32 = make_pso(lib, "moe_gemv_f32"); m_pso_gemv_q80 = make_pso(lib, "moe_gemv_q80"); m_pso_gemv_q50 = make_pso(lib, "moe_gemv_q50")
	m_pso_moe_topk = make_pso(lib, "moe_softmax_topk"); m_pso_moe_combine = make_pso(lib, "moe_combine")
	m_pso_gemv_f32_b = make_pso(lib, "gemv_f32_b"); m_pso_moe_topk_b = make_pso(lib, "moe_softmax_topk_b")
	m_pso_conv = make_pso(lib, "conv1d_step_silu"); m_pso_l2norm = make_pso(lib, "l2norm_scale")
	m_pso_delta = make_pso(lib, "delta_recurrent"); m_pso_rmsgated = make_pso(lib, "rmsnorm_gated")
	m_pso_q4k = make_pso(lib, "gemm_q4k_f32"); m_pso_q6k = make_pso(lib, "gemm_q6k_f32")
	m_pso_q4k_b64 = make_pso(lib, "gemm_q4k_f32_b64")
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

	if c.is_moe {
		moe_metal_init(t)
	} else if g_tune_enabled {
		autotune(t, model_path)
	}

	// Memory budget: report model + KV + linear-state footprint against the
	// machine's physical RAM; warn when the engine would eat >85% of it
	// (the 8 GB MacBook Air case — see docs/MEMORY.md).
	if pi := NS.ProcessInfo_processInfo(); pi != nil {
		phys := u64(NS.ProcessInfo_physicalMemory(pi))
		if fi, ferr := os.stat(model_path, context.allocator); ferr == os.ERROR_NONE {
			kv_bytes := 2 * u64(c.n_full) * u64(c.seq_len) * u64(c.n_kv_heads) * u64(c.head_dim) * 4
			conv_dim := c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
			lin_bytes :=
				u64(c.n_linear) * u64(conv_dim) * u64(c.lin_conv_kernel - 1) * 4 +
				u64(c.n_linear) * u64(c.lin_n_v_heads) * u64(c.lin_head_k_dim) * u64(c.lin_head_v_dim) * 4
			total := u64(fi.size) + kv_bytes + lin_bytes
			frac := f64(total) / f64(phys) * 100.0
			level := "OK"
			if frac > 85.0 { level = "WARNING: tight for this machine" }
			fmt.eprintfln("mem: model=%.2f GB kv=%.2f GB lin=%.2f GB total=%.2f GB / physical=%.2f GB (%.0f%%) %s",
				f64(fi.size) / 1e9, f64(kv_bytes) / 1e9, f64(lin_bytes) / 1e9, f64(total) / 1e9, f64(phys) / 1e9, frac, level)
		}
	}

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
	case .Q4_K:
		if g_gemv3 { return m_pso_q4_k3 }
		if g_gemv2r { return m_pso_q4_k2r }
		return m_pso_q4_k
	case .Q6_K:
		if g_gemv3 { return m_pso_q6_k3 }
		if g_gemv2r { return m_pso_q6_k2r }
		return m_pso_q6_k
	}
	return nil
}

@(private = "file")
bytes_of :: proc(p: rawptr, n: int) -> []u8 { return ([^]u8)(p)[:n] }

@(private = "file")
woff :: proc(p: rawptr) -> NS.UInteger { return NS.UInteger(uintptr(p) - m_mmap_base) }

@(private = "file")
tune_pso :: proc(kind: ggml.GGML_Type, cand: Tune_Cand) -> (pso: ^MTL.ComputePipelineState, rows_tg: int) {
	rows_tg = GEMV_ROWS
	switch cand {
	case .Prod:
		pso = pso_for(kind)
	case .R2:
		pso = kind == .Q4_K ? m_pso_q4_k2r : m_pso_q6_k2r
		rows_tg = 64
	case .V3:
		pso = kind == .Q4_K ? m_pso_q4_k3 : m_pso_q6_k3
		rows_tg = 64
	case .O:
		pso = m_pso_q4_k_o
		rows_tg = 32
	}
	return
}

Tune_Shape :: struct { kind: ggml.GGML_Type, d, n: int, w_off: NS.UInteger }

@(private = "file")
tune_add_shape :: proc(shapes: ^[dynamic]Tune_Shape, kind: ggml.GGML_Type, w: rawptr, d, n: int) {
	if d < 512 { return }
	for s in shapes {
		if s.kind == kind && s.d == d && s.n == n { return }
	}
	append(shapes, Tune_Shape{kind, d, n, woff(w)})
}

@(private = "file")
collect_tune_shapes :: proc(t: ^Transformer) -> [dynamic]Tune_Shape {
	if t.config.is_moe {
		return make([dynamic]Tune_Shape)
	}
	c := t.config
	w := &t.weights
	shapes := make([dynamic]Tune_Shape, 0, 8)
	att_head_dim := c.n_heads * c.head_dim
	conv_dim := c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
	for l in 0 ..< c.n_layers {
		lw := &w.layers[l]
		tune_add_shape(&shapes, lw.w1.kind, raw_data(lw.w1.data), c.hidden_dim, c.dim)
		tune_add_shape(&shapes, lw.w2.kind, raw_data(lw.w2.data), c.dim, c.hidden_dim)
		switch lw.layer_type {
		case .Full_Attention:
			tune_add_shape(&shapes, lw.full.wq.kind, raw_data(lw.full.wq.data), att_head_dim, c.dim)
		case .Linear_Attention:
			tune_add_shape(&shapes, lw.linear.in_qkv.kind, raw_data(lw.linear.in_qkv.data), conv_dim, c.dim)
		}
	}
	tune_add_shape(&shapes, w.output.kind, raw_data(w.output.data), c.vocab_size, c.dim)
	return shapes
}

tune_time_ms :: proc() -> i64 { return time.to_unix_nanoseconds(time.now()) / 1_000_000 }

// Wall-clock timing, 10 dispatches per command buffer, min of 3 buffers.
// (GPU timer counters proved unreliable for single short dispatches.)
@(private = "file")
time_candidate :: proc(kind: ggml.GGML_Type, cand: Tune_Cand, w_off: NS.UInteger, n, d: int, xb, ob: ^MTL.Buffer) -> f64 {
	pso, rows_tg := tune_pso(kind, cand)
	best := math.INF_F64
	dims := [2]u32{u32(n), u32(d)}
	tg_x := (d + rows_tg - 1) / rows_tg
	for iter in 0 ..< 3 {
		t0 := tune_time_ms()
		cmd := m_queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(m_weights, w_off, 0); enc->setBuffer(xb, 0, 1); enc->setBuffer(ob, 0, 2)
		enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
		for rep in 0 ..< 10 {
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
		}
		enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
		ms := f64(tune_time_ms() - t0) / 10.0
		if ms < best { best = ms }
	}
	return best
}

tune_cache_dir :: proc() -> string {
	dir := os.get_env("ODIN_TUNE_DIR", context.allocator)
	if dir == "" {
		home := os.get_env("HOME", context.allocator)
		dir = strings.concatenate({home, "/.odin-infer"}, context.allocator)
	}
	return dir
}

tune_cache_path :: proc(model_path: string) -> string {
	base := model_path
	if idx := strings.last_index(model_path, "/"); idx >= 0 {
		base = model_path[idx + 1:]
	}
	return strings.concatenate({tune_cache_dir(), "/", base, ".tune"}, context.allocator)
}

tune_kind_str :: proc(kind: ggml.GGML_Type) -> string {
	#partial switch kind {
	case .Q4_K: return "Q4_K"
	case .Q6_K: return "Q6_K"
	case:       return "?"
	}
}

tune_cand_str :: proc(cand: Tune_Cand) -> string {
	switch cand {
	case .Prod: return "prod"
	case .R2:   return "r2"
	case .V3:   return "v3"
	case .O:    return "o"
	case:       return "?"
	}
}

tune_load :: proc(path: string) -> bool {
	data, err := os.read_entire_file(path, context.allocator)
	if err != os.ERROR_NONE { return false }
	for line in strings.split_lines(string(data)) {
		fields := strings.fields(line)
		if len(fields) < 4 { continue }
		kind: ggml.GGML_Type
		if fields[0] == "Q4_K" { kind = .Q4_K }
		else if fields[0] == "Q6_K" { kind = .Q6_K } else { continue }
		d, ok1 := strconv.parse_int(fields[1])
		n, ok2 := strconv.parse_int(fields[2])
		if !ok1 || !ok2 { continue }
		cand: Tune_Cand
		switch fields[3] {
		case "prod": cand = .Prod
		case "r2":   cand = .R2
		case "v3":   cand = .V3
		case "o":    cand = .O
		case:        continue
		}
		append(&g_tune, Tune_Entry{kind, d, n, cand})
	}
	return len(g_tune) > 0
}

tune_save :: proc(path: string) {
	_ = os.make_directory(tune_cache_dir())
	f, err := os.open(path, os.O_CREATE | os.O_WRONLY | os.O_TRUNC)
	if err != os.ERROR_NONE {
		fmt.eprintfln("tune: cannot write {} ({})", path, err)
		return
	}
	defer os.close(f)
	st := os.to_stream(f)
	for e in g_tune {
		fmt.wprintfln(st, "{} {} {} {}", tune_kind_str(e.kind), e.d, e.n, tune_cand_str(e.cand))
	}
	fmt.eprintfln("tune: saved {} kernel picks to {}", len(g_tune), path)
}

// First-run autotune: time each compiled GEMV variant on the real weight
// shapes and pick the fastest per (kind, rows, cols). Results are cached in
// ~/.odin-infer/<model>.tune (ODIN_TUNE_DIR overrides) and reused on later
// starts — this is what adapts the engine to M1/M2/M3/M4 GPU generations.
autotune :: proc(t: ^Transformer, model_path: string) {
	path := tune_cache_path(model_path)
	if tune_load(path) {
		fmt.eprintfln("tune: loaded {} kernel picks from {} (set ODIN_TUNE_DIR to retune)", len(g_tune), path)
		return
	}
	shapes := collect_tune_shapes(t)
	max_n, max_d := 0, 0
	for s in shapes {
		max_n = max(max_n, s.n)
		max_d = max(max_d, s.d)
	}
	xb := new_shared(max_n)
	ob := new_shared(max_d)
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< max_n { xs[i] = f32(i % 7) * 0.01 }

	for s in shapes {
		best := Tune_Cand.Prod
		best_ms := math.INF_F64
		cands := []Tune_Cand{.Prod, .R2, .V3}
		if s.kind == .Q4_K { cands = []Tune_Cand{.Prod, .R2, .V3, .O} }
		row: strings.Builder
		strings.builder_init(&row, context.allocator)
		for cand in cands {
			ms := time_candidate(s.kind, cand, s.w_off, s.n, s.d, xb, ob)
			strings.write_string(&row, fmt.tprintf(" {}={:.2f}ms", tune_cand_str(cand), ms))
			if ms < best_ms {
				best_ms = ms
				best = cand
			}
		}
		append(&g_tune, Tune_Entry{s.kind, s.d, s.n, best})
		fmt.eprintfln("tune: {} {}x{} -> {}{}", tune_kind_str(s.kind), s.d, s.n, tune_cand_str(best), strings.to_string(row))
	}
	tune_save(path)
}

// ============ MoE (qwen35moe) Metal path ============
// Router weights and the shared expert are dequantized at load (F32) and
// uploaded once; expert tensors stay in m_weights and are dispatched with
// per-expert offsets. Router + top-k run on the GPU; the selected expert ids
// are read back once per layer (a single sync) to drive the expert dispatches.

moe_metal_init :: proc(t: ^Transformer) {
	c := t.config
	w := &t.weights
	router_stride := c.n_expert * c.dim
	m_b_router = new_shared(c.n_layers * router_stride * 4)
	rs := m_b_router->contentsAsSlice([]f32)
	shexp_up_len := c.n_ff_shexp * c.dim
	shexp_down_len := c.dim * c.n_ff_shexp
	shexp_stride := shexp_up_len * 2 + shexp_down_len + c.dim
	m_b_shexp = new_shared(c.n_layers * shexp_stride * 4)
	ss := m_b_shexp->contentsAsSlice([]f32)
	for l in 0 ..< c.n_layers {
		mw := &w.layers[l].moe
		copy(rs[l * router_stride:], mw.gate_inp)
		if mw.shexp_up != nil {
			copy(ss[l * shexp_stride:], mw.shexp_up)
			copy(ss[l * shexp_stride + shexp_up_len:], mw.shexp_gate)
			copy(ss[l * shexp_stride + 2 * shexp_up_len:], mw.shexp_down)
			copy(ss[l * shexp_stride + 2 * shexp_up_len + shexp_down_len:], mw.shexp_gate_inp)
		}
	}
	m_b_moe_logits = new_shared(max(c.n_expert, 1) * 4)
	m_b_moe_sel = new_shared(max(c.n_expert_used, 1) * 4)
	m_b_moe_w = new_shared(max(c.n_expert_used, 1) * 4)
	m_b_moe_shexp_out = new_shared(max(c.dim, 1) * 4)
	m_b_moe_shexp_gate = new_shared(4)
	m_moe_hbuf = make([dynamic]^MTL.Buffer, c.n_expert_used)
	m_moe_gbuf = make([dynamic]^MTL.Buffer, c.n_expert_used)
	for k in 0 ..< c.n_expert_used {
		m_moe_hbuf[k] = new_shared(max(c.n_ff_exp, 1) * 4)
		m_moe_gbuf[k] = new_shared(max(c.n_ff_exp, 1) * 4)
	}
	m_b_moe_down_all = new_shared(max(c.n_expert_used * c.dim, 1) * 4)
	fmt.eprintfln("moe metal: router=%d KB shexp=%d KB", c.n_layers * router_stride * 4 / 1024, c.n_layers * shexp_stride * 4 / 1024)
}

@(private = "file")
enc_gemv_f32 :: proc(enc: ^MTL.ComputeCommandEncoder, wb: ^MTL.Buffer, w_off: NS.UInteger,
	xb: ^MTL.Buffer, x_off: NS.UInteger, ob: ^MTL.Buffer, o_off: NS.UInteger, n, d: int) {
	enc->setComputePipelineState(m_pso_gemv_f32)
	enc->setBuffer(wb, w_off, 0); enc->setBuffer(xb, x_off, 1); enc->setBuffer(ob, o_off, 2)
	dims := [2]u32{u32(n), u32(d)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	tg_x := (d + GEMV_ROWS - 1) / GEMV_ROWS
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
}

@(private = "file")
enc_gemv_q80 :: proc(enc: ^MTL.ComputeCommandEncoder, w_off: NS.UInteger,
	xb: ^MTL.Buffer, x_off: NS.UInteger, ob: ^MTL.Buffer, o_off: NS.UInteger, n, d: int) {
	enc->setComputePipelineState(m_pso_gemv_q80)
	enc->setBuffer(m_weights, w_off, 0); enc->setBuffer(xb, x_off, 1); enc->setBuffer(ob, o_off, 2)
	dims := [2]u32{u32(n), u32(d)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	tg_x := (d + GEMV_ROWS - 1) / GEMV_ROWS
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
}

@(private = "file")
enc_gemv_q50 :: proc(enc: ^MTL.ComputeCommandEncoder, w_off: NS.UInteger,
	xb: ^MTL.Buffer, x_off: NS.UInteger, ob: ^MTL.Buffer, o_off: NS.UInteger, n, d: int) {
	enc->setComputePipelineState(m_pso_gemv_q50)
	enc->setBuffer(m_weights, w_off, 0); enc->setBuffer(xb, x_off, 1); enc->setBuffer(ob, o_off, 2)
	dims := [2]u32{u32(n), u32(d)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	tg_x := (d + GEMV_ROWS - 1) / GEMV_ROWS
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
}

// Batched router: one GEMV over T tokens; logits are [T x n_expert].
@(private = "file")
enc_gemv_f32_b :: proc(enc: ^MTL.ComputeCommandEncoder, wb: ^MTL.Buffer, w_off: NS.UInteger,
	xb: ^MTL.Buffer, x_off: NS.UInteger, ob: ^MTL.Buffer, o_off: NS.UInteger, n, d, tcount: int) {
	enc->setComputePipelineState(m_pso_gemv_f32_b)
	enc->setBuffer(wb, w_off, 0); enc->setBuffer(xb, x_off, 1); enc->setBuffer(ob, o_off, 2)
	P := [3]u32{u32(n), u32(d), u32(tcount)}
	enc->setBytes(bytes_of(&P, size_of(P)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer((d * tcount + 255) / 256), 1, 1}, MTL.Size{256, 1, 1})
}

@(private = "file")
enc_moe_topk_b :: proc(enc: ^MTL.ComputeCommandEncoder, logits: ^MTL.Buffer, l_off: NS.UInteger,
	sel: ^MTL.Buffer, s_off: NS.UInteger, w: ^MTL.Buffer, w_off: NS.UInteger, n_expert, k, tcount: int) {
	enc->setComputePipelineState(m_pso_moe_topk_b)
	enc->setBuffer(logits, l_off, 0)
	enc->setBuffer(sel, s_off, 1)
	enc->setBuffer(w, w_off, 2)
	P := [3]u32{u32(n_expert), u32(k), u32(tcount)}
	enc->setBytes(bytes_of(&P, size_of(P)), 3)
	enc->dispatchThreadgroups(MTL.Size{1, 1, 1}, MTL.Size{NS.Integer(tcount), 1, 1})
}

@(private = "file")
enc_moe_topk :: proc(enc: ^MTL.ComputeCommandEncoder, n_expert, k: int) {
	enc->setComputePipelineState(m_pso_moe_topk)
	enc->setBuffer(m_b_moe_logits, 0, 0)
	enc->setBuffer(m_b_moe_sel, 0, 1)
	enc->setBuffer(m_b_moe_w, 0, 2)
	P := [2]u32{u32(n_expert), u32(k)}
	enc->setBytes(bytes_of(&P, size_of(P)), 3)
	enc->dispatchThreadgroups(MTL.Size{1, 1, 1}, MTL.Size{1, 1, 1})
}

@(private = "file")
enc_moe_combine :: proc(enc: ^MTL.ComputeCommandEncoder, k, has_shexp, dim: int, out: ^MTL.Buffer) {
	enc->setComputePipelineState(m_pso_moe_combine)
	enc->setBuffer(m_b_moe_down_all, 0, 0)
	enc->setBuffer(out, 0, 1)
	enc->setBuffer(m_b_moe_w, 0, 2)
	enc->setBuffer(m_b_moe_shexp_out, 0, 3)
	P := [3]u32{u32(k), u32(has_shexp), u32(dim)}
	enc->setBytes(bytes_of(&P, size_of(P)), 4)
	enc->setBuffer(m_b_moe_shexp_gate, 0, 5)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer((dim + 255) / 256), 1, 1}, MTL.Size{256, 1, 1})
}

// MoE FFN for one layer. Encodes rmsnorm + router + top-k into the caller's
// command buffer, syncs once to read the selected experts, runs the expert
// GEMVs + shared expert + weighted combine in a second command buffer, then
// hands back a fresh encoder for the remaining layers.
moe_ffn_gpu :: proc(enc: ^^MTL.ComputeCommandEncoder, cmd: ^^MTL.CommandBuffer, t: ^Transformer, l: int, nonorm: bool) {
	p := &t.config
	w := &t.weights
	lw := &w.layers[l]
	mw := &lw.moe
	dim := p.dim
	n_exp := p.n_expert
	used := p.n_expert_used
	n_ff := p.n_ff_exp

	if !nonorm {
		enc_rmsnorm(enc^, m_b_x, m_b_xb, raw_data(lw.ffn_norm), dim, 1, p.rms_eps)
	}
	enc_gemv_f32(enc^, m_b_router, NS.UInteger(l * n_exp * dim * 4), m_b_xb, 0, m_b_moe_logits, 0, dim, n_exp)
	enc_moe_topk(enc^, n_exp, used)
	enc^->endEncoding()
	cmd^->commit()
	cmd^->waitUntilCompleted()

	sel := m_b_moe_sel->contentsAsSlice([]i32)[:used]

	cmd2 := m_queue->commandBuffer()
	enc2 := cmd2->computeCommandEncoder()
	up_rb := ggml.row_byte_size(mw.up_exps.kind, dim)
	down_rb := ggml.row_byte_size(mw.down_exps.kind, n_ff)
	up_base := woff(raw_data(mw.up_exps.data))
	gate_base := woff(raw_data(mw.gate_exps.data))
	down_base := woff(raw_data(mw.down_exps.data))
	for k in 0 ..< used {
		e := int(sel[k])
		// NOTE: expert offsets are relative to each tensor; m_weights is the
		// whole-model mmap, so add the tensor's own base offset.
		enc_gemv(enc2, mw.up_exps.kind, up_base + NS.UInteger(e * n_ff * up_rb), m_b_xb, 0, m_moe_hbuf[k], 0, dim, n_ff)
		enc_gemv(enc2, mw.gate_exps.kind, gate_base + NS.UInteger(e * n_ff * up_rb), m_b_xb, 0, m_moe_gbuf[k], 0, dim, n_ff)
		// swiglu kernel: buf0 = silu(buf0) * buf1 -> pass (gate, up) so the
		// result is silu(gate) * up (llama convention; same trap as CPU MoE)
		enc_elementwise(enc2, m_pso_swiglu, m_moe_gbuf[k], m_moe_hbuf[k], n_ff)
		#partial switch mw.down_exps.kind {
		case .Q8_0:
			enc_gemv_q80(enc2, down_base + NS.UInteger(e * dim * down_rb), m_moe_gbuf[k], 0, m_b_moe_down_all, NS.UInteger(k * dim * 4), n_ff, dim)
		case .Q5_0:
			enc_gemv_q50(enc2, down_base + NS.UInteger(e * dim * down_rb), m_moe_gbuf[k], 0, m_b_moe_down_all, NS.UInteger(k * dim * 4), n_ff, dim)
		case:
			enc_gemv(enc2, mw.down_exps.kind, down_base + NS.UInteger(e * dim * down_rb), m_moe_gbuf[k], 0, m_b_moe_down_all, NS.UInteger(k * dim * 4), n_ff, dim)
		}
	}
	has_shexp := 0
	if mw.shexp_up != nil {
		has_shexp = 1
		base := l * (2 * p.n_ff_shexp * dim + dim * p.n_ff_shexp + dim)
		enc_gemv_f32(enc2, m_b_shexp, NS.UInteger(base * 4), m_b_xb, 0, m_moe_hbuf[0], 0, dim, p.n_ff_shexp)
		enc_gemv_f32(enc2, m_b_shexp, NS.UInteger((base + p.n_ff_shexp * dim) * 4), m_b_xb, 0, m_moe_gbuf[0], 0, dim, p.n_ff_shexp)
		enc_elementwise(enc2, m_pso_swiglu, m_moe_gbuf[0], m_moe_hbuf[0], p.n_ff_shexp)
		enc_gemv_f32(enc2, m_b_shexp, NS.UInteger((base + 2 * p.n_ff_shexp * dim) * 4), m_moe_gbuf[0], 0, m_b_moe_shexp_out, 0, p.n_ff_shexp, dim)
		enc_gemv_f32(enc2, m_b_shexp, NS.UInteger((base + 2 * p.n_ff_shexp * dim + dim * p.n_ff_shexp) * 4), m_b_xb, 0, m_b_moe_shexp_gate, 0, dim, 1)
	}
	enc_moe_combine(enc2, used, has_shexp, dim, m_b_xb)
	enc_elementwise(enc2, m_pso_residual, m_b_x, m_b_xb, dim)
	enc2->endEncoding()
	cmd2->commit()
	cmd2->waitUntilCompleted()

	cmd3 := m_queue->commandBuffer()
	enc3 := cmd3->computeCommandEncoder()
	cmd^ = cmd3
	enc^ = enc3
}

enc_gemv :: proc(enc: ^MTL.ComputeCommandEncoder, kind: ggml.GGML_Type, w_off: NS.UInteger,
	xb: ^MTL.Buffer, x_off: NS.UInteger, ob: ^MTL.Buffer, o_off: NS.UInteger, n, d: int) {
	pso := pso_for(kind)
	rows_tg := GEMV_ROWS
	// Kernel selection order: env overrides (QGEMV3/QGEMV2R, handled inside
	// pso_for) > first-run autotune table > production default. The autotune
	// table is per (kind, rows, cols) and adapts to the GPU generation.
	if (kind == .Q4_K || kind == .Q6_K) && !g_gemv3 && !g_gemv2r && len(g_tune) > 0 {
		for e in g_tune {
			if e.kind == kind && e.d == d && e.n == n {
				pso, rows_tg = tune_pso(kind, e.cand)
				break
			}
		}
	} else if (kind == .Q4_K || kind == .Q6_K) && (g_gemv3 || g_gemv2r) {
		rows_tg = 64
	}
	enc->setComputePipelineState(pso)
	enc->setBuffer(m_weights, w_off, 0); enc->setBuffer(xb, x_off, 1); enc->setBuffer(ob, o_off, 2)
	dims := [2]u32{u32(n), u32(d)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	tg_x := (d + rows_tg - 1) / rows_tg
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
}

// Norm-fused GEMV: the kernel recomputes rms(x) cooperatively and folds the
// norm weight (at wn_off in m_weights) into the activation. Requires the fused
// kernel variants for Q4_K/Q6_K; caller guards on g_normfuse.
@(private = "file")
enc_gemv_norm :: proc(enc: ^MTL.ComputeCommandEncoder, kind: ggml.GGML_Type, w_off, wn_off: NS.UInteger,
	xb: ^MTL.Buffer, x_off: NS.UInteger, ob: ^MTL.Buffer, o_off: NS.UInteger, n, d: int, eps: f32) {
	pso := m_pso_q4_knorm
	if kind == .Q6_K { pso = m_pso_q6_knorm }
	enc->setComputePipelineState(pso)
	enc->setBuffer(m_weights, w_off, 0); enc->setBuffer(xb, x_off, 1); enc->setBuffer(ob, o_off, 2)
	dims := [2]u32{u32(n), u32(d)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	enc->setBuffer(m_weights, wn_off, 4)
	eps_local := eps
	enc->setBytes(bytes_of(&eps_local, size_of(eps_local)), 5)
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
// Autotuned (gemm_autotune): M%64==0 -> bm=64 bk=64 kernel (+13-18%); else v3 bm=32.
@(private = "file")
enc_q4k :: proc(enc: ^MTL.ComputeCommandEncoder, w: rawptr, B: ^MTL.Buffer, B_off: NS.UInteger, C: ^MTL.Buffer, C_off: NS.UInteger, M, N, K: int) {
	enc->setBuffer(m_weights, woff(w), 0); enc->setBuffer(B, B_off, 1); enc->setBuffer(C, C_off, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	if M % 64 == 0 {
		enc->setComputePipelineState(m_pso_q4k_b64)
		enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / 8), NS.Integer(M / 64), 1}, MTL.Size{256, 1, 1})
		if !g_b64_seen {
			g_b64_seen = true
			fmt.eprintfln("[b64] gemm_q4k_f32_b64 dispatched: M={} N={} K={}", M, N, K)
		}
	} else {
		enc->setComputePipelineState(m_pso_q4k)
		enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / 8), NS.Integer(M / 32), 1}, MTL.Size{128, 1, 1})
	}
}

// Q6_K MMQ variant (Q4_K_M stores ffn_down/output/qkv as Q6_K). Stays on v3:
// autotune shows bm=64 bk=64 is a ~0.66x REGRESSION for Q6K (w2 4096x512x12288),
// v3 bm=32 bk=32 is already at the platform ceiling (~400 GFLOPS).
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

	// ablation switches: read /tmp/qprof_flags per call so they can be toggled
	// WITHOUT restart (comma-separated: noout, nomlp, nosmall)
	prof: []u8
	if data, ferr := os.read_entire_file("/tmp/qprof_flags", context.temp_allocator); ferr == os.ERROR_NONE {
		prof = data
	}
	prof_s := string(prof)
	noout := strings.contains(prof_s, "noout")
	nomlp := strings.contains(prof_s, "nomlp")
	nosmall := strings.contains(prof_s, "nosmall")
	nonorm := nosmall || strings.contains(prof_s, "nonorm")
	noattn := nosmall || strings.contains(prof_s, "noattn")
	nodelta := nosmall || strings.contains(prof_s, "nodelta")

	cmd := m_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	kv_layer_bytes := NS.UInteger(seq_len * kv_dim * 2)
	conv_state_stride := conv_dim * (kernel - 1)
	rec_state_stride := n_vh * head_k_dim * head_v_dim

	for l in 0 ..< p.n_layers {
		lw := &w.layers[l]
		// shared: input norm + MLP (post_attention_norm)
		if !nonorm && !g_normfuse {
			enc_rmsnorm(enc, m_b_x, m_b_xb, raw_data(lw.attn_norm), dim, 1, eps)
		}

		switch lw.layer_type {
		case .Full_Attention:
			fa := &lw.full
			slot := transformer.full_slot[l]
			kv_loff := NS.UInteger(slot) * kv_layer_bytes
			if g_normfuse {
				enc_gemv_norm(enc, fa.wq.kind, woff(raw_data(fa.wq.data)), woff(raw_data(lw.attn_norm)), m_b_x, 0, m_b_qproj, 0, dim, n_heads * head_dim * 2, eps)
				enc_gemv_norm(enc, fa.wk.kind, woff(raw_data(fa.wk.data)), woff(raw_data(lw.attn_norm)), m_b_x, 0, m_b_ktmp, 0, dim, kv_dim, eps)
				enc_gemv_norm(enc, fa.wv.kind, woff(raw_data(fa.wv.data)), woff(raw_data(lw.attn_norm)), m_b_x, 0, m_b_vtmp, 0, dim, kv_dim, eps)
			} else {
				enc_gemv(enc, fa.wq.kind, woff(raw_data(fa.wq.data)), m_b_xb, 0, m_b_qproj, 0, dim, n_heads * head_dim * 2)
				enc_gemv(enc, fa.wk.kind, woff(raw_data(fa.wk.data)), m_b_xb, 0, m_b_ktmp, 0, dim, kv_dim)
				enc_gemv(enc, fa.wv.kind, woff(raw_data(fa.wv.data)), m_b_xb, 0, m_b_vtmp, 0, dim, kv_dim)
			}
			// extract + norm q (per head, strided in qproj); norm k (contiguous per kv head)
			if !noattn {
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
			} // !nosmall
			enc_gemv(enc, fa.wo.kind, woff(raw_data(fa.wo.data)), m_b_xb3, 0, m_b_xb2, 0, att_head_dim, dim)
			enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb2, dim)

		case .Linear_Attention:
			la := &lw.linear
			slot := transformer.lin_slot[l]
			conv_off := NS.UInteger(slot * conv_state_stride * 4)
			rec_off := NS.UInteger(slot * rec_state_stride * 4)
			if g_normfuse {
				enc_gemv_norm(enc, la.in_qkv.kind, woff(raw_data(la.in_qkv.data)), woff(raw_data(lw.attn_norm)), m_b_x, 0, m_b_qkv, 0, dim, conv_dim, eps)
				enc_gemv_norm(enc, la.in_z.kind, woff(raw_data(la.in_z.data)), woff(raw_data(lw.attn_norm)), m_b_x, 0, m_b_z, 0, dim, value_dim, eps)
				enc_gemv_norm(enc, la.in_b.kind, woff(raw_data(la.in_b.data)), woff(raw_data(lw.attn_norm)), m_b_x, 0, m_b_b, 0, dim, n_vh, eps)
				enc_gemv_norm(enc, la.in_a.kind, woff(raw_data(la.in_a.data)), woff(raw_data(lw.attn_norm)), m_b_x, 0, m_b_a, 0, dim, n_vh, eps)
			} else {
				enc_gemv(enc, la.in_qkv.kind, woff(raw_data(la.in_qkv.data)), m_b_xb, 0, m_b_qkv, 0, dim, conv_dim)
				enc_gemv(enc, la.in_z.kind, woff(raw_data(la.in_z.data)), m_b_xb, 0, m_b_z, 0, dim, value_dim)
				enc_gemv(enc, la.in_b.kind, woff(raw_data(la.in_b.data)), m_b_xb, 0, m_b_b, 0, dim, n_vh)
				enc_gemv(enc, la.in_a.kind, woff(raw_data(la.in_a.data)), m_b_xb, 0, m_b_a, 0, dim, n_vh)
			}
			// conv1d step + silu (weight is F32 in mmap)
			if !nodelta {
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
			} // !nosmall
			enc_gemv(enc, la.out.kind, woff(raw_data(la.out.data)), m_b_lout, 0, m_b_xb2, 0, value_dim, dim)
			enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb2, dim)
		}

		// shared MLP (post_attention_norm)
		if !nonorm && !g_normfuse && !p.is_moe {
			enc_rmsnorm(enc, m_b_x, m_b_xb, raw_data(lw.ffn_norm), dim, 1, eps)
		}
		if !nomlp {
			if p.is_moe {
				moe_ffn_gpu(&enc, &cmd, transformer, l, nonorm)
			} else if g_normfuse {
				enc_gemv_norm(enc, lw.w1.kind, woff(raw_data(lw.w1.data)), woff(raw_data(lw.ffn_norm)), m_b_x, 0, m_b_hb, 0, dim, hidden_dim, eps)
				enc_gemv_norm(enc, lw.w3.kind, woff(raw_data(lw.w3.data)), woff(raw_data(lw.ffn_norm)), m_b_x, 0, m_b_hb2, 0, dim, hidden_dim, eps)
			} else {
				enc_gemv(enc, lw.w1.kind, woff(raw_data(lw.w1.data)), m_b_xb, 0, m_b_hb, 0, dim, hidden_dim)
				enc_gemv(enc, lw.w3.kind, woff(raw_data(lw.w3.data)), m_b_xb, 0, m_b_hb2, 0, dim, hidden_dim)
			}
			if !p.is_moe {
				if !nosmall {
					enc_elementwise(enc, m_pso_swiglu, m_b_hb, m_b_hb2, hidden_dim)
				}
				enc_gemv(enc, lw.w2.kind, woff(raw_data(lw.w2.data)), m_b_hb, 0, m_b_xb, 0, hidden_dim, dim)
				enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb, dim)
			}
		}
	}

	if !nonorm && !g_normfuse {
		enc_rmsnorm(enc, m_b_x, m_b_x, raw_data(w.output_norm), dim, 1, eps)
	}
	if !noout {
		if g_normfuse {
			enc_gemv_norm(enc, w.output.kind, woff(raw_data(w.output.data)), woff(raw_data(w.output_norm)), m_b_x, 0, m_b_logits, 0, dim, p.vocab_size, eps)
		} else {
			enc_gemv(enc, w.output.kind, woff(raw_data(w.output.data)), m_b_x, 0, m_b_logits, 0, dim, p.vocab_size)
		}
	}

	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
	if g_timing {
		gpu_ms := f64(cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0
		fmt.eprintfln("[timing] forward_gpu pos={} GPU={:.2f} ms", pos, gpu_ms)
	}
	return m_b_logits->contentsAsSlice([]f32)[:p.vocab_size]
}

// Batched prefill (BATCHED_PREFILL Stage 1a): process T tokens at once, batching
// the MLP projections (w1/w3/w2) via Q4_K MMQ. Attention (full + linear) still
// runs per-token on single-token scratch (its projections are GEMV; batching them
// is Stage 1b). T must be a multiple of 8 and <= MAX_BATCH_T (caller chunks; the
// trailing <8 remainder goes through forward_gpu). Returns the LAST token's logits.
forward_gpu_batch :: proc(transformer: ^Transformer, tokens: []int, pos_start: int, n_real: int) -> []f32 {
	NS.scoped_autoreleasepool()
	p := &transformer.config
	w := &transformer.weights

	T := len(tokens)
	if T == 0 { fmt.eprintln("forward_gpu_batch: empty"); os.exit(1) }
	if T > MAX_BATCH_T || (T % 8) != 0 {
		fmt.eprintf("forward_gpu_batch: T=%d must be in (0,%d] and %%8==0\n", T, MAX_BATCH_T)
		os.exit(1)
	}
	if n_real < 1 || n_real > T {
		fmt.eprintf("forward_gpu_batch: n_real=%d must be in [1,%d]\n", n_real, T)
		os.exit(1)
	}
	if n_real < p.lin_conv_kernel - 1 {
		// conv1d_update_state slides to the last km1 inputs and needs n_real >= km1
		fmt.eprintf("forward_gpu_batch: n_real=%d too small for conv update (need >= %d)\n", n_real, p.lin_conv_kernel - 1)
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
			for t in 0 ..< n_real {
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
			for t in 0 ..< n_real {
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
			enc_conv1d_update(enc, m_b_qkv_t, m_b_conv_states, conv_off, conv_dim, n_real, kernel)
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

	// final norm + output projection: LAST REAL token only (padded tail: skip filler)
	last_off := NS.UInteger(n_real - 1) * dim_bytes
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

// Save full engine state (KV cache + recurrent + conv) to memory for prefix caching.
// Returns heap-allocated byte slices that the caller owns.
metal_save_full_state :: proc() -> (kv_k: []u8, kv_v: []u8, conv: []u8, rec: []u8) {
	if !metal_enabled { return }
	kc := m_b_kc->contentsAsSlice([]u8)
	vc := m_b_vc->contentsAsSlice([]u8)
	kv_k = make([]u8, len(kc))
	kv_v = make([]u8, len(vc))
	copy(kv_k, kc)
	copy(kv_v, vc)
	conv, rec = metal_save_spec_state()
	return
}

// Restore full engine state from memory snapshot.
metal_restore_full_state :: proc(kv_k, kv_v, conv, rec: []u8) {
	if !metal_enabled { return }
	kc := m_b_kc->contentsAsSlice([]u8)
	vc := m_b_vc->contentsAsSlice([]u8)
	copy(kc, kv_k)
	copy(vc, kv_v)
	metal_restore_spec_state(conv, rec)
}

// Save recurrent + conv states for speculative decoding rollback.
// Returns heap-allocated byte slices that the caller owns.
metal_save_spec_state :: proc() -> (conv: []u8, rec: []u8) {
	if !metal_enabled { return }
	conv_src := m_b_conv_states->contentsAsSlice([]u8)
	rec_src := m_b_rec_states->contentsAsSlice([]u8)
	conv = make([]u8, len(conv_src))
	rec = make([]u8, len(rec_src))
	copy(conv, conv_src)
	copy(rec, rec_src)
	return
}

// Restore recurrent + conv states from snapshot.
metal_restore_spec_state :: proc(conv: []u8, rec: []u8) {
	if !metal_enabled { return }
	conv_dst := m_b_conv_states->contentsAsSlice([]u8)
	rec_dst := m_b_rec_states->contentsAsSlice([]u8)
	copy(conv_dst, conv)
	copy(rec_dst, rec)
}

// Verify batch: extract logits for ALL positions in the last forward_gpu_batch.
// m_b_bx holds [T × dim] hidden states after batch processing. For each
// position, do RMSNorm + output GEMV to get that position's logits.
// Returns a flat [T × vocab] f32 slice (caller slices into rows).
// Cost: ~0.2ms × T on M3 (just GEMVs, layers already computed).
metal_verify_batch_logits :: proc(t: ^Transformer, T: int) -> []f32 {
	if !metal_enabled || T <= 0 { return {} }
	p := t.config
	w := t.weights
	dim := p.dim
	vocab := p.vocab_size
	dim_bytes := NS.UInteger(dim * 4)

	// Need a larger logits buffer: [T × vocab] instead of [1 × vocab]
	bytes_needed := NS.UInteger(T * vocab * 4)
	if m_b_verify_logits == nil {
		m_b_verify_logits = new_shared_bytes(T * vocab * 4)
	} else if m_b_verify_logits->length() < bytes_needed {
		m_b_verify_logits->release()
		m_b_verify_logits = new_shared_bytes(T * vocab * 4)
	}

	for i in 0 ..< T {
		cmd := m_queue->commandBuffer()
		enc := cmd->computeCommandEncoder()

		// Copy hidden state i from batch buffer to m_b_x
		off := NS.UInteger(i) * dim_bytes
		enc_copy(enc, m_b_bx, off, m_b_x, 0, dim)
		// RMSNorm
		enc_rmsnorm(enc, m_b_x, m_b_x, raw_data(w.output_norm), dim, 1, p.rms_eps)
		// GEMV → m_b_logits (reuse single-row buffer, then copy)
		enc_gemv(enc, w.output.kind, woff(raw_data(w.output.data)),
			m_b_x, 0, m_b_logits, 0, dim, vocab)

		enc->endEncoding()
		cmd->commit()
		cmd->waitUntilCompleted()

		// Copy from m_b_logits to position i in the verify buffer
		verify_slice := m_b_verify_logits->contentsAsSlice([]f32)[i * vocab : (i + 1) * vocab]
		logit_slice := m_b_logits->contentsAsSlice([]f32)[:vocab]
		copy(verify_slice, logit_slice)
	}

	return m_b_verify_logits->contentsAsSlice([]f32)[:T * vocab]
}
