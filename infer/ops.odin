/* Elementwise ops shared by forward pass and sampler */

package infer

import "core:math"

rmsnorm :: proc(o, x, weight: []f32, eps: f32) {
	size := len(x)
	ss: f32 = 0
	for j in 0 ..< size {
		ss += x[j] * x[j]
	}
	ss /= f32(size)
	ss += eps
	ss = 1.0 / math.sqrt_f32(ss)
	for j in 0 ..< size {
		o[j] = weight[j] * (ss * x[j])
	}
}

softmax :: proc(x: []f32) {
	if len(x) == 0 do return
	max_val := x[0]
	for i in 1 ..< len(x) {
		if x[i] > max_val {
			max_val = x[i]
		}
	}
	sum: f32 = 0
	for i in 0 ..< len(x) {
		x[i] = math.exp_f32(x[i] - max_val)
		sum += x[i]
	}
	for i in 0 ..< len(x) {
		x[i] /= sum
	}
}
