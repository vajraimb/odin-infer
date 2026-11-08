/* Multithreaded matrix-vector multiply with on-the-fly dequantization */

package infer

import ggml "ggml:ggml"

import "base:intrinsics"
import "core:thread"

MAX_MATMUL_N :: 65536 // per-thread dequant scratch length

Matmul_Context :: struct {
	kind:      ggml.GGML_Type,
	w:         []u8,
	x:         []f32,
	xout:      []f32,
	n:         int,
	d:         int,
	row_bytes: int,
	next_row:  int,
}

matmul_pool: thread.Pool
matmul_num_threads: int
matmul_pool_inited: bool
matmul_scratch: [][]f32

matmul_set_threads :: proc(n: int) {
	matmul_num_threads = max(n, 1)
}

ensure_matmul :: proc() {
	if matmul_scratch == nil {
		n := max(matmul_num_threads, 1)
		matmul_scratch = make([][]f32, n)
		for i in 0 ..< n {
			matmul_scratch[i] = make([]f32, MAX_MATMUL_N)
		}
	}
	// The calling thread participates, so the pool needs num_threads-1 workers.
	if matmul_num_threads > 1 && !matmul_pool_inited {
		thread.pool_init(&matmul_pool, context.allocator, matmul_num_threads - 1)
		thread.pool_start(&matmul_pool)
		matmul_pool_inited = true
	}
}

destroy_matmul_pool :: proc() {
	if matmul_pool_inited {
		thread.pool_finish(&matmul_pool)
		thread.pool_destroy(&matmul_pool)
		matmul_pool_inited = false
	}
	if matmul_scratch != nil {
		for s in matmul_scratch {
			delete(s)
		}
		delete(matmul_scratch)
		matmul_scratch = nil
	}
}

run_rows :: proc(ctx: ^Matmul_Context, scratch_idx: int) {
	scratch := matmul_scratch[scratch_idx]
	for {
		row := intrinsics.atomic_add(&ctx.next_row, 1)
		if row >= ctx.d do break
		ctx.xout[row] = ggml.dot_row(ctx.kind, ctx.w[row * ctx.row_bytes:], ctx.x, ctx.n, scratch)
	}
}

matmul_worker :: proc(task: thread.Task) {
	run_rows(cast(^Matmul_Context)task.data, task.user_index)
}

// xout[d] = W[d,n] * x[n], where W is a (possibly quantized) tensor.
matmul_t :: proc(xout, x: []f32, w: ^Tensor, n, d: int) {
	ensure_matmul()
	rb := ggml.row_byte_size(w.kind, n)

	ctx := Matmul_Context {
		kind      = w.kind,
		w         = w.data,
		x         = x,
		xout      = xout,
		n         = n,
		d         = d,
		row_bytes = rb,
		next_row  = 0,
	}

	if matmul_num_threads <= 1 {
		run_rows(&ctx, 0)
		return
	}

	for t in 0 ..< matmul_num_threads - 1 {
		thread.pool_add_task(&matmul_pool, context.allocator, matmul_worker, &ctx, t)
	}
	// Calling thread does its share using the last scratch slot.
	run_rows(&ctx, matmul_num_threads - 1)

	for thread.pool_num_outstanding(&matmul_pool) > 0 {
		thread.yield()
	}
	for thread.pool_num_done(&matmul_pool) > 0 {
		thread.pool_pop_done(&matmul_pool)
	}
}
