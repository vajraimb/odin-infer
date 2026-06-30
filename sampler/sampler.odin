/* Sampling strategies for token generation */

package sampler

import infer "infer:infer"

import "core:math"
import "core:mem"
import "core:slice"

Prob_Index :: struct {
	prob:  f32,
	index: int,
}

Sampler :: struct {
	vocab_size: int,
	probindex:  []Prob_Index,
	temperature: f32,
	topp:        f32,
	rng_state:   u64,
	repeat_penalty: f32,      // 1.0 = off; >1 penalises tokens seen in last_tokens
	last_tokens:    [dynamic]int, // recent token ids (prompt + generated) to penalise
}

build_sampler :: proc(s: ^Sampler, vocab_size: int, temperature, topp: f32, rng_seed: u64) {
	s.vocab_size = vocab_size
	s.temperature = temperature
	s.topp = topp
	s.rng_state = rng_seed
	s.repeat_penalty = 1.0
	s.probindex = make([]Prob_Index, vocab_size)
}

// Enable CTRL-style repetition penalty: any token present in the recent window
// has its logit divided (if positive) or multiplied (if negative) by `penalty`.
enable_repeat_penalty :: proc(s: ^Sampler, penalty: f32) {
	p := penalty < 1.0 ? 1.0 : penalty
	s.repeat_penalty = p
	if s.last_tokens == nil {
		s.last_tokens = make([dynamic]int, 0, 128)
	}
}

record_token :: proc(s: ^Sampler, token: int, window: int = 64) {
	if s.repeat_penalty == 1.0 do return
	append(&s.last_tokens, token)
	for len(s.last_tokens) > window {
		ordered_remove(&s.last_tokens, 0)
	}
}

free_sampler :: proc(s: ^Sampler) {
	delete(s.probindex)
	if s.last_tokens != nil {
		delete(s.last_tokens)
	}
}

random_u32 :: proc(state: ^u64) -> u32 {
	state^ ~= state^ >> 12
	state^ ~= state^ << 25
	state^ ~= state^ >> 27
	return u32((state^ * 0x2545F4914F6CDD1D) >> 32)
}

random_f32 :: proc(state: ^u64) -> f32 {
	return f32(random_u32(state) >> 8) / 16777216.0
}

sample_argmax :: proc(probabilities: []f32) -> int {
	max_i := 0
	max_p := probabilities[0]
	for i in 1 ..< len(probabilities) {
		if probabilities[i] > max_p {
			max_i = i
			max_p = probabilities[i]
		}
	}
	return max_i
}

sample_mult :: proc(probabilities: []f32, coin: f32) -> int {
	cdf: f32 = 0
	for i in 0 ..< len(probabilities) {
		cdf += probabilities[i]
		if coin < cdf {
			return i
		}
	}
	return len(probabilities) - 1
}

prob_index_cmp :: proc(a, b: Prob_Index) -> bool {
	return a.prob > b.prob
}

sample_topp :: proc(probabilities: []f32, topp: f32, probindex: []Prob_Index, coin: f32) -> int {
	n := len(probabilities)
	n0 := 0
	cutoff := (1.0 - topp) / f32(n - 1)
	for i in 0 ..< n {
		if probabilities[i] >= cutoff {
			probindex[n0].index = i
			probindex[n0].prob = probabilities[i]
			n0 += 1
		}
	}

	slice.sort_by(probindex[:n0], prob_index_cmp)

	cumulative_prob: f32 = 0
	last_idx := n0 - 1
	for i in 0 ..< n0 {
		cumulative_prob += probindex[i].prob
		if cumulative_prob > topp {
			last_idx = i
			break
		}
	}

	r := coin * cumulative_prob
	cdf: f32 = 0
	for i in 0 ..= last_idx {
		cdf += probindex[i].prob
		if r < cdf {
			return probindex[i].index
		}
	}
	return probindex[last_idx].index
}

sample :: proc(s: ^Sampler, logits: []f32) -> int {
	// CTRL repetition penalty: penalise each unique token in the recent window.
	if s.repeat_penalty != 1.0 && len(s.last_tokens) > 0 {
		seen := make([dynamic]int, 0, 64, allocator = context.temp_allocator)
		defer delete(seen)
		for tok in s.last_tokens {
			if tok < 0 || tok >= s.vocab_size do continue
			dup := false
			for st in seen {
				if st == tok {
					dup = true
					break
				}
			}
			if dup do continue
			append(&seen, tok)
			if logits[tok] > 0 {
				logits[tok] /= s.repeat_penalty
			} else {
				logits[tok] *= s.repeat_penalty
			}
		}
	}

	if s.temperature == 0.0 {
		return sample_argmax(logits)
	}

	for i in 0 ..< s.vocab_size {
		logits[i] /= s.temperature
	}
	infer.softmax(logits[:s.vocab_size])

	coin := random_f32(&s.rng_state)
	if s.topp <= 0 || s.topp >= 1 {
		return sample_mult(logits[:s.vocab_size], coin)
	}
	return sample_topp(logits[:s.vocab_size], s.topp, s.probindex, coin)
}
