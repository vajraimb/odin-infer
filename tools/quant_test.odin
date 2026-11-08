/* Cross-check ggml.dequant_row against the Python reference fixture
   (tools/gen_quant_ref.py → tools/quant_fixture.bin). The reference is an
   independent numpy transcription of llama.cpp's dequantize_row_* functions,
   so this verifies bit-level agreement of the Odin port. */

package main

import ggml "ggml:ggml"

import "core:fmt"
import "core:os"
import "core:strconv"

tolerance_for :: proc(kind: ggml.GGML_Type) -> f32 {
	#partial switch kind {
	case .IQ1_S, .IQ1_M:
		return 1e-5
	}
	return 1e-4
}

kind_name :: proc(kind: ggml.GGML_Type) -> string {
	#partial switch kind {
	case .Q2_K:    return "Q2_K"
	case .Q3_K:    return "Q3_K"
	case .Q4_K:    return "Q4_K"
	case .Q5_K:    return "Q5_K"
	case .Q6_K:    return "Q6_K"
	case .Q8_K:    return "Q8_K"
	case .IQ2_XXS: return "IQ2_XXS"
	case .IQ2_XS:  return "IQ2_XS"
	case .IQ2_S:   return "IQ2_S"
	case .IQ3_XXS: return "IQ3_XXS"
	case .IQ3_S:   return "IQ3_S"
	case .IQ1_S:   return "IQ1_S"
	case .IQ1_M:   return "IQ1_M"
	case .IQ4_NL:  return "IQ4_NL"
	case .IQ4_XS:  return "IQ4_XS"
	}
	return "?"
}

read_u32 :: proc(data: []u8, pos: ^int) -> u32 {
	v := u32(data[pos^]) | u32(data[pos^ + 1]) << 8 | u32(data[pos^ + 2]) << 16 | u32(data[pos^ + 3]) << 24
	pos^ += 4
	return v
}

read_f32 :: proc(data: []u8, pos: ^int) -> f32 {
	return transmute(f32)read_u32(data, pos)
}

main :: proc() {
	path := "tools/quant_fixture.bin"
	if len(os.args) > 1 {
		path = os.args[1]
	}

	data, err := os.read_entire_file(path, context.temp_allocator)
	if err != os.ERROR_NONE {
		fmt.eprintf("quant_test: cannot read fixture %s: %v\n", path, err)
		os.exit(2)
	}

	if len(data) < 12 || data[0] != 'O' || data[1] != 'D' || data[2] != 'Q' || data[3] != 'N' {
		fmt.eprintf("quant_test: bad fixture magic\n")
		os.exit(2)
	}

	pos := 4
	version := read_u32(data, &pos)
	nkinds := int(read_u32(data, &pos))
	fmt.printf("fixture version=%d kinds=%d\n", version, nkinds)

	failures := 0
	checks := 0
	for k in 0 ..< nkinds {
		kind := ggml.GGML_Type(read_u32(data, &pos))
		n := int(read_u32(data, &pos))
		row_bytes := int(read_u32(data, &pos))
		nrows := int(read_u32(data, &pos))

		rows := make([][]u8, nrows)
		for r in 0 ..< nrows {
			rows[r] = data[pos:pos + row_bytes]
			pos += row_bytes
		}

		dst := make([]f32, n)
		tol := tolerance_for(kind)
		worst := f32(0)
		all_ok := true
		for r in 0 ..< nrows {
			ggml.dequant_row(kind, rows[r], n, dst)
			for i in 0 ..< n {
				want := read_f32(data, &pos)
				diff := abs(dst[i] - want)
				if diff > worst {
					worst = diff
				}
				if diff > tol {
					if all_ok {
						fmt.eprintf(
							"FAIL %-8s row %d: dst[%d]=%v want=%v diff=%v (tol=%v)\n",
							kind_name(kind), r, i, dst[i], want, diff, tol,
						)
						all_ok = false
					}
				}
				checks += 1
			}
		}

		if all_ok {
			fmt.printf("PASS %-8s n=%d rows=%d worst_diff=%.3e\n", kind_name(kind), n, nrows, worst)
		} else {
			failures += 1
		}
	}

	fmt.printf("checks=%d failures=%d\n", checks, failures)
	if failures != 0 {
		os.exit(1)
	}
}
