/* tools/metal_smoke.odin — model-free Metal shader smoke test (CI gate).

   1. Compiles the embedded MSL library (fastMath on and off).
   2. Builds a compute pipeline for every kernel in it.
   3. Cross-checks that every make_pso(lib, "<name>") referenced by the engine
      sources actually exists in the compiled library.

   Exit 0 on success, 1 on any failure. Run:
     odin run tools/metal_smoke.odin -file <collections>
*/
package metal_smoke

import "core:fmt"
import "core:os"
import "core:strings"
import q35 "qwen3_5:qwen3_5"

// Engine sources whose make_pso(...) names must resolve in the library.
SOURCES := [?]string{
	#load("../packages/qwen3_5/metal.odin", string),
	#load("../packages/qwen3_5/moe.odin", string),
}

main :: proc() {
	names, ok := q35.metal_shader_selftest()
	have := make(map[string]bool)
	for n in names { have[n] = true }

	needle :: `make_pso(lib, "`
	referenced, missing := 0, 0
	for src in SOURCES {
		rest := src
		for {
			i := strings.index(rest, needle)
			if i < 0 { break }
			rest = rest[i + len(needle):]
			j := strings.index_byte(rest, '"')
			if j < 0 { break }
			name := rest[:j]
			referenced += 1
			if !have[name] {
				fmt.eprintfln("metal smoke: engine references kernel '%s' but it is not in the library", name)
				missing += 1
			}
			rest = rest[j:]
		}
	}
	fmt.eprintfln("metal smoke: %d kernels in library, %d make_pso references, %d missing", len(names), referenced, missing)

	if !ok || missing > 0 || referenced == 0 {
		fmt.eprintln("metal smoke: FAILED")
		os.exit(1)
	}
	fmt.eprintln("metal smoke: PASSED")
}
