/* Print the token ids of a prompt using the embedded Qwen3.5 tokenizer.
   odin build tools/encode_print.odin -file -out:bin/encode_print ...collections
   encode_print "prompt text" */
package main

import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:fmt"
import "core:os"

main :: proc() {
	if len(os.args) < 2 {
		fmt.eprintln("usage: encode_print <prompt>")
		os.exit(1)
	}
	prompt := os.args[1]
	tok: tok35.Tokenizer
	tok35.build_tokenizer(&tok)
	defer tok35.free_tokenizer(&tok)
	encoded, err := tok35.encode(&tok, prompt)
	if err != nil || len(encoded) < 2 {
		fmt.eprintln("encode_print: encode failed")
		os.exit(1)
	}
	defer delete(encoded)
	for t in encoded {
		fmt.printf("%d ", t)
	}
	fmt.println()
}
