/* Runtime-loaded CUDA driver API + NVRTC.
   No nvcc and no libcuda link at Odin compile time. The .so is dlopen'd.
   NVRTC load order:
     1. $ODINFER_NVRTC_PATH if set (file or directory). Exclusive.
     2. $NVRTC_DIR if set (directory or a full soname path), then the
        loader sonames, then a pip nvidia-cuda-nvrtc-cu12 tree.
     3. otherwise versioned sonames, then bare libnvrtc.so, then pip.
   compute_70 / compute_72 (sm_70, Tesla V100) cannot be compiled by CUDA 13
   NVRTC, which dropped archs below sm_75. Those compiles search NVRTC 12.x
   only (libnvrtc.so.12, then a pip nvidia-cuda-nvrtc-cu12 tree). A 13.x
   driver can still JIT the 12.9 PTX. Callers pass
   --gpu-architecture=compute_XX; CUDA 13's default is sm_75. */

package cuda

import "core:c"
import "core:dynlib"
import "core:fmt"
import "core:os"
import "core:strings"

Error :: distinct i32

OK :: Error(0)
ERR_LIBRARY :: Error(-1)
ERR_SYMBOL :: Error(-2)
ERR_COMPILE :: Error(-3)

Device :: distinct i32
Context :: distinct rawptr
Module :: distinct rawptr
Function :: distinct rawptr
Device_Ptr :: distinct u64

// cuGetErrorString writes a static C string; the pointer is stable.
Driver :: struct {
	Init:                       proc "c" (flags: c.uint) -> Error,
	GetErrorString:             proc "c" (err: Error, str: ^cstring) -> Error,
	DeviceGet:                  proc "c" (dev: ^Device, ordinal: c.int) -> Error,
	DeviceGetCount:             proc "c" (count: ^c.int) -> Error,
	DeviceGetName:              proc "c" (name: [^]u8, len: c.int, dev: Device) -> Error,
	DeviceGetAttribute:         proc "c" (pi: ^c.int, attrib: c.int, dev: Device) -> Error,
	DeviceTotalMem_v2:          proc "c" (bytes: ^c.size_t, dev: Device) -> Error,
	DevicePrimaryCtxRetain:     proc "c" (ctx: ^Context, dev: Device) -> Error,
	DevicePrimaryCtxRelease_v2: proc "c" (dev: Device) -> Error,
	CtxSetCurrent:              proc "c" (ctx: Context) -> Error,
	CtxGetCurrent:              proc "c" (ctx: ^Context) -> Error,
	CtxSynchronize:             proc "c" () -> Error,
	MemAlloc_v2:                proc "c" (ptr: ^Device_Ptr, bytes: c.size_t) -> Error,
	MemFree_v2:                 proc "c" (ptr: Device_Ptr) -> Error,
	MemcpyHtoD_v2:              proc "c" (dst: Device_Ptr, src: rawptr, bytes: c.size_t) -> Error,
	MemcpyDtoH_v2:              proc "c" (dst: rawptr, src: Device_Ptr, bytes: c.size_t) -> Error,
	MemcpyDtoD_v2:              proc "c" (dst: Device_Ptr, src: Device_Ptr, bytes: c.size_t) -> Error,
	MemsetD8_v2:                proc "c" (ptr: Device_Ptr, value: c.uchar, n: c.size_t) -> Error,
	ModuleLoadDataEx:           proc "c" (mod: ^Module, image: rawptr, nopts: c.uint, opts: [^]c.int, vals: [^]rawptr) -> Error,
	ModuleGetFunction:          proc "c" (fn: ^Function, mod: Module, name: cstring) -> Error,
	ModuleUnload:               proc "c" (mod: Module) -> Error,
	LaunchKernel:               proc "c" (fn: Function, gx, gy, gz, bx, by, bz, shared: c.uint, stream: rawptr, params: [^]rawptr, extra: rawptr) -> Error,
	__handle:                   dynlib.Library,
}

Nvrtc :: struct {
	Version:           proc "c" (major, minor: ^c.int) -> Error,
	GetErrorString:    proc "c" (result: Error) -> cstring,
	CreateProgram:     proc "c" (prog: ^rawptr, src: cstring, name: cstring, num_headers: c.int, headers: [^]cstring, include_names: [^]cstring) -> Error,
	// nvrtcDestroyProgram takes nvrtcProgram* and nulls it.
	DestroyProgram:    proc "c" (prog: ^rawptr) -> Error,
	CompileProgram:    proc "c" (prog: rawptr, num_options: c.int, options: [^]cstring) -> Error,
	GetPTXSize:        proc "c" (prog: rawptr, size: ^c.size_t) -> Error,
	GetPTX:            proc "c" (prog: rawptr, ptx: [^]u8) -> Error,
	GetProgramLogSize: proc "c" (prog: rawptr, size: ^c.size_t) -> Error,
	GetProgramLog:     proc "c" (prog: rawptr, log: [^]u8) -> Error,
	__handle:          dynlib.Library,
}

drv: Driver
nv: Nvrtc
nv_loaded_from: string

ATTR_MULTIPROCESSOR_COUNT :: c.int(16)
ATTR_COMPUTE_CAPABILITY_MAJOR :: c.int(75)
ATTR_COMPUTE_CAPABILITY_MINOR :: c.int(76)

// CU_JIT_ERROR_LOG_BUFFER / SIZE
JIT_ERROR_LOG_BUFFER :: c.int(5)
JIT_ERROR_LOG_BUFFER_SIZE_BYTES :: c.int(6)

BACKEND_TAG :: "cuda"

driver_symbol_count :: proc() -> int {
	n := 0
	// Keep in sync with Driver fields that are procedures. The handle is not a symbol.
	n = size_of(Driver) / size_of(rawptr) - 1
	return n
}

nvrtc_symbol_count :: proc() -> int {
	return size_of(Nvrtc) / size_of(rawptr) - 1
}

load_driver :: proc() -> Error {
	if drv.__handle != nil {
		return OK
	}
	names: []string
	when ODIN_OS == .Windows {
		names = []string{"nvcuda.dll"}
	} else when ODIN_OS == .Linux {
		names = []string{"libcuda.so.1", "libcuda.so"}
	} else {
		names = []string{}
	}
	for name in names {
		count, ok := dynlib.initialize_symbols(&drv, name, "cu")
		if drv.__handle == nil {
			drv = {}
			continue
		}
		if !ok || count != driver_symbol_count() {
			dynlib.unload_library(drv.__handle)
			drv = {}
			fmt.eprintf("cuda: %s is missing driver symbols (got %d, want %d)\n", name, count, driver_symbol_count())
			return ERR_SYMBOL
		}
		return OK
	}
	return ERR_LIBRARY
}

@(private)
preload_builtins :: proc(dir: string) {
	when ODIN_OS == .Linux {
		if len(dir) == 0 {
			return
		}
		// Wheels and CUDA 12.x/13.x ship libnvrtc-builtins.so.<major>.<minor>
		// (for example libnvrtc-builtins.so.12.9). The unversioned ".12" name
		// is not always present. Scan the directory and skip the ".alt." copy.
		fixed := [5]string{
			"libnvrtc-builtins.so.13",
			"libnvrtc-builtins.so.12",
			"libnvrtc-builtins.so",
			"libnvrtc-builtins.so.13.0",
			"libnvrtc-builtins.so.12.9",
		}
		for name in fixed {
			lib, ok := dynlib.load_library(join_path(dir, name), true)
			if ok && lib != nil {
				return
			}
		}
		f, err := os.open(dir)
		if err != os.ERROR_NONE {
			return
		}
		defer os.close(f)
		files, rerr := os.read_directory(f, -1, context.temp_allocator)
		if rerr != os.ERROR_NONE {
			return
		}
		fallback := ""
		for info in files {
			name := info.name
			if !strings.has_prefix(name, "libnvrtc-builtins") {
				continue
			}
			if strings.contains(name, ".alt.") {
				if fallback == "" {
					fallback = name
				}
				continue
			}
			lib, ok := dynlib.load_library(join_path(dir, name), true)
			if ok && lib != nil {
				return
			}
		}
		if fallback != "" {
			dynlib.load_library(join_path(dir, fallback), true)
		}
	}
}

@(private)
is_soname_path :: proc(path: string) -> bool {
	return strings.has_suffix(path, ".so") || strings.contains(path, ".so.")
}

@(private)
nvrtc_major :: proc() -> int {
	if nv.Version == nil {
		return 0
	}
	maj, min: c.int
	if nv.Version(&maj, &min) != OK {
		return 0
	}
	return int(maj)
}

@(private)
unload_nv :: proc() {
	if nv.__handle != nil {
		dynlib.unload_library(nv.__handle)
	}
	nv = {}
	if nv_loaded_from != "" {
		delete(nv_loaded_from)
		nv_loaded_from = ""
	}
}

// CUDA 13 NVRTC cannot emit compute_70 / compute_72. A 13.x driver can JIT
// PTX produced by NVRTC 12.9.
nvrtc12_hint :: proc() {
	fmt.eprintln("cuda: CUDA 13 NVRTC cannot target compute_70 (sm_70, Tesla V100). Archs below sm_75 were removed.")
	fmt.eprintln("cuda: a 13.x driver can still JIT PTX from NVRTC 12.9. Install that library and set ODINFER_NVRTC_PATH:")
	fmt.eprintln("cuda:   python3 -m pip install --user nvidia-cuda-nvrtc-cu12")
	fmt.eprintln("cuda:   export ODINFER_NVRTC_PATH=\"$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12\"")
}

@(private)
append_linux_names :: proc(dst: ^[dynamic]string, dir: string, prefer12: bool) {
	if prefer12 {
		append(dst, join_path(dir, "libnvrtc.so.12"))
		append(dst, join_path(dir, "libnvrtc.so"))
		return
	}
	append(dst, join_path(dir, "libnvrtc.so.13"))
	append(dst, join_path(dir, "libnvrtc.so.12"))
	append(dst, join_path(dir, "libnvrtc.so"))
}

// pip nvidia-cuda-nvrtc-cu12 drops libnvrtc.so.12 under site-packages.
@(private)
append_pip_nvrtc :: proc(dst: ^[dynamic]string, prefer12: bool) {
	home := os.get_env("HOME", context.temp_allocator)
	py3 := "/usr/lib/python3/dist-packages/nvidia/cuda_nvrtc/lib"
	if os.exists(py3) {
		append_linux_names(dst, py3, prefer12)
	}
	vers := [?]string{"3.14", "3.13", "3.12", "3.11", "3.10", "3.9", "3.8"}
	for v in vers {
		cands := [3]string{
			fmt.tprintf("%s/.local/lib/python%s/site-packages/nvidia/cuda_nvrtc/lib", home, v),
			fmt.tprintf("/usr/local/lib/python%s/dist-packages/nvidia/cuda_nvrtc/lib", v),
			fmt.tprintf("/usr/lib/python%s/dist-packages/nvidia/cuda_nvrtc/lib", v),
		}
		for d in cands {
			if os.exists(d) {
				append_linux_names(dst, d, prefer12)
			}
		}
	}
}

@(private)
try_nvrtc_candidates :: proc(candidates: []string, prefer12: bool) -> Error {
	want := nvrtc_symbol_count()
	for path in candidates {
		dir := path
		if is_soname_path(path) {
			dir = parent_dir(path)
		}
		preload_builtins(dir)
		count, ok := dynlib.initialize_symbols(&nv, path, "nvrtc")
		if nv.__handle == nil {
			nv = {}
			continue
		}
		if !ok || count != want {
			dynlib.unload_library(nv.__handle)
			nv = {}
			continue
		}
		if prefer12 && nvrtc_major() >= 13 {
			fmt.eprintfln("cuda: skipping NVRTC %d at %s (cannot target compute_70)", nvrtc_major(), path)
			dynlib.unload_library(nv.__handle)
			nv = {}
			continue
		}
		if nv_loaded_from != "" {
			delete(nv_loaded_from)
		}
		nv_loaded_from = strings.clone(path)
		maj, min: c.int
		if nv.Version != nil && nv.Version(&maj, &min) == OK {
			fmt.eprintfln("cuda: NVRTC %d.%d from %s", int(maj), int(min), path)
		}
		return OK
	}
	return ERR_LIBRARY
}

// load_nvrtc opens NVRTC. `dir` overrides the env paths when non-empty (tests).
// prefer12 limits the search to NVRTC 12.x, which is required for compute_70.
load_nvrtc :: proc(dir := "", prefer12 := false) -> Error {
	if nv.__handle != nil {
		if !prefer12 || nvrtc_major() < 13 {
			return OK
		}
		fmt.eprintfln("cuda: loaded NVRTC %d from %s cannot target compute_70; looking for NVRTC 12.x", nvrtc_major(), nv_loaded_from)
		unload_nv()
	}

	// ODINFER_NVRTC_PATH and the dir argument are exclusive. NVRTC_DIR is
	// only the first place to look: on sm_70 a directory that holds NVRTC
	// 13 must not hide a pip-installed libnvrtc.so.12.
	from := dir
	exclusive := dir != ""
	if from == "" {
		from = os.get_env("ODINFER_NVRTC_PATH", context.temp_allocator)
		if from != "" {
			exclusive = true
		}
	}
	if from == "" {
		from = os.get_env("NVRTC_DIR", context.temp_allocator)
	}

	candidates := make([dynamic]string, context.temp_allocator)
	defer delete(candidates)

	when ODIN_OS == .Windows {
		if from != "" {
			append(&candidates, from)
			append(&candidates, join_path(from, "nvrtc64_120_0.dll"))
			if !prefer12 {
				append(&candidates, join_path(from, "nvrtc64_130_0.dll"))
			}
		}
		if !exclusive {
			append(&candidates, "nvrtc64_120_0.dll")
			if !prefer12 {
				append(&candidates, "nvrtc64_130_0.dll")
			}
		}
	} else when ODIN_OS == .Linux {
		if from != "" {
			if is_soname_path(from) {
				append(&candidates, from)
			} else {
				append_linux_names(&candidates, from, prefer12)
			}
		}
		if !exclusive {
			if prefer12 {
				// pip's 12.x wheel before the loader, which on a CUDA 13 box
				// resolves libnvrtc.so to NVRTC 13 (no compute_70).
				append_pip_nvrtc(&candidates, true)
				append(&candidates, "libnvrtc.so.12")
				append(&candidates, "libnvrtc.so")
			} else {
				append(&candidates, "libnvrtc.so.13")
				append(&candidates, "libnvrtc.so.12")
				append(&candidates, "libnvrtc.so")
				append_pip_nvrtc(&candidates, false)
			}
		}
	}

	err := try_nvrtc_candidates(candidates[:], prefer12)
	if err != OK && prefer12 {
		if from != "" {
			fmt.eprintfln("cuda: ODINFER_NVRTC_PATH/NVRTC_DIR %s has no usable NVRTC 12.x", from)
		}
		nvrtc12_hint()
	}
	return err
}

@(private)
options_need_nvrtc12 :: proc(options: []string) -> bool {
	for o in options {
		if strings.contains(o, "compute_70") || strings.contains(o, "compute_72") || strings.contains(o, "sm_70") || strings.contains(o, "sm_72") {
			return true
		}
	}
	return false
}

parent_dir :: proc(path: string) -> string {
	i := len(path) - 1
	for i >= 0 {
		if path[i] == '/' {
			return path[:i]
		}
		i -= 1
	}
	return ""
}

join_path :: proc(dir, name: string) -> string {
	if len(dir) == 0 {
		return name
	}
	if strings.has_suffix(dir, "/") {
		return strings.concatenate({dir, name}, context.temp_allocator)
	}
	return strings.concatenate({dir, "/", name}, context.temp_allocator)
}

error_string :: proc(err: Error) -> string {
	if err == OK {
		return "ok"
	}
	if err == ERR_LIBRARY {
		return "library not found"
	}
	if err == ERR_SYMBOL {
		return "missing symbol"
	}
	if err == ERR_COMPILE {
		return "nvrtc compile failed"
	}
	if drv.GetErrorString != nil {
		msg: cstring
		if drv.GetErrorString(err, &msg) == OK && msg != nil {
			return string(msg)
		}
	}
	return fmt.tprintf("cuda error %d", int(err))
}

// compile_ptx compiles CUDA C++ to PTX. `options` must include
// --gpu-architecture=compute_XX derived from the device. `log` is always
// populated when NVRTC produced a diagnostic (success or failure).
compile_ptx :: proc(source, name: string, options: []string, allocator := context.allocator) -> (ptx: string, log: string, err: Error) {
	want12 := options_need_nvrtc12(options)
	if e := load_nvrtc("", want12); e != OK {
		return "", "", e
	}
	ptx, log, err = compile_loaded(source, name, options, allocator)
	// compute_70 asked for 12.x already. A compile that still failed on
	// NVRTC 13 (arch was not in the option string, or 13 was already loaded)
	// gets one retry against 12.x.
	if err == OK || nvrtc_major() < 13 {
		return
	}
	fmt.eprintln("cuda: NVRTC compile failed on a 13.x library. CUDA 13 NVRTC cannot target compute_70. Retrying with NVRTC 12.x.")
	unload_nv()
	if e := load_nvrtc("", true); e != OK {
		return "", log, err
	}
	return compile_loaded(source, name, options, allocator)
}

@(private)
compile_loaded :: proc(source, name: string, options: []string, allocator := context.allocator) -> (ptx: string, log: string, err: Error) {
	csrc := strings.clone_to_cstring(source, context.temp_allocator)
	cname := strings.clone_to_cstring(name, context.temp_allocator)
	copts := make([]cstring, len(options), context.temp_allocator)
	for o, i in options {
		copts[i] = strings.clone_to_cstring(o, context.temp_allocator)
	}
	prog: rawptr
	if e := nv.CreateProgram(&prog, csrc, cname, 0, nil, nil); e != OK {
		return "", "", e
	}
	defer nv.DestroyProgram(&prog)

	cerr := nv.CompileProgram(prog, c.int(len(copts)), raw_data(copts))
	log_size: c.size_t
	if nv.GetProgramLogSize(prog, &log_size) == OK && log_size > 1 {
		buf := make([]u8, int(log_size), allocator)
		nv.GetProgramLog(prog, raw_data(buf))
		log = string(buf[:int(log_size) - 1])
	}
	if cerr != OK {
		return "", log, cerr
	}
	ptx_size: c.size_t
	if e := nv.GetPTXSize(prog, &ptx_size); e != OK {
		return "", log, e
	}
	buf := make([]u8, int(ptx_size), allocator)
	if e := nv.GetPTX(prog, raw_data(buf)); e != OK {
		delete(buf, allocator)
		return "", log, e
	}
	// Keep the trailing NUL inside the allocation; the returned string omits it
	// but module_load passes the raw bytes with an explicit terminator.
	ptx = string(buf[:int(ptx_size) - 1])
	return
}

module_load_ptx :: proc(ptx: string) -> (mod: Module, jit_log: string, err: Error) {
	// cuModuleLoadDataEx wants a NUL-terminated image.
	img := make([]u8, len(ptx) + 1, context.temp_allocator)
	copy(img, transmute([]u8)(ptx))
	logbuf := make([]u8, 1 << 16, context.temp_allocator)
	logsize := c.size_t(len(logbuf))
	opts := [2]c.int{JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES}
	vals := [2]rawptr{raw_data(logbuf), &logsize}
	err = drv.ModuleLoadDataEx(&mod, raw_data(img), 2, raw_data(opts[:]), raw_data(vals[:]))
	n := 0
	for n < len(logbuf) && logbuf[n] != 0 {
		n += 1
	}
	if n > 0 {
		jit_log = strings.clone(string(logbuf[:n]))
	}
	return
}

module_function :: proc(mod: Module, name: string) -> (fn: Function, err: Error) {
	cname := strings.clone_to_cstring(name, context.temp_allocator)
	err = drv.ModuleGetFunction(&fn, mod, cname)
	return
}

launch :: proc(fn: Function, gx, gy, gz, bx, by, bz: u32, shared: u32, args: []rawptr) -> Error {
	if gx == 0 || bx == 0 {
		return OK
	}
	return drv.LaunchKernel(fn, c.uint(gx), c.uint(gy), c.uint(gz), c.uint(bx), c.uint(by), c.uint(bz), c.uint(shared), nil, raw_data(args), nil)
}

alloc :: proc(nbytes: int) -> (Device_Ptr, Error) {
	p: Device_Ptr
	if nbytes <= 0 {
		return 0, OK
	}
	err := drv.MemAlloc_v2(&p, c.size_t(nbytes))
	return p, err
}

free :: proc(p: Device_Ptr) -> Error {
	if p == 0 {
		return OK
	}
	return drv.MemFree_v2(p)
}

upload :: proc(dst: Device_Ptr, src: rawptr, nbytes: int) -> Error {
	if nbytes <= 0 {
		return OK
	}
	return drv.MemcpyHtoD_v2(dst, src, c.size_t(nbytes))
}

download :: proc(dst: rawptr, src: Device_Ptr, nbytes: int) -> Error {
	if nbytes <= 0 {
		return OK
	}
	return drv.MemcpyDtoH_v2(dst, src, c.size_t(nbytes))
}

memset :: proc(p: Device_Ptr, value: u8, nbytes: int) -> Error {
	if nbytes <= 0 || p == 0 {
		return OK
	}
	return drv.MemsetD8_v2(p, c.uchar(value), c.size_t(nbytes))
}

sync :: proc() -> Error {
	return drv.CtxSynchronize()
}

ptr_add :: proc(p: Device_Ptr, bytes: int) -> Device_Ptr {
	return Device_Ptr(u64(p) + u64(bytes))
}
