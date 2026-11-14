/* Vulkan compute runtime for the Qwen3.5 backend.
   vulkan-1.dll / libvulkan.so.1 is loaded at runtime. Nothing links vulkan-1.lib,
   so a Windows build needs only the Odin compiler. The driver supplies the DLL. */

package vkr

import "core:dynlib"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"
import vk "vendor:vulkan"

Buffer :: struct {
	buf:      vk.Buffer,
	mem:      vk.DeviceMemory,
	ptr:      [^]byte,
	size:     int,
	coherent: bool,
}

Info :: struct {
	name:             string,
	api:              u32,
	uma:              bool,
	subgroup:         u32,
	subgroup_arith:   bool,
	intdot:           bool,
	coopmat:          bool,
	max_storage:      u32,
	max_wg:           u32,
	max_push:         u32,
	noncoherent_atom: int,
	device_local:     bool,
}

@(private)
State :: struct {
	lib:        dynlib.Library,
	instance:   vk.Instance,
	phys:       vk.PhysicalDevice,
	device:     vk.Device,
	queue:      vk.Queue,
	family:     u32,
	pool:       vk.CommandPool,
	cb:         vk.CommandBuffer,
	fence:      vk.Fence,
	sem:        [2]vk.Semaphore,
	submissions: int,
	signal_idx: int,
	recording:  bool,
	layout:     vk.PipelineLayout,
	dset_layout: vk.DescriptorSetLayout,
	dpool:      vk.DescriptorPool,
	sets:       [dynamic]vk.DescriptorSet,
	pipes:      [dynamic]vk.Pipeline,
	modules:    [dynamic]vk.ShaderModule,
	alive:      [dynamic]Buffer,
	info:       Info,
	ready:      bool,
	memprops:   vk.PhysicalDeviceMemoryProperties,
}

@(private)
g: State

device_info :: proc() -> Info {
	return g.info
}

n_sets :: proc() -> int {
	return len(g.sets)
}

boot :: proc() -> (Info, bool) {
	if g.ready {
		return g.info, true
	}
	if !load_lib() {
		return {}, false
	}
	if !create_instance() {
		shutdown()
		return {}, false
	}
	if !pick_and_create_device() {
		shutdown()
		return {}, false
	}
	if !create_commands() {
		shutdown()
		return {}, false
	}
	if !create_layout() {
		shutdown()
		return {}, false
	}
	g.ready = true
	fmt.eprintfln(
		"vulkan: device=%s api=%d.%d uma=%v subgroup=%d arith=%v intdot=%v coopmat=%v storage=%dMB max_wg=%d",
		g.info.name,
		(g.info.api >> 22) & 0x3ff,
		(g.info.api >> 12) & 0x3ff,
		g.info.uma,
		g.info.subgroup,
		g.info.subgroup_arith,
		g.info.intdot,
		g.info.coopmat,
		g.info.max_storage / (1024 * 1024),
		g.info.max_wg,
	)
	return g.info, true
}

shutdown :: proc() {
	if g.device != nil {
		if vk.DeviceWaitIdle != nil {
			_ = vk.DeviceWaitIdle(g.device)
		}
	}
	for p in g.pipes {
		if p != 0 && vk.DestroyPipeline != nil {
			vk.DestroyPipeline(g.device, p, nil)
		}
	}
	delete(g.pipes)
	for m in g.modules {
		if m != 0 && vk.DestroyShaderModule != nil {
			vk.DestroyShaderModule(g.device, m, nil)
		}
	}
	delete(g.modules)
	if g.dpool != 0 {
		vk.DestroyDescriptorPool(g.device, g.dpool, nil)
	}
	delete(g.sets)
	if g.layout != 0 {
		vk.DestroyPipelineLayout(g.device, g.layout, nil)
	}
	if g.dset_layout != 0 {
		vk.DestroyDescriptorSetLayout(g.device, g.dset_layout, nil)
	}
	// Copy the handles out: free_buffer swaps the alive list.
	alive := g.alive
	g.alive = nil
	for b in alive {
		bb := b
		free_buffer(&bb)
	}
	delete(alive)
	if g.fence != 0 {
		vk.DestroyFence(g.device, g.fence, nil)
	}
	if g.sem[0] != 0 {
		vk.DestroySemaphore(g.device, g.sem[0], nil)
	}
	if g.sem[1] != 0 {
		vk.DestroySemaphore(g.device, g.sem[1], nil)
	}
	if g.pool != 0 {
		vk.DestroyCommandPool(g.device, g.pool, nil)
	}
	if g.device != nil && vk.DestroyDevice != nil {
		vk.DestroyDevice(g.device, nil)
	}
	if g.instance != nil && vk.DestroyInstance != nil {
		vk.DestroyInstance(g.instance, nil)
	}
	if g.lib != nil {
		dynlib.unload_library(g.lib)
	}
	delete(g.info.name)
	g = {}
}

make_buffer :: proc(size: int) -> (Buffer, bool) {
	b: Buffer
	if size <= 0 || !g.ready {
		fmt.eprintln("vulkan: make_buffer bad size")
		return b, false
	}
	atom := max(g.info.noncoherent_atom, 1)
	bytes := align_up(size, atom)
	ci := vk.BufferCreateInfo {
		sType = .BUFFER_CREATE_INFO,
		size = vk.DeviceSize(bytes),
		usage = {.STORAGE_BUFFER},
		sharingMode = .EXCLUSIVE,
	}
	if !ok(vk.CreateBuffer(g.device, &ci, nil, &b.buf), "CreateBuffer") {
		return b, false
	}
	req: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(g.device, b.buf, &req)
	idx, coherent, devlocal, found := pick_memory(req.memoryTypeBits)
	if !found {
		fmt.eprintln("vulkan: no host-visible memory type for a storage buffer")
		vk.DestroyBuffer(g.device, b.buf, nil)
		b.buf = 0
		return b, false
	}
	alloc_size := int(req.size)
	if alloc_size < bytes {
		alloc_size = bytes
	}
	ai := vk.MemoryAllocateInfo {
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = req.size,
		memoryTypeIndex = idx,
	}
	if ai.allocationSize < vk.DeviceSize(bytes) {
		ai.allocationSize = vk.DeviceSize(bytes)
	}
	if !ok(vk.AllocateMemory(g.device, &ai, nil, &b.mem), "AllocateMemory") {
		vk.DestroyBuffer(g.device, b.buf, nil)
		b.buf = 0
		return b, false
	}
	if !ok(vk.BindBufferMemory(g.device, b.buf, b.mem, 0), "BindBufferMemory") {
		vk.FreeMemory(g.device, b.mem, nil)
		vk.DestroyBuffer(g.device, b.buf, nil)
		return {}, false
	}
	mapped: rawptr
	if !ok(vk.MapMemory(g.device, b.mem, 0, vk.DeviceSize(vk.WHOLE_SIZE), {}, &mapped), "MapMemory") {
		vk.FreeMemory(g.device, b.mem, nil)
		vk.DestroyBuffer(g.device, b.buf, nil)
		return {}, false
	}
	b.ptr = cast([^]byte)mapped
	b.size = int(ai.allocationSize)
	b.coherent = coherent
	if devlocal {
		g.info.device_local = true
		g.info.uma = true
	}
	mem.zero(mapped, b.size)
	flush_range(b, 0, b.size)
	append(&g.alive, b)
	return b, true
}

free_buffer :: proc(b: ^Buffer) {
	if b.buf == 0 {
		return
	}
	if b.ptr != nil {
		vk.UnmapMemory(g.device, b.mem)
	}
	vk.DestroyBuffer(g.device, b.buf, nil)
	vk.FreeMemory(g.device, b.mem, nil)
	for i := 0; i < len(g.alive); i += 1 {
		if g.alive[i].buf == b.buf {
			ordered_remove(&g.alive, i)
			break
		}
	}
	b^ = {}
}

// spec_local, when use_spec is set, overrides specialization constant 0
// (local_size_x_id). Cooperative-matrix uses that so one subgroup owns the tile.
make_pipeline :: proc(code: []u8, spec_local: u32, use_spec: bool) -> (vk.Pipeline, bool) {
	pipe: vk.Pipeline
	if len(code) == 0 || len(code) % 4 != 0 {
		fmt.eprintln("vulkan: SPIR-V is empty or not a multiple of 4")
		return pipe, false
	}
	words := make([]u32, len(code) / 4)
	defer delete(words)
	mem.copy(raw_data(words), raw_data(code), len(code))
	smi := vk.ShaderModuleCreateInfo {
		sType = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(code),
		pCode = cast(^u32)raw_data(words),
	}
	mod: vk.ShaderModule
	if !ok(vk.CreateShaderModule(g.device, &smi, nil, &mod), "CreateShaderModule") {
		return pipe, false
	}
	append(&g.modules, mod)
	local := spec_local
	entry := vk.SpecializationMapEntry {
		constantID = 0,
		offset = 0,
		size = 4,
	}
	spec := vk.SpecializationInfo {
		mapEntryCount = 1,
		pMapEntries = &entry,
		dataSize = 4,
		pData = &local,
	}
	stage := vk.PipelineShaderStageCreateInfo {
		sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage = {.COMPUTE},
		module = mod,
		pName = "main",
	}
	if use_spec {
		stage.pSpecializationInfo = &spec
	}
	ci := vk.ComputePipelineCreateInfo {
		sType = .COMPUTE_PIPELINE_CREATE_INFO,
		stage = stage,
		layout = g.layout,
	}
	if vk.CreateComputePipelines(g.device, 0, 1, &ci, nil, &pipe) != .SUCCESS {
		return 0, false
	}
	append(&g.pipes, pipe)
	return pipe, true
}

// One descriptor set per weight buffer. Binding 0 is that buffer, binding 1
// is the shared activation buffer. Sets are recorded into the command buffer,
// so a later update would not apply to an already-recorded dispatch.
bind_storage :: proc(weights: []Buffer, act: Buffer) -> bool {
	if len(weights) == 0 || act.buf == 0 {
		fmt.eprintln("vulkan: bind_storage needs a weight buffer and an activation buffer")
		return false
	}
	if g.dpool != 0 {
		vk.DestroyDescriptorPool(g.device, g.dpool, nil)
		g.dpool = 0
		clear(&g.sets)
	}
	ps := [1]vk.DescriptorPoolSize{{type = .STORAGE_BUFFER, descriptorCount = u32(len(weights) * 2)}}
	pi := vk.DescriptorPoolCreateInfo {
		sType = .DESCRIPTOR_POOL_CREATE_INFO,
		maxSets = u32(len(weights)),
		poolSizeCount = 1,
		pPoolSizes = raw_data(ps[:]),
	}
	if !ok(vk.CreateDescriptorPool(g.device, &pi, nil, &g.dpool), "CreateDescriptorPool") {
		return false
	}
	layouts := make([]vk.DescriptorSetLayout, len(weights))
	defer delete(layouts)
	for i in 0 ..< len(weights) {
		layouts[i] = g.dset_layout
	}
	ai := vk.DescriptorSetAllocateInfo {
		sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool = g.dpool,
		descriptorSetCount = u32(len(weights)),
		pSetLayouts = raw_data(layouts),
	}
	resize(&g.sets, len(weights))
	if !ok(vk.AllocateDescriptorSets(g.device, &ai, raw_data(g.sets)), "AllocateDescriptorSets") {
		return false
	}
	for i in 0 ..< len(weights) {
		wb := vk.DescriptorBufferInfo {
			buffer = weights[i].buf,
			offset = 0,
			range = vk.DeviceSize(vk.WHOLE_SIZE),
		}
		ab := vk.DescriptorBufferInfo {
			buffer = act.buf,
			offset = 0,
			range = vk.DeviceSize(vk.WHOLE_SIZE),
		}
		wr := [2]vk.WriteDescriptorSet {
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = g.sets[i],
				dstBinding = 0,
				descriptorCount = 1,
				descriptorType = .STORAGE_BUFFER,
				pBufferInfo = &wb,
			},
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = g.sets[i],
				dstBinding = 1,
				descriptorCount = 1,
				descriptorType = .STORAGE_BUFFER,
				pBufferInfo = &ab,
			},
		}
		vk.UpdateDescriptorSets(g.device, 2, raw_data(wr[:]), 0, nil)
	}
	return true
}

begin :: proc() -> bool {
	if !g.ready {
		return false
	}
	if g.recording {
		fmt.eprintln("vulkan: begin while a command buffer is open")
		return false
	}
	if !ok(vk.ResetCommandBuffer(g.cb, {}), "ResetCommandBuffer") {
		return false
	}
	bi := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}
	if !ok(vk.BeginCommandBuffer(g.cb, &bi), "BeginCommandBuffer") {
		return false
	}
	mb := vk.MemoryBarrier {
		sType = .MEMORY_BARRIER,
		srcAccessMask = {.HOST_WRITE},
		dstAccessMask = {.SHADER_READ},
	}
	vk.CmdPipelineBarrier(g.cb, {.HOST}, {.COMPUTE_SHADER}, {}, 1, &mb, 0, nil, 0, nil)
	g.recording = true
	return true
}

// Shader write of a previous dispatch becomes visible to the next read.
barrier :: proc() {
	if !g.recording {
		return
	}
	mb := vk.MemoryBarrier {
		sType = .MEMORY_BARRIER,
		srcAccessMask = {.SHADER_WRITE},
		dstAccessMask = {.SHADER_READ, .SHADER_WRITE},
	}
	vk.CmdPipelineBarrier(g.cb, {.COMPUTE_SHADER}, {.COMPUTE_SHADER}, {}, 1, &mb, 0, nil, 0, nil)
}

dispatch :: proc(pipe: vk.Pipeline, gx, gy, gz: u32, push: rawptr, push_n: int, set_i: int) {
	if !g.recording || pipe == 0 || gx == 0 || gy == 0 || gz == 0 {
		return
	}
	if set_i < 0 || set_i >= len(g.sets) {
		fmt.eprintfln("vulkan: descriptor set %d out of range (%d)", set_i, len(g.sets))
		return
	}
	vk.CmdBindPipeline(g.cb, .COMPUTE, pipe)
	set := g.sets[set_i]
	vk.CmdBindDescriptorSets(g.cb, .COMPUTE, g.layout, 0, 1, &set, 0, nil)
	if push != nil && push_n > 0 {
		vk.CmdPushConstants(g.cb, g.layout, {.COMPUTE}, 0, u32(push_n), push)
	}
	vk.CmdDispatch(g.cb, gx, gy, gz)
}

submit_wait :: proc() -> bool {
	if !g.recording {
		return true
	}
	mb := vk.MemoryBarrier {
		sType = .MEMORY_BARRIER,
		srcAccessMask = {.SHADER_WRITE},
		dstAccessMask = {.HOST_READ, .SHADER_READ},
	}
	vk.CmdPipelineBarrier(g.cb, {.COMPUTE_SHADER}, {.HOST, .COMPUTE_SHADER}, {}, 1, &mb, 0, nil, 0, nil)
	if !ok(vk.EndCommandBuffer(g.cb), "EndCommandBuffer") {
		g.recording = false
		return false
	}
	g.recording = false
	if !ok(vk.ResetFences(g.device, 1, &g.fence), "ResetFences") {
		return false
	}
	wait_stage := vk.PipelineStageFlags{.COMPUTE_SHADER}
	si := vk.SubmitInfo {
		sType = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers = &g.cb,
		pSignalSemaphores = &g.sem[g.signal_idx],
		signalSemaphoreCount = 1,
	}
	if g.submissions > 0 {
		prev := 1 - g.signal_idx
		si.waitSemaphoreCount = 1
		si.pWaitSemaphores = &g.sem[prev]
		si.pWaitDstStageMask = &wait_stage
	}
	if !ok(vk.QueueSubmit(g.queue, 1, &si, g.fence), "QueueSubmit") {
		return false
	}
	if !ok(vk.WaitForFences(g.device, 1, &g.fence, true, ~u64(0)), "WaitForFences") {
		return false
	}
	g.submissions += 1
	g.signal_idx = 1 - g.signal_idx
	return true
}

idle :: proc() {
	if g.queue != nil && vk.QueueWaitIdle != nil {
		_ = vk.QueueWaitIdle(g.queue)
	}
}

flush_range :: proc(b: Buffer, off, n: int) {
	sync_range(b, off, n, true)
}

invalidate_range :: proc(b: Buffer, off, n: int) {
	sync_range(b, off, n, false)
}

@(private)
sync_range :: proc(b: Buffer, off, n: int, flush: bool) {
	if b.coherent || b.mem == 0 || n <= 0 || g.device == nil {
		return
	}
	atom := max(g.info.noncoherent_atom, 1)
	start := off & ~(atom - 1)
	end := align_up(off + n, atom)
	if start < 0 {
		start = 0
	}
	if end > b.size {
		end = b.size
	}
	if end <= start {
		return
	}
	rng := vk.MappedMemoryRange {
		sType = .MAPPED_MEMORY_RANGE,
		memory = b.mem,
		offset = vk.DeviceSize(start),
		size = vk.DeviceSize(end - start),
	}
	if flush {
		_ = vk.FlushMappedMemoryRanges(g.device, 1, &rng)
	} else {
		_ = vk.InvalidateMappedMemoryRanges(g.device, 1, &rng)
	}
}

@(private)
align_up :: proc(n, a: int) -> int {
	if a <= 1 {
		return n
	}
	return (n + a - 1) & ~(a - 1)
}

@(private)
ok :: proc(r: vk.Result, what: string) -> bool {
	if r == .SUCCESS {
		return true
	}
	fmt.eprintfln("vulkan: %s failed (%v)", what, r)
	return false
}

@(private)
load_lib :: proc() -> bool {
	loaded := false
	when ODIN_OS == .Windows {
		g.lib, loaded = dynlib.load_library("vulkan-1.dll")
	} else when ODIN_OS == .Darwin {
		g.lib, loaded = dynlib.load_library("libvulkan.1.dylib")
		if !loaded {
			g.lib, loaded = dynlib.load_library("libvulkan.dylib")
		}
	} else {
		g.lib, loaded = dynlib.load_library("libvulkan.so.1")
		if !loaded {
			g.lib, loaded = dynlib.load_library("libvulkan.so")
		}
	}
	if !loaded {
		fmt.eprintln("vulkan: could not load vulkan-1.dll / libvulkan.so.1 (the driver runtime, not the SDK)")
		return false
	}
	sym, found := dynlib.symbol_address(g.lib, "vkGetInstanceProcAddr")
	if !found || sym == nil {
		fmt.eprintln("vulkan: vkGetInstanceProcAddr missing")
		return false
	}
	vk.load_proc_addresses_global(sym)
	if vk.CreateInstance == nil {
		fmt.eprintln("vulkan: vkCreateInstance missing")
		return false
	}
	return true
}

@(private)
create_instance :: proc() -> bool {
	apis := [2]u32{vk.API_VERSION_1_3, vk.API_VERSION_1_2}
	for api in apis {
		app := vk.ApplicationInfo {
			sType = .APPLICATION_INFO,
			pApplicationName = "odinfer",
			apiVersion = api,
		}
		ci := vk.InstanceCreateInfo {
			sType = .INSTANCE_CREATE_INFO,
			pApplicationInfo = &app,
		}
		if vk.CreateInstance(&ci, nil, &g.instance) == .SUCCESS {
			vk.load_proc_addresses_instance(g.instance)
			return true
		}
		g.instance = nil
	}
	fmt.eprintln("vulkan: vkCreateInstance failed for 1.3 and 1.2")
	return false
}

@(private)
pick_and_create_device :: proc() -> bool {
	n: u32
	if !ok(vk.EnumeratePhysicalDevices(g.instance, &n, nil), "EnumeratePhysicalDevices") || n == 0 {
		fmt.eprintln("vulkan: no physical devices")
		return false
	}
	devs := make([]vk.PhysicalDevice, n)
	defer delete(devs)
	if !ok(vk.EnumeratePhysicalDevices(g.instance, &n, raw_data(devs)), "EnumeratePhysicalDevices") {
		return false
	}
	names := make([]string, n)
	defer {
		for s in names {
			delete(s)
		}
		delete(names)
	}
	for i in 0 ..< int(n) {
		pr: vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(devs[i], &pr)
		names[i] = cstr_bytes(pr.deviceName[:])
		fmt.eprintfln("vulkan: [%d] %s type=%v api=%d.%d", i, names[i], pr.deviceType, (pr.apiVersion >> 22) & 0x3ff, (pr.apiVersion >> 12) & 0x3ff)
	}
	idx := 0
	if s := env_str("ODINFER_VULKAN_DEVICE"); s != "" {
		if v, parsed := parse_int(s); parsed && v >= 0 && v < int(n) {
			idx = v
		} else {
			fmt.eprintfln("vulkan: ODINFER_VULKAN_DEVICE=%s is not a device index; using the default", s)
		}
	} else {
		for i in 0 ..< int(n) {
			pr: vk.PhysicalDeviceProperties
			vk.GetPhysicalDeviceProperties(devs[i], &pr)
			low := strings.to_lower(names[i], context.temp_allocator)
			if pr.deviceType != .CPU && !strings.contains(low, "llvmpipe") {
				idx = i
				break
			}
		}
	}
	g.phys = devs[idx]
	props: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(g.phys, &props)
	if props.apiVersion < vk.API_VERSION_1_2 {
		fmt.eprintfln("vulkan: %s exposes API below 1.2", names[idx])
		return false
	}
	g.info.name = strings.clone(names[idx])
	g.info.api = props.apiVersion
	g.info.max_storage = props.limits.maxStorageBufferRange
	g.info.max_wg = props.limits.maxComputeWorkGroupCount[0]
	if g.info.max_wg == 0 {
		g.info.max_wg = 65535
	}
	g.info.max_push = props.limits.maxPushConstantsSize
	g.info.noncoherent_atom = int(props.limits.nonCoherentAtomSize)
	if g.info.noncoherent_atom <= 0 {
		g.info.noncoherent_atom = 1
	}
	if g.info.max_push < 128 {
		fmt.eprintfln("vulkan: maxPushConstantsSize %d < 128", g.info.max_push)
		return false
	}

	sub: vk.PhysicalDeviceSubgroupProperties
	sub.sType = .PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
	props2: vk.PhysicalDeviceProperties2
	props2.sType = .PHYSICAL_DEVICE_PROPERTIES_2
	props2.pNext = &sub
	vk.GetPhysicalDeviceProperties2(g.phys, &props2)
	g.info.subgroup = sub.subgroupSize
	g.info.subgroup_arith = .ARITHMETIC in sub.supportedOperations && .COMPUTE in sub.supportedStages && g.info.subgroup > 0 && 64 % g.info.subgroup == 0

	vk11: vk.PhysicalDeviceVulkan11Features
	vk11.sType = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
	vk12: vk.PhysicalDeviceVulkan12Features
	vk12.sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
	vk11.pNext = &vk12
	vk13: vk.PhysicalDeviceVulkan13Features
	vk13.sType = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
	coop: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR
	coop.sType = .PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
	api13 := props.apiVersion >= vk.API_VERSION_1_3
	if api13 {
		vk12.pNext = &vk13
	}
	has_coop_ext := device_has_ext("VK_KHR_cooperative_matrix")
	if has_coop_ext {
		if api13 {
			vk13.pNext = &coop
		} else {
			vk12.pNext = &coop
		}
	}
	feats2: vk.PhysicalDeviceFeatures2
	feats2.sType = .PHYSICAL_DEVICE_FEATURES_2
	feats2.pNext = &vk11
	vk.GetPhysicalDeviceFeatures2(g.phys, &feats2)

	want_intdot := api13 && vk13.shaderIntegerDotProduct == true
	want_coop := has_coop_ext && coop.cooperativeMatrix == true && vk11.storageBuffer16BitAccess == true && vk12.shaderFloat16 == true
	g.info.intdot = want_intdot
	g.info.coopmat = want_coop

	// Fresh chain so device creation enables only the features this backend uses.
	c11: vk.PhysicalDeviceVulkan11Features
	c11.sType = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
	c12: vk.PhysicalDeviceVulkan12Features
	c12.sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
	c11.pNext = &c12
	c13: vk.PhysicalDeviceVulkan13Features
	c13.sType = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
	ccoop: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR
	ccoop.sType = .PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
	if want_coop {
		c11.storageBuffer16BitAccess = true
		c12.shaderFloat16 = true
		ccoop.cooperativeMatrix = true
	}
	if want_intdot {
		c12.pNext = &c13
		c13.shaderIntegerDotProduct = true
		if want_coop {
			c13.pNext = &ccoop
		}
	} else if want_coop {
		c12.pNext = &ccoop
	}

	qf_n: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(g.phys, &qf_n, nil)
	qfs := make([]vk.QueueFamilyProperties, qf_n)
	defer delete(qfs)
	vk.GetPhysicalDeviceQueueFamilyProperties(g.phys, &qf_n, raw_data(qfs))
	family := -1
	for i in 0 ..< int(qf_n) {
		if .COMPUTE in qfs[i].queueFlags && qfs[i].queueCount > 0 {
			family = i
			break
		}
	}
	if family < 0 {
		fmt.eprintln("vulkan: no compute queue")
		return false
	}
	g.family = u32(family)
	prio := f32(1)
	qi := vk.DeviceQueueCreateInfo {
		sType = .DEVICE_QUEUE_CREATE_INFO,
		queueFamilyIndex = g.family,
		queueCount = 1,
		pQueuePriorities = &prio,
	}
	ext_lit: string = vk.KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
	ext_name := cstring(raw_data(ext_lit))
	dci := vk.DeviceCreateInfo {
		sType = .DEVICE_CREATE_INFO,
		pNext = &c11,
		queueCreateInfoCount = 1,
		pQueueCreateInfos = &qi,
	}
	if want_coop {
		dci.enabledExtensionCount = 1
		dci.ppEnabledExtensionNames = &ext_name
	}
	if !ok(vk.CreateDevice(g.phys, &dci, nil, &g.device), "CreateDevice") {
		// 1.3 feature chain can be refused. Retry without integer-dot and coopmat.
		g.info.intdot = false
		g.info.coopmat = false
		c11b: vk.PhysicalDeviceVulkan11Features
		c11b.sType = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
		dci.pNext = &c11b
		dci.enabledExtensionCount = 0
		dci.ppEnabledExtensionNames = nil
		if !ok(vk.CreateDevice(g.phys, &dci, nil, &g.device), "CreateDevice (fallback)") {
			return false
		}
	}
	vk.load_proc_addresses_device(g.device)
	vk.GetDeviceQueue(g.device, g.family, 0, &g.queue)
	vk.GetPhysicalDeviceMemoryProperties(g.phys, &g.memprops)
	return g.queue != nil
}

@(private)
device_has_ext :: proc(name: string) -> bool {
	n: u32
	if vk.EnumerateDeviceExtensionProperties(g.phys, nil, &n, nil) != .SUCCESS || n == 0 {
		return false
	}
	exts := make([]vk.ExtensionProperties, n)
	defer delete(exts)
	if vk.EnumerateDeviceExtensionProperties(g.phys, nil, &n, raw_data(exts)) != .SUCCESS {
		return false
	}
	for i in 0 ..< len(exts) {
		name_buf := exts[i].extensionName
		if cstr_bytes_tmp(name_buf[:]) == name {
			return true
		}
	}
	return false
}

@(private)
create_commands :: proc() -> bool {
	ci := vk.CommandPoolCreateInfo {
		sType = .COMMAND_POOL_CREATE_INFO,
		flags = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = g.family,
	}
	if !ok(vk.CreateCommandPool(g.device, &ci, nil, &g.pool), "CreateCommandPool") {
		return false
	}
	ai := vk.CommandBufferAllocateInfo {
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool = g.pool,
		level = .PRIMARY,
		commandBufferCount = 1,
	}
	if !ok(vk.AllocateCommandBuffers(g.device, &ai, &g.cb), "AllocateCommandBuffers") {
		return false
	}
	fi := vk.FenceCreateInfo{sType = .FENCE_CREATE_INFO}
	if !ok(vk.CreateFence(g.device, &fi, nil, &g.fence), "CreateFence") {
		return false
	}
	si := vk.SemaphoreCreateInfo{sType = .SEMAPHORE_CREATE_INFO}
	if !ok(vk.CreateSemaphore(g.device, &si, nil, &g.sem[0]), "CreateSemaphore") {
		return false
	}
	if !ok(vk.CreateSemaphore(g.device, &si, nil, &g.sem[1]), "CreateSemaphore") {
		return false
	}
	return true
}

@(private)
create_layout :: proc() -> bool {
	binds := [2]vk.DescriptorSetLayoutBinding {
		{binding = 0, descriptorType = .STORAGE_BUFFER, descriptorCount = 1, stageFlags = {.COMPUTE}},
		{binding = 1, descriptorType = .STORAGE_BUFFER, descriptorCount = 1, stageFlags = {.COMPUTE}},
	}
	li := vk.DescriptorSetLayoutCreateInfo {
		sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		bindingCount = 2,
		pBindings = raw_data(binds[:]),
	}
	if !ok(vk.CreateDescriptorSetLayout(g.device, &li, nil, &g.dset_layout), "CreateDescriptorSetLayout") {
		return false
	}
	pc := vk.PushConstantRange {
		stageFlags = {.COMPUTE},
		offset = 0,
		size = 128,
	}
	pi := vk.PipelineLayoutCreateInfo {
		sType = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount = 1,
		pSetLayouts = &g.dset_layout,
		pushConstantRangeCount = 1,
		pPushConstantRanges = &pc,
	}
	return ok(vk.CreatePipelineLayout(g.device, &pi, nil, &g.layout), "CreatePipelineLayout")
}

@(private)
pick_memory :: proc(bits: u32) -> (idx: u32, coherent, devlocal, found: bool) {
	// Prefer the UMA heap: device-local and host-visible, coherent if the
	// driver offers it. Lavapipe is host-visible and usually not device-local.
	best := -1
	best_score := -1
	for i in 0 ..< int(g.memprops.memoryTypeCount) {
		if bits & (u32(1) << u32(i)) == 0 {
			continue
		}
		f := g.memprops.memoryTypes[i].propertyFlags
		if .HOST_VISIBLE not_in f {
			continue
		}
		score := 1
		if .HOST_COHERENT in f {
			score += 2
		}
		if .DEVICE_LOCAL in f {
			score += 4
		}
		if score > best_score {
			best_score = score
			best = i
		}
	}
	if best < 0 {
		return 0, false, false, false
	}
	f := g.memprops.memoryTypes[best].propertyFlags
	return u32(best), .HOST_COHERENT in f, .DEVICE_LOCAL in f, true
}

@(private)
cstr_bytes :: proc(b: []byte) -> string {
	n := 0
	for n < len(b) && b[n] != 0 {
		n += 1
	}
	return strings.clone(string(b[:n]))
}

@(private)
cstr_bytes_tmp :: proc(b: []byte) -> string {
	n := 0
	for n < len(b) && b[n] != 0 {
		n += 1
	}
	return string(b[:n])
}

@(private)
env_str :: proc(key: string) -> string {
	return os.get_env(key, context.temp_allocator)
}

@(private)
parse_int :: proc(s: string) -> (int, bool) {
	return strconv.parse_int(s)
}
