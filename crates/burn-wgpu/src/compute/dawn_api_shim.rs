#![allow(missing_docs)]

use crate::compute::{
    dawn_native_bindings::*, webgpu_api::*, WgpuServer, WgpuStorage,
};
use crate::{create_client, GraphicsApi, RuntimeOptions, WgpuDevice};
use alloc::sync::Arc;
use burn_compute::{
    channel::MutexComputeChannel, client::ComputeClient, memory_management::SimpleMemoryManagement,
    ComputeRuntime,
};
use burn_jit::compute::WorkGroup;
use std::num::NonZeroU64;

#[derive(Debug)]
pub struct DawnApi {}

#[derive(Debug)]
pub struct DawnAdapter {
    adapter: WGPUAdapter,
}

impl Adapter<DawnAdapterInfo> for DawnAdapter {
    fn get_info(&self) -> DawnAdapterInfo {
        let mut adapter_info = WGPUAdapterProperties {
            nextInChain: std::ptr::null_mut::<WGPUChainedStructOut>(),
            vendorID: 0,
            vendorName: std::ptr::null(),
            architecture: std::ptr::null(),
            deviceID: 0,
            name: std::ptr::null(),
            driverDescription: std::ptr::null(),
            adapterType: 0,
            backendType: 0,
            compatibilityMode: 0,
        };
        unsafe {
            wgpuAdapterGetProperties(self.adapter, &mut adapter_info);
        }
        DawnAdapterInfo { adapter_info }
    }
}

#[derive(Debug)]
pub struct DawnAdapterInfo {
    adapter_info: WGPUAdapterProperties,
}

impl AdapterInfo<DawnBackend> for DawnAdapterInfo {
    fn backend(&self) -> DawnBackend {
        DawnBackend::from_u32(self.adapter_info.backendType)
    }

    fn device(&self) -> DeviceId {
        self.adapter_info.deviceID
    }
}

#[derive(Debug)]
pub enum DawnBackend {
    Undefined = WGPUBackendType_WGPUBackendType_Undefined as isize,
    Null = WGPUBackendType_WGPUBackendType_Null as isize,
    WebGPU = WGPUBackendType_WGPUBackendType_WebGPU as isize,
    D3D11 = WGPUBackendType_WGPUBackendType_D3D11 as isize,
    D3D12 = WGPUBackendType_WGPUBackendType_D3D12 as isize,
    Metal = WGPUBackendType_WGPUBackendType_Metal as isize,
    Vulkan = WGPUBackendType_WGPUBackendType_Vulkan as isize,
    OpenGL = WGPUBackendType_WGPUBackendType_OpenGL as isize,
    OpenGLES = WGPUBackendType_WGPUBackendType_OpenGLES as isize,
}

impl DawnBackend {
    #[allow(non_upper_case_globals)]
    fn from_u32(val: u32) -> DawnBackend {
        match val {
            WGPUBackendType_WGPUBackendType_Undefined => DawnBackend::Undefined,
            WGPUBackendType_WGPUBackendType_Null => DawnBackend::Null,
            WGPUBackendType_WGPUBackendType_WebGPU => DawnBackend::WebGPU,
            WGPUBackendType_WGPUBackendType_D3D11 => DawnBackend::D3D11,
            WGPUBackendType_WGPUBackendType_D3D12 => DawnBackend::D3D12,
            WGPUBackendType_WGPUBackendType_Metal => DawnBackend::Metal,
            WGPUBackendType_WGPUBackendType_Vulkan => DawnBackend::Vulkan,
            WGPUBackendType_WGPUBackendType_OpenGL => DawnBackend::OpenGL,
            WGPUBackendType_WGPUBackendType_OpenGLES => DawnBackend::OpenGLES,
            _ => panic!("Unknown Dawn backend type: {}", val),
        }
    }
}

impl core::convert::AsRef<str> for DawnBackend {
    fn as_ref(&self) -> &'static str {
        match self {
            DawnBackend::Undefined => "undefined",
            DawnBackend::Null => "null",
            DawnBackend::WebGPU => "webgpu",
            DawnBackend::D3D11 => "dx11",
            DawnBackend::D3D12 => "dx12",
            DawnBackend::Metal => "metal",
            DawnBackend::Vulkan => "vulkan",
            DawnBackend::OpenGL => "opengl",
            DawnBackend::OpenGLES => "opengles",
        }
    }
}

#[derive(Debug)]
pub struct DawnBindGroup {
    bind_group: WGPUBindGroup,
}
unsafe impl Send for DawnBindGroup {}
impl BindGroup for DawnBindGroup {}

#[derive(Debug)]
pub struct DawnBindGroupLayout {
    layout: WGPUBindGroupLayout,
}
impl BindGroupLayout for DawnBindGroupLayout {}

#[derive(Debug)]
pub struct DawnBuffer {
    buffer: WGPUBuffer,
    size: u64,
}
unsafe impl Send for DawnBuffer {}
unsafe impl Sync for DawnBuffer {}

impl Buffer<DawnBuffer, DawnDevice> for DawnBuffer {
    fn as_entire_buffer_binding(&self) -> BufferBinding<'_, DawnBuffer> {
        BufferBinding {
            buffer: self,
            offset: 0,
            size: Some(NonZeroU64::new((*self).size).unwrap()),
        }
    }

    fn destroy(&self) {
        unsafe {
            wgpuBufferDestroy((*self).buffer.into());
        }
    }

    async fn read(&self, device: &DawnDevice) -> Vec<u8> {
        let mut read_data = BufferReadData {
            read_done: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        };
        unsafe {
            let data_ptr = std::mem::transmute::<*mut BufferReadData, *mut std::os::raw::c_void>(
                std::ptr::addr_of_mut!(read_data),
            );
            let mut sz = (*self).size;
            if sz % 4 != 0 {
                sz += 2;
            }
            wgpuBufferMapAsync(
                (*self).buffer.into(),
                WGPUMapMode_WGPUMapMode_Read,
                0,
                sz as usize,
                Some(buffer_reader_cb),
                data_ptr,
            );

            let mut read_done = read_data.read_done.lock().unwrap();
            let should_process = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
            let spt = should_process.clone();
            let instance = DawnInstance {
                instance: wgpuAdapterGetInstance(wgpuDeviceGetAdapter((*device).device)),
            };
            let handle = std::thread::spawn(move || {
                let inst = instance;
                while spt.load(std::sync::atomic::Ordering::Relaxed) {
                    wgpuInstanceProcessEvents(inst.instance);
                    std::thread::sleep(std::time::Duration::from_micros(10));
                }
            });
            while !*read_done {
                let res = read_data
                    .cv
                    .wait_timeout(read_done, std::time::Duration::from_micros(100))
                    .unwrap();
                read_done = res.0;
            }
            should_process.store(false, std::sync::atomic::Ordering::Relaxed);
            handle.join().unwrap();

            let mpd_rng =
                wgpuBufferGetConstMappedRange((*self).buffer.into(), 0, (*self).size as usize);
            let slice = std::slice::from_raw_parts(mpd_rng as *const u8, (*self).size as usize);
            slice.to_vec()
        }
    }

    fn size(&self) -> u64 {
        (*self).size
    }
}

pub type DawnBufferUsages = u32;
pub const MAP_READ: DawnBufferUsages = WGPUBufferUsage_WGPUBufferUsage_MapRead;
pub const COPY_SRC: DawnBufferUsages = WGPUBufferUsage_WGPUBufferUsage_CopySrc;
pub const COPY_DST: DawnBufferUsages = WGPUBufferUsage_WGPUBufferUsage_CopyDst;
pub const STORAGE: DawnBufferUsages = WGPUBufferUsage_WGPUBufferUsage_Storage;

#[derive(Debug)]
pub struct DawnCommandBuffer {
    buffer: WGPUCommandBuffer,
}
impl CommandBuffer for DawnCommandBuffer {}

#[derive(Debug)]
pub struct DawnCommandEncoder {
    encoder: WGPUCommandEncoder,
}
unsafe impl Send for DawnCommandEncoder {}
unsafe impl Sync for DawnCommandEncoder {}

impl CommandEncoder<DawnBindGroup, DawnBuffer, DawnCommandBuffer, DawnComputePipeline>
    for DawnCommandEncoder
{
    fn dispatch_compute_pass(
        &mut self,
        desc: &ComputePassDescriptor,
        pipeline: Arc<DawnComputePipeline>,
        bind_group: DawnBindGroup,
        work_group: WorkGroup,
    ) {
        let label = match desc.label {
            Some(name) => name,
            None => "",
        };
        let pass_desc = WGPUComputePassDescriptor {
            nextInChain: std::ptr::null(),
            label: std::ffi::CString::new(label).unwrap().into_raw(),
            timestampWrites: std::ptr::null(),
        };
        let pass: WGPUComputePassEncoder;
        unsafe {
            pass = wgpuCommandEncoderBeginComputePass(self.encoder.into(), &pass_desc);
        }
        unsafe {
            wgpuComputePassEncoderSetPipeline(pass, pipeline.pipeline.into());
            wgpuComputePassEncoderSetBindGroup(
                pass,
                0,
                bind_group.bind_group.into(),
                0,
                (&[]).as_ptr(),
            );
            wgpuComputePassEncoderDispatchWorkgroups(
                pass,
                work_group.x,
                work_group.y,
                work_group.z,
            );
        }
        unsafe {
            wgpuComputePassEncoderEnd(pass);
        }
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: &DawnBuffer,
        src_offset: u64,
        dest: &DawnBuffer,
        dest_offset: u64,
        size: u64,
    ) {
        unsafe {
            wgpuCommandEncoderCopyBufferToBuffer(
                (*self).encoder.into(),
                (*src).buffer.into(),
                src_offset,
                (*dest).buffer.into(),
                dest_offset,
                size,
            );
        }
    }

    fn finish(self) -> DawnCommandBuffer {
        let cmd_buf_desc = WGPUCommandBufferDescriptor {
            nextInChain: std::ptr::null(),
            label: std::ptr::null(),
        };
        let cmd_buf: WGPUCommandBuffer;
        unsafe {
            cmd_buf = wgpuCommandEncoderFinish(self.encoder.into(), &cmd_buf_desc);
        }
        DawnCommandBuffer { buffer: cmd_buf }
    }
}

#[derive(Debug)]
pub struct DawnComputePipeline {
    pipeline: WGPUComputePipeline,
}
unsafe impl Send for DawnComputePipeline {}
unsafe impl Sync for DawnComputePipeline {}

impl ComputePipeline<DawnBindGroupLayout> for DawnComputePipeline {
    fn get_bind_group_layout(&self, id: u32) -> DawnBindGroupLayout {
        let layout: WGPUBindGroupLayout;
        unsafe {
            layout = wgpuComputePipelineGetBindGroupLayout((*self).pipeline.into(), id);
        }
        DawnBindGroupLayout { layout: layout }
    }
}

#[derive(Debug)]
pub struct DawnDevice {
    device: WGPUDevice,
}
unsafe impl Send for DawnDevice {}
unsafe impl Sync for DawnDevice {}

impl
    Device<
        DawnBindGroup,
        DawnBindGroupLayout,
        DawnBuffer,
        DawnCommandEncoder,
        DawnComputePipeline,
        DawnPipelineLayout,
        DawnShaderModule,
    > for DawnDevice
{
    fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor<'_, DawnBindGroupLayout, DawnBuffer>,
    ) -> DawnBindGroup {
        let entries = (*desc)
            .entries
            .iter()
            .map(|entry| {
                let resource = match &entry.resource {
                    BindingResource::Buffer(res) => res,
                };
                WGPUBindGroupEntry {
                    nextInChain: std::ptr::null(),
                    binding: entry.binding,
                    buffer: resource.buffer.buffer.into(),
                    offset: resource.offset,
                    size: resource.size.unwrap().get(),
                    sampler: std::ptr::null_mut(),
                    textureView: std::ptr::null_mut(),
                }
            })
            .collect::<Vec<_>>();
        let label = match desc.label {
            None => std::ptr::null(),
            Some(name) => std::ffi::CString::new(name).unwrap().into_raw(),
        };
        let bg_desc = WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null(),
            label: label,
            layout: (*desc).layout.layout,
            entryCount: entries.len(),
            entries: entries.as_ptr(),
        };
        let bind_group: WGPUBindGroup;
        unsafe {
            bind_group = wgpuDeviceCreateBindGroup((*self).device.into(), &bg_desc);
        }
        DawnBindGroup {
            bind_group: bind_group,
        }
    }

    fn create_buffer(&self, desc: &BufferDescriptor) -> DawnBuffer {
        let label = match desc.label {
            None => std::ptr::null(),
            Some(name) => std::ffi::CString::new(name).unwrap().into_raw(),
        };
        let buf_desc = WGPUBufferDescriptor {
            nextInChain: std::ptr::null(),
            label: label,
            usage: (*desc).usage,
            size: (*desc).size,
            mappedAtCreation: (*desc).mapped_at_creation as u32,
        };
        let buffer: WGPUBuffer;
        unsafe {
            buffer = wgpuDeviceCreateBuffer((*self).device.into(), &buf_desc);
        }
        DawnBuffer {
            buffer: buffer,
            size: (*desc).size,
        }
    }

    fn create_buffer_init(&self, desc: &BufferInitDescriptor) -> DawnBuffer {
        let label = match desc.label {
            None => std::ptr::null(),
            Some(name) => std::ffi::CString::new(name).unwrap().into_raw(),
        };
        let buf_desc = WGPUBufferDescriptor {
            nextInChain: std::ptr::null(),
            label: label,
            usage: (*desc).usage,
            size: (*desc).contents.len() as u64,
            mappedAtCreation: 1,
        };
        let buffer: WGPUBuffer;
        unsafe {
            buffer = wgpuDeviceCreateBuffer((*self).device.into(), &buf_desc);
            let data = wgpuBufferGetMappedRange(buffer, 0, (*desc).contents.len());
            let src_ptr = &(*desc).contents[0] as *const u8;
            std::ptr::copy_nonoverlapping(src_ptr, data as *mut u8, (*desc).contents.len());
            wgpuBufferUnmap(buffer);
        }
        DawnBuffer {
            buffer: buffer,
            size: (*desc).contents.len() as u64,
        }
    }

    fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> DawnCommandEncoder {
        let label = match desc.label {
            None => std::ptr::null(),
            Some(name) => std::ffi::CString::new(name).unwrap().into_raw(),
        };
        let encoder_desc = WGPUCommandEncoderDescriptor {
            nextInChain: std::ptr::null(),
            label: label,
        };
        let encoder: WGPUCommandEncoder;
        unsafe {
            encoder = wgpuDeviceCreateCommandEncoder((*self).device.into(), &encoder_desc);
        }
        DawnCommandEncoder { encoder: encoder }
    }

    fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor<DawnPipelineLayout, DawnShaderModule>,
    ) -> DawnComputePipeline {
        let label = match desc.label {
            None => std::ptr::null(),
            Some(name) => std::ffi::CString::new(name).unwrap().into_raw(),
        };
        let layout = match desc.layout {
            None => std::ptr::null_mut(),
            Some(layout) => layout.layout,
        };
        let pip_desc = WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null(),
            label: label,
            layout: layout,
            compute: WGPUProgrammableStageDescriptor {
                nextInChain: std::ptr::null(),
                module: (*(*desc).module).module,
                entryPoint: std::ffi::CString::new((*desc).entry_point)
                    .unwrap()
                    .into_raw(),
                constantCount: 0,
                constants: std::ptr::null(),
            },
        };
        let pipeline: WGPUComputePipeline;
        unsafe {
            pipeline = wgpuDeviceCreateComputePipeline((*self).device.into(), &pip_desc);
        }
        DawnComputePipeline { pipeline: pipeline }
    }

    fn create_shader_module(&self, desc: &ShaderModuleDescriptor) -> DawnShaderModule {
        let label = match desc.label {
            None => std::ptr::null(),
            Some(name) => std::ffi::CString::new(name).unwrap().into_raw(),
        };
        let src = match &desc.source {
            ShaderSource::Wgsl(source) => source.to_string(),
        };
        let wgsl_desc = WGPUShaderModuleWGSLDescriptor {
            chain: WGPUChainedStruct {
                next: std::ptr::null(),
                sType: WGPUSType_WGPUSType_ShaderModuleWGSLDescriptor,
            },
            code: std::ffi::CString::new(src).unwrap().into_raw(),
        };
        let module: WGPUShaderModule;
        unsafe {
            let sh_desc = WGPUShaderModuleDescriptor {
                nextInChain: std::mem::transmute::<
                    *const WGPUShaderModuleWGSLDescriptor,
                    *const WGPUChainedStruct,
                >(&wgsl_desc),
                label: label,
            };
            module = wgpuDeviceCreateShaderModule((*self).device.into(), &sh_desc);
        }
        DawnShaderModule { module: module }
    }
}

#[derive(Debug)]
pub struct DawnInstance {
    instance: WGPUInstance,
}
unsafe impl Send for DawnInstance {}

#[derive(Debug)]
pub struct DawnPipelineLayout {
    layout: WGPUPipelineLayout,
}
impl PipelineLayout for DawnPipelineLayout {}

#[derive(Debug)]
pub struct DawnQueue {
    queue: WGPUQueue,
}
unsafe impl Send for DawnQueue {}

impl Queue<DawnBuffer, DawnCommandBuffer> for DawnQueue {
    fn submit(&self, buf: Option<DawnCommandBuffer>) {
        match buf {
            None => (),
            Some(buf) => unsafe {
                wgpuQueueSubmit((*self).queue.into(), 1, std::ptr::addr_of!(buf.buffer));
            },
        };
    }

    fn write_buffer(&self, buffer: &DawnBuffer, offset: u64, data: &[u8]) {
        unsafe {
            let data_ptr =
                std::mem::transmute::<*const u8, *const std::os::raw::c_void>(data.as_ptr());
            let mut sz = data.len();
            if sz % 4 != 0 {
                sz += 2;
            }
            wgpuQueueWriteBuffer(
                (*self).queue.into(),
                (*buffer).buffer.into(),
                offset,
                data_ptr,
                sz,
            );
        }
    }
}

#[derive(Debug)]
pub struct DawnShaderModule {
    module: WGPUShaderModule,
}
impl ShaderModule for DawnShaderModule {}

/// The compute instance is shared across all [dawn runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = WgpuServer<DawnApi, SimpleMemoryManagement<WgpuStorage<DawnApi>>>;

impl WebGPUApi for DawnApi {
    type Adapter = DawnAdapter;
    type AdapterInfo = DawnAdapterInfo;
    type Backend = DawnBackend;
    type BindGroup = DawnBindGroup;
    type BindGroupLayout = DawnBindGroupLayout;
    type Buffer = DawnBuffer;
    type CommandBuffer = DawnCommandBuffer;
    type CommandEncoder = DawnCommandEncoder;
    type ComputePipeline = DawnComputePipeline;
    type Device = DawnDevice;
    type PipelineLayout = DawnPipelineLayout;
    type Queue = DawnQueue;
    type ShaderModule = DawnShaderModule;

    const MAP_READ: u32 = MAP_READ;
    const COPY_SRC: u32 = COPY_SRC;
    const COPY_DST: u32 = COPY_DST;
    const STORAGE: u32 = STORAGE;

    type Server = WgpuServer<Self, SimpleMemoryManagement<WgpuStorage<Self>>>;
    type Channel = MutexComputeChannel<WgpuServer<Self, SimpleMemoryManagement<WgpuStorage<Self>>>>;

    fn client<G: GraphicsApi>(device: &WgpuDevice) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            pollster::block_on(create_client::<DawnApi, G>(
                device,
                RuntimeOptions::default(),
            ))
        })
    }

    async fn select_device(adapter: &DawnAdapter) -> (DawnDevice, DawnQueue) {
        let mut req_data = DevRequestData {
            device: std::ptr::null::<WGPUDevice>() as WGPUDevice,
            is_set: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        };
        let desc = WGPUDeviceDescriptor {
            nextInChain: std::ptr::null(),
            label: std::ptr::null(),
            requiredFeatureCount: 1,
            requiredFeatures: &WGPUFeatureName_WGPUFeatureName_ShaderF16,
            requiredLimits: std::ptr::null(),
            defaultQueue: WGPUQueueDescriptor {
                nextInChain: std::ptr::null(),
                label: std::ptr::null(),
            },
            deviceLostCallback: None,
            deviceLostCallbackInfo: WGPUDeviceLostCallbackInfo {
                nextInChain: std::ptr::null(),
                mode: 0,
                callback: None,
                userdata: std::ptr::null_mut(),
            },
            deviceLostUserdata: std::ptr::null_mut(),
            uncapturedErrorCallbackInfo: WGPUUncapturedErrorCallbackInfo {
                nextInChain: std::ptr::null(),
                callback: None,
                userdata: std::ptr::null_mut(),
            },
        };

        unsafe {
            let data_ptr = std::mem::transmute::<*mut DevRequestData, *mut std::os::raw::c_void>(
                std::ptr::addr_of_mut!(req_data),
            );
            wgpuAdapterRequestDevice(
                (*adapter).adapter.into(),
                &desc,
                Some(request_device_cb),
                data_ptr,
            );
        }

        let mut is_set = req_data.is_set.lock().unwrap();
        while !*is_set {
            is_set = req_data.cv.wait(is_set).unwrap();
        }

        unsafe {
            wgpuDeviceSetUncapturedErrorCallback(
                req_data.device,
                Some(device_error_callback),
                std::ptr::null_mut(),
            );
            wgpuDeviceSetLoggingCallback(
                req_data.device,
                Some(device_logging_callback),
                std::ptr::null_mut(),
            );
        }

        let dev = DawnDevice {
            device: req_data.device,
        };
        let queue: WGPUQueue;
        unsafe {
            queue = wgpuDeviceGetQueue(dev.device.into());
        }
        (dev, DawnQueue { queue })
    }

    fn select_adapter<G: GraphicsApi>(_: &WgpuDevice) -> DawnAdapter {
        let instance: WGPUInstance;
        let instance_desc = WGPUInstanceDescriptor {
            nextInChain: std::ptr::null(),
            features: WGPUInstanceFeatures {
                nextInChain: std::ptr::null(),
                timedWaitAnyEnable: 0,
                timedWaitAnyMaxCount: 0,
            },
        };
        unsafe {
            instance = wgpuCreateInstance(&instance_desc);
        }
        let mut req_data = AdapterRequestData {
            adapter: std::ptr::null::<WGPUAdapter>() as WGPUAdapter,
            is_set: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        };
        unsafe {
            let data_ptr = std::mem::transmute::<*mut AdapterRequestData, *mut std::os::raw::c_void>(
                std::ptr::addr_of_mut!(req_data),
            );
            wgpuInstanceRequestAdapter(
                instance,
                std::ptr::null(),
                Some(request_adapter_cb),
                data_ptr,
            );
        }

        let mut is_set = req_data.is_set.lock().unwrap();
        while !*is_set {
            is_set = req_data.cv.wait(is_set).unwrap();
        }

        DawnAdapter {
            adapter: req_data.adapter,
        }
    }

    fn device_poll(device: &DawnDevice) {
        let instance: WGPUInstance;
        let dev = (*device).device;
        unsafe {
            instance = wgpuAdapterGetInstance(wgpuDeviceGetAdapter(dev.into()));
            wgpuInstanceProcessEvents(instance.into());
            wgpuDeviceTick(dev.into());
        }
    }

    fn init_sync<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
        let device = Arc::new(device);
        let client = pollster::block_on(create_client::<Self, G>(&device, options));

        RUNTIME.register(&device, client)
    }

    async fn init_async<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
        let device = Arc::new(device);
        let client = create_client::<Self, G>(&device, options).await;

        RUNTIME.register(&device, client)
    }
}

#[allow(non_upper_case_globals)]
extern "C" fn device_error_callback(
    type_: WGPUErrorType,
    message: *const ::std::os::raw::c_char,
    _userdata: *mut ::std::os::raw::c_void,
) {
    let type_str = match type_ {
        WGPUErrorType_WGPUErrorType_Validation => "Validation",
        WGPUErrorType_WGPUErrorType_OutOfMemory => "Out of memory",
        WGPUErrorType_WGPUErrorType_Internal => "Internal",
        WGPUErrorType_WGPUErrorType_Unknown => "Unknown",
        WGPUErrorType_WGPUErrorType_DeviceLost => "Device lost",
        _ => "",
    };
    unsafe {
        let msg_str = std::ffi::CStr::from_ptr(message).to_str().unwrap();
        println!("{} error: {}", type_str, msg_str);
    }
}

extern "C" fn device_logging_callback(
    _type_: WGPULoggingType,
    message: *const ::std::os::raw::c_char,
    _userdata: *mut ::std::os::raw::c_void,
) {
    unsafe {
        let msg_str = std::ffi::CStr::from_ptr(message).to_str().unwrap();
        println!("Device log: {}", msg_str);
    }
}

extern "C" fn request_device_cb(
    _status: WGPURequestDeviceStatus,
    device: WGPUDevice,
    _message: *const ::std::os::raw::c_char,
    userdata: *mut ::std::os::raw::c_void,
) {
    unsafe {
        let req_data =
            std::mem::transmute::<*mut std::os::raw::c_void, *mut DevRequestData>(userdata);
        (*req_data).device = device;
        let mut is_set = (*req_data).is_set.lock().unwrap();
        *is_set = true;
        (*req_data).cv.notify_one();
    }
}

#[repr(C)]
struct DevRequestData {
    device: WGPUDevice,
    is_set: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

extern "C" fn request_adapter_cb(
    _status: WGPURequestAdapterStatus,
    adapter: WGPUAdapter,
    _message: *const ::std::os::raw::c_char,
    userdata: *mut ::std::os::raw::c_void,
) {
    unsafe {
        let req_data =
            std::mem::transmute::<*mut std::os::raw::c_void, *mut AdapterRequestData>(userdata);
        (*req_data).adapter = adapter;
        let mut is_set = (*req_data).is_set.lock().unwrap();
        *is_set = true;
        (*req_data).cv.notify_one();
    }
}

#[repr(C)]
struct AdapterRequestData {
    adapter: WGPUAdapter,
    is_set: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

#[repr(C)]
struct BufferReadData {
    read_done: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

unsafe extern "C" fn buffer_reader_cb(
    _status: WGPUBufferMapAsyncStatus,
    userdata: *mut ::std::os::raw::c_void,
) {
    unsafe {
        let read_data =
            std::mem::transmute::<*mut std::os::raw::c_void, *mut BufferReadData>(userdata);
        let mut read_done = (*read_data).read_done.lock().unwrap();
        (*read_done) = true;
        (*read_data).cv.notify_one();
    }
}
