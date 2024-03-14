#![allow(missing_docs)]

use crate::compute::dawn_native_bindings::*;
use crate::{GraphicsApi, WgpuDevice};

use std::{borrow::Cow, num::NonZeroU64};

#[derive(Debug)]
pub struct WebGPUAdapter {
    adapter: WGPUAdapter,
}

impl WebGPUAdapter {
    pub fn get_info(&self) -> WebGPUAdapterInfo {
        let mut properties = WGPUAdapterProperties {
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
            wgpuAdapterGetProperties(self.adapter, &mut properties);
        }
        WebGPUAdapterInfo {
            device: properties.deviceID,
            backend: WebGPUBackend::from_u32(properties.backendType),
        }
    }
}

#[derive(Debug)]
pub struct WebGPUAdapterInfo {
    pub device: u32,
    pub backend: WebGPUBackend,
}

#[derive(Debug)]
pub enum WebGPUBackend {
    Undefined = wgpu_BackendType_Undefined as isize,
    Null = wgpu_BackendType_Null as isize,
    WebGPU = wgpu_BackendType_WebGPU as isize,
    D3D11 = wgpu_BackendType_D3D11 as isize,
    D3D12 = wgpu_BackendType_D3D12 as isize,
    Metal = wgpu_BackendType_Metal as isize,
    Vulkan = wgpu_BackendType_Vulkan as isize,
    OpenGL = wgpu_BackendType_OpenGL as isize,
    OpenGLES = wgpu_BackendType_OpenGLES as isize,
}

impl WebGPUBackend {
    #[allow(non_upper_case_globals)]
    fn from_u32(val: u32) -> WebGPUBackend {
        match val {
            wgpu_BackendType_Undefined => WebGPUBackend::Undefined,
            wgpu_BackendType_Null => WebGPUBackend::Null,
            wgpu_BackendType_WebGPU => WebGPUBackend::WebGPU,
            wgpu_BackendType_D3D11 => WebGPUBackend::D3D11,
            wgpu_BackendType_D3D12 => WebGPUBackend::D3D12,
            wgpu_BackendType_Metal => WebGPUBackend::Metal,
            wgpu_BackendType_Vulkan => WebGPUBackend::Vulkan,
            wgpu_BackendType_OpenGL => WebGPUBackend::OpenGL,
            wgpu_BackendType_OpenGLES => WebGPUBackend::OpenGLES,
            _ => panic!("Unknown WebGPU backend type: {}", val),
        }
    }

    pub const fn to_str(self) -> &'static str {
        match self {
            WebGPUBackend::Undefined => "undefined",
            WebGPUBackend::Null => "null",
            WebGPUBackend::WebGPU => "webgpu",
            WebGPUBackend::D3D11 => "dx11",
            WebGPUBackend::D3D12 => "dx12",
            WebGPUBackend::Metal => "metal",
            WebGPUBackend::Vulkan => "vulkan",
            WebGPUBackend::OpenGL => "opengl",
            WebGPUBackend::OpenGLES => "opengles",
        }
    }
}

#[derive(Debug)]
pub struct WebGPUBindGroup {
    bind_group: WGPUBindGroup,
}
unsafe impl Send for WebGPUBindGroup {}
unsafe impl Sync for WebGPUBindGroup {}

#[derive(Debug)]
pub struct WebGPUBindGroupDescriptor<'a> {
    pub label: Option<&'a str>,
    pub layout: &'a WebGPUBindGroupLayout,
    pub entries: &'a Vec<WebGPUBindGroupEntry<'a>>,
}

#[derive(Debug)]
pub struct WebGPUBindGroupEntry<'a> {
    pub binding: u32,
    pub resource: WebGPUBindingResource<'a>,
}

#[derive(Debug)]
pub struct WebGPUBindGroupLayout {
    layout: WGPUBindGroupLayout,
}

#[derive(Debug)]
pub enum WebGPUBindingResource<'a> {
    Buffer(WebGPUBufferBinding<'a>),
}

#[derive(Debug)]
pub struct WebGPUBuffer {
    buffer: WGPUBuffer,
    size: u64,
}
unsafe impl Send for WebGPUBuffer {}
unsafe impl Sync for WebGPUBuffer {}

impl WebGPUBuffer {
    pub fn as_entire_buffer_binding(&self) -> WebGPUBufferBinding {
        WebGPUBufferBinding {
            buffer: self,
            offset: 0,
            size: Some(NonZeroU64::new((*self).size).unwrap()),
        }
    }

    pub fn destroy(&self) {
        unsafe {
            wgpuBufferDestroy((*self).buffer.into());
        }
    }

    pub fn size(&self) -> u64 {
        (*self).size
    }
}

pub type WebGPUBufferAddress = u64;

#[derive(Clone, Copy, Debug)]
pub struct WebGPUBufferBinding<'a> {
    pub buffer: &'a WebGPUBuffer,
    pub offset: u64,
    pub size: Option<NonZeroU64>,
}

pub type WebGPUBufferSize = NonZeroU64;

#[derive(Debug)]
pub struct WebGPUBufferDescriptor<'a> {
    pub label: Option<&'a str>,
    pub size: u64,
    pub usage: WebGPUBufferUsages,
    pub mapped_at_creation: bool,
}

pub type WebGPUBufferUsages = u32;
pub const MAP_READ: WebGPUBufferUsages = wgpu_BufferUsage_MapRead;
pub const COPY_SRC: WebGPUBufferUsages = wgpu_BufferUsage_CopySrc;
pub const COPY_DST: WebGPUBufferUsages = wgpu_BufferUsage_CopyDst;
pub const STORAGE: WebGPUBufferUsages = wgpu_BufferUsage_Storage;

#[derive(Debug)]
pub struct WebGPUCommandBuffer {
    buffer: WGPUCommandBuffer,
}

#[derive(Debug)]
pub struct WebGPUCommandEncoder {
    encoder: WGPUCommandEncoder,
}
unsafe impl Send for WebGPUCommandEncoder {}
unsafe impl Sync for WebGPUCommandEncoder {}

impl WebGPUCommandEncoder {
    pub fn begin_compute_pass(&self, desc: &WebGPUComputePassDescriptor) -> WebGPUComputePass {
        let label = match desc.label {
            Some(name) => name,
            None => "",
        };
        let timestamp_writes = match desc.timestamp_writes {
            Some(_) => panic!("binding for setting timestamp_writes is not implemented yet"),
            None => std::ptr::null(),
        };
        let pass_desc = WGPUComputePassDescriptor {
            nextInChain: std::ptr::null(),
            label: std::ffi::CString::new(label).unwrap().into_raw(),
            timestampWrites: timestamp_writes,
        };
        let pass: WGPUComputePassEncoder;
        unsafe {
            pass = wgpuCommandEncoderBeginComputePass(self.encoder.into(), &pass_desc);
        }
        WebGPUComputePass { pass: pass }
    }

    pub fn copy_buffer_to_buffer(
        &self,
        src: &WebGPUBuffer,
        src_offset: u64,
        dest: &WebGPUBuffer,
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

    pub fn finish(&self) -> WebGPUCommandBuffer {
        let cmd_buf_desc = WGPUCommandBufferDescriptor {
            nextInChain: std::ptr::null(),
            label: std::ptr::null(),
        };
        let cmd_buf: WGPUCommandBuffer;
        unsafe {
            cmd_buf = wgpuCommandEncoderFinish((*self).encoder.into(), &cmd_buf_desc);
        }
        WebGPUCommandBuffer { buffer: cmd_buf }
    }
}

#[derive(Debug)]
pub struct WebGPUCommandEncoderDescriptor<'a> {
    pub label: Option<&'a str>,
}

#[derive(Debug)]
pub struct WebGPUComputePass {
    pass: WGPUComputePassEncoder,
}

impl WebGPUComputePass {
    pub fn set_pipeline(&self, pipeline: &WebGPUComputePipeline) {
        unsafe {
            wgpuComputePassEncoderSetPipeline((*self).pass, (*pipeline).pipeline.into());
        }
    }

    pub fn set_bind_group(&self, id: u32, bind_group: &WebGPUBindGroup, offsets: &[u32]) {
        unsafe {
            wgpuComputePassEncoderSetBindGroup(
                (*self).pass,
                id,
                (*bind_group).bind_group.into(),
                offsets.len(),
                offsets.as_ptr(),
            );
        }
    }

    pub fn dispatch_workgroups(&self, x: u32, y: u32, z: u32) {
        unsafe {
            wgpuComputePassEncoderDispatchWorkgroups((*self).pass, x, y, z);
        }
    }
}

impl std::ops::Drop for WebGPUComputePass {
    fn drop(&mut self) {
        unsafe {
            wgpuComputePassEncoderEnd((*self).pass);
        }
    }
}

#[derive(Debug)]
pub struct WebGPUComputePassDescriptor<'a> {
    pub label: Option<&'a str>,
    pub timestamp_writes: Option<()>,
}

#[derive(Debug)]
pub struct WebGPUComputePipeline {
    pipeline: WGPUComputePipeline,
}
unsafe impl Send for WebGPUComputePipeline {}
unsafe impl Sync for WebGPUComputePipeline {}

impl WebGPUComputePipeline {
    pub fn get_bind_group_layout(&self, id: u32) -> WebGPUBindGroupLayout {
        let layout: WGPUBindGroupLayout;
        unsafe {
            layout = wgpuComputePipelineGetBindGroupLayout((*self).pipeline.into(), id);
        }
        WebGPUBindGroupLayout { layout: layout }
    }
}

#[derive(Debug)]
pub struct WebGPUComputePipelineDescriptor<'a> {
    pub label: Option<&'a str>,
    pub layout: Option<&'a WebGPUPipelineLayout>,
    pub module: &'a WebGPUShaderModule,
    pub entry_point: &'a str,
}

#[derive(Debug)]
pub struct WebGPUDevice {
    device: WGPUDevice,
}
unsafe impl Send for WebGPUDevice {}
unsafe impl Sync for WebGPUDevice {}

impl WebGPUDevice {
    pub fn create_bind_group(&self, desc: &WebGPUBindGroupDescriptor) -> WebGPUBindGroup {
        let entries = (*desc)
            .entries
            .iter()
            .map(|entry| {
                let resource = match entry.resource {
                    WebGPUBindingResource::Buffer(res) => res,
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
        WebGPUBindGroup {
            bind_group: bind_group,
        }
    }

    pub fn create_buffer(&self, desc: &WebGPUBufferDescriptor) -> WebGPUBuffer {
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
        WebGPUBuffer {
            buffer: buffer,
            size: (*desc).size,
        }
    }

    pub fn create_command_encoder(
        &self,
        desc: &WebGPUCommandEncoderDescriptor,
    ) -> WebGPUCommandEncoder {
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
        WebGPUCommandEncoder { encoder: encoder }
    }

    pub fn create_compute_pipeline(
        &self,
        desc: &WebGPUComputePipelineDescriptor,
    ) -> WebGPUComputePipeline {
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
        WebGPUComputePipeline { pipeline: pipeline }
    }

    pub fn create_shader_module(&self, desc: WebGPUShaderModuleDescriptor) -> WebGPUShaderModule {
        let label = match desc.label {
            None => std::ptr::null(),
            Some(name) => std::ffi::CString::new(name).unwrap().into_raw(),
        };
        let src = match desc.source {
            WebGPUShaderSource::Wgsl(source) => source.to_string(),
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
        WebGPUShaderModule { module: module }
    }

    pub fn get_queue(&self) -> WebGPUQueue {
        let queue: WGPUQueue;
        unsafe {
            queue = wgpuDeviceGetQueue((*self).device.into());
        }
        WebGPUQueue { queue: queue }
    }
}

#[derive(Debug)]
pub struct WebGPUInstance {
    instance: WGPUInstance,
}
unsafe impl Send for WebGPUInstance {}
unsafe impl Sync for WebGPUInstance {}

#[derive(Debug)]
pub struct WebGPUPipelineLayout {
    layout: WGPUPipelineLayout,
}
unsafe impl Send for WebGPUPipelineLayout {}
unsafe impl Sync for WebGPUPipelineLayout {}

#[derive(Debug)]
pub struct WebGPUQueue {
    queue: WGPUQueue,
}
unsafe impl Send for WebGPUQueue {}
unsafe impl Sync for WebGPUQueue {}

impl WebGPUQueue {
    pub fn write_buffer(&self, buffer: &WebGPUBuffer, offset: u64, data: &[u8]) {
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
pub struct WebGPUShaderModule {
    module: WGPUShaderModule,
}

pub struct WebGPUShaderModuleDescriptor<'a> {
    pub label: Option<&'a str>,
    pub source: WebGPUShaderSource<'a>,
}

impl WebGPUQueue {
    pub fn submit(&self, cmd_buf: Option<WebGPUCommandBuffer>) {
        match cmd_buf {
            None => (),
            Some(cmd_buf) => unsafe {
                wgpuQueueSubmit((*self).queue.into(), 1, std::ptr::addr_of!(cmd_buf.buffer));
            },
        };
    }
}

pub enum WebGPUShaderSource<'a> {
    Wgsl(Cow<'a, str>),
}

pub async fn webgpu_select_device(adapter: &WebGPUAdapter) -> (WebGPUDevice, WebGPUQueue) {
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
        deviceLostUserdata: std::ptr::null_mut(),
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

    let dev = WebGPUDevice {
        device: req_data.device,
    };
    let queue = dev.get_queue();
    (dev, queue)
}

pub fn webgpu_select_adapter<G: GraphicsApi>(_: &WgpuDevice) -> WebGPUAdapter {
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
        dawnProcSetProcs(dawn_native_GetProcs());
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

    WebGPUAdapter {
        adapter: req_data.adapter,
    }
}

pub fn webgpu_read_buffer(buffer: &WebGPUBuffer, device: &WebGPUDevice) -> Vec<u8> {
    let mut read_data = BufferReadData {
        read_done: std::sync::Mutex::new(false),
        cv: std::sync::Condvar::new(),
    };
    unsafe {
        let data_ptr = std::mem::transmute::<*mut BufferReadData, *mut std::os::raw::c_void>(
            std::ptr::addr_of_mut!(read_data),
        );
        let mut sz = (*buffer).size;
        if sz % 4 != 0 {
            sz += 2;
        }
        wgpuBufferMapAsync(
            (*buffer).buffer.into(),
            wgpu_MapMode_Read,
            0,
            sz as usize,
            Some(buffer_reader_cb),
            data_ptr,
        );

        let mut read_done = read_data.read_done.lock().unwrap();
        let should_process = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let spt = should_process.clone();
        let instance = WebGPUInstance {
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
            wgpuBufferGetConstMappedRange((*buffer).buffer.into(), 0, (*buffer).size as usize);
        let slice = std::slice::from_raw_parts(mpd_rng as *const u8, (*buffer).size as usize);
        slice.to_vec()
    }
}

pub fn webgpu_device_poll(device: &WebGPUDevice) {
    let instance: WGPUInstance;
    let dev = (*device).device;
    unsafe {
        instance = wgpuAdapterGetInstance(wgpuDeviceGetAdapter(dev.into()));
        wgpuInstanceProcessEvents(instance.into());
        wgpuDeviceTick(dev.into());
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
