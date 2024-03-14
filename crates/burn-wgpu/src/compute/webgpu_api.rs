use crate::{GraphicsApi, RuntimeOptions, WgpuDevice};
use burn_compute::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};
use burn_jit::compute::WorkGroup;
use std::borrow::Cow;
use alloc::sync::Arc;

pub trait Adapter<AdapterInfo>: core::fmt::Debug {
    fn get_info(&self) -> AdapterInfo;
}

pub trait AdapterInfo<Backend>: core::fmt::Debug {
    fn backend(&self) -> Backend;
    fn device(&self) -> DeviceId;
}

pub trait BindGroup: Send + core::fmt::Debug {}

pub struct BindGroupDescriptor<'a, BindGroupLayout, Buffer> {
    pub label: Option<&'a str>,
    pub layout: &'a BindGroupLayout,
    pub entries: &'a Vec<BindGroupEntry<'a, Buffer>>,
}

pub struct BindGroupEntry<'a, Buffer> {
    pub binding: u32,
    pub resource: BindingResource<'a, Buffer>,
}

pub trait BindGroupLayout: core::fmt::Debug {}

pub enum BindingResource<'a, Buffer> {
    Buffer(BufferBinding<'a, Buffer>),
}

pub trait Buffer<Buffer, Device>: Send + Sync + core::fmt::Debug {
    fn as_entire_buffer_binding(&self) -> BufferBinding<'_, Buffer>;
    fn destroy(&self);
    #[allow(async_fn_in_trait)]
    async fn read(&self, device: &Device) -> Vec<u8>;
    fn size(&self) -> u64;
}

pub struct BufferBinding<'a, Buffer> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: Option<std::num::NonZeroU64>,
}

pub struct BufferDescriptor<'a> {
    pub label: Option<&'a str>,
    pub size: u64,
    pub usage: u32,
    pub mapped_at_creation: bool,
}

pub struct BufferInitDescriptor<'a> {
    pub label: Option<&'a str>,
    pub contents: &'a [u8],
    pub usage: u32,
}

pub trait CommandBuffer: core::fmt::Debug {}

pub struct CommandEncoderDescriptor<'a> {
    pub label: Option<&'a str>,
}

pub trait CommandEncoder<BindGroup, Buffer, CommandBuffer, ComputePipeline>:
    Send + Sync + core::fmt::Debug
{
    fn dispatch_compute_pass(
        &mut self,
        desc: &ComputePassDescriptor,
        pipeline: Arc<ComputePipeline>,
        bind_group: BindGroup,
        work_group: WorkGroup,
    );
    fn copy_buffer_to_buffer(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        size: u64,
    );
    fn finish(self) -> CommandBuffer;
}

pub struct ComputePassDescriptor<'a> {
    pub label: Option<&'a str>,
}

pub trait ComputePipeline<BindGroupLayout>: Send + Sync + core::fmt::Debug {
    fn get_bind_group_layout(&self, id: u32) -> BindGroupLayout;
}

pub struct ComputePipelineDescriptor<'a, PipelineLayout, ShaderModule> {
    pub label: Option<&'a str>,
    pub layout: Option<&'a PipelineLayout>,
    pub module: &'a ShaderModule,
    pub entry_point: &'a str,
}

pub trait Device<
    BindGroup,
    BindGroupLayout,
    Buffer,
    CommandEncoder,
    ComputePipeline,
    PipelineLayout,
    ShaderModule,
>: Send + Sync + core::fmt::Debug
{
    fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor<'_, BindGroupLayout, Buffer>,
    ) -> BindGroup;
    fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer;
    fn create_buffer_init(&self, desc: &BufferInitDescriptor) -> Buffer;
    fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> CommandEncoder;
    fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor<PipelineLayout, ShaderModule>,
    ) -> ComputePipeline;
    fn create_shader_module(&self, desc: &ShaderModuleDescriptor) -> ShaderModule;
}

pub type DeviceId = u32;

pub trait PipelineLayout: core::fmt::Debug {}

pub trait Queue<Buffer, CommandBuffer>: Send + core::fmt::Debug {
    fn submit(&self, buf: Option<CommandBuffer>);
    fn write_buffer(&self, buf: &Buffer, offset: u64, data: &[u8]);
}

pub enum ShaderSource<'a> {
    Wgsl(Cow<'a, str>),
}

pub trait ShaderModule: core::fmt::Debug {}

pub struct ShaderModuleDescriptor<'a> {
    pub label: Option<&'a str>,
    pub source: ShaderSource<'a>,
}

pub trait WebGPUApi: Send + Sync + core::fmt::Debug + 'static {
    type Adapter: Adapter<Self::AdapterInfo>;
    type AdapterInfo: AdapterInfo<Self::Backend>;
    type Backend: core::convert::AsRef<str>;
    type BindGroup: BindGroup;
    type BindGroupLayout: BindGroupLayout;
    type Buffer: Buffer<Self::Buffer, Self::Device>;
    type CommandBuffer: CommandBuffer;
    type CommandEncoder: CommandEncoder<
        Self::BindGroup,
        Self::Buffer,
        Self::CommandBuffer,
        Self::ComputePipeline,
    >;
    type ComputePipeline: ComputePipeline<Self::BindGroupLayout>;
    type Device: Device<
        Self::BindGroup,
        Self::BindGroupLayout,
        Self::Buffer,
        Self::CommandEncoder,
        Self::ComputePipeline,
        Self::PipelineLayout,
        Self::ShaderModule,
    >;
    type PipelineLayout: PipelineLayout;
    type Queue: Queue<Self::Buffer, Self::CommandBuffer>;
    type ShaderModule: ShaderModule;

    const MAP_READ: u32;
    const COPY_SRC: u32;
    const COPY_DST: u32;
    const STORAGE: u32;

    type Server: ComputeServer<
        Kernel = burn_jit::compute::Kernel,
        AutotuneKey = burn_jit::compute::JitAutotuneKey,
    >;
    type Channel: ComputeChannel<Self::Server>;

    fn client<G: GraphicsApi>(device: &WgpuDevice) -> ComputeClient<Self::Server, Self::Channel>;
    #[allow(async_fn_in_trait)]
    async fn select_device(adapter: &Self::Adapter) -> (Self::Device, Self::Queue);
    #[allow(async_fn_in_trait)]
    #[cfg(target_family = "wasm")]
    async fn select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> Self::Adapter;
    #[cfg(not(target_family = "wasm"))]
    fn select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> Self::Adapter;
    fn device_poll(device: &Self::Device);

    fn init_sync<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions);
    #[allow(async_fn_in_trait)]
    async fn init_async<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions);
}
