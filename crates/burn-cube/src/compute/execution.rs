use crate::compute::{FullCompilationPhase, Kernel, WorkGroup};
use crate::{GpuComputeShaderPhase, Runtime};
use burn_compute::client::ComputeClient;
use burn_compute::server::{Binding, Handle};

pub struct KernelLauncher<R: Runtime> {
    pub arrays: Vec<Binding<R::Server>>,
    pub array_lengths: Vec<u32>,
    pub info: Vec<u32>,
    pub scalar_bf16: Vec<half::bf16>,
    pub scalar_f16: Vec<half::f16>,
    pub scalar_f32: Vec<f32>,
    pub scalar_f64: Vec<f64>,
    pub scalar_usize: Vec<usize>,
    pub scalar_u64: Vec<u64>,
    pub scalar_u32: Vec<u32>,
    pub scalar_u16: Vec<u16>,
    pub scalar_i32: Vec<i32>,
    pub scalar_i16: Vec<i16>,
}

impl<R: Runtime> KernelLauncher<R> {
    pub fn new() -> Self {
        Self {
            arrays: Vec::new(),
            array_lengths: Vec::new(),
            info: Vec::new(),
            scalar_bf16: Vec::new(),
            scalar_f16: Vec::new(),
            scalar_f32: Vec::new(),
            scalar_f64: Vec::new(),
            scalar_usize: Vec::new(),
            scalar_u64: Vec::new(),
            scalar_u32: Vec::new(),
            scalar_u16: Vec::new(),
            scalar_i32: Vec::new(),
            scalar_i16: Vec::new(),
        }
    }
}

impl<R: Runtime> KernelLauncher<R> {
    pub fn launch<K: GpuComputeShaderPhase>(
        self,
        workgroup: WorkGroup,
        kernel: K,
        client: ComputeClient<R::Server, R::Channel>,
    ) {
        let settings = self.into_settings(&client, workgroup);

        let mut handles = settings.handles_tensors;
        let workgroup = settings.workgroup;

        handles.push(settings.handle_info.binding());

        for handle in settings.handles_scalars.into_iter() {
            handles.push(handle.binding());
        }

        let kernel = Kernel::JitGpu(Box::new(FullCompilationPhase::<R::Compiler, K>::new(
            kernel, workgroup,
        )));

        client.execute(kernel, handles);
    }

    fn into_settings(
        mut self,
        client: &ComputeClient<R::Server, R::Channel>,
        workgroup: WorkGroup,
    ) -> ExecuteSettings<R> {
        if R::require_array_lengths() {
            for len in self.array_lengths {
                self.info.push(len);
            }
        }

        let info = client.create(bytemuck::cast_slice(&self.info));

        let mut handles_scalars = Vec::new();

        if !self.scalar_bf16.is_empty() {
            handles_scalars.push(client.create(bytemuck::cast_slice(&self.scalar_bf16)));
        }

        if !self.scalar_f16.is_empty() {
            handles_scalars.push(client.create(bytemuck::cast_slice(&self.scalar_bf16)));
        }

        if !self.scalar_f32.is_empty() {
            handles_scalars.push(client.create(bytemuck::cast_slice(&self.scalar_f32)));
        }

        if !self.scalar_f64.is_empty() {
            handles_scalars.push(client.create(bytemuck::cast_slice(&self.scalar_f64)));
        }

        if !self.scalar_u64.is_empty() {
            handles_scalars.push(client.create(bytemuck::cast_slice(&self.scalar_u64)));
        }

        if !self.scalar_u32.is_empty() {
            handles_scalars.push(client.create(bytemuck::cast_slice(&self.scalar_u32)));
        }

        if !self.scalar_u16.is_empty() {
            handles_scalars.push(client.create(bytemuck::cast_slice(&self.scalar_u16)));
        }

        ExecuteSettings {
            handles_tensors: self.arrays,
            handle_info: info,
            handles_scalars,
            workgroup,
        }
    }
}

struct ExecuteSettings<R: Runtime> {
    handles_tensors: Vec<Binding<R::Server>>,
    handle_info: Handle<R::Server>,
    handles_scalars: Vec<Handle<R::Server>>,
    workgroup: WorkGroup,
}
