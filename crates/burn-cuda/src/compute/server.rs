use super::storage::Binding;
use super::storage::CudaStorage;
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_cube::ir::CubeDim;
use burn_cube::prelude::*;
use burn_cube::FeatureSet;
use burn_jit::JitAutotuneKey;
use burn_tensor::backend::SyncType;
use burn_tensor::reader_from_concrete;
use burn_tensor::Reader;
use cudarc::driver::sys::CUctx_st;
use cudarc::driver::sys::CUfunc_st;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;

#[derive(Debug)]
pub struct CudaServer<MM: MemoryManagement<CudaStorage>> {
    state: CudaServerState<MM>,
    pub(crate) archs: Vec<i32>,
    pub(crate) minimum_arch_version: i32,
}

pub(crate) enum CudaServerState<MM: MemoryManagement<CudaStorage>> {
    Uninitialized {
        device_index: usize,
        init: Box<dyn Fn(usize) -> CudaContext<MM>>,
    },
    Initialized {
        ctx: CudaContext<MM>,
    },
}

impl<MM: MemoryManagement<CudaStorage>> core::fmt::Debug for CudaServerState<MM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Context")
    }
}

#[derive(Debug)]
pub(crate) struct CudaContext<MM: MemoryManagement<CudaStorage>> {
    context: *mut CUctx_st,
    stream: cudarc::driver::sys::CUstream,
    memory_management: MM,
    module_names: HashMap<String, CompiledKernel>,
}

#[derive(Debug)]
struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

unsafe impl<MM: MemoryManagement<CudaStorage>> Send for CudaServer<MM> {}

impl<MM: MemoryManagement<CudaStorage>> ComputeServer for CudaServer<MM> {
    type Kernel = Box<dyn CubeTask>;
    type DispatchCount = CubeCount<Self>;
    type Storage = CudaStorage;
    type MemoryManagement = MM;
    type AutotuneKey = JitAutotuneKey;
    type FeatureSet = FeatureSet;

    fn read(&mut self, binding: server::Binding<Self>) -> Reader {
        let ctx = self.get_context();
        let resource = ctx.memory_management.get(binding.memory);

        // TODO: Check if it is possible to make this faster
        let mut data = vec![0; resource.size() as usize];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(&mut data, resource.ptr, ctx.stream).unwrap();
        };
        ctx.sync();
        reader_from_concrete(data)
    }

    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(data.len(), || unsafe {
            cudarc::driver::result::stream::synchronize(ctx.stream).unwrap();
        });
        let handle = server::Handle::new(handle);
        let binding = handle.clone().binding().memory;
        let resource = ctx.memory_management.get(binding);

        unsafe {
            cudarc::driver::result::memcpy_htod_async(resource.ptr, data, ctx.stream).unwrap();
        }

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size, || unsafe {
            cudarc::driver::result::stream::synchronize(ctx.stream).unwrap();
        });
        server::Handle::new(handle)
    }

    fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: Self::DispatchCount,
        bindings: Vec<server::Binding<Self>>,
    ) {
        let arch = self.minimum_arch_version;

        let kernel_id = kernel.id();

        let count = match count {
            CubeCount::Fixed(x, y, z) => (x, y, z),
            // TODO: There should be a way to have cuda use the GPU values without doing a readback,
            // but I'm not sure.
            CubeCount::Dynamic(handle) => {
                let data = burn_tensor::try_read_sync(self.read(handle.binding())).unwrap();
                let data: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                (data[0], data[1], data[2])
            }
        };

        let ctx = self.get_context();

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, arch);
        }

        let bindings = bindings
            .into_iter()
            .map(|binding| ctx.memory_management.get(binding.memory).as_binding())
            .collect();

        ctx.execute_task(kernel_id, count, bindings);
        // TODO: fix this
        // self.memory_management.storage().perform_deallocations();
    }

    fn sync(&mut self, sync_type: SyncType) {
        match sync_type {
            // Synchronize the stream if waiting.
            SyncType::Wait => {
                let ctx = self.get_context();
                ctx.sync();
            }
            // Nothing to do - all tasks are already submitted to the stream.
            SyncType::Flush => (),
        }
    }

    fn get_resource(
        &mut self,
        binding: server::Binding<Self>,
    ) -> <Self::Storage as burn_compute::storage::ComputeStorage>::Resource {
        let ctx = self.get_context();
        ctx.memory_management.get(binding.memory)
    }
}

impl<MM: MemoryManagement<CudaStorage>> CudaContext<MM> {
    pub fn new(
        memory_management: MM,
        stream: cudarc::driver::sys::CUstream,
        context: *mut CUctx_st,
    ) -> Self {
        Self {
            context,
            memory_management,
            module_names: HashMap::new(),
            stream,
        }
    }

    fn sync(&mut self) {
        unsafe {
            cudarc::driver::result::stream::synchronize(self.stream).unwrap();
        };
    }

    fn compile_kernel(&mut self, kernel_id: &str, kernel: Box<dyn CubeTask>, arch: i32) {
        let kernel_compiled = kernel.compile();
        let shared_mem_bytes = kernel_compiled.shared_mem_bytes;
        let cube_dim = kernel_compiled.cube_dim;
        let arch = format!("--gpu-architecture=sm_{}", arch);

        #[cfg(target_os = "linux")]
        let options = &[
            arch.as_str(),
            "--include-path=/usr/include",
            "--include-path=/usr/include/cuda",
            "--include-path=/usr/local/include/cuda",
        ];
        #[cfg(not(target_os = "linux"))] // TODO: add include-path for other OS.
        let options = &[arch.as_str()];

        let ptx = unsafe {
            let program = cudarc::nvrtc::result::create_program(kernel_compiled.source).unwrap();
            if cudarc::nvrtc::result::compile_program(program, options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).unwrap();
                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                let source = kernel.compile().source;
                panic!("{message}\n[Source]  \n{source}");
            };
            cudarc::nvrtc::result::get_ptx(program).unwrap()
        };

        let func_name = CString::new("kernel".to_string()).unwrap();
        let func = unsafe {
            let module =
                cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _).unwrap();
            cudarc::driver::result::module::get_function(module, func_name).unwrap()
        };

        self.module_names.insert(
            kernel_id.to_string(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );
    }

    fn execute_task(
        &mut self,
        kernel_id: String,
        dispatch_count: (u32, u32, u32),
        mut bindings: Vec<Binding>,
    ) {
        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        unsafe {
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                kernel.shared_mem_bytes as u32,
                self.stream,
                &mut bindings,
            )
            .unwrap();
        };
    }
}

impl<MM: MemoryManagement<CudaStorage>> CudaServer<MM> {
    /// Create a new cuda server.
    pub(crate) fn new(index: usize, init: Box<dyn Fn(usize) -> CudaContext<MM>>) -> Self {
        let archs = unsafe {
            let mut num_supported_arg: core::ffi::c_int = 0;
            cudarc::nvrtc::sys::lib()
                .nvrtcGetNumSupportedArchs(core::ptr::from_mut(&mut num_supported_arg));

            let mut archs: Vec<core::ffi::c_int> = vec![0; num_supported_arg as usize];
            cudarc::nvrtc::sys::lib().nvrtcGetSupportedArchs(core::ptr::from_mut(&mut archs[0]));
            archs
        };

        let minimum_arch_version = archs[0];

        Self {
            state: CudaServerState::Uninitialized {
                device_index: index,
                init,
            },
            archs,
            minimum_arch_version,
        }
    }

    fn get_context(&mut self) -> &mut CudaContext<MM> {
        if let CudaServerState::Uninitialized { device_index, init } = &self.state {
            let ctx = init(*device_index);
            self.state = CudaServerState::Initialized { ctx };
        }
        if let CudaServerState::Initialized { ctx } = &mut self.state {
            unsafe {
                cudarc::driver::result::ctx::set_current(ctx.context).unwrap();
            };
            ctx
        } else {
            panic!("Context should be initialized");
        }
    }
}
