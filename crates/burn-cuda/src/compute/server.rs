use super::storage::Binding;
use super::storage::CudaStorage;
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_jit::compute::{JitAutotuneKey, Kernel, WorkGroup};
use burn_jit::gpu::WorkgroupSize;
use cudarc::driver::sys::CUctx_st;
use cudarc::driver::sys::CUfunc_st;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;

#[derive(Debug)]
pub struct CudaServer<MM: MemoryManagement<CudaStorage>> {
    state: CudaServerState<MM>,
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
    workgroup_size: WorkgroupSize,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

unsafe impl<MM: MemoryManagement<CudaStorage>> Send for CudaServer<MM> {}

impl<MM: MemoryManagement<CudaStorage>> ComputeServer for CudaServer<MM> {
    type Kernel = Kernel;
    type Storage = CudaStorage;
    type MemoryManagement = MM;
    type AutotuneKey = JitAutotuneKey;

    fn read(&mut self, binding: server::Binding<Self>) -> burn_tensor::Reader<Vec<u8>> {
        let ctx = self.get_context();
        let resource = ctx.memory_management.get(binding.memory);
        // TODO: Check if it is possible to make this faster
        let mut data = vec![0; resource.size() as usize];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(&mut data, resource.ptr, ctx.stream).unwrap();
        };

        ctx.sync();

        burn_tensor::Reader::Concrete(data)
    }

    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(data.len());
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
        let handle = ctx.memory_management.reserve(size);
        server::Handle::new(handle)
    }

    fn execute(&mut self, kernel: Self::Kernel, bindings: Vec<server::Binding<Self>>) {
        let ctx = self.get_context();
        let kernel_id = kernel.id();
        let settings = kernel.launch_settings();

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel);
        }

        let bindings = bindings
            .into_iter()
            .map(|binding| ctx.memory_management.get(binding.memory).as_binding())
            .collect();

        ctx.execute_task(kernel_id, settings.workgroup, bindings);
        // TODO: fix this
        // self.memory_management.storage().perform_deallocations();
    }

    fn sync(&mut self) {
        let ctx = self.get_context();
        ctx.sync();
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

    fn compile_kernel(&mut self, kernel_id: &str, kernel: Kernel) {
        let kernel_compiled = kernel.compile();
        let shared_mem_bytes = kernel_compiled.shared_mem_bytes;
        let workgroup_size = kernel_compiled.workgroup_size;

        let ptx = unsafe {
            let program = cudarc::nvrtc::result::create_program(kernel_compiled.source).unwrap();
            if cudarc::nvrtc::result::compile_program::<Vec<_>>(program, &[]).is_err() {
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
                workgroup_size,
                shared_mem_bytes,
                func,
            },
        );
    }

    fn execute_task(
        &mut self,
        kernel_id: String,
        workgroup: WorkGroup,
        mut bindings: Vec<Binding>,
    ) {
        let kernel = self.module_names.get(&kernel_id).unwrap();
        let workgroup_size = kernel.workgroup_size;

        unsafe {
            cudarc::driver::result::launch_kernel(
                kernel.func,
                (workgroup.x, workgroup.y, workgroup.z),
                (workgroup_size.x, workgroup_size.y, workgroup_size.z),
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
        Self {
            state: CudaServerState::Uninitialized {
                device_index: index,
                init,
            },
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
