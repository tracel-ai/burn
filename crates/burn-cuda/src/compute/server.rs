use super::storage::Binding;
use super::storage::CudaStorage;
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_jit::compute::{JitAutotuneKey, Kernel, WorkGroup};
use burn_jit::gpu::WorkgroupSize;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::{driver::CudaDevice, nvrtc::compile_ptx};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug)]
pub struct CudaServer<MM: MemoryManagement<CudaStorage>> {
    memory_management: MM,
    device: Arc<CudaDevice>,
    manual_available: HashMap<usize, Vec<server::Handle<Self>>>,
    manual_taken: Vec<(usize, server::Handle<Self>)>,
    module_names: HashMap<String, (String, WorkgroupSize)>,
}

impl<MM: MemoryManagement<CudaStorage>> ComputeServer for CudaServer<MM> {
    type Kernel = Kernel;
    type Storage = CudaStorage;
    type MemoryManagement = MM;
    type AutotuneKey = JitAutotuneKey;

    fn read(&mut self, handle: &server::Handle<Self>) -> burn_tensor::Reader<Vec<u8>> {
        self.sync();

        let resource = self.memory_management.get(&handle.memory);
        let data = self.device.dtoh_sync_copy(resource.buffer).unwrap();
        burn_tensor::Reader::Concrete(data)
    }

    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let handle = self.manual_reserve(data.len());
        let resource = self.memory_management.get(&handle.memory);
        self.device
            .htod_copy_into(data.to_vec(), resource.buffer)
            .unwrap();

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        server::Handle::new(self.memory_management.reserve(size))
    }

    fn execute(&mut self, kernel: Self::Kernel, handles: &[&server::Handle<Self>]) {
        let kernel_id = kernel.id();

        if !self.module_names.contains_key(&kernel_id) {
            let name = format!("m{}", self.module_names.len());
            let kernel = kernel.compile();
            let source = kernel.source;
            let workgroup_size = kernel.workgroup_size;
            println!("Source {source}");
            let kernel = compile_ptx(source).unwrap();

            self.device.load_ptx(kernel, &name, &["kernel"]).unwrap();
            self.module_names
                .insert(kernel_id.clone(), (name, workgroup_size));
        }

        let task = ComputeTask::new(
            kernel_id,
            kernel.launch_settings().workgroup,
            handles
                .iter()
                .map(|h| self.memory_management.get(&h.memory).as_binding())
                .collect(),
        );

        self.execute_task(task);
        // self.free_manual_allocations();
        // self.memory_management.storage().perform_deallocations();
        self.sync();
    }

    fn sync(&mut self) {
        self.device.synchronize().unwrap();
    }
}

#[derive(new, Debug)]
struct ComputeTask {
    kernel_id: String,
    workgroup: WorkGroup,
    bindings: Vec<Binding>,
}

impl<MM: MemoryManagement<CudaStorage>> CudaServer<MM> {
    pub fn new(device: Arc<CudaDevice>, memory_management: MM) -> Self {
        Self {
            memory_management,
            device,
            manual_available: HashMap::new(),
            manual_taken: Vec::new(),
            module_names: HashMap::new(),
        }
    }

    fn manual_reserve(&mut self, size: usize) -> server::Handle<Self> {
        let handle = self
            .manual_available
            .get_mut(&size)
            .and_then(|h| h.pop())
            .unwrap_or_else(|| {
                let memory = self.memory_management.alloc(size);
                server::Handle::new(memory)
            });

        self.manual_taken.push((size, handle.clone()));

        handle
    }

    fn free_manual_allocations(&mut self) {
        let mut manual_taken_tmp = Vec::new();
        core::mem::swap(&mut manual_taken_tmp, &mut self.manual_taken);

        for (size, handle) in manual_taken_tmp.drain(..) {
            if handle.can_mut() {
                self.register_manual(size, handle);
            } else {
                self.manual_taken.push((size, handle));
            }
        }
    }

    fn register_manual(&mut self, size: usize, handle: server::Handle<Self>) {
        if let Some(handles) = self.manual_available.get_mut(&size) {
            handles.push(handle);
        } else {
            self.manual_available.insert(size, [handle].into());
        }
    }

    fn execute_task(&mut self, task: ComputeTask) {
        let workgroup = task.workgroup;
        let (module_name, grid) = self.module_names.get(&task.kernel_id).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (workgroup.x, workgroup.y, workgroup.z),
            block_dim: (grid.x, grid.y, grid.z),
            shared_mem_bytes: 0,
        };

        unsafe {
            let func = self.device.get_func(module_name, "kernel").unwrap();
            let bindings = task.bindings;
            //println!("Execute task {module_name} - {bindings:?}");

            match bindings.len() {
                1 => func.launch(cfg, (bindings[0],)),
                2 => func.launch(cfg, (bindings[0], bindings[1])),
                3 => func.launch(cfg, (bindings[0], bindings[1], bindings[2])),
                4 => func.launch(cfg, (bindings[0], bindings[1], bindings[2], bindings[3])),
                5 => func.launch(
                    cfg,
                    (
                        bindings[0],
                        bindings[1],
                        bindings[2],
                        bindings[3],
                        bindings[4],
                    ),
                ),
                6 => func.launch(
                    cfg,
                    (
                        bindings[0],
                        bindings[1],
                        bindings[2],
                        bindings[3],
                        bindings[4],
                        bindings[5],
                    ),
                ),
                7 => func.launch(
                    cfg,
                    (
                        bindings[0],
                        bindings[1],
                        bindings[2],
                        bindings[3],
                        bindings[4],
                        bindings[5],
                        bindings[6],
                    ),
                ),
                8 => func.launch(
                    cfg,
                    (
                        bindings[0],
                        bindings[1],
                        bindings[2],
                        bindings[3],
                        bindings[4],
                        bindings[5],
                        bindings[6],
                        bindings[7],
                    ),
                ),
                9 => func.launch(
                    cfg,
                    (
                        bindings[0],
                        bindings[1],
                        bindings[2],
                        bindings[3],
                        bindings[4],
                        bindings[5],
                        bindings[6],
                        bindings[7],
                        bindings[8],
                    ),
                ),
                _ => panic!(),
            }
            .unwrap();
        }
    }
}
