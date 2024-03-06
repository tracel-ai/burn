use super::storage::Binding;
use std::{collections::HashMap, sync::Arc};

use super::storage::CudaStorage;
use burn_compute::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use burn_jit::compute::{JitAutotuneKey, Kernel, WorkGroup};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::{driver::CudaDevice, nvrtc::compile_ptx};

#[derive(Debug)]
pub struct CudaServer<MM: MemoryManagement<CudaStorage>> {
    memory_management: MM,
    device: Arc<CudaDevice>,
    manual_available: HashMap<usize, Vec<server::Handle<Self>>>,
    manual_taken: Vec<(usize, server::Handle<Self>)>,
    module_names: HashMap<String, String>,
}

#[derive(new, Debug)]
struct ComputeTask {
    kernel_id: String,
    workgroup: WorkGroup,
    bindings: Vec<Binding>,
    grid: (u32, u32, u32),
}

impl<MM: MemoryManagement<CudaStorage>> CudaServer<MM> {
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

    fn execute_task(&mut self, task: ComputeTask) {
        let workgroup = task.workgroup;
        let cfg = LaunchConfig {
            block_dim: (workgroup.x, workgroup.y, workgroup.z),
            grid_dim: task.grid,
            shared_mem_bytes: 0,
        };
        let module_name = self.module_names.get(&task.kernel_id).unwrap();

        unsafe {
            let func = self.device.get_func(module_name, "main").unwrap();
            let bindings = task.bindings;

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
                        bindings[1],
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
                        bindings[1],
                        bindings[1],
                        bindings[2],
                        bindings[3],
                        bindings[4],
                        bindings[5],
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

impl<MM: MemoryManagement<CudaStorage>> ComputeServer for CudaServer<MM> {
    type Kernel = Box<dyn Kernel>;
    type Storage = CudaStorage;
    type MemoryManagement = MM;
    type AutotuneKey = JitAutotuneKey;

    fn read(&mut self, handle: &server::Handle<Self>) -> burn_tensor::Reader<Vec<u8>> {
        // TODO: Sync.
        let resource = self.memory_management.get(&handle.memory);
        let data = self.device.dtoh_sync_copy(&resource.view()).unwrap();
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
            let name = format!("{}", self.module_names.len());
            let kernel = compile_ptx(kernel.source().complete()).unwrap();

            self.device.load_ptx(kernel, &name, &["main"]).unwrap();
            self.module_names.insert(kernel_id.clone(), name);
        }

        let task = ComputeTask::new(
            kernel_id,
            kernel.workgroup(),
            handles
                .iter()
                .map(|h| self.memory_management.get(&h.memory).as_binding())
                .collect(),
            (32, 32, 1),
        );

        self.execute_task(task);
    }

    fn sync(&mut self) {
        todo!()
    }
}
