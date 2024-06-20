use super::memory_pool::{
    MemoryExtensionStrategy, MemoryPool, MemoryPoolBinding, MemoryPoolHandle, RoundingStrategy,
};
use crate::storage::ComputeStorage;

use super::MemoryManagement;

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct DynamicMemoryManagement<Storage> {
    small_memory_pool: MemoryPool,
    main_memory_pool: MemoryPool,
    storage: Storage,
}

impl<Storage: ComputeStorage> DynamicMemoryManagement<Storage> {
    /// Creates a new instance using the given storage, merging_strategy strategy and slice strategy.
    pub fn new(storage: Storage) -> Self {
        let main_memory_pool = MemoryPool::new(
            MemoryExtensionStrategy::new_period_tick(10),
            RoundingStrategy::RoundUp,
            1024 * 1024 * 1024 * 2,
            true,
        );
        let small_memory_pool = MemoryPool::new(
            MemoryExtensionStrategy::Never,
            RoundingStrategy::None,
            1024 * 1024 * 512,
            false,
        );

        Self {
            main_memory_pool,
            small_memory_pool,
            storage,
        }
    }
}

impl<Storage> core::fmt::Debug for DynamicMemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "DynamicMemoryManagement {:?}",
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> for DynamicMemoryManagement<Storage> {
    type Handle = MemoryPoolHandle;
    type Binding = MemoryPoolBinding;

    fn get(&mut self, binding: Self::Binding) -> Storage::Resource {
        if let Some(handle) = self.small_memory_pool.get(&mut self.storage, &binding) {
            return handle;
        }

        if let Some(handle) = self.main_memory_pool.get(&mut self.storage, &binding) {
            return handle;
        }

        panic!("No handle found in the small and main memory pool");
    }

    fn reserve<Sync: FnOnce()>(&mut self, size: usize, sync: Sync) -> Self::Handle {
        if size < 512 {
            self.small_memory_pool
                .reserve(&mut self.storage, size, sync)
        } else {
            self.main_memory_pool.reserve(&mut self.storage, size, sync)
        }
    }

    fn alloc<Sync: FnOnce()>(&mut self, size: usize, sync: Sync) -> Self::Handle {
        if size < 512 {
            self.small_memory_pool.alloc(&mut self.storage, size, sync)
        } else {
            self.main_memory_pool.alloc(&mut self.storage, size, sync)
        }
    }

    fn dealloc(&mut self, _binding: Self::Binding) {
        // Can't dealloc slices.
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }
}
