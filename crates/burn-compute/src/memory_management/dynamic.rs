use super::memory_pool::{
    MemoryExtensionStrategy, MemoryPool, MemoryPoolBinding, MemoryPoolHandle, RoundingStrategy,
    SmallMemoryPool,
};
use crate::storage::ComputeStorage;
use alloc::vec::Vec;

use super::MemoryManagement;

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct DynamicMemoryManagement<Storage> {
    min_chunk_alignment_offset: usize,
    small_memory_pool: SmallMemoryPool,
    pools: Vec<MemoryPool>,
    options: Vec<MemoryPoolOptions>,
    storage: Storage,
}

/// Options to initialize a [dynamic memory management](DynamicMemoryManagement).
#[derive(new, Debug)]
pub struct DynamicMemoryManagementOptions {
    pools: Vec<MemoryPoolOptions>,
    min_chunk_alignment_offset: usize,
}

/// Options to create a memory pool.
#[derive(Debug)]
pub struct MemoryPoolOptions {
    /// The amount of bytes used for each chunk in the memory pool.
    pub chunk_size: usize,
    /// The number of chunks allocated directly at creation.
    ///
    /// Useful when you know in advance how much memory you'll need.
    pub chunk_num_prealloc: usize,
    /// The max size in bytes a slice can take in the pool.
    pub slice_max_size: usize,
}

impl DynamicMemoryManagementOptions {
    /// Creates the options from device limits.
    pub fn preset(max_chunk_size: usize, min_chunk_alignment_offset: usize) -> Self {
        // Rounding down to a factor of 8.
        let max_chunk_size = (max_chunk_size / 8) * 8;

        const MB: usize = 1024 * 1024;

        let mut pools = Vec::new();

        pools.push(MemoryPoolOptions {
            chunk_size: max_chunk_size,
            chunk_num_prealloc: 0,
            slice_max_size: max_chunk_size,
        });

        let mut current = max_chunk_size;

        while current >= 32 * MB {
            current /= 4;

            pools.push(MemoryPoolOptions {
                chunk_size: current,
                chunk_num_prealloc: 0,
                // Creating max slices lower than the chunk size reduces fragmentation.
                slice_max_size: current / 2usize.pow(pools.len() as u32),
            });
        }

        Self {
            pools,
            min_chunk_alignment_offset,
        }
    }
}

impl<Storage: ComputeStorage> DynamicMemoryManagement<Storage> {
    /// Creates a new instance using the given storage, merging_strategy strategy and slice strategy.
    pub fn new(mut storage: Storage, mut options: DynamicMemoryManagementOptions) -> Self {
        options
            .pools
            .sort_by(|pool1, pool2| usize::cmp(&pool1.slice_max_size, &pool2.slice_max_size));

        let min_chunk_alignment_offset = options.min_chunk_alignment_offset;

        let pools = options
            .pools
            .iter()
            .map(|option| {
                let mut pool = MemoryPool::new(
                    MemoryExtensionStrategy::Never,
                    RoundingStrategy::FixedAmount(option.chunk_size),
                    min_chunk_alignment_offset,
                );

                for _ in 0..option.chunk_num_prealloc {
                    pool.alloc(&mut storage, option.chunk_size, || {});
                }

                pool
            })
            .collect();

        Self {
            min_chunk_alignment_offset,
            small_memory_pool: SmallMemoryPool::new(min_chunk_alignment_offset),
            pools,
            options: options.pools,
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

        for pool in &mut self.pools {
            if let Some(handle) = pool.get(&mut self.storage, &binding) {
                return handle;
            }
        }

        panic!("No handle found in memory pools");
    }

    fn reserve<Sync: FnOnce()>(&mut self, size: usize, sync: Sync) -> Self::Handle {
        if size <= self.min_chunk_alignment_offset {
            return self
                .small_memory_pool
                .reserve(&mut self.storage, size, sync);
        }

        for (index, option) in self.options.iter().enumerate() {
            if size <= option.slice_max_size {
                let pool = &mut self.pools[index];
                return pool.reserve(&mut self.storage, size, sync);
            }
        }

        panic!("No memory pool big enough to reserve {size} bytes.");
    }

    fn alloc<Sync: FnOnce()>(&mut self, size: usize, sync: Sync) -> Self::Handle {
        if size <= self.min_chunk_alignment_offset {
            return self.small_memory_pool.alloc(&mut self.storage, size, sync);
        }

        for (index, option) in self.options.iter().enumerate() {
            if size <= option.slice_max_size {
                let pool = &mut self.pools[index];
                return pool.alloc(&mut self.storage, size, sync);
            }
        }

        panic!("No memory pool big enough to alloc {size} bytes.");
    }

    fn dealloc(&mut self, _binding: Self::Binding) {
        // Can't dealloc slices.
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }
}
