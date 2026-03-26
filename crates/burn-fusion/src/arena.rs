use std::{cell::UnsafeCell, sync::Arc};

/// The raw storage for the item, potentially uninitialized.
#[repr(C, align(64))]
pub struct Bytes<const MAX_ITEM_SIZE: usize> {
    bytes: [u8; MAX_ITEM_SIZE],
}

/// A circular, allocation arena for reusable memory blocks.
///
/// The `Arena` manages a fixed-capacity pool of [`Bytes`]. It uses a cursor-based
/// search strategy to find and reuse available slots, minimizing allocation overhead
/// after the initial lazy initialization.
///
/// # Notes
///
/// This can be used to replace `Arc<dyn Trait>`.
pub struct Arena<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> {
    buffer: Vec<UnsafeCell<Bytes<MAX_ITEM_SIZE>>>,
    counts: Vec<Arc<()>>,
    cursor: usize,
}

/// The initialized reserved memory.
pub struct ReservedMemory<const MAX_ITEM_SIZE: usize> {
    data: *mut Bytes<MAX_ITEM_SIZE>,
    count: Arc<()>,
    drop_fn: fn(&mut Bytes<MAX_ITEM_SIZE>),
}

/// The uninitialized reserved memory.
///
/// This type isn't Send/Sync and should be initialized on the same thread as it was reserved.
pub struct UninitReservedMemory<const MAX_ITEM_SIZE: usize> {
    data: *mut Bytes<MAX_ITEM_SIZE>,
    count: Arc<()>,
    /// Used to assert the position in the arena.
    #[cfg(test)]
    pub index: usize,
}

impl<const MAX_ITEM_SIZE: usize> UninitReservedMemory<MAX_ITEM_SIZE> {
    /// Initialize the reserved memory.
    ///
    /// # Panics
    ///
    /// If the given object isn't safe to store in this arena.
    pub fn init<O>(self, obj: O) -> ReservedMemory<MAX_ITEM_SIZE> {
        assert!(
            accept_obj::<O, MAX_ITEM_SIZE>(),
            "Object isn't safe to store in this arena"
        );

        self.init_with_func(
            |bytes| {
                let ptr = core::ptr::from_mut(bytes);
                unsafe {
                    core::ptr::write(ptr as *mut O, obj);
                };
            },
            |bytes| {
                let ptr = core::ptr::from_mut(bytes);
                unsafe {
                    core::ptr::drop_in_place(ptr as *mut O);
                }
            },
        )
    }

    /// Initialize the reserved memory.
    fn init_with_func<F>(
        self,
        init_data: F,
        drop_fn: fn(&mut Bytes<MAX_ITEM_SIZE>),
    ) -> ReservedMemory<MAX_ITEM_SIZE>
    where
        F: FnOnce(&mut Bytes<MAX_ITEM_SIZE>),
    {
        // # Safety
        //
        // We read the cell pointer that is only available to the client, not the
        // areana.
        assert_eq!(
            Arc::strong_count(&self.count),
            2,
            "We can only initialize reserved memory when there is a single writer."
        );

        let mut bytes_mut = unsafe { self.data.as_mut().unwrap() };
        init_data(&mut bytes_mut);

        ReservedMemory {
            data: self.data.clone(),
            count: self.count.clone(),
            drop_fn,
        }
    }
}

impl<const MAX_ITEM_SIZE: usize> core::fmt::Debug for ReservedMemory<MAX_ITEM_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReservedMemory")
            .field("data", &self.data)
            .field("count", &self.count)
            .field("drop_fn", &self.drop_fn)
            .finish()
    }
}

impl<const MAX_ITEM_SIZE: usize> Clone for ReservedMemory<MAX_ITEM_SIZE> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            count: self.count.clone(),
            drop_fn: self.drop_fn.clone(),
        }
    }
}

impl<const MAX_ITEM_SIZE: usize> Drop for ReservedMemory<MAX_ITEM_SIZE> {
    fn drop(&mut self) {
        if Arc::strong_count(&self.count) == 2 {
            // # Safety
            //
            // We read the cell pointer that is only available to the client, not the
            // areana.
            let mut bytes_mut = unsafe { self.data.as_mut().unwrap() };
            (self.drop_fn)(&mut bytes_mut)
        }
    }
}

/// The reserved data is readonly and protected with ref counting.
unsafe impl<const MAX_ITEM_SIZE: usize> Send for ReservedMemory<MAX_ITEM_SIZE> {}
/// The reserved data is readonly and protected with ref counting.
unsafe impl<const MAX_ITEM_SIZE: usize> Sync for ReservedMemory<MAX_ITEM_SIZE> {}

impl<const MAX_ITEM_SIZE: usize> ReservedMemory<MAX_ITEM_SIZE> {
    /// Gets the reserved bytes.
    pub fn as_ref(&self) -> &Bytes<MAX_ITEM_SIZE> {
        // The pointer is valid and the data is readonly.
        unsafe { self.data.as_ref().unwrap() }
    }
}

impl<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE> {
    /// Creates a new, empty `Arena`.
    ///
    /// The internal buffer is not allocated until the first call to [`reserve`](Self::reserve).
    pub const fn new() -> Self {
        Self {
            buffer: Vec::new(),
            counts: Vec::new(),
            cursor: 0,
        }
    }

    pub const fn accept<O>() -> bool {
        accept_obj::<O, MAX_ITEM_SIZE>()
    }

    /// Attempts to reserve an available item in the arena.
    ///
    /// If the arena is empty, it lazily initializes the buffer to `MAX_ITEM_COUNT`.
    /// It searches starting from the current `cursor` position for an item with a
    /// `count` of 0.
    ///
    /// The drop function is only called when the reserved memory is initialized.
    ///
    /// # Returns
    /// - `Some((index, *mut Item))`: The index and a raw pointer to the reserved item.
    ///   The item's `count` is automatically set to `2`.
    /// - `None`: If all items in the arena are currently reserved or active.
    pub fn reserve(&mut self) -> Option<UninitReservedMemory<MAX_ITEM_SIZE>> {
        if self.buffer.is_empty() {
            for _ in 0..MAX_ITEM_COUNT {
                self.buffer.push(UnsafeCell::new(Bytes {
                    bytes: [0; MAX_ITEM_SIZE],
                }));
                self.counts.push(Arc::new(()));
            }
        }

        for i in 0..MAX_ITEM_COUNT {
            let i = (i + self.cursor) % MAX_ITEM_COUNT;
            let count = &self.counts[i];

            if Arc::strong_count(count) == 1 {
                self.cursor = (i + 1) % MAX_ITEM_COUNT;
                let count = count.clone();

                let bytes = self.buffer[i].get();

                return Some(UninitReservedMemory {
                    data: bytes,
                    count,
                    #[cfg(test)]
                    index: i,
                });
            }
        }

        None
    }
}

const fn accept_obj<O, const MAX_ITEM_SIZE: usize>() -> bool {
    size_of::<O>() <= size_of::<Bytes<MAX_ITEM_SIZE>>()
        && align_of::<O>() <= align_of::<Bytes<MAX_ITEM_SIZE>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_ITEM_SIZE: usize = 2048;

    #[test]
    fn test_lazy_initialization() {
        let mut arena = Arena::<10, MAX_ITEM_SIZE>::new();
        assert_eq!(
            arena.buffer.len(),
            0,
            "Buffer should be empty before first reservation"
        );

        arena.reserve();

        assert_eq!(
            arena.buffer.len(),
            10,
            "Buffer should be initialized to size"
        );
    }

    #[test]
    fn test_sequential_allocation_moves_cursor() {
        let mut arena = Arena::<3, MAX_ITEM_SIZE>::new();

        // First allocation
        let _ = arena.reserve().expect("Should allocate");
        assert_eq!(arena.cursor, 1);

        // Second allocation
        let _ = arena.reserve().expect("Should allocate");
        assert_eq!(arena.cursor, 2);
    }

    #[test]
    fn test_reuse_of_freed_data() {
        let mut arena = Arena::<2, MAX_ITEM_SIZE>::new();

        // Fill the arena
        let data0 = arena.reserve().unwrap();
        let _data1 = arena.reserve().unwrap();

        // Arena is now full (counts are 2)
        assert!(arena.reserve().is_none(), "Should be full");

        // Manually "free" index 0 by setting count to 0 (simulating ManagedOperation drop)
        let data0_index = data0.index;
        core::mem::drop(data0);

        // Should now be able to reserve again, and it should pick up index 0
        let data2 = arena.reserve().expect("Should reuse index 0");
        assert_eq!(data0_index, data2.index);
    }

    #[test]
    fn test_circular_cursor_search() {
        let mut arena = Arena::<3, MAX_ITEM_SIZE>::new();

        // Fill 0, 1, 2
        let _d0 = arena.reserve().unwrap();
        let d1 = arena.reserve().unwrap();
        let _d2 = arena.reserve().unwrap();

        // Free index 1 (the middle)
        core::mem::drop(d1);

        // Currently cursor is at 2. The search starts at (cursor + i) % size.
        // It should wrap around and find index 1.
        let _ = arena.reserve().expect("Should find the hole at index 1");
        assert_eq!(arena.cursor, 2);
    }

    #[test]
    fn test_full_arena_returns_none() {
        let mut arena = Arena::<5, MAX_ITEM_SIZE>::new();

        let mut reserved = Vec::new();

        for _ in 0..5 {
            let item = arena.reserve();
            assert!(item.is_some());
            reserved.push(item);
        }

        // Next one should fail
        assert!(arena.reserve().is_none());
    }
}
