use std::{
    mem::MaybeUninit,
    sync::atomic::{AtomicU32, Ordering},
};

/// A fixed-size memory slot with high alignment.
///
/// `Item` provides a raw byte buffer of size `MAX_ITEM_SIZE`.
/// It uses a 512-byte alignment to avoid false sharing and to satisfy
/// strict alignment requirements for specialized hardware or SIMD operations.
#[repr(C, align(512))]
pub struct Item<const MAX_ITEM_SIZE: usize> {
    /// The raw storage for the item, potentially uninitialized.
    pub bytes: [MaybeUninit<u8>; MAX_ITEM_SIZE],
    /// An atomic reference count or state tracker.
    ///
    /// - `0`: The item is free for reuse.
    /// - `1`: The item is pending destruction/cleanup.
    /// - `2+`: The item is currently reserved and active.
    pub count: AtomicU32,
}

/// A circular, lock-free-ish allocation arena for reusable memory blocks.
///
/// The `Arena` manages a fixed-capacity pool of [`Item`]s. It uses a cursor-based
/// search strategy to find and reuse available slots, minimizing allocation overhead
/// after the initial lazy initialization.
pub struct Arena<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> {
    buffer: Vec<Item<MAX_ITEM_SIZE>>,
    cursor: usize,
}

impl<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE> {
    /// Creates a new, empty `Arena`.
    ///
    /// The internal buffer is not allocated until the first call to [`reserve`](Self::reserve).
    pub const fn new() -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
        }
    }

    /// Attempts to reserve an available item in the arena.
    ///
    /// If the arena is empty, it lazily initializes the buffer to `MAX_ITEM_COUNT`.
    /// It searches starting from the current `cursor` position for an item with a
    /// `count` of 0.
    ///
    /// # Returns
    /// - `Some((index, *mut Item))`: The index and a raw pointer to the reserved item.
    ///   The item's `count` is automatically set to `2`.
    /// - `None`: If all items in the arena are currently reserved or active.
    pub fn reserve(&mut self) -> Option<(usize, *mut Item<MAX_ITEM_SIZE>)> {
        if self.buffer.is_empty() {
            for _ in 0..MAX_ITEM_COUNT {
                self.buffer.push(Item {
                    bytes: [MaybeUninit::uninit(); MAX_ITEM_SIZE],
                    count: AtomicU32::new(0),
                });
            }
        }

        for i in 0..MAX_ITEM_COUNT {
            let i = (i + self.cursor) % MAX_ITEM_COUNT;
            let data = &mut self.buffer[i];

            if data.count.load(Ordering::Relaxed) == 0 {
                self.cursor = (i + 1) % MAX_ITEM_COUNT;
                data.count.store(2, Ordering::Relaxed);

                return Some((i, &mut self.buffer[i]));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

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
        let (idx1, _) = arena.reserve().expect("Should allocate");
        assert_eq!(idx1, 0);
        assert_eq!(arena.cursor, 0);

        // Second allocation
        let (idx2, _) = arena.reserve().expect("Should allocate");
        assert_eq!(idx2, 1);
        assert_eq!(arena.cursor, 1);
    }

    #[test]
    fn test_reuse_of_freed_data() {
        let mut arena = Arena::<2, MAX_ITEM_SIZE>::new();

        // Fill the arena
        let (_idx0, data0) = arena.reserve().unwrap();
        let (_idx1, _data1) = arena.reserve().unwrap();

        // Arena is now full (counts are 2)
        assert!(arena.reserve().is_none(), "Should be full");

        // Manually "free" index 0 by setting count to 0 (simulating ManagedOperation drop)
        unsafe { data0.as_ref().unwrap().count.store(0, Ordering::Relaxed) };

        // Should now be able to reserve again, and it should pick up index 0
        let (new_idx, _) = arena.reserve().expect("Should reuse index 0");
        assert_eq!(new_idx, 0);
    }

    #[test]
    fn test_circular_cursor_search() {
        let mut arena = Arena::<3, MAX_ITEM_SIZE>::new();

        // Fill 0, 1, 2
        let (_, _d0) = arena.reserve().unwrap();
        let (_, d1) = arena.reserve().unwrap();
        let (_, _d2) = arena.reserve().unwrap();

        // Free index 1 (the middle)
        unsafe { d1.as_ref().unwrap().count.store(0, Ordering::Relaxed) };

        // Currently cursor is at 2. The search starts at (cursor + i) % size.
        // It should wrap around and find index 1.
        let (reused_idx, _) = arena.reserve().expect("Should find the hole at index 1");
        assert_eq!(reused_idx, 1);
        assert_eq!(arena.cursor, 1);
    }

    #[test]
    fn test_full_arena_returns_none() {
        let mut arena = Arena::<5, MAX_ITEM_SIZE>::new();

        for _ in 0..5 {
            assert!(arena.reserve().is_some());
        }

        // Next one should fail
        assert!(arena.reserve().is_none());
    }
}
