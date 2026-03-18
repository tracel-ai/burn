use std::{
    cell::RefCell,
    sync::atomic::{AtomicU32, Ordering},
};

const MAX_SIZE: usize = 2048;
const MAX_ITEM: usize = 256;

type Bytes = [u128; MAX_SIZE / 16];

pub struct Item {
    pub bytes: Bytes,
    pub count: AtomicU32,
}

pub struct Arena<const SIZE: usize> {
    buffer: Vec<Item>,
    cursor: usize,
    size: usize,
    init: bool,
}

impl Arena {
    pub const fn new(size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            size,
            init: false,
        }
    }

    pub fn reserve_data(&mut self) -> Option<(usize, *mut Item)> {
        if !self.init {
            for _ in 0..self.size {
                self.buffer.push(Item {
                    bytes: [0; MAX_SIZE / 16],
                    count: AtomicU32::new(0),
                });
            }
            self.init = true;
        }

        for i in 0..self.size {
            let i = (i + self.cursor) % self.size;
            let data = &mut self.buffer[i];

            if data.count.load(Ordering::Relaxed) == 0 {
                self.cursor = i;
                // We start with a ref count of 2, so a ref count of 1 mean the drop must be
                // executed and a ref count of 0 means free for reuse.
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

    #[test]
    fn test_lazy_initialization() {
        let mut arena = Arena::new(10);
        assert_eq!(
            arena.buffer.len(),
            0,
            "Buffer should be empty before first reservation"
        );

        arena.reserve_data();

        assert_eq!(
            arena.buffer.len(),
            10,
            "Buffer should be initialized to size"
        );
        assert!(arena.init);
    }

    #[test]
    fn test_sequential_allocation_moves_cursor() {
        let mut arena = Arena::new(3);

        // First allocation
        let (idx1, _) = arena.reserve_data().expect("Should allocate");
        assert_eq!(idx1, 0);
        assert_eq!(arena.cursor, 0);

        // Second allocation
        let (idx2, _) = arena.reserve_data().expect("Should allocate");
        assert_eq!(idx2, 1);
        assert_eq!(arena.cursor, 1);
    }

    #[test]
    fn test_reuse_of_freed_data() {
        let mut arena = Arena::new(2);

        // Fill the arena
        let (_idx0, data0) = arena.reserve_data().unwrap();
        let (_idx1, _data1) = arena.reserve_data().unwrap();

        // Arena is now full (counts are 2)
        assert!(arena.reserve_data().is_none(), "Should be full");

        // Manually "free" index 0 by setting count to 0 (simulating ManagedOperation drop)
        unsafe { data0.as_ref().unwrap().count.store(0, Ordering::Relaxed) };

        // Should now be able to reserve again, and it should pick up index 0
        let (new_idx, _) = arena.reserve_data().expect("Should reuse index 0");
        assert_eq!(new_idx, 0);
    }

    #[test]
    fn test_circular_cursor_search() {
        let mut arena = Arena::new(3);

        // Fill 0, 1, 2
        let (_, _d0) = arena.reserve_data().unwrap();
        let (_, d1) = arena.reserve_data().unwrap();
        let (_, _d2) = arena.reserve_data().unwrap();

        // Free index 1 (the middle)
        unsafe { d1.as_ref().unwrap().count.store(0, Ordering::Relaxed) };

        // Currently cursor is at 2. The search starts at (cursor + i) % size.
        // It should wrap around and find index 1.
        let (reused_idx, _) = arena
            .reserve_data()
            .expect("Should find the hole at index 1");
        assert_eq!(reused_idx, 1);
        assert_eq!(arena.cursor, 1);
    }

    #[test]
    fn test_full_arena_returns_none() {
        let size = 5;
        let mut arena = Arena::new(size);

        for _ in 0..size {
            assert!(arena.reserve_data().is_some());
        }

        // Next one should fail
        assert!(arena.reserve_data().is_none());
    }
}
