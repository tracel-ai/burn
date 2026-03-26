use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    sync::{Arc, atomic::AtomicBool},
};

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
    /// The arc here is only to have a stable pointer, since the buffer can be drop when the thread
    /// is done, but some bytes might still live, so the Arc handles that gracefully.
    buffer: Vec<Arc<UnsafeCell<Bytes<MAX_ITEM_SIZE>>>>,
    alive: Option<Arc<AtomicBool>>,
    cursor: usize,
}

/// The initialized reserved memory.
pub struct ReservedMemory<const MAX_ITEM_SIZE: usize> {
    data: Arc<UnsafeCell<Bytes<MAX_ITEM_SIZE>>>,
    alive: Arc<AtomicBool>,
    drop_fn: fn(&mut Bytes<MAX_ITEM_SIZE>),
}

/// The uninitialized reserved memory.
///
/// This type isn't Send/Sync and should be initialized on the same thread as it was reserved.
pub struct UninitReservedMemory<const MAX_ITEM_SIZE: usize> {
    data: Arc<UnsafeCell<Bytes<MAX_ITEM_SIZE>>>,
    alive: Arc<AtomicBool>,
    /// Used to assert the position in the arena.
    #[cfg(test)]
    index: usize,
    // Add this type to make sure the object is `!Sync`.
    not_sync: PhantomData<*const ()>,
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
        // arena.
        assert_eq!(
            Arc::strong_count(&self.data),
            2,
            "We can only initialize reserved memory when there is a single writer."
        );

        let bytes_mut = unsafe { self.data.as_ref().get().as_mut().unwrap() };
        init_data(bytes_mut);

        ReservedMemory {
            data: self.data,
            alive: self.alive,
            drop_fn,
        }
    }
}

impl<const MAX_ITEM_SIZE: usize> core::fmt::Debug for ReservedMemory<MAX_ITEM_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReservedMemory")
            .field("data", &self.data)
            .field("drop_fn", &self.drop_fn)
            .finish()
    }
}

impl<const MAX_ITEM_SIZE: usize> Clone for ReservedMemory<MAX_ITEM_SIZE> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            alive: self.alive.clone(),
            drop_fn: self.drop_fn,
        }
    }
}

impl<const MAX_ITEM_SIZE: usize> Drop for ReservedMemory<MAX_ITEM_SIZE> {
    fn drop(&mut self) {
        // We take the strong count BEFORE we take the alive atomic.
        let drop_fn = || {
            // SAFETY: We are the last user of this slot. The data pointer is valid,
            // initialized, and no other `ReservedMemory` clone exists.
            let bytes_mut = unsafe { self.data.get().as_mut().unwrap() };
            (self.drop_fn)(bytes_mut);
        };

        if self.alive.load(std::sync::atomic::Ordering::Acquire) {
            if Arc::strong_count(&self.data) == 2 {
                drop_fn();
            }
        } else {
            if Arc::strong_count(&self.data) == 1 {
                drop_fn();
            }
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
        unsafe { self.data.as_ref().get().as_ref().unwrap() }
    }
}

impl<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE> {
    /// Creates a new, empty `Arena`.
    ///
    /// The internal buffer is not allocated until the first call to [`reserve`](Self::reserve).
    pub const fn new() -> Self {
        Self {
            buffer: Vec::new(),
            alive: None,
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
                self.buffer.push(Arc::new(UnsafeCell::new(Bytes {
                    bytes: [0; MAX_ITEM_SIZE],
                })));
            }
            self.alive = Some(Arc::new(AtomicBool::new(true)));
        }

        for i in 0..MAX_ITEM_COUNT {
            let i = (i + self.cursor) % MAX_ITEM_COUNT;
            let item = &self.buffer[i];

            if Arc::strong_count(item) == 1 {
                self.cursor = (i + 1) % MAX_ITEM_COUNT;
                let data = item.clone();

                return Some(UninitReservedMemory {
                    data,
                    alive: self.alive.as_ref().unwrap().clone(),
                    #[cfg(test)]
                    index: i,
                    not_sync: PhantomData,
                });
            }
        }

        None
    }
}

impl<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> Drop
    for Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE>
{
    fn drop(&mut self) {
        // We start by dropping the buffers.
        self.buffer.clear();

        // Then we set the alive boolean.
        self.alive
            .as_ref()
            .unwrap()
            .store(false, std::sync::atomic::Ordering::Release);
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

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::sync::{Arc, Barrier, Mutex};
    use std::thread;

    const MAX_ITEM_SIZE: usize = 2048;

    /// Wraps an arena in a Mutex for shared cross-thread access.
    fn shared_arena<const N: usize>() -> Arc<Mutex<Arena<N, MAX_ITEM_SIZE>>> {
        Arc::new(Mutex::new(Arena::<N, MAX_ITEM_SIZE>::new()))
    }

    /// Verifies that drop_fn is called exactly once even when multiple threads
    /// hold clones and release them concurrently.
    #[test]
    fn test_drop_called_exactly_once_under_contention() {
        let drop_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let arena = shared_arena::<4>();

        let uninit = arena.lock().unwrap().reserve().unwrap();

        struct Probe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let reserved = uninit.init(Probe(drop_count.clone()));

        // Spawn 32 threads, each clones and drops ReservedMemory concurrently.
        let barrier = Arc::new(Barrier::new(32));
        let mut handles = vec![];

        for _ in 0..32 {
            let r = reserved.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait(); // all threads drop at the same time
                drop(r);
            }));
        }

        drop(reserved); // drop the original too
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            drop_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "drop_fn must be called exactly once"
        );
    }

    /// Verifies that a slot becomes available for reuse after all ReservedMemory
    /// clones are dropped across threads.
    #[test]
    fn test_slot_reuse_after_concurrent_drop() {
        let arena = shared_arena::<1>();
        let uninit = arena.lock().unwrap().reserve().unwrap();
        let reserved = uninit.init(42u64);

        let barrier = Arc::new(Barrier::new(8));
        let mut handles = vec![];

        for _ in 0..8 {
            let r = reserved.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                drop(r);
            }));
        }

        drop(reserved);
        for h in handles {
            h.join().unwrap();
        }

        // All clones dropped — the single slot should be free again.
        assert!(
            arena.lock().unwrap().reserve().is_some(),
            "Slot should be available after all clones are dropped"
        );
    }

    /// Verifies that ReservedMemory clones dropped after the arena is dropped
    /// still correctly run drop_fn (the count == 1 case).
    #[test]
    fn test_drop_after_arena_dropped() {
        let drop_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        struct Probe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let reserved = {
            let mut arena = Arena::<4, MAX_ITEM_SIZE>::new();
            let uninit = arena.reserve().unwrap();
            uninit.init(Probe(drop_count.clone()))
            // arena drops here
        };

        // Spawn threads that hold clones past the arena's lifetime.
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = vec![];

        for _ in 0..8 {
            let r = reserved.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                drop(r);
            }));
        }

        drop(reserved);
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            drop_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "drop_fn must fire exactly once even when arena is dropped first"
        );
    }
}
