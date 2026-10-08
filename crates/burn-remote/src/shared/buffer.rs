use std::{
    collections::{TryReserveError, VecDeque},
    mem,
    ops::{Deref, DerefMut},
    sync::{Mutex, MutexGuard, PoisonError},
};

use bytes::Bytes;

use crate::transport::link::MAX_FRAME_SIZE;

/// One pool for the whole process, so a buffer one connection frees serves the next message on any.
pub static BUFFERS: BufferPool = BufferPool::new();

/// Buffers for large messages, kept once their bytes are dropped for the next message to reuse:
/// handing megabytes back to the allocator per message lets glibc trim them to the OS, and the
/// next message then faults every page back in.
pub struct BufferPool {
    kept: Mutex<KeptBuffers>,
}

impl BufferPool {
    /// The most bytes kept, counted by capacity; a buffer larger than this is freed, so a rare huge
    /// message is not held for the life of the process.
    const MAX_KEPT: usize = 32 * 1024 * 1024;

    /// The bytes the pool hands out before a buffer kept all along is stale, and may make way for
    /// one returned to a full pool: about what a round trip of a message as large as the pool hands
    /// out, so buffers every round trip reuses are taken again before they go stale.
    const STALE_AFTER: u64 = 4 * Self::MAX_KEPT as u64;

    const fn new() -> Self {
        Self {
            kept: Mutex::new(KeptBuffers::new()),
        }
    }

    /// An empty buffer for `capacity` bytes, for an encoding to be written into.
    pub fn take(&'static self, capacity: usize) -> Result<PooledBuffer, TryReserveError> {
        let mut buffer = self.take_to_overwrite(capacity)?;
        buffer.clear();
        Ok(buffer)
    }

    /// A buffer for `capacity` bytes still holding what it last held, for a read that overwrites
    /// every byte it keeps: lengthening it zeroes only the bytes past those.
    pub fn take_to_overwrite(
        &'static self,
        capacity: usize,
    ) -> Result<PooledBuffer, TryReserveError> {
        let class = Self::size_class(capacity);
        // Its own statement, so the lock is released before a new buffer is allocated.
        let kept = if class <= Self::MAX_KEPT {
            self.lock().take(class)
        } else {
            None
        };
        let buffer = match kept {
            Some(buffer) => buffer,
            None => {
                let mut buffer = Vec::new();
                buffer.try_reserve_exact(class)?;
                buffer
            }
        };
        Ok(PooledBuffer { buffer, pool: self })
    }

    /// The capacity a buffer for `len` bytes has, so that nearby lengths share buffers: a power of
    /// two up to a frame, a multiple of a frame past it.
    fn size_class(len: usize) -> usize {
        if len <= MAX_FRAME_SIZE {
            return len.next_power_of_two();
        }
        // A length too large to round up is refused by the allocation all the same.
        len.checked_next_multiple_of(MAX_FRAME_SIZE).unwrap_or(len)
    }

    fn put_back(&self, buffer: Vec<u8>) {
        if buffer.capacity() > Self::MAX_KEPT {
            return;
        }
        let freed = self.lock().put_back(buffer);
        // Freed once the lock is released: unmapping megabytes is too slow to hold it through.
        drop(freed);
    }

    fn lock(&self) -> MutexGuard<'_, KeptBuffers> {
        self.kept.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The buffers a pool holds, oldest first, their capacity in total, and the bytes handed out so
/// far, the clock a kept buffer's idleness is measured by.
struct KeptBuffers {
    buffers: VecDeque<KeptBuffer>,
    bytes: usize,
    handed_out: u64,
}

/// A buffer in a pool, and the bytes the pool had handed out when it was put back.
struct KeptBuffer {
    buffer: Vec<u8>,
    kept_at: u64,
}

impl KeptBuffers {
    const fn new() -> Self {
        Self {
            buffers: VecDeque::new(),
            bytes: 0,
            handed_out: 0,
        }
    }

    /// The newest buffer of exactly `capacity`: a larger one would be held by a smaller message,
    /// such as a send segment QUIC keeps until it is acknowledged.
    fn take(&mut self, capacity: usize) -> Option<Vec<u8>> {
        self.handed_out += capacity as u64;
        let index = self
            .buffers
            .iter()
            .rposition(|kept| kept.buffer.capacity() == capacity)?;
        let kept = self.buffers.remove(index)?;
        self.bytes -= kept.buffer.capacity();
        Some(kept.buffer)
    }

    /// Keeps `buffer` once stale buffers make room for it, and returns the buffers to free: `buffer`
    /// itself when buffers still in use fill the pool, since evicting one of those to keep it would
    /// only free a buffer the next messages take.
    fn put_back(&mut self, buffer: Vec<u8>) -> Vec<Vec<u8>> {
        let mut freed = Vec::new();
        while !self.has_room_for(buffer.capacity())
            && let Some(stale) = self.evict_stale()
        {
            freed.push(stale);
        }
        if self.has_room_for(buffer.capacity()) {
            self.bytes += buffer.capacity();
            self.buffers.push_back(KeptBuffer {
                buffer,
                kept_at: self.handed_out,
            });
        } else {
            freed.push(buffer);
        }
        freed
    }

    fn has_room_for(&self, capacity: usize) -> bool {
        self.bytes + capacity <= BufferPool::MAX_KEPT
    }

    /// The oldest buffer, removed if it is stale.
    fn evict_stale(&mut self) -> Option<Vec<u8>> {
        let oldest = self.buffers.front()?;
        if self.handed_out - oldest.kept_at <= BufferPool::STALE_AFTER {
            return None;
        }
        let oldest = self.buffers.pop_front()?;
        self.bytes -= oldest.buffer.capacity();
        Some(oldest.buffer)
    }
}

/// A buffer taken from a [`BufferPool`], put back when dropped.
pub struct PooledBuffer {
    buffer: Vec<u8>,
    pool: &'static BufferPool,
}

impl PooledBuffer {
    /// The bytes written so far, sharing this buffer: it goes back to its pool once the last of
    /// them is dropped.
    pub fn into_bytes(self) -> Bytes {
        Bytes::from_owner(self)
    }
}

impl Deref for PooledBuffer {
    type Target = Vec<u8>;

    fn deref(&self) -> &Vec<u8> {
        &self.buffer
    }
}

impl DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        self.pool.put_back(mem::take(&mut self.buffer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_is_reused_once_its_bytes_are_dropped() {
        static POOL: BufferPool = BufferPool::new();
        let mut buffer = POOL.take(1024).unwrap();
        buffer.extend_from_slice(b"message");
        let address = buffer.as_ptr();

        let bytes = buffer.into_bytes();
        assert_eq!(&bytes[..], b"message");
        drop(bytes);

        let reused = POOL.take(1000).unwrap();
        assert_eq!(reused.as_ptr(), address);
        assert!(reused.is_empty());
    }

    #[test]
    fn lengths_within_one_size_class_share_a_buffer() {
        static POOL: BufferPool = BufferPool::new();
        let buffer = POOL.take(MAX_FRAME_SIZE + 1).unwrap();
        let address = buffer.as_ptr();
        drop(buffer);

        assert_eq!(POOL.take(2 * MAX_FRAME_SIZE).unwrap().as_ptr(), address);
    }

    #[test]
    fn a_kept_buffer_of_a_larger_class_is_not_taken() {
        static POOL: BufferPool = BufferPool::new();
        let large = POOL.take(2 * MAX_FRAME_SIZE).unwrap();
        let large_address = large.as_ptr();
        drop(large);

        let segment = POOL.take(MAX_FRAME_SIZE).unwrap();

        assert_ne!(segment.as_ptr(), large_address);
        assert_eq!(segment.capacity(), MAX_FRAME_SIZE);
    }

    #[test]
    fn a_buffer_past_what_the_pool_keeps_is_freed() {
        static POOL: BufferPool = BufferPool::new();
        drop(POOL.take(BufferPool::MAX_KEPT + 1).unwrap());

        assert!(POOL.lock().buffers.is_empty());
    }

    #[test]
    fn the_cap_counts_every_kept_buffer_by_capacity() {
        static POOL: BufferPool = BufferPool::new();
        let first = POOL.take(BufferPool::MAX_KEPT / 2 + 1).unwrap();
        let second = POOL.take(BufferPool::MAX_KEPT / 2 + 1).unwrap();
        let first_address = first.as_ptr();

        drop(first);
        drop(second);

        let kept = POOL.lock();
        assert_eq!(kept.buffers.len(), 1);
        assert_eq!(kept.buffers[0].buffer.as_ptr(), first_address);
        assert_eq!(kept.bytes, kept.buffers[0].buffer.capacity());
    }

    /// Fills `pool` with frame-sized buffers, all just put back.
    fn fill_with_segments(pool: &'static BufferPool) {
        let segments: Vec<_> = (0..BufferPool::MAX_KEPT / MAX_FRAME_SIZE)
            .map(|_| pool.take(MAX_FRAME_SIZE).unwrap())
            .collect();
        drop(segments);
    }

    #[test]
    fn a_buffer_returned_to_a_pool_full_of_buffers_in_use_is_freed() {
        static POOL: BufferPool = BufferPool::new();
        fill_with_segments(&POOL);
        let large = POOL.take(4 * MAX_FRAME_SIZE).unwrap();
        let address = large.as_ptr();

        drop(large);

        let kept = POOL.lock();
        assert!(
            kept.buffers
                .iter()
                .all(|segment| segment.buffer.as_ptr() != address)
        );
        assert_eq!(kept.bytes, BufferPool::MAX_KEPT);
    }

    #[test]
    fn a_buffer_returned_to_a_full_pool_evicts_buffers_left_unused() {
        static POOL: BufferPool = BufferPool::new();
        fill_with_segments(&POOL);
        let large = 4 * MAX_FRAME_SIZE;
        for _ in 0..=BufferPool::STALE_AFTER / large as u64 {
            drop(POOL.take(large).unwrap());
        }
        let buffer = POOL.take(large).unwrap();
        let address = buffer.as_ptr();

        drop(buffer);

        assert_eq!(POOL.take(large).unwrap().as_ptr(), address);
        assert!(POOL.lock().bytes <= BufferPool::MAX_KEPT);
    }

    #[test]
    fn bytes_keep_their_buffer_out_of_the_pool_while_any_slice_of_them_lives() {
        static POOL: BufferPool = BufferPool::new();
        let mut buffer = POOL.take(1024).unwrap();
        buffer.extend_from_slice(b"message");
        let address = buffer.as_ptr();

        let slice = buffer.into_bytes().slice(1..4);
        let other = POOL.take(1024).unwrap();

        assert_ne!(other.as_ptr(), address);
        assert_eq!(&slice[..], b"ess");
        drop(slice);
        assert_eq!(POOL.take(1024).unwrap().as_ptr(), address);
    }
}
