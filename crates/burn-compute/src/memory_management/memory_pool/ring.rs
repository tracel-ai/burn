use alloc::vec::Vec;
use core::marker::PhantomData;
use hashbrown::HashMap;

use super::{ChunkId, SliceId};

#[derive(Debug)]
pub struct RingBuffer<C: MemoryChunk<S>, S: MemorySlice> {
    queue: Vec<ChunkId>,
    chunk_positions: HashMap<ChunkId, usize>,
    cursor_slice: usize,
    cursor_chunk: usize,
    _s: PhantomData<S>,
    _c: PhantomData<C>,
}

pub trait MemoryChunk<S: MemorySlice> {
    fn merge_next_slice(&mut self, slice_position: usize, slices: &mut HashMap<SliceId, S>)
        -> bool;
    fn slice(&self, index: usize) -> Option<SliceId>;
    fn insert_slice(&mut self, position: usize, slice_id: SliceId);
}

pub trait MemorySlice {
    fn is_free(&self) -> bool;
    fn size(&self) -> usize;
    fn split(&mut self, offset: usize) -> Self;
    fn id(&self) -> SliceId;
}

impl<C: MemoryChunk<S>, S: MemorySlice> RingBuffer<C, S> {
    pub fn new() -> Self {
        Self {
            queue: Vec::new(),
            chunk_positions: HashMap::new(),
            cursor_slice: 0,
            cursor_chunk: 0,
            _s: PhantomData,
            _c: PhantomData,
        }
    }

    pub fn push_chunk(&mut self, chunk_id: ChunkId) {
        self.queue.push(chunk_id);
        self.chunk_positions.insert(chunk_id, self.queue.len() - 1);
    }

    pub fn remove_chunk(&mut self, chunk_id: ChunkId) {
        if let Some(position) = self.chunk_positions.remove(&chunk_id) {
            self.queue.remove(position);
        }

        self.chunk_positions.clear();

        for (pos, id) in self.queue.iter().enumerate() {
            self.chunk_positions.insert(*id, pos);
        }
    }

    pub fn find_free_slice(
        &mut self,
        size: usize,
        chunks: &mut HashMap<ChunkId, C>,
        slices: &mut HashMap<SliceId, S>,
    ) -> Option<SliceId> {
        let max_second = self.cursor_chunk;
        let result = self.find_free_slice_in_all_chunks(size, chunks, slices, self.queue.len());

        if result.is_some() {
            return result;
        }

        self.cursor_chunk = 0;
        self.cursor_slice = 0;
        self.find_free_slice_in_all_chunks(size, chunks, slices, max_second)
    }

    fn find_free_slice_in_chunk(
        &mut self,
        size: usize,
        chunk: &mut C,
        slices: &mut HashMap<SliceId, S>,
        mut slice_index: usize,
    ) -> Option<(usize, SliceId)> {
        while let Some(slice_id) = chunk.slice(slice_index) {
            let slice = slices.get_mut(&slice_id).unwrap();

            let is_big_enough = slice.size() >= size;
            let is_free = slice.is_free();

            if is_big_enough && is_free {
                if slice.size() > size {
                    let new_slice = slice.split(size);
                    chunk.insert_slice(slice_index + 1, new_slice.id());
                    slices.insert(new_slice.id(), new_slice);
                }

                return Some((slice_index, slice_id));
            }

            if is_free && chunk.merge_next_slice(slice_index, slices) {
                continue;
            }

            slice_index += 1;
        }

        None
    }

    fn find_free_slice_in_all_chunks(
        &mut self,
        size: usize,
        chunks: &mut HashMap<ChunkId, C>,
        slices: &mut HashMap<SliceId, S>,
        max_cursor_position: usize,
    ) -> Option<SliceId> {
        let start = self.cursor_chunk;
        let end = usize::min(self.queue.len(), max_cursor_position);
        let mut slice_index = self.cursor_slice;

        for chunk_index in start..end {
            if chunk_index > start {
                slice_index = 0;
            }

            if let Some(id) = self.queue.get(chunk_index) {
                let chunk = chunks.get_mut(id).unwrap();
                let result = self.find_free_slice_in_chunk(size, chunk, slices, slice_index);

                if let Some((cursor_slice, slice)) = result {
                    self.cursor_slice = cursor_slice + 1;
                    self.cursor_chunk = chunk_index;
                    return Some(slice);
                }
            }
            self.cursor_chunk = chunk_index;
            self.cursor_slice = 0;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::stub::*;
    use super::*;
    use alloc::vec;

    #[test]
    fn simple_1() {
        let mut ring = RingBuffer::<TestChunk, TestSlice>::new();

        let slice_1 = new_slice(0, 100);
        let slice_2 = new_slice(1, 200);
        let chunk_1 = new_chunk(0, vec![0, 1]);

        let mut slices = HashMap::from([(slice_1.id, slice_1), (slice_2.id, slice_2)]);
        let mut chunks = HashMap::from([(chunk_1.id, chunk_1)]);

        ring.push_chunk(ChunkId { value: 0 });

        let slice = ring.find_free_slice(50, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, SliceId { value: 0 });
        assert_eq!(slices.get(&slice).unwrap().size, 50);
        assert_eq!(slices.len(), 3);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 3);
    }

    #[test]
    fn simple_2() {
        let mut ring = RingBuffer::<TestChunk, TestSlice>::new();

        let slice_1 = new_slice(0, 100);
        let slice_2 = new_slice(1, 200);
        let chunk_1 = new_chunk(0, vec![0, 1]);

        let mut slices = HashMap::from([(slice_1.id, slice_1), (slice_2.id, slice_2)]);
        let mut chunks = HashMap::from([(chunk_1.id, chunk_1)]);

        ring.push_chunk(ChunkId { value: 0 });

        let slice = ring.find_free_slice(150, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, SliceId { value: 0 });
        assert_eq!(slices.get(&slice).unwrap().size, 150);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 2);
    }

    #[test]
    fn multiple_chunks() {
        let mut ring = RingBuffer::<TestChunk, TestSlice>::new();

        let slice_1 = new_slice(0, 100);
        let slice_2 = new_slice(1, 200);
        let slice_3 = new_slice(2, 200);
        let slice_4 = new_slice(3, 200);
        let chunk_1 = new_chunk(0, vec![0, 1]);
        let chunk_2 = new_chunk(1, vec![2, 3]);

        let mut slices = HashMap::from([
            (slice_1.id, slice_1),
            (slice_2.id, slice_2),
            (slice_3.id, slice_3),
            (slice_4.id, slice_4),
        ]);
        let mut chunks = HashMap::from([(chunk_1.id, chunk_1), (chunk_2.id, chunk_2)]);

        ring.push_chunk(ChunkId { value: 0 });
        ring.push_chunk(ChunkId { value: 1 });

        slices.get_mut(&SliceId { value: 0 }).unwrap().is_free = true;
        slices.get_mut(&SliceId { value: 1 }).unwrap().is_free = false;
        slices.get_mut(&SliceId { value: 3 }).unwrap().is_free = false;

        let slice = ring.find_free_slice(200, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, SliceId { value: 2 });

        let slice = ring.find_free_slice(100, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, SliceId { value: 0 });
    }

    #[test]
    fn find_free_slice_with_exact_fit() {
        let mut ring = RingBuffer::<TestChunk, TestSlice>::new();

        let slice_1 = new_slice(0, 100);
        let slice_2 = new_slice(1, 200);
        let chunk_1 = new_chunk(0, vec![0, 1]);

        let mut slices = HashMap::from([(slice_1.id, slice_1), (slice_2.id, slice_2)]);
        let mut chunks = HashMap::from([(chunk_1.id, chunk_1)]);

        ring.push_chunk(ChunkId { value: 0 });

        slices.get_mut(&SliceId { value: 0 }).unwrap().is_free = false;
        slices.get_mut(&SliceId { value: 1 }).unwrap().is_free = true;

        let slice = ring.find_free_slice(200, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, SliceId { value: 1 });
        assert_eq!(slices.get(&slice).unwrap().size, 200);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 2);
    }

    #[test]
    fn find_free_slice_with_merging() {
        let mut ring = RingBuffer::<TestChunk, TestSlice>::new();

        let slice_1 = new_slice(0, 100);
        let slice_2 = new_slice(1, 50);
        let slice_3 = new_slice(2, 100);
        let chunk_1 = new_chunk(0, vec![0, 1, 2]);

        let mut slices = HashMap::from([
            (slice_1.id, slice_1),
            (slice_2.id, slice_2),
            (slice_3.id, slice_3),
        ]);
        let mut chunks = HashMap::from([(chunk_1.id, chunk_1)]);

        ring.push_chunk(ChunkId { value: 0 });

        slices.get_mut(&SliceId { value: 0 }).unwrap().is_free = true;
        slices.get_mut(&SliceId { value: 1 }).unwrap().is_free = true;
        slices.get_mut(&SliceId { value: 2 }).unwrap().is_free = true;

        let slice = ring.find_free_slice(250, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, SliceId { value: 0 });
        assert_eq!(slices.get(&slice).unwrap().size, 250);
        assert_eq!(slices.len(), 1);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 1);
    }

    #[test]
    fn find_free_slice_with_multiple_chunks_and_merging() {
        let mut ring = RingBuffer::<TestChunk, TestSlice>::new();

        let slice_1 = new_slice(0, 50);
        let slice_2 = new_slice(1, 50);
        let chunk_1 = new_chunk(0, vec![0, 1]);

        let slice_3 = new_slice(2, 100);
        let slice_4 = new_slice(3, 50);
        let chunk_2 = new_chunk(1, vec![2, 3]);

        let mut slices = HashMap::from([
            (slice_1.id, slice_1),
            (slice_2.id, slice_2),
            (slice_3.id, slice_3),
            (slice_4.id, slice_4),
        ]);
        let mut chunks = HashMap::from([(chunk_1.id, chunk_1), (chunk_2.id, chunk_2)]);

        ring.push_chunk(ChunkId { value: 0 });
        ring.push_chunk(ChunkId { value: 1 });

        slices.get_mut(&SliceId { value: 0 }).unwrap().is_free = true;
        slices.get_mut(&SliceId { value: 1 }).unwrap().is_free = true;
        slices.get_mut(&SliceId { value: 2 }).unwrap().is_free = true;
        slices.get_mut(&SliceId { value: 3 }).unwrap().is_free = true;

        let slice = ring.find_free_slice(150, &mut chunks, &mut slices).unwrap();

        assert_eq!(slices.get(&slice).unwrap().size, 150);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 1);
    }

    fn new_slice(id: usize, size: usize) -> TestSlice {
        TestSlice {
            id: SliceId { value: id },
            is_free: true,
            size,
        }
    }

    fn new_chunk(id: usize, slices: Vec<usize>) -> TestChunk {
        TestChunk {
            id: ChunkId { value: id },
            slices: slices.into_iter().map(|i| SliceId { value: i }).collect(),
        }
    }
}

#[cfg(test)]
mod stub {
    use super::*;
    use burn_common::*;

    #[derive(Debug)]
    pub struct TestChunk {
        pub id: ChunkId,
        pub slices: Vec<SliceId>,
    }

    #[derive(Debug)]
    pub struct TestSlice {
        pub id: SliceId,
        pub is_free: bool,
        pub size: usize,
    }

    impl MemorySlice for TestSlice {
        fn is_free(&self) -> bool {
            self.is_free
        }

        fn size(&self) -> usize {
            self.size
        }

        fn split(&mut self, offset: usize) -> Self {
            let size_remained = self.size - offset;
            self.size = offset;

            Self {
                id: SliceId {
                    value: rand::gen_random(),
                },
                is_free: true,
                size: size_remained,
            }
        }

        fn id(&self) -> SliceId {
            self.id
        }
    }

    impl MemoryChunk<TestSlice> for TestChunk {
        fn merge_next_slice(
            &mut self,
            from_slice_index: usize,
            slices: &mut HashMap<SliceId, TestSlice>,
        ) -> bool {
            let slice_id_current = self.slices.get(from_slice_index).unwrap();
            let slice_id_next = self.slices.get(from_slice_index + 1);
            let slice_id_next = match slice_id_next {
                Some(val) => val,
                None => return false,
            };

            let slice_next = slices.get(slice_id_next).unwrap();
            let is_free = slice_next.is_free;
            let size = slice_next.size;

            let slice_current = slices.get_mut(slice_id_current).unwrap();

            if is_free {
                slice_current.size += size;
                slices.remove(slice_id_next);
                self.slices.remove(from_slice_index + 1);

                return true;
            }

            false
        }

        fn slice(&self, index: usize) -> Option<SliceId> {
            self.slices.get(index).copied()
        }

        fn insert_slice(&mut self, position: usize, slice_id: SliceId) {
            self.slices.insert(position, slice_id);
        }
    }
}
