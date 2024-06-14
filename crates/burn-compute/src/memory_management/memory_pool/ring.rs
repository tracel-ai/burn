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
    fn merge_slices(&mut self, from_slice_index: usize, slices: &mut HashMap<SliceId, S>);
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
        let result = self.find_free_slice_in_chunks(size, chunks, slices, self.queue.len());

        if result.is_some() {
            return result;
        }

        self.cursor_chunk = 0;
        self.cursor_slice = 0;
        self.find_free_slice_in_chunks(size, chunks, slices, max_second)
    }

    fn find_free_slice_in_chunk(
        &mut self,
        size: usize,
        chunk: &mut C,
        slices: &mut HashMap<SliceId, S>,
        mut slice_index: usize,
    ) -> Option<(usize, SliceId)> {
        let mut merged = false;

        loop {
            let slice_id = if let Some(slice_id) = chunk.slice(slice_index) {
                slice_id
            } else {
                break;
            };

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

            if is_free && !merged {
                chunk.merge_slices(slice_index, slices);
                merged = true;
            } else {
                slice_index += 1;
            }
        }

        None
    }

    fn find_free_slice_in_chunks(
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
    use super::*;

    #[derive(Debug)]
    struct TestChunk {
        id: ChunkId,
        slices: Vec<SliceId>,
    }

    #[derive(Debug)]
    struct TestSlice {
        id: SliceId,
        is_free: bool,
        size: usize,
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
                    value: rand::random(),
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
        fn merge_slices(
            &mut self,
            from_slice_index: usize,
            slices: &mut HashMap<SliceId, TestSlice>,
        ) {
            let mut slices_updated = Vec::with_capacity(self.slices.len());

            let mut current: Option<TestSlice> = None;

            for (index, slice_id) in self.slices.drain(..).enumerate() {
                if index < from_slice_index {
                    slices_updated.push(slice_id);
                    continue;
                }

                let slice = slices.remove(&slice_id).unwrap();

                if slice.is_free {
                    match current.take() {
                        Some(mut val) => {
                            val.size += slice.size;
                            current = Some(val);
                        }
                        None => {
                            current = Some(slice);
                        }
                    };
                } else {
                    if let Some(s) = current.take() {
                        slices_updated.push(s.id);
                        slices.insert(s.id, s);
                    }

                    slices_updated.push(slice.id);
                    slices.insert(slice.id, slice);
                }
            }

            if let Some(s) = current.take() {
                slices_updated.push(s.id);
                slices.insert(s.id, s);
            }

            self.slices = slices_updated;
        }

        fn slice(&self, index: usize) -> Option<SliceId> {
            self.slices.get(index).map(|slice| slice.clone())
        }

        fn insert_slice(&mut self, position: usize, slice_id: SliceId) {
            self.slices.insert(position, slice_id);
        }
    }

    #[test]
    fn simple_1() {
        let mut ring = RingBuffer::<TestChunk, TestSlice>::new();

        let slice_1 = new_slice(0, 100);
        let slice_2 = new_slice(1, 200);
        let chunk_1 = new_chunk(0, vec![0, 1]);

        let mut slices = HashMap::from([(slice_1.id, slice_1), (slice_2.id, slice_2)]);
        let mut chunks = HashMap::from([(chunk_1.id, chunk_1)]);

        ring.push_chunk(ChunkId { value: 0 });

        let slice = ring
            .find_free_slice(50, &mut chunks, &mut slices)
            .unwrap();

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

        let slice = ring
            .find_free_slice(150, &mut chunks, &mut slices)
            .unwrap();

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

        let slice = ring
            .find_free_slice(200, &mut chunks, &mut slices)
            .unwrap();

        assert_eq!(slice, SliceId { value: 2 });

        let slice = ring
            .find_free_slice(100, &mut chunks, &mut slices)
            .unwrap();

        assert_eq!(slice, SliceId { value: 0 });
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
