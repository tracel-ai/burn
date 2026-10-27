use crate::{
    Backend, TensorMetadata,
    distributed::ReduceOperation,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use alloc::vec::Vec;
use burn_std::{Bytes, DType, Slice, TensorData};
use core::{marker::PhantomData, ops::Range};

/// Collectives over a group of devices, called once with every member's tensor.
///
/// Member `i` is position `i` of the tensors given, each on its own member's device and every one
/// of one shape. A chunk is a member's share of a dimension, as even as possible with the first
/// members holding one more; a dimension split into chunks holds at least one element per member,
/// since a backend never receives an empty slice. The defaults emulate each collective with
/// transfers, slices, concatenations and additions, so every backend runs them; a backend with a
/// native collective between its devices overrides them.
pub trait CollectiveOps<B: Backend> {
    /// Every member receives `op` over every member's tensor.
    fn float_all_reduce(tensors: Vec<FloatTensor<B>>, op: ReduceOperation) -> Vec<FloatTensor<B>> {
        EmulatedCollectives::<B>::float_all_reduce(tensors, op)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn float_all_gather(tensors: Vec<FloatTensor<B>>, dim: usize) -> Vec<FloatTensor<B>> {
        EmulatedCollectives::<B>::float_all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `dim` of `op` over every member's tensor.
    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<B>> {
        EmulatedCollectives::<B>::float_reduce_scatter(tensors, dim, op)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn float_all_to_all(
        tensors: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        EmulatedCollectives::<B>::float_all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn float_broadcast(tensor: FloatTensor<B>, devices: &[Device<B>]) -> Vec<FloatTensor<B>> {
        EmulatedCollectives::<B>::float_broadcast(tensor, devices)
    }

    /// Every member receives `op` over every member's tensor.
    fn int_all_reduce(tensors: Vec<IntTensor<B>>, op: ReduceOperation) -> Vec<IntTensor<B>> {
        EmulatedCollectives::<B>::int_all_reduce(tensors, op)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn int_all_gather(tensors: Vec<IntTensor<B>>, dim: usize) -> Vec<IntTensor<B>> {
        EmulatedCollectives::<B>::int_all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `dim` of `op` over every member's tensor.
    fn int_reduce_scatter(
        tensors: Vec<IntTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<B>> {
        EmulatedCollectives::<B>::int_reduce_scatter(tensors, dim, op)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn int_all_to_all(
        tensors: Vec<IntTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<B>> {
        EmulatedCollectives::<B>::int_all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn int_broadcast(tensor: IntTensor<B>, devices: &[Device<B>]) -> Vec<IntTensor<B>> {
        EmulatedCollectives::<B>::int_broadcast(tensor, devices)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn bool_all_gather(tensors: Vec<BoolTensor<B>>, dim: usize) -> Vec<BoolTensor<B>> {
        EmulatedCollectives::<B>::bool_all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn bool_all_to_all(
        tensors: Vec<BoolTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<B>> {
        EmulatedCollectives::<B>::bool_all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn bool_broadcast(tensor: BoolTensor<B>, devices: &[Device<B>]) -> Vec<BoolTensor<B>> {
        EmulatedCollectives::<B>::bool_broadcast(tensor, devices)
    }
}

/// The collectives of every tensor kind emulated with ordinary tensor ops, transfers between
/// devices, slices, concatenations and additions. Its methods are [`CollectiveOps`]' defaults, by a
/// name an override can call when it falls back.
pub struct EmulatedCollectives<B> {
    _backend: PhantomData<B>,
}

/// Dimension `dim`, of length `len`, cut into one chunk per member, a member being one of the
/// tensors a collective is given: as even as possible, with the first members holding one more.
///
/// A caller that splits a tensor itself must cut it the same way, so this is public.
#[derive(Clone, Copy, Debug)]
pub struct DimSplit {
    /// The dimension the chunks cut along.
    pub dim: usize,
    /// How many elements that dimension holds.
    pub len: usize,
    /// How many chunks it is cut into, one per member.
    pub members: usize,
}

impl<B: Backend> EmulatedCollectives<B> {
    /// See [`CollectiveOps::float_all_reduce`].
    pub fn float_all_reduce(
        tensors: Vec<FloatTensor<B>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<B>> {
        Self::all_reduce(tensors, B::float_to_device, |parts| {
            Self::float_reduced(parts, op)
        })
    }

    /// See [`CollectiveOps::float_all_gather`].
    pub fn float_all_gather(tensors: Vec<FloatTensor<B>>, dim: usize) -> Vec<FloatTensor<B>> {
        Self::all_gather(tensors, dim, B::float_to_device, B::float_cat)
    }

    /// See [`CollectiveOps::float_reduce_scatter`].
    pub fn float_reduce_scatter(
        tensors: Vec<FloatTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<B>> {
        Self::reduce_scatter(tensors, dim, B::float_to_device, B::float_slice, |parts| {
            Self::float_reduced(parts, op)
        })
    }

    /// See [`CollectiveOps::float_all_to_all`].
    pub fn float_all_to_all(
        tensors: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        Self::all_to_all(
            tensors,
            split_dim,
            concat_dim,
            B::float_to_device,
            B::float_slice,
            B::float_cat,
        )
    }

    /// See [`CollectiveOps::float_broadcast`].
    pub fn float_broadcast(tensor: FloatTensor<B>, devices: &[Device<B>]) -> Vec<FloatTensor<B>> {
        Self::broadcast(tensor, devices, B::float_to_device)
    }

    /// See [`CollectiveOps::int_all_reduce`].
    pub fn int_all_reduce(tensors: Vec<IntTensor<B>>, op: ReduceOperation) -> Vec<IntTensor<B>> {
        Self::all_reduce(tensors, B::int_to_device, |parts| {
            Self::int_reduced(parts, op)
        })
    }

    /// See [`CollectiveOps::int_all_gather`].
    pub fn int_all_gather(tensors: Vec<IntTensor<B>>, dim: usize) -> Vec<IntTensor<B>> {
        Self::all_gather(tensors, dim, B::int_to_device, B::int_cat)
    }

    /// See [`CollectiveOps::int_reduce_scatter`].
    pub fn int_reduce_scatter(
        tensors: Vec<IntTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<B>> {
        Self::reduce_scatter(tensors, dim, B::int_to_device, B::int_slice, |parts| {
            Self::int_reduced(parts, op)
        })
    }

    /// See [`CollectiveOps::int_all_to_all`].
    pub fn int_all_to_all(
        tensors: Vec<IntTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<B>> {
        Self::all_to_all(
            tensors,
            split_dim,
            concat_dim,
            B::int_to_device,
            B::int_slice,
            B::int_cat,
        )
    }

    /// See [`CollectiveOps::int_broadcast`].
    pub fn int_broadcast(tensor: IntTensor<B>, devices: &[Device<B>]) -> Vec<IntTensor<B>> {
        Self::broadcast(tensor, devices, B::int_to_device)
    }

    /// See [`CollectiveOps::bool_all_gather`].
    pub fn bool_all_gather(tensors: Vec<BoolTensor<B>>, dim: usize) -> Vec<BoolTensor<B>> {
        Self::all_gather(tensors, dim, B::bool_to_device, B::bool_cat)
    }

    /// See [`CollectiveOps::bool_all_to_all`].
    pub fn bool_all_to_all(
        tensors: Vec<BoolTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<B>> {
        Self::all_to_all(
            tensors,
            split_dim,
            concat_dim,
            B::bool_to_device,
            B::bool_slice,
            B::bool_cat,
        )
    }

    /// See [`CollectiveOps::bool_broadcast`].
    pub fn bool_broadcast(tensor: BoolTensor<B>, devices: &[Device<B>]) -> Vec<BoolTensor<B>> {
        Self::broadcast(tensor, devices, B::bool_to_device)
    }

    /// Reduces once on the first member and broadcasts the result, so each tensor crosses a link
    /// once each way rather than to every member.
    fn all_reduce<T, D, R>(tensors: Vec<T>, to_device: D, reduce: R) -> Vec<T>
    where
        T: TensorMetadata<Device = Device<B>>,
        D: Fn(T, &Device<B>) -> T,
        R: Fn(Vec<T>) -> T,
    {
        let devices = Self::devices(&tensors);
        let Some(root) = devices.first() else {
            return Vec::new();
        };
        let reduced = reduce(Self::moved_to(&tensors, root, &to_device));
        Self::broadcast(reduced, &devices, to_device)
    }

    fn all_gather<T, D, C>(tensors: Vec<T>, dim: usize, to_device: D, cat: C) -> Vec<T>
    where
        T: TensorMetadata<Device = Device<B>>,
        D: Fn(T, &Device<B>) -> T,
        C: Fn(Vec<T>, usize) -> T,
    {
        Self::devices(&tensors)
            .iter()
            .map(|device| cat(Self::moved_to(&tensors, device, &to_device), dim))
            .collect()
    }

    /// Only each member's chunk moves to it.
    fn reduce_scatter<T, D, S, R>(
        tensors: Vec<T>,
        dim: usize,
        to_device: D,
        slice: S,
        reduce: R,
    ) -> Vec<T>
    where
        T: TensorMetadata<Device = Device<B>>,
        D: Fn(T, &Device<B>) -> T,
        S: Fn(T, &[Slice]) -> T,
        R: Fn(Vec<T>) -> T,
    {
        let split = DimSplit::new(&tensors, dim);
        Self::devices(&tensors)
            .iter()
            .enumerate()
            .map(|(member, device)| {
                let chunks =
                    Self::member_chunks(&tensors, &split, member, device, &to_device, &slice);
                reduce(chunks)
            })
            .collect()
    }

    /// Only the chunks that change member move.
    fn all_to_all<T, D, S, C>(
        tensors: Vec<T>,
        split_dim: usize,
        concat_dim: usize,
        to_device: D,
        slice: S,
        cat: C,
    ) -> Vec<T>
    where
        T: TensorMetadata<Device = Device<B>>,
        D: Fn(T, &Device<B>) -> T,
        S: Fn(T, &[Slice]) -> T,
        C: Fn(Vec<T>, usize) -> T,
    {
        let split = DimSplit::new(&tensors, split_dim);
        Self::devices(&tensors)
            .iter()
            .enumerate()
            .map(|(member, device)| {
                let chunks =
                    Self::member_chunks(&tensors, &split, member, device, &to_device, &slice);
                cat(chunks, concat_dim)
            })
            .collect()
    }

    fn broadcast<T, D>(tensor: T, devices: &[Device<B>], to_device: D) -> Vec<T>
    where
        T: Clone,
        D: Fn(T, &Device<B>) -> T,
    {
        devices
            .iter()
            .map(|device| to_device(tensor.clone(), device))
            .collect()
    }

    fn devices<T>(tensors: &[T]) -> Vec<Device<B>>
    where
        T: TensorMetadata<Device = Device<B>>,
    {
        tensors.iter().map(TensorMetadata::device).collect()
    }

    fn moved_to<T, D>(tensors: &[T], device: &Device<B>, to_device: &D) -> Vec<T>
    where
        T: Clone,
        D: Fn(T, &Device<B>) -> T,
    {
        tensors
            .iter()
            .map(|tensor| to_device(tensor.clone(), device))
            .collect()
    }

    /// `member`'s chunk of every member's tensor, moved to `device`.
    fn member_chunks<T, D, S>(
        tensors: &[T],
        split: &DimSplit,
        member: usize,
        device: &Device<B>,
        to_device: &D,
        slice: &S,
    ) -> Vec<T>
    where
        T: Clone,
        D: Fn(T, &Device<B>) -> T,
        S: Fn(T, &[Slice]) -> T,
    {
        let slices = split.slices(member);
        tensors
            .iter()
            .map(|tensor| to_device(slice(tensor.clone(), &slices), device))
            .collect()
    }

    /// A mean divides each part first, as NCCL averages floats, so a sum past the type's range
    /// still averages.
    fn float_reduced(parts: Vec<FloatTensor<B>>, op: ReduceOperation) -> FloatTensor<B> {
        let members = parts.len();
        let parts = parts.into_iter().map(|part| match op {
            ReduceOperation::Sum => part,
            ReduceOperation::Mean => B::float_div_scalar(part, members.into()),
        });
        parts.reduce(B::float_add).expect("a group has a member")
    }

    /// A mean divides the sum, so it truncates once, as the backend's integer division does.
    fn int_reduced(parts: Vec<IntTensor<B>>, op: ReduceOperation) -> IntTensor<B> {
        let members = parts.len();
        let sum = parts
            .into_iter()
            .reduce(B::int_add)
            .expect("a group has a member");
        match op {
            ReduceOperation::Sum => sum,
            ReduceOperation::Mean => B::int_div_scalar(sum, members.into()),
        }
    }
}

impl DimSplit {
    fn new<T: TensorMetadata>(tensors: &[T], dim: usize) -> Self {
        let split = Self {
            dim,
            len: tensors.first().map_or(0, |tensor| tensor.shape()[dim]),
            members: tensors.len(),
        };
        assert!(
            split.len >= split.members,
            "dimension {dim} holds {} elements, fewer than the {} members it splits over",
            split.len,
            split.members,
        );
        split
    }

    /// The indices along `dim` that `member` holds.
    pub fn range(&self, member: usize) -> Range<usize> {
        let base = self.len / self.members;
        let longer = self.len % self.members;
        let start = member * base + member.min(longer);
        start..start + base + usize::from(member < longer)
    }

    /// The slices that select `member`'s chunk.
    pub fn slices(&self, member: usize) -> Vec<Slice> {
        let range = self.range(member);
        (0..=self.dim)
            .map(|axis| match axis == self.dim {
                true => Slice::new(range.start as isize, Some(range.end as isize), 1),
                false => Slice::full(),
            })
            .collect()
    }

    /// Each member's chunk of `data`, in member order, cut on the host one at a time.
    ///
    /// # Panics
    ///
    /// When `data` is quantized, since its packed values and scales do not cut along a dim, or
    /// when its `dim` does not hold `len` elements.
    pub fn data_chunks(self, data: &TensorData) -> impl Iterator<Item = TensorData> + '_ {
        assert!(
            !matches!(data.dtype(), DType::QFloat(_)),
            "A quantized tensor is not cut along a dim"
        );
        let shape = data.shape();
        assert!(
            shape.as_slice().get(self.dim) == Some(&self.len),
            "dimension {} of a tensor of shape {shape:?} does not hold {} elements",
            self.dim,
            self.len,
        );
        (0..self.members).map(move |member| {
            let mut chunk_shape = shape.clone();
            chunk_shape[self.dim] = self.range(member).len();
            TensorData::from_bytes(self.chunk_bytes(data, member), chunk_shape, data.dtype())
        })
    }

    /// `member`'s bytes of `data`. A chunk in one piece is a window into `data` where its storage
    /// has windows, so a file-backed tensor reads only that range.
    fn chunk_bytes(&self, data: &TensorData, member: usize) -> Bytes {
        let pieces = self.byte_ranges(data, member);
        if let [piece] = pieces.as_slice()
            && let Ok(window) = data.bytes().view(piece.start, piece.end)
        {
            return window;
        }
        concatenated(data.as_bytes(), &pieces)
    }

    /// Where `member`'s chunk sits in `data`: one byte range for each index of the dims before
    /// `dim`.
    fn byte_ranges(&self, data: &TensorData, member: usize) -> Vec<Range<usize>> {
        let shape = data.shape().as_slice();
        let blocks: usize = shape[..self.dim].iter().product();
        let row_size = shape[self.dim + 1..].iter().product::<usize>() * data.dtype().size();
        let rows = self.range(member);
        (0..blocks)
            .map(|block| {
                let start = (block * self.len + rows.start) * row_size;
                start..start + rows.len() * row_size
            })
            .collect()
    }
}

/// The `ranges` of `bytes` end to end, in a u128 buffer: it is aligned to `MAX_ALIGN`, so the
/// result is not copied again to reach it.
fn concatenated(bytes: &[u8], ranges: &[Range<usize>]) -> Bytes {
    let len: usize = ranges.iter().map(Range::len).sum();
    let mut concatenated =
        Bytes::from_elems(Vec::<u128>::with_capacity(len.div_ceil(size_of::<u128>())));
    for range in ranges {
        concatenated.extend_from_byte_slice(&bytes[range.clone()]);
    }
    concatenated
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[cfg(feature = "std")]
    use burn_std::Reader;

    #[test]
    fn a_split_gives_the_first_members_one_more() {
        let split = DimSplit {
            dim: 0,
            len: 7,
            members: 3,
        };
        let ranges: Vec<_> = (0..3).map(|member| split.range(member)).collect();

        assert_eq!(ranges, [0..3, 3..5, 5..7]);
    }

    #[test]
    fn data_chunks_cut_every_block_along_an_inner_dim() {
        let data = TensorData::new((0..12).collect::<Vec<i32>>(), [2, 3, 2]);
        let split = DimSplit {
            dim: 1,
            len: 3,
            members: 2,
        };

        let chunks: Vec<_> = split.data_chunks(&data).collect();

        assert_eq!(
            chunks,
            [
                TensorData::new(vec![0, 1, 2, 3, 6, 7, 8, 9], [2, 2, 2]),
                TensorData::new(vec![4, 5, 10, 11], [2, 1, 2]),
            ]
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn data_chunks_along_the_first_dim_read_only_their_range_of_a_file() {
        let whole = TensorData::new((0..12).collect::<Vec<i32>>(), [4, 3]);
        let path = std::env::temp_dir().join(format!("burn-data-chunks-{}", std::process::id()));
        std::fs::write(&path, whole.as_bytes()).unwrap();
        let bytes = Bytes::from_file(&path, 48, 0);
        let data = TensorData::from_bytes(bytes, [4, 3], DType::I32);
        let split = DimSplit {
            dim: 0,
            len: 4,
            members: 2,
        };

        let chunks: Vec<_> = split.data_chunks(&data).collect();

        assert_eq!(
            chunks,
            [
                TensorData::new(vec![0, 1, 2, 3, 4, 5], [2, 3]),
                TensorData::new(vec![6, 7, 8, 9, 10, 11], [2, 3]),
            ]
        );
        assert!(data.bytes().read(Reader::new().no_copy()).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[should_panic(expected = "dimension 2 of a tensor of shape")]
    fn data_chunks_refuse_a_dim_the_tensor_lacks() {
        let data = TensorData::new((0..12).collect::<Vec<i32>>(), [2, 6]);
        let split = DimSplit {
            dim: 2,
            len: 4,
            members: 2,
        };

        let _ = split.data_chunks(&data);
    }
}
