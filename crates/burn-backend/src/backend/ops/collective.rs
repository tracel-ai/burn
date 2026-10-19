use crate::{
    Backend, TensorMetadata,
    distributed::ReduceOperation,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use alloc::vec::Vec;
use burn_std::Slice;
use core::{marker::PhantomData, ops::Range};

/// Collectives over a group of devices, called once with every member's tensor.
///
/// Member `i` is position `i` of the tensors given, each on its own member's device and every one
/// of one shape. A chunk is a member's share of a dimension, as even as possible with the first
/// members holding one more; a dimension split into chunks holds at least one element per member,
/// since a backend never receives an empty slice. The defaults emulate each collective with
/// transfers, slices, concatenations and additions, so every backend runs them; a backend with a
/// native collective between its devices overrides them.
///
/// They must not run beside [`DistributedOps::all_reduce`](crate::distributed::DistributedOps::all_reduce)
/// on a device with a transport: both queue on that device's one transport, so the two can
/// deadlock.
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
struct DimSplit {
    dim: usize,
    len: usize,
    members: usize,
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

    fn range(&self, member: usize) -> Range<usize> {
        let base = self.len / self.members;
        let longer = self.len % self.members;
        let start = member * base + member.min(longer);
        start..start + base + usize::from(member < longer)
    }

    /// The slices that select `member`'s chunk.
    fn slices(&self, member: usize) -> Vec<Slice> {
        let range = self.range(member);
        (0..=self.dim)
            .map(|axis| match axis == self.dim {
                true => Slice::new(range.start as isize, Some(range.end as isize), 1),
                false => Slice::full(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
