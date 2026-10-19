use crate::{
    Backend, Scalar, TensorMetadata,
    distributed::ReduceOperation,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use alloc::vec::Vec;
use burn_std::Slice;
use core::ops::Range;

/// Collectives over a group of devices, called once with every member's tensor.
///
/// Member `i` is position `i` of the tensors given, each on its own member's device and every one
/// of one shape. A chunk is a member's share of a dimension, as even as possible with the first
/// members holding one more. The defaults compose each collective from transfers, slices,
/// concatenations and additions, so every backend runs them; a backend with a transport between
/// its devices overrides them.
///
/// They must not run beside [`DistributedOps::all_reduce`](crate::distributed::DistributedOps::all_reduce)
/// on a device with a transport: both queue on that device's one transport, so the two can
/// deadlock.
pub trait CollectiveOps<B: Backend> {
    /// Every member receives `op` over every member's tensor.
    fn float_all_reduce(tensors: Vec<FloatTensor<B>>, op: ReduceOperation) -> Vec<FloatTensor<B>> {
        ComposedReductions::<B, _>::float().all_reduce(tensors, op)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn float_all_gather(tensors: Vec<FloatTensor<B>>, dim: usize) -> Vec<FloatTensor<B>> {
        ComposedCollectives::<B, _>::float().all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `dim` of `op` over every member's tensor.
    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<B>> {
        ComposedReductions::<B, _>::float().reduce_scatter(tensors, dim, op)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn float_all_to_all(
        tensors: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        ComposedCollectives::<B, _>::float().all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn float_broadcast(tensor: FloatTensor<B>, devices: &[Device<B>]) -> Vec<FloatTensor<B>> {
        ComposedCollectives::<B, _>::float().broadcast(tensor, devices)
    }

    /// Every member receives `op` over every member's tensor.
    fn int_all_reduce(tensors: Vec<IntTensor<B>>, op: ReduceOperation) -> Vec<IntTensor<B>> {
        ComposedReductions::<B, _>::int().all_reduce(tensors, op)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn int_all_gather(tensors: Vec<IntTensor<B>>, dim: usize) -> Vec<IntTensor<B>> {
        ComposedCollectives::<B, _>::int().all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `dim` of `op` over every member's tensor.
    fn int_reduce_scatter(
        tensors: Vec<IntTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<B>> {
        ComposedReductions::<B, _>::int().reduce_scatter(tensors, dim, op)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn int_all_to_all(
        tensors: Vec<IntTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<B>> {
        ComposedCollectives::<B, _>::int().all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn int_broadcast(tensor: IntTensor<B>, devices: &[Device<B>]) -> Vec<IntTensor<B>> {
        ComposedCollectives::<B, _>::int().broadcast(tensor, devices)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn bool_all_gather(tensors: Vec<BoolTensor<B>>, dim: usize) -> Vec<BoolTensor<B>> {
        ComposedCollectives::<B, _>::bool().all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn bool_all_to_all(
        tensors: Vec<BoolTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<B>> {
        ComposedCollectives::<B, _>::bool().all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn bool_broadcast(tensor: BoolTensor<B>, devices: &[Device<B>]) -> Vec<BoolTensor<B>> {
        ComposedCollectives::<B, _>::bool().broadcast(tensor, devices)
    }
}

/// The collectives of one tensor kind composed from its transfers, slices and concatenations:
/// what a backend without a transport of its own runs, and what an override falls back to.
pub struct ComposedCollectives<B: Backend, T> {
    to_device: fn(T, &Device<B>) -> T,
    slice: fn(T, &[Slice]) -> T,
    cat: fn(Vec<T>, usize) -> T,
}

/// The reductions of one tensor kind composed from its collectives and additions.
pub struct ComposedReductions<B: Backend, T> {
    collectives: ComposedCollectives<B, T>,
    add: fn(T, T) -> T,
    div_scalar: fn(T, Scalar) -> T,
    mean: MeanDivides,
}

/// What a mean divides by the member count.
enum MeanDivides {
    /// Each part, as NCCL averages floats, so a sum past the type's range still averages.
    EachPart,
    /// The sum, so an integer mean truncates once.
    Sum,
}

/// A dimension's length split over a group's members.
struct Chunks {
    len: usize,
    members: usize,
}

impl<B: Backend> ComposedCollectives<B, FloatTensor<B>> {
    /// The float collectives.
    pub fn float() -> Self {
        Self {
            to_device: B::float_to_device,
            slice: B::float_slice,
            cat: B::float_cat,
        }
    }
}

impl<B: Backend> ComposedCollectives<B, IntTensor<B>> {
    /// The int collectives.
    pub fn int() -> Self {
        Self {
            to_device: B::int_to_device,
            slice: B::int_slice,
            cat: B::int_cat,
        }
    }
}

impl<B: Backend> ComposedCollectives<B, BoolTensor<B>> {
    /// The bool collectives.
    pub fn bool() -> Self {
        Self {
            to_device: B::bool_to_device,
            slice: B::bool_slice,
            cat: B::bool_cat,
        }
    }
}

impl<B: Backend, T: TensorMetadata<Device = Device<B>>> ComposedCollectives<B, T> {
    /// See [`CollectiveOps::float_all_gather`].
    pub fn all_gather(&self, tensors: Vec<T>, dim: usize) -> Vec<T> {
        Self::devices(&tensors)
            .iter()
            .map(|device| (self.cat)(self.moved_to(&tensors, device), dim))
            .collect()
    }

    /// See [`CollectiveOps::float_all_to_all`]. Only the chunks that change member move.
    pub fn all_to_all(&self, tensors: Vec<T>, split_dim: usize, concat_dim: usize) -> Vec<T> {
        let chunks = Chunks::new(&tensors, split_dim);
        Self::devices(&tensors)
            .iter()
            .enumerate()
            .map(|(member, device)| {
                let received = tensors
                    .iter()
                    .map(|tensor| {
                        let chunk = self.chunk(tensor.clone(), split_dim, chunks.range(member));
                        (self.to_device)(chunk, device)
                    })
                    .collect();
                (self.cat)(received, concat_dim)
            })
            .collect()
    }

    /// See [`CollectiveOps::float_broadcast`].
    pub fn broadcast(&self, tensor: T, devices: &[Device<B>]) -> Vec<T> {
        devices
            .iter()
            .map(|device| (self.to_device)(tensor.clone(), device))
            .collect()
    }

    fn devices(tensors: &[T]) -> Vec<Device<B>> {
        tensors.iter().map(TensorMetadata::device).collect()
    }

    fn moved_to(&self, tensors: &[T], device: &Device<B>) -> Vec<T> {
        tensors
            .iter()
            .map(|tensor| (self.to_device)(tensor.clone(), device))
            .collect()
    }

    fn chunk(&self, tensor: T, dim: usize, range: Range<usize>) -> T {
        let slices: Vec<Slice> = (0..=dim)
            .map(|axis| match axis == dim {
                true => Slice::new(range.start as isize, Some(range.end as isize), 1),
                false => Slice::full(),
            })
            .collect();
        (self.slice)(tensor, &slices)
    }
}

impl<B: Backend> ComposedReductions<B, FloatTensor<B>> {
    /// The float reductions.
    pub fn float() -> Self {
        Self {
            collectives: ComposedCollectives::float(),
            add: B::float_add,
            div_scalar: B::float_div_scalar,
            mean: MeanDivides::EachPart,
        }
    }
}

impl<B: Backend> ComposedReductions<B, IntTensor<B>> {
    /// The int reductions, whose mean divides as the backend's integer division does.
    pub fn int() -> Self {
        Self {
            collectives: ComposedCollectives::int(),
            add: B::int_add,
            div_scalar: B::int_div_scalar,
            mean: MeanDivides::Sum,
        }
    }
}

impl<B: Backend, T: TensorMetadata<Device = Device<B>>> ComposedReductions<B, T> {
    /// See [`CollectiveOps::float_all_reduce`].
    pub fn all_reduce(&self, tensors: Vec<T>, op: ReduceOperation) -> Vec<T> {
        ComposedCollectives::<B, T>::devices(&tensors)
            .iter()
            .map(|device| self.reduced(self.collectives.moved_to(&tensors, device), op))
            .collect()
    }

    /// See [`CollectiveOps::float_reduce_scatter`]. Only each member's chunk moves to it.
    pub fn reduce_scatter(&self, tensors: Vec<T>, dim: usize, op: ReduceOperation) -> Vec<T> {
        let chunks = Chunks::new(&tensors, dim);
        ComposedCollectives::<B, T>::devices(&tensors)
            .iter()
            .enumerate()
            .map(|(member, device)| {
                let parts = tensors
                    .iter()
                    .map(|tensor| {
                        let chunk =
                            self.collectives
                                .chunk(tensor.clone(), dim, chunks.range(member));
                        (self.collectives.to_device)(chunk, device)
                    })
                    .collect();
                self.reduced(parts, op)
            })
            .collect()
    }

    fn reduced(&self, parts: Vec<T>, op: ReduceOperation) -> T {
        let members = parts.len();
        match (op, &self.mean) {
            (ReduceOperation::Sum, _) => self.sum(parts),
            (ReduceOperation::Mean, MeanDivides::EachPart) => self.sum(
                parts
                    .into_iter()
                    .map(|part| (self.div_scalar)(part, members.into()))
                    .collect(),
            ),
            (ReduceOperation::Mean, MeanDivides::Sum) => {
                (self.div_scalar)(self.sum(parts), members.into())
            }
        }
    }

    fn sum(&self, parts: Vec<T>) -> T {
        parts
            .into_iter()
            .reduce(self.add)
            .expect("a group has a member")
    }
}

impl Chunks {
    fn new<T: TensorMetadata>(tensors: &[T], dim: usize) -> Self {
        Self {
            len: tensors.first().map_or(0, |tensor| tensor.shape()[dim]),
            members: tensors.len(),
        }
    }

    fn range(&self, member: usize) -> Range<usize> {
        let base = self.len / self.members;
        let longer = self.len % self.members;
        let start = member * base + member.min(longer);
        start..start + base + usize::from(member < longer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_give_the_first_members_one_more() {
        let chunks = Chunks { len: 7, members: 3 };
        let ranges: Vec<_> = (0..3).map(|member| chunks.range(member)).collect();

        assert_eq!(ranges, [0..3, 3..5, 5..7]);
    }
}
