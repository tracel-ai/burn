use crate::{
    Backend, TensorMetadata,
    distributed::ReduceOperation,
    tensor::{Bool, BoolTensor, Device, Float, FloatTensor, Int, IntTensor},
};
use alloc::vec::Vec;
use burn_std::Slice;
use core::{marker::PhantomData, ops::Range};

/// Collectives over a group of devices, called once with every member's tensor.
///
/// Member `i` is position `i` of the tensors given, each on its own member's device and every one
/// of one shape. A chunk is a member's share of a dimension, as even as possible with the first
/// members holding one more; a dimension split into chunks holds at least one element per member,
/// since a backend never receives an empty slice. The defaults compose each collective from
/// transfers, slices, concatenations and additions, so every backend runs them; a backend with a
/// transport between its devices overrides them.
///
/// They must not run beside [`DistributedOps::all_reduce`](crate::distributed::DistributedOps::all_reduce)
/// on a device with a transport: both queue on that device's one transport, so the two can
/// deadlock.
pub trait CollectiveOps<B: Backend> {
    /// Every member receives `op` over every member's tensor.
    fn float_all_reduce(tensors: Vec<FloatTensor<B>>, op: ReduceOperation) -> Vec<FloatTensor<B>> {
        TransferCollectives::<B, Float>::all_reduce(tensors, op)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn float_all_gather(tensors: Vec<FloatTensor<B>>, dim: usize) -> Vec<FloatTensor<B>> {
        TransferCollectives::<B, Float>::all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `dim` of `op` over every member's tensor.
    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<B>> {
        TransferCollectives::<B, Float>::reduce_scatter(tensors, dim, op)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn float_all_to_all(
        tensors: Vec<FloatTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<B>> {
        TransferCollectives::<B, Float>::all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn float_broadcast(tensor: FloatTensor<B>, devices: &[Device<B>]) -> Vec<FloatTensor<B>> {
        TransferCollectives::<B, Float>::broadcast(tensor, devices)
    }

    /// Every member receives `op` over every member's tensor.
    fn int_all_reduce(tensors: Vec<IntTensor<B>>, op: ReduceOperation) -> Vec<IntTensor<B>> {
        TransferCollectives::<B, Int>::all_reduce(tensors, op)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn int_all_gather(tensors: Vec<IntTensor<B>>, dim: usize) -> Vec<IntTensor<B>> {
        TransferCollectives::<B, Int>::all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `dim` of `op` over every member's tensor.
    fn int_reduce_scatter(
        tensors: Vec<IntTensor<B>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<B>> {
        TransferCollectives::<B, Int>::reduce_scatter(tensors, dim, op)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn int_all_to_all(
        tensors: Vec<IntTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<B>> {
        TransferCollectives::<B, Int>::all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn int_broadcast(tensor: IntTensor<B>, devices: &[Device<B>]) -> Vec<IntTensor<B>> {
        TransferCollectives::<B, Int>::broadcast(tensor, devices)
    }

    /// Every member receives every member's tensor, concatenated along `dim` in member order.
    fn bool_all_gather(tensors: Vec<BoolTensor<B>>, dim: usize) -> Vec<BoolTensor<B>> {
        TransferCollectives::<B, Bool>::all_gather(tensors, dim)
    }

    /// Member `i` receives chunk `i` along `split_dim` of every member's tensor, concatenated along
    /// `concat_dim` in member order.
    fn bool_all_to_all(
        tensors: Vec<BoolTensor<B>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<B>> {
        TransferCollectives::<B, Bool>::all_to_all(tensors, split_dim, concat_dim)
    }

    /// Each of `devices` receives `tensor`.
    fn bool_broadcast(tensor: BoolTensor<B>, devices: &[Device<B>]) -> Vec<BoolTensor<B>> {
        TransferCollectives::<B, Bool>::broadcast(tensor, devices)
    }
}

/// What [`TransferCollectives`] needs from a tensor kind to move its tensors, so each collective
/// is written once for float, int and bool tensors.
pub trait TransferKind<B: Backend> {
    /// The kind's tensor primitive.
    type Tensor: TensorMetadata<Device = Device<B>>;

    /// Moves `tensor` to `device`.
    fn to_device(tensor: Self::Tensor, device: &Device<B>) -> Self::Tensor;

    /// Slices `tensor`.
    fn slice(tensor: Self::Tensor, slices: &[Slice]) -> Self::Tensor;

    /// Concatenates `tensors` along `dim`.
    fn cat(tensors: Vec<Self::Tensor>, dim: usize) -> Self::Tensor;
}

/// What [`TransferCollectives`] also needs from a tensor kind to reduce its tensors.
pub trait ReduceKind<B: Backend>: TransferKind<B> {
    /// The elementwise sum of `parts`, all on one device.
    fn sum(parts: Vec<Self::Tensor>) -> Self::Tensor;

    /// The elementwise mean of `parts`, all on one device.
    fn mean(parts: Vec<Self::Tensor>) -> Self::Tensor;
}

/// The collectives of tensor kind `K` built from transfers between devices, slices,
/// concatenations and additions: what a backend without a transport runs, and what an override
/// falls back to.
pub struct TransferCollectives<B, K> {
    _kind: PhantomData<(B, K)>,
}

/// A dimension's length split over a group's members.
struct Chunks {
    len: usize,
    members: usize,
}

impl<B: Backend> TransferKind<B> for Float {
    type Tensor = FloatTensor<B>;

    fn to_device(tensor: Self::Tensor, device: &Device<B>) -> Self::Tensor {
        B::float_to_device(tensor, device)
    }

    fn slice(tensor: Self::Tensor, slices: &[Slice]) -> Self::Tensor {
        B::float_slice(tensor, slices)
    }

    fn cat(tensors: Vec<Self::Tensor>, dim: usize) -> Self::Tensor {
        B::float_cat(tensors, dim)
    }
}

impl<B: Backend> ReduceKind<B> for Float {
    fn sum(parts: Vec<Self::Tensor>) -> Self::Tensor {
        parts
            .into_iter()
            .reduce(B::float_add)
            .expect("a group has a member")
    }

    /// Divides each part first, as NCCL averages floats, so a sum past the type's range still
    /// averages.
    fn mean(parts: Vec<Self::Tensor>) -> Self::Tensor {
        let members = parts.len();
        let parts = parts
            .into_iter()
            .map(|part| B::float_div_scalar(part, members.into()));
        <Self as ReduceKind<B>>::sum(parts.collect())
    }
}

impl<B: Backend> TransferKind<B> for Int {
    type Tensor = IntTensor<B>;

    fn to_device(tensor: Self::Tensor, device: &Device<B>) -> Self::Tensor {
        B::int_to_device(tensor, device)
    }

    fn slice(tensor: Self::Tensor, slices: &[Slice]) -> Self::Tensor {
        B::int_slice(tensor, slices)
    }

    fn cat(tensors: Vec<Self::Tensor>, dim: usize) -> Self::Tensor {
        B::int_cat(tensors, dim)
    }
}

impl<B: Backend> ReduceKind<B> for Int {
    fn sum(parts: Vec<Self::Tensor>) -> Self::Tensor {
        parts
            .into_iter()
            .reduce(B::int_add)
            .expect("a group has a member")
    }

    /// Divides the sum, so the mean truncates once, as the backend's integer division does.
    fn mean(parts: Vec<Self::Tensor>) -> Self::Tensor {
        let members = parts.len();
        B::int_div_scalar(<Self as ReduceKind<B>>::sum(parts), members.into())
    }
}

impl<B: Backend> TransferKind<B> for Bool {
    type Tensor = BoolTensor<B>;

    fn to_device(tensor: Self::Tensor, device: &Device<B>) -> Self::Tensor {
        B::bool_to_device(tensor, device)
    }

    fn slice(tensor: Self::Tensor, slices: &[Slice]) -> Self::Tensor {
        B::bool_slice(tensor, slices)
    }

    fn cat(tensors: Vec<Self::Tensor>, dim: usize) -> Self::Tensor {
        B::bool_cat(tensors, dim)
    }
}

impl<B: Backend, K: TransferKind<B>> TransferCollectives<B, K> {
    /// See [`CollectiveOps::float_all_gather`].
    pub fn all_gather(tensors: Vec<K::Tensor>, dim: usize) -> Vec<K::Tensor> {
        Self::devices(&tensors)
            .iter()
            .map(|device| K::cat(Self::moved_to(&tensors, device), dim))
            .collect()
    }

    /// See [`CollectiveOps::float_all_to_all`]. Only the chunks that change member move.
    pub fn all_to_all(
        tensors: Vec<K::Tensor>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<K::Tensor> {
        let chunks = Chunks::new(&tensors, split_dim);
        Self::devices(&tensors)
            .iter()
            .enumerate()
            .map(|(member, device)| {
                let received = tensors
                    .iter()
                    .map(|tensor| {
                        let chunk = Self::chunk(tensor.clone(), split_dim, chunks.range(member));
                        K::to_device(chunk, device)
                    })
                    .collect();
                K::cat(received, concat_dim)
            })
            .collect()
    }

    /// See [`CollectiveOps::float_broadcast`].
    pub fn broadcast(tensor: K::Tensor, devices: &[Device<B>]) -> Vec<K::Tensor> {
        devices
            .iter()
            .map(|device| K::to_device(tensor.clone(), device))
            .collect()
    }

    fn devices(tensors: &[K::Tensor]) -> Vec<Device<B>> {
        tensors.iter().map(TensorMetadata::device).collect()
    }

    fn moved_to(tensors: &[K::Tensor], device: &Device<B>) -> Vec<K::Tensor> {
        tensors
            .iter()
            .map(|tensor| K::to_device(tensor.clone(), device))
            .collect()
    }

    fn chunk(tensor: K::Tensor, dim: usize, range: Range<usize>) -> K::Tensor {
        let slices: Vec<Slice> = (0..=dim)
            .map(|axis| match axis == dim {
                true => Slice::new(range.start as isize, Some(range.end as isize), 1),
                false => Slice::full(),
            })
            .collect();
        K::slice(tensor, &slices)
    }
}

impl<B: Backend, K: ReduceKind<B>> TransferCollectives<B, K> {
    /// See [`CollectiveOps::float_all_reduce`]. Reduces once on the first member and broadcasts
    /// the result, so each tensor crosses a link once each way rather than to every member.
    pub fn all_reduce(tensors: Vec<K::Tensor>, op: ReduceOperation) -> Vec<K::Tensor> {
        let devices = Self::devices(&tensors);
        let Some(root) = devices.first() else {
            return Vec::new();
        };
        let reduced = Self::reduced(Self::moved_to(&tensors, root), op);
        Self::broadcast(reduced, &devices)
    }

    /// See [`CollectiveOps::float_reduce_scatter`]. Only each member's chunk moves to it.
    pub fn reduce_scatter(
        tensors: Vec<K::Tensor>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<K::Tensor> {
        let chunks = Chunks::new(&tensors, dim);
        Self::devices(&tensors)
            .iter()
            .enumerate()
            .map(|(member, device)| {
                let parts = tensors
                    .iter()
                    .map(|tensor| {
                        let chunk = Self::chunk(tensor.clone(), dim, chunks.range(member));
                        K::to_device(chunk, device)
                    })
                    .collect();
                Self::reduced(parts, op)
            })
            .collect()
    }

    fn reduced(parts: Vec<K::Tensor>, op: ReduceOperation) -> K::Tensor {
        match op {
            ReduceOperation::Sum => K::sum(parts),
            ReduceOperation::Mean => K::mean(parts),
        }
    }
}

impl Chunks {
    fn new<T: TensorMetadata>(tensors: &[T], dim: usize) -> Self {
        let chunks = Self {
            len: tensors.first().map_or(0, |tensor| tensor.shape()[dim]),
            members: tensors.len(),
        };
        assert!(
            chunks.len >= chunks.members,
            "dimension {dim} holds {} elements, fewer than the {} members it splits over",
            chunks.len,
            chunks.members,
        );
        chunks
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
