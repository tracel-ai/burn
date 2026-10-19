use burn_backend::{
    DeviceOps,
    cubecl::{dtype_to_elem_type, reduce_operation_to_cubecl},
    distributed::ReduceOperation,
    ops::{CollectiveOps, ComposedCollectives, ComposedReductions},
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::Shape;
use cubecl::client::{ClientGroup, MemberBuffers};

use crate::{
    CubeBackend,
    kernel::{copy_contiguous, into_contiguous},
    ops::{numeric::empty_device_contiguous_dtype, permute},
    tensor::CubeTensor,
};

impl CollectiveOps<Self> for CubeBackend {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        Members::all_reduce(tensors, op, |tensors| {
            ComposedReductions::<Self, _>::float().all_reduce(tensors, op)
        })
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        Members::new(tensors).all_gather(dim, |tensors| {
            ComposedCollectives::<Self, _>::float().all_gather(tensors, dim)
        })
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        Members::new(tensors).reduce_scatter(dim, op, |tensors| {
            ComposedReductions::<Self, _>::float().reduce_scatter(tensors, dim, op)
        })
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        Members::new(tensors).all_to_all(split_dim, concat_dim, |tensors| {
            ComposedCollectives::<Self, _>::float().all_to_all(tensors, split_dim, concat_dim)
        })
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        Members::all_reduce(tensors, op, |tensors| {
            ComposedReductions::<Self, _>::int().all_reduce(tensors, op)
        })
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        Members::new(tensors).all_gather(dim, |tensors| {
            ComposedCollectives::<Self, _>::int().all_gather(tensors, dim)
        })
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        Members::new(tensors).reduce_scatter(dim, op, |tensors| {
            ComposedReductions::<Self, _>::int().reduce_scatter(tensors, dim, op)
        })
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        Members::new(tensors).all_to_all(split_dim, concat_dim, |tensors| {
            ComposedCollectives::<Self, _>::int().all_to_all(tensors, split_dim, concat_dim)
        })
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        Members::new(tensors).all_gather(dim, |tensors| {
            ComposedCollectives::<Self, _>::bool().all_gather(tensors, dim)
        })
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        Members::new(tensors).all_to_all(split_dim, concat_dim, |tensors| {
            ComposedCollectives::<Self, _>::bool().all_to_all(tensors, split_dim, concat_dim)
        })
    }
}

/// Every member's tensor, and the group of their devices when cubecl runs collectives over it:
/// distinct devices in ascending order, on one runtime with a transport, holding tensors of one
/// shape and type. Anything else runs `composed`.
struct Members {
    tensors: Vec<CubeTensor>,
    group: Option<ClientGroup>,
}

/// A member order and the ascending device order cubecl's groups take, for a collective whose
/// result does not depend on the order.
struct AscendingOrder {
    members: Vec<usize>,
}

/// Every member's buffers for one collective, and the tensors its destinations become.
struct GroupBuffers {
    buffers: Vec<MemberBuffers>,
    outputs: Vec<CubeTensor>,
}

impl Members {
    fn new(tensors: Vec<CubeTensor>) -> Self {
        let alike = tensors.windows(2).all(|pair| {
            pair[0].meta.shape() == pair[1].meta.shape() && pair[0].dtype == pair[1].dtype
        });
        let clients = tensors.iter().map(|tensor| tensor.client.clone()).collect();
        let group = alike.then(|| ClientGroup::new(clients).ok()).flatten();
        Self { tensors, group }
    }

    fn all_reduce(
        tensors: Vec<CubeTensor>,
        op: ReduceOperation,
        composed: impl FnOnce(Vec<CubeTensor>) -> Vec<CubeTensor>,
    ) -> Vec<CubeTensor> {
        let order = AscendingOrder::new(&tensors);
        let members = Self::new(order.sorted(tensors));
        let reduced = match members.reducing() {
            Some(group) => {
                let group = group.clone();
                let elem = dtype_to_elem_type(members.tensors[0].dtype);
                let parts = members.in_place_where_possible();
                group.all_reduce(parts.buffers, elem, reduce_operation_to_cubecl(op));
                parts.outputs
            }
            None => composed(members.tensors),
        };
        order.restored(reduced)
    }

    fn all_gather(
        self,
        dim: usize,
        composed: impl FnOnce(Vec<CubeTensor>) -> Vec<CubeTensor>,
    ) -> Vec<CubeTensor> {
        let Some(group) = self.group else {
            return composed(self.tensors);
        };
        let members = self.tensors.len();
        let front = Front::new(dim);
        let parts = front.buffers(self.tensors, |len| len * members);
        group.all_gather(parts.buffers);
        front.restored(parts.outputs)
    }

    fn reduce_scatter(
        self,
        dim: usize,
        op: ReduceOperation,
        composed: impl FnOnce(Vec<CubeTensor>) -> Vec<CubeTensor>,
    ) -> Vec<CubeTensor> {
        let members = self.tensors.len();
        let Some(group) = self.reducing().filter(|_| self.splits_evenly(dim)).cloned() else {
            return composed(self.tensors);
        };
        let elem = dtype_to_elem_type(self.tensors[0].dtype);
        let front = Front::new(dim);
        let parts = front.buffers(self.tensors, |len| len / members);
        group.reduce_scatter(parts.buffers, elem, reduce_operation_to_cubecl(op));
        front.restored(parts.outputs)
    }

    fn all_to_all(
        self,
        split_dim: usize,
        concat_dim: usize,
        composed: impl FnOnce(Vec<CubeTensor>) -> Vec<CubeTensor>,
    ) -> Vec<CubeTensor> {
        // cubecl splits and concatenates along one dimension, so two different ones compose.
        let runs = split_dim == concat_dim && self.splits_evenly(split_dim);
        let Some(group) = self.group.filter(|_| runs) else {
            return composed(self.tensors);
        };
        let front = Front::new(split_dim);
        let parts = front.buffers(self.tensors, |len| len);
        group.all_to_all(parts.buffers);
        front.restored(parts.outputs)
    }

    /// The group when its devices reduce the members' type.
    fn reducing(&self) -> Option<&ClientGroup> {
        let group = self.group.as_ref()?;
        let elem = dtype_to_elem_type(self.tensors.first()?.dtype);
        group.reduces(elem).then_some(group)
    }

    fn splits_evenly(&self, dim: usize) -> bool {
        let members = self.tensors.len();
        self.tensors
            .first()
            .map(|tensor| tensor.meta.shape()[dim])
            .is_some_and(|len| len >= members && len.is_multiple_of(members))
    }

    /// Contiguous sources, each its own destination when nothing else holds it.
    fn in_place_where_possible(self) -> GroupBuffers {
        GroupBuffers::from_members(self.tensors.into_iter().map(|tensor| {
            let src = Self::dense(tensor);
            let dst = match src.can_mut() {
                true => src.clone(),
                false => Self::empty_like(&src, src.meta.shape().clone()),
            };
            [src, dst]
        }))
    }

    /// The tensor with its elements filling its buffer in row-major order, since cubecl's
    /// collectives move whole buffers.
    fn dense(tensor: CubeTensor) -> CubeTensor {
        let tensor = into_contiguous(tensor);
        match tensor.is_contiguous_buffer() {
            true => tensor,
            // `is_contiguous` ignores a dimension of size one, which can still hold a pitched row.
            false => copy_contiguous(tensor),
        }
    }

    fn empty_like(tensor: &CubeTensor, shape: Shape) -> CubeTensor {
        empty_device_contiguous_dtype(
            tensor.client.clone(),
            tensor.device.clone(),
            shape,
            tensor.dtype,
        )
    }
}

impl GroupBuffers {
    /// Each member's source and destination, in member order.
    fn from_members(members: impl Iterator<Item = [CubeTensor; 2]>) -> Self {
        let mut parts = Self {
            buffers: Vec::new(),
            outputs: Vec::new(),
        };
        for [src, dst] in members {
            parts.buffers.push(MemberBuffers {
                src: src.handle.clone(),
                dst: dst.handle.clone(),
            });
            parts.outputs.push(dst);
        }
        parts
    }
}

/// A dimension a collective runs along, moved to the front: cubecl's collectives cut and join
/// whole buffers, which is cutting and joining along the front dimension of a contiguous tensor.
struct Front {
    dim: usize,
}

impl Front {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Each member's source with the dimension in front, and a destination whose front length
    /// is `dst_len` of the source's.
    fn buffers(&self, tensors: Vec<CubeTensor>, dst_len: impl Fn(usize) -> usize) -> GroupBuffers {
        GroupBuffers::from_members(tensors.into_iter().map(|tensor| {
            let src = Members::dense(self.moved_to_front(tensor));
            let mut shape = src.meta.shape().clone();
            shape[0] = dst_len(shape[0]);
            let dst = Members::empty_like(&src, shape);
            [src, dst]
        }))
    }

    /// The destinations with the dimension back where it was.
    fn restored(&self, tensors: Vec<CubeTensor>) -> Vec<CubeTensor> {
        if self.dim == 0 {
            return tensors;
        }
        tensors
            .into_iter()
            .map(|tensor| {
                let axes: Vec<usize> = (0..tensor.meta.shape().num_dims())
                    .map(|axis| match axis.cmp(&self.dim) {
                        core::cmp::Ordering::Less => axis + 1,
                        core::cmp::Ordering::Equal => 0,
                        core::cmp::Ordering::Greater => axis,
                    })
                    .collect();
                permute(tensor, &axes)
            })
            .collect()
    }

    fn moved_to_front(&self, tensor: CubeTensor) -> CubeTensor {
        if self.dim == 0 {
            return tensor;
        }
        let axes: Vec<usize> = core::iter::once(self.dim)
            .chain((0..tensor.meta.shape().num_dims()).filter(|axis| *axis != self.dim))
            .collect();
        permute(tensor, &axes)
    }
}

impl AscendingOrder {
    fn new(tensors: &[CubeTensor]) -> Self {
        let mut members: Vec<usize> = (0..tensors.len()).collect();
        members.sort_by_key(|member| tensors[*member].device.id());
        Self { members }
    }

    fn sorted(&self, tensors: Vec<CubeTensor>) -> Vec<CubeTensor> {
        let mut tensors: Vec<Option<CubeTensor>> = tensors.into_iter().map(Some).collect();
        self.members
            .iter()
            .map(|member| tensors[*member].take().expect("each member once"))
            .collect()
    }

    fn restored(&self, sorted: Vec<CubeTensor>) -> Vec<CubeTensor> {
        let mut restored: Vec<Option<CubeTensor>> = (0..sorted.len()).map(|_| None).collect();
        for (member, tensor) in self.members.iter().zip(sorted) {
            restored[*member] = Some(tensor);
        }
        restored
            .into_iter()
            .map(|tensor| tensor.expect("each member once"))
            .collect()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::CubeDevice;
    use burn_backend::DType;
    use cubecl::cuda::CudaDevice;

    /// NCCL runs only where a group forms, reduces the type and splits evenly; anything else must
    /// compose, and a choice that silently stopped holding would pass every value test.
    #[test]
    #[ignore = "needs two CUDA devices and NCCL"]
    fn nccl_runs_only_where_the_members_allow_it() {
        let devices = [0, 1].map(|index| CubeDevice::Cuda(CudaDevice { index }));
        let tensor = |member: usize, rows: usize, dtype: DType| {
            let device = devices[member].clone();
            empty_device_contiguous_dtype(device.client(), device, Shape::new([rows, 2]), dtype)
        };
        let members = |order: [usize; 2], rows: [usize; 2], dtype: DType| {
            let tensors = order.iter().zip(rows);
            Members::new(
                tensors
                    .map(|(member, rows)| tensor(*member, rows, dtype))
                    .collect(),
            )
        };

        let even = members([0, 1], [2, 2], DType::F32);
        assert!(even.group.is_some() && even.reducing().is_some());
        assert!(even.splits_evenly(0) && even.splits_evenly(1));
        assert!(!members([0, 1], [3, 3], DType::F32).splits_evenly(0));

        let unreduced = members([0, 1], [2, 2], DType::I16);
        assert!(unreduced.group.is_some() && unreduced.reducing().is_none());

        assert!(members([1, 0], [2, 2], DType::F32).group.is_none());
        assert!(members([0, 0], [2, 2], DType::F32).group.is_none());
        assert!(members([0, 1], [2, 3], DType::F32).group.is_none());
    }
}
