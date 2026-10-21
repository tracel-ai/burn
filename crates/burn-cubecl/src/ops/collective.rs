//! A member is one of the tensors a collective is given, as
//! [`CollectiveOps`](burn_backend::ops::CollectiveOps) defines it: member `i` is position `i` of
//! the input, on its own device, and receives position `i` of the output.

use burn_backend::{
    DeviceOps,
    cubecl::{dtype_to_elem_type, reduce_operation_to_cubecl},
    distributed::ReduceOperation,
    ops::{CollectiveOps, EmulatedCollectives},
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use core::cmp::Ordering;
use cubecl::{
    client::{ClientGroup, MemberBuffers},
    ir::ElemType,
};

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
        CollectivePlan::in_device_order(tensors, |tensors| {
            match CollectivePlan::all_reduce(tensors) {
                CollectivePlan::Native(group) => group.all_reduce(op),
                CollectivePlan::Emulated(tensors) => {
                    EmulatedCollectives::<Self>::float_all_reduce(tensors, op)
                }
            }
        })
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        match CollectivePlan::all_gather(tensors) {
            CollectivePlan::Native(group) => group.all_gather(dim),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::float_all_gather(tensors, dim)
            }
        }
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        match CollectivePlan::reduce_scatter(tensors, dim) {
            CollectivePlan::Native(group) => group.reduce_scatter(dim, op),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::float_reduce_scatter(tensors, dim, op)
            }
        }
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        match CollectivePlan::all_to_all(tensors, split_dim, concat_dim) {
            CollectivePlan::Native(group) => group.all_to_all(split_dim),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::float_all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        CollectivePlan::in_device_order(tensors, |tensors| {
            match CollectivePlan::all_reduce(tensors) {
                CollectivePlan::Native(group) => group.all_reduce(op),
                CollectivePlan::Emulated(tensors) => {
                    EmulatedCollectives::<Self>::int_all_reduce(tensors, op)
                }
            }
        })
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        match CollectivePlan::all_gather(tensors) {
            CollectivePlan::Native(group) => group.all_gather(dim),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::int_all_gather(tensors, dim)
            }
        }
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        match CollectivePlan::reduce_scatter(tensors, dim) {
            CollectivePlan::Native(group) => group.reduce_scatter(dim, op),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::int_reduce_scatter(tensors, dim, op)
            }
        }
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        match CollectivePlan::all_to_all(tensors, split_dim, concat_dim) {
            CollectivePlan::Native(group) => group.all_to_all(split_dim),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::int_all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        match CollectivePlan::all_gather(tensors) {
            CollectivePlan::Native(group) => group.all_gather(dim),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::bool_all_gather(tensors, dim)
            }
        }
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        match CollectivePlan::all_to_all(tensors, split_dim, concat_dim) {
            CollectivePlan::Native(group) => group.all_to_all(split_dim),
            CollectivePlan::Emulated(tensors) => {
                EmulatedCollectives::<Self>::bool_all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }
}

/// How a collective over the members runs.
enum CollectivePlan {
    /// Natively, by cubecl over a group of the members' devices: NCCL on CUDA.
    Native(NativeGroup),
    /// Emulated with transfers between devices, slices, concatenations and additions.
    Emulated(Vec<CubeTensor>),
}

/// The members' tensors with the cubecl group that runs a collective over them natively: alike
/// tensors on distinct devices of one runtime with a native collective, in ascending order.
struct NativeGroup {
    tensors: Vec<CubeTensor>,
    clients: ClientGroup,
}

/// A collective's dimension moved first: cubecl cuts and joins whole buffers, which is cutting and
/// joining along the leading dimension of a dense tensor.
struct LeadingDim {
    dim: usize,
}

impl CollectivePlan {
    fn all_reduce(tensors: Vec<CubeTensor>) -> Self {
        Self::native_if(tensors, NativeGroup::reduces)
    }

    /// Runs `all_reduce` over the members in the ascending device order a native group takes, and
    /// returns each member's output at its position. Every member receives the same result.
    fn in_device_order(
        tensors: Vec<CubeTensor>,
        all_reduce: impl FnOnce(Vec<CubeTensor>) -> Vec<CubeTensor>,
    ) -> Vec<CubeTensor> {
        let mut members: Vec<_> = tensors.into_iter().enumerate().collect();
        members.sort_by_key(|(_, tensor)| tensor.device.id());
        let (positions, sorted): (Vec<_>, Vec<_>) = members.into_iter().unzip();
        let mut outputs: Vec<_> = positions.into_iter().zip(all_reduce(sorted)).collect();
        outputs.sort_by_key(|(position, _)| *position);
        outputs.into_iter().map(|(_, output)| output).collect()
    }

    fn all_gather(tensors: Vec<CubeTensor>) -> Self {
        Self::native_if(tensors, |_| true)
    }

    fn reduce_scatter(tensors: Vec<CubeTensor>, dim: usize) -> Self {
        Self::native_if(tensors, |group| group.reduces() && group.splits_evenly(dim))
    }

    fn all_to_all(tensors: Vec<CubeTensor>, split_dim: usize, concat_dim: usize) -> Self {
        // cubecl splits and concatenates along one dimension, so two different ones are emulated.
        Self::native_if(tensors, |group| {
            split_dim == concat_dim && group.splits_evenly(split_dim)
        })
    }

    /// Native when the members form a group that `supports` the collective, emulated otherwise.
    fn native_if(tensors: Vec<CubeTensor>, supports: impl FnOnce(&NativeGroup) -> bool) -> Self {
        match NativeGroup::new(tensors) {
            Ok(group) if supports(&group) => Self::Native(group),
            Ok(group) => Self::Emulated(group.tensors),
            Err(tensors) => Self::Emulated(tensors),
        }
    }
}

impl NativeGroup {
    /// The group over the members' devices, or the tensors back when cubecl cannot group them.
    fn new(tensors: Vec<CubeTensor>) -> Result<Self, Vec<CubeTensor>> {
        let alike = tensors.windows(2).all(|pair| {
            pair[0].meta.shape() == pair[1].meta.shape() && pair[0].dtype == pair[1].dtype
        });
        let clients = tensors.iter().map(|tensor| tensor.client.clone()).collect();
        match alike.then(|| ClientGroup::new(clients).ok()).flatten() {
            Some(clients) => Ok(Self { tensors, clients }),
            None => Err(tensors),
        }
    }

    fn reduces(&self) -> bool {
        self.clients.reduces(self.elem())
    }

    fn splits_evenly(&self, dim: usize) -> bool {
        let members = self.tensors.len();
        let len = self.tensors[0].meta.shape()[dim];
        len >= members && len.is_multiple_of(members)
    }

    /// Each source is its own output when nothing else holds it.
    fn all_reduce(self, op: ReduceOperation) -> Vec<CubeTensor> {
        let elem = self.elem();
        let sources: Vec<CubeTensor> = self.tensors.into_iter().map(Self::dense).collect();
        let outputs: Vec<CubeTensor> = sources
            .iter()
            .map(|source| match source.can_mut() {
                true => source.clone(),
                false => Self::output_like(source, |len| len),
            })
            .collect();
        let buffers = Self::buffers(&sources, &outputs);
        self.clients
            .all_reduce(buffers, elem, reduce_operation_to_cubecl(op));
        outputs
    }

    fn all_gather(self, dim: usize) -> Vec<CubeTensor> {
        let members = self.tensors.len();
        self.along(
            dim,
            |len| len * members,
            |clients, buffers| clients.all_gather(buffers),
        )
    }

    fn reduce_scatter(self, dim: usize, op: ReduceOperation) -> Vec<CubeTensor> {
        let members = self.tensors.len();
        let elem = self.elem();
        self.along(
            dim,
            |len| len / members,
            |clients, buffers| {
                clients.reduce_scatter(buffers, elem, reduce_operation_to_cubecl(op))
            },
        )
    }

    fn all_to_all(self, dim: usize) -> Vec<CubeTensor> {
        self.along(
            dim,
            |len| len,
            |clients, buffers| clients.all_to_all(buffers),
        )
    }

    /// Runs `collective` with `dim` leading, each output's leading length `len` of its source's.
    fn along(
        self,
        dim: usize,
        len: impl Fn(usize) -> usize,
        collective: impl FnOnce(&ClientGroup, Vec<MemberBuffers>),
    ) -> Vec<CubeTensor> {
        let leading = LeadingDim::new(dim);
        let sources: Vec<CubeTensor> = self
            .tensors
            .into_iter()
            .map(|tensor| Self::dense(leading.moved_ahead(tensor)))
            .collect();
        let outputs: Vec<CubeTensor> = sources
            .iter()
            .map(|source| Self::output_like(source, &len))
            .collect();
        collective(&self.clients, Self::buffers(&sources, &outputs));
        leading.restored(outputs)
    }

    fn elem(&self) -> ElemType {
        dtype_to_elem_type(self.tensors[0].dtype)
    }

    fn buffers(sources: &[CubeTensor], outputs: &[CubeTensor]) -> Vec<MemberBuffers> {
        sources
            .iter()
            .zip(outputs)
            .map(|(source, output)| MemberBuffers {
                src: source.handle.clone(),
                dst: output.handle.clone(),
            })
            .collect()
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

    /// An empty dense tensor like `source`, its leading length `len` of the source's.
    fn output_like(source: &CubeTensor, len: impl Fn(usize) -> usize) -> CubeTensor {
        let mut shape = source.meta.shape().clone();
        shape[0] = len(shape[0]);
        empty_device_contiguous_dtype(
            source.client.clone(),
            source.device.clone(),
            shape,
            source.dtype,
        )
    }
}

impl LeadingDim {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn moved_ahead(&self, tensor: CubeTensor) -> CubeTensor {
        if self.dim == 0 {
            return tensor;
        }
        let axes: Vec<usize> = core::iter::once(self.dim)
            .chain((0..tensor.meta.shape().num_dims()).filter(|axis| *axis != self.dim))
            .collect();
        permute(tensor, &axes)
    }

    /// The outputs with the dimension back where it was.
    fn restored(&self, tensors: Vec<CubeTensor>) -> Vec<CubeTensor> {
        if self.dim == 0 {
            return tensors;
        }
        tensors
            .into_iter()
            .map(|tensor| {
                let axes: Vec<usize> = (0..tensor.meta.shape().num_dims())
                    .map(|axis| match axis.cmp(&self.dim) {
                        Ordering::Less => axis + 1,
                        Ordering::Equal => 0,
                        Ordering::Greater => axis,
                    })
                    .collect();
                permute(tensor, &axes)
            })
            .collect()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::CubeDevice;
    use burn_backend::DType;
    use burn_std::Shape;
    use cubecl::cuda::CudaDevice;

    /// A collective runs natively only where a group forms, reduces the type and splits evenly;
    /// anything else must be emulated, and a plan that silently stopped holding would pass every
    /// value test.
    #[test]
    #[ignore = "needs two CUDA devices and NCCL"]
    fn each_collective_runs_natively_only_where_the_members_allow_it() {
        let devices = [0, 1].map(|index| CubeDevice::Cuda(CudaDevice { index }));
        let members = |order: [usize; 2], rows: [usize; 2], dtype: DType| -> Vec<CubeTensor> {
            order
                .iter()
                .zip(rows)
                .map(|(member, rows)| {
                    let device = devices[*member].clone();
                    let shape = Shape::new([rows, 2]);
                    empty_device_contiguous_dtype(device.client(), device, shape, dtype)
                })
                .collect()
        };
        let native = |plan: CollectivePlan| matches!(plan, CollectivePlan::Native(_));

        let even = || members([0, 1], [2, 2], DType::F32);
        assert!(native(CollectivePlan::all_reduce(even())));
        assert!(native(CollectivePlan::all_gather(even())));
        assert!(native(CollectivePlan::reduce_scatter(even(), 0)));
        assert!(native(CollectivePlan::all_to_all(even(), 1, 1)));
        assert!(!native(CollectivePlan::all_to_all(even(), 0, 1)));

        let uneven = || members([0, 1], [3, 3], DType::F32);
        assert!(native(CollectivePlan::all_gather(uneven())));
        assert!(!native(CollectivePlan::reduce_scatter(uneven(), 0)));
        assert!(!native(CollectivePlan::all_to_all(uneven(), 0, 0)));

        let unreduced = || members([0, 1], [2, 2], DType::I16);
        assert!(native(CollectivePlan::all_gather(unreduced())));
        assert!(!native(CollectivePlan::all_reduce(unreduced())));

        let descending = || members([1, 0], [2, 2], DType::F32);
        assert!(!native(CollectivePlan::all_reduce(descending())));
        let mut sorted_native = false;
        let outputs = CollectivePlan::in_device_order(descending(), |sorted| {
            sorted_native = native(CollectivePlan::all_reduce(sorted.clone()));
            sorted
        });
        assert!(sorted_native);
        let output_devices: Vec<_> = outputs.iter().map(|output| output.device.clone()).collect();
        assert_eq!(output_devices, [devices[1].clone(), devices[0].clone()]);
        assert!(!native(CollectivePlan::all_gather(members(
            [0, 0],
            [2, 2],
            DType::F32
        ))));
        assert!(!native(CollectivePlan::all_gather(members(
            [0, 1],
            [2, 3],
            DType::F32
        ))));
    }
}
