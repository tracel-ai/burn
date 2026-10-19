use burn_backend::{
    DeviceOps,
    cubecl::{dtype_to_elem_type, reduce_operation_to_cubecl},
    distributed::ReduceOperation,
    ops::{CollectiveOps, TransferCollectives},
    tensor::{Bool, BoolTensor, Float, FloatTensor, Int, IntTensor},
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
        let order = DeviceOrder::new(&tensors);
        let reduced = match Route::all_reduce(order.sorted(tensors)) {
            Route::Group(group) => group.all_reduce(op),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Float>::all_reduce(tensors, op)
            }
        };
        order.restored(reduced)
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        match Route::all_gather(tensors) {
            Route::Group(group) => group.all_gather(dim),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Float>::all_gather(tensors, dim)
            }
        }
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        match Route::reduce_scatter(tensors, dim) {
            Route::Group(group) => group.reduce_scatter(dim, op),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Float>::reduce_scatter(tensors, dim, op)
            }
        }
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        match Route::all_to_all(tensors, split_dim, concat_dim) {
            Route::Group(group) => group.all_to_all(split_dim),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Float>::all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        let order = DeviceOrder::new(&tensors);
        let reduced = match Route::all_reduce(order.sorted(tensors)) {
            Route::Group(group) => group.all_reduce(op),
            Route::Transfers(tensors) => TransferCollectives::<Self, Int>::all_reduce(tensors, op),
        };
        order.restored(reduced)
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        match Route::all_gather(tensors) {
            Route::Group(group) => group.all_gather(dim),
            Route::Transfers(tensors) => TransferCollectives::<Self, Int>::all_gather(tensors, dim),
        }
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        match Route::reduce_scatter(tensors, dim) {
            Route::Group(group) => group.reduce_scatter(dim, op),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Int>::reduce_scatter(tensors, dim, op)
            }
        }
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        match Route::all_to_all(tensors, split_dim, concat_dim) {
            Route::Group(group) => group.all_to_all(split_dim),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Int>::all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        match Route::all_gather(tensors) {
            Route::Group(group) => group.all_gather(dim),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Bool>::all_gather(tensors, dim)
            }
        }
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        match Route::all_to_all(tensors, split_dim, concat_dim) {
            Route::Group(group) => group.all_to_all(split_dim),
            Route::Transfers(tensors) => {
                TransferCollectives::<Self, Bool>::all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }
}

/// Where a collective over the members runs.
enum Route {
    /// Over cubecl's group of the members' devices: NCCL on CUDA.
    Group(DeviceGroup),
    /// Built from transfers between devices, slices, concatenations and additions.
    Transfers(Vec<CubeTensor>),
}

/// The members' tensors, alike in shape and type, on distinct devices of one runtime with a
/// transport, in ascending order: what cubecl runs a collective over.
struct DeviceGroup {
    tensors: Vec<CubeTensor>,
    clients: ClientGroup,
}

/// The order that sorts members by device, as cubecl's groups require, and back. Only an
/// all-reduce takes it, since only its result is the same in any member order.
struct DeviceOrder {
    members: Vec<usize>,
}

/// A collective's dimension moved first: cubecl cuts and joins whole buffers, which is cutting and
/// joining along the leading dimension of a dense tensor.
struct LeadingDim {
    dim: usize,
}

impl Route {
    fn all_reduce(tensors: Vec<CubeTensor>) -> Self {
        Self::group_if(tensors, DeviceGroup::reduces)
    }

    fn all_gather(tensors: Vec<CubeTensor>) -> Self {
        Self::group_if(tensors, |_| true)
    }

    fn reduce_scatter(tensors: Vec<CubeTensor>, dim: usize) -> Self {
        Self::group_if(tensors, |group| group.reduces() && group.splits_evenly(dim))
    }

    fn all_to_all(tensors: Vec<CubeTensor>, split_dim: usize, concat_dim: usize) -> Self {
        // cubecl splits and concatenates along one dimension, so two different ones compose.
        Self::group_if(tensors, |group| {
            split_dim == concat_dim && group.splits_evenly(split_dim)
        })
    }

    /// The group, when the members form one and it `runs` the collective.
    fn group_if(tensors: Vec<CubeTensor>, runs: impl FnOnce(&DeviceGroup) -> bool) -> Self {
        let Some(clients) = DeviceGroup::clients(&tensors) else {
            return Self::Transfers(tensors);
        };
        let group = DeviceGroup { tensors, clients };
        match runs(&group) {
            true => Self::Group(group),
            false => Self::Transfers(group.tensors),
        }
    }
}

impl DeviceGroup {
    /// cubecl's group of the members' devices, when the members hold alike tensors on devices
    /// it groups.
    fn clients(tensors: &[CubeTensor]) -> Option<ClientGroup> {
        let alike = tensors.windows(2).all(|pair| {
            pair[0].meta.shape() == pair[1].meta.shape() && pair[0].dtype == pair[1].dtype
        });
        let clients = tensors.iter().map(|tensor| tensor.client.clone()).collect();
        alike.then(|| ClientGroup::new(clients).ok()).flatten()
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

impl DeviceOrder {
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

    /// NCCL runs only where a group forms, reduces the type and splits evenly; anything else must
    /// compose, and a route that silently stopped holding would pass every value test.
    #[test]
    #[ignore = "needs two CUDA devices and NCCL"]
    fn each_collective_routes_over_nccl_only_where_the_members_allow_it() {
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
        let grouped = |route: Route| matches!(route, Route::Group(_));

        let even = || members([0, 1], [2, 2], DType::F32);
        assert!(grouped(Route::all_reduce(even())));
        assert!(grouped(Route::all_gather(even())));
        assert!(grouped(Route::reduce_scatter(even(), 0)));
        assert!(grouped(Route::all_to_all(even(), 1, 1)));
        assert!(!grouped(Route::all_to_all(even(), 0, 1)));

        let uneven = || members([0, 1], [3, 3], DType::F32);
        assert!(grouped(Route::all_gather(uneven())));
        assert!(!grouped(Route::reduce_scatter(uneven(), 0)));
        assert!(!grouped(Route::all_to_all(uneven(), 0, 0)));

        let unreduced = || members([0, 1], [2, 2], DType::I16);
        assert!(grouped(Route::all_gather(unreduced())));
        assert!(!grouped(Route::all_reduce(unreduced())));

        assert!(!grouped(Route::all_gather(members(
            [1, 0],
            [2, 2],
            DType::F32
        ))));
        assert!(!grouped(Route::all_gather(members(
            [0, 0],
            [2, 2],
            DType::F32
        ))));
        assert!(!grouped(Route::all_gather(members(
            [0, 1],
            [2, 3],
            DType::F32
        ))));
    }
}
