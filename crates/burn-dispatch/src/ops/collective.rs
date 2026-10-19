use alloc::vec::Vec;
use burn_backend::{
    DeviceOps, TensorMetadata,
    distributed::ReduceOperation,
    ops::{BoolTensors, CollectiveOps, FloatTensors, IntTensors, TransferCollectives},
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_backend_extension::backend_dispatch;

use crate::{Dispatch, DispatchDevice, DispatchTensor};

// Members on one backend run its collective; members spanning backends compose, since only
// dispatch's own transfers cross them. A broadcast always composes, as its devices may span them.
impl CollectiveOps<Self> for Dispatch {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::float_all_reduce_dispatch(tensors, op),
            false => TransferCollectives::<Self, FloatTensors>::all_reduce(tensors, op),
        }
    }

    fn float_all_gather(tensors: Vec<FloatTensor<Self>>, dim: usize) -> Vec<FloatTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::float_all_gather_dispatch(tensors, dim),
            false => TransferCollectives::<Self, FloatTensors>::all_gather(tensors, dim),
        }
    }

    fn float_reduce_scatter(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::float_reduce_scatter_dispatch(tensors, dim, op),
            false => TransferCollectives::<Self, FloatTensors>::reduce_scatter(tensors, dim, op),
        }
    }

    fn float_all_to_all(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::float_all_to_all_dispatch(tensors, split_dim, concat_dim),
            false => TransferCollectives::<Self, FloatTensors>::all_to_all(
                tensors, split_dim, concat_dim,
            ),
        }
    }

    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::int_all_reduce_dispatch(tensors, op),
            false => TransferCollectives::<Self, IntTensors>::all_reduce(tensors, op),
        }
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::int_all_gather_dispatch(tensors, dim),
            false => TransferCollectives::<Self, IntTensors>::all_gather(tensors, dim),
        }
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::int_reduce_scatter_dispatch(tensors, dim, op),
            false => TransferCollectives::<Self, IntTensors>::reduce_scatter(tensors, dim, op),
        }
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::int_all_to_all_dispatch(tensors, split_dim, concat_dim),
            false => {
                TransferCollectives::<Self, IntTensors>::all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::bool_all_gather_dispatch(tensors, dim),
            false => TransferCollectives::<Self, BoolTensors>::all_gather(tensors, dim),
        }
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        match Self::on_one_backend(&tensors) {
            true => Self::bool_all_to_all_dispatch(tensors, split_dim, concat_dim),
            false => {
                TransferCollectives::<Self, BoolTensors>::all_to_all(tensors, split_dim, concat_dim)
            }
        }
    }
}

#[backend_dispatch]
impl Dispatch {
    #[backend_dispatch(skip)]
    fn on_one_backend(tensors: &[DispatchTensor]) -> bool {
        let backend = |tensor: &DispatchTensor| {
            DispatchDevice::decode_type_id(tensor.device().id().type_id).0
        };
        !tensors.is_empty()
            && tensors
                .windows(2)
                .all(|pair| backend(&pair[0]) == backend(&pair[1]))
    }

    fn float_all_reduce_dispatch(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        B::float_all_reduce(tensors, op)
    }

    fn float_all_gather_dispatch(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        B::float_all_gather(tensors, dim)
    }

    fn float_reduce_scatter_dispatch(
        tensors: Vec<FloatTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        B::float_reduce_scatter(tensors, dim, op)
    }

    fn float_all_to_all_dispatch(
        tensors: Vec<FloatTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<FloatTensor<Self>> {
        B::float_all_to_all(tensors, split_dim, concat_dim)
    }

    fn int_all_reduce_dispatch(
        tensors: Vec<IntTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        B::int_all_reduce(tensors, op)
    }

    fn int_all_gather_dispatch(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        B::int_all_gather(tensors, dim)
    }

    fn int_reduce_scatter_dispatch(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        B::int_reduce_scatter(tensors, dim, op)
    }

    fn int_all_to_all_dispatch(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        B::int_all_to_all(tensors, split_dim, concat_dim)
    }

    fn bool_all_gather_dispatch(
        tensors: Vec<BoolTensor<Self>>,
        dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        B::bool_all_gather(tensors, dim)
    }

    fn bool_all_to_all_dispatch(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        B::bool_all_to_all(tensors, split_dim, concat_dim)
    }
}
