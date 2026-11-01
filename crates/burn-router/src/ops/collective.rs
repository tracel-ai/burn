use alloc::vec::Vec;
use burn_backend::{distributed::ReduceOperation, ops::CollectiveOps, tensor::FloatTensor};

use crate::{BackendRouter, RouterChannel};

impl<R: RouterChannel> CollectiveOps<Self> for BackendRouter<R> {
    fn float_all_reduce(
        tensors: Vec<FloatTensor<Self>>,
        op: ReduceOperation,
    ) -> Vec<FloatTensor<Self>> {
        R::float_all_reduce(tensors, op)
    }
}
