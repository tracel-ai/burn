use alloc::sync::Arc;

use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    std::throughput::roofline_bounds,
    tune::{BoundsGenerator, Thresholds},
};
use cubek::matmul::{
    definition::{MatmulCost, MatmulGlobalElems},
    strategy::MatmulAutotuneKey,
};

use crate::{CubeRuntime, kernel::matmul::tune::base::Inputs};

type BoundsGen<R> = dyn BoundsGenerator<MatmulAutotuneKey, Inputs<R>> + Send + Sync;

/// Fraction of the modeled roofline a matmul candidate is expected to reach.
const THRESHOLDS: Thresholds = Thresholds::uniform(1.0);

/// Creates a closure that calculates performance bounds for matrix multiplication autotuning.
pub(super) fn create_matmul_bounds<R: CubeRuntime>() -> Arc<BoundsGen<R>> {
    Arc::new(|_key: &MatmulAutotuneKey, tensors: &Inputs<R>| {
        let client = &tensors.0.client;
        let cost = cost(tensors);

        roofline_bounds(client, cost.compute_key(client), cost.work(), THRESHOLDS)
    })
}

fn cost<R: CubeRuntime>((lhs, rhs, out): &Inputs<R>) -> MatmulCost {
    let lhs_shape = lhs.meta.shape();
    let rhs_shape = rhs.meta.shape();
    let ndims = lhs_shape.len();

    MatmulCost {
        batches: lhs_shape[..ndims - 2].iter().product(),
        m: lhs_shape[ndims - 2],
        k: lhs_shape[ndims - 1],
        n: rhs_shape[ndims - 1],
        elems: MatmulGlobalElems {
            lhs: dtype_to_storage_type(lhs.dtype),
            rhs: dtype_to_storage_type(rhs.dtype),
            out: dtype_to_storage_type(out.dtype),
        },
    }
}
