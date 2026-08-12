use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{std::throughput::roofline_bounds, tune::TunableSet};
use cubek::matmul::{
    definition::{MatmulCost, MatmulGlobalElems},
    strategy::MatmulAutotuneKey,
};

use crate::{CubeRuntime, kernel::autotune_bounds, kernel::matmul::tune::base::Inputs};

type MatmulTunables<R, Out> = TunableSet<MatmulAutotuneKey, Inputs<R>, Out>;

/// Registers the performance bounds used for matrix multiplication autotuning.
pub(super) fn with_matmul_bounds<R: CubeRuntime, Out: 'static>(
    set: MatmulTunables<R, Out>,
) -> MatmulTunables<R, Out> {
    autotune_bounds::with_bounds(set, |_key, tensors: &Inputs<R>, thresholds| {
        let client = &tensors.0.client;
        let cost = cost(tensors);

        roofline_bounds(client, cost.compute_key(client), cost.work(), thresholds)
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
