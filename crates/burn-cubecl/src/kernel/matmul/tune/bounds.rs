use std::sync::Arc;

use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    client::ComputeClient,
    ir::StorageType,
    std::throughput::{measure_launch_overhead, measure_peak_throughput},
    throughput::{CmmaDims, ComputeCmmaConfig, ThroughputKey, ThroughputMode},
    tune::{AutotuneBound, Bounds, BoundsGenerator},
};
use cubek::matmul::strategy::MatmulAutotuneKey;

use crate::{CubeRuntime, kernel::matmul::tune::base::Inputs};

type BoundsGen<R> = dyn BoundsGenerator<MatmulAutotuneKey, Inputs<R>, AutotuneBound> + Send + Sync;

/// Creates a closure that calculates performance bounds for matrix multiplication autotuning.
pub(super) fn create_matmul_bounds<R: CubeRuntime>(client: &ComputeClient<R>) -> Arc<BoundsGen<R>> {
    let owned_client = client.clone();

    let launch_overhead = measure_launch_overhead(&owned_client);

    Arc::new(
        move |_key: &MatmulAutotuneKey, tensors: &Inputs<R>| Bounds {
            bounds: autotune_bounds(&owned_client, tensors),
            launch_overhead,
        },
    )
}

/// Calculates the theoretical compute and memory throughput bounds for a specific matrix multiplication operation.
fn autotune_bounds<R: CubeRuntime>(
    client: &ComputeClient<R>,
    (lhs, rhs, out): &Inputs<R>,
) -> Vec<AutotuneBound> {
    let elem_lhs = dtype_to_storage_type(lhs.dtype);
    let elem_rhs = dtype_to_storage_type(rhs.dtype);
    let elem_out = dtype_to_storage_type(out.dtype);

    let compute_throughput = measure_peak_throughput(
        client,
        ThroughputKey {
            mode: compute_mode(client, &(elem_lhs, elem_rhs, elem_out)),
            dtype: elem_out.elem_type(),
        },
    );

    let memory_throughput = measure_peak_throughput(
        client,
        ThroughputKey {
            mode: ThroughputMode::Memory,
            dtype: elem_out.elem_type(),
        },
    );

    let lhs_shape = lhs.meta.shape();
    let rhs_shape = rhs.meta.shape();
    let ndims = lhs_shape.len();

    let m = lhs_shape[ndims - 2];
    let k = lhs_shape[ndims - 1];
    let n = rhs_shape[ndims - 1];
    let batches = lhs_shape[..ndims - 2].iter().product::<usize>();

    vec![
        AutotuneBound {
            ops_count: batches * m * n * (2 * k - 1), // Theoretical matmul compute operations
            throughput: compute_throughput.ops_per_s(),
            threshold: 1.0,
        },
        AutotuneBound {
            ops_count: lhs.meta.num_elements() + rhs.meta.num_elements() + out.meta.num_elements(), // Theoretical matmul memory reads and writes operations
            throughput: memory_throughput.ops_per_s(),
            threshold: 1.0,
        },
    ]
}

/// Determines the optimal compute mode for the given operation.
fn compute_mode<R: CubeRuntime>(
    client: &ComputeClient<R>,
    (elem_lhs, elem_rhs, elem_out): &(StorageType, StorageType, StorageType),
) -> ThroughputMode {
    let max_cmma = client
        .properties()
        .features
        .matmul
        .cmma
        .iter()
        .filter(|c| c.a_type == *elem_lhs && c.b_type == *elem_rhs && c.cd_type == *elem_out)
        .max_by_key(|c| c.m * c.n * c.k);

    if let Some(config) = max_cmma {
        ThroughputMode::ComputeCmma(ComputeCmmaConfig {
            accumulator_type: elem_out.elem_type(),
            cmma_dims: CmmaDims {
                m: config.m as usize,
                n: config.n as usize,
                k: config.k as usize,
            },
        })
    } else {
        ThroughputMode::ComputeDirect
    }
}
