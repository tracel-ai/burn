use alloc::sync::Arc;

use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    client::ComputeClient,
    ir::StorageType,
    std::throughput::{measure_launch_overhead, measure_peak_throughput},
    throughput::{CmmaDims, ComputeCmmaConfig, ThroughputKey, ThroughputMode},
    tune::{AutotuneBound, Bounds, BoundsGenerator},
};
use cubek::matmul::{
    components::tile::TileMatmulKind,
    definition::{MatmulElems, MatmulGlobalElems},
    strategy::MatmulAutotuneKey,
};

use crate::{CubeRuntime, kernel::matmul::tune::base::Inputs};

type BoundsGen<R> = dyn BoundsGenerator<MatmulAutotuneKey, Inputs<R>, AutotuneBound> + Send + Sync;

const THRESHOLD: f32 = 0.85;

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

    let lhs_shape = lhs.meta.shape();
    let rhs_shape = rhs.meta.shape();
    let ndims = lhs_shape.len();

    let m = lhs_shape[ndims - 2];
    let k = lhs_shape[ndims - 1];
    let n = rhs_shape[ndims - 1];
    let batches = lhs_shape[..ndims - 2].iter().product::<usize>();

    let compute =
        matmul_compute_throughput_selection(client, elem_lhs, elem_rhs, elem_out, (m, n, k));
    let compute_mode = match compute.cmma_tile {
        Some((tile_m, tile_n, tile_k)) => ThroughputMode::ComputeCmma(ComputeCmmaConfig {
            accumulator_type: compute.acc.elem_type(),
            cmma_dims: CmmaDims {
                m: tile_m as usize,
                n: tile_n as usize,
                k: tile_k as usize,
            },
        }),
        None => ThroughputMode::ComputeDirect,
    };

    let compute_throughput = measure_peak_throughput(
        client,
        ThroughputKey {
            mode: compute_mode,
            dtype: compute.acc.elem_type(),
        },
    );

    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
        dtype: elem_out.elem_type(),
    };

    let memory_throughput = measure_peak_throughput(client, memory_key);

    let lhs_bytes = lhs.meta.num_elements() * elem_lhs.elem_type().size();
    let rhs_bytes = rhs.meta.num_elements() * elem_rhs.elem_type().size();
    let out_bytes = out.meta.num_elements() * elem_out.elem_type().size();

    vec![
        AutotuneBound {
            ops_count: batches * m * n * (2 * k - 1),
            throughput: compute_throughput.ops_per_s(),
            threshold: THRESHOLD,
        },
        AutotuneBound {
            ops_count: lhs_bytes + rhs_bytes + out_bytes,
            throughput: memory_throughput.bytes_per_s(&memory_key),
            threshold: THRESHOLD,
        },
    ]
}

/// Compute-throughput selection for a matmul problem, resolved from the same register
/// element types and cmma tile availability the matmul kernel uses.
///
/// Keeps the autotune compute bound anchored to the kernel that will actually run, rather
/// than an independent scan of the hardware cmma set that can pick a peak no available
/// kernel reaches.
#[derive(Debug, Clone, Copy)]
pub struct MatmulComputeThroughputSelection {
    /// Accumulator register type: the element type the throughput probe should measure.
    pub acc: StorageType,
    /// The accelerated (cmma) tile `(m, n, k)` to measure, or `None` when the matmul will
    /// fall back to a non-accelerated (direct) kernel for this problem.
    pub cmma_tile: Option<(u32, u32, u32)>,
}

/// Resolves the compute-throughput selection for the given global element types and
/// problem size, matching how the matmul resolves register types and cmma availability.
pub fn matmul_compute_throughput_selection<R: CubeRuntime>(
    client: &ComputeClient<R>,
    lhs: StorageType,
    rhs: StorageType,
    out: StorageType,
    (m, n, k): (usize, usize, usize),
) -> MatmulComputeThroughputSelection {
    let elems = MatmulElems::from_globals(&MatmulGlobalElems { lhs, rhs, out });

    let cmma_tile = TileMatmulKind::Cmma
        .supported_sizes(
            client,
            elems.lhs_register,
            elems.rhs_register,
            elems.acc_register,
        )
        .into_iter()
        .filter(|tile| m >= tile.m() as usize && n >= tile.n() as usize && k >= tile.k() as usize)
        .max_by_key(|tile| tile.m() as u64 * tile.n() as u64 * tile.k() as u64)
        .map(|tile| (tile.m(), tile.n(), tile.k()));

    MatmulComputeThroughputSelection {
        acc: elems.acc_register,
        cmma_tile,
    }
}
