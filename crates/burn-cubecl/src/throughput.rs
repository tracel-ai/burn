use alloc::vec::Vec;
use cubecl::{
    client::ComputeClient,
    ir::{ElemType, StorageType},
    std::throughput::measure_peak_throughput,
    throughput::{CmmaDims, ComputeCmmaConfig, ThroughputMode},
    tune::AutotuneBound,
};

pub use cubecl::throughput::{ThroughputKey, ThroughputValue};

use crate::CubeRuntime;

/// Measure peak throughput on `device` for each of the given `keys`.
pub fn device_throughput<R: CubeRuntime>(
    device: &R::Device,
    keys: &[ThroughputKey],
) -> Vec<ThroughputValue> {
    let client = R::client(device);
    keys.iter()
        .map(|key| measure_peak_throughput::<R>(&client, *key))
        .collect()
}

/// Resolves the largest supported CMMA or MMA tile size `(m, n, k)`.
pub fn select_cmma_tile<R: CubeRuntime>(
    client: &ComputeClient<R>,
    lhs: StorageType,
    rhs: StorageType,
    acc: StorageType,
    (m, n, k): (usize, usize, usize),
) -> Option<(u32, u32, u32)> {
    let props = client.properties();

    props
        .features
        .matmul
        .cmma
        .iter()
        // Combine both CMMA and MMA supported hardware features.
        .chain(props.features.matmul.mma.iter())
        // Filter for instructions matching the exact input/accumulator data types.
        .filter(|it| it.a_type == lhs && it.b_type == rhs && it.cd_type == acc)
        // Ensure the hardware tile size actually fits within our problem dimensions.
        .filter(|it| m >= it.m as usize && n >= it.n as usize && k >= it.k as usize)
        // Select the tile with the largest volume to maximize throughput.
        .max_by_key(|it| it.m as u64 * it.n as u64 * it.k as u64)
        .map(|it| (it.m as u32, it.n as u32, it.k as u32))
}

/// Constructs a compute [`ThroughputKey`] based on CMMA tile availability and types.
pub fn compute_throughput_key(
    cmma_tile: Option<(u32, u32, u32)>,
    input_elem_type: ElemType,
    acc_elem_type: ElemType,
) -> ThroughputKey {
    let mode = match cmma_tile {
        Some((tile_m, tile_n, tile_k)) => ThroughputMode::ComputeCmma(ComputeCmmaConfig {
            accumulator_type: acc_elem_type,
            cmma_dims: CmmaDims {
                m: tile_m as usize,
                n: tile_n as usize,
                k: tile_k as usize,
            },
        }),
        None => ThroughputMode::ComputeDirect,
    };

    ThroughputKey {
        mode,
        dtype: match mode {
            ThroughputMode::ComputeCmma(_) => input_elem_type,
            ThroughputMode::ComputeDirect => acc_elem_type,
            ThroughputMode::Memory => unreachable!(),
        },
    }
}

/// Standardizes the creation of compute and memory [`AutotuneBound`]s.
pub fn calculate_bounds(
    compute_throughput: &ThroughputValue,
    compute_ops: usize,
    compute_threshold: f32,
    memory_throughput: &ThroughputValue,
    memory_key: &ThroughputKey,
    memory_bytes: usize,
    memory_threshold: f32,
) -> Vec<AutotuneBound> {
    vec![
        AutotuneBound {
            ops_count: compute_ops,
            throughput: compute_throughput.ops_per_s(),
            threshold: compute_threshold,
        },
        AutotuneBound {
            ops_count: memory_bytes,
            throughput: memory_throughput.bytes_per_s(memory_key),
            threshold: memory_threshold,
        },
    ]
}
