//! Attention over a key/value cache through cubek's autotune over its tile-DSL routines
//! ([`launch_tuned`]), for the calls [`cache_serves`] admits.

use alloc::vec::Vec;

use super::{routines_take, scoring};
use crate::{
    CubeDevice,
    kernel::cast,
    ops::{numeric::full_device_dtype, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::ops::AttentionModuleOptions;
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use burn_std::{id::StreamId, sync::Mutex};
use cubecl::prelude::InputScalar;
use cubek::attention::kernels::select::problem::QueryPositions;
use cubek::attention::kernels::tune::{AttentionTuneInputs, launch_tuned};
use cubek::std::autotune::GroupPriority;

/// The `u32` counters a decode's cubes count themselves in at, kept per device and stream.
///
/// Every launch leaves them at zero, so one buffer serves every call on its stream, and a captured
/// decode replays over the counters it recorded. A stream of its own keeps two decodes running at
/// once from counting into the same buffer.
struct ArrivalCounters {
    kept: Vec<(CubeDevice, StreamId, CubeTensor)>,
}

static ARRIVALS: Mutex<ArrivalCounters> = Mutex::new(ArrivalCounters { kept: Vec::new() });

impl ArrivalCounters {
    /// At least `count` counters at zero on `query`'s device, for the current stream: the ones
    /// kept, unless they are too few, in which case they are replaced by `count` new ones.
    fn at_least(&mut self, query: &CubeTensor, count: usize) -> CubeTensor {
        let stream = StreamId::current();
        let kept = self
            .kept
            .iter()
            .position(|(device, on, _)| *device == query.device && *on == stream);
        if let Some(index) = kept {
            let counters = &self.kept[index].2;
            if counters.meta.shape()[0] >= count {
                return counters.clone();
            }
            self.kept.swap_remove(index);
        }
        let counters = full_device_dtype(
            query.client.clone(),
            Shape::new([count]),
            query.device.clone(),
            InputScalar::new(0u32, dtype_to_storage_type(DType::U32)),
            DType::U32,
        );
        self.kept
            .push((query.device.clone(), stream, counters.clone()));
        counters
    }
}

/// Whether the tile routines serve an attention of `query` over the cache of `key` and `value`
/// with `options`: operands they [take](routines_take), and no softcap.
pub fn cache_serves(
    query: &CubeTensor,
    key: &CubeTensor,
    value: &CubeTensor,
    options: &AttentionModuleOptions,
) -> bool {
    routines_take(query, key, value) && options.softcap.is_none()
}

/// Executes [`attention_over_cache`](burn_backend::ops::ModuleOps::attention_over_cache) through
/// cubek's autotune over its tile-DSL routines: a single query among the routines that serve one,
/// counting in at counters kept for its device and stream, and more among those that serve many.
///
/// # Panics
///
/// A call [`cache_serves`] does not admit.
pub fn attention_over_cache_autotune(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    pages: Option<CubeTensor>,
    lengths: CubeTensor,
    options: AttentionModuleOptions,
) -> CubeTensor {
    let (client, device, dtype) = (query.client.clone(), query.device.clone(), query.dtype);
    let [batch, heads, seq_q, head_dim] = query.meta.shape().dims::<4>();
    let positions = QueryPositions::of(seq_q);
    let arrivals = matches!(positions, QueryPositions::One)
        .then(|| ARRIVALS.lock().at_least(&query, batch * heads));
    let inputs = AttentionTuneInputs::new(
        &client,
        query.plain(),
        key.plain(),
        value.plain(),
        pages.map(|pages| cast(pages, DType::U32).plain()),
        cast(lengths, DType::U32).plain(),
        arrivals.map(CubeTensor::plain),
        scoring(&options, head_dim),
    )
    .unwrap_or_else(|why| panic!("no tile-DSL attention routine serves this cache: {why:?}"));
    let out = launch_tuned(positions, &device, inputs, GroupPriority::DEFAULT);
    // The routines write `[b, q, h, dv]`; burn's attention is `[b, h, q, dv]`, viewed without a
    // copy.
    let out = CubeTensor::new(client, out.handle, *out.metadata, device, dtype);
    swap_dims(out, 1, 2)
}
