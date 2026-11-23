//! The arrival counters of a single query's tile routines over a key/value cache.

use alloc::vec::Vec;

use crate::{CubeDevice, ops::numeric::full_device_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use burn_std::{id::StreamId, sync::Mutex};
use cubecl::{client::Client, prelude::InputScalar};

/// The `u32` counters a decode's cubes count themselves in at, kept per device and stream.
///
/// Every launch leaves them at zero, so one buffer serves every call on its stream, and a captured
/// decode replays over the counters it recorded. A stream of its own keeps two decodes running at
/// once from counting into the same buffer.
struct ArrivalCounters {
    kept: Vec<(CubeDevice, StreamId, CubeTensor)>,
}

static ARRIVALS: Mutex<ArrivalCounters> = Mutex::new(ArrivalCounters { kept: Vec::new() });

/// At least `count` counters at zero on `device`, for the current stream: the ones kept, unless
/// they are too few, in which case they are replaced by `count` new ones.
pub(crate) fn kept_arrivals(client: &Client, device: &CubeDevice, count: usize) -> CubeTensor {
    let mut arrivals = ARRIVALS.lock();
    let stream = StreamId::current();
    let kept = arrivals
        .kept
        .iter()
        .position(|(on_device, on_stream, _)| on_device == device && *on_stream == stream);
    if let Some(index) = kept {
        let counters = &arrivals.kept[index].2;
        if counters.meta.shape()[0] >= count {
            return counters.clone();
        }
        arrivals.kept.swap_remove(index);
    }
    let counters = full_device_dtype(
        client.clone(),
        Shape::new([count]),
        device.clone(),
        InputScalar::new(0u32, dtype_to_storage_type(DType::U32)),
        DType::U32,
    );
    arrivals
        .kept
        .push((device.clone(), stream, counters.clone()));
    counters
}
