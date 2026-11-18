//! A link probe between two remote devices, which their servers run.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use burn_backend::{LinkPoint, PeakError};
use burn_router::get_client;

use super::runner::{RemoteDevice, session_order};
use super::{RemoteChannel, RemoteClient};
use crate::PeerId;
use crate::shared::{LinkPull, TransferCapability};

const SMALLEST_PULL_BYTES: u64 = 4 << 10;
const LARGEST_PULL_BYTES: u64 = 64 << 20;
/// Each size's first download only opens the connection, so it is left out of the rate.
const DOWNLOADS_PER_PULL: u32 = 11;

/// A device, by its server and its index there.
type LinkEnd = (PeerId, u32);

type MeasuredLinks = HashMap<(LinkEnd, LinkEnd), Vec<LinkPoint>>;

/// Curves between two servers, which no server keeps: the source's cubecl cache keeps those
/// within one.
static MEASURED: LazyLock<Mutex<MeasuredLinks>> = LazyLock::new(Default::default);

/// What the link from `from`'s device to `to` delivers across transfer sizes: within one server,
/// as its backend measures it; between two, as the target times downloads from the source.
pub(crate) fn measure_link(
    from: &RemoteClient,
    to: &RemoteDevice,
) -> Result<Vec<LinkPoint>, PeakError> {
    let source = &from.device;
    if source.peer_id() == to.peer_id() {
        return measure_within_server(from, to);
    }
    let ends = (link_end(source), link_end(to));
    if let Some(curve) = measured().get(&ends) {
        return Ok(curve.clone());
    }
    let curve = measure_between_servers(from, to)?;
    measured().insert(ends, curve.clone());
    Ok(curve)
}

fn measure_within_server(
    from: &RemoteClient,
    to: &RemoteDevice,
) -> Result<Vec<LinkPoint>, PeakError> {
    let index = to.device_index;
    from.handle
        .submit_blocking(move |s| s.measure_link_local(index))
        .unwrap_or(Err(PeakError::Unsupported))
}

/// Unsupported between an Iroh server and a WebSocket one, which no transfer crosses.
fn measure_between_servers(
    from: &RemoteClient,
    to: &RemoteDevice,
) -> Result<Vec<LinkPoint>, PeakError> {
    if from.device.peer_addr().is_iroh() != to.peer_addr().is_iroh() {
        return Err(PeakError::Unsupported);
    }
    let pulls = link_pulls();
    let target_client = get_client::<RemoteChannel>(to);
    let order = session_order([&from.device, to]);
    let (target, exposed) = (to.peer_id(), pulls.clone());
    from.handle
        .submit(move |s| s.expose_link_probe(target, exposed));
    from.handle.flush_queue();
    let source = from.device.peer_addr();
    let link = target_client
        .handle
        .submit_blocking(move |s| s.measure_link_pull(source, pulls))
        .unwrap_or(Err(PeakError::Unsupported));
    drop(order);
    link
}

/// One pull per size, by fours from the smallest to the largest.
fn link_pulls() -> Vec<LinkPull> {
    core::iter::successors(Some(SMALLEST_PULL_BYTES), |bytes| Some(bytes * 4))
        .take_while(|bytes| *bytes <= LARGEST_PULL_BYTES)
        .map(|bytes| LinkPull {
            bytes,
            downloads: DOWNLOADS_PER_PULL,
            capability: TransferCapability::random(),
        })
        .collect()
}

fn link_end(device: &RemoteDevice) -> LinkEnd {
    (device.peer_id(), device.device_index)
}

fn measured() -> std::sync::MutexGuard<'static, MeasuredLinks> {
    MEASURED.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulls_span_four_kibibytes_to_sixty_four_mebibytes() {
        let sizes: Vec<u64> = link_pulls().iter().map(|pull| pull.bytes).collect();

        assert_eq!(sizes.first(), Some(&SMALLEST_PULL_BYTES));
        assert_eq!(sizes.last(), Some(&LARGEST_PULL_BYTES));
        assert_eq!(sizes.len(), 8);
    }
}
