//! A link probe between two remote devices, which their servers run.

use core::time::Duration;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use burn_backend::{LinkPoint, PeakError};
use burn_router::get_client;

use super::runner::{RemoteDevice, session_order};
use super::{RemoteChannel, RemoteClient};
use crate::PeerId;
use crate::shared::{LinkPull, TransferCapability};

const SMALLEST_PULL_BYTES: u64 = 4 << 10;
/// A size whose fastest download takes longer ends the sweep: the next would take four times as
/// long, for a rate the link has already settled on.
const SLOWEST_PULL: Duration = Duration::from_secs(1);

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
    match from.device.peer_id() == to.peer_id() {
        true => measure_within_server(from, to),
        false => measure_between_servers(from, to),
    }
}

/// `to`'s session grants its device under a capability only this client holds, so the server
/// probes a device only for a client it admitted there.
fn measure_within_server(
    from: &RemoteClient,
    to: &RemoteDevice,
) -> Result<Vec<LinkPoint>, PeakError> {
    let capability = TransferCapability::random();
    let granted = get_client::<RemoteChannel>(to)
        .handle
        .submit_blocking(move |s| s.grant_link(capability))
        .unwrap_or(false);
    if !granted {
        return Err(PeakError::Unsupported);
    }
    from.handle
        .submit_blocking(move |s| s.measure_link_local(capability))
        .unwrap_or(Err(PeakError::Unsupported))
}

/// Unsupported between an Iroh server and a WebSocket one, which no transfer crosses.
///
/// One at a time in the process, so a second caller for the pair finds the first one's curve
/// whichever sessions it reaches the servers through.
fn measure_between_servers(
    from: &RemoteClient,
    to: &RemoteDevice,
) -> Result<Vec<LinkPoint>, PeakError> {
    if from.device.peer_addr().is_iroh() != to.peer_addr().is_iroh() {
        return Err(PeakError::Unsupported);
    }
    let target_client = get_client::<RemoteChannel>(to);
    let mut measured = measured();
    let ends = (link_end(&from.device), link_end(to));
    if let Some(curve) = measured.get(&ends) {
        return Ok(curve.clone());
    }
    let _order = session_order([&from.device, to]);
    let mut curve = Vec::new();
    for bytes in pull_sizes() {
        let point = pull_once(from, &target_client, to.peer_id(), bytes)?;
        curve.push(point);
        if point.bytes as f64 / point.per_second > SLOWEST_PULL.as_secs_f64() {
            break;
        }
    }
    measured.insert(ends, curve.clone());
    Ok(curve)
}

/// One size of the sweep: exposed on the source, timed on the target, then withdrawn from the
/// source whatever downloads the target left. A target that could not be asked may still be
/// downloading, so the buffer is left to expire on the source instead.
fn pull_once(
    from: &RemoteClient,
    target_client: &RemoteClient,
    target: PeerId,
    bytes: u64,
) -> Result<LinkPoint, PeakError> {
    let pull = LinkPull {
        bytes,
        downloads: LinkPull::MAX_DOWNLOADS,
        capability: TransferCapability::random(),
    };
    let exposed = from
        .handle
        .submit_blocking(move |s| s.expose_link_pull(target, pull))
        .unwrap_or(false);
    if !exposed {
        return Err(PeakError::Unsupported);
    }
    let source = from.device.peer_addr();
    let point = target_client
        .handle
        .submit_blocking(move |s| s.measure_link_pull(source, pull))
        .ok()
        .flatten()
        .ok_or(PeakError::Unsupported)?;
    from.handle
        .submit(move |s| s.withdraw_link_pull(pull.capability));
    from.handle.flush_queue();
    point
}

/// From the smallest size up to the largest a server exposes, by fours.
fn pull_sizes() -> impl Iterator<Item = u64> {
    core::iter::successors(Some(SMALLEST_PULL_BYTES), |bytes| Some(bytes * 4))
        .take_while(|bytes| *bytes <= LinkPull::MAX_BYTES)
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
    fn pulls_span_four_kibibytes_to_the_largest_a_server_exposes() {
        let sizes: Vec<u64> = pull_sizes().collect();

        assert_eq!(sizes.first(), Some(&SMALLEST_PULL_BYTES));
        assert_eq!(sizes.last(), Some(&LinkPull::MAX_BYTES));
        assert_eq!(sizes.len(), 8);
    }
}
