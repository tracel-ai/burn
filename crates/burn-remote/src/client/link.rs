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
/// A warmup and three timed downloads, however slow the size.
const MIN_DOWNLOADS: u32 = 4;
/// How long a size's downloads should take, the next size's expected from the last one's.
const SAMPLING_BUDGET: Duration = Duration::from_millis(500);

/// A device, by its server and its index there.
type LinkEnd = (PeerId, u32);

type MeasuredLinks = HashMap<(LinkEnd, LinkEnd), Vec<LinkPoint>>;

/// Curves between two servers, which no server keeps: the source's cubecl cache keeps those
/// within one.
static MEASURED: LazyLock<Mutex<MeasuredLinks>> = LazyLock::new(Default::default);

/// What the link from `from`'s device to `to` delivers across transfer sizes: within one server,
/// as its backend measures it; between two, as the target times downloads from the source.
///
/// Unsupported unless both devices' sessions are open and their servers admitted them, checked
/// before anything is exposed or granted on either.
pub(crate) fn measure_link(
    from: &RemoteClient,
    to: &RemoteDevice,
) -> Result<Vec<LinkPoint>, PeakError> {
    if !admitted(from) || !admitted(&get_client::<RemoteChannel>(to)) {
        return Err(PeakError::Unsupported);
    }
    match from.device.peer_id() == to.peer_id() {
        true => measure_within_server(from, to),
        false => measure_between_servers(from, to),
    }
}

/// Whether `client`'s session is open, connecting it first: a device a listing named but never
/// connected may still be refused.
fn admitted(client: &RemoteClient) -> bool {
    client.device.session_error().is_none()
        && client
            .handle
            .submit_blocking(|s| s.try_connect().is_ok())
            .unwrap_or(false)
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
        let downloads = downloads_after(curve.last());
        let point = pull_once(from, &target_client, to.peer_id(), bytes, downloads)?;
        curve.push(point);
        if point.bytes as f64 / point.per_second > SLOWEST_PULL.as_secs_f64() {
            break;
        }
    }
    measured.insert(ends, curve.clone());
    Ok(curve)
}

/// One size of the sweep: exposed on the source and timed on the target, which makes every
/// download so the source's transport releases the buffer after the last. Withdrawn once the
/// target answered, for one that failed partway: a target that could not be asked may still be
/// downloading, and withdrawing under it would leave its next download waiting.
fn pull_once(
    from: &RemoteClient,
    target_client: &RemoteClient,
    target: PeerId,
    bytes: u64,
    downloads: u32,
) -> Result<LinkPoint, PeakError> {
    let pull = LinkPull {
        bytes,
        downloads,
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

/// Downloads for the size after `previous`, four times larger: as many as fit the sampling budget
/// at its fastest rate, within what a server serves.
fn downloads_after(previous: Option<&LinkPoint>) -> u32 {
    let Some(previous) = previous else {
        return LinkPull::MAX_DOWNLOADS;
    };
    let next_seconds = 4.0 * previous.bytes as f64 / previous.per_second;
    let fitting = (SAMPLING_BUDGET.as_secs_f64() / next_seconds) as u32;
    fitting
        .saturating_add(1)
        .clamp(MIN_DOWNLOADS, LinkPull::MAX_DOWNLOADS)
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

    fn point(bytes: u64, seconds: f64) -> LinkPoint {
        LinkPoint {
            bytes,
            per_second: bytes as f64 / seconds,
            relative_std_dev: None,
        }
    }

    #[test]
    fn a_fast_link_takes_every_download_a_server_serves() {
        assert_eq!(downloads_after(None), LinkPull::MAX_DOWNLOADS);
        assert_eq!(
            downloads_after(Some(&point(4096, 1e-6))),
            LinkPull::MAX_DOWNLOADS
        );
    }

    #[test]
    fn a_slow_link_takes_a_warmup_and_three_timed_downloads() {
        assert_eq!(downloads_after(Some(&point(1 << 20, 0.9))), MIN_DOWNLOADS);
    }

    #[test]
    fn the_downloads_fit_the_budget_at_the_last_sizes_rate() {
        // The next size takes 40 ms a download, so 12 fit 500 ms past the warmup: capped at 11.
        assert_eq!(
            downloads_after(Some(&point(4096, 0.01))),
            LinkPull::MAX_DOWNLOADS
        );
        // 200 ms a download: 2 timed fit, raised to three.
        assert_eq!(downloads_after(Some(&point(4096, 0.05))), MIN_DOWNLOADS);
        // 80 ms a download: 6 timed fit, plus the warmup.
        assert_eq!(downloads_after(Some(&point(4096, 0.02))), 7);
    }
}
