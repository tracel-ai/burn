//! A link probe between two remote devices, which their servers run.

use core::time::Duration;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use burn_backend::{Peak, PeakError};
use burn_router::get_client;

use super::runner::{RemoteDevice, session_order};
use super::{RemoteChannel, RemoteClient};
use crate::PeerId;
use crate::shared::{LinkPull, TransferCapability};

/// A warmup and three timed downloads, however slow the size.
const MIN_DOWNLOADS: u32 = 4;
/// How long a size's downloads should take, at the rate of the size below it.
const SAMPLING_BUDGET: Duration = Duration::from_millis(500);

/// A device, by its server and its index there.
type LinkEnd = (PeerId, u32);

type MeasuredLinks = HashMap<(LinkEnd, LinkEnd, u64), Peak>;

/// Peaks between two servers, by size, which no server keeps: the source's cubecl cache keeps
/// those within one.
static MEASURED: LazyLock<Mutex<MeasuredLinks>> = LazyLock::new(Default::default);

/// Times the link from one remote device to another, through each one's client: within one
/// server, as its backend measures it; between two, as the target times downloads from the
/// source.
pub(crate) struct LinkProber<'a> {
    source: &'a RemoteClient,
    target: RemoteClient,
}

impl<'a> LinkProber<'a> {
    pub(crate) fn new(source: &'a RemoteClient, target: &RemoteDevice) -> Self {
        Self {
            source,
            target: get_client::<RemoteChannel>(target),
        }
    }

    /// The link's peak throughput for each of `sizes`, in order.
    ///
    /// Unsupported between an Iroh server and a WebSocket one, which no transfer crosses, and
    /// unless both devices' sessions are open and their servers admitted them, checked before
    /// anything is exposed or granted on either.
    pub(crate) fn measure(&self, sizes: &[u64]) -> Vec<Result<Peak, PeakError>> {
        if sizes.is_empty() {
            return Vec::new();
        }
        let (source, target) = (&self.source.device, &self.target.device);
        if source.peer_addr().is_iroh() != target.peer_addr().is_iroh() {
            return unsupported(sizes);
        }
        if !self.source.admitted() || !self.target.admitted() {
            return unsupported(sizes);
        }
        match source.peer_id() == target.peer_id() {
            true => self.measure_within_server(sizes),
            false => self.measure_between_servers(sizes),
        }
    }

    /// The target's session grants its device under a capability only this client holds, so the
    /// server probes a device only for a client it admitted there.
    fn measure_within_server(&self, sizes: &[u64]) -> Vec<Result<Peak, PeakError>> {
        let capability = TransferCapability::random();
        let granted = self
            .target
            .handle
            .submit_blocking(move |s| s.grant_link(capability))
            .unwrap_or(false);
        if !granted {
            return unsupported(sizes);
        }
        let asked = sizes.to_vec();
        self.source
            .handle
            .submit_blocking(move |s| s.measure_link_local(capability, asked))
            .unwrap_or_else(|_| unsupported(sizes))
    }

    /// A size no server serves is unsupported on its own. The rest go smallest first, so each
    /// size's downloads are fitted to the rate of the one below it, and the first that fails ends
    /// the probe, its error answering every larger size not cached: no further buffer is exposed
    /// for a target that may be gone. One probe at a time in the process, so a second caller for the pair finds the first one's
    /// peaks whichever sessions it reaches the servers through.
    fn measure_between_servers(&self, sizes: &[u64]) -> Vec<Result<Peak, PeakError>> {
        let mut measured = measured();
        let ends = (link_end(&self.source.device), link_end(&self.target.device));
        let _order = session_order([&self.source.device, &self.target.device]);
        let mut peaks: HashMap<u64, Result<Peak, PeakError>> = sizes
            .iter()
            .filter(|bytes| !LinkPull::serves_bytes(**bytes))
            .map(|bytes| (*bytes, Err(PeakError::Unsupported)))
            .collect();
        let mut ascending: Vec<u64> = sizes
            .iter()
            .copied()
            .filter(|bytes| LinkPull::serves_bytes(*bytes))
            .collect();
        ascending.sort_unstable();
        ascending.dedup();
        let mut below = None;
        let mut failed = None;
        for bytes in ascending {
            let key = (ends.0.clone(), ends.1.clone(), bytes);
            let peak = match (measured.get(&key), failed) {
                (Some(peak), _) => Ok(*peak),
                (None, Some(error)) => Err(error),
                (None, None) => self.pull(bytes, downloads_at(bytes, below)),
            };
            peaks.insert(bytes, peak);
            failed = failed.or(peak.err());
            below = peak.ok().or(below);
            if let Ok(peak) = peak {
                measured.insert(key, peak);
            }
        }
        sizes.iter().map(|bytes| peaks[bytes]).collect()
    }

    /// One size: exposed on the source and timed on the target, which makes every download so the
    /// source's transport releases the buffer after the last. Withdrawn once the target answered,
    /// for one that failed partway: a target that could not be asked may still be downloading,
    /// and withdrawing under it would leave its next download waiting.
    fn pull(&self, bytes: u64, downloads: u32) -> Result<Peak, PeakError> {
        let pull = LinkPull {
            bytes,
            downloads,
            capability: TransferCapability::random(),
        };
        let target = self.target.device.peer_id();
        let exposed = self
            .source
            .handle
            .submit_blocking(move |s| s.expose_link_pull(target, pull))
            .unwrap_or(false);
        if !exposed {
            return Err(PeakError::Unsupported);
        }
        let source = self.source.device.peer_addr();
        let peak = self
            .target
            .handle
            .submit_blocking(move |s| s.measure_link_pull(source, pull))
            .ok()
            .flatten()
            .ok_or(PeakError::Unsupported)?;
        self.source
            .handle
            .submit(move |s| s.withdraw_link_pull(pull.capability));
        self.source.handle.flush_queue();
        peak
    }
}

impl core::fmt::Debug for LinkProber<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LinkProber")
            .field("source", &self.source.device)
            .field("target", &self.target.device)
            .finish()
    }
}

/// Downloads of `bytes`: as many as fit the sampling budget at the peak of the size `below` it,
/// within what a server serves, and every one a server serves with no size below.
fn downloads_at(bytes: u64, below: Option<Peak>) -> u32 {
    let Some(peak) = below else {
        return LinkPull::MAX_DOWNLOADS;
    };
    let seconds = bytes as f64 / peak.per_second;
    let fitting = (SAMPLING_BUDGET.as_secs_f64() / seconds) as u32;
    fitting
        .saturating_add(1)
        .clamp(MIN_DOWNLOADS, LinkPull::MAX_DOWNLOADS)
}

fn unsupported(sizes: &[u64]) -> Vec<Result<Peak, PeakError>> {
    sizes.iter().map(|_| Err(PeakError::Unsupported)).collect()
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

    /// The peak of a size below, moved in `seconds` at its fastest.
    fn below(bytes: u64, seconds: f64) -> Option<Peak> {
        let per_second = bytes as f64 / seconds;
        Some(Peak { per_second })
    }

    #[test]
    fn a_fast_link_takes_every_download_a_server_serves() {
        assert_eq!(downloads_at(4096, None), LinkPull::MAX_DOWNLOADS);
        assert_eq!(
            downloads_at(16384, below(4096, 1e-6)),
            LinkPull::MAX_DOWNLOADS
        );
    }

    #[test]
    fn a_slow_link_takes_a_warmup_and_three_timed_downloads() {
        assert_eq!(downloads_at(4 << 20, below(1 << 20, 0.9)), MIN_DOWNLOADS);
    }

    #[test]
    fn the_downloads_fit_the_budget_at_the_rate_below() {
        // 40 ms a download, so 12 fit 500 ms past the warmup: capped at 11.
        assert_eq!(
            downloads_at(16384, below(4096, 0.01)),
            LinkPull::MAX_DOWNLOADS
        );
        // 200 ms a download: 2 timed fit, raised to three.
        assert_eq!(downloads_at(16384, below(4096, 0.05)), MIN_DOWNLOADS);
        // 80 ms a download: 6 timed fit, plus the warmup.
        assert_eq!(downloads_at(16384, below(4096, 0.02)), 7);
    }
}
