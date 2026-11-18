//! The server-to-server half of a link probe: the source exposes a buffer per transfer size, and
//! the target times its downloads of each.

use burn_backend::{Peak, PeakError, TensorData};
use burn_ir::BackendIr;
use burn_std::profile::Instant;
use core::time::Duration;

use crate::PeerAddr;
use crate::server::transfer::TensorTransfer;
use crate::shared::LinkPull;

/// The buffer a source exposes for `pull`, which only its size matters for.
pub(crate) fn link_buffer(pull: &LinkPull) -> TensorData {
    TensorData::zeros::<u8, _>([pull.bytes as usize])
}

/// The peak throughput of the link from `source` at the pull's size, timing every download after
/// the first. It makes every download the pull names, so the source's transport releases the buffer
/// after the last one; a link lost partway answers with the downloads it timed.
///
/// # Errors
///
/// [`Unsupported`](PeakError::Unsupported) for a pull outside the bounds a client sends, and
/// [`Transfer`](PeakError::Transfer) when a download fails before one is timed.
pub(crate) async fn time_link_pull<B: BackendIr, T: TensorTransfer<B>>(
    transfer: &T,
    source: &PeerAddr,
    pull: &LinkPull,
) -> Result<Peak, PeakError> {
    if !pull.is_bounded() {
        return Err(PeakError::Unsupported);
    }
    let mut samples = Vec::with_capacity(pull.downloads as usize);
    for _ in 0..pull.downloads {
        let start = Instant::now();
        if transfer
            .download_tensor(source.clone(), pull.capability)
            .await
            .is_none()
        {
            break;
        }
        samples.push(start.elapsed());
    }
    match samples.get(1..) {
        Some(timed) if !timed.is_empty() => fastest_rate(pull.bytes, timed),
        _ => Err(PeakError::Transfer),
    }
}

/// The link's rate at its fastest sample of moving `bytes`.
fn fastest_rate(bytes: u64, samples: &[Duration]) -> Result<Peak, PeakError> {
    let fastest = samples
        .iter()
        .copied()
        .filter(|sample| !sample.is_zero())
        .min()
        .ok_or(PeakError::NoTiming)?;
    Ok(Peak {
        per_second: bytes as f64 / fastest.as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rate_is_the_fastest_sample() {
        let samples = [Duration::from_millis(4), Duration::from_millis(2)];

        let peak = fastest_rate(1000, &samples).unwrap();

        assert_eq!(peak.per_second, 500_000.0);
    }

    #[test]
    fn samples_with_no_elapsed_time_have_no_rate() {
        assert_eq!(
            fastest_rate(1000, &[Duration::ZERO]).unwrap_err(),
            PeakError::NoTiming
        );
    }

    #[cfg(feature = "websocket")]
    mod lost_link {
        use super::*;
        use crate::PeerId;
        use crate::shared::TransferCapability;
        use burn_flex::Flex;
        use std::sync::atomic::{AtomicU32, Ordering};

        /// Serves `served` downloads, then fails every later one.
        struct LostAfter {
            served: AtomicU32,
        }

        impl TensorTransfer<Flex> for LostAfter {
            async fn expose_data(&self, _: TensorData, _: u32, _: TransferCapability, _: PeerId) {}

            async fn download_tensor(
                &self,
                _: PeerAddr,
                _: TransferCapability,
            ) -> Option<TensorData> {
                let left = self.served.load(Ordering::SeqCst);
                self.served.store(left.saturating_sub(1), Ordering::SeqCst);
                (left > 0).then(|| TensorData::zeros::<u8, _>([1]))
            }

            async fn fail(&self, _: TransferCapability, _: PeerId, _: String) {}

            async fn withdraw(&self, _: TransferCapability) {}
        }

        async fn pull_lost_after(served: u32) -> Result<Peak, PeakError> {
            let transfer = LostAfter {
                served: AtomicU32::new(served),
            };
            let source = PeerAddr::WebSocket(burn_communication::Address::from("ws://127.0.0.1:0"));
            let pull = LinkPull {
                bytes: 4096,
                downloads: 4,
                capability: TransferCapability::random(),
            };
            time_link_pull::<Flex, _>(&transfer, &source, &pull).await
        }

        #[tokio::test]
        async fn a_link_lost_after_a_timed_download_answers_with_it() {
            assert!(pull_lost_after(2).await.is_ok());
        }

        #[tokio::test]
        async fn a_link_lost_before_a_timed_download_is_a_transfer_failure() {
            assert_eq!(pull_lost_after(1).await, Err(PeakError::Transfer));
        }
    }
}
