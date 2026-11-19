//! The server-to-server half of a link probe: the source exposes a buffer per transfer size, and
//! the target times its downloads of each.

use burn_backend::{Peak, PeakError, TensorData};
use burn_ir::BackendIr;
use burn_std::profile::Instant;

use crate::server::transfer::TensorTransfer;
use crate::shared::{LinkPull, fastest_rate};
use crate::{PeerAddr, PeerId};

/// Serves a link probe's pulls through the server's transfers: as the source, it exposes a buffer
/// per pull; as the target, it times its downloads of one.
pub(crate) struct LinkPuller<'a, T> {
    transfer: &'a T,
}

impl<'a, T> LinkPuller<'a, T> {
    pub(crate) fn new(transfer: &'a T) -> Self {
        Self { transfer }
    }

    /// Exposes a buffer of the pull's size for `target` to download as many times as the pull
    /// names, and whether it did: not for a pull outside the bounds a client sends.
    pub(crate) async fn expose<B: BackendIr>(&self, target: PeerId, pull: &LinkPull) -> bool
    where
        T: TensorTransfer<B>,
    {
        if !pull.is_bounded() {
            return false;
        }
        let buffer = TensorData::zeros::<u8, _>([pull.bytes as usize]);
        self.transfer
            .expose_data(buffer, pull.downloads, pull.capability, target)
            .await;
        true
    }

    /// The peak throughput of the link from `source` at the pull's size, timing every download
    /// after the first. It makes every download the pull names, so the source's transport
    /// releases the buffer after the last one; a link lost partway answers with the downloads it
    /// timed.
    ///
    /// # Errors
    ///
    /// [`Unsupported`](PeakError::Unsupported) for a pull outside the bounds a client sends, and
    /// [`Transfer`](PeakError::Transfer) when a download fails before one is timed.
    pub(crate) async fn time<B: BackendIr>(
        &self,
        source: &PeerAddr,
        pull: &LinkPull,
    ) -> Result<Peak, PeakError>
    where
        T: TensorTransfer<B>,
    {
        if !pull.is_bounded() {
            return Err(PeakError::Unsupported);
        }
        let mut samples = Vec::with_capacity(pull.downloads as usize);
        for _ in 0..pull.downloads {
            let start = Instant::now();
            if self
                .transfer
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
}

impl<T> core::fmt::Debug for LinkPuller<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LinkPuller").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "websocket")]
    mod lost_link {
        use super::*;
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
            LinkPuller::new(&transfer)
                .time::<Flex>(&source, &pull)
                .await
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
