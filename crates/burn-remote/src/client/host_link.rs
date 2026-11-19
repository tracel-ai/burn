//! A link probe between the calling process and a remote device, which the client runs.

use core::time::Duration;

use burn_backend::{HostTransfer, Peak, PeakError, TensorData};
use burn_router::RouterClient;
use burn_std::future::block_on;
use burn_std::profile::Instant;

use super::RemoteClient;
use crate::shared::fastest_rate;

/// Timed transfers taken of one size before the time budget can end the sampling.
const MIN_TIMED: usize = 3;
const MAX_TIMED: usize = 10;
const SAMPLING_BUDGET: Duration = Duration::from_millis(500);

/// Times the link between the calling process and a remote device through the device's client:
/// an upload until the server holds the tensor on its device, a download by reading one back.
pub(crate) struct HostLinkProber<'a> {
    client: &'a RemoteClient,
}

impl<'a> HostLinkProber<'a> {
    pub(crate) fn new(client: &'a RemoteClient) -> Self {
        Self { client }
    }

    /// The link's peak throughput for each of `sizes`, in order, moved the way `transfer` names.
    /// Once a transfer fails, every size after it reports the same failure untried.
    pub(crate) fn measure(
        &self,
        transfer: HostTransfer,
        sizes: &[u64],
    ) -> Vec<Result<Peak, PeakError>> {
        let mut failed = false;
        sizes
            .iter()
            .map(|&bytes| {
                if failed {
                    return Err(PeakError::Transfer);
                }
                let peak = match transfer {
                    HostTransfer::Upload => self.upload(bytes),
                    HostTransfer::Download => self.download(bytes),
                };
                failed = peak == Err(PeakError::Transfer);
                peak
            })
            .collect()
    }

    /// The peak rate of uploading `bytes`.
    ///
    /// # Errors
    ///
    /// [`Unsupported`](PeakError::Unsupported) for no bytes and in the browser, and
    /// [`Transfer`](PeakError::Transfer) when the session fails under an upload.
    fn upload(&self, bytes: u64) -> Result<Peak, PeakError> {
        let data = buffer(bytes)?;
        let samples = sample(|| {
            let data = data.clone();
            let start = Instant::now();
            let tensor = self.client.register_tensor_data(data);
            self.client.sync().map_err(|_| PeakError::Transfer)?;
            let elapsed = start.elapsed();
            drop(tensor);
            Ok(elapsed)
        })?;
        fastest_rate(bytes, &samples)
    }

    /// The peak rate of downloading `bytes` the device holds.
    ///
    /// # Errors
    ///
    /// [`Unsupported`](PeakError::Unsupported) for no bytes and in the browser, and
    /// [`Transfer`](PeakError::Transfer) when the session fails under a download.
    fn download(&self, bytes: u64) -> Result<Peak, PeakError> {
        let tensor = self.client.register_tensor_data(buffer(bytes)?);
        self.client.sync().map_err(|_| PeakError::Transfer)?;
        let samples = sample(|| {
            // A second handle keeps the read from consuming the tensor the next sample reads.
            let ir = tensor.clone().into_ir();
            let start = Instant::now();
            block_on(self.client.read_tensor_async(ir)).map_err(|_| PeakError::Transfer)?;
            Ok(start.elapsed())
        })?;
        fastest_rate(bytes, &samples)
    }
}

impl core::fmt::Debug for HostLinkProber<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostLinkProber")
            .field("device", &self.client.device)
            .finish()
    }
}

/// A buffer of `bytes`, which only its size matters for. None in the browser, where a sync
/// returns before the server has run what came before it, so nothing could be timed.
fn buffer(bytes: u64) -> Result<TensorData, PeakError> {
    if cfg!(target_family = "wasm") {
        return Err(PeakError::Unsupported);
    }
    let len = usize::try_from(bytes).map_err(|_| PeakError::Unsupported)?;
    if len == 0 {
        return Err(PeakError::Unsupported);
    }
    Ok(TensorData::zeros::<u8, _>([len]))
}

/// One warmup, then timed transfers until the time budget runs out past the fewest, or the most.
fn sample(
    mut transfer: impl FnMut() -> Result<Duration, PeakError>,
) -> Result<Vec<Duration>, PeakError> {
    transfer()?;
    let started = Instant::now();
    let mut samples = Vec::with_capacity(MAX_TIMED);
    while samples.len() < MAX_TIMED
        && (samples.len() < MIN_TIMED || started.elapsed() < SAMPLING_BUDGET)
    {
        samples.push(transfer()?);
    }
    Ok(samples)
}
