//! The server-to-server half of a link probe: the source exposes a buffer per transfer size, and
//! the target times its downloads of each.

use burn_backend::{LinkPoint, PeakError, TensorData};
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

/// Timed downloads a target takes of one size before its time budget can end the sampling.
const MIN_TIMED_DOWNLOADS: usize = 3;
const SAMPLING_BUDGET: Duration = Duration::from_millis(500);
/// Past this, one timed download is enough: a link that slowed mid-sweep must still finish the
/// size before an Iroh source lets its capability expire, five minutes after exposing it.
const SAMPLING_DEADLINE: Duration = Duration::from_secs(60);

/// What the link from `source` delivers at the pull's size, timing every download after the first
/// until the time budget runs out or the pull's downloads do.
///
/// # Errors
///
/// [`Unsupported`](PeakError::Unsupported) for a pull outside the bounds a client sends, and
/// [`Transfer`](PeakError::Transfer) when a download fails.
pub(crate) async fn time_link_pull<B: BackendIr, T: TensorTransfer<B>>(
    transfer: &T,
    source: &PeerAddr,
    pull: &LinkPull,
) -> Result<LinkPoint, PeakError> {
    if !pull.is_bounded() {
        return Err(PeakError::Unsupported);
    }
    let started = Instant::now();
    let mut samples = Vec::with_capacity(pull.downloads as usize);
    for _ in 0..pull.downloads {
        let start = Instant::now();
        transfer
            .download_tensor(source.clone(), pull.capability)
            .await
            .ok_or(PeakError::Transfer)?;
        samples.push(start.elapsed());
        if sampled_enough(samples.len() - 1, started.elapsed()) {
            break;
        }
    }
    link_point(pull.bytes, samples.get(1..).unwrap_or_default())
}

fn sampled_enough(timed: usize, elapsed: Duration) -> bool {
    (timed >= MIN_TIMED_DOWNLOADS && elapsed >= SAMPLING_BUDGET)
        || (timed >= 1 && elapsed >= SAMPLING_DEADLINE)
}

/// The link's rate at its fastest sample of moving `bytes`, and how much the samples spread.
fn link_point(bytes: u64, samples: &[Duration]) -> Result<LinkPoint, PeakError> {
    let fastest = samples
        .iter()
        .copied()
        .filter(|sample| !sample.is_zero())
        .min()
        .ok_or(PeakError::NoTiming)?;
    Ok(LinkPoint {
        bytes,
        per_second: bytes as f64 / fastest.as_secs_f64(),
        relative_std_dev: relative_std_dev(samples),
    })
}

/// The samples' standard deviation over their mean, `None` under two samples.
fn relative_std_dev(samples: &[Duration]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let seconds = samples.iter().map(Duration::as_secs_f64);
    let mean = seconds.clone().sum::<f64>() / samples.len() as f64;
    let variance =
        seconds.map(|sample| (sample - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    Some(variance.sqrt() / mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rate_is_the_fastest_sample() {
        let samples = [Duration::from_millis(4), Duration::from_millis(2)];

        let point = link_point(1000, &samples).unwrap();

        assert_eq!(point.per_second, 500_000.0);
    }

    #[test]
    fn identical_samples_do_not_spread() {
        let samples = [Duration::from_millis(3); 4];

        assert_eq!(relative_std_dev(&samples), Some(0.0));
    }

    #[test]
    fn a_slow_link_stops_at_one_timed_download_past_the_deadline() {
        assert!(!sampled_enough(1, SAMPLING_BUDGET));
        assert!(sampled_enough(1, SAMPLING_DEADLINE));
        assert!(!sampled_enough(0, SAMPLING_DEADLINE));
    }

    #[test]
    fn one_sample_has_no_spread() {
        assert_eq!(relative_std_dev(&[Duration::from_millis(3)]), None);
    }

    #[test]
    fn samples_with_no_elapsed_time_have_no_rate() {
        assert_eq!(
            link_point(1000, &[Duration::ZERO]).unwrap_err(),
            PeakError::NoTiming
        );
    }
}
