use burn_backend::{Peak, PeakError};
use core::time::Duration;

/// The rate at the fastest of `samples` that each moved `bytes`.
pub(crate) fn fastest_rate(bytes: u64, samples: &[Duration]) -> Result<Peak, PeakError> {
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
}
