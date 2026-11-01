//! The device a run used, and what it can do.

use std::fmt::{self, Display};

use burn_instrument::PeakFinding;

use crate::Report;

/// The page's first lines: the device, its peaks, and what a reader must be told about them
/// before any percentage below is read.
pub(super) struct DeviceHeader<'r, 'a> {
    report: &'r Report<'a>,
}

impl<'r, 'a> DeviceHeader<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>) -> Self {
        Self { report }
    }
}

impl Display for DeviceHeader<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let measurement = self.report.measurement();
        let peaks = self.report.roofline().peaks();
        writeln!(
            f,
            "device   {} · {} · {}",
            measurement.device,
            match peaks.copy_bytes_per_s {
                Some(copy) => format!("{:.1} GB/s", copy / 1e9),
                None => "unprobed".to_string(),
            },
            match peaks.flops {
                Some(flops) => format!("{:.1} TFLOP/s at {:?}", flops.per_s / 1e12, flops.dtype),
                None => "unprobed".to_string(),
            },
        )?;
        // The two ceilings a bytes component is actually divided by, beside the copy
        // figure above. A percentage whose denominator is nowhere on the page
        // asks to be trusted about the very thing a reader doubts — and the
        // copy ceiling alone would be worse than none, since it is not what any
        // component was divided by. Absent where the device reported no directions,
        // in which case the copy ceiling above *is* the denominator.
        if let (Some(read), Some(write)) = (peaks.read_bytes_per_s, peaks.write_bytes_per_s) {
            writeln!(
                f,
                "         scored against {:.1} GB/s read and {:.1} GB/s written, each charged",
                read / 1e9,
                write / 1e9,
            )?;
            writeln!(
                f,
                "         its own direction; the copy ceiling is reported, not divided by."
            )?;
        }
        if let Some(launch) = peaks.launch_seconds {
            writeln!(
                f,
                "         one launch takes {:.2} µs back to back; no floor below charges it",
                launch * 1e6,
            )?;
        }
        if measurement.peaks.is_none() {
            writeln!(
                f,
                "         not probed: no component is scored, and the ladder is the measured time"
            )?;
        }
        for finding in self.report.findings() {
            for line in note(finding) {
                writeln!(f, "         {line}")?;
            }
        }
        Ok(())
    }
}

/// What the header says about one finding, one string per printed line.
///
/// Named, not swallowed: every ratio in a run shares the probe, so one bad
/// reading invalidates all of them at once, and a reader who sees only the copy
/// ceiling has no way to tell a device that reported no directional peaks from
/// one whose probe contradicted itself — or to know why no component is held against
/// `flops`.
fn note(finding: &PeakFinding) -> Vec<String> {
    match *finding {
        PeakFinding::ImpossibleDirections {
            read_bytes_per_s: read,
            write_bytes_per_s: write,
        } => vec![
            format!(
                "REJECTED the per-direction peaks: the probe reported {:.1} GB/s",
                write / 1e9
            ),
            format!(
                "written against {:.1} GB/s read. Nothing writes faster than",
                read / 1e9
            ),
            "it reads — a store moves at least a load's bytes, and a".to_string(),
            "write-allocate cache makes DRAM see more — so every component below".to_string(),
            "is scored against the copy ceiling instead.".to_string(),
        ],
        // A caveat, not a fallback: the components below were still divided by the two
        // directions, because nothing in three disagreeing readings says which
        // of them is wrong.
        PeakFinding::UnpredictedCopyCeiling {
            measured_bytes_per_s: measured,
            predicted_bytes_per_s: predicted,
        } => vec![
            format!(
                "CAVEAT: the two directions predict a {:.1} GB/s copy ceiling and",
                predicted / 1e9
            ),
            format!(
                "the probe measured {:.1}. One of the three is wrong and the",
                measured / 1e9
            ),
            "readings do not say which, so the components below are still scored".to_string(),
            "on the directions. Check them against an independent stream".to_string(),
            "before believing the absolutes.".to_string(),
        ],
        PeakFinding::ImpossibleCompute { flops_per_s } => vec![
            format!(
                "REJECTED the compute peak: the probe reported {:.4} TFLOP/s,",
                flops_per_s / 1e12
            ),
            "below this device's own memory peak. Nothing computes fewer".to_string(),
            "FLOP/s than it moves bytes/s, so the probe failed — every component".to_string(),
            "below is scored on memory alone.".to_string(),
        ],
    }
}
