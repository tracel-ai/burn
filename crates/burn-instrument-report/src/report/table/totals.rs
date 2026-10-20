//! What the run came to, beneath the component table.

use std::fmt::{self, Display};

use burn_instrument::Kernel;

use crate::Report;
use crate::millis::Millis;
use crate::reading::HostSplit;
use crate::share::Share;

/// The run's totals, and what each is short by: the clean run, the device time the
/// components are shares of, the timings that went missing, and the host split.
pub(super) struct Totals<'r, 'a> {
    report: &'r Report<'a>,
    /// The width of the component column the totals line up under.
    width: usize,
}

impl<'r, 'a> Totals<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>, width: usize) -> Self {
        Self { report, width }
    }
}

impl Display for Totals<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let measurement = self.report.measurement();
        let width = self.width;
        let text = self.report.reading().text();
        let kernels = measurement.breakdown.kernels().count();
        let launches: usize = measurement
            .breakdown
            .kernels()
            .map(|kernel| kernel.launches)
            .sum();
        if let Some(wall) = measurement.clean_wall {
            writeln!(
                f,
                "  {:<width$} {:>8} ms   {}",
                "wall",
                Millis(wall),
                text.wall,
            )?;
        }
        writeln!(
            f,
            "  {:<width$} {:>8} ms   {kernels} distinct kernels, {launches} launches, {}",
            match measurement.breakdown.host_timed {
                true => "on device (SOME HOST-TIMED)",
                false => "on device",
            },
            Millis(measurement.breakdown.total_time()),
            text.on_device,
        )?;
        if measurement.breakdown.host_timed {
            for line in [
                "the runtime could not get a device timestamp for every launch and fell",
                "back to a host timer, which measures submission and sync as well as the",
                "kernel — so the proportions above are skewed by an unknown amount.",
            ] {
                writeln!(f, "    {line}")?;
            }
        }
        // Not in any component above, and the total every component is a share of. Said
        // out loud rather than left to shrink the denominator quietly.
        // Lost rather than misplaced, and the reason a table can read a matmul
        // at 0.0 ms beside a counter add at 78: said here, where the total it
        // shortens is printed, and keyed on below.
        let untimed: Vec<&Kernel> = measurement.breakdown.untimed().collect();
        if !untimed.is_empty() {
            let launches: usize = untimed.iter().map(|kernel| kernel.launches).sum();
            writeln!(
                f,
                "  {:<width$} {:>8}      {} kernels, {launches} launches, measured no time: every component and share above is short by them",
                "untimed",
                "",
                untimed.len(),
            )?;
        }
        if measurement.breakdown.unplaced.timings > 0 {
            writeln!(
                f,
                "  {:<width$} {:>8} ms   {} timings arrived after their component closed",
                "unplaced",
                Millis(measurement.breakdown.unplaced.total),
                measurement.breakdown.unplaced.timings
            )?;
        }
        // The host/device split: the clean run less what its kernels took on
        // the device. Printed only where that device time is a pipelined
        // run's — see `HostSplit` — and a dash that says why
        // otherwise, rather than a subtraction of the wrong thing.
        match self.report.host_split() {
            HostSplit::Measured { host, .. } => writeln!(
                f,
                "  {:<width$} {:>8} ms   {} of the run the device spent waiting on the host",
                "host",
                Millis(host),
                Share {
                    part: host,
                    whole: self.report.wall(),
                },
            ),
            unsplit => {
                for (index, line) in unsplit.why_not_separable().iter().enumerate() {
                    match index {
                        0 => writeln!(
                            f,
                            "  {:<width$} {:>8}      not separable: {line}",
                            "host", "—"
                        )?,
                        _ => writeln!(f, "  {:<width$}               {line}", "")?,
                    }
                }
                Ok(())
            }
        }
    }
}
