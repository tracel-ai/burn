//! What the pass came to, beneath the component table.

use std::fmt::{self, Display};

use burn_instrument::Kernel;

use crate::Report;
use crate::millis::Millis;
use crate::reading::HostSplit;
use crate::share::Share;

/// The pass's totals, and what each is short by: the uncollected run, the device time the
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
        let pass = self.report.pass();
        let width = self.width;
        let text = self.report.reading().text();
        let kernels = pass.breakdown.kernels().count();
        let launches: usize = pass.breakdown.kernels().map(|kernel| kernel.launches).sum();
        if let Some(wall) = pass.clean_wall {
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
            match pass.breakdown.host_timed {
                true => "measured (SOME HOST-TIMED)",
                false => "measured",
            },
            Millis(pass.breakdown.total_time()),
            text.measured,
        )?;
        if pass.breakdown.host_timed {
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
        let untimed: Vec<&Kernel> = pass.breakdown.untimed().collect();
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
        if pass.breakdown.unplaced.timings > 0 {
            writeln!(
                f,
                "  {:<width$} {:>8} ms   {} timings arrived after their component closed",
                "unplaced",
                Millis(pass.breakdown.unplaced.total),
                pass.breakdown.unplaced.timings
            )?;
        }
        // The host/device split: the clean pass less what its kernels took on
        // the device. Printed only where that device time is a pipelined
        // pass's — see `HostSplit` — and a dash that says why
        // otherwise, rather than a subtraction of the wrong thing.
        match self.report.host_split() {
            HostSplit::Measured { host, .. } => writeln!(
                f,
                "  {:<width$} {:>8} ms   {} of the pass the device spent waiting on the host",
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
