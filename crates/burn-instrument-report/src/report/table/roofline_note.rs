//! What a component past its measured peak means.

use std::fmt::{self, Display};

use crate::Report;

/// The note under a table where some component reads past its peak (`*`), or nothing where
/// none does.
///
/// Where priced work timed nothing, kernel time was lost, and a component past its peak is far
/// likelier a time that came in short than a kernel past the hardware — "nothing to win" would
/// wave off the one component that shows it.
pub(super) struct RooflineNote<'r, 'a> {
    report: &'r Report<'a>,
}

impl<'r, 'a> RooflineNote<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>) -> Self {
        Self { report }
    }
}

impl Display for RooflineNote<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let measurement = self.report.measurement();
        let past_peak = measurement
            .breakdown
            .tree
            .iter()
            .flat_map(|component| component.walk())
            .any(|(_, component)| {
                self.report
                    .roofline()
                    .score_of(component)
                    .is_some_and(|score| score.achieved > 1.0)
            });
        if !past_peak {
            return Ok(());
        }
        // One `writeln!` per line: a `\n` inside a format string is
        // indented by rustfmt into the output, which puts the second line
        // of every note halfway across the page.
        writeln!(f)?;
        let lines: &[&str] = match measurement.breakdown.untimed().next().is_none() {
            true => &[
                "* past the measured peak, which charges every byte to DRAM. How far",
                "  past is the reading. A few percent over is a component at the roof seen",
                "  through a probe that came in low, since the peaks are measured and",
                "  not looked up. Well over means the bytes were not all paid for — a",
                "  small, freshly written working set served from cache, which no bound",
                "  stated in bytes can say — or its kernels are too brief to time at this",
                "  resolution, or its bounds are wrong. Nothing to win here either way.",
            ],
            false => &[
                "* past the measured peak, in a run that lost kernel time (`untimed`",
                "  above): a component past its peak here is a time that came in short, not",
                "  a component at the roof. Read none of these percentages.",
            ],
        };
        for line in lines {
            writeln!(f, "  {line}")?;
        }
        // The one reading a fused run adds: a component part of whose work fused
        // with a neighbour's took less than its floor prices, since that
        // part runs in a kernel recorded elsewhere.
        if !measurement.breakdown.unfused {
            for line in [
                "  In a fused run it can also be a component part of whose work fused with a",
                "  neighbour's: that part ran in a kernel listed on a component above it, or",
                "  outside every component (`· fuses ↑`).",
            ] {
                writeln!(f, "  {line}")?;
            }
        }
        Ok(())
    }
}
