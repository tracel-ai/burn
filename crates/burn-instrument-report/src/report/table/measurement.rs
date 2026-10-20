//! How the pass was measured.

use std::fmt::{self, Display};

use crate::Report;
use crate::millis::Millis;

/// How the pass was measured, one line per question a reader asks of a number.
///
/// Three independent axes, so each gets its own word and none borrows another's: what the
/// clean pass took, which graph ran (fused or not), and how the kernels were timed. "Eager",
/// which in burn means *unfused*, is the word this section does not use.
pub(super) struct Measurement<'r, 'a> {
    report: &'r Report<'a>,
}

impl<'r, 'a> Measurement<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>) -> Self {
        Self { report }
    }
}

impl Display for Measurement<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pass = self.report.pass();
        writeln!(
            f,
            "\n  {:<10}{}",
            "pass",
            match pass.clean_wall {
                Some(wall) => format!(
                    "{} ms run uncollected, the time the components below break down",
                    Millis(wall)
                ),
                None => "no uncollected run measured: no host/device split".to_string(),
            }
        )?;
        let graph = match pass.breakdown.unfused {
            true => "unfused — one kernel per operation: not the kernels a fused build runs",
            false => "fused — the kernels a fused build runs",
        };
        writeln!(f, "  {:<10}{graph}", "graph")?;
        let [first, second] = self.report.reading().text().timings;
        writeln!(f, "  {:<10}{first}", "timings")?;
        writeln!(f, "  {:<10}{second}", "")
    }
}
