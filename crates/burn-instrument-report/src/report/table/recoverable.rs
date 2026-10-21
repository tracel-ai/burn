//! How much of a run is on the table, lever by lever.

use std::fmt::{self, Display};

use crate::millis::Millis;
use crate::roofline::Ladder;

/// The ladder as the page prints it: what the run could give back, one rung per lever, or why
/// that is unavailable.
pub(super) struct Recoverable {
    ladder: Ladder,
}

impl Recoverable {
    pub(super) fn new(ladder: Ladder) -> Self {
        Self { ladder }
    }
}

impl Display for Recoverable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ladder = &self.ladder;
        if !ladder.coherent() {
            // Saturating the subtractions would print "0.0 on the table" under
            // rungs that plainly exceed the run — the most misleading thing
            // this report could say, since it reads as "nothing to win".
            writeln!(f)?;
            for line in [
                "how much is on the table   unavailable".to_string(),
                format!(
                    "  the floors ({} ms / {} ms) do not fit inside the {} ms",
                    Millis(ladder.kernels),
                    Millis(ladder.fused),
                    Millis(ladder.device())
                ),
                "  the device spent on this run, so a denominator is wrong".to_string(),
            ] {
                writeln!(f, "  {line}")?;
            }
            return Ok(());
        }
        writeln!(
            f,
            "\n  how much is on the table{:>22} ms",
            Millis(ladder.total())
        )?;
        // The outermost lever where the split is known: time the device
        // spent idle, which no kernel and no fusion recovers.
        if !ladder.host.is_zero() {
            writeln!(
                f,
                "    {:<34} {:>8} ms   ← fewer launches {:>10} ms",
                "the device never waiting",
                Millis(ladder.device()),
                Millis(ladder.host)
            )?;
        }
        writeln!(
            f,
            "    {:<34} {:>8} ms   ← faster kernels {:>10} ms",
            "kernels at their own roofs",
            Millis(ladder.kernels),
            Millis(ladder.from_kernels())
        )?;
        writeln!(
            f,
            "    {:<34} {:>8} ms   ← fewer round-trips {:>7} ms",
            "perfect fusion inside components",
            Millis(ladder.fused),
            Millis(ladder.from_fusion())
        )
    }
}
