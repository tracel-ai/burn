//! What each member of a sharded compute device did, beneath the totals, where the run had
//! several: the components sum every member's kernels, so no line above is one device's time.
//!
//! Idle time is the instrumented run's, whose host pays a round trip per launch, so it runs
//! past the clean run's wall; what the wall holds beyond the slowest member is the clean run's.

use std::fmt::{self, Display};

use crate::Report;
use crate::millis::Millis;

/// Each member's device time and idle time, and the slowest member against the wall.
pub(super) struct Members<'r, 'a> {
    report: &'r Report<'a>,
    /// The width of the component column the lines sit under.
    width: usize,
}

impl<'r, 'a> Members<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>, width: usize) -> Self {
        Self { report, width }
    }
}

impl Display for Members<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let measurement = self.report.measurement();
        let Some(per_member) = &measurement.breakdown.per_member else {
            return Ok(());
        };
        let width = self.width;
        // A run whose launches were timed alone was serialized: no member ran beside another,
        // so their times say nothing about the run as it overlaps.
        if measurement.breakdown.timed_alone {
            return writeln!(
                f,
                "  {:<width$} {:>8}      not timed apart: launches timed alone ran one at a time",
                "members", "—"
            );
        }
        for (index, member) in per_member.members.iter().enumerate() {
            writeln!(
                f,
                "  {:<width$} {:>8} ms   on device, {} ms idle in the instrumented run",
                format!("member {index}"),
                Millis(member.on_device),
                Millis(member.idle),
            )?;
        }
        if !per_member.off_members.is_zero() {
            writeln!(
                f,
                "  {:<width$} {:>8} ms   launched off the members' workers, on no member's line",
                "off members",
                Millis(per_member.off_members),
            )?;
        }
        let slowest = per_member
            .members
            .iter()
            .map(|member| member.on_device)
            .max()
            .unwrap_or_default();
        match measurement.clean_wall {
            Some(wall) if wall >= slowest => writeln!(
                f,
                "  {:<width$} {:>8} ms   of the clean run outside the slowest member's kernels: \
                 collectives, the host, and imbalance between members, at most",
                "not computing",
                Millis(wall - slowest),
            ),
            Some(_) => writeln!(
                f,
                "  {:<width$} {:>8}      the slowest member's kernels measured more than the run \
                 took, so the two clocks disagree",
                "not computing", "—"
            ),
            None => Ok(()),
        }
    }
}
