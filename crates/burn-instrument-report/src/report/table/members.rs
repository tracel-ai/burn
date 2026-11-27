//! What each member did, beneath the totals, where the run had several: the components are the
//! lead's where the members ran at once and were read at the lead, and sum every member's
//! kernels otherwise, which is the run's device time only for members that took turns.
//!
//! Idle time is the instrumented run's, whose host pays a round trip per launch, so it runs
//! past the clean run's wall; what the wall holds beyond the slowest member is the clean run's.

use std::fmt::{self, Display};

use burn_instrument::{MemberSchedule, PerMember};

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
                "  {:<width$} {:>8} ms   ran on no member's device, on no member's line",
                "off members",
                Millis(per_member.off_members),
            )?;
        }
        let (computing, outside) = computing(per_member);
        match measurement.clean_wall {
            Some(wall) if wall >= computing => writeln!(
                f,
                "  {:<width$} {:>8} ms   of the clean run outside {outside}, at most",
                "not computing",
                Millis(wall - computing),
            ),
            Some(_) => writeln!(
                f,
                "  {:<width$} {:>8}      the members' kernels measured more than the run took, \
                 so the two clocks disagree",
                "not computing", "—"
            ),
            None => Ok(()),
        }
    }
}

/// How long the run computed, and what the clean run outside it went to: the slowest member's
/// kernels where the members worked at once, all of theirs where they took turns.
fn computing(per_member: &PerMember) -> (std::time::Duration, &'static str) {
    let times = per_member.members.iter().map(|member| member.on_device);
    match per_member.schedule {
        MemberSchedule::Concurrent => (
            times.max().unwrap_or_default(),
            "the slowest member's kernels: collectives, the host, and imbalance between members",
        ),
        MemberSchedule::Sequential => (
            times.sum(),
            "every member's kernels: the host, and the hops between members",
        ),
    }
}
