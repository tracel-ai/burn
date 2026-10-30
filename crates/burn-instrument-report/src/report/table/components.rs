//! The component tree, each component scored against the wall it is bound by.

use std::fmt::{self, Display};

use burn_instrument::{Component, FusedShare, Kernel};

use crate::Report;
use crate::components::{ComponentReading, KernelReading, UnclaimedReading, WorkShare};
use crate::millis::Millis;
use crate::served_by::ServedBy;

/// The table's body: each component's line, its children's beneath it, then the kernels it
/// launched itself, and the kernels no component claimed.
///
/// The component column sizes itself to the widest heading, variants included, and the rules
/// and the totals below follow it — so one long execution name widens the table rather than
/// pushing every number on one component out of its column.
pub(super) struct ComponentTable<'r, 'a> {
    report: &'r Report<'a>,
    /// The run's components, read once: what every line prints, and what the column's width
    /// and the legend are measured from.
    components: Vec<ComponentReading<'a>>,
    /// The kernels no component claimed, read once.
    unclaimed: Vec<UnclaimedReading<'a>>,
    /// The width of the component column.
    width: usize,
}

impl<'r, 'a> ComponentTable<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>) -> Self {
        let components = report.components();
        Self {
            report,
            width: column_width(&components),
            unclaimed: report.unclaimed(),
            components,
        }
    }

    /// The width of the component column, which the totals beneath the table line up under.
    pub(super) fn width(&self) -> usize {
        self.width
    }

    /// A component's line, and its children's beneath it, `depth` components below the root.
    fn write_component(
        &self,
        f: &mut fmt::Formatter<'_>,
        reading: &ComponentReading<'_>,
        depth: usize,
    ) -> fmt::Result {
        let fused = reading.fused;
        let indent = "  ".repeat(depth);
        let label = fit(
            heading(reading.component),
            self.width.saturating_sub(depth * 2),
        );
        let (bound, achieved, floor, gain) = match (reading.score, reading.floor) {
            (Some(score), _) => (
                score.floor.bound.to_string(),
                // A component above 100% is marked and keeps its figure. The peaks are
                // measured rather than looked up, so 105% is a component at the roof read
                // through a probe that came in a little low, and 568% is a declared
                // bound that is wrong — different findings, and one number
                // separates them where a shared ">100%" gave both the same answer.
                // Marked rather than suppressed: this is usually the most
                // interesting component on the page. The systemically broken denominator
                // is caught at the peak, not here.
                //
                // Only the bar is clamped, because a bar cannot say more than full.
                match score.achieved > 1.0 {
                    true => format!("{} {:>3.0}%*", bar(1.0), score.achieved * 100.0),
                    false => {
                        format!("{} {:>3.0}% ", bar(score.achieved), score.achieved * 100.0)
                    }
                },
                Millis(score.floor.time).to_string(),
                Millis(score.gain(reading.time.unwrap_or_default())).to_string(),
            ),
            // Priced but not scored — it measured no time of its own, or only part
            // of its work's, the rest fused into a kernel listed below it, which says
            // so where the percentage would be. The floor is still the
            // declaration's, and its parent's floor counts it, so the component says
            // what that is rather than leaving the parent's figure to be reconciled
            // against blanks.
            (None, Some(floor)) => (
                floor.bound.to_string(),
                match fused {
                    Some(FusedShare::Part) => "partly fused ↓".to_string(),
                    Some(FusedShare::All) | None => String::new(),
                },
                Millis(floor.time).to_string(),
                "—".to_string(),
            ),
            // A component nobody priced is timed and says so, rather than being
            // scored against a floor no kernel could meet.
            (None, None) => (
                "—".to_string(),
                String::new(),
                "—".to_string(),
                "—".to_string(),
            ),
        };
        // `fused ↓` rather than `0.0`: a zero reads as a component that cost
        // nothing, and this one's cost is in a fused kernel listed below it — on a
        // component above, after that component's children, or outside every component.
        let (time, of_work, host) = match reading.time {
            None => ("fused ↓".to_string(), String::new(), String::new()),
            Some(time) => (
                Millis(time).to_string(),
                share_column(reading.share),
                reading
                    .host
                    .map(|host| Millis(host).to_string())
                    .unwrap_or_default(),
            ),
        };
        writeln!(
            f,
            "  {indent}{:<width$} {:>8} {:>7} {:>7}  {:<8} {:<18} {:>8} {:>8}",
            label,
            time,
            of_work,
            host,
            bound,
            achieved,
            floor,
            gain,
            width = self.width.saturating_sub(depth * 2),
        )?;
        for child in &reading.children {
            self.write_component(f, child, depth + 1)?;
        }
        // The kernels this component launched itself, after the components it
        // entered. A kernel here that sits above a component's own component is one that
        // fused across it — the fuser's behaviour, read off the indentation rather
        // than asked about.
        for kernel in &reading.kernels {
            writeln!(
                f,
                "  {}· {:<width$} {:>8} {:>7}",
                "  ".repeat(depth + 1),
                fit(
                    launched(kernel),
                    self.width.saturating_sub(depth * 2 + KERNEL_INDENT)
                ),
                Millis(kernel.time),
                share_column(kernel.share),
                width = self.width.saturating_sub(depth * 2 + KERNEL_INDENT),
            )?;
        }
        Ok(())
    }
}

impl Display for ComponentTable<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self.width;
        let rule = width + 70;
        writeln!(
            f,
            "\n  {:<width$} {:>8} {:>7} {:>7}  {:<8} {:<18} {:>8} {:>8}",
            "component", "ms", "share", "host", "bound", "achieved", "floor ms", "gain ms"
        )?;
        // What the two bare-percentage columns are *of*. `share` and
        // `achieved` are both percentages and mean different things, and a
        // reader who takes one for the other reads a component at 3% of the
        // memory peak as 3% of the run.
        writeln!(
            f,
            "  {:<width$} {:>8} {:>7} {:>7}  {:<8} {:<18} {:>8} {:>8}",
            "",
            self.report.reading().text().time_of,
            "of work",
            "waited",
            "",
            "of peak",
            "at best",
            "recoverable"
        )?;
        writeln!(f, "  {}", "─".repeat(rule))?;
        for component in &self.components {
            self.write_component(f, component, 0)?;
        }
        // Kernels nobody's component claimed, and the fused work the last
        // components left for the sync that ends the run. Work the device did
        // is still work, and a component saying so beats a silent omission.
        for group in &self.unclaimed {
            writeln!(f, "  {:<width$}", group.unclaimed)?;
            for kernel in &group.kernels {
                writeln!(
                    f,
                    "    · {:<width$} {:>8}",
                    fit(launched(kernel), width.saturating_sub(KERNEL_INDENT)),
                    Millis(kernel.time),
                    width = width.saturating_sub(KERNEL_INDENT),
                )?;
            }
        }
        // The marks a fused run adds, explained where they are read and only
        // where they appear.
        let readings: Vec<&ComponentReading<'_>> = self
            .components
            .iter()
            .flat_map(ComponentReading::walk)
            .map(|(_, reading)| reading)
            .collect();
        let fuses = readings
            .iter()
            .flat_map(|reading| &reading.kernels)
            .chain(self.unclaimed.iter().flat_map(|group| &group.kernels))
            .any(|kernel| !kernel.kernel.fuses.is_empty());
        if fuses {
            writeln!(f)?;
            writeln!(
                f,
                "  · fuses ↑ a + b   one kernel did the named components' work; it sits on their closest parent, or outside every component"
            )?;
            writeln!(
                f,
                "  fused ↓           a component whose work all ran in such a kernel below it; its floor is set aside"
            )?;
            let partly_fused = readings
                .iter()
                .any(|reading| reading.fused == Some(FusedShare::Part));
            if partly_fused {
                writeln!(
                    f,
                    "  partly fused ↓    a component part of whose work ran in such a kernel; its time is only the rest, so it is not scored"
                )?;
            }
        }
        writeln!(f, "  {}", "─".repeat(rule))
    }
}

/// A ten-cell bar. The two of them on a component are the whole point of the column:
/// one number cannot separate a `bytes` floor from a `flops` one from neither.
fn bar(fraction: f64) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * 10.0).round() as usize;
    format!("{}{}", "▓".repeat(filled), "░".repeat(10 - filled))
}

/// A component's heading: what it is called, how many times the run entered it,
/// and which executions served those entries.
///
/// The variant sits beside the count rather than on a line of its own because a
/// component that routed one way is one fact about that component, and a second line per
/// component would double a 40-component table to say it.
fn heading(component: &Component) -> String {
    let mut heading = match component.entries {
        1 => component.label.clone(),
        many => format!("{} ×{many}", component.label),
    };
    let served_by = ServedBy(component);
    if !served_by.is_empty() {
        heading.push_str(&format!(" · {served_by}"));
    }
    heading
}

/// How wide the component column has to be for `components` to fit in it.
///
/// Computed rather than fixed: a heading carries its variants, and the longest
/// of those is not knowable in advance. Floored so a shallow tree still lines
/// up with the summary beneath it, and capped so one pathological name cannot
/// push every number off the right of a terminal. The cap is wide enough for a
/// nested component that routed three ways — the case the variant column exists for —
/// and a heading past it is elided by [`fit`] rather than overflowing, so the
/// columns hold whatever a component is called.
fn column_width(components: &[ComponentReading<'_>]) -> usize {
    const NARROWEST: usize = 28;
    const WIDEST: usize = 64;

    components
        .iter()
        .flat_map(ComponentReading::walk)
        .flat_map(|(depth, reading)| {
            // The kernel lines share this column with the component headings,
            // one indent deeper, and a kernel that names the components it
            // fused is the longest thing in the table. Measured here because
            // the alternative is a `· fuses ↑ a + b` elided down to `· fuse…` —
            // the annotation cut off is exactly the one nothing else states.
            let kernels: Vec<usize> = reading
                .kernels
                .iter()
                .map(|kernel| depth * 2 + KERNEL_INDENT + launched(kernel).chars().count())
                .collect();
            std::iter::once(depth * 2 + heading(reading.component).chars().count()).chain(kernels)
        })
        .max()
        .unwrap_or(NARROWEST)
        .clamp(NARROWEST, WIDEST)
}

/// `heading` in at most `width` columns.
///
/// Only a heading past [`column_width`]'s cap is cut, and it is cut visibly: a
/// component that quietly overflowed would push its own figures out of the columns
/// every other component is aligned to, so the one line a reader cannot parse would
/// be the busiest one in the table.
fn fit(heading: String, width: usize) -> String {
    match heading.chars().count() <= width {
        true => heading,
        false => heading
            .chars()
            .take(width.saturating_sub(1))
            .chain(['…'])
            .collect(),
    }
}

/// How much further in a kernel's name starts than its component's heading:
/// one level of indent and the `· ` that marks a kernel. Taken off the name's
/// width, so the kernel's figures line up under its component's.
const KERNEL_INDENT: usize = 4;

/// A kernel's line: its name, how many times it launched, and — where it
/// fused components' work — whose, listed above it.
fn launched(reading: &KernelReading<'_>) -> String {
    let (name, kernel): (&str, &Kernel) = (&reading.name, reading.kernel);
    match kernel.fuses.as_slice() {
        [] => format!("{name} ×{}", kernel.launches),
        fuses => format!(
            "{name} ×{} · fuses ↑ {}",
            kernel.launches,
            fuses
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" + ")
        ),
    }
}

/// A share as the table's column prints it, blank where the run timed nothing.
fn share_column(share: Option<WorkShare>) -> String {
    share.map(|share| share.to_string()).unwrap_or_default()
}
