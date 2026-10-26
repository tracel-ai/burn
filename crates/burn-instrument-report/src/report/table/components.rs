//! The component tree, each component scored against the wall it is bound by.

use std::fmt::{self, Display};
use std::time::Duration;

use burn_instrument::{Component, FusedShare, Kernel};

use crate::Report;
use crate::millis::Millis;
use crate::served_by::ServedBy;
use crate::share::Share;

/// The table's body: each component's line, its children's beneath it, then the kernels it
/// launched itself, and the kernels no component claimed.
///
/// The component column sizes itself to the widest heading, variants included, and the rules
/// and the totals below follow it — so one long execution name widens the table rather than
/// pushing every number on one component out of its column.
pub(super) struct ComponentTable<'r, 'a> {
    report: &'r Report<'a>,
    /// The device time every share is of.
    whole: Duration,
    /// The width of the component column.
    width: usize,
}

impl<'r, 'a> ComponentTable<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>) -> Self {
        let breakdown = &report.measurement().breakdown;
        Self {
            report,
            whole: breakdown.total_time(),
            width: column_width(&breakdown.tree),
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
        component: &Component,
        depth: usize,
    ) -> fmt::Result {
        let fused = self.report.roofline().fused_elsewhere().share(component);
        let indent = "  ".repeat(depth);
        let label = fit(heading(component), self.width.saturating_sub(depth * 2));
        let scored = self.report.roofline().score_of(component);
        let (bound, achieved, floor, gain) =
            match (scored, self.report.roofline().floor_of(component)) {
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
                    Millis(score.gain(self.report.reading().apportioned(component))).to_string(),
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
        let (time, of_work, host) = match fused {
            Some(FusedShare::All) => ("fused ↓".to_string(), String::new(), String::new()),
            Some(FusedShare::Part) | None => (
                Millis(self.report.reading().apportioned(component)).to_string(),
                Share {
                    part: component.total_time(),
                    whole: self.whole,
                }
                .to_string(),
                self.report
                    .host_split()
                    .share(component.total_wait())
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
        for child in &component.children {
            self.write_component(f, child, depth + 1)?;
        }
        // The kernels this component launched itself, after the components it
        // entered. A kernel here that sits above a component's own component is one that
        // fused across it — the fuser's behaviour, read off the indentation rather
        // than asked about.
        for (name, kernel) in named(&component.kernels) {
            writeln!(
                f,
                "  {}· {:<width$} {:>8} {:>7}",
                "  ".repeat(depth + 1),
                fit(
                    launched(&name, kernel),
                    self.width.saturating_sub(depth * 2 + KERNEL_INDENT)
                ),
                Millis(self.report.reading().apportion(kernel.total)),
                Share {
                    part: kernel.total,
                    whole: self.whole,
                },
                width = self.width.saturating_sub(depth * 2 + KERNEL_INDENT),
            )?;
        }
        Ok(())
    }
}

impl Display for ComponentTable<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let breakdown = &self.report.measurement().breakdown;
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
        for component in &breakdown.tree {
            self.write_component(f, component, 0)?;
        }
        // Kernels nobody's component claimed, and the fused work the last
        // components left for the sync that ends the run. Work the device did
        // is still work, and a component saying so beats a silent omission.
        for (heading, kernels) in [
            ("end of run", &breakdown.end_of_run),
            ("untagged", &breakdown.untagged),
        ] {
            if kernels.is_empty() {
                continue;
            }
            writeln!(f, "  {heading:<width$}")?;
            for (name, kernel) in named(kernels) {
                writeln!(
                    f,
                    "    · {:<width$} {:>8}",
                    fit(launched(&name, kernel), width.saturating_sub(KERNEL_INDENT)),
                    Millis(self.report.reading().apportion(kernel.total)),
                    width = width.saturating_sub(KERNEL_INDENT),
                )?;
            }
        }
        // The marks a fused run adds, explained where they are read and only
        // where they appear.
        if breakdown.kernels().any(|kernel| !kernel.fuses.is_empty()) {
            writeln!(f)?;
            writeln!(
                f,
                "  · fuses ↑ a + b   one kernel did the named components' work; it sits on their closest parent, or outside every component"
            )?;
            writeln!(
                f,
                "  fused ↓           a component whose work all ran in such a kernel below it; its floor is set aside"
            )?;
            let partly_fused = breakdown
                .tree
                .iter()
                .flat_map(|component| component.walk())
                .any(|(_, component)| {
                    self.report.roofline().fused_elsewhere().share(component)
                        == Some(FusedShare::Part)
                });
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

/// How wide the component column has to be for `tree` to fit in it.
///
/// Computed rather than fixed: a heading carries its variants, and the longest
/// of those is not knowable in advance. Floored so a shallow tree still lines
/// up with the summary beneath it, and capped so one pathological name cannot
/// push every number off the right of a terminal. The cap is wide enough for a
/// nested component that routed three ways — the case the variant column exists for —
/// and a heading past it is elided by [`fit`] rather than overflowing, so the
/// columns hold whatever a component is called.
fn column_width(tree: &[Component]) -> usize {
    const NARROWEST: usize = 28;
    const WIDEST: usize = 64;

    tree.iter()
        .flat_map(|component| component.walk())
        .flat_map(|(depth, component)| {
            // The kernel lines share this column with the component headings,
            // one indent deeper, and a kernel that names the components it
            // fused is the longest thing in the table. Measured here because
            // the alternative is a `· fuses ↑ a + b` elided down to `· fuse…` —
            // the annotation cut off is exactly the one nothing else states.
            let kernels: Vec<usize> = named(&component.kernels)
                .into_iter()
                .map(|(name, kernel)| {
                    depth * 2 + KERNEL_INDENT + launched(&name, kernel).chars().count()
                })
                .collect();
            std::iter::once(depth * 2 + heading(component).chars().count()).chain(kernels)
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
fn launched(name: &str, kernel: &Kernel) -> String {
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

/// A component's kernels under the shortest names that still tell them apart.
///
/// A kernel's type name carries its whole generic instantiation and runs to
/// several hundred characters, so the table would be unreadable at full
/// length. `Short` — the last path segment before the generics — is what reads
/// well, but three matmul families instantiate one `MatmulEntry` and collapsing
/// them would print two different kernels under one name, which is worse than
/// a long name. So `Short` is used only where it stays unambiguous *within the
/// component*, and the fuller form serves the rest.
fn named(kernels: &[Kernel]) -> Vec<(String, &Kernel)> {
    use cubecl::config::{TypeNameFormatLevel, type_name_format};

    let short: Vec<String> = kernels
        .iter()
        .map(|kernel| type_name_format(&kernel.name, TypeNameFormatLevel::Short))
        .collect();
    let distinct = short.iter().collect::<std::collections::HashSet<_>>().len();
    let level = match distinct == kernels.len() {
        true => TypeNameFormatLevel::Short,
        false => TypeNameFormatLevel::Balanced,
    };
    kernels
        .iter()
        .map(|kernel| (type_name_format(&kernel.name, level), kernel))
        .collect()
}
