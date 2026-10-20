//! How a report lays itself out as text.

use std::fmt::{self, Display, Write as _};
use std::time::Duration;

use burn_instrument::{Component, Kernel, PeakFinding};

use super::worklist::Worklist;
use super::{HostSplit, Report, TimeReading};

impl Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pass = self.pass();

        let peak = |value: Option<f64>, scale: f64, unit: &str| match value {
            Some(value) => format!("{:.1} {unit}", value / scale),
            None => "unprobed".to_string(),
        };
        writeln!(
            f,
            "device   {} · {} · {}",
            pass.device,
            peak(self.peaks().copy_bytes_per_s, 1e9, "GB/s"),
            match self.peaks().flops {
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
        if let (Some(read), Some(write)) = (
            self.peaks().read_bytes_per_s,
            self.peaks().write_bytes_per_s,
        ) {
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
        if pass.peaks.is_none() {
            writeln!(
                f,
                "         not probed: no component is scored, and the ladder is the measured time"
            )?;
        }
        for finding in self.findings() {
            for line in note(finding) {
                writeln!(f, "         {line}")?;
            }
        }

        let device = pass.breakdown.total_time();
        let untimed: Vec<&Kernel> = pass.breakdown.untimed().collect();

        // What ran, one line per question a reader asks of a number. Three
        // independent axes, so each gets its own word and none borrows
        // another's: what the clean pass took, which graph ran (fused or not),
        // and how the kernels were timed. "Eager", which in burn means
        // *unfused*, is the word this block does not use.
        writeln!(
            f,
            "\n  {:<10}{}",
            "pass",
            match pass.clean_wall {
                Some(wall) => format!(
                    "{} ms run uncollected, the time the components below break down",
                    ms(wall)
                ),
                None => "no uncollected run measured: no host/device split".to_string(),
            }
        )?;
        let graph = match pass.breakdown.unfused {
            true => "unfused — one kernel per operation: not the kernels a fused build runs",
            false => "fused — the kernels a fused build runs",
        };
        let timings: &[&str] = match self.reading() {
            TimeReading::InPlace => &[
                "in place — read back after the pass, the kernels ran back to back: a component is",
                "its kernels' device time, and `framework` is the device waiting on the host",
            ],
            TimeReading::ApportionedOnto(_) => &[
                "alone — each read as it launched, so the kernels ran one at a time: a component is",
                "its share of the uncollected run, good to a few percent",
            ],
            TimeReading::Serialized => &[
                "alone — each read as it launched, so the kernels ran one at a time: a component is",
                "its kernels' serialized time, and only the proportions mean anything",
            ],
        };
        writeln!(f, "  {:<10}{graph}", "graph")?;
        writeln!(f, "  {:<10}{}", "timings", timings[0])?;
        for line in &timings[1..] {
            writeln!(f, "  {:<10}{line}", "")?;
        }

        // The component column sizes itself to the widest heading, variants
        // included, and the rule and the summary below follow it — so one long
        // execution name widens the table rather than pushing every number on
        // one component out of its column.
        let width = column_width(&pass.breakdown.tree);
        let rule = width + 70;
        writeln!(
            f,
            "\n  {:<width$} {:>8} {:>7} {:>7}  {:<8} {:<18} {:>8} {:>8}",
            "component", "ms", "share", "host", "bound", "achieved", "floor ms", "gain ms"
        )?;
        // What the two bare-percentage columns are *of*. `share` and
        // `achieved` are both percentages and mean different things, and a
        // reader who takes one for the other reads a component at 3% of the
        // memory peak as 3% of the pass.
        writeln!(
            f,
            "  {:<width$} {:>8} {:>7} {:>7}  {:<8} {:<18} {:>8} {:>8}",
            "",
            match self.reading() {
                TimeReading::ApportionedOnto(_) => "of wall",
                TimeReading::InPlace | TimeReading::Serialized => "on device",
            },
            "of work",
            "waited",
            "",
            "of peak",
            "at best",
            "recoverable"
        )?;
        writeln!(f, "  {}", "─".repeat(rule))?;
        let mut body = String::new();
        let lines = ComponentLines {
            report: self,
            whole: device,
            width,
        };
        for component in &pass.breakdown.tree {
            lines.write(&mut body, component, &[])?;
        }
        f.write_str(&body)?;
        // Kernels nobody's component claimed, and the fused work the last
        // components left for the sync that ends the pass. Work the device did
        // is still work, and a component saying so beats a silent omission.
        for (heading, kernels) in [
            ("end of pass", &pass.breakdown.end_of_pass),
            ("untagged", &pass.breakdown.untagged),
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
                    ms(self.apportion(kernel.total)),
                    width = width.saturating_sub(KERNEL_INDENT),
                )?;
            }
        }
        // The two marks a fused pass adds, explained where they are read and
        // only where they appear.
        let fused_kernels = pass
            .breakdown
            .kernels()
            .any(|kernel| !kernel.fuses.is_empty());
        if fused_kernels {
            writeln!(f)?;
            writeln!(
                f,
                "  · fuses ↑ a + b   one kernel did the named components' work; it sits on their closest parent, or outside every component"
            )?;
            writeln!(
                f,
                "  fused ↓           a component whose work all ran in such a kernel below it; its floor is set aside"
            )?;
        }
        writeln!(f, "  {}", "─".repeat(rule))?;
        let kernels = pass.breakdown.kernels().count();
        let launches: usize = pass.breakdown.kernels().map(|kernel| kernel.launches).sum();
        if let Some(wall) = pass.clean_wall {
            writeln!(
                f,
                "  {:<width$} {:>8} ms   {}",
                "wall",
                ms(wall),
                match self.reading() {
                    TimeReading::ApportionedOnto(_) => {
                        "the pass run uncollected, and what the components above sum to"
                    }
                    TimeReading::InPlace | TimeReading::Serialized => "the pass run uncollected",
                }
            )?;
        }
        writeln!(
            f,
            "  {:<width$} {:>8} ms   {kernels} distinct kernels, {launches} launches, {}",
            match pass.breakdown.host_timed {
                true => "measured (SOME HOST-TIMED)",
                false => "measured",
            },
            ms(device),
            match self.reading() {
                TimeReading::InPlace => "timed in place — what the components above sum to",
                TimeReading::ApportionedOnto(_) | TimeReading::Serialized => "each timed alone",
            }
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
                ms(pass.breakdown.unplaced.total),
                pass.breakdown.unplaced.timings
            )?;
        }
        // The host/device split: the clean pass less what its kernels took on
        // the device. Printed only where that device time is a pipelined
        // pass's — see `Report::framework` — and a dash that says why
        // otherwise, rather than a subtraction of the wrong thing.
        let not_separable = |f: &mut fmt::Formatter<'_>, why: &[&str]| -> fmt::Result {
            writeln!(
                f,
                "  {:<width$} {:>8}      not separable: {}",
                "framework", "—", why[0]
            )?;
            for line in &why[1..] {
                writeln!(f, "  {:<width$}               {line}", "")?;
            }
            Ok(())
        };
        match self.host_split() {
            HostSplit::Measured(framework) => writeln!(
                f,
                "  {:<width$} {:>8} ms   {} of the pass the device spent waiting on the host",
                "framework",
                ms(framework),
                share(framework, self.wall()),
            )?,
            HostSplit::NoCleanRun => not_separable(f, &["no uncollected run was measured"])?,
            HostSplit::TimedAlone => not_separable(
                f,
                &[
                    "a launch timed alone serializes the",
                    "pass, so the measured total is not a pipelined pass's",
                ],
            )?,
            HostSplit::NothingTimed => not_separable(f, &["nothing was timed on the device"])?,
            HostSplit::ClocksDisagree => not_separable(
                f,
                &[
                    "the kernels measured more than the",
                    "pass took, so the two clocks disagree",
                ],
            )?,
        }

        let ladder = self.ladder();
        let at_the_roof = self
            .pass()
            .breakdown
            .tree
            .iter()
            .flat_map(|component| component.walk())
            .any(|(_, component)| {
                self.score_of(component)
                    .is_some_and(|score| score.achieved > 1.0)
            });
        if at_the_roof {
            // One `writeln!` per line: a `\n` inside a format string is
            // indented by rustfmt into the output, which puts the second line
            // of every note halfway across the page.
            writeln!(f)?;
            // Where priced work timed nothing, kernel time was lost, and a component
            // past its peak is far likelier a time that came in short than a
            // kernel past the hardware — "nothing to win" would wave off the
            // one component that shows it.
            let lines: &[&str] = match untimed.is_empty() {
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
                    "* past the measured peak, in a pass that lost kernel time (`untimed`",
                    "  above): a component past its peak here is a time that came in short, not",
                    "  a component at the roof. Read none of these percentages.",
                ],
            };
            for line in lines {
                writeln!(f, "  {line}")?;
            }
            // The one reading a fused pass adds: a component part of whose work fused
            // with a neighbour's took less than its floor prices, since that
            // part runs in a kernel recorded elsewhere.
            if !pass.breakdown.unfused {
                for line in [
                    "  In a fused pass it can also be a component part of whose work fused with a",
                    "  neighbour's: that part ran in a kernel listed on a component above it, or",
                    "  outside every component (`· fuses ↑`).",
                ] {
                    writeln!(f, "  {line}")?;
                }
            }
        }

        if ladder.coherent() {
            writeln!(
                f,
                "\n  how much is on the table{:>22} ms",
                ms(ladder.total())
            )?;
            // The outermost lever where the split is known: time the device
            // spent idle, which no kernel and no fusion recovers.
            if !ladder.host.is_zero() {
                writeln!(
                    f,
                    "    {:<34} {:>8} ms   ← fewer launches {:>10} ms",
                    "the device never waiting",
                    ms(ladder.device()),
                    ms(ladder.host)
                )?;
            }
            writeln!(
                f,
                "    {:<34} {:>8} ms   ← faster kernels {:>10} ms",
                "kernels at their own roofs",
                ms(ladder.kernels),
                ms(ladder.from_kernels())
            )?;
            writeln!(
                f,
                "    {:<34} {:>8} ms   ← fewer round-trips {:>7} ms",
                "perfect fusion inside components",
                ms(ladder.fused),
                ms(ladder.from_fusion())
            )?;
        } else {
            // Saturating the subtractions would print "0.0 on the table" under
            // rungs that plainly exceed the pass — the most misleading thing
            // this report could say, since it reads as "nothing to win".
            writeln!(f)?;
            for line in [
                "how much is on the table   unavailable".to_string(),
                format!(
                    "  the floors ({} ms / {} ms) do not fit inside the {} ms",
                    ms(ladder.kernels),
                    ms(ladder.fused),
                    ms(ladder.device())
                ),
                "  the device spent on this pass, so a denominator is wrong".to_string(),
            ] {
                writeln!(f, "  {line}")?;
            }
        }

        // Where one component was fed from two places. After the table, because it
        // is a fact about a component a reader has already seen, and never instead of
        // one: the tree is unchanged and the merge stands.
        for (under, component) in merged(&pass.breakdown.tree) {
            let sites: Vec<String> = component.sites.iter().map(|it| site(it)).collect();
            let under = match under.is_empty() {
                true => String::new(),
                false => format!(" under `{under}`"),
            };
            writeln!(
                f,
                "\n  note: `{}`{under} was entered from {} sites ({})",
                component.label,
                sites.len(),
                sites.join(", ")
            )?;
        }

        write!(f, "{}", Worklist::new(self))?;
        Ok(())
    }
}

/// A ten-cell bar. The two of them on a component are the whole point of the column:
/// one number cannot separate a `bytes` floor from a `flops` one from neither.
fn bar(fraction: f64) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * 10.0).round() as usize;
    format!("{}{}", "▓".repeat(filled), "░".repeat(10 - filled))
}

/// Milliseconds, at the precision a reader can act on.
pub(super) fn ms(duration: Duration) -> String {
    format!("{:.1}", duration.as_secs_f64() * 1e3)
}

/// A share of the whole, or blank when there is no whole to take one of.
fn share(part: Duration, whole: Duration) -> String {
    if whole.is_zero() {
        return String::new();
    }
    format!("{:.1}%", part.as_secs_f64() / whole.as_secs_f64() * 100.0)
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

/// A component's heading: what it is called, how many times the pass entered it,
/// which executions served those entries, and whether its floor includes
/// observed traffic.
///
/// The variant sits beside the count rather than on a line of its own because a
/// component that routed one way is one fact about that component, and a second line per
/// component would double a 40-component table to say it.
pub(super) fn heading(component: &Component) -> String {
    let mut heading = match component.entries {
        1 => component.label.clone(),
        many => format!("{} ×{many}", component.label),
    };
    if let Some(variants) = variants(component) {
        heading.push_str(" · ");
        heading.push_str(&variants);
    }
    if component.observed.is_some() {
        heading.push_str(" · floor read from the pass");
    }
    heading
}

/// Which executions served a component's entries, or `None` where it declares no
/// variant — which is most components.
///
/// A single execution that served every entry prints as its bare name: its
/// count would only repeat the component's own. Several print with theirs, because
/// *how the entries split* is the whole reason to look — `tile ×8, tensor ops
/// ×1` says a routing decision went one way eight times and the other once, and
/// that is a finding.
fn variants(component: &Component) -> Option<String> {
    match component.variants.as_slice() {
        [] => None,
        [only] if only.entries == component.entries => Some(only.name.clone()),
        several => Some(
            several
                .iter()
                .map(|variant| format!("{} ×{}", variant.name, variant.entries))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    }
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

/// Components fed from more than one call site, each with the path that names it.
///
/// A merge is not a fault — one component entered 36 times is what makes a stack
/// readable, and two sites under one label is the same mechanism. It is worth a
/// note only because it is also how one component comes to hold two different jobs:
/// two entry points that both write a cache under one label would sum two
/// different writes under one heading.
///
/// A site is a function, so this reports two *entry points*, never two roles of
/// one entry point — that is what a caller-supplied label is for.
fn merged(tree: &[Component]) -> Vec<(String, &Component)> {
    fn descend<'a>(component: &'a Component, under: &str, out: &mut Vec<(String, &'a Component)>) {
        if component.sites.len() > 1 {
            out.push((under.to_string(), component));
        }
        for child in &component.children {
            descend(child, &component.label, out);
        }
    }
    let mut out = Vec::new();
    for root in tree {
        descend(root, "", &mut out);
    }
    out
}

/// A site under the shortest path that still says where to look.
///
/// The crate is dropped, since a model's components mostly come from one or two, and
/// so is a `base` module, which holds a folder's one implementation — so
/// `my_model::rope::base::apply` reads as `rope::apply`, which is what a
/// reader greps for.
pub(super) fn site(site: &str) -> String {
    site.split("::")
        .skip(1)
        .filter(|segment| *segment != "base")
        .collect::<Vec<_>>()
        .join("::")
}

/// The component tree as the table prints it: each component's line, its children's beneath
/// it, then the kernels it launched itself.
struct ComponentLines<'r, 'a> {
    report: &'r Report<'a>,
    /// The device time every share is of.
    whole: Duration,
    /// The width of the component column.
    width: usize,
}

impl ComponentLines<'_, '_> {
    /// A component's line, and its children's beneath it. `ancestors` are the
    /// components it sits under, outermost first — its depth, and where a kernel
    /// that fused its work would be.
    fn write(
        &self,
        out: &mut String,
        component: &Component,
        ancestors: &[&Component],
    ) -> fmt::Result {
        let depth = ancestors.len();
        let fused_elsewhere = self.report.is_fused_elsewhere(component);
        let indent = "  ".repeat(depth);
        let label = fit(heading(component), self.width.saturating_sub(depth * 2));
        let scored = self.report.score_of(component);
        let (bound, achieved, floor, gain) = match (scored, self.report.floor_of(component)) {
            (Some(score), _) => (
                score.bound.to_string(),
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
                    false => format!("{} {:>3.0}% ", bar(score.achieved), score.achieved * 100.0),
                },
                ms(score.floor),
                ms(score.gain(self.report.apportioned(component))),
            ),
            // Priced but not scored — it measured no time of its own. The floor is
            // still the declaration's, and its parent's floor counts it, so the
            // component says what that is rather than leaving the parent's figure to be
            // reconciled against blanks.
            (None, Some(floor)) => (
                floor.bound.to_string(),
                String::new(),
                ms(floor.time),
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
        let (time, of_work, host) = match fused_elsewhere {
            true => ("fused ↓".to_string(), String::new(), String::new()),
            false => (
                ms(self.report.apportioned(component)),
                share(component.total_time(), self.whole),
                self.report.host(component).map(ms).unwrap_or_default(),
            ),
        };
        writeln!(
            out,
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
        let under: Vec<&Component> = ancestors.iter().copied().chain([component]).collect();
        for child in &component.children {
            self.write(out, child, &under)?;
        }
        // The kernels this component launched itself, after the components it
        // entered. A kernel here that sits above a component's own component is one that
        // fused across it — the fuser's behaviour, read off the indentation rather
        // than asked about.
        for (name, kernel) in named(&component.kernels) {
            writeln!(
                out,
                "  {}· {:<width$} {:>8} {:>7}",
                "  ".repeat(depth + 1),
                fit(
                    launched(&name, kernel),
                    self.width.saturating_sub(depth * 2 + KERNEL_INDENT)
                ),
                ms(self.report.apportion(kernel.total)),
                share(kernel.total, self.whole),
                width = self.width.saturating_sub(depth * 2 + KERNEL_INDENT),
            )?;
        }
        Ok(())
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
