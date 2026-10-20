//! The table a collected pass prints.
//!
//! One page: the device and what it can do, how the pass ran, the component
//! tree with each row scored against the wall it is bound by, the ladder, and
//! a worklist ranked by what is actually recoverable. It is plain text,
//! greppable, and says what it does not know rather than leaving a column
//! blank.

use std::fmt::{self, Display, Write as _};
use std::time::Duration;

use burn_instrument::{
    Bounds, CollectedPass, Component, ComponentPath, Kernel, PeakFinding, Peaks,
};

use crate::roofline::{Floor, Ladder, Score, score};

/// A collected pass, scored against its device's peaks, and the table that
/// renders it ([`Display`]).
///
/// A front door that knows what the pass was (a model, a batch, a phase)
/// prints that above this table; the table says only what the file does.
pub struct Report<'a> {
    pass: &'a CollectedPass,
    /// The pass's peaks, with any impossible reading dropped.
    peaks: Peaks,
    /// What the header must say about the probe set before its numbers are
    /// read, already ordered — see [`Peaks::findings`].
    findings: Vec<PeakFinding>,
    /// The rows whose work all ran in a fused kernel recorded elsewhere. See
    /// [`Breakdown::fused_elsewhere`](burn_instrument::Breakdown::fused_elsewhere).
    fused_elsewhere: Vec<&'a Component>,
}

impl<'a> Report<'a> {
    /// Score `pass` against the peaks its device measured, or against none
    /// where it was not probed.
    ///
    /// An impossible peak is dropped here rather than divided by — see
    /// [`Peaks::trusted`]. The findings are read off the raw peaks first, so
    /// the header can say what was rejected: a probe that half-ran is a fact
    /// about the run, and silently falling back to a coarser wall would leave a
    /// reader wondering why nothing is held against `flops`, or why every row
    /// moved at once.
    pub fn new(pass: &'a CollectedPass) -> Self {
        let peaks = pass.peaks.unwrap_or_default();
        Self {
            pass,
            findings: peaks.findings(),
            peaks: peaks.trusted(),
            fused_elsewhere: pass.breakdown.fused_elsewhere(),
        }
    }

    /// What the pass took: the clean pass's wall time where it was measured,
    /// and the collected pass's device time otherwise.
    pub fn wall(&self) -> Duration {
        self.pass
            .clean_wall
            .unwrap_or_else(|| self.pass.breakdown.total_time())
    }

    /// Of the clean pass, the time the device spent waiting on the host — the
    /// launches, the op recording, everything between kernels — where the
    /// collected pass can say.
    ///
    /// It is the clean pass's wall less the device time its kernels took,
    /// which holds only when that device time is a pipelined pass's: `None`
    /// where no clean pass was measured, where any launch was timed alone,
    /// where nothing was measured on the device at all, and where the kernels
    /// add up to more than the clean pass took — two clocks disagreeing, not a
    /// negative host.
    pub fn framework(&self) -> Option<Duration> {
        let wall = self.pass.clean_wall?;
        let device = self.pass.breakdown.total_time();
        let timed_alone = self.pass.breakdown.timed_alone;
        (!timed_alone && !device.is_zero() && device <= wall).then(|| wall - device)
    }

    /// The clean pass the rows are apportioned onto, where they need to be
    /// and one was measured. See [`apportioned`](Self::apportioned).
    fn apportioned_onto(&self) -> Option<Duration> {
        self.pass
            .clean_wall
            .filter(|_| self.pass.breakdown.timed_alone)
    }

    /// Whether all of `node`'s work ran in a fused kernel recorded elsewhere —
    /// on a row above it, or outside every component.
    ///
    /// Such a row ran nothing, so it has no time to score and its declared
    /// floor is set aside: the work it declared is in the kernel that carried
    /// it out, which stands at what it took on the row that holds it. Counting
    /// the floor as well would price that work twice.
    fn is_fused_elsewhere(&self, node: &Component) -> bool {
        // By identity rather than by path: every caller holds a node borrowed
        // from this report, and none of them its path. A tree that kept its
        // collector's row indices would make both unnecessary.
        self.fused_elsewhere
            .iter()
            .any(|row| std::ptr::eq(*row, node))
    }

    /// The ladder for this run: what it took, what faster kernels reach, and
    /// what perfect fusion inside each component reaches past that.
    pub fn ladder(&self) -> Ladder {
        Ladder {
            measured: self.wall(),
            host: self.framework().unwrap_or_default(),
            kernels: self
                .pass
                .breakdown
                .tree
                .iter()
                .map(|node| self.deepest_floor(node))
                .sum(),
            fused: self
                .pass
                .breakdown
                .tree
                .iter()
                .map(|node| self.outermost_floor(node))
                .sum(),
        }
    }

    /// The floor of the outermost priced components under `node` — each at
    /// its own floor, which assumes perfect fusion inside it.
    ///
    /// Descends through a component that declares none, so an unpriced root
    /// — the model itself — does not hide the priced components under it and
    /// leave the fusion rung equal to the kernel one. What such a component
    /// launched itself stands at what the device spent on it, as in
    /// [`deepest_floor`](Self::deepest_floor).
    fn outermost_floor(&self, node: &Component) -> Duration {
        if let Some(floor) = self.own_floor(node) {
            return floor;
        }
        match self.composite(node) {
            true => {
                node.children
                    .iter()
                    .map(|child| self.outermost_floor(child))
                    .sum::<Duration>()
                    + self.apportion(self.unpriced_kernel_time(node))
            }
            false => self.apportioned(node),
        }
    }

    /// This component's own floor, when it declared bounds and the device was
    /// probed — whatever it measured. See [`Floor`].
    fn own_floor(&self, node: &Component) -> Option<Duration> {
        self.floor_of(node).map(|floor| floor.time)
    }

    /// This component's floor and the wall that sets it.
    fn floor_of(&self, node: &Component) -> Option<Floor> {
        Floor::new(self.priced(node)?, &self.peaks)
    }

    /// What `node` is held to, unless its work all ran elsewhere. See
    /// [`is_fused_elsewhere`](Self::is_fused_elsewhere).
    fn priced(&self, node: &Component) -> Option<Bounds> {
        match self.is_fused_elsewhere(node) {
            true => None,
            false => Self::held(node),
        }
    }

    /// What `node` is held to: its declared bounds plus the traffic observed
    /// beyond them, with at least the arithmetic of its parts.
    ///
    /// Fusion removes traffic, never arithmetic, so a composite that prices
    /// only the traffic it must move — the whole model does, rather than
    /// restate every operator's formula — still has every multiply-add beneath
    /// it to do. Everything a row is scored against goes through here, so its
    /// floor and its percentage cannot be of two different bounds.
    fn held(node: &Component) -> Option<Bounds> {
        let own = node.bounds? + node.observed.unwrap_or_default();
        let flops = node
            .parts_bounds()
            .map_or(own.flops, |parts| own.flops.max(parts.flops));
        Some(Bounds { flops, ..own })
    }

    /// The floor of the finest-grained declarations under `node` — every
    /// component at its own roof, taken as deep as the tree prices one.
    ///
    /// Falls back to the node's own floor where nothing below declares one, and
    /// to its measured time where nothing does at all: a component nobody
    /// priced cannot be claimed back, so it counts at what it cost.
    fn deepest_floor(&self, node: &Component) -> Duration {
        if self.composite(node) {
            let children: Duration = node
                .children
                .iter()
                .map(|child| self.deepest_floor(child))
                .sum();
            // Whatever the children did not account for is this component's
            // own work, and no declaration prices it — so it stands, at what
            // the device spent on it.
            return children + self.apportion(self.unpriced_kernel_time(node));
        }
        self.own_floor(node).unwrap_or(self.apportioned(node))
    }

    /// Device time in `node`'s own kernels that no declaration beneath it
    /// prices — what a rung counts at measured time because nothing else
    /// counts it.
    ///
    /// A kernel that fused components' work is recorded on their closest
    /// common parent, and where every component it names is still priced —
    /// it ran kernels of its own besides — the rung already counts that work
    /// at their floors: counting the kernel's time as well would price the
    /// same work twice and lift the rung above the pass. Where any of them is
    /// not, the kernel stands at what it took, like the rest of this
    /// component's own work. A component whose work all ran in it is not:
    /// its floor is set aside, so the kernel is what counts its work. And a
    /// kernel that also carried this component's own work stands at what it
    /// took whatever it names, since no floor below prices that part of it.
    fn unpriced_kernel_time(&self, node: &Component) -> Duration {
        let priced: Vec<ComponentPath> = node
            .descendants()
            .into_iter()
            .filter(|(_, below)| self.priced(below).is_some())
            .map(|(path, _)| path)
            .collect();
        node.kernels
            .iter()
            .filter(|kernel| {
                kernel.fuses.is_empty()
                    || kernel.fuses_own
                    || !kernel.fuses.iter().all(|path| priced.contains(path))
            })
            .map(|kernel| kernel.total)
            .sum()
    }

    /// Whether anything under `node` is priced — whether, that is, its own
    /// floor has parts to be read against.
    ///
    /// The question [`deepest_floor`](Self::deepest_floor) descends on and the
    /// worklist ranks on, asked once so the two cannot disagree about which
    /// rows are composites.
    fn composite(&self, node: &Component) -> bool {
        node.children
            .iter()
            .any(|child| child.bounds.is_some() || !child.children.is_empty())
    }

    /// This component's milliseconds: its kernels' own device time where they
    /// were measured in place, and its share of the clean pass where any
    /// launch was timed alone.
    ///
    /// In place, nothing waited on a kernel while the pass ran, so each
    /// measurement is the kernel as it runs in a full queue and needs no
    /// correcting. The rows then sum to the device's time, and the clean pass's
    /// remainder is the host's — the `framework` line.
    ///
    /// **Timed alone, the collected pass has no absolute clock.** Its per-launch device
    /// times are each measured with the kernel running *alone*, since timing a
    /// launch blocks on it, so two collections of the same work can read twice
    /// apart against a clean pass that barely moved.
    /// What the collected pass measures reliably is the **proportions**,
    /// because every kernel in it is measured the same way.
    ///
    /// So the proportions are apportioned onto the one unperturbed number
    /// there is: the median wall time of the clean pass. The result is
    /// absolute, sums to that wall, and is the model's time rather than the
    /// instrumentation's.
    ///
    /// What it assumes is that serialization inflates every kernel alike. It
    /// does not, quite — a fixed cost per launch over-weights a component that
    /// runs many small kernels against one that runs a few large ones — so a
    /// row is good to a few percent and not to the last digit.
    ///
    /// Timed alone with no clean pass measured, there is nothing to apportion
    /// onto: the rows are the serialized kernels' own times, and only their
    /// proportions mean anything, which the header says.
    fn apportioned(&self, node: &Component) -> Duration {
        self.apportion(node.total_time())
    }

    /// One measured duration in a row's milliseconds. See
    /// [`apportioned`](Self::apportioned).
    fn apportion(&self, measured: Duration) -> Duration {
        let Some(wall) = self.apportioned_onto() else {
            return measured;
        };
        let total = self.pass.breakdown.total_time();
        if total.is_zero() {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(wall.as_secs_f64() * measured.as_secs_f64() / total.as_secs_f64())
    }

    /// This component's share of the clean pass's host time: the `framework`
    /// line, split by how long the device waited before each component's
    /// kernels in the collected pass. `None` where the pass has no host/device
    /// split — see [`framework`](Self::framework).
    ///
    /// Split rather than read off directly, because the collected pass's host
    /// pays a round trip per launch to time it, so its idle stretches are
    /// longer than the clean pass's by a cost every launch shares. As shares
    /// they still say which components the device waited on; as absolutes they
    /// would describe the instrumentation. Where that shared cost is most of
    /// every stretch the split falls back to counting launches, which is the
    /// honest reading of a device waiting on the same work each time.
    fn host(&self, node: &Component) -> Option<Duration> {
        self.host_share(node.total_wait())
    }

    /// The clean pass's host time charged to `waited` of the collected pass's.
    fn host_share(&self, waited: Duration) -> Option<Duration> {
        let framework = self.framework()?;
        let total = self.pass.breakdown.total_wait();
        (!total.is_zero()).then(|| {
            Duration::from_secs_f64(
                framework.as_secs_f64() * waited.as_secs_f64() / total.as_secs_f64(),
            )
        })
    }

    /// Score one node against its binding wall, over its apportioned share of
    /// the clean pass.
    ///
    /// `None` where nothing under the component was timed: a percentage over
    /// zero measured time is not a percentage.
    fn score_of(&self, node: &Component) -> Option<Score> {
        let measured = self.apportioned(node);
        (!measured.is_zero())
            .then(|| score(self.priced(node)?, measured, &self.peaks))
            .flatten()
    }
}

/// A ten-cell bar. The two of them on a row are the whole point of the column:
/// one number cannot separate a `bytes` floor from a `flops` one from neither.
fn bar(fraction: f64) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * 10.0).round() as usize;
    format!("{}{}", "▓".repeat(filled), "░".repeat(10 - filled))
}

/// Milliseconds, at the precision a reader can act on.
fn ms(duration: Duration) -> String {
    format!("{:.1}", duration.as_secs_f64() * 1e3)
}

/// A share of the whole, or blank when there is no whole to take one of.
fn share(part: Duration, whole: Duration) -> String {
    if whole.is_zero() {
        return String::new();
    }
    format!("{:.1}%", part.as_secs_f64() / whole.as_secs_f64() * 100.0)
}

/// What the header says about one finding, a line per row of the note.
///
/// Named, not swallowed: every ratio in a run shares the probe, so one bad
/// reading invalidates all of them at once, and a reader who sees only the copy
/// ceiling has no way to tell a device that reported no directional peaks from
/// one whose probe contradicted itself — or to know why no row is held against
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
            "write-allocate cache makes DRAM see more — so every row below".to_string(),
            "is scored against the copy ceiling instead.".to_string(),
        ],
        // A caveat, not a fallback: the rows below were still divided by the two
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
            "readings do not say which, so the rows below are still scored".to_string(),
            "on the directions. Check them against an independent stream".to_string(),
            "before believing the absolutes.".to_string(),
        ],
        PeakFinding::ImpossibleCompute { flops_per_s } => vec![
            format!(
                "REJECTED the compute peak: the probe reported {:.4} TFLOP/s,",
                flops_per_s / 1e12
            ),
            "below this device's own memory peak. Nothing computes fewer".to_string(),
            "FLOP/s than it moves bytes/s, so the probe failed — every row".to_string(),
            "below is scored on memory alone.".to_string(),
        ],
    }
}

/// A row's heading: what it is called, how many times the pass entered it,
/// which executions served those entries, and whether its floor includes
/// observed traffic.
///
/// The variant sits beside the count rather than on a line of its own because a
/// row that routed one way is one fact about that row, and a second line per
/// row would double a 40-row table to say it.
fn heading(node: &Component) -> String {
    let mut heading = match node.entries {
        1 => node.label.clone(),
        many => format!("{} ×{many}", node.label),
    };
    if let Some(variants) = variants(node) {
        heading.push_str(" · ");
        heading.push_str(&variants);
    }
    if node.observed.is_some() {
        heading.push_str(" · floor read from the pass");
    }
    heading
}

/// Which executions served a row's entries, or `None` where it declares no
/// variant — which is most rows.
///
/// A single execution that served every entry prints as its bare name: its
/// count would only repeat the row's own. Several print with theirs, because
/// *how the entries split* is the whole reason to look — `tile ×8, tensor ops
/// ×1` says a routing decision went one way eight times and the other once, and
/// that is a finding.
fn variants(node: &Component) -> Option<String> {
    match node.variants.as_slice() {
        [] => None,
        [only] if only.entries == node.entries => Some(only.name.clone()),
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
/// nested row that routed three ways — the case the variant column exists for —
/// and a heading past it is elided by [`fit`] rather than overflowing, so the
/// columns hold whatever a row is called.
fn column_width(tree: &[Component]) -> usize {
    const NARROWEST: usize = 28;
    const WIDEST: usize = 64;

    tree.iter()
        .flat_map(|node| node.walk())
        .flat_map(|(depth, node)| {
            // The kernel lines share this column with the component headings,
            // one indent deeper, and a kernel that names the components it
            // fused is the longest thing in the table. Measured here because
            // the alternative is a `· fuses ↑ a + b` elided down to `· fuse…` —
            // the annotation cut off is exactly the one nothing else states.
            let kernels: Vec<usize> = named(&node.kernels)
                .into_iter()
                .map(|(name, kernel)| {
                    depth * 2 + KERNEL_INDENT + launched(&name, kernel).chars().count()
                })
                .collect();
            std::iter::once(depth * 2 + heading(node).chars().count()).chain(kernels)
        })
        .max()
        .unwrap_or(NARROWEST)
        .clamp(NARROWEST, WIDEST)
}

/// `heading` in at most `width` columns.
///
/// Only a heading past [`column_width`]'s cap is cut, and it is cut visibly: a
/// row that quietly overflowed would push its own figures out of the columns
/// every other row is aligned to, so the one line a reader cannot parse would
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

/// Rows fed from more than one call site, each with the path that names it.
///
/// A merge is not a fault — one row entered 36 times is what makes a stack
/// readable, and two sites under one label is the same mechanism. It is worth a
/// note only because it is also how one row comes to hold two different jobs:
/// two entry points that both write a cache under one label would sum two
/// different writes under one heading.
///
/// A site is a function, so this reports two *entry points*, never two roles of
/// one entry point — that is what a caller-supplied label is for.
fn merged(tree: &[Component]) -> Vec<(String, &Component)> {
    fn descend<'a>(node: &'a Component, under: &str, out: &mut Vec<(String, &'a Component)>) {
        if node.sites.len() > 1 {
            out.push((under.to_string(), node));
        }
        for child in &node.children {
            descend(child, &node.label, out);
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
/// The crate is dropped, since a model's rows mostly come from one or two, and
/// so is a `base` module, which holds a folder's one implementation — so
/// `my_model::rope::base::apply` reads as `rope::apply`, which is what a
/// reader greps for.
fn site(site: &str) -> String {
    site.split("::")
        .skip(1)
        .filter(|segment| *segment != "base")
        .collect::<Vec<_>>()
        .join("::")
}

/// A component's line, and its children's beneath it. `ancestors` are the
/// components it sits under, outermost first — its depth, and where a kernel
/// that fused its work would be.
fn rows(
    out: &mut String,
    rendered: &Report<'_>,
    node: &Component,
    ancestors: &[&Component],
    whole: Duration,
    width: usize,
) -> fmt::Result {
    let depth = ancestors.len();
    let fused_elsewhere = rendered.is_fused_elsewhere(node);
    let indent = "  ".repeat(depth);
    let label = fit(heading(node), width.saturating_sub(depth * 2));
    let scored = rendered.score_of(node);
    let (bound, achieved, floor, gain) = match (scored, rendered.floor_of(node)) {
        (Some(score), _) => (
            score.bound.label().to_string(),
            // A row above 100% is marked and keeps its figure. The peaks are
            // measured rather than looked up, so 105% is a row at the roof read
            // through a probe that came in a little low, and 568% is a declared
            // bound that is wrong — different findings, and one number
            // separates them where a shared ">100%" gave both the same answer.
            // Marked rather than suppressed: this is usually the most
            // interesting row on the page. The systemically broken denominator
            // is caught at the peak, not here.
            //
            // Only the bar is clamped, because a bar cannot say more than full.
            match score.achieved > 1.0 {
                true => format!("{} {:>3.0}%*", bar(1.0), score.achieved * 100.0),
                false => format!("{} {:>3.0}% ", bar(score.achieved), score.achieved * 100.0),
            },
            ms(score.floor),
            ms(score.gain(rendered.apportioned(node))),
        ),
        // Priced but not scored — it measured no time of its own. The floor is
        // still the declaration's, and its parent's floor counts it, so the
        // row says what that is rather than leaving the parent's figure to be
        // reconciled against blanks.
        (None, Some(floor)) => (
            floor.bound.label().to_string(),
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
    // row above, after that row's components, or outside every component.
    let (time, of_work, host) = match fused_elsewhere {
        true => ("fused ↓".to_string(), String::new(), String::new()),
        false => (
            ms(rendered.apportioned(node)),
            share(node.total_time(), whole),
            rendered.host(node).map(ms).unwrap_or_default(),
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
        width = width.saturating_sub(depth * 2),
    )?;
    let under: Vec<&Component> = ancestors.iter().copied().chain([node]).collect();
    for child in &node.children {
        rows(out, rendered, child, &under, whole, width)?;
    }
    // The kernels this component launched itself, after the components it
    // entered. A kernel here that sits above a component's own row is one that
    // fused across it — the fuser's behaviour, read off the indentation rather
    // than asked about.
    for (name, kernel) in named(&node.kernels) {
        writeln!(
            out,
            "  {}· {:<width$} {:>8} {:>7}",
            "  ".repeat(depth + 1),
            fit(
                launched(&name, kernel),
                width.saturating_sub(depth * 2 + KERNEL_INDENT)
            ),
            ms(rendered.apportion(kernel.total)),
            share(kernel.total, whole),
            width = width.saturating_sub(depth * 2 + KERNEL_INDENT),
        )?;
    }
    Ok(())
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

/// One component's line on the worklist: what it could give back, and which
/// lever gives it.
struct Recoverable {
    label: String,
    gain: Duration,
    /// The lever, named — the two are different work for different people.
    lever: &'static str,
    /// Where it stands against its binding wall.
    against: String,
}

/// Every priced component by what it could still give back, **counting each
/// millisecond once**.
///
/// A composite's own gain contains its children's, so ranking it by that would
/// list the same time again under every ancestor — and a reader adding the list
/// up would find more on the table than the pass took. So each row is ranked by
/// the lever that is only available at *its* level: a leaf by what a faster
/// kernel would buy, a composite by the round-trip between its parts that no
/// faster kernel can reach. Those are the ladder's two rungs, split per
/// component, and they are disjoint by construction.
fn worklist(rendered: &Report<'_>) -> Vec<Recoverable> {
    fn visit(rendered: &Report<'_>, node: &Component, out: &mut Vec<Recoverable>) {
        // The third lever: what the device waited on the host for before this
        // component's own kernels. Its own, not its subtree's, so a
        // millisecond is listed once however deep the component sits.
        if let Some(host) = rendered
            .host_share(node.own_wait)
            .filter(|host| !host.is_zero())
        {
            let launches: usize = node.kernels.iter().map(|kernel| kernel.launches).sum();
            // A wait before a single launch is the host's work ahead of it,
            // which no launch can be taken away from.
            let (lever, against) = match launches {
                1 => ("less host work", "before its one launch".to_string()),
                _ => ("fewer launches", format!("{launches} launches")),
            };
            out.push(Recoverable {
                label: node.label.clone(),
                gain: host,
                lever,
                against,
            });
        }
        if let Some(score) = rendered.score_of(node) {
            // A component at the roof has nothing to give back, so it sorts to
            // the bottom on its own — filtering it out would hide the row that
            // most deserves to be read.
            let (gain, lever) = match rendered.composite(node) {
                true => (
                    rendered.deepest_floor(node).saturating_sub(score.floor),
                    "fusing inside",
                ),
                false => (score.gain(rendered.apportioned(node)), "a faster kernel"),
            };
            out.push(Recoverable {
                label: node.label.clone(),
                gain,
                lever,
                against: format!(
                    "{:>3.0}% of {} peak",
                    score.achieved * 100.0,
                    score.bound.label()
                ),
            });
        }
        for child in &node.children {
            visit(rendered, child, out);
        }
    }
    let mut out = Vec::new();
    for node in &rendered.pass.breakdown.tree {
        visit(rendered, node, &mut out);
    }
    out.sort_by_key(|row| std::cmp::Reverse(row.gain));
    out
}

impl Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pass = self.pass;

        let peak = |value: Option<f64>, scale: f64, unit: &str| match value {
            Some(value) => format!("{:.1} {unit}", value / scale),
            None => "unprobed".to_string(),
        };
        writeln!(
            f,
            "device   {} · {} · {} at {:?}",
            pass.device,
            peak(self.peaks.copy_bytes_per_s, 1e9, "GB/s"),
            peak(self.peaks.flops_per_s, 1e12, "TFLOP/s"),
            self.peaks.compute_dtype,
        )?;
        // The two ceilings a bytes row is actually divided by, beside the copy
        // figure above. A percentage whose denominator is nowhere on the page
        // asks to be trusted about the very thing a reader doubts — and the
        // copy ceiling alone would be worse than none, since it is not what any
        // row was divided by. Absent where the device reported no directions,
        // in which case the copy ceiling above *is* the denominator.
        if let (Some(read), Some(write)) =
            (self.peaks.read_bytes_per_s, self.peaks.write_bytes_per_s)
        {
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
                "         not probed: no row is scored, and the ladder is the measured time"
            )?;
        }
        for finding in &self.findings {
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
                    "{} ms run uncollected, the time the rows below break down",
                    ms(wall)
                ),
                None => "no uncollected run measured: no host/device split".to_string(),
            }
        )?;
        let graph = match pass.breakdown.unfused {
            true => "unfused — one kernel per operation: not the kernels a fused build runs",
            false => "fused — the kernels a fused build runs",
        };
        let timings: &[&str] = match (pass.breakdown.timed_alone, pass.clean_wall) {
            (false, _) => &[
                "in place — read back after the pass, the kernels ran back to back: a row is",
                "its kernels' device time, and `framework` is the device waiting on the host",
            ],
            (true, Some(_)) => &[
                "alone — each read as it launched, so the kernels ran one at a time: a row is",
                "its share of the uncollected run, good to a few percent",
            ],
            (true, None) => &[
                "alone — each read as it launched, so the kernels ran one at a time: a row is",
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
        // one row out of its column.
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
            match self.apportioned_onto().is_some() {
                true => "of wall",
                false => "on device",
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
        for node in &pass.breakdown.tree {
            rows(&mut body, self, node, &[], device, width)?;
        }
        f.write_str(&body)?;
        // Kernels nobody's component claimed, and the fused work the last
        // components left for the sync that ends the pass. Work the device did
        // is still work, and a row saying so beats a silent omission.
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
                    ms(kernel.total),
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
                "  · fuses ↑ a + b   one kernel did the named rows' work; it sits on their closest parent, or outside every component"
            )?;
            writeln!(
                f,
                "  fused ↓           a row whose work all ran in such a kernel below it; its floor is set aside"
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
                match pass.breakdown.timed_alone {
                    true => "the pass run uncollected, and what the rows above sum to",
                    false => "the pass run uncollected",
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
            match pass.breakdown.timed_alone {
                true => "each timed alone",
                false => "timed in place — what the rows above sum to",
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
        // Not in any row above, and the total every row is a share of. Said
        // out loud rather than left to shrink the denominator quietly.
        // Lost rather than misplaced, and the reason a table can read a matmul
        // at 0.0 ms beside a counter add at 78: said here, where the total it
        // shortens is printed, and keyed on below.
        if !untimed.is_empty() {
            let launches: usize = untimed.iter().map(|kernel| kernel.launches).sum();
            writeln!(
                f,
                "  {:<width$} {:>8}      {} kernels, {launches} launches, measured no time: every row and share above is short by them",
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
        match self.framework() {
            Some(framework) => writeln!(
                f,
                "  {:<width$} {:>8} ms   {} of the pass the device spent waiting on the host",
                "framework",
                ms(framework),
                share(framework, self.wall()),
            )?,
            None if pass.clean_wall.is_none() => writeln!(
                f,
                "  {:<width$} {:>8}      not separable: no uncollected run was measured",
                "framework", "—"
            )?,
            None if pass.breakdown.timed_alone => writeln!(
                f,
                "  {:<width$} {:>8}      not separable: a launch timed alone serializes the\n  \
                 {:<width$}               pass, so the measured total is not a pipelined pass's",
                "framework", "—", ""
            )?,
            None if device.is_zero() => writeln!(
                f,
                "  {:<width$} {:>8}      not separable: nothing was timed on the device",
                "framework", "—"
            )?,
            None => writeln!(
                f,
                "  {:<width$} {:>8}      not separable: the kernels measured more than the\n  \
                 {:<width$}               pass took, so the two clocks disagree",
                "framework", "—", ""
            )?,
        }

        let ladder = self.ladder();
        let at_the_roof = self
            .pass
            .breakdown
            .tree
            .iter()
            .flat_map(|node| node.walk())
            .any(|(_, node)| {
                self.score_of(node)
                    .is_some_and(|score| score.achieved > 1.0)
            });
        if at_the_roof {
            // One `writeln!` per line: a `\n` inside a format string is
            // indented by rustfmt into the output, which puts the second line
            // of every note halfway across the page.
            writeln!(f)?;
            // Where priced work timed nothing, kernel time was lost, and a row
            // past its peak is far likelier a time that came in short than a
            // kernel past the hardware — "nothing to win" would wave off the
            // one row that shows it.
            let lines: &[&str] = match untimed.is_empty() {
                true => &[
                    "* past the measured peak, which charges every byte to DRAM. How far",
                    "  past is the reading. A few percent over is a row at the roof seen",
                    "  through a probe that came in low, since the peaks are measured and",
                    "  not looked up. Well over means the bytes were not all paid for — a",
                    "  small, freshly written working set served from cache, which no bound",
                    "  stated in bytes can say — or its kernels are too brief to time at this",
                    "  resolution, or its bounds are wrong. Nothing to win here either way.",
                ],
                false => &[
                    "* past the measured peak, in a pass that lost kernel time (`untimed`",
                    "  above): a row past its peak here is a time that came in short, not",
                    "  a row at the roof. Read none of these percentages.",
                ],
            };
            for line in lines {
                writeln!(f, "  {line}")?;
            }
            // The one reading a fused pass adds: a row part of whose work fused
            // with a neighbour's took less than its floor prices, since that
            // part runs in a kernel recorded elsewhere.
            if !pass.breakdown.unfused {
                for line in [
                    "  In a fused pass it can also be a row part of whose work fused with a",
                    "  neighbour's: that part ran in a kernel listed on a row above it, or",
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

        // Where one row was fed from two places. After the table, because it
        // is a fact about a row a reader has already seen, and never instead of
        // one: the tree is unchanged and the merge stands.
        for (under, node) in merged(&pass.breakdown.tree) {
            let sites: Vec<String> = node.sites.iter().map(|it| site(it)).collect();
            let under = match under.is_empty() {
                true => String::new(),
                false => format!(" under `{under}`"),
            };
            writeln!(
                f,
                "\n  note: `{}`{under} was entered from {} sites ({})",
                node.label,
                sites.len(),
                sites.join(", ")
            )?;
        }

        let worklist = worklist(self);
        if !worklist.is_empty() {
            writeln!(f, "\n  worth reading first    recoverable   by")?;
            // Its gains are differences against those timings and its waits
            // are gaps between them, so it carries the same caveat.
            if !untimed.is_empty() {
                writeln!(
                    f,
                    "    (from a pass that lost kernel time — see `untimed`; re-measure first)"
                )?;
            }
            for row in worklist.iter().take(5) {
                writeln!(
                    f,
                    "    {:<22} {:>6} ms   {:<16} {}",
                    row.label,
                    ms(row.gain),
                    row.lever,
                    row.against
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roofline::Bound;
    use burn_instrument::{Breakdown, Unplaced, Variant};

    fn peaks() -> Peaks {
        Peaks {
            copy_bytes_per_s: Some(198.9e9),
            flops_per_s: Some(49.3e12),
            ..Peaks::default()
        }
    }

    /// A pass with a small tree, **built rather than measured**.
    ///
    /// The table is a function of a [`CollectedPass`], so a rendering test
    /// needs one, not a run that produces one: collecting a pass here would
    /// leave every number at whatever this machine happened to do. What the
    /// collector produces from a real pass is `burn-instrument`'s to test;
    /// these tests own the rendering, and their input is stated.
    fn pass() -> CollectedPass {
        CollectedPass {
            device: "cpu".to_string(),
            peaks: Some(peaks()),
            clean_wall: Some(Duration::from_micros(2_000)),
            breakdown: Breakdown {
                tree: tree(),
                untagged: vec![kernel("Reshape", 1, 20)],
                end_of_pass: Vec::new(),
                outside_wait: Duration::ZERO,
                host_timed: false,
                // Sized against a 2.0 ms wall that its rows are apportioned
                // onto; the in-place reading has tests of its own.
                timed_alone: true,
                unplaced: Unplaced::default(),
                unfused: false,
            },
        }
    }

    /// The rows of [`pass`]: a two-layer stack between an embedding and a
    /// head, priced the way the components price themselves — a composite below
    /// the sum of its parts, and the leaves carrying the kernels.
    ///
    /// The floors are sized against [`peaks`]: bytes ÷ 198.9 GB/s and flops ÷
    /// 49.3 TFLOP/s, so a leaf here is a tenth to a quarter of a millisecond
    /// and the rungs land inside the 2.0 ms the pass took. Numbers a thousand
    /// times smaller would render as `0.0` and make the ladder read as
    /// incoherent, which is a fact about the fixture and not about the code.
    /// `attention engine` is the one row held against `flops`, so a table has
    /// one of each wall.
    fn tree() -> Vec<Component> {
        vec![
            Component {
                label: "embedding".into(),
                bounds: Some(Bounds {
                    read: 15_000_000,
                    write: 5_000_000,
                    flops: 0,
                }),
                kernels: vec![kernel("Gather", 1, 50)],
                ..row()
            },
            Component {
                label: "transformer block".into(),
                observed: None,
                entries: 2,
                bounds: Some(Bounds {
                    read: 82_500_000,
                    write: 27_500_000,
                    flops: 3_000_000_000,
                }),
                children: vec![
                    Component {
                        label: "attn norm".into(),
                        observed: None,
                        entries: 2,
                        bounds: Some(Bounds {
                            read: 15_000_000,
                            write: 5_000_000,
                            flops: 0,
                        }),
                        kernels: vec![kernel("RmsNormKernel", 2, 70)],
                        ..row()
                    },
                    Component {
                        label: "attention".into(),
                        observed: None,
                        entries: 2,
                        bounds: Some(Bounds {
                            read: 45_000_000,
                            write: 15_000_000,
                            flops: 900_000_000,
                        }),
                        children: vec![
                            Component {
                                label: "query".into(),
                                observed: None,
                                entries: 2,
                                bounds: Some(Bounds {
                                    read: 30_000_000,
                                    write: 10_000_000,
                                    flops: 300_000_000,
                                }),
                                kernels: vec![kernel("Matmul", 2, 160)],
                                ..row()
                            },
                            Component {
                                label: "attention engine".into(),
                                observed: None,
                                entries: 2,
                                bounds: Some(Bounds {
                                    read: 30_000_000,
                                    write: 10_000_000,
                                    flops: 12_000_000_000,
                                }),
                                kernels: vec![kernel("FlashAttention", 2, 220)],
                                ..row()
                            },
                        ],
                        ..row()
                    },
                    Component {
                        label: "feed forward".into(),
                        observed: None,
                        entries: 2,
                        bounds: Some(Bounds {
                            read: 45_000_000,
                            write: 15_000_000,
                            flops: 2_000_000_000,
                        }),
                        children: vec![
                            Component {
                                label: "gate".into(),
                                observed: None,
                                entries: 2,
                                bounds: Some(Bounds {
                                    read: 30_000_000,
                                    write: 10_000_000,
                                    flops: 800_000_000,
                                }),
                                kernels: vec![kernel("Matmul", 2, 240)],
                                ..row()
                            },
                            Component {
                                label: "down".into(),
                                observed: None,
                                entries: 2,
                                bounds: Some(Bounds {
                                    read: 30_000_000,
                                    write: 10_000_000,
                                    flops: 800_000_000,
                                }),
                                kernels: vec![kernel("Matmul", 2, 240)],
                                ..row()
                            },
                        ],
                        ..row()
                    },
                ],
                ..row()
            },
            Component {
                label: "final norm".into(),
                bounds: Some(Bounds {
                    read: 7_500_000,
                    write: 2_500_000,
                    flops: 0,
                }),
                kernels: vec![kernel("RmsNormKernel", 1, 30)],
                ..row()
            },
            Component {
                label: "lm head".into(),
                bounds: Some(Bounds {
                    read: 22_500_000,
                    write: 7_500_000,
                    flops: 1_000_000_000,
                }),
                children: vec![Component {
                    label: "head projection".into(),
                    bounds: Some(Bounds {
                        read: 30_000_000,
                        write: 10_000_000,
                        flops: 1_000_000_000,
                    }),
                    kernels: vec![kernel("Matmul", 1, 260)],
                    ..row()
                }],
                ..row()
            },
        ]
    }

    /// A row is held to its declared floor plus the traffic observed beyond
    /// it, and its heading says so.
    #[test]
    fn observed_traffic_raises_the_floor_and_marks_the_row() {
        let declared = Bounds {
            read: 100,
            write: 10,
            flops: 50,
        };
        let observed = Bounds {
            read: 300,
            write: 0,
            flops: 0,
        };
        let plain = Component {
            label: "routed experts".into(),
            bounds: Some(declared),
            ..row()
        };
        let read = Component {
            observed: Some(observed),
            ..plain.clone()
        };

        assert_eq!(Report::held(&plain), Some(declared));
        assert_eq!(Report::held(&read), Some(declared + observed));
        assert_eq!(heading(&plain), "routed experts");
        assert_eq!(heading(&read), "routed experts · floor read from the pass");
    }

    /// An empty row, so a fixture states only the fields it is about.
    fn row() -> Component {
        Component {
            label: String::new(),
            variants: Vec::new(),
            sites: Vec::new(),
            bounds: None,
            observed: None,
            entries: 1,
            children: Vec::new(),
            kernels: Vec::new(),
            own_wait: Duration::ZERO,
        }
    }

    fn kernel(name: &str, launches: usize, micros: u64) -> Kernel {
        Kernel {
            name: name.to_string(),
            launches,
            total: Duration::from_micros(micros),
            fuses: Vec::new(),
            fuses_own: false,
        }
    }

    /// Without an uncollected run there is no host to split off, and a pass
    /// timed alone has nothing to apportion its rows onto: they stay the
    /// serialized kernels' own times, and the table says only their proportions
    /// mean anything rather than print a `framework` line it cannot know.
    #[test]
    fn a_pass_without_a_clean_run_splits_nothing_and_says_so() {
        let pass = CollectedPass {
            clean_wall: None,
            ..pass()
        };
        let report = Report::new(&pass);
        assert_eq!(report.framework(), None);
        assert_eq!(
            report.wall(),
            pass.breakdown.total_time(),
            "the ladder starts from what the device measured"
        );
        let embedding = &pass.breakdown.tree[0];
        assert_eq!(
            report.apportioned(embedding),
            embedding.total_time(),
            "nothing to apportion onto"
        );

        let table = report.to_string();
        assert!(table.contains("no uncollected run measured"), "{table}");
        assert!(
            table.contains("only the proportions mean anything"),
            "{table}"
        );
        assert!(
            table.contains("not separable: no uncollected run was measured"),
            "{table}"
        );
        assert!(
            !table.contains("\n  wall "),
            "no clean wall to print:\n{table}"
        );
    }

    /// A pass nobody probed is timed and not scored, and the header says why
    /// every roofline column is blank.
    #[test]
    fn an_unprobed_pass_says_why_nothing_is_scored() {
        let pass = CollectedPass {
            peaks: None,
            ..pass()
        };
        let report = Report::new(&pass);
        for (_, component) in pass.breakdown.tree.iter().flat_map(|root| root.walk()) {
            assert!(report.score_of(component).is_none(), "{}", component.label);
        }
        assert!(report.to_string().contains("not probed"));
    }

    /// Print a real table, so its shape can be read rather than only asserted
    /// on. A column that lines up in a test never proves it lines up on a page.
    #[test]
    #[ignore = "a reporting aid, not an assertion"]
    fn show() {
        println!("{}", Report::new(&pass()));
    }

    #[test]
    fn a_table_carries_every_column_a_reader_acts_on() {
        let pass = pass();
        let table = Report::new(&pass).to_string();

        for column in [
            "component",
            "ms",
            "share",
            "bound",
            "achieved",
            "floor",
            "gain",
        ] {
            assert!(table.contains(column), "missing {column}:\n{table}");
        }
        assert!(table.contains("pass"), "what the rows break down:\n{table}");
        assert!(table.contains("framework"), "the host half:\n{table}");
        assert!(
            table.contains("how much is on the table"),
            "the ladder:\n{table}"
        );
    }

    #[test]
    fn an_unfused_breakdown_says_so_on_the_table() {
        let mut pass = pass();
        pass.breakdown.unfused = true;
        let table = Report::new(&pass).to_string();
        assert!(
            table.contains("graph     unfused"),
            "a reader must not take an unfused breakdown for a fused one:\n{table}"
        );
    }

    /// A fused table says so, and a component whose work all ran in a kernel
    /// its parent carries reads `fused ↓` with no floor — not `0.0`, which
    /// would read as a component that cost nothing, and not its declared
    /// floor, which the kernel below it now counts.
    #[test]
    fn a_component_whose_work_fused_into_its_parent_says_so() {
        let mut pass = pass();
        // Every op the norm registered fused with the query projection's, so
        // the kernel that did both is on the block and none is on the norm.
        pass.breakdown.tree[1].children[0].kernels.clear();
        pass.breakdown.tree[1].kernels = vec![Kernel {
            fuses: vec![
                ComponentPath::new(["attn norm"]),
                ComponentPath::new(["attention", "query"]),
            ],
            ..kernel("ElemwiseMatmul", 2, 70)
        }];

        let table = Report::new(&pass).to_string();
        assert!(table.contains("graph     fused"), "{table}");
        let norm = table
            .lines()
            .find(|line| line.contains("attn norm"))
            .expect("the norm's row");
        assert!(norm.contains("fused ↓"), "{norm}");
        assert!(!norm.contains("bytes"), "and sets its floor aside:\n{norm}");

        // A component with no kernel that no parent's kernel names did no
        // device work — a view, a reshape — and its zero is its time. A
        // `fused ↓` there would point at a kernel that carries nothing of it.
        pass.breakdown.tree[1].kernels.clear();
        let table = Report::new(&pass).to_string();
        let norm = table
            .lines()
            .find(|line| line.contains("attn norm"))
            .expect("the norm's row");
        assert!(!norm.contains("fused ↓"), "{norm}");
    }

    /// A fused kernel counts its work once. Where every component it names
    /// ran nothing else, their floors are set aside and the kernel stands at
    /// what it took on the parent that holds it; where every one still ran
    /// kernels of its own, they stay priced and the kernel is not counted
    /// again on top of their floors.
    #[test]
    fn a_fused_kernel_on_a_parent_is_not_counted_twice() {
        let pass = pass();
        let mut fused = pass.clone();
        let norm = std::mem::take(&mut fused.breakdown.tree[1].children[0].kernels);
        let query = std::mem::take(&mut fused.breakdown.tree[1].children[1].children[0].kernels);
        let total = norm.iter().chain(&query).map(|kernel| kernel.total).sum();
        let carried = Kernel {
            fuses: vec![
                ComponentPath::new(["attn norm"]),
                ComponentPath::new(["attention", "query"]),
            ],
            total,
            ..kernel("ElemwiseMatmul", 2, 0)
        };
        fused.breakdown.tree[1].kernels = vec![carried.clone()];

        let rendered = Report::new(&fused);
        let block = &fused.breakdown.tree[1];
        assert_eq!(
            rendered.unpriced_kernel_time(block),
            total,
            "their work is the kernel's, at what it took"
        );
        assert_eq!(
            rendered.deepest_floor(&block.children[0]),
            Duration::ZERO,
            "and the norm's floor is not counted beside it"
        );

        // Both still ran a kernel of their own besides: still priced, so the
        // shared kernel is counted at their floors and not again.
        let mut partly = pass.clone();
        partly.breakdown.tree[1].kernels = vec![carried];
        let rendered = Report::new(&partly);
        assert_eq!(
            rendered.unpriced_kernel_time(&partly.breakdown.tree[1]),
            Duration::ZERO
        );
    }

    /// A residual add fused with its child's work lands on the parent naming
    /// only the child: the child ran nothing, so its floor is set aside and
    /// the kernel — the child's work and the parent's own — counts once, at
    /// what it took.
    #[test]
    fn a_parent_fused_with_its_child_counts_shared_work_once() {
        let pass = pass();
        let mut fused = pass.clone();
        let norm = std::mem::take(&mut fused.breakdown.tree[1].children[0].kernels);
        let total: Duration =
            norm.iter().map(|kernel| kernel.total).sum::<Duration>() + Duration::from_micros(5);
        fused.breakdown.tree[1].kernels = vec![Kernel {
            fuses: vec![ComponentPath::new(["attn norm"])],
            fuses_own: true,
            total,
            ..kernel("ElemwiseFuse", 1, 0)
        }];

        let rendered = Report::new(&fused);
        let block = &fused.breakdown.tree[1];
        assert!(rendered.is_fused_elsewhere(&block.children[0]));
        assert_eq!(rendered.unpriced_kernel_time(block), total);
        assert_eq!(rendered.deepest_floor(&block.children[0]), Duration::ZERO);
        let table = rendered.to_string();
        assert!(
            table.contains("ElemwiseFuse ×1 · fuses ↑ attn norm"),
            "{table}"
        );
    }

    /// A block with work from outside every component is recorded untagged,
    /// naming the row whose work it carried by its path from the root: that
    /// row reads `fused ↓` with its floor set aside, as it would under a
    /// parent's kernel.
    #[test]
    fn a_row_whose_work_ran_in_an_untagged_kernel_says_so() {
        let mut pass = pass();
        let norm = std::mem::take(&mut pass.breakdown.tree[1].children[0].kernels);
        pass.breakdown.untagged.push(Kernel {
            fuses: vec![ComponentPath::new(["transformer block", "attn norm"])],
            ..norm[0].clone()
        });

        let rendered = Report::new(&pass);
        let block = &pass.breakdown.tree[1];
        assert!(rendered.is_fused_elsewhere(&block.children[0]));
        assert_eq!(rendered.deepest_floor(&block.children[0]), Duration::ZERO);
        let table = rendered.to_string();
        let row = table
            .lines()
            .find(|line| line.contains("attn norm") && !line.contains("fuses"))
            .expect("the norm's row");
        assert!(row.contains("fused ↓"), "{row}");
        assert!(!row.contains("bytes"), "and sets its floor aside:\n{row}");
    }

    /// A kernel that carried a parent's own work stands at what it took even
    /// where the child it names is still priced — it ran a kernel of its own
    /// besides — since no floor below prices the parent's part of it.
    #[test]
    fn a_parents_own_work_fused_with_a_priced_child_still_counts() {
        let mut pass = pass();
        let residual = Kernel {
            fuses: vec![ComponentPath::new(["attn norm"])],
            fuses_own: true,
            ..kernel("ElemwiseFuse", 1, 15)
        };
        pass.breakdown.tree[1].kernels = vec![residual.clone()];

        let rendered = Report::new(&pass);
        let block = &pass.breakdown.tree[1];
        assert!(!rendered.is_fused_elsewhere(&block.children[0]));
        assert_eq!(rendered.unpriced_kernel_time(block), residual.total);
    }

    /// A kernel that carried out several components' work says whose, on the
    /// row that holds all of it.
    #[test]
    fn a_fused_kernel_names_the_components_it_carried_out() {
        let mut pass = pass();
        pass.breakdown.tree[1].kernels = vec![Kernel {
            fuses: vec![
                ComponentPath::new(["attn norm"]),
                ComponentPath::new(["attention", "query"]),
            ],
            ..kernel("ElemwiseMatmul", 2, 90)
        }];

        let table = Report::new(&pass).to_string();
        assert!(
            table.contains("ElemwiseMatmul ×2 · fuses ↑ attn norm + attention › query"),
            "{table}"
        );
    }

    /// The ladder's "kernels at their own roofs" is a sum of floors, and a
    /// floor is the declaration's. Taking it from the row's score dropped every
    /// row whose work fused forward, since a score needs measured time — and
    /// pulled the rung below what the pass's own declarations cost.
    #[test]
    fn a_floor_counts_on_the_ladder_whoever_ran_its_work() {
        let pass = pass();
        let before = Report::new(&pass).ladder();

        let mut fused = pass.clone();
        let moved = std::mem::take(&mut fused.breakdown.tree[1].children[0].kernels);
        // Onto the query projection, the leaf that sealed them.
        fused.breakdown.tree[1].children[1].children[0]
            .kernels
            .extend(moved);
        let after = Report::new(&fused).ladder();

        assert_eq!(
            before.kernels, after.kernels,
            "the same declarations are the same floors"
        );
        assert_eq!(before.fused, after.fused);
    }

    /// Measured in place, a row's milliseconds are its kernels' own and the
    /// clean pass splits into the device's time and the host's: the fixture's
    /// kernels come to 1.29 ms of a 2.0 ms pass, so the device waited 0.71 ms.
    #[test]
    fn a_pass_timed_in_place_splits_the_clean_pass_into_device_and_host() {
        let mut pass = pass();
        pass.breakdown.timed_alone = false;
        assert_eq!(
            Report::new(&pass).framework(),
            Some(Duration::from_micros(710))
        );

        let rendered = Report::new(&pass);
        let head = &pass.breakdown.tree[3].children[0];
        assert_eq!(
            rendered.apportioned(head),
            Duration::from_micros(260),
            "the kernel's own time, not a share of the wall"
        );
        assert_eq!(rendered.ladder().host, Duration::from_micros(710));

        let table = rendered.to_string();
        let framework = table
            .lines()
            .find(|line| line.trim_start().starts_with("framework"))
            .expect("the framework line");
        assert!(framework.contains("0.7 ms"), "{framework}");
        assert!(framework.contains("35.5%"), "{framework}");
        assert!(table.contains("on device"), "{table}");
        assert!(table.contains("timed in place"), "{table}");
    }

    /// The clean pass's host time is split by how long the device waited
    /// before each component's kernels, and each component's own share is on
    /// the worklist under the lever that recovers it.
    #[test]
    fn the_host_time_is_split_by_where_the_device_waited() {
        let mut pass = pass();
        pass.breakdown.timed_alone = false;
        let micros = Duration::from_micros;
        // The query's launches kept the device waiting three times as long as
        // the head's: 30 µs against 10 µs of the collected pass's idle time.
        pass.breakdown.tree[1].children[1].children[0].own_wait = micros(30);
        pass.breakdown.tree[3].children[0].own_wait = micros(10);

        let rendered = Report::new(&pass);
        let query = &pass.breakdown.tree[1].children[1].children[0];
        // Three quarters of the 710 µs the device waited in the clean pass.
        let host = rendered.host(query).expect("a pass timed in place splits");
        assert!(
            host.abs_diff(Duration::from_nanos(532_500)) < Duration::from_nanos(10),
            "{host:?}"
        );
        assert_eq!(
            rendered.host(&pass.breakdown.tree[1]),
            rendered.host(query),
            "a component's host time is its children's as well"
        );

        let table = rendered.to_string();
        assert!(table.contains("waited"), "{table}");
        assert!(
            table
                .lines()
                .any(|line| line.contains("query") && line.contains("fewer launches")),
            "the worklist names the lever:\n{table}"
        );

        // Timed alone there is no host time to split.
        pass.breakdown.timed_alone = true;
        assert_eq!(Report::new(&pass).host(query), None);
    }

    /// Timed alone, the kernels' sum is a serialized pass's, so subtracting it
    /// from the clean pass would print the lost overlap as host time.
    #[test]
    fn a_pass_timed_alone_does_not_split_off_the_host() {
        let pass = pass();
        assert_eq!(Report::new(&pass).framework(), None);
        assert_eq!(Report::new(&pass).ladder().host, Duration::ZERO);

        let table = Report::new(&pass).to_string();
        let framework = table
            .lines()
            .find(|line| line.trim_start().starts_with("framework"))
            .expect("the framework line");
        assert!(framework.contains('—'), "{framework}");
    }

    /// With nothing measured on the device there is no split to make — a pass
    /// on a target the runtime does not observe would otherwise read as all
    /// host.
    #[test]
    fn a_pass_with_no_device_time_splits_nothing() {
        let mut pass = pass();
        pass.breakdown.timed_alone = false;
        for node in pass.breakdown.tree.iter_mut() {
            clear(node);
        }
        pass.breakdown.untagged.clear();
        assert_eq!(Report::new(&pass).framework(), None);

        fn clear(node: &mut Component) {
            node.kernels.clear();
            node.children.iter_mut().for_each(clear);
        }
    }

    /// A composite that prices only the traffic it must move is still held
    /// to every multiply-add beneath it: fusion removes traffic, never
    /// arithmetic. The whole model declares its floor that way.
    #[test]
    fn a_composite_is_held_to_the_arithmetic_of_its_parts() {
        let pass = pass();
        let parts: u64 = pass
            .breakdown
            .tree
            .iter()
            .filter_map(|node| node.bounds)
            .map(|bounds| bounds.flops)
            .sum();
        let model = Component {
            label: "transformer".into(),
            bounds: Some(Bounds {
                read: 1_000_000,
                write: 1_000,
                flops: 0,
            }),
            children: pass.breakdown.tree.clone(),
            ..row()
        };

        let held = Report::held(&model).expect("the model declared bounds");
        assert_eq!(held.flops, parts);
        assert_eq!(
            (held.read, held.write),
            (1_000_000, 1_000),
            "its traffic is its own"
        );
    }

    /// An unpriced root — the model itself — does not hide the priced
    /// components under it: the fusion rung still takes each at its own floor
    /// rather than falling to the kernel rung.
    #[test]
    fn an_unpriced_root_keeps_the_fusion_rung() {
        let pass = pass();
        let flat = Report::new(&pass).ladder();

        let mut rooted = pass.clone();
        rooted.breakdown.tree = vec![Component {
            label: "transformer".into(),
            children: std::mem::take(&mut rooted.breakdown.tree),
            ..row()
        }];
        let under_a_root = Report::new(&rooted).ladder();

        assert_eq!(flat.fused, under_a_root.fused);
        assert_eq!(flat.kernels, under_a_root.kernels);
        assert!(under_a_root.from_fusion() > Duration::ZERO);
    }

    /// The fused work the last components left for the sync ending the pass
    /// is printed under its own heading and counted in what the device did.
    #[test]
    fn the_end_of_the_pass_is_printed_and_counted() {
        let mut pass = pass();
        let before = pass.breakdown.total_time();
        pass.breakdown.end_of_pass = vec![kernel("FusedMatmul", 1, 40)];

        let table = Report::new(&pass).to_string();
        assert!(table.contains("end of pass"), "{table}");
        assert_eq!(
            pass.breakdown.total_time(),
            before + Duration::from_micros(40)
        );
    }

    #[test]
    fn an_unprobed_device_scores_nothing_rather_than_printing_a_guess() {
        let mut pass = pass();
        pass.peaks = None;
        let table = Report::new(&pass).to_string();
        assert!(table.contains("unprobed"), "the header says so:\n{table}");
        // Every roofline column reads as absent rather than as zero.
        assert!(!table.contains('%') || table.contains('—'), "{table}");
    }

    /// The wgpu reading that prompted the guard: a write ceiling above the
    /// device's own read ceiling, on a bus that carries neither. The pass must
    /// say the pair was dropped rather than quietly fall back, or a reader
    /// cannot tell it from a device that reported no directions at all.
    #[test]
    fn an_impossible_write_ceiling_is_reported_not_divided_by() {
        let broken = Peaks {
            copy_bytes_per_s: Some(156.0e9),
            read_bytes_per_s: Some(227.0e9),
            write_bytes_per_s: Some(281.0e9),
            flops_per_s: Some(49.3e12),
            ..Peaks::default()
        };
        let pass = CollectedPass {
            peaks: Some(broken),
            ..pass()
        };
        let table = Report::new(&pass).to_string();

        assert!(
            table.contains("REJECTED the per-direction peaks"),
            "the header says so:\n{table}"
        );
        // The rejection names both readings — that is the point of naming it —
        // so what must be absent is the line presenting them as the denominator,
        // not the numbers themselves.
        assert!(
            !table.contains("GB/s read and"),
            "and does not present them as denominators:\n{table}"
        );
        // And it is not also reported as merely disagreeing: one finding, named
        // once, in the terms that say what happened to the denominator.
        assert!(!table.contains("CAVEAT"), "named once:\n{table}");
    }

    /// Three ceilings that cannot all be measurements of one device are named,
    /// and still divided by — nothing in the readings says which is wrong.
    #[test]
    fn ceilings_that_do_not_predict_each_other_are_a_caveat_not_a_fallback() {
        // Plausible in their ordering, so nothing is rejected; but 234 and 212
        // predict a 222 GB/s copy ceiling against a measured 140.
        let odd = Peaks {
            copy_bytes_per_s: Some(140.0e9),
            read_bytes_per_s: Some(234.0e9),
            write_bytes_per_s: Some(212.0e9),
            flops_per_s: Some(49.3e12),
            ..Peaks::default()
        };
        let pass = CollectedPass {
            peaks: Some(odd),
            ..pass()
        };
        let table = Report::new(&pass).to_string();

        assert!(table.contains("CAVEAT"), "the header says so:\n{table}");
        assert!(
            !table.contains("REJECTED the per-direction peaks"),
            "and nothing was dropped:\n{table}"
        );
        assert!(
            table.contains("234.0 GB/s read"),
            "the rows are still scored on the directions:\n{table}"
        );
    }

    /// The failure a real ROCm run hit: the f16 CMMA probe returned a tiny
    /// positive number, so every row read as thousands of percent of peak with
    /// a floor longer than the pass. The pass must say the denominator broke
    /// rather than render it.
    #[test]
    fn a_broken_compute_peak_is_reported_not_divided_by() {
        let broken = Peaks {
            copy_bytes_per_s: Some(166.1e9),
            flops_per_s: Some(4.5e9),
            ..Peaks::default()
        };
        let pass = CollectedPass {
            peaks: Some(broken),
            ..pass()
        };
        let table = Report::new(&pass).to_string();

        assert!(table.contains("REJECTED"), "the header says so:\n{table}");
        assert!(
            table.contains("unprobed"),
            "and does not print the rejected peak as though it were one:\n{table}"
        );
        // Falls back to the `bytes` wall rather than producing no rows at all.
        assert!(table.contains("bytes"), "{table}");
        // Structurally, not by string: every row falls back to the `bytes`
        // wall, since the one it would otherwise be held against has no
        // denominator anyone measured.
        let rendered = Report::new(&pass);
        for (_, node) in pass.breakdown.tree.iter().flat_map(|node| node.walk()) {
            if let Some(score) = rendered.score_of(node) {
                assert_eq!(score.bound, Bound::Bytes, "{}", node.label);
            }
        }
    }

    /// A component that beats its floor is at the roof with a probe reading
    /// slightly low — the head, streaming 311 MB of weights, does this. It is
    /// usually the most interesting row on the page, so it is marked rather
    /// than blanked: suppressing its numbers would lose the finding.
    ///
    /// And it keeps its percentage, because how far past is the whole reading:
    /// this row is at 118%, which is a probe a fifth low; a row at 568% is a
    /// declared bound that is wrong. A shared ">100%" gave both the same
    /// answer.
    #[test]
    fn a_row_past_the_peak_is_marked_and_still_shows_its_numbers() {
        let mut pass = pass();
        // One row whose kernels ran just under its own floor. The roofline is
        // scored on device time, so the row needs a timed kernel rather than a
        // wall time — a component with no kernel has nothing to be a
        // percentage of.
        pass.breakdown.tree.truncate(1);
        pass.breakdown.tree[0].bounds = Some(Bounds {
            read: 240_000_000,
            write: 80_000_000,
            flops: 0,
        });
        pass.breakdown.tree[0].children.clear();
        pass.breakdown.tree[0].kernels = vec![kernel("TheHead", 1, 1_700)];
        pass.breakdown.untagged.clear();
        // The one component is the whole pass, so its apportioned share is the
        // wall — and 1.7 ms against a 2.0 ms floor is a row at the roof.
        pass.clean_wall = Some(Duration::from_micros(1_700));

        let peaks = Peaks {
            copy_bytes_per_s: Some(160.0e9),
            flops_per_s: Some(32.0e12),
            ..Peaks::default()
        };
        pass.peaks = Some(peaks);
        let table = Report::new(&pass).to_string();
        // 320 MB against a 160 GB/s ceiling is a 2.0 ms floor, reached in 1.7.
        assert!(
            table.contains("118%*"),
            "the row is marked and keeps its figure:\n{table}"
        );
        assert!(
            table.contains("past the measured peak"),
            "and the mark is explained:\n{table}"
        );
        // The row keeps its numbers. An earlier version blanked such a row as
        // "denominator broken", which on a real run hid the LM head — the best
        // row on the page — behind two dashes.
        let row = table
            .lines()
            .find(|line| line.contains('*'))
            .expect("the marked row");
        assert!(
            !row.contains('—'),
            "a row at the roof keeps its floor and gain:\n{row}"
        );
    }

    /// A row fed from two places says so, and names them.
    ///
    /// Without the note a merge is invisible: two entry points under one
    /// label read exactly like one entered twice. The note leaves the tree
    /// alone — merging is what makes a 36-layer stack readable — and points at
    /// the sites, which is what a split is made from.
    #[test]
    fn a_row_fed_from_two_sites_says_where_from() {
        let mut pass = pass();
        pass.breakdown.tree[1].children[1].sites = vec![
            "my_model::attention::base::forward_cached".to_string(),
            "my_model::attention::base::decode_batch".to_string(),
        ];

        let table = Report::new(&pass).to_string();
        assert!(
            table.contains(
                "note: `attention` under `transformer block` was entered from 2 sites \
                 (attention::forward_cached, attention::decode_batch)"
            ),
            "the note names the row, its parent and the sites:\n{table}"
        );
    }

    /// The ordinary row — one site — carries no note. A footnote on every row
    /// is a footnote a reader stops reading.
    #[test]
    fn a_row_fed_from_one_site_carries_no_note() {
        let table = Report::new(&pass()).to_string();
        assert!(
            !table.contains("was entered from"),
            "nothing to say about a row nothing merged into:\n{table}"
        );
    }

    /// A site is shortened to what a reader greps for: the crate is the same
    /// for most rows in the tree, and a `base` module holds a folder's one
    /// implementation, so neither tells anyone anything.
    #[test]
    fn a_site_is_named_by_its_module_and_entry() {
        assert_eq!(site("my_model::rope::base::apply"), "rope::apply");
        assert_eq!(
            site("my_model::attention::cache::base::append"),
            "attention::cache::append"
        );
    }

    /// A row whose execution is chosen at call time says which one ran.
    ///
    /// The attention engine is one label serving four executions — the tile fold,
    /// the flash op, tensor ops and the decode kernel. Without the variant they
    /// render as the same word, and the row a reader opened the table for reads
    /// as though nothing were decided in it.
    #[test]
    fn a_row_prints_which_executions_served_it() {
        let mut pass = pass();
        pass.breakdown.tree[0].label = "attention engine".to_string();
        pass.breakdown.tree[0].children.clear();
        pass.breakdown.tree[0].entries = 9;
        pass.breakdown.tree[0].variants = vec![
            Variant {
                name: "tile".to_string(),
                entries: 8,
            },
            Variant {
                name: "tensor-ops".to_string(),
                entries: 1,
            },
        ];

        let table = Report::new(&pass).to_string();
        assert!(
            table.contains("attention engine ×9 · tile ×8, tensor-ops ×1"),
            "the split is beside the entry count:\n{table}"
        );
    }

    /// One execution that served every entry needs no count: it would only
    /// repeat the row's own, and the reading — "this row took the decode
    /// kernel, all nine times" — is the same either way.
    #[test]
    fn a_row_served_one_way_prints_the_bare_name() {
        let mut pass = pass();
        pass.breakdown.tree[0].label = "attention engine".to_string();
        pass.breakdown.tree[0].children.clear();
        pass.breakdown.tree[0].entries = 28;
        pass.breakdown.tree[0].variants = vec![Variant {
            name: "decode".to_string(),
            entries: 28,
        }];

        let table = Report::new(&pass).to_string();
        assert!(
            table.contains("attention engine ×28 · decode"),
            "the name alone:\n{table}"
        );
        assert!(
            !table.contains("decode ×28"),
            "and not its count twice:\n{table}"
        );
    }

    /// The component column sizes itself to what it has to hold, so a long
    /// variant widens the table rather than pushing one row's numbers out of
    /// their columns and leaving the rest aligned to nothing.
    #[test]
    fn every_row_keeps_its_columns_when_a_heading_grows() {
        let mut pass = pass();
        pass.breakdown.tree[0].label = "attention engine".to_string();
        pass.breakdown.tree[0].entries = 9;
        pass.breakdown.tree[0].variants = vec![
            Variant {
                name: "prefill-fold".to_string(),
                entries: 8,
            },
            Variant {
                name: "tensor-ops".to_string(),
                entries: 1,
            },
        ];

        let table = Report::new(&pass).to_string();
        // Where a right-aligned column ends, on the header and on the
        // subheading beneath it.
        let ends = |line: &str, word: &str| {
            table
                .lines()
                .find(|found| found.contains(line))
                .and_then(|found| found.find(word))
                .map(|at| at + word.len())
        };
        assert_eq!(
            ends("gain ms", "share"),
            ends("recoverable", "of work"),
            "the header and its subheading share one column layout:\n{table}"
        );
        // And the rule follows the column rather than staying at the width a
        // narrower table used, which would leave it ending mid-heading.
        let rule = table
            .lines()
            .map(|line| line.trim())
            .find(|line| !line.is_empty() && line.chars().all(|glyph| glyph == '─'))
            .expect("the rule under the header")
            .chars()
            .count();
        assert!(
            rule > 88,
            "the rule widened with the column ({rule} columns):\n{table}"
        );
        // And the heading ends inside its column rather than running into the
        // figures — which is the failure the fixed width would have produced.
        let widened = table
            .lines()
            .find(|line| line.contains("attention engine"))
            .expect("the widened row");
        let after = widened
            .split_once("tensor-ops ×1")
            .expect("the heading, in full")
            .1;
        assert!(
            after.starts_with(' '),
            "the figures do not butt up against the heading:\n{widened}"
        );
    }

    /// A kernel's figures sit in its component's columns, whatever the depth:
    /// a kernel line two columns to the right reads its milliseconds as the
    /// share beside them.
    #[test]
    fn a_kernel_line_keeps_its_components_columns() {
        let pass = pass();
        let table = Report::new(&pass).to_string();
        // The column, in characters, where the milliseconds figure after the
        // label ends: `·` and `×` are wider than a byte.
        let ms_end = |needle: &str| {
            let line = table
                .lines()
                .find(|line| line.contains(needle))
                .unwrap_or_else(|| panic!("no line with {needle}:\n{table}"));
            let (before, after) = line.split_once(needle).expect("found above");
            let digits = after
                .find(|glyph: char| glyph.is_ascii_digit())
                .expect("a figure");
            let figure = after[digits..].split(' ').next().expect("a figure");
            before.chars().count()
                + needle.chars().count()
                + after[..digits].chars().count()
                + figure.chars().count()
        };
        for (component, kernel) in [
            ("embedding", "· Gather ×1"),
            ("query ×2", "· Matmul ×2"),
            ("attn norm ×2", "· RmsNormKernel ×2"),
        ] {
            assert_eq!(ms_end(component), ms_end(kernel), "{kernel}:\n{table}");
        }
        assert_eq!(
            ms_end("embedding"),
            ms_end("· Reshape ×1"),
            "an untagged kernel too:\n{table}"
        );
    }

    /// Past the column's cap a heading is cut, and cut visibly.
    ///
    /// The cap exists so one name cannot push every number off a terminal; the
    /// row it caps still has to line up with the rest, because the row with the
    /// most to say — the one that routed four ways — is the one a reader came
    /// for, and it would be the only unreadable line in the table.
    #[test]
    fn a_heading_past_the_cap_is_elided_rather_than_overflowing() {
        let mut pass = pass();
        pass.breakdown.tree[0].label = "an operator with a very long name indeed".to_string();
        pass.breakdown.tree[0].children.clear();
        pass.breakdown.tree[0].entries = 4;
        pass.breakdown.tree[0].variants = ["tile", "cubek", "prefill-fold", "tensor-ops"]
            .into_iter()
            .map(|name| Variant {
                name: name.to_string(),
                entries: 1,
            })
            .collect();

        let table = Report::new(&pass).to_string();
        let row = table
            .lines()
            .find(|line| line.contains("an operator with"))
            .expect("the capped row");
        assert!(
            row.contains('…'),
            "the cut is marked rather than silent:\n{row}"
        );

        // And its figures sit in the columns the header set, which is the
        // property the cut protects. Measured in columns rather than bytes:
        // `×` and `…` are wider than one byte, and an offset would make the
        // elided row look misaligned when it is not.
        let header = table
            .lines()
            .find(|line| line.contains("component") && line.contains("gain ms"))
            .expect("the header");
        // The bare `ms` column, right-aligned like every value under it.
        let at = header.find(" ms").expect("the ms column") + 3;
        let column = header[..at].chars().count();

        let ordinary = table
            .lines()
            .find(|line| line.contains("final norm"))
            .expect("a row with a short heading");
        for line in [row, ordinary] {
            let glyph = |column: usize| line.chars().nth(column);
            assert!(
                glyph(column - 1).is_some_and(|glyph| glyph.is_ascii_digit()),
                "a figure ends where the header's `ms` does:\n{line}"
            );
            assert_eq!(
                glyph(column),
                Some(' '),
                "and the next column starts after it:\n{line}"
            );
        }
    }

    /// Saturating the ladder's subtractions would print "0.0 on the table"
    /// under rungs that plainly exceed the pass — which reads as "nothing to
    /// win", the opposite of what happened.
    #[test]
    fn an_incoherent_ladder_says_so_rather_than_reporting_nothing_to_win() {
        let inverted = Ladder {
            measured: Duration::from_millis(715),
            host: Duration::ZERO,
            kernels: Duration::from_millis(88_535),
            fused: Duration::from_millis(102_582),
        };
        assert!(!inverted.coherent());
        assert_eq!(
            inverted.total(),
            Duration::ZERO,
            "the saturation is why this must not be printed as a result"
        );

        let real = Ladder {
            measured: Duration::from_millis(1000),
            host: Duration::ZERO,
            kernels: Duration::from_millis(600),
            fused: Duration::from_millis(400),
        };
        assert!(real.coherent());
    }

    /// A priced row that measured no time keeps its bound and floor: its
    /// parent's floor counts them, and blanks would leave that figure with
    /// nothing to be reconciled against.
    #[test]
    fn an_unscored_priced_row_keeps_its_floor() {
        let mut pass = pass();
        pass.breakdown.tree[1].children[1].children[0].kernels[0].total = Duration::ZERO;

        let rendered = Report::new(&pass);
        let query = &pass.breakdown.tree[1].children[1].children[0];
        assert!(
            rendered.score_of(query).is_none(),
            "nothing measured to score"
        );
        let floor = rendered.floor_of(query).expect("the query is priced");
        let table = rendered.to_string();
        let row = table
            .lines()
            .find(|line| line.trim_start().starts_with("query"))
            .expect("the query's row");
        assert!(row.contains(floor.bound.label()), "{row}");
        assert!(row.contains(&ms(floor.time)), "{row}");
    }

    /// A kernel that launched and measured nothing is a lost timing, said in
    /// the summary — and the marks and the worklist that read those timings
    /// say they cannot be trusted rather than "nothing to win".
    #[test]
    fn a_kernel_that_measured_no_time_is_reported_lost() {
        let mut pass = pass();
        assert_eq!(pass.breakdown.untimed().count(), 0);
        let table = Report::new(&pass).to_string();
        assert!(!table.contains("untimed"), "{table}");

        pass.breakdown.tree[1].children[1].children[0].kernels[0].total = Duration::ZERO;
        assert_eq!(pass.breakdown.untimed().count(), 1);
        let table = Report::new(&pass).to_string();
        assert!(
            table.contains("untimed") && table.contains("1 kernels, 2 launches"),
            "{table}"
        );
        assert!(!table.contains("Nothing to win"), "{table}");
    }

    /// A wait before a single launch is the host's work ahead of it; no
    /// launch can be taken away from one, so it is not "fewer launches".
    #[test]
    fn a_wait_before_one_launch_is_host_work() {
        let mut pass = pass();
        pass.breakdown.timed_alone = false;
        pass.breakdown.tree[0].own_wait = Duration::from_micros(30);

        let table = Report::new(&pass).to_string();
        let line = table
            .lines()
            .find(|line| line.contains("embedding") && line.contains("host work"))
            .unwrap_or_else(|| panic!("the embedding's worklist line:\n{table}"));
        assert!(line.contains("before its one launch"), "{line}");
    }

    #[test]
    fn the_ladder_never_promises_more_than_the_pass_took() {
        let pass = pass();
        let rendered = Report::new(&pass);
        let ladder = rendered.ladder();
        assert!(
            ladder.total() <= ladder.measured,
            "a gain larger than the time it came from is not a gain"
        );
        assert!(
            ladder.fused <= ladder.kernels,
            "perfect fusion cannot be slower than each part at its own roof"
        );
    }

    #[test]
    fn a_repeated_component_says_how_many_times_it_ran() {
        let pass = pass();
        let table = Report::new(&pass).to_string();
        // Two layers, one row: without the count a reader would divide by the
        // wrong number to get a per-layer figure.
        assert!(
            table.contains("transformer block ×2"),
            "the stack's row carries its layer count:\n{table}"
        );
    }
}
