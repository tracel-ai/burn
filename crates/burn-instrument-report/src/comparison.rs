//! Two measurements, side by side.
//!
//! Every performance change is a comparison, and a table on its own is not one: a reader holds
//! the previous measurement in a scrollback buffer and subtracts by eye, which is where a
//! regression in a component nobody was looking at survives.
//!
//! Components are keyed on their **path** from the root: a label alone is ambiguous (`down` sits
//! under a feed-forward in every layer), and a component is merged on its label only among its
//! siblings. Each side's milliseconds are the ones its own [`Report`] prints, so the comparison
//! cannot disagree with either table.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};

use burn_instrument::{Component, ComponentPath, Measurement};

use crate::Report;
use crate::millis::Millis;
use crate::served_by::ServedBy;

/// A measurement beside the one it is being held against: what [`Report::against`] returns.
pub struct Comparison<'a> {
    now: Report<'a>,
    was: Report<'a>,
    /// Every component of `now`, depth-first, with its path from the root.
    now_components: Vec<(ComponentPath, &'a Component)>,
    /// Every component of `was`, depth-first, with its path from the root.
    was_components: Vec<(ComponentPath, &'a Component)>,
}

/// One component of the current measurement beside the same path in the saved one.
struct Compared<'a> {
    path: &'a ComponentPath,
    now: &'a Component,
    /// The same path in the saved measurement, where it had one. `None` is a component this
    /// measurement has and the saved one did not: added, renamed or newly annotated, and not a
    /// delta of any kind.
    was: Option<&'a Component>,
}

impl<'a> Comparison<'a> {
    /// `now`'s measurement against `saved`. Two different types, so the two sides cannot be
    /// swapped by accident: a swap would flip the sign of every delta.
    pub(crate) fn new(now: Report<'a>, saved: &'a Measurement) -> Self {
        Self {
            now_components: now.measurement().breakdown.components(),
            was_components: saved.breakdown.components(),
            now,
            was: Report::new(saved),
        }
    }

    /// Each component of the current measurement beside the same path in the saved one.
    fn compared(&self) -> Vec<Compared<'_>> {
        let was: HashMap<&ComponentPath, &Component> = self
            .was_components
            .iter()
            .map(|(path, component)| (path, *component))
            .collect();
        self.now_components
            .iter()
            .map(|(path, now)| Compared {
                path,
                now,
                was: was.get(path).copied(),
            })
            .collect()
    }

    /// Whether `compared`'s work changed hands between the two measurements: fused into an
    /// ancestor's kernel in one and launched on its own in the other.
    ///
    /// Not a delta of any kind. A component that fused away since the saved measurement reads zero
    /// because its work moved up, and the ancestor that absorbed it gained that time, so
    /// subtracting gives a −100% win beside an equal regression: one fusion decision counted
    /// twice with opposite signs.
    fn changed_hands(&self, compared: &Compared<'_>) -> bool {
        compared.was.is_some_and(|was| {
            self.was.roofline().fused_elsewhere().contains(was)
                != self.now.roofline().fused_elsewhere().contains(compared.now)
        })
    }

    /// What the two measurements differ in that makes their difference a number about nothing.
    fn mismatches(&self) -> Vec<Mismatch<'_>> {
        let (now, was) = (self.now.measurement(), self.was.measurement());
        let mut mismatches = Vec::new();
        if now.device != was.device {
            mismatches.push(Mismatch::Device {
                was: &was.device,
                now: &now.device,
            });
        }
        if now.breakdown.unfused != was.breakdown.unfused {
            mismatches.push(Mismatch::Graph {
                was_unfused: was.breakdown.unfused,
            });
        }
        if now.breakdown.timed_alone != was.breakdown.timed_alone {
            mismatches.push(Mismatch::Timing {
                was_alone: was.breakdown.timed_alone,
            });
        }
        if now.clean_wall.is_some() != was.clean_wall.is_some() {
            mismatches.push(Mismatch::CleanRun {
                was_measured: was.clean_wall.is_some(),
            });
        }
        mismatches
    }
}

/// One way two measurements differ in what they measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mismatch<'a> {
    /// They ran on different devices.
    Device { was: &'a str, now: &'a str },
    /// One ran fused and the other unfused; `was_unfused` is the saved measurement's.
    Graph { was_unfused: bool },
    /// One timed its kernels in place and the other alone; `was_alone` is the saved measurement's.
    Timing { was_alone: bool },
    /// One measured a clean run and the other did not, so one side's milliseconds are that run's
    /// and the other's the instrumented run's own; `was_measured` is the saved measurement's.
    CleanRun { was_measured: bool },
}

impl Display for Mismatch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let graph = |unfused: bool| match unfused {
            true => "unfused",
            false => "fused",
        };
        let timed = |alone: bool| match alone {
            true => "alone",
            false => "in place",
        };
        let measured = |measured: bool| match measured {
            true => "measured",
            false => "not measured",
        };
        match *self {
            Mismatch::Device { was, now } => write!(f, "device was {was}, now {now}"),
            Mismatch::Graph { was_unfused } => write!(
                f,
                "graph was {}, now {}",
                graph(was_unfused),
                graph(!was_unfused)
            ),
            Mismatch::Timing { was_alone } => write!(
                f,
                "kernels were timed {}, now {}",
                timed(was_alone),
                timed(!was_alone)
            ),
            Mismatch::CleanRun { was_measured } => write!(
                f,
                "clean run was {}, now {}",
                measured(was_measured),
                measured(!was_measured)
            ),
        }
    }
}

impl Display for Comparison<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let compared = self.compared();
        let width = compared
            .iter()
            .map(|component| component.path.to_string().chars().count())
            .chain([28])
            .max()
            .unwrap_or(28)
            .min(56);

        // Named before anything is subtracted: two measurements that differ in what
        // they measured have a difference about nothing, and a reader who is
        // not told reads it as a result.
        let mismatches = self.mismatches();
        writeln!(
            f,
            "\n  against a saved measurement: {}",
            match mismatches.is_empty() {
                true => "same device, graph, timing and clean run".to_string(),
                false => format!(
                    "DIFFERENT — {}; the deltas below compare two different measurements",
                    mismatches
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        )?;

        writeln!(
            f,
            "\n  {:<width$} {:>8} {:>8} {:>13}  served by",
            "component", "was ms", "now ms", "delta"
        )?;
        writeln!(f, "  {}", "─".repeat(width + 42))?;
        for component in &compared {
            let now = self.now.reading().apportioned(component.now);
            let delta = match component.was {
                // A component the saved measurement did not have. Dashes, not zeroes:
                // zero is a measurement and this is the absence of one.
                None => ("—".to_string(), "new".to_string()),
                // Both measurements have it, but its work changed hands, so there is
                // a `was` to print and no delta to draw from it.
                Some(was) if self.changed_hands(component) => (
                    format!("{:.3}", Millis(self.was.reading().apportioned(was))),
                    match self
                        .now
                        .roofline()
                        .fused_elsewhere()
                        .contains(component.now)
                    {
                        true => "fused ↓".to_string(),
                        false => "unfused".to_string(),
                    },
                ),
                Some(was) => {
                    let was = self.was.reading().apportioned(was);
                    let change = (now.as_secs_f64() - was.as_secs_f64()) * 1e3;
                    // No ratio against nothing: "infinitely worse" about a
                    // component that may have moved a microsecond.
                    let delta = match was.is_zero() {
                        true => format!("{change:+.3}"),
                        false => format!(
                            "{change:+.3} {:+.0}%",
                            change / (was.as_secs_f64() * 1e3) * 100.0
                        ),
                    };
                    (format!("{:.3}", Millis(was)), delta)
                }
            };
            writeln!(
                f,
                "  {:<width$} {:>8} {:>8.3} {:>13}  {}",
                component.path,
                delta.0,
                Millis(now),
                delta.1,
                ServedBy(component.now)
            )?;
        }

        // Components the saved measurement had and this one does not: the other half
        // of a rename, which a per-component delta cannot show because it walks
        // the components this measurement produced.
        let present: HashSet<&ComponentPath> =
            self.now_components.iter().map(|(path, _)| path).collect();
        let gone: Vec<String> = self
            .was_components
            .iter()
            .filter(|(path, _)| !present.contains(path))
            .map(|(path, _)| path.to_string())
            .collect();
        if !gone.is_empty() {
            writeln!(
                f,
                "\n  note: {} component(s) of the saved measurement are absent here ({})",
                gone.len(),
                gone.join(", ")
            )?;
        }

        let (now, was) = (self.now.wall(), self.was.wall());
        let relative = match was.is_zero() {
            true => String::new(),
            false => format!(
                " ({:+.1}%)",
                (now.as_secs_f64() - was.as_secs_f64()) / was.as_secs_f64() * 100.0
            ),
        };
        writeln!(
            f,
            "\n  {:<width$} {:>8.3} {:>8.3} ms{relative}",
            "run",
            Millis(was),
            Millis(now)
        )
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use burn_instrument::{Breakdown, Kernel, Unplaced, Variant};

    use super::*;

    /// A measurement, stated rather than taken: the comparison is a function of two of these, so
    /// the tests own both sides and every number in them.
    fn measurement(tree: Vec<Component>) -> Measurement {
        Measurement {
            device: "cpu".to_string(),
            peaks: None,
            clean_wall: None,
            breakdown: Breakdown {
                tree,
                untagged: Vec::new(),
                end_of_pass: Vec::new(),
                outside_wait: Duration::ZERO,
                host_timed: false,
                timed_alone: false,
                unplaced: Unplaced::default(),
                unfused: false,
            },
        }
    }

    /// A component whose one kernel took `micros`, with `children` under it.
    fn component(label: &str, micros: u64, children: Vec<Component>) -> Component {
        Component {
            label: label.to_string(),
            variants: Vec::new(),
            sites: Vec::new(),
            bounds: None,
            observed: None,
            entries: 1,
            children,
            kernels: match micros {
                0 => Vec::new(),
                micros => vec![kernel(label, micros, Vec::new())],
            },
            own_wait: Duration::ZERO,
        }
    }

    fn kernel(name: &str, micros: u64, fuses: Vec<ComponentPath>) -> Kernel {
        Kernel {
            name: name.to_string(),
            launches: 1,
            total: Duration::from_micros(micros),
            fuses,
            fuses_own: false,
        }
    }

    fn line<'t>(table: &'t str, path: &str) -> &'t str {
        table
            .lines()
            .find(|line| line.trim_start().starts_with(path))
            .unwrap_or_else(|| panic!("no line for {path}:\n{table}"))
    }

    /// The delta a reader came for: what moved, by how much, and against what.
    #[test]
    fn a_component_that_moved_prints_what_it_moved_by() {
        let was = measurement(vec![component("norm", 100, Vec::new())]);
        let now = measurement(vec![component("norm", 150, Vec::new())]);
        let table = Report::new(&now).against(&was).to_string();
        let norm = line(&table, "norm");
        assert!(norm.contains("0.100"), "{norm}");
        assert!(norm.contains("0.150"), "{norm}");
        assert!(norm.contains("+0.050 +50%"), "{norm}");
        assert!(
            table.contains("same device, graph, timing and clean run"),
            "{table}"
        );
    }

    /// Keyed on where a component sits, not on what it is called: `down` under two different
    /// parents is two components.
    #[test]
    fn components_are_keyed_on_their_path_not_their_label() {
        let was = measurement(vec![
            component("dense", 0, vec![component("down", 100, Vec::new())]),
            component("sparse", 0, vec![component("down", 900, Vec::new())]),
        ]);
        let now = measurement(vec![
            component("dense", 0, vec![component("down", 110, Vec::new())]),
            component("sparse", 0, vec![component("down", 900, Vec::new())]),
        ]);
        let table = Report::new(&now).against(&was).to_string();
        assert!(
            line(&table, "dense › down").contains("+0.010 +10%"),
            "{table}"
        );
        assert!(line(&table, "sparse › down").contains("+0.000"), "{table}");
    }

    /// A component the saved measurement did not have is `new`: printing its whole time as a
    /// regression would put the biggest number in the table on the component that least deserves
    /// it.
    #[test]
    fn a_component_the_saved_measurement_lacked_is_new_rather_than_a_regression() {
        let was = measurement(vec![component("norm", 100, Vec::new())]);
        let now = measurement(vec![
            component("norm", 100, Vec::new()),
            component("rope", 400, Vec::new()),
        ]);
        let table = Report::new(&now).against(&was).to_string();
        let rope = line(&table, "rope");
        assert!(rope.contains("new"), "{rope}");
        assert!(!rope.contains('+'), "{rope}");
    }

    /// The other half of a rename: a component the saved measurement had and this one does not
    /// would otherwise read as one that silently got free.
    #[test]
    fn a_component_that_disappeared_is_named() {
        let was = measurement(vec![
            component("norm", 100, Vec::new()),
            component("old name", 100, Vec::new()),
        ]);
        let now = measurement(vec![component("norm", 100, Vec::new())]);
        let table = Report::new(&now).against(&was).to_string();
        assert!(table.contains("1 component(s)"), "{table}");
        assert!(table.contains("(old name)"), "{table}");
    }

    /// A component whose work fused into its parent's kernel is not a win, and the parent that
    /// absorbed it is not a regression of the same size: the change is named, not subtracted.
    #[test]
    fn work_that_changed_hands_is_named_rather_than_subtracted() {
        let was = measurement(vec![component(
            "block",
            200,
            vec![component("norm", 100, Vec::new())],
        )]);
        let mut block = component("block", 0, vec![component("norm", 0, Vec::new())]);
        block.kernels = vec![kernel("fused", 300, vec![ComponentPath::new(["norm"])])];
        let now = measurement(vec![block]);

        let table = Report::new(&now).against(&was).to_string();
        let norm = line(&table, "block › norm");
        assert!(norm.contains("fused ↓"), "{norm}");
        assert!(!norm.contains("-0.100"), "{norm}");
    }

    /// A saved measurement with no clean run beside one with: one side's `run` line is that
    /// run's wall and the other its kernels' device time, so their difference is the host.
    #[test]
    fn a_clean_run_on_one_side_only_is_named() {
        let was = measurement(vec![component("norm", 100, Vec::new())]);
        let mut now = measurement(vec![component("norm", 100, Vec::new())]);
        now.clean_wall = Some(Duration::from_micros(300));

        let table = Report::new(&now).against(&was).to_string();
        assert!(table.contains("DIFFERENT"), "{table}");
        assert!(
            table.contains("clean run was not measured, now measured"),
            "{table}"
        );
    }

    /// A delta carries the executions that served the component, because a routing change is
    /// the most common reason for one.
    #[test]
    fn a_delta_carries_the_execution_that_served_it() {
        let was = measurement(vec![component("attention", 100, Vec::new())]);
        let mut attention = component("attention", 300, Vec::new());
        attention.variants = vec![Variant {
            name: "tensor-ops".to_string(),
            entries: 1,
        }];
        let now = measurement(vec![attention]);
        let table = Report::new(&now).against(&was).to_string();
        assert!(line(&table, "attention").ends_with("tensor-ops"), "{table}");
    }

    /// Two measurements of different things are announced before a number is printed,
    /// each difference named.
    #[test]
    fn measurements_of_different_things_are_announced_first() {
        let was = measurement(vec![component("norm", 100, Vec::new())]);
        let mut now = was.clone();
        now.device = "Rocm(0)".to_string();
        now.breakdown.unfused = true;
        let table = Report::new(&now).against(&was).to_string();
        let header = table
            .lines()
            .find(|line| line.contains("against a saved measurement"))
            .expect("the header");
        assert!(header.contains("DIFFERENT"), "{header}");
        assert!(header.contains("device was cpu, now Rocm(0)"), "{header}");
        assert!(header.contains("graph was fused, now unfused"), "{header}");
        assert!(
            table.find("DIFFERENT").unwrap() < table.find("norm").unwrap(),
            "before any component:\n{table}"
        );
    }
}
