//! What a collected pass leaves behind: the components it entered, the
//! kernels each launched, and what the collection itself can say about how
//! far those numbers are to be trusted.

use std::io::{Read, Write};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::Bounds;
use crate::row_path::RowPath;

/// One collected pass, broken down by component.
///
/// What [`SharedCollector::collect`](crate::SharedCollector::collect) returns:
/// data only, so a breakdown collected from a run and one written out in a
/// test are the same kind of value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Breakdown {
    /// The components the collected pass entered.
    pub tree: Vec<Node>,
    /// Kernels the pass launched under no component at all, and fused kernels
    /// that carried out work from outside every component or across two roots,
    /// named for the components they did carry out (see [`Kernel::fuses`]).
    pub untagged: Vec<Kernel>,
    /// Kernels the sync ending the pass ran: fused work the last components
    /// left queued, which no op after them was there to seal. See
    /// [`SharedCollector::collect`](crate::SharedCollector::collect).
    pub end_of_pass: Vec<Kernel>,
    /// How long the device waited on the host before the untagged kernels
    /// and the end of the pass's — the idle time no row is charged with. See
    /// [`Node::waited`].
    #[serde(with = "nanoseconds")]
    pub waited_outside: Duration,
    /// Whether any launch was timed on the host rather than the device.
    ///
    /// wgpu budgets its timestamp query sets and falls back to a host timer
    /// when they run out, and a host timing is wall around a blocking submit —
    /// submission and sync included, so a pass with one reads several times
    /// its device time.
    pub host_timed: bool,
    /// Whether any launch was timed alone — read back as it launched, or by a
    /// host timer — so the kernels ran one at a time and their sum is not a
    /// pipelined pass's device time.
    ///
    /// Decides how the rows are read: in place, a row's milliseconds are its
    /// kernels' own and the clean pass splits into device and host; alone, the
    /// device times are only good as proportions and are apportioned onto the
    /// clean pass instead.
    pub timed_alone: bool,
    /// Kernel time that reached the collector under a different row from the
    /// launch it measured, or a measurement that came back empty, so no row
    /// could account for it.
    ///
    /// Each measurement is kept beside the row that counted its launch and read
    /// back once the pass has run, so this is zero unless a runtime hands one
    /// over somewhere else. Reported rather than assumed, because
    /// [`device`](Self::device) is the denominator of every apportioned
    /// millisecond in a report.
    pub unplaced: Unplaced,
    /// Whether the collected pass ran without fusion — no fusion linked, or a
    /// device whose ops do not go through it.
    ///
    /// Each pass has its own way of being read wrong, and the breakdown says
    /// which it ran. Unfused, it is a graph production does not run: what a
    /// fused kernel would do across two components is charged to each of them.
    /// Fused, it is production's graph, and a kernel that carried out several
    /// components' work is charged to their closest common parent, or outside
    /// every component where they share none — so a component whose work all
    /// fused reads zero, and the kernel that did it is listed elsewhere.
    pub unfused: bool,
}

impl Breakdown {
    /// Every kernel of the pass: the ones under a component, then the
    /// untagged ones and the ones the end of the pass ran.
    pub fn kernels(&self) -> impl Iterator<Item = &Kernel> {
        self.tree
            .iter()
            .flat_map(Node::walk)
            .flat_map(|(_, node)| &node.kernels)
            .chain(&self.untagged)
            .chain(&self.end_of_pass)
    }

    /// The kernels that launched and measured no time at all.
    ///
    /// A launch the device ran took some time, so a zero is a timing the
    /// collection lost rather than a fast kernel — and every row it sits under,
    /// every share and the device total are short by it. Distinct from
    /// [`unplaced`](Self::unplaced), which is time that arrived with no row to
    /// go to: this is a row whose time never arrived.
    pub fn untimed(&self) -> impl Iterator<Item = &Kernel> {
        self.kernels()
            .filter(|kernel| kernel.launches > 0 && kernel.total.is_zero())
    }

    /// What the device spent on this pass: every kernel, summed.
    pub fn device(&self) -> Duration {
        self.kernels().map(|kernel| kernel.total).sum()
    }

    /// How long the device waited on the host across the collected pass,
    /// between every pair of kernels: what [`Node::waited`] is apportioned
    /// by.
    pub fn waited(&self) -> Duration {
        self.tree.iter().map(Node::waited_total).sum::<Duration>() + self.waited_outside
    }

    /// The rows whose work all ran in a fused kernel recorded elsewhere: no
    /// kernel under them, and named by a kernel on a row above them or outside
    /// every component. Such a row ran nothing, so its floor is the kernel's
    /// that carried its work, and a report sets the row's own aside.
    ///
    /// Asked of the kernels rather than inferred from a zero: a component that
    /// registered no device work — views and reshapes — also has no kernel, and
    /// its zero is its time.
    pub fn fused_elsewhere(&self) -> Vec<&Node> {
        fn visit<'a>(
            breakdown: &'a Breakdown,
            node: &'a Node,
            ancestors: &mut Vec<&'a Node>,
            out: &mut Vec<&'a Node>,
        ) {
            let named_above = ancestors.iter().enumerate().any(|(at, ancestor)| {
                // What this ancestor's kernels call `node`: the components between
                // the two, then `node`. A bare label would also match a same-named
                // row under another branch of the same ancestor, and a mixed
                // stack's dense and sparse `down` are both `down`.
                let path = RowPath::new(
                    ancestors[at + 1..]
                        .iter()
                        .chain([&node])
                        .map(|row| row.label.as_str()),
                );
                ancestor
                    .kernels
                    .iter()
                    .any(|kernel| kernel.fuses.contains(&path))
            });
            // Outside every component, a kernel names a row by its path from the
            // root.
            let path = RowPath::new(
                ancestors
                    .iter()
                    .chain([&node])
                    .map(|row| row.label.as_str()),
            );
            let named_outside = breakdown
                .untagged
                .iter()
                .chain(&breakdown.end_of_pass)
                .any(|kernel| kernel.fuses.contains(&path));
            if node.device_time().is_zero() && (named_above || named_outside) {
                out.push(node);
            }
            ancestors.push(node);
            for child in &node.children {
                visit(breakdown, child, ancestors, out);
            }
            ancestors.pop();
        }
        let mut out = Vec::new();
        for root in &self.tree {
            visit(self, root, &mut Vec::new(), &mut out);
        }
        out
    }
}

/// One kernel a component launched: what it is called, how many times it ran,
/// and what the device spent on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kernel {
    /// The name the kernel gives itself, in full — shortened for display only,
    /// so two kernels that abbreviate alike are still counted apart.
    pub name: String,
    /// Launches, across every entry of the component.
    pub launches: usize,
    /// Device time across those launches.
    #[serde(with = "nanoseconds")]
    pub total: Duration,
    /// The components below the row it is recorded on whose work it carried
    /// out, in first-seen order, each by its path from that row — or from the
    /// root, for a kernel outside every component.
    ///
    /// A fused kernel is recorded on the closest common parent of the
    /// components whose work it did, and that row's own work is its by
    /// placement, so a kernel on a parent naming one child did the child's
    /// work and the parent's. Empty for a kernel that did one component's work
    /// on that component's row.
    pub fuses: Vec<RowPath>,
    /// Whether, besides the rows [`fuses`](Self::fuses) names, it carried out
    /// work the row it is recorded on registered itself — a residual add fused
    /// with its child's. That work is priced by no row below, so the kernel
    /// stands at what it took even where every row it names is priced.
    pub fuses_own: bool,
}

/// Which execution served some of a row's entries, and how many.
///
/// A row is merged on its **label**, which is its identity; the variant says
/// which of several executions ran under it. So `attention core ×9` staying one
/// row while eight entries took the tile kernel and one took tensor ops is the
/// intended reading — the alternative, a row per execution, would be two rows
/// whose times a reader has to add back together to get the component's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Variant {
    /// What the execution calls itself.
    pub name: String,
    /// Entries it served, across the whole pass.
    pub entries: usize,
}

/// One component's row: what it declared, and what it took.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    /// What the component is called — its label, or its type.
    pub label: String,
    /// Which executions served its entries, in first-seen order. Empty for a
    /// component that declares no variant, which is most of them.
    pub variants: Vec<Variant>,
    /// The call sites this row was entered from, in first-seen order.
    ///
    /// One for almost every row. More than one is a **merge**: two entries
    /// written in different places landed on one label, and the row now holds
    /// the sum of both. That is what a reader wants for a stack of layers and is also
    /// how a row comes to hold two different jobs, so the merge stands and a
    /// report names the sites, which is what points at a split when there is
    /// one to make.
    ///
    /// A site is a *function*: one entered many times is one site, so this sees
    /// two entry points and not two roles.
    pub sites: Vec<String>,
    /// The sum of what every entry declared it should cost at best, or `None`
    /// when the component declares none.
    ///
    /// Summed rather than checked for agreement because a component is legally
    /// entered at several shapes in one pass: chunked prefill runs a full chunk
    /// and a padded tail, and both are real work the row has to account for.
    pub bounds: Option<Bounds>,
    /// Traffic observed beyond [`bounds`](Self::bounds) under this component,
    /// such as the experts a sparse layer's routes read past one row's. `None`
    /// where nothing under it observed any, or where it declares no floor.
    pub observed: Option<Bounds>,
    /// How many times the pass entered it.
    pub entries: usize,
    /// The components it entered, in first-seen order.
    pub children: Vec<Node>,
    /// The kernels recorded on this row, in first-seen order: its own work, and
    /// fused kernels that carried out several of its descendants' work, which
    /// land on their closest common parent (see [`Kernel::fuses`]).
    ///
    /// A component with children of its own can still launch kernels directly
    /// — the reshapes and residual adds between them — and those are exactly
    /// the work no child accounts for.
    pub kernels: Vec<Kernel>,
    /// How long the device sat idle, in the collected pass, before the kernels
    /// this component launched directly — waiting on the host to issue them.
    ///
    /// Idle time is charged to the kernel that ended it: that launch is what
    /// the device was waiting for. It is the collected pass's, whose host also
    /// records every launch, so as an absolute it includes the instrumentation;
    /// as a share of every gap it says which components the device waited on,
    /// which is what a clean pass's host time is split by.
    #[serde(with = "nanoseconds")]
    pub waited: Duration,
}

impl Node {
    /// Every row below this one, depth-first, each with its path from here —
    /// the form a kernel on this row names it by in [`Kernel::fuses`].
    ///
    /// Not [`walk`](Self::walk), which yields a depth and not a path: two rows
    /// at one depth under different parents can share a label, and telling those
    /// apart is the whole reason a kernel records a path.
    pub fn descendants(&self) -> Vec<(RowPath, &Node)> {
        let mut out = Vec::new();
        self.descend(&RowPath::default(), &mut out);
        out
    }

    /// The rows below this one, which sits at `at`, onto `out`. See
    /// [`descendants`](Self::descendants).
    fn descend<'a>(&'a self, at: &RowPath, out: &mut Vec<(RowPath, &'a Node)>) {
        for child in &self.children {
            let path = at.child(&child.label);
            out.push((path.clone(), child));
            child.descend(&path, out);
        }
    }

    /// How long the device waited on the host before every kernel under this
    /// component, its children's included. See [`waited`](Self::waited).
    pub fn waited_total(&self) -> Duration {
        self.waited
            + self
                .children
                .iter()
                .map(|child| child.waited_total())
                .sum::<Duration>()
    }

    /// Device time in the kernels this component launched directly.
    pub fn kernel_time(&self) -> Duration {
        self.kernels.iter().map(|kernel| kernel.total).sum()
    }

    /// Device time in every kernel under this component, its children's
    /// included.
    ///
    /// **This is the only time a component has.** Nothing drains at its
    /// boundaries, so a clock around its host call would measure how fast the
    /// host registered ops, not what the device spent. The kernels are what the
    /// device actually ran, and their sum across a pass comes out within a few
    /// percent of the clean pass's wall time, which is the cross-check that says
    /// so.
    ///
    /// Zero for a component whose work all fused with another's: the kernel
    /// that did both is on their closest common parent, and counted there.
    pub fn device_time(&self) -> Duration {
        self.kernel_time()
            + self
                .children
                .iter()
                .map(|child| child.device_time())
                .sum::<Duration>()
    }

    /// The sum of this component's children's bounds, which is what it would
    /// cost with no fusion between them.
    ///
    /// A composite's own bounds assume perfect fusion *within itself* and so
    /// sits strictly below this. The gap is the intermediate traffic a perfect
    /// fuser would delete — the rung of the ladder that no faster kernel can
    /// recover.
    pub fn parts_bounds(&self) -> Option<Bounds> {
        let parts: Vec<Bounds> = self
            .children
            .iter()
            .filter_map(|child| child.bounds)
            .collect();
        (!parts.is_empty()).then(|| parts.into_iter().sum())
    }

    /// Every node under and including this one, depth-first, with its depth.
    pub fn walk(&self) -> Vec<(usize, &Node)> {
        fn descend<'a>(node: &'a Node, depth: usize, out: &mut Vec<(usize, &'a Node)>) {
            out.push((depth, node));
            for child in &node.children {
                descend(child, depth + 1, out);
            }
        }
        let mut out = Vec::new();
        descend(self, 0, &mut out);
        out
    }
}

/// Kernel time that could not be credited to a row — a timing that reached the
/// collector under a different attribution from the launch it measured, or a
/// window that came back with no measurement in it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unplaced {
    /// How many timings had nowhere to go.
    pub timings: usize,
    /// What they came to.
    #[serde(with = "nanoseconds")]
    pub total: Duration,
}

/// What a breakdown file says it is, so a reader can refuse anything else before decoding it.
const FORMAT: &str = "burn-instrument breakdown";

/// The version of the breakdown file format this build writes and reads.
///
/// Every change to what a breakdown serializes as (a field added, removed, renamed or
/// re-encoded) is a new version, so a saved pass is read back as it was collected or not at all.
pub const BREAKDOWN_FILE_VERSION: u32 = 1;

/// A breakdown as it is written to a file.
#[derive(Serialize)]
struct FileOut<'a> {
    format: &'static str,
    version: u32,
    breakdown: &'a Breakdown,
}

/// What a breakdown file declares before its breakdown is decoded.
#[derive(Deserialize)]
struct FileHeader {
    format: String,
    version: u32,
}

/// A breakdown file's contents, once its header has been accepted.
#[derive(Deserialize)]
struct FileIn {
    breakdown: Breakdown,
}

/// Why a breakdown could not be written or read back.
#[derive(Debug, thiserror::Error)]
pub enum BreakdownFileError {
    /// The file could not be written or read.
    #[error("breakdown file: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not JSON, or does not hold what its header says.
    #[error("breakdown file is malformed: {0}")]
    Malformed(#[from] serde_json::Error),
    /// The file is not a breakdown.
    #[error("not a breakdown file: its format is `{found}`")]
    NotABreakdown {
        /// The format the file declares.
        found: String,
    },
    /// The file is a breakdown in a version this build does not read.
    #[error(
        "breakdown file version {found} is not supported; this build reads version {supported}"
    )]
    UnsupportedVersion {
        /// The version the file declares.
        found: u32,
        /// The version this build reads, [`BREAKDOWN_FILE_VERSION`].
        supported: u32,
    },
}

impl Breakdown {
    /// Write this breakdown as JSON, under a header naming the format and its
    /// [`BREAKDOWN_FILE_VERSION`], so a pass collected on one machine can be read on another.
    pub fn write(&self, writer: impl Write) -> Result<(), BreakdownFileError> {
        let file = FileOut {
            format: FORMAT,
            version: BREAKDOWN_FILE_VERSION,
            breakdown: self,
        };
        serde_json::to_writer_pretty(writer, &file)?;
        Ok(())
    }

    /// Read back a breakdown [`write`](Self::write) wrote, refusing a file of another format or
    /// version before decoding its contents.
    pub fn read(mut reader: impl Read) -> Result<Self, BreakdownFileError> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        let header: FileHeader = serde_json::from_str(&text)?;
        if header.format != FORMAT {
            return Err(BreakdownFileError::NotABreakdown {
                found: header.format,
            });
        }
        if header.version != BREAKDOWN_FILE_VERSION {
            return Err(BreakdownFileError::UnsupportedVersion {
                found: header.version,
                supported: BREAKDOWN_FILE_VERSION,
            });
        }
        let file: FileIn = serde_json::from_str(&text)?;
        Ok(file.breakdown)
    }
}

/// A [`Duration`] as a whole number of nanoseconds, which any reader of the file can use.
mod nanoseconds {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        // Saturating: a pass would have to run for 584 years to overflow.
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        serializer.serialize_u64(nanoseconds)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        u64::deserialize(deserializer).map(Duration::from_nanos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A breakdown with one of everything the file carries.
    fn breakdown() -> Breakdown {
        Breakdown {
            tree: vec![Node {
                label: "attention".to_string(),
                variants: vec![Variant {
                    name: "flash".to_string(),
                    entries: 2,
                }],
                sites: vec!["model::attention::forward".to_string()],
                bounds: Some(Bounds {
                    read: 64,
                    write: 16,
                    flops: 128,
                }),
                observed: None,
                entries: 2,
                children: Vec::new(),
                kernels: vec![Kernel {
                    name: "matmul".to_string(),
                    launches: 2,
                    total: Duration::from_nanos(1_500),
                    fuses: vec![RowPath::new(["query"])],
                    fuses_own: true,
                }],
                waited: Duration::from_nanos(20),
            }],
            untagged: Vec::new(),
            end_of_pass: Vec::new(),
            waited_outside: Duration::from_nanos(7),
            host_timed: false,
            timed_alone: false,
            unplaced: Unplaced::default(),
            unfused: false,
        }
    }

    fn written(breakdown: &Breakdown) -> String {
        let mut bytes = Vec::new();
        breakdown.write(&mut bytes).expect("written");
        String::from_utf8(bytes).expect("JSON is UTF-8")
    }

    #[test]
    fn a_written_breakdown_reads_back_unchanged() {
        let breakdown = breakdown();
        let read = Breakdown::read(written(&breakdown).as_bytes()).expect("read back");
        assert_eq!(read, breakdown);
    }

    /// The format version 1 is, field for field. A change that fails this test is a new
    /// [`BREAKDOWN_FILE_VERSION`], not an edit to the expected text.
    #[test]
    fn version_one_is_pinned() {
        let expected = r#"{
  "format": "burn-instrument breakdown",
  "version": 1,
  "breakdown": {
    "tree": [
      {
        "label": "attention",
        "variants": [
          {
            "name": "flash",
            "entries": 2
          }
        ],
        "sites": [
          "model::attention::forward"
        ],
        "bounds": {
          "read": 64,
          "write": 16,
          "flops": 128
        },
        "observed": null,
        "entries": 2,
        "children": [],
        "kernels": [
          {
            "name": "matmul",
            "launches": 2,
            "total": 1500,
            "fuses": [
              [
                "query"
              ]
            ],
            "fuses_own": true
          }
        ],
        "waited": 20
      }
    ],
    "untagged": [],
    "end_of_pass": [],
    "waited_outside": 7,
    "host_timed": false,
    "timed_alone": false,
    "unplaced": {
      "timings": 0,
      "total": 0
    },
    "unfused": false
  }
}"#;
        assert_eq!(written(&breakdown()), expected);
    }

    #[test]
    fn another_version_is_refused_before_its_contents_are_decoded() {
        let file = r#"{"format": "burn-instrument breakdown", "version": 2, "breakdown": "?"}"#;
        assert!(matches!(
            Breakdown::read(file.as_bytes()),
            Err(BreakdownFileError::UnsupportedVersion {
                found: 2,
                supported: 1
            })
        ));
    }

    #[test]
    fn another_format_is_refused() {
        let file = r#"{"format": "something else", "version": 1}"#;
        assert!(matches!(
            Breakdown::read(file.as_bytes()),
            Err(BreakdownFileError::NotABreakdown { found }) if found == "something else"
        ));
    }
}
