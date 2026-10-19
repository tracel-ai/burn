//! Where a collected pass places the kernels burn's fuser builds across
//! component boundaries.
//!
//! These components are made of plain burn operations so the fuser can
//! cross every kind of boundary a tree has — siblings, a parent and its child,
//! cousins, repeated entries of one row, a reduction, a matmul's epilogue, and
//! work from outside every component — and each [`Boundary`] states where its
//! kernels land with fusion and without.
//!
//! Which build ran is read off the collector, not a cargo feature: the device
//! decides whether a fusion server sits in front of it, so on a device without
//! one the same test checks the unfused placement: run it on a cubecl device
//! (`BURN_DEVICE`) with `fusion` for the fused half. Nothing here names a
//! kernel or reads a duration: what the fuser chooses to launch is the
//! fuser's, and the placement of whatever it launches is this crate's.
#![cfg(feature = "collector-tests")]

use std::fmt;
use std::sync::Mutex;

use burn_instrument::{
    Breakdown, Component, ComponentPath, Kernel, PassCollector, Unplaced, compute,
};
use burn_tensor::activation::{gelu, sigmoid};
use burn_tensor::{Device, Distribution, Tensor};
use cubecl::config::{TypeNameFormatLevel, type_name_format};

/// What `#[compute]` expands to, as a crate depending on `burn` sees it.
mod burn {
    pub use burn_instrument as instrument;
}

/// The runtimes' launch and fusion observers are process-wide slots, so two
/// collected passes at once would each record the other's kernels.
static SERIAL: Mutex<()> = Mutex::new(());

/// Rows and columns of every input and of the projection's weight.
const SIDE: usize = 64;

/// Uncollected passes before the collected one, so every shape is tuned.
const WARMUP: usize = 5;

/// One kind of boundary the fuser can cross, and the components that make it.
#[derive(Clone, Copy, Debug)]
enum Boundary {
    /// Several operations inside one component: the control, and the case
    /// a block built around custom kernels falls in.
    WithinOne,
    /// Two siblings' work in one kernel.
    Siblings,
    /// A parent's own operation fused with its child's.
    ParentAndChild,
    /// Two components two levels apart, whose closest common parent is not
    /// the parent of either.
    Cousins,
    /// The tail of one entry of a row fused with the head of the next entry
    /// of the same row.
    RepeatedEntries,
    /// An elementwise component fused into the reduction after it, as a
    /// log-sum-exp. The exponent has no other reader, so the fuser can read it
    /// inside the reduction instead of writing it out; and the parent's own
    /// `log` follows the sum, because burn's fuser runs a reduction nothing is
    /// written after as plain kernels, each a block of its own.
    IntoReduction,
    /// A projection's matmul fused with the activation after it.
    MatmulEpilogue,
    /// Work registered outside every component fused with work inside one.
    OutsideAnyComponent,
}

/// Where a case's work lands.
struct Expected {
    /// Unfused, every one of these rows launches kernels of its own.
    launching: &'static [&'static [&'static str]],
    /// Fused, where the kernel carrying the case's work lands.
    fused: Placement,
}

/// Where a fused kernel lands, and the components it names.
enum Placement {
    /// On the row at this path, naming these paths from it. Naming none, the
    /// whole case is one launch there: no name says the fuser ran, so the
    /// count does.
    Component {
        at: &'static [&'static str],
        fuses: &'static [&'static [&'static str]],
    },
    /// Outside every component, naming these paths from the root.
    Untagged {
        fuses: &'static [&'static [&'static str]],
    },
}

impl Boundary {
    /// The case's components, built on `device` — before the pass, so
    /// building them is not work the pass collects.
    fn build(self, device: &Device) -> Box<dyn Fn(Tensor<2>) -> Tensor<2>> {
        match self {
            Boundary::WithinOne => Box::new(|x| Chain.forward(x)),
            Boundary::Siblings => Box::new(|x| Pair::default().forward(x)),
            Boundary::ParentAndChild => Box::new(|x| Residual::default().forward(x)),
            Boundary::Cousins => Box::new(|x| Branches::default().forward(x)),
            Boundary::RepeatedEntries => Box::new(|x| Stack::default().forward(x)),
            Boundary::IntoReduction => Box::new(|x| LogSumExp::default().forward(x)),
            Boundary::MatmulEpilogue => {
                let mlp = Mlp::new(device);
                Box::new(move |x| mlp.forward(x))
            }
            // The scaling is the work outside every component.
            Boundary::OutsideAnyComponent => Box::new(|x| Inner.forward(x.mul_scalar(3.0))),
        }
    }

    fn expected(self) -> Expected {
        match self {
            Boundary::WithinOne => Expected {
                launching: &[&["chain"]],
                fused: Placement::Component {
                    at: &["chain"],
                    fuses: &[],
                },
            },
            Boundary::Siblings => Expected {
                launching: &[&["pair", "scale"], &["pair", "shift"]],
                fused: Placement::Component {
                    at: &["pair"],
                    fuses: &[&["scale"], &["shift"]],
                },
            },
            Boundary::ParentAndChild => Expected {
                launching: &[&["residual"], &["residual", "gate"]],
                fused: Placement::Component {
                    at: &["residual"],
                    fuses: &[&["gate"]],
                },
            },
            Boundary::Cousins => Expected {
                launching: &[
                    &["branches", "left", "leaf"],
                    &["branches", "right", "leaf"],
                ],
                fused: Placement::Component {
                    at: &["branches"],
                    fuses: &[&["left", "leaf"], &["right", "leaf"]],
                },
            },
            // Every entry is one row, so a kernel spanning entries did that
            // row's work alone: it lands on the row, not the stack, and names
            // nothing.
            Boundary::RepeatedEntries => Expected {
                launching: &[&["stack", "layer"]],
                fused: Placement::Component {
                    at: &["stack", "layer"],
                    fuses: &[],
                },
            },
            Boundary::IntoReduction => Expected {
                launching: &[&["log sum exp", "exponent"], &["log sum exp", "row sum"]],
                fused: Placement::Component {
                    at: &["log sum exp"],
                    fuses: &[&["exponent"], &["row sum"]],
                },
            },
            Boundary::MatmulEpilogue => Expected {
                launching: &[&["mlp", "projection"], &["mlp", "activation"]],
                fused: Placement::Component {
                    at: &["mlp"],
                    fuses: &[&["projection"], &["activation"]],
                },
            },
            Boundary::OutsideAnyComponent => Expected {
                launching: &[&["inner"]],
                fused: Placement::Untagged {
                    fuses: &[&["inner"]],
                },
            },
        }
    }
}

/// One case's collected pass, read against what the case expects.
struct CaseBreakdown {
    boundary: Boundary,
    breakdown: Breakdown,
}

impl CaseBreakdown {
    /// Run `boundary`'s components collected, on a model already warmed and
    /// settled.
    fn new(boundary: Boundary) -> Self {
        let serial = SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let device = Device::default();
        let input = Tensor::<2>::random([SIDE, SIDE], Distribution::Default, &device);
        let model = boundary.build(&device);
        // Uncollected first: the first sighting of a reduction's or a matmul's
        // shape tunes it, and every candidate the tuner times is a launch.
        // Each output is held to the sync, as the collected pass holds its
        // own: one dropped before its block runs lets the fuser leave its
        // write out, which is a different kernel tuned under a different key.
        let warmed: Vec<Tensor<2>> = (0..WARMUP).map(|_| model(input.clone())).collect();
        // Collected on settled work: building the input and the model, and
        // warming them, is not the case's.
        device.sync().expect("the device answers before the pass");
        core::mem::drop(warmed);

        let mut output = None;
        let breakdown = PassCollector::new(&device).collect(|| output = Some(model(input)));
        // Held across the collection: an output dropped inside the pass would
        // let the fuser skip the work that wrote it.
        core::mem::drop(output);
        core::mem::drop(serial);
        Self {
            boundary,
            breakdown,
        }
    }

    /// Check the pass against the case: what holds in both builds, then what
    /// holds in the one that ran.
    fn check(&self) {
        let expected = self.boundary.expected();
        // Which half ran is the device's call, so a run says which it checked.
        let unfused = self.breakdown.unfused;
        eprintln!(
            "{:?}: collected {}",
            self.boundary,
            if unfused { "unfused" } else { "fused" }
        );
        self.assert_placed();
        match unfused {
            true => self.assert_unfused(&expected),
            false => self.assert_fused(&expected),
        }
    }

    /// The row at `path`, from the roots.
    fn row(&self, path: &[&str]) -> &Component {
        let (first, rest) = path.split_first().expect("a path names a row");
        let mut node = find(&self.breakdown.tree, first);
        for label in rest {
            node = find(&node.children, label);
        }
        node
    }

    fn launches(&self) -> usize {
        self.breakdown.kernels().map(|kernel| kernel.launches).sum()
    }

    /// What holds whichever build ran: every timing found its row, no block
    /// was left to the sync ending the pass, and every name a fused kernel
    /// records is a row below the one it is on.
    fn assert_placed(&self) {
        let boundary = self.boundary;
        assert_eq!(
            self.breakdown.unplaced,
            Unplaced::default(),
            "{boundary:?}: a timing found no row"
        );
        assert!(
            self.breakdown.end_of_pass.is_empty(),
            "{boundary:?}: work that ran at the end of the pass went back to no component\n{self}"
        );
        assert!(
            self.launches() > 0,
            "{boundary:?}: the pass launched nothing"
        );
        for root in &self.breakdown.tree {
            for (_, node) in root.walk() {
                let below: Vec<ComponentPath> = node
                    .descendants()
                    .into_iter()
                    .map(|(path, _)| path)
                    .collect();
                for kernel in &node.kernels {
                    for path in &kernel.fuses {
                        assert!(
                            below.contains(path),
                            "{boundary:?}: `{}` names `{path}`, which is no row below it",
                            node.label
                        );
                    }
                }
            }
        }
        // Outside every component, a kernel names a row by its path from the
        // root.
        let rows: Vec<ComponentPath> = self
            .breakdown
            .tree
            .iter()
            .flat_map(|root| {
                let below = root.descendants().into_iter().map(|(path, _)| {
                    ComponentPath::new(std::iter::once(root.label.as_str()).chain(path.labels()))
                });
                std::iter::once(ComponentPath::new([root.label.as_str()])).chain(below)
            })
            .collect();
        for kernel in &self.breakdown.untagged {
            for path in &kernel.fuses {
                assert!(
                    rows.contains(path),
                    "{boundary:?}: an untagged kernel names `{path}`, which is no row\n{self}"
                );
            }
        }
    }

    /// Unfused, every component launches its own work and no kernel names
    /// another.
    fn assert_unfused(&self, expected: &Expected) {
        let boundary = self.boundary;
        for path in expected.launching {
            assert!(
                !self.row(path).kernels.is_empty(),
                "{boundary:?}: {path:?} launched nothing of its own unfused\n{self}"
            );
        }
        assert!(
            self.breakdown
                .kernels()
                .all(|kernel| kernel.fuses.is_empty()),
            "{boundary:?}: an unfused kernel names several components\n{self}"
        );
    }

    /// Fused, a kernel lands where [`Placement`] says naming what it says,
    /// and every row it names has no kernel of its own left — which is what
    /// says the fuser crossed the boundary, however many kernels its tuning
    /// chose to launch for the work. A placement naming nothing is the whole
    /// case in one launch.
    fn assert_fused(&self, expected: &Expected) {
        let boundary = self.boundary;
        if let Placement::Component { fuses: [], .. } = expected.fused {
            assert_eq!(
                self.launches(),
                1,
                "{boundary:?}: the case's work did not run as one kernel\n{self}"
            );
        }
        let (kernels, fuses, named): (&[Kernel], _, Vec<&Component>) = match expected.fused {
            Placement::Component { at, fuses } => {
                let named = fuses.iter().map(|path| self.row(&[at, path].concat()));
                (&self.row(at).kernels, fuses, named.collect())
            }
            Placement::Untagged { fuses } => {
                let named = fuses.iter().map(|path| self.row(path));
                (&self.breakdown.untagged, fuses, named.collect())
            }
        };
        let wanted: Vec<ComponentPath> = fuses
            .iter()
            .map(|path| ComponentPath::new(path.iter().copied()))
            .collect();
        // A kernel's names are a set, recorded in first-seen order.
        assert!(
            kernels.iter().any(|kernel| {
                kernel.fuses.len() == wanted.len()
                    && wanted.iter().all(|path| kernel.fuses.contains(path))
            }),
            "{boundary:?}: no kernel naming {:?} where it belongs\n{self}",
            printed(&wanted)
        );
        for row in named {
            assert!(
                row.kernels.is_empty(),
                "{boundary:?}: `{}` kept kernels of its own though its work was fused\n{self}",
                row.label,
            );
        }
    }
}

/// Every row and the kernels it holds, one per line, so a failure shows where
/// the work went.
impl fmt::Display for CaseBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kernel = |f: &mut fmt::Formatter<'_>, indent: &str, kernel: &Kernel| {
            writeln!(
                f,
                "{indent}· {} ×{} fuses {:?}",
                type_name_format(&kernel.name, TypeNameFormatLevel::Short),
                kernel.launches,
                printed(&kernel.fuses)
            )
        };
        for root in &self.breakdown.tree {
            for (depth, node) in root.walk() {
                let indent = "  ".repeat(depth);
                writeln!(f, "{indent}{} ×{}", node.label, node.entries)?;
                for launched in &node.kernels {
                    kernel(f, &format!("{indent}  "), launched)?;
                }
            }
        }
        for (heading, kernels) in [
            ("untagged ", &self.breakdown.untagged),
            ("end of pass ", &self.breakdown.end_of_pass),
        ] {
            for launched in kernels {
                kernel(f, heading, launched)?;
            }
        }
        Ok(())
    }
}

fn find<'a>(nodes: &'a [Component], label: &str) -> &'a Component {
    nodes
        .iter()
        .find(|node| node.label == label)
        .unwrap_or_else(|| panic!("no row `{label}` among {:?}", labels(nodes)))
}

/// Paths as a table prints them, for a failure message.
fn printed(paths: &[ComponentPath]) -> Vec<String> {
    paths.iter().map(ToString::to_string).collect()
}

fn labels(nodes: &[Component]) -> Vec<&str> {
    nodes.iter().map(|node| node.label.as_str()).collect()
}

#[test]
fn within_one_component() {
    CaseBreakdown::new(Boundary::WithinOne).check();
}

#[test]
fn across_siblings() {
    CaseBreakdown::new(Boundary::Siblings).check();
}

#[test]
fn across_a_parent_and_its_child() {
    CaseBreakdown::new(Boundary::ParentAndChild).check();
}

#[test]
fn across_cousins() {
    CaseBreakdown::new(Boundary::Cousins).check();
}

#[test]
fn across_repeated_entries_of_one_row() {
    CaseBreakdown::new(Boundary::RepeatedEntries).check();
}

#[test]
fn into_a_reduction() {
    CaseBreakdown::new(Boundary::IntoReduction).check();
}

#[test]
fn into_a_matmul_epilogue() {
    CaseBreakdown::new(Boundary::MatmulEpilogue).check();
}

#[test]
fn from_outside_every_component() {
    CaseBreakdown::new(Boundary::OutsideAnyComponent).check();
}

// The components. Each does only what its case needs the fuser to see, and
// each is named for the role its row reads as.

struct Chain;

impl Chain {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.mul_scalar(2.0).add_scalar(1.0).exp()
    }
}

#[derive(Default)]
struct Pair {
    scale: Scale,
    shift: Shift,
}

impl Pair {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.shift.forward(self.scale.forward(x))
    }
}

#[derive(Default)]
struct Scale;

impl Scale {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.mul_scalar(0.5).exp()
    }
}

#[derive(Default)]
struct Shift;

impl Shift {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.add_scalar(1.0).sqrt()
    }
}

#[derive(Default)]
struct Residual {
    gate: Gate,
}

impl Residual {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.gate.forward(x.clone()) + x
    }
}

#[derive(Default)]
struct Gate;

impl Gate {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        sigmoid(x)
    }
}

#[derive(Default)]
struct Branches {
    left: Left,
    right: Right,
}

impl Branches {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.right.forward(self.left.forward(x))
    }
}

#[derive(Default)]
struct Left {
    leaf: Leaf,
}

impl Left {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.leaf.forward(x)
    }
}

#[derive(Default)]
struct Right {
    leaf: Leaf,
}

impl Right {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.leaf.forward(x)
    }
}

#[derive(Default)]
struct Leaf;

impl Leaf {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.mul_scalar(0.9).add_scalar(0.1)
    }
}

struct Stack {
    layers: Vec<Layer>,
}

impl Stack {
    /// Entries of the one `layer` row a pass makes.
    const DEPTH: usize = 4;

    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.layers.iter().fold(x, |x, layer| layer.forward(x))
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self {
            layers: (0..Self::DEPTH).map(|_| Layer).collect(),
        }
    }
}

struct Layer;

impl Layer {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.mul_scalar(0.9).add_scalar(0.1)
    }
}

#[derive(Default)]
struct LogSumExp {
    exponent: Exponent,
    total: RowSum,
}

impl LogSumExp {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.total.forward(self.exponent.forward(x)).log()
    }
}

#[derive(Default)]
struct Exponent;

impl Exponent {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.exp()
    }
}

#[derive(Default)]
struct RowSum;

impl RowSum {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.sum_dim(1)
    }
}

struct Mlp {
    projection: Projection,
    activation: Activation,
}

impl Mlp {
    fn new(device: &Device) -> Self {
        Self {
            projection: Projection {
                weight: Tensor::random([SIDE, SIDE], Distribution::Default, device),
            },
            activation: Activation,
        }
    }

    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.activation.forward(self.projection.forward(x))
    }
}

struct Projection {
    weight: Tensor<2>,
}

impl Projection {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.weight.clone())
    }
}

struct Activation;

impl Activation {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        gelu(x)
    }
}

struct Inner;

impl Inner {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.add_scalar(1.0).exp()
    }
}
