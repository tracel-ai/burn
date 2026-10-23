//! What `#[compute]` generates, on components small enough to read in full: the component's name, the
//! execution it declares, the floor it prices, that children nest under whoever called them, and
//! that the body computes what it did before.

// A parameter a declaration reads and the body does not must not warn, and nor must a `mut`
// parameter the body mutates. Denied rather than asserted, because the absence of a warning is
// not something a test can read: `Mixer::rotate` and `Mixer::accumulate` are the cases, and an
// expansion that warns on either fails this module's compilation.
#![deny(unused_variables, unused_mut, unused_braces)]

use crate::observed::{entered, installed};
use crate::{Bounds, compute};

/// What the expansion names, as a crate depending on `burn` sees it.
mod burn {
    pub use crate as instrument;
}

/// Which way a `Mixer` was asked to turn: a label, and nothing the mixing itself reads.
#[derive(Clone, Copy)]
enum Turn {
    Left,
    Right,
}

impl Turn {
    fn label(self) -> &'static str {
        match self {
            Turn::Left => "turn left",
            Turn::Right => "turn right",
        }
    }
}

/// A leaf: a plain forward, and entries that are different jobs.
struct Mixer {
    scale: u32,
}

impl Mixer {
    /// The floor for mixing `x`, taking only what it prices.
    fn priced(&self, x: u32) -> Bounds {
        Bounds {
            read: x as u64 * 6,
            write: x as u64 * 2,
            flops: 4,
        }
    }

    /// Receives every parameter of the forward by reference, as a bare path does.
    fn bounds(&self, x: &u32) -> Bounds {
        Bounds {
            read: *x as u64 * 3,
            write: *x as u64,
            flops: 2,
        }
    }

    /// No label: the component is named for the type.
    #[compute(bounds = Self::bounds)]
    fn forward(&self, x: u32) -> u32 {
        x * self.scale
    }

    /// Not the component's forward, so it says what it is, and which of two layouts served it.
    #[compute(label = "conv taps", variant = self.layout())]
    fn taps(&self, x: u32) -> u32 {
        x + 1
    }

    /// A floor written as the call that computes it.
    #[compute(label = "mix", bounds = self.priced(x))]
    fn mix(&self, x: u32, _weight: u32) -> u32 {
        x * self.scale
    }

    /// The same job reached another way, under the same label: two entries, one component.
    #[compute(label = "conv taps")]
    fn taps_at(&self, x: u32) -> u32 {
        x + 2
    }

    fn layout(&self) -> &'static str {
        "packed"
    }

    /// A component named by an argument, because the caller knows the role. The body never reads
    /// `turn`.
    #[compute(label = turn.label())]
    fn rotate(&self, x: u32, turn: Turn) -> u32 {
        x * self.scale
    }

    /// A floor no entry changes, declared once.
    const FLOOR: Bounds = Bounds {
        read: 5,
        write: 1,
        flops: 0,
    };

    /// Consumes the component: a bare-path floor is handed `&self`, so pricing it leaves
    /// `self` to the body.
    #[compute(label = "consume", bounds = Self::bounds)]
    fn consume(self, x: u32) -> u32 {
        x * self.scale
    }

    /// Declines a floor by naming none.
    #[compute(label = "decline", bounds = None)]
    fn decline(&self, x: u32) -> u32 {
        x + self.scale
    }

    /// A floor that is a constant, not a function to call.
    #[compute(label = "fixed", bounds = Self::FLOOR)]
    fn fixed(&self, x: u32) -> u32 {
        x + self.scale
    }

    /// A floor its caller already priced, handed in as a parameter.
    #[compute(label = "given", bounds = floor)]
    fn given(&self, x: u32, floor: Bounds) -> u32 {
        x + self.scale
    }

    /// A parameter the body mutates, declared `mut` as the author wrote it.
    #[compute(label = "accumulate")]
    fn accumulate(&self, mut x: u32) -> u32 {
        x += self.scale;
        x
    }

    /// A body on one line, as an author may write it: kept on one line so the expansion is checked
    /// against `unused_braces`.
    #[rustfmt::skip]
    #[compute(label = "single")]
    fn single(&self, x: u32) -> u32 { x * self.scale }

    /// A parameter whose name the expansion could use for its own binding.
    #[compute(label = observing)]
    fn named_by(&self, x: u32, observing: &str) -> u32 {
        x + self.scale
    }
}

/// A component whose forward is a trait's.
trait Step {
    fn step(&self, x: u32) -> u32;
}

impl Step for Mixer {
    #[compute(label = "step")]
    fn step(&self, x: u32) -> u32 {
        x + self.scale
    }
}

/// A composite, to show the tree is a consequence of composition.
struct MixerStack {
    mixer: Mixer,
}

impl MixerStack {
    #[compute]
    fn forward(&self, x: u32) -> u32 {
        self.mixer.forward(x)
    }
}

/// An annotated forward with no arguments is a component named for its type, priced by its bounds
/// function, and computes what its body computes.
#[test]
fn a_forward_with_no_label_is_a_component_named_for_its_type() {
    let (seen, observation) = installed();
    let computed = Mixer { scale: 3 }.forward(7);
    drop(observation);

    assert_eq!(computed, 21, "the body computes what it did before");
    assert_eq!(seen.events(), ["+mixer:-:28", "-"]);
}

/// A one-line body runs as written, and its braces are the author's, not a nested block rustc
/// would call unnecessary (the module denies `unused_braces`).
#[test]
fn a_one_line_body_runs_as_written() {
    let seen = entered(|| assert_eq!(Mixer { scale: 3 }.single(7), 21));

    assert_eq!(seen.events(), ["+single:-:-", "-"]);
}

/// The declaration reads the method's parameters as the author named them, whatever names the
/// expansion binds for itself.
#[test]
fn a_declaration_reads_the_parameter_it_names() {
    let seen = entered(|| assert_eq!(Mixer { scale: 3 }.named_by(7, "query"), 10));

    assert_eq!(seen.events(), ["+query:-:-", "-"]);
}

/// A bare-path floor is priced from `&self`, so a component that consumes itself still runs its
/// body with `self` intact.
#[test]
fn a_component_that_consumes_itself_is_priced_by_reference() {
    let seen = entered(|| {
        assert_eq!(Mixer { scale: 3 }.consume(7), 21);
    });

    assert_eq!(seen.events(), ["+consume:-:28", "-"]);
}

/// A path that names a value is that value, not a function to call: a component can decline a
/// floor with `None`, declare a constant, or take one its caller priced.
#[test]
fn a_floor_can_be_a_value_named_by_a_path() {
    let mixer = Mixer { scale: 3 };
    let given = Bounds {
        read: 2,
        write: 2,
        flops: 0,
    };
    let seen = entered(|| {
        mixer.decline(7);
        mixer.fixed(7);
        mixer.given(7, given);
    });

    assert_eq!(
        seen.events(),
        ["+decline:-:-", "-", "+fixed:-:6", "-", "+given:-:4", "-"]
    );
}

/// A `mut` parameter stays the body's to mutate, with nothing left for the expansion to warn
/// about.
#[test]
fn a_mut_parameter_is_the_bodys_to_mutate() {
    let seen = entered(|| {
        assert_eq!(Mixer { scale: 3 }.accumulate(7), 10);
    });

    assert_eq!(seen.events(), ["+accumulate:-:-", "-"]);
}

/// A component's forward can be a trait method, the shape most models' layers take.
#[test]
fn a_trait_method_can_be_a_component() {
    let seen = entered(|| {
        assert_eq!(Mixer { scale: 3 }.step(7), 10);
    });

    assert_eq!(seen.events(), ["+step:-:-", "-"]);
}

/// A label and a variant written in the attribute are what the entry declares.
#[test]
fn a_label_and_a_variant_are_what_the_component_declares() {
    let seen = entered(|| {
        Mixer { scale: 3 }.taps(7);
    });

    assert_eq!(seen.events(), ["+conv taps:packed:-", "-"]);
}

/// An expression names what it prices, and a parameter it reads that the body does not still
/// does not warn: `_weight` keeps its underscore and `x` does not.
#[test]
fn a_floor_can_be_the_call_that_computes_it() {
    let seen = entered(|| {
        Mixer { scale: 3 }.mix(7, 2);
    });

    assert_eq!(seen.events(), ["+mix:-:56", "-"]);
}

/// A label can come from an argument the body never reads, so a caller can name the role
/// without the body warning about it.
#[test]
fn a_component_can_be_named_by_an_argument_the_body_ignores() {
    let mixer = Mixer { scale: 3 };
    let seen = entered(|| {
        assert_eq!(mixer.rotate(7, Turn::Left), 21);
        mixer.rotate(7, Turn::Right);
    });

    assert_eq!(
        seen.events(),
        ["+turn left:-:-", "-", "+turn right:-:-", "-"],
        "the label is the argument's, so one function is two components"
    );
}

/// Two entries written in different places under one label are one component, and the component remembers
/// both sites, in the order they first ran.
#[test]
fn a_component_fed_from_two_places_records_both_sites() {
    let mixer = Mixer { scale: 3 };
    let seen = entered(|| {
        mixer.taps(7);
        mixer.taps_at(7);
    });

    assert_eq!(
        seen.lines(),
        ["conv taps (unpriced)", "conv taps (unpriced)"]
    );
    let sites = seen.sites("conv taps");
    assert!(
        sites.iter().all(|site| site.starts_with(module_path!())),
        "a site names the module the entry is written in: {sites:?}"
    );
    let entries: Vec<&str> = sites
        .iter()
        .map(|site| site.rsplit("::").next().expect("a function name"))
        .collect();
    assert_eq!(entries, ["taps", "taps_at"]);
}

/// A site is recorded once however many times it is entered: what a reader acts on is which
/// places fed a component.
#[test]
fn a_site_entered_repeatedly_is_recorded_once() {
    let mixer = Mixer { scale: 3 };
    let seen = entered(|| {
        mixer.taps(1);
        mixer.taps(2);
        mixer.taps(3);
    });

    assert_eq!(seen.lines().len(), 3, "three entries");
    assert_eq!(seen.sites("conv taps").len(), 1, "of one site");
}

/// Nesting follows composition: a forward that calls its child's forward is its parent, with
/// nothing declaring the tree.
#[test]
fn a_child_nests_under_whoever_called_it() {
    let stack = MixerStack {
        mixer: Mixer { scale: 3 },
    };
    let mut computed = 0;
    let seen = entered(|| computed = stack.forward(7));

    assert_eq!(computed, 21);
    assert_eq!(seen.events(), ["+mixer stack:-:-", "+mixer:-:28", "-", "-"]);
}

/// With no observer installed, an annotated forward records nothing and computes what its body
/// computes: the annotation costs a production run nothing it would notice.
#[test]
fn an_unobserved_forward_records_nothing_and_computes_the_same() {
    let seen = entered(|| {});

    assert_eq!(Mixer { scale: 3 }.forward(7), 21);
    assert!(seen.events().is_empty());
}
