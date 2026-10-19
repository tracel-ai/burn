//! What `#[compute]` generates, on components small enough to read in full: the row's name, the
//! execution it declares, the floor it prices, that children nest under whoever called them, and
//! that the body computes what it did before.

// A parameter a declaration reads and the body does not must not warn, and nor must a `mut`
// parameter the body mutates. Denied rather than asserted, because the absence of a warning is
// not something a test can read: `Mixer::rotate` and `Mixer::accumulate` are the cases, and an
// expansion that warns on either fails this module's compilation.
#![deny(unused_variables, unused_mut)]

use crate::recording::{entered, recording};
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

    /// No label: the row is named for the type.
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

    /// The same job reached another way, under the same label: two entries, one row.
    #[compute(label = "conv taps")]
    fn taps_at(&self, x: u32) -> u32 {
        x + 2
    }

    fn layout(&self) -> &'static str {
        "packed"
    }

    /// A row named by an argument, because the caller knows the role. The body never reads
    /// `turn`.
    #[compute(label = turn.label())]
    fn rotate(&self, x: u32, turn: Turn) -> u32 {
        x * self.scale
    }

    /// A parameter the body mutates, declared `mut` as the author wrote it.
    #[compute(label = "accumulate")]
    fn accumulate(&self, mut x: u32) -> u32 {
        x += self.scale;
        x
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

#[test]
fn a_forward_with_no_label_is_a_row_named_for_its_type() {
    let (recorder, collecting) = recording();
    let computed = Mixer { scale: 3 }.forward(7);
    drop(collecting);

    assert_eq!(computed, 21, "the body computes what it did before");
    assert_eq!(recorder.events(), ["+mixer:-:28", "-"]);
}

#[test]
fn a_mut_parameter_is_the_bodys_to_mutate() {
    let recorder = entered(|| {
        assert_eq!(Mixer { scale: 3 }.accumulate(7), 10);
    });

    assert_eq!(recorder.events(), ["+accumulate:-:-", "-"]);
}

#[test]
fn a_trait_method_can_be_a_component() {
    let recorder = entered(|| {
        assert_eq!(Mixer { scale: 3 }.step(7), 10);
    });

    assert_eq!(recorder.events(), ["+step:-:-", "-"]);
}

#[test]
fn a_label_and_a_variant_are_what_the_row_declares() {
    let recorder = entered(|| {
        Mixer { scale: 3 }.taps(7);
    });

    assert_eq!(recorder.events(), ["+conv taps:packed:-", "-"]);
}

/// An expression names what it prices, and a parameter it reads that the body does not still
/// does not warn: `_weight` keeps its underscore and `x` does not.
#[test]
fn a_floor_can_be_the_call_that_computes_it() {
    let recorder = entered(|| {
        Mixer { scale: 3 }.mix(7, 2);
    });

    assert_eq!(recorder.events(), ["+mix:-:56", "-"]);
}

#[test]
fn a_row_can_be_named_by_an_argument_the_body_ignores() {
    let mixer = Mixer { scale: 3 };
    let recorder = entered(|| {
        assert_eq!(mixer.rotate(7, Turn::Left), 21);
        mixer.rotate(7, Turn::Right);
    });

    assert_eq!(
        recorder.events(),
        ["+turn left:-:-", "-", "+turn right:-:-", "-"],
        "the label is the argument's, so one component is two rows"
    );
}

/// Two entries written in different places under one label are one row, and the row remembers
/// both sites, in the order they first ran.
#[test]
fn a_row_fed_from_two_places_records_both_sites() {
    let mixer = Mixer { scale: 3 };
    let recorder = entered(|| {
        mixer.taps(7);
        mixer.taps_at(7);
    });

    assert_eq!(
        recorder.rows(),
        ["conv taps (unpriced)", "conv taps (unpriced)"]
    );
    let sites = recorder.sites("conv taps");
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

#[test]
fn a_site_entered_repeatedly_is_recorded_once() {
    let mixer = Mixer { scale: 3 };
    let recorder = entered(|| {
        mixer.taps(1);
        mixer.taps(2);
        mixer.taps(3);
    });

    assert_eq!(recorder.rows().len(), 3, "three entries");
    assert_eq!(recorder.sites("conv taps").len(), 1, "of one site");
}

#[test]
fn a_child_nests_under_whoever_called_it() {
    let stack = MixerStack {
        mixer: Mixer { scale: 3 },
    };
    let mut computed = 0;
    let recorder = entered(|| computed = stack.forward(7));

    assert_eq!(computed, 21);
    assert_eq!(
        recorder.events(),
        ["+mixer stack:-:-", "+mixer:-:28", "-", "-"]
    );
}

#[test]
fn a_disarmed_forward_records_nothing_and_computes_the_same() {
    let recorder = entered(|| {});

    assert_eq!(Mixer { scale: 3 }.forward(7), 21);
    assert!(recorder.events().is_empty());
}
