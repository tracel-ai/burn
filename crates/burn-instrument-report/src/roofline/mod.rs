//! Scoring a component against what the device can do.
//!
//! A [`Bounds`](burn_instrument::Bounds) says what a component must move and compute at best; the device's [`Peaks`](burn_instrument::Peaks) say
//! how fast either can go. The longer of the two floors is the **floor**, the time the component
//! cannot go below, and the wall that produced it is the one it is bound by: taking the max finds
//! the binding constraint and names it in one step.
//!
//! The bytes floor charges each direction at its own ceiling, `read / read_peak + write /
//! write_peak`, because almost no component splits its traffic evenly and a single mixed figure
//! is wrong both ways at once. Where a device reports no per-direction ceilings, the copy ceiling
//! over the total is the fallback.
//!
//! Every floor is a floor for the component **in isolation**. A composite's therefore sits below
//! the sum of its children's, and the difference is fusion headroom rather than an error: see
//! [`Ladder`]. `flops` is logical arithmetic, so a quantized component's dequantization is absent
//! from it.

mod base;
mod run;

pub use base::*;
pub(crate) use run::*;
