//! How a kernel family races its candidates: the priority band that orders a round, and the
//! autotune a family builds its candidates into ([`Autotune`]), measured against the problem's
//! roofline ([`Roofline`]).
//!
//! Every autotune of cubek's routines is built here, at the level of the backend, so that a race
//! can hold combinations of backend functions; cubek states its routines, their settings and their
//! launches, and measures nothing.

mod base;
mod group;
mod kernel;
mod priority;
mod roofline;
#[cfg(test)]
mod tests;

pub use base::*;
pub use group::*;
pub use priority::*;
pub use roofline::*;
