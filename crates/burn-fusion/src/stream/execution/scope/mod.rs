//! One scope around one unit of work.
//!
//! Without it, every execution site keeps the same bookkeeping by hand: decide
//! whether the inputs can be trusted, claim the write set on each failure path,
//! leave it alone on the success path, catch the panic, and hope no early
//! return forgot one of the four. The scope makes forgetting loud instead of
//! silent — there is one way in, one way out, and the claim is not something a
//! caller can omit.
//!
//! A scope is opened one of two ways, and the choice is made once, in the
//! constructor, which is why the two can never interleave:
//!
//! - work whose inputs all read cleanly **enters**, and runs;
//! - work whose input a failure claims **skips**: its write set takes that same
//!   failure one hop down, and the body never runs.
//!
//! The body is a closure rather than the scope being a guard value, because a
//! guard enforces nothing here: `#[must_use]` says nothing about a bound value
//! on a path that returns early. A closure has exactly one exit, and the scope
//! owns what happens at it.

mod base;

pub use base::*;
