//! A session's worker: it handles the session's tasks one at a time, in the order the client
//! submitted them across all its streams.
//!
//! That order is the protocol's. A tensor produced on one client stream and consumed on another
//! exists only because the producer ran first, and the interpreter panics on a missing input, so
//! a task's stream picks the [server stream](crate::server::ServerStreams) it runs on, never its
//! turn. Reads, profile measurements and remote exposures start in that order but finish when the
//! device does, so their replies can arrive out of order.
//!
//! The worker has an OS thread of its own: a same-host transfer or an all-reduce waits for other
//! sessions, and on the shared runtime enough of those waits would leave no thread for the
//! sessions they wait on. In the browser it runs on the event loop instead, so a browser peer
//! cannot host both ends of such a wait.
//!
//! Off the browser, a task that panics ends its session, while a reply whose future panics
//! answers with the panic as an error.

mod handler;
mod responder;

pub use handler::{SessionServices, spawn_session_worker};
