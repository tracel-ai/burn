//! The two websocket connection handlers, one per route, plus the session-layer traits they
//! depend on.
//!
//! A client opens a pair of sockets per session: a `/submit` socket carrying the stream of
//! [`RemoteMessage`](crate::shared::RemoteMessage)s the client submits, and a `/fetch` socket
//! carrying results back. [`SubmitHandler`] drives the former and [`FetchHandler`] the latter.
//! Both are otherwise plain loops; the only real state is on the submit side, where a
//! connection tracks which session it is bound to and the channel to that session's worker —
//! that state machine lives in [`policy`].
//!
//! Both handlers are written against two abstractions rather than concrete types: the socket is
//! a [`CommunicationChannel`](burn_communication::CommunicationChannel), and the session layer
//! they talk to is a [`SubmitService`] / [`FetchService`]. The production service is the
//! [`SessionManager`](super::session::SessionManager), but the traits let the submit policy be
//! exercised against a fake service with no backend and no live socket (see the [`policy`]
//! tests).

mod base;
mod fetch;
#[cfg(feature = "websocket")]
mod policy;
mod submit;

#[cfg(feature = "iroh")]
pub(crate) use base::parse_init_handshake;
#[cfg(feature = "websocket")]
pub(crate) use fetch::FetchHandler;
pub(crate) use fetch::FetchService;
#[cfg(feature = "websocket")]
pub(crate) use submit::SubmitHandler;
pub(crate) use submit::SubmitService;
