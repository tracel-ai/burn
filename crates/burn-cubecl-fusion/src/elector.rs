//! The routines the backend's unfused autotunes elected, as a fused operation reads them
//! ([`RoutineElector`]).
//!
//! The autotunes of cubek's routines are the backend's, which depends on this crate, so a fused
//! operation cannot reach their tuners; the backend installs an elector that reads them instead.

use cubecl::{Device, client::Client};
use cubek::reduce::kernels::{ReduceCandidate, ReduceKey};
use std::sync::OnceLock;

/// The elector the backend installed ([`RoutineElector::install`]).
static ELECTOR: OnceLock<&'static dyn RoutineElector> = OnceLock::new();

/// Reads which routine the backend's unfused autotune elected for a problem, so a fused operation
/// launches the routine the unfused one settled.
///
/// # Guarantees
///
/// Where no elector is installed, no routine is elected for any problem, and every fused
/// operation that launches an elected routine runs its fallback instead.
pub trait RoutineElector: Send + Sync {
    /// The reduce routine, at its setting, the unfused reduce's autotune elected for the
    /// reductions `key` describes on `client`'s `device`: `None` where no round in this process
    /// has settled the key, or the floor won it.
    fn elect_reduce(
        &self,
        client: &Client,
        device: &Device,
        key: &ReduceKey,
    ) -> Option<ReduceCandidate>;

    /// Make this the elector every fused operation reads, for the rest of the process. The first
    /// install holds; a later one changes nothing.
    fn install(&'static self)
    where
        Self: Sized,
    {
        let _ = ELECTOR.set(self);
    }
}

/// The reduce routine the installed elector reads as elected for `key`
/// ([`RoutineElector::elect_reduce`]), or `None` where none is installed.
pub(crate) fn elected_reduce(
    client: &Client,
    device: &Device,
    key: &ReduceKey,
) -> Option<ReduceCandidate> {
    ELECTOR.get()?.elect_reduce(client, device, key)
}
