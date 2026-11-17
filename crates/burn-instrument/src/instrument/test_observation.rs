//! The process-wide observer slot, shared by the tests that install an observer.

use std::sync::{Arc, Mutex, MutexGuard};

use cubecl::logging::{
    Communication, Execution, ExecutionId, ExecutionKind, ExecutionObservation, ExecutionObserver,
    ExecutionScope, TransferRoute,
};
use cubecl_common::device::{DeviceId, ServiceId};

/// One observation at a time, in a process where every test runs alongside the others.
static SERIAL: Mutex<()> = Mutex::new(());

pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `N` execution ids, in the order cubecl minted them.
///
/// cubecl mints an id only for what it issues while observed, so each comes from a scope opened
/// only to mint it; a test then builds the executions it feeds an observer around them.
pub fn minted<const N: usize>() -> [ExecutionId; N] {
    let _serial = serial();
    let minted = Arc::new(Minted::default());
    let observation = ExecutionObservation::new(minted.clone());
    let anywhere = ServiceId::of::<()>(DeviceId::new(0, 0));
    for _ in 0..N {
        drop(ExecutionScope::communication(|| Communication::Transfer {
            from: anywhere,
            to: anywhere,
            bytes: 0,
            route: TransferRoute::Host,
        }));
    }
    drop(observation);
    let ids = minted.0.lock().expect("no test panicked minting").clone();
    ids.try_into().expect("one id per scope")
}

#[derive(Default)]
struct Minted(Mutex<Vec<ExecutionId>>);

impl ExecutionObserver for Minted {
    fn issued(&self, execution: &Execution) {
        if let ExecutionKind::Communication(_) = execution.kind {
            self.0
                .lock()
                .expect("no test panicked minting")
                .push(execution.id);
        }
    }
}
