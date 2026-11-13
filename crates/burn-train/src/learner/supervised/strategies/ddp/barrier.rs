use std::sync::{Condvar, Mutex};

/// A barrier that releases every worker waiting on it once a failed peer aborts it.
pub(crate) struct WorkerBarrier {
    workers: usize,
    state: Mutex<BarrierState>,
    released: Condvar,
}

#[derive(Default)]
struct BarrierState {
    arrived: usize,
    generation: usize,
    aborted: bool,
}

/// The barrier was aborted: a peer failed and will never reach it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Aborted;

impl WorkerBarrier {
    pub fn new(workers: usize) -> Self {
        Self {
            workers,
            state: Mutex::new(BarrierState::default()),
            released: Condvar::new(),
        }
    }

    /// Blocks until every worker arrives. Returns whether this worker arrived last.
    pub fn wait(&self) -> Result<bool, Aborted> {
        let mut state = self.state.lock().unwrap();
        if state.aborted {
            return Err(Aborted);
        }
        state.arrived += 1;
        if state.arrived == self.workers {
            state.arrived = 0;
            state.generation += 1;
            self.released.notify_all();
            return Ok(true);
        }
        let generation = state.generation;
        let state = self
            .released
            .wait_while(state, |state| {
                state.generation == generation && !state.aborted
            })
            .unwrap();
        // A round completed before the abort still counts as released.
        if state.generation == generation {
            Err(Aborted)
        } else {
            Ok(false)
        }
    }

    /// Releases the current and every later wait with [`Aborted`].
    pub fn abort(&self) {
        self.state.lock().unwrap().aborted = true;
        self.released.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    #[test]
    fn releases_every_round_with_one_leader() {
        let barrier = Arc::new(WorkerBarrier::new(2));
        let peer = {
            let barrier = barrier.clone();
            std::thread::spawn(move || [barrier.wait(), barrier.wait()])
        };
        let main = [barrier.wait(), barrier.wait()];
        let peer = peer.join().unwrap();
        for round in 0..2 {
            let leaders = [&main[round], &peer[round]]
                .into_iter()
                .filter(|result| *result == &Ok(true))
                .count();
            assert_eq!(leaders, 1);
            assert!(main[round].is_ok() && peer[round].is_ok());
        }
    }

    #[test]
    fn abort_releases_a_waiting_worker_and_later_waits() {
        let barrier = Arc::new(WorkerBarrier::new(2));
        let (sender, receiver) = mpsc::channel();
        let waiting = {
            let barrier = barrier.clone();
            std::thread::spawn(move || sender.send(barrier.wait()).unwrap())
        };
        barrier.abort();
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
            Err(Aborted)
        );
        waiting.join().unwrap();
        assert_eq!(barrier.wait(), Err(Aborted));
    }
}
