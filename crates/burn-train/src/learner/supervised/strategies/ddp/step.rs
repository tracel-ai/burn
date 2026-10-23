use std::sync::{Barrier, Mutex};

use burn_core::data::dataset::DatasetError;

use crate::Interrupter;

/// Admits a backward pass only when every worker has a batch and agrees to continue.
pub(crate) struct StepCoordinator {
    workers: usize,
    barrier: Barrier,
    state: Mutex<StepState>,
}

#[derive(Default)]
struct StepState {
    ready: usize,
    proceed: bool,
}

impl StepCoordinator {
    pub fn new(workers: usize) -> Self {
        Self {
            workers,
            barrier: Barrier::new(workers),
            state: Mutex::new(StepState::default()),
        }
    }

    /// Loads each worker's next batch, then publishes one decision for the whole group.
    /// No worker may independently stop after this returns a batch: its peers may already
    /// be waiting for it in a collective. Uneven exhaustion is an error, not a partial step.
    pub fn next<T>(
        &self,
        iterator: &mut impl Iterator<Item = Result<T, DatasetError>>,
        interrupter: &Interrupter,
    ) -> Option<T> {
        let item = if interrupter.should_stop() {
            None
        } else {
            match iterator.next() {
                Some(Ok(item)) => Some(item),
                Some(Err(error)) => {
                    interrupter.fail(error);
                    None
                }
                None => None,
            }
        };
        if item.is_some() {
            self.state.lock().unwrap().ready += 1;
        }

        if self.barrier.wait().is_leader() {
            let mut state = self.state.lock().unwrap();
            if !interrupter.should_stop() && state.ready != 0 && state.ready != self.workers {
                interrupter.fail(DatasetError::new(std::io::Error::other(
                    "DDP workers must yield the same number of training batches",
                )));
            }
            state.proceed = !interrupter.should_stop() && state.ready == self.workers;
            state.ready = 0;
        }
        self.barrier.wait();

        // The next round cannot change this decision until every worker has read it
        // and reached the first barrier again.
        if self.state.lock().unwrap().proceed {
            item
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, mpsc},
        time::Duration,
    };

    #[test]
    fn workers_exhaust_together_and_can_start_another_epoch() {
        let steps = Arc::new(StepCoordinator::new(2));
        let interrupter = Interrupter::new();
        for len in [3, 0, 2] {
            let expected: Vec<_> = (0..len).collect();
            for batches in run_epoch(&steps, &interrupter, [len; 2]) {
                assert_eq!(batches, expected);
            }
        }
        assert!(!interrupter.should_stop());
    }

    #[test]
    fn uneven_exhaustion_stops_all_workers_before_an_unmatched_backward() {
        let steps = Arc::new(StepCoordinator::new(2));
        let interrupter = Interrupter::new();
        for batches in run_epoch(&steps, &interrupter, [1, 2]) {
            assert_eq!(batches, [0]);
        }
        assert!(
            interrupter
                .error()
                .unwrap()
                .to_string()
                .contains("same number of training batches")
        );
    }

    fn run_epoch(
        steps: &Arc<StepCoordinator>,
        interrupter: &Interrupter,
        lengths: [usize; 2],
    ) -> Vec<Vec<usize>> {
        let (sender, receiver) = mpsc::channel();
        let workers: Vec<_> = lengths
            .into_iter()
            .map(|len| {
                let steps = steps.clone();
                let interrupter = interrupter.clone();
                let sender = sender.clone();
                std::thread::spawn(move || {
                    let mut batches = (0..len).map(Ok);
                    let admitted: Vec<_> =
                        std::iter::from_fn(|| steps.next(&mut batches, &interrupter)).collect();
                    sender.send(admitted).unwrap();
                })
            })
            .collect();
        let batches = (0..2)
            .map(|_| receiver.recv_timeout(Duration::from_secs(5)).unwrap())
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        batches
    }
}
