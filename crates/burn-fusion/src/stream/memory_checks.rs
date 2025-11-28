use hashbrown::HashMap;
use std::{
    fmt::Display,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::SyncSender,
    },
    thread::JoinHandle,
    time::Duration,
};

use burn_ir::{HandleContainer, TensorId, TensorStatus};
use burn_std::id::StreamId;

use crate::FusionRuntime;

use super::Stream;

/// Memory checks struct to validate there is no memory leak with the fusion runtime.
#[derive(Clone)]
pub(crate) struct MemoryChecks {
    sender: SyncSender<Message>,
    num_queued: Arc<AtomicU64>,
    // Keeps track of its thread.
    _handle: Arc<JoinHandle<()>>,
}

enum Message {
    Register(StreamAnalyses),
    Check(SyncSender<MemoryReport>),
}

enum MemoryReport {
    Success,
    NotReady,
    NotStarted,
    Fail(String),
}

#[derive(Default)]
struct StreamAnalyses {
    streams: HashMap<StreamId, Analysis>,
    num_handles: usize,
}

impl Display for StreamAnalyses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("\n==== Fusion Memory Report ====\n")?;
        f.write_fmt(format_args!(" - Handles: {}\n", self.num_handles))?;
        f.write_fmt(format_args!(" - Streams: {}\n", self.streams.len()))?;

        for (id, analysis) in self.streams.iter() {
            f.write_fmt(format_args!(
                "  - {} => operations: {} cursor: {}\n",
                id, analysis.num_operations, analysis.cursor
            ))?;
            for (tid, (origin, status)) in analysis.variables.iter() {
                f.write_fmt(format_args!(
                    "   - {tid} => origin: {origin} status: {status:?}\n",
                ))?;
            }
        }

        f.write_str("==============================\n")
    }
}

#[derive(Default, Debug)]
struct Analysis {
    variables: HashMap<TensorId, (StreamId, TensorStatus)>,
    num_operations: usize,
    cursor: u64,
}

#[macro_export]
/// Export memory checks tests.
macro_rules! memory_checks {
    () => {
        #[cfg(test)]
        mod memory_checks {
            #[test]
            fn test_memory_leaks() {
                burn_fusion::stream::memory_checks::check_memory_leaks();
            }
        }
    };
}

static INSTANCE: spin::Mutex<Option<MemoryChecks>> = spin::Mutex::new(None);

/// Performs memory checks and panics if a leak is discovered.
pub fn check_memory_leaks() {
    let mut num_try_uninit = 0;
    let max_try = 25;

    loop {
        let report = fetch_memory_report();
        match report {
            MemoryReport::Success => return,
            MemoryReport::NotReady => {
                num_try_uninit = 0;
                std::thread::sleep(Duration::from_millis(100))
            }
            MemoryReport::NotStarted => {
                if num_try_uninit >= max_try {
                    // Nothing is running on the fusion runtime.
                    return;
                }
                num_try_uninit += 1;
                std::thread::sleep(Duration::from_millis(100))
            }
            MemoryReport::Fail(msg) => panic!("{msg}"),
        }
    }
}

fn fetch_memory_report() -> MemoryReport {
    let report = INSTANCE.lock();

    let report = match report.as_ref() {
        Some(client) => client,
        None => return MemoryReport::NotStarted,
    };

    let (sender, rec) = std::sync::mpsc::sync_channel(1);
    match report.sender.send(Message::Check(sender)) {
        Ok(_) => {}
        Err(err) => {
            panic!("Channel closed can't send the check call: {err:?}")
        }
    };

    match rec.recv() {
        Ok(report) => report,
        Err(err) => panic!("Received an error from fetching check results: {err}"),
    }
}

impl Default for MemoryChecks {
    fn default() -> Self {
        let mut instance = INSTANCE.lock();
        let result = match instance.as_mut() {
            Some(client) => client.clone(),
            None => {
                let this = Self::spawn_new();
                *instance = Some(this.clone());
                this
            }
        };
        core::mem::drop(instance);
        result
    }
}

impl MemoryChecks {
    pub(crate) fn check<R: FusionRuntime>(
        &mut self,
        streams: &HashMap<StreamId, Stream<R>>,
        handles: &HandleContainer<R::FusionHandle>,
    ) {
        let mut analyses = StreamAnalyses {
            num_handles: handles.num_handles(),
            streams: Default::default(),
        };

        for (id, s) in streams.iter() {
            let analysis = Analysis {
                variables: s.queue.variables.clone(),
                num_operations: s.queue.global.len(),
                cursor: s.cursor,
            };
            analyses.streams.insert(*id, analysis);
        }

        self.num_queued.fetch_add(1, Ordering::Relaxed);
        match self.sender.send(Message::Register(analyses)) {
            Ok(..) => {}
            Err(err) => {
                panic!("Can't register memory checks analysis: {err:?}")
            }
        }
    }

    fn spawn_new() -> Self {
        let (sender, rec) = std::sync::mpsc::sync_channel(100);
        let num_queued = Arc::new(AtomicU64::new(0));
        let num_queued_moved = num_queued.clone();

        let handle = std::thread::spawn(move || {
            let mut last_analyses = None;

            loop {
                let payload = match rec.recv() {
                    Err(_err) => {
                        // A client has panic, safe to skip as it may be normal.
                        continue;
                    }
                    Ok(payload) => payload,
                };
                match payload {
                    Message::Register(payload) => {
                        last_analyses = Some(payload);
                        num_queued_moved.fetch_sub(1, Ordering::Relaxed);
                    }
                    Message::Check(callback) => {
                        if num_queued_moved.load(Ordering::Relaxed) > 1 {
                            callback.send(MemoryReport::NotReady).unwrap();
                            continue;
                        }

                        // We assume that if nothing has been registered in the last second
                        // while being at a count of 1, it's the end.
                        std::thread::sleep(Duration::from_secs(5));

                        if num_queued_moved.load(Ordering::Relaxed) <= 1 {
                            match last_analyses.take() {
                                Some(val) => {
                                    callback.send(Self::final_check(val)).unwrap();
                                }
                                None => {
                                    callback
                                        .send(MemoryReport::Fail("No analyses".into()))
                                        .unwrap();
                                }
                            }
                        } else {
                            callback.send(MemoryReport::NotReady).unwrap();
                        }
                    }
                }
            }
        });

        Self {
            sender,
            num_queued,
            _handle: Arc::new(handle),
        }
    }

    fn final_check(analyses: StreamAnalyses) -> MemoryReport {
        if !analyses.streams.is_empty() || analyses.num_handles > 0 {
            return MemoryReport::Fail(format!("{analyses}"));
        }

        MemoryReport::Success
    }
}
