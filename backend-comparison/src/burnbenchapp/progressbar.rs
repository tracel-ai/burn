use core::fmt;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use indicatif::{style::ProgressTracker, ProgressBar, ProgressState, ProgressStyle};

pub(crate) struct RunnerProgressBar {
    pb: ProgressBar,
    tracker: ThreadSafeTracker,
}

impl RunnerProgressBar {
    pub(crate) fn new(total: u64) -> Self {
        let tracker = CountTracker::default();
        let thread_safe_tracker = ThreadSafeTracker::new(tracker);
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::with_template(
                "\n{msg}\n{wide_bar:.yellow/red}  {pos}/{len}  ({counter})",
            )
            .unwrap()
            .with_key("counter", thread_safe_tracker.clone())
            .progress_chars("▬▶―"),
        );
        Self {
            pb,
            tracker: thread_safe_tracker,
        }
    }

    pub(crate) fn message(&self, msg: String) {
        self.pb.set_message(msg);
    }

    pub(crate) fn inc_by_one(&self) {
        self.pb.inc(1);
    }

    pub(crate) fn successed_inc(&mut self) {
        self.tracker.inner.lock().unwrap().successed += 1;
    }

    pub(crate) fn failed_inc(&mut self) {
        self.tracker.inner.lock().unwrap().failed += 1;
    }

    pub(crate) fn finish(&self) {
        let success = self.tracker.inner.lock().unwrap().failed == 0;
        let msg = format!(
            "\n{{msg}}\n{{wide_bar:.{}}}",
            if success { "green" } else { "red" }
        );
        self.pb.set_style(
            ProgressStyle::with_template(&msg)
                .unwrap()
                .progress_chars("▬▶―"),
        );
        self.pb.finish_with_message(if success {
            "Benchmarks Complete!"
        } else {
            "Some benchmarks failed!"
        });
    }
}

#[derive(Clone, Default)]
struct CountTracker {
    pub failed: u64,
    pub successed: u64,
}

#[derive(Clone)]
struct ThreadSafeTracker {
    inner: Arc<Mutex<CountTracker>>,
}

impl ThreadSafeTracker {
    pub fn new(tracker: CountTracker) -> Self {
        Self {
            inner: Arc::new(Mutex::new(tracker)),
        }
    }
}

impl ProgressTracker for ThreadSafeTracker {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(Self {
            inner: Arc::clone(&self.inner),
        })
    }

    fn tick(&mut self, _: &ProgressState, _: Instant) {}

    fn reset(&mut self, _: &ProgressState, _: Instant) {}

    fn write(&self, _state: &ProgressState, w: &mut dyn fmt::Write) {
        let tracker = self.inner.lock().unwrap();
        write!(w, "{}✅ {}❌", tracker.successed, tracker.failed).unwrap();
    }
}
