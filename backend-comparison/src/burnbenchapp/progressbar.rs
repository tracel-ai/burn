use core::fmt;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use indicatif::{style::ProgressTracker, ProgressBar, ProgressState, ProgressStyle};

pub(crate) struct RunnerProgressBar {
    pb: ProgressBar,
    succeeded: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
}

impl RunnerProgressBar {
    pub(crate) fn new(total: u64) -> Self {
        let pb = ProgressBar::new(total);
        let succeeded = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicU64::new(0));
        pb.set_style(
            ProgressStyle::default_spinner()
                .template(
                    "\n{msg}\n{spinner}{wide_bar:.yellow/red} {pos}/{len} {succeeded} {failed}\n ",
                )
                .unwrap()
                .with_key("succeeded", CountTracker::new(succeeded.clone(), '✅'))
                .with_key("failed", CountTracker::new(failed.clone(), '❌'))
                .progress_chars("▬▬―")
                .tick_strings(&[
                    "🕛 ", "🕐 ", "🕑 ", "🕒 ", "🕓 ", "🕔 ", "🕕 ", "🕖 ", "🕗 ", "🕘 ", "🕙 ",
                    "🕚 ",
                ]),
        );
        Self {
            pb,
            succeeded: succeeded.clone(),
            failed: failed.clone(),
        }
    }

    pub(crate) fn message(&self, msg: String) {
        self.pb.set_message(msg);
    }

    pub(crate) fn advance_spinner(&self) {
        self.pb.tick();
    }

    /// make the spinner to spin automatically
    pub(crate) fn start_spinner(&self) {
        self.pb.enable_steady_tick(Duration::from_millis(100));
    }

    /// stop the spinner to spin automatically
    pub(crate) fn stop_spinner(&self) {
        self.pb.disable_steady_tick();
    }

    pub(crate) fn inc_by_one(&self) {
        self.pb.inc(1);
    }

    pub(crate) fn succeeded_inc(&mut self) {
        self.succeeded.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn failed_inc(&mut self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn finish(&self) {
        let success = self.failed.load(Ordering::SeqCst) == 0;
        let msg = format!(
            "\n{{msg}}\n{{wide_bar:.{}}}",
            if success { "green" } else { "red" }
        );
        self.pb.set_style(
            ProgressStyle::with_template(&msg)
                .unwrap()
                .progress_chars("▬▬―"),
        );
        self.pb.finish_with_message(if success {
            "Benchmarks Complete!"
        } else {
            "Some benchmarks failed!"
        });
    }
}

#[derive(Clone)]
struct CountTracker {
    count: Arc<AtomicU64>,
    icon: char,
}

impl CountTracker {
    pub fn new(count: Arc<AtomicU64>, icon: char) -> Self {
        Self { count, icon }
    }
}

impl ProgressTracker for CountTracker {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(self.clone())
    }

    fn tick(&mut self, _: &ProgressState, _: Instant) {}

    fn reset(&mut self, _: &ProgressState, _: Instant) {}

    fn write(&self, _state: &ProgressState, w: &mut dyn fmt::Write) {
        write!(w, "{}{}", self.count.load(Ordering::Relaxed), self.icon).unwrap();
    }
}
