//! What the device spent on each launch: the measurements a run hands over
//! unread, and what they say about the run once read back.

use std::time::{Duration, Instant};

use cubecl::profile::{ProfileDuration, ProfileTicks, TimingMethod};

use super::cursor::Attribution;
use crate::breakdown::Unplaced;

/// The measurements a run kept, and how far they are to be trusted.
#[derive(Default)]
pub struct Timings {
    /// Measurements handed over unread, each beside the component its launch was
    /// counted on.
    kept: Vec<Kept>,
    /// Whether any launch was timed on the host rather than the device.
    ///
    /// One is enough to taint the totals: a host timing is wall around a
    /// blocking submit, so it carries submission and sync and is several times
    /// the kernel it names.
    host_timed: bool,
    /// Whether any launch's measurement was read back as it launched, which
    /// runs the kernels one at a time. See [`Breakdown::timed_alone`](crate::Breakdown::timed_alone).
    timed_alone: bool,
    /// Kernel time that could not be credited to a component, and how many timings
    /// that was.
    ///
    /// A measurement is filed under the attribution of the thread it arrives
    /// on, and the runtime hands it over on the thread that issued the launch,
    /// inside the same call — so it finds the component its launch was counted on.
    /// Anything else has nowhere to go, nor does a window that comes back with
    /// no measurement in it, and [`Breakdown::total_time`](crate::Breakdown::total_time)
    /// is the denominator of every share, so a silent drop would move every
    /// number read off it. Counted and reported, like a host timing.
    unplaced: Unplaced,
}

impl Timings {
    /// Keep a launch's measurement unread, beside the component that counted it.
    pub fn keep(
        &mut self,
        attribution: Attribution,
        kernel: &'static str,
        profile: ProfileDuration,
    ) {
        self.host_timed |= !matches!(profile.timing_method(), TimingMethod::Device);
        self.kept.push(Kept {
            attribution,
            kernel,
            profile,
        });
    }

    /// A measurement was read back as its kernel launched, timed by `method`:
    /// that kernel ran alone.
    pub fn read_at_launch(&mut self, method: TimingMethod) {
        self.host_timed |= !matches!(method, TimingMethod::Device);
        self.timed_alone = true;
    }

    /// Every measurement kept so far, to read back.
    pub fn take_kept(&mut self) -> Vec<Kept> {
        std::mem::take(&mut self.kept)
    }

    /// A timing of `duration` that no component could account for.
    pub fn unplace(&mut self, duration: Duration) {
        self.unplaced.timings += 1;
        self.unplaced.total += duration;
    }

    pub fn quality(&self) -> TimingQuality {
        TimingQuality {
            host_timed: self.host_timed,
            // A host timer syncs at both ends of every window, so a launch it
            // timed ran alone too.
            timed_alone: self.timed_alone || self.host_timed,
            unplaced: self.unplaced,
        }
    }
}

/// What a breakdown says about how far its timings are to be trusted. See the
/// fields of [`Breakdown`](crate::Breakdown) of the same names. The default is
/// a run whose every timing was taken on the device, in a full queue, and
/// found its component.
#[derive(Clone, Copy, Default)]
pub struct TimingQuality {
    pub host_timed: bool,
    pub timed_alone: bool,
    pub unplaced: Unplaced,
}

/// A launch's measurement, not yet read back, and the component it belongs to.
pub struct Kept {
    attribution: Attribution,
    kernel: &'static str,
    profile: ProfileDuration,
}

impl Kept {
    /// Read the measurement back. Waits for its kernel, so a run reads them
    /// once it has run.
    pub fn resolve(self) -> Measured {
        Measured {
            attribution: self.attribution,
            kernel: self.kernel,
            ticks: cubecl::future::block_on(self.profile.resolve()),
        }
    }
}

/// A measurement read back, or `None` for a window that came back empty.
pub struct Measured {
    pub attribution: Attribution,
    pub kernel: &'static str,
    pub ticks: Option<ProfileTicks>,
}

/// The windows the device ran the run's kernels in, on one clock.
pub struct Timeline {
    /// Before the run, so every window can be placed after it and the gaps
    /// between them read off.
    epoch: Instant,
    windows: Vec<Window>,
}

/// One kernel's window on the device, placed after the timeline's epoch.
struct Window {
    attribution: Attribution,
    start: Duration,
    end: Duration,
}

impl Timeline {
    pub fn new(epoch: Instant) -> Self {
        Self {
            epoch,
            windows: Vec::new(),
        }
    }

    /// Place the window `ticks` measured for a kernel on the component under
    /// `attribution`.
    pub fn push(&mut self, attribution: Attribution, ticks: &ProfileTicks) {
        self.windows.push(Window {
            attribution,
            start: ticks.start_duration_since(self.epoch),
            end: ticks.end_duration_since(self.epoch),
        });
    }

    /// Every stretch the device sat idle, each beside the component whose kernel
    /// ended it: that launch is what the device was waiting for.
    ///
    /// In start order rather than launch order: the fusion worker and the
    /// model thread both launch, and the device runs them as they arrive. A
    /// window that starts before the last one ended overlapped it, and the
    /// device was not idle for it at all.
    pub fn idle(mut self) -> Vec<(Attribution, Duration)> {
        self.windows.sort_by_key(|window| window.start);
        let mut idle = Vec::with_capacity(self.windows.len());
        let mut busy_until: Option<Duration> = None;
        for window in self.windows {
            if let Some(busy) = busy_until {
                idle.push((window.attribution, window.start.saturating_sub(busy)));
            }
            busy_until = Some(busy_until.map_or(window.end, |busy| busy.max(window.end)));
        }
        idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device's idle stretches are charged to the kernel that ended each:
    /// that launch is what it was waiting for. A kernel that starts before
    /// the last one finished overlapped it, and the device waited for nothing.
    #[test]
    fn idle_time_is_charged_to_the_kernel_that_ended_it() {
        let epoch = Instant::now();
        let mut timeline = Timeline::new(epoch);
        let mut tree = super::super::component_tree::ComponentTree::default();
        let mut component = |label| {
            let component = tree.enter(crate::Declaration {
                label,
                variant: None,
                site: "test::forward",
                bounds: None,
            });
            tree.exit();
            component
        };
        let (norm, matmul) = (component("norm"), component("matmul"));
        let window = |start: u64, end: u64| {
            let micros = Duration::from_micros;
            ProfileTicks::from_start_end(epoch + micros(start), epoch + micros(end))
        };
        // Out of order on purpose: the device runs them by start time.
        timeline.push(matmul, &window(15, 20));
        timeline.push(norm, &window(0, 10));
        timeline.push(Attribution::Untagged, &window(18, 30));
        timeline.push(norm, &window(40, 50));

        let micros = Duration::from_micros;
        assert_eq!(
            timeline.idle(),
            [
                (matmul, micros(5)),
                (Attribution::Untagged, Duration::ZERO),
                (norm, micros(10)),
            ],
            "idle from 10 to 15 before the matmul, none before the untagged \
             kernel that overlapped it, and from 30 to 40 before the norm"
        );
    }
}
