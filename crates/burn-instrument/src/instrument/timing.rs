//! What the device spent on each launch: the timings a run hands over
//! unread, and what they say about the run once read back.

use std::time::{Duration, Instant};

use cubecl::profile::{ProfileDuration, ProfileTicks, TimingMethod};

use super::cursor::{Attribution, Member, MemberCount};
use crate::breakdown::{MemberTime, PerMember, Unplaced};

/// The timings a run kept, and how far they are to be trusted.
#[derive(Default)]
pub struct Timings {
    /// Timings handed over unread, each beside the component its launch was
    /// counted on.
    kept: Vec<Kept>,
    /// Whether any launch was timed on the host rather than the device.
    ///
    /// One is enough to taint the totals: a host timing is wall around a
    /// blocking submit, so it carries submission and sync and is several times
    /// the kernel it names.
    host_timed: bool,
    /// Whether any launch's timing was read back as it launched, which
    /// runs the kernels one at a time. See [`Breakdown::timed_alone`](crate::Breakdown::timed_alone).
    timed_alone: bool,
    /// Kernel time that could not be credited to a component, and how many timings
    /// that was.
    ///
    /// A timing is filed under the attribution of the thread it arrives
    /// on, and the runtime hands it over on the thread that issued the launch,
    /// inside the same call — so it finds the component its launch was counted on.
    /// Anything else has nowhere to go, nor does a window that comes back with
    /// no timing in it, and [`Breakdown::total_time`](crate::Breakdown::total_time)
    /// is the denominator of every share, so a silent drop would move every
    /// number read off it. Counted and reported, like a host timing.
    unplaced: Unplaced,
}

impl Timings {
    /// Keep a launch's timing unread, beside the component that counted it.
    pub fn keep(
        &mut self,
        attribution: Attribution,
        member: Option<Member>,
        kernel: &'static str,
        profile: ProfileDuration,
    ) {
        self.host_timed |= !matches!(profile.timing_method(), TimingMethod::Device);
        self.kept.push(Kept {
            attribution,
            member,
            kernel,
            profile,
        });
    }

    /// A timing was read back as its kernel launched, timed by `method`:
    /// that kernel ran alone.
    pub fn read_at_launch(&mut self, method: TimingMethod) {
        self.host_timed |= !matches!(method, TimingMethod::Device);
        self.timed_alone = true;
    }

    /// Every timing kept so far, to read back.
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

/// A launch's timing, not yet read back, and the component it belongs to.
pub struct Kept {
    attribution: Attribution,
    member: Option<Member>,
    kernel: &'static str,
    profile: ProfileDuration,
}

impl Kept {
    /// Read the timing back. Waits for its kernel, so a run reads them
    /// once it has run.
    pub fn resolve(self) -> Timed {
        Timed {
            attribution: self.attribution,
            member: self.member,
            kernel: self.kernel,
            ticks: cubecl::future::block_on(self.profile.resolve()),
        }
    }
}

/// A timing read back, or `None` for a window that came back empty.
pub struct Timed {
    pub attribution: Attribution,
    pub member: Option<Member>,
    pub kernel: &'static str,
    pub ticks: Option<ProfileTicks>,
}

/// The windows each member's device ran the run's kernels in, on one clock.
pub struct Timeline {
    /// Before the run, so every window can be placed after it and the gaps
    /// between them read off.
    epoch: Instant,
    members: MemberCount,
    windows: Vec<Window>,
}

/// One kernel's window, beside the device it ran on, placed after the timeline's epoch.
struct Window {
    attribution: Attribution,
    ran_on: RanOn,
    start: Duration,
    end: Duration,
}

/// The device a window ran on, as far as its launching thread tells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RanOn {
    /// The one device a device of one runs everything on, whichever thread launched it.
    Device,
    /// A member, launched from its fusion worker.
    Member(Member),
    /// Some member, which one unknown: launched from a thread that is no member's worker, so it
    /// fills no member's gaps and opens none.
    UnknownMember,
}

/// What a timeline says about the run once every window is on it.
pub struct Settled {
    /// Every stretch a device sat idle, beside the component whose kernel ended it.
    pub idle: Vec<(Attribution, Duration)>,
    /// What each member did, over several members.
    pub per_member: Option<PerMember>,
}

impl Timeline {
    pub fn new(epoch: Instant, members: MemberCount) -> Self {
        Self {
            epoch,
            members,
            windows: Vec::new(),
        }
    }

    /// Place the window `ticks` measured for a kernel launched for `member` on the component
    /// under `attribution`.
    pub fn push(&mut self, attribution: Attribution, member: Option<Member>, ticks: &ProfileTicks) {
        let ran_on = match (self.members.several(), member) {
            (false, _) => RanOn::Device,
            (true, Some(member)) => RanOn::Member(member),
            (true, None) => RanOn::UnknownMember,
        };
        self.windows.push(Window {
            attribution,
            ran_on,
            start: ticks.start_duration_since(self.epoch),
            end: ticks.end_duration_since(self.epoch),
        });
    }

    /// Every stretch a device sat idle, each beside the component whose kernel
    /// ended it: that launch is what the device was waiting for. And over several members, what
    /// each did.
    ///
    /// In start order rather than launch order: the fusion worker and the
    /// model thread both launch, and the device runs them as they arrive. A
    /// window that starts before the last one ended overlapped it, and the
    /// device was not idle for it at all. Each member's device is idle on its own: another
    /// member's kernel running meanwhile does not fill the gap.
    pub fn settle(mut self) -> Settled {
        self.windows
            .sort_by_key(|window| (window.ran_on, window.start));
        let mut idle = Vec::with_capacity(self.windows.len());
        let mut per_member = self.members.several().then(|| PerMember {
            members: vec![MemberTime::default(); self.members.0],
            off_members: Duration::ZERO,
        });
        let mut busy_until: Option<(RanOn, Duration)> = None;
        for window in &self.windows {
            let length = window.end.saturating_sub(window.start);
            let gap = match busy_until {
                Some((ran_on, busy)) if ran_on == window.ran_on => {
                    Some(window.start.saturating_sub(busy))
                }
                _ => None,
            };
            busy_until = Some(match busy_until {
                Some((ran_on, busy)) if ran_on == window.ran_on => (ran_on, busy.max(window.end)),
                _ => (window.ran_on, window.end),
            });
            match (window.ran_on, per_member.as_mut()) {
                (RanOn::UnknownMember, Some(per_member)) => per_member.off_members += length,
                (RanOn::Member(member), Some(per_member)) => {
                    if let Some(time) = per_member.members.get_mut(member.0) {
                        time.on_device += length;
                        time.idle += gap.unwrap_or_default();
                    }
                    idle.extend(gap.map(|gap| (window.attribution, gap)));
                }
                (RanOn::Device, _) | (_, None) => {
                    idle.extend(gap.map(|gap| (window.attribution, gap)))
                }
            }
        }
        Settled { idle, per_member }
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
        let mut timeline = Timeline::new(epoch, MemberCount(1));
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
        timeline.push(matmul, None, &window(15, 20));
        timeline.push(norm, None, &window(0, 10));
        timeline.push(Attribution::Untagged, None, &window(18, 30));
        timeline.push(norm, None, &window(40, 50));

        let micros = Duration::from_micros;
        assert_eq!(
            timeline.settle().idle,
            [
                (matmul, micros(5)),
                (Attribution::Untagged, Duration::ZERO),
                (norm, micros(10)),
            ],
            "idle from 10 to 15 before the matmul, none before the untagged \
             kernel that overlapped it, and from 30 to 40 before the norm"
        );
    }

    /// A member waiting on a collective is idle while the other computes, which one merged
    /// timeline would read as filling the gap. A kernel launched off the members' workers is on
    /// no member: it fills no member's gap, and is counted apart.
    #[test]
    fn each_member_is_idle_on_its_own() {
        let epoch = Instant::now();
        let mut timeline = Timeline::new(epoch, MemberCount(2));
        let micros = Duration::from_micros;
        let window = |start: u64, end: u64| {
            ProfileTicks::from_start_end(epoch + micros(start), epoch + micros(end))
        };
        let at = Attribution::Untagged;
        timeline.push(at, Some(Member(0)), &window(0, 10));
        timeline.push(at, Some(Member(1)), &window(0, 30));
        timeline.push(at, None, &window(12, 28));
        timeline.push(at, Some(Member(0)), &window(30, 40));
        timeline.push(at, Some(Member(1)), &window(35, 40));

        let settled = timeline.settle();
        assert_eq!(settled.idle, [(at, micros(20)), (at, micros(5))]);
        assert_eq!(
            settled.per_member,
            Some(PerMember {
                members: vec![
                    MemberTime {
                        on_device: micros(20),
                        idle: micros(20),
                    },
                    MemberTime {
                        on_device: micros(35),
                        idle: micros(5),
                    },
                ],
                off_members: micros(16),
            })
        );
    }
}
