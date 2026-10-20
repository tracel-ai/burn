use std::sync::Arc;
use std::time::{Duration, Instant};

use burn_tensor::{Device, FloatDType};

use super::recorder::RunRecorder;
use super::watching::{Claim, Watching};
use crate::{Breakdown, ComponentObservation, Measurement, Peaks};

/// Measures a model on a device into a [`Measurement`]: one instrumented
/// run's [`Breakdown`], the device's [`Peaks`], and the wall time of a clean
/// run, with nothing instrumenting it.
///
/// ```rust,ignore
/// let measurement = Instrument::new(&device).measure(|| model.forward(input.clone()));
/// measurement.write(std::fs::File::create("model.json")?)?;
/// ```
///
/// See the [crate docs](crate#measuring-a-model) for what an instrumented run
/// records and how far its numbers are to be trusted.
pub struct Instrument {
    device: Device,
    claim: Claim,
    warmup: usize,
    samples: usize,
    dtype: FloatDType,
}

impl Instrument {
    /// An instrument for work running on `device`, and the process's one until
    /// it is dropped.
    ///
    /// It warms the run up 5 times, takes the clean wall as the median of 10
    /// clean runs, and probes the arithmetic peak at the device's
    /// default float dtype; see [`warmup`](Self::warmup),
    /// [`samples`](Self::samples) and [`dtype`](Self::dtype).
    ///
    /// # Panics
    ///
    /// If another instrument exists in the process. Refused here, before the
    /// device is synced or its fusion worker pointed anywhere: either would
    /// disturb the run the other is instrumenting.
    ///
    /// If `device` fuses and this crate was built without its `fusion`
    /// feature: the instrument could not see the blocks, and would place every
    /// fused kernel on whichever component sealed it.
    pub fn new(device: &Device) -> Self {
        #[cfg(not(feature = "fusion"))]
        assert!(
            device.as_dispatch().fusion_worker().is_none(),
            "the device fuses, so instrumenting it needs burn-instrument's `fusion` feature"
        );
        Self {
            claim: Claim::take(),
            device: device.clone(),
            warmup: 5,
            samples: 10,
            dtype: device.settings().float_dtype,
        }
    }

    /// Run the model `runs` times before anything is measured. The first run
    /// compiles and autotunes, and a tuner's candidates instrumented as the
    /// run's would be recorded as untagged kernels.
    pub fn warmup(mut self, runs: usize) -> Self {
        self.warmup = runs;
        self
    }

    /// Take the clean wall as the median of `runs` clean runs, or leave
    /// it unmeasured for `0`.
    pub fn samples(mut self, runs: usize) -> Self {
        self.samples = runs;
        self
    }

    /// Probe the arithmetic peak at `dtype`, which should be the one the run
    /// computes in.
    pub fn dtype(mut self, dtype: FloatDType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Warm `model` up, measure the device's peaks and its clean wall, then
    /// run it once more instrumented.
    ///
    /// `model` runs once per warmup, once per sample and once instrumented, and
    /// what it returns is held until the run it belongs to has ended: an
    /// output dropped inside the run would let the fuser skip the work that
    /// wrote it. Each clean run is synced, so call it on work already
    /// synced and keep other threads off the device: see
    /// [`breakdown`](Self::breakdown).
    ///
    /// # Panics
    ///
    /// If the device stops answering, or if the work producing an observed
    /// value failed.
    pub fn measure<T>(self, mut model: impl FnMut() -> T) -> Measurement {
        for _ in 0..self.warmup {
            self.run_clean(&mut model);
        }
        // Before anything is timed: the probes are saturating kernels that
        // move clocks and thermals for whatever runs after them.
        let peaks = Peaks::new(&self.device, self.dtype);
        let clean_wall = self.clean_wall(&mut model);
        let device = format!("{:?}", self.device);
        Measurement {
            device,
            peaks: Some(peaks),
            clean_wall,
            breakdown: self.breakdown(model),
        }
    }

    /// Run `model` instrumented, and end the run, without warming it up or
    /// measuring what it is scored against: every component it enters arrives
    /// from this thread and every kernel the runtime issues for it from
    /// whichever thread issued it, until the run's last queued segment has
    /// executed and its timings are read back. What `model` returns is
    /// held until then.
    ///
    /// Instrumented, the model runs the graph it runs in production — nothing
    /// is drained at a boundary — so call it on work already synced: an
    /// operation queued before it would be watched too. Run it once
    /// beforehand, so the tuner's work is not recorded as the run's, and keep other threads off
    /// the device while it runs: see the [crate docs](crate#measuring-a-model).
    ///
    /// Taken by value: an instrument holds one run, and its breakdown is read
    /// once that run has ended.
    ///
    /// # Panics
    ///
    /// If the device stops answering at the end of the run, or if the work
    /// producing an observed value failed.
    pub fn breakdown<T>(self, model: impl FnOnce() -> T) -> Breakdown {
        // One recorder, three observations: the components arrive on this
        // thread and the kernels and fusion blocks on whichever thread runs
        // them, so the tree is written from both. Each kernel lands under the
        // attribution of the thread that issued it — see `LaunchCursor`.
        // Built here rather than in `new`, so its epoch is the run's.
        let recorder = Arc::new(RunRecorder::new(&self.device));
        let watching = Watching::new(&recorder, self.claim);
        let observation = ComponentObservation::new(Box::new(Arc::clone(&recorder)));
        let output = model();
        drop(observation);
        let breakdown = recorder.finish(watching);
        drop(output);
        breakdown
    }

    /// The median wall time of the configured samples, or `None` for none.
    fn clean_wall<T>(&self, model: &mut impl FnMut() -> T) -> Option<Duration> {
        let mut samples: Vec<Duration> = (0..self.samples)
            .map(|_| {
                let start = Instant::now();
                self.run_clean(model);
                start.elapsed()
            })
            .collect();
        samples.sort();
        samples.get(samples.len() / 2).copied()
    }

    /// Run `model` clean to its end, its output held to the sync.
    fn run_clean<T>(&self, model: &mut impl FnMut() -> T) {
        let output = model();
        self.device.sync().expect("the device answers");
        drop(output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second instrument is refused before it touches the device: built while
    /// another exists, it would sync the device and point the fusion worker
    /// away from the run the first is instrumenting, before `breakdown` could
    /// refuse it.
    #[test]
    fn a_second_instrument_is_refused_before_it_touches_the_device() {
        let device = Device::flex();
        let first = Instrument::new(&device);
        let second = std::panic::catch_unwind(|| Instrument::new(&Device::flex()));
        assert!(second.is_err(), "refused while the first exists");

        drop(first);
        drop(Instrument::new(&device));
    }
}
