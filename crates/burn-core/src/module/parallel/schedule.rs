use alloc::vec::Vec;

/// The order each stage of a pipeline runs every microbatch's forward and backward pass in.
///
/// Both schedules idle the same fraction of a step, `(p - 1) / (m + p - 1)` for `p` stages and `m`
/// microbatches, when hops between stages are free. They differ in how far a stage runs ahead of
/// its backwards, which sets how many microbatches' activations it holds and how much of a hop's
/// latency the step waits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// Every forward, then every backward. Each stage holds all `m` microbatches at once, and a
    /// hop's latency is paid only while the pipeline fills and drains.
    GPipe,
    /// Each stage runs ahead only as far as keeps the stages after it busy, then alternates one
    /// forward with one backward. Stage `s` holds at most `p - s` microbatches, so `m` can grow
    /// to shrink the idle fraction, but a hop's round trip lands on every microbatch.
    OneForwardOneBackward,
}

/// How many stages a pipeline has and how many microbatches one step splits its batch into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineShape {
    /// The runs of consecutive layers on one device, input to output.
    pub num_stages: usize,
    /// The microbatches each stage runs a forward and a backward of per step.
    pub num_microbatches: usize,
}

/// One stage's pass over one microbatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagePass {
    /// Waits for the previous stage's output for `microbatch`, unless this is the first stage.
    Forward {
        /// The microbatch, counted from zero within the step.
        microbatch: usize,
    },
    /// Waits for the next stage's input gradient for `microbatch`, unless this is the last stage.
    Backward {
        /// The microbatch, counted from zero within the step.
        microbatch: usize,
    },
}

impl PipelineSchedule {
    /// The passes each stage runs in one step, in order, first stage first.
    pub fn passes(self, shape: PipelineShape) -> Vec<Vec<StagePass>> {
        let forward = |microbatch| StagePass::Forward { microbatch };
        let backward = |microbatch| StagePass::Backward { microbatch };
        let num_microbatches = shape.num_microbatches;

        (0..shape.num_stages)
            .map(|stage| {
                let warmup = self.warmup(stage, shape);
                (0..warmup)
                    .map(forward)
                    .chain(
                        (warmup..num_microbatches)
                            .flat_map(|next| [forward(next), backward(next - warmup)]),
                    )
                    .chain((num_microbatches - warmup..num_microbatches).map(backward))
                    .collect()
            })
            .collect()
    }

    /// How many forwards `stage` runs before its first backward.
    fn warmup(self, stage: usize, shape: PipelineShape) -> usize {
        match self {
            Self::GPipe => shape.num_microbatches,
            Self::OneForwardOneBackward => {
                (shape.num_stages - stage - 1).min(shape.num_microbatches)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const SCHEDULES: [PipelineSchedule; 2] = [
        PipelineSchedule::GPipe,
        PipelineSchedule::OneForwardOneBackward,
    ];

    fn shapes() -> impl Iterator<Item = PipelineShape> {
        (1..=5).flat_map(|num_stages| {
            (1..=9).map(move |num_microbatches| PipelineShape {
                num_stages,
                num_microbatches,
            })
        })
    }

    /// The most microbatches a stage has run the forward of and not yet the backward.
    fn peak_in_flight(passes: &[StagePass]) -> usize {
        passes
            .iter()
            .scan(0usize, |in_flight, pass| {
                match pass {
                    StagePass::Forward { .. } => *in_flight += 1,
                    StagePass::Backward { .. } => *in_flight -= 1,
                }
                Some(*in_flight)
            })
            .max()
            .unwrap_or(0)
    }

    /// How long each pass takes on every stage, and each hop between two stages.
    #[derive(Clone, Copy)]
    struct Timing {
        forward: f64,
        backward: f64,
        hop: f64,
    }

    impl Timing {
        /// When the step ends, each stage running its passes in order and each pass starting once
        /// its stage is free and its input has arrived.
        ///
        /// # Panics
        ///
        /// Panics when no stage can start its next pass: the passes deadlock.
        fn step(self, stages: &[Vec<StagePass>]) -> f64 {
            let mut ends: Vec<Vec<f64>> = vec![Vec::new(); stages.len()];
            let ended = |ends: &[Vec<f64>], stage: usize, pass: StagePass| {
                let index = stages[stage].iter().position(|&other| other == pass)?;
                ends[stage].get(index).copied()
            };

            while ends
                .iter()
                .zip(stages)
                .any(|(ended, passes)| ended.len() < passes.len())
            {
                let mut progressed = false;
                for (stage, passes) in stages.iter().enumerate() {
                    let Some(&pass) = passes.get(ends[stage].len()) else {
                        continue;
                    };
                    let (input_stage, cost) = match pass {
                        StagePass::Forward { .. } => (stage.checked_sub(1), self.forward),
                        StagePass::Backward { .. } => (
                            Some(stage + 1).filter(|&next| next < stages.len()),
                            self.backward,
                        ),
                    };
                    let arrived = match input_stage {
                        None => 0.0,
                        Some(input_stage) => match ended(&ends, input_stage, pass) {
                            Some(end) => end + self.hop,
                            None => continue,
                        },
                    };
                    let start = f64::max(ends[stage].last().copied().unwrap_or(0.0), arrived);
                    ends[stage].push(start + cost);
                    progressed = true;
                }
                assert!(progressed, "the passes deadlock");
            }
            ends.iter()
                .filter_map(|ended| ended.last().copied())
                .fold(0.0, f64::max)
        }
    }

    #[test]
    fn every_stage_runs_each_microbatch_forward_once_then_backward_once() {
        for schedule in SCHEDULES {
            for shape in shapes() {
                for passes in schedule.passes(shape) {
                    let position = |wanted| passes.iter().position(|&pass| pass == wanted);
                    for microbatch in 0..shape.num_microbatches {
                        let forward = position(StagePass::Forward { microbatch });
                        let backward = position(StagePass::Backward { microbatch });
                        assert!(
                            forward.is_some() && forward < backward,
                            "{schedule:?} {shape:?} microbatch {microbatch}"
                        );
                    }
                    assert_eq!(passes.len(), 2 * shape.num_microbatches);
                }
            }
        }
    }

    #[test]
    fn gpipe_holds_every_microbatch_and_one_forward_one_backward_at_most_the_stages_left() {
        for shape in shapes() {
            let gpipe = PipelineSchedule::GPipe.passes(shape);
            let one_one = PipelineSchedule::OneForwardOneBackward.passes(shape);
            for stage in 0..shape.num_stages {
                assert_eq!(peak_in_flight(&gpipe[stage]), shape.num_microbatches);
                assert_eq!(
                    peak_in_flight(&one_one[stage]),
                    (shape.num_stages - stage).min(shape.num_microbatches),
                    "{shape:?} stage {stage}"
                );
            }
        }
    }

    #[test]
    fn with_free_hops_both_schedules_idle_only_while_filling_and_draining() {
        let timing = Timing {
            forward: 1.0,
            backward: 2.0,
            hop: 0.0,
        };
        for schedule in SCHEDULES {
            for shape in shapes() {
                let ideal = (shape.num_microbatches + shape.num_stages - 1) as f64
                    * (timing.forward + timing.backward);
                assert_eq!(
                    timing.step(&schedule.passes(shape)),
                    ideal,
                    "{schedule:?} {shape:?}"
                );
            }
        }
    }

    #[test]
    fn gpipe_pays_hop_latency_once_per_step_and_one_forward_one_backward_on_every_microbatch() {
        let free = Timing {
            forward: 1.0,
            backward: 2.0,
            hop: 0.0,
        };
        let slow = Timing { hop: 0.5, ..free };
        let shape = PipelineShape {
            num_stages: 4,
            num_microbatches: 16,
        };
        let added = |schedule: PipelineSchedule| {
            let passes = schedule.passes(shape);
            slow.step(&passes) - free.step(&passes)
        };

        assert_eq!(
            added(PipelineSchedule::GPipe),
            2.0 * (shape.num_stages - 1) as f64 * slow.hop
        );
        assert!(
            added(PipelineSchedule::OneForwardOneBackward)
                >= shape.num_microbatches as f64 * slow.hop
        );
    }
}
