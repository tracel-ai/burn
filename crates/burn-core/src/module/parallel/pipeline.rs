use alloc::vec::Vec;
use std::{
    sync::mpsc::{self, Receiver},
    thread::{self, Scope},
};

use burn_tensor::Device;

use super::{
    DistributedLayer, DistributedLayeredModel, HeldStage, HiddenSignal, InputSignal,
    LayerParallelism, LayerPlacement, OutputSignal, StagedLayers,
};
use crate::module::Module;

/// How many batches a stage may finish before the next stage takes the first of them. It
/// bounds the activations waiting on each device, and one is enough to keep the next stage fed.
const BATCHES_AHEAD: usize = 1;

/// A [`DistributedLayeredModel`] run as a pipeline: several batches stream through its stages, each
/// stage on a thread of its own, so one device works on a batch while the next works on the one
/// before.
///
/// The model stays the [`Module`] to load, save and train; this is another way to run it. It pays
/// most where a hop between devices blocks the thread that issues it, as a copy between two
/// runtimes does, since that hop then stalls only its own stage.
///
/// ```rust,ignore
/// let pipeline = InferencePipeline::new(model);
/// let outputs = pipeline.forward(batches);
/// ```
#[derive(Debug)]
pub struct InferencePipeline<M: LayerParallelism> {
    layers: StagedLayers<M>,
    placement: LayerPlacement,
}

impl<M: LayerParallelism> InferencePipeline<M> {
    /// The stages of `model`, as its placement cuts them.
    pub fn new(model: DistributedLayeredModel<M>) -> Self {
        Self {
            placement: model.placement().clone(),
            layers: model.into_stages(),
        }
    }

    /// Run each of `batches` as [`DistributedLayeredModel::forward`] does, every stage on a
    /// thread of its own. The outputs come back in the order of `batches`.
    pub fn forward(&self, batches: Vec<InputSignal<M>>) -> Vec<OutputSignal<M>> {
        let stages = &self.layers.stages;
        if stages.len() < 2 {
            return batches
                .into_iter()
                .map(|batch| self.forward_sequential(batch))
                .collect();
        }
        let first = &stages[0];
        let middle = &stages[1..stages.len() - 1];
        let last = &stages[stages.len() - 1];

        thread::scope(|scope| {
            let mut from_previous = spawn_first_stage(
                scope,
                self.layers.input.clone(),
                first.clone(),
                self.placement.input.clone(),
                batches,
            );
            for stage in middle {
                from_previous = spawn_stage(scope, stage.clone(), from_previous);
            }
            from_previous
                .into_iter()
                .map(|signal| self.finish(last.forward(signal)))
                .collect()
        })
    }

    /// One batch through every layer on the calling thread, for a pipeline of fewer than two
    /// stages, which has nothing to overlap.
    fn forward_sequential(&self, batch: InputSignal<M>) -> OutputSignal<M> {
        let signal = self
            .layers
            .input
            .forward(batch.to_device(&self.placement.input));
        self.finish(
            self.layers
                .stages
                .iter()
                .fold(signal, |signal, stage| stage.forward(signal)),
        )
    }

    /// The output layer on what the last stage returned.
    fn finish(&self, signal: HiddenSignal<M>) -> OutputSignal<M> {
        self.layers
            .output
            .forward(signal.to_device(&self.placement.output))
    }
}

/// `input` and `stage` on a thread of their own, running each of `batches` in turn, moved to
/// `device` first. Returns where the next stage receives what `stage` returns.
fn spawn_first_stage<'scope, M: LayerParallelism + 'scope>(
    scope: &'scope Scope<'scope, '_>,
    input: M::InputLayer,
    stage: HeldStage<M>,
    device: Device,
    batches: Vec<InputSignal<M>>,
) -> Receiver<HiddenSignal<M>> {
    let (to_next, from_this_stage) = mpsc::sync_channel(BATCHES_AHEAD);
    scope.spawn(move || {
        for batch in batches {
            let signal = input.forward(batch.to_device(&device));
            if to_next.send(stage.forward(signal)).is_err() {
                break;
            }
        }
    });
    from_this_stage
}

/// `stage` on a thread of its own, running each signal `from_previous` brings. Returns where the
/// next stage receives what this one returns.
fn spawn_stage<'scope, M: LayerParallelism + 'scope>(
    scope: &'scope Scope<'scope, '_>,
    stage: HeldStage<M>,
    from_previous: Receiver<HiddenSignal<M>>,
) -> Receiver<HiddenSignal<M>> {
    let (to_next, from_this_stage) = mpsc::sync_channel(BATCHES_AHEAD);
    scope.spawn(move || {
        for signal in from_previous {
            if to_next.send(stage.forward(signal)).is_err() {
                break;
            }
        }
    });
    from_this_stage
}
