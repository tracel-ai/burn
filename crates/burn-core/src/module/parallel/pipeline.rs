use alloc::vec::Vec;
use std::{
    sync::mpsc::{self, Receiver},
    thread::{self, Scope},
};

use super::{
    DistributedLayeredModel, HiddenSignal, InputPart, InputSignal, LayerParallelism, MiddlePart,
    OutputSignal, PipelineParts,
};

/// How many batches a stage may finish before the next stage takes the first of them. It
/// bounds the activations waiting on each device, and one is enough to keep the next stage fed.
const BATCHES_AHEAD: usize = 1;

/// A [`DistributedLayeredModel`] run as a pipeline: several batches stream through its stages, each
/// stage on a thread of its own, so one device works on a batch while the next works on the one
/// before.
///
/// The model stays the [`Module`](crate::module::Module) to load, save and train; this is another
/// way to run it. It pays most where a hop between devices blocks the thread that issues it, as a
/// copy between two runtimes does, since that hop then stalls only its own stage.
///
/// ```rust,ignore
/// let pipeline = InferencePipeline::new(model);
/// let outputs = pipeline.forward(batches);
/// ```
#[derive(Debug)]
pub struct InferencePipeline<M: LayerParallelism> {
    parts: PipelineParts<M>,
}

impl<M: LayerParallelism> InferencePipeline<M> {
    /// The parts of `model`, as its placement cuts them.
    pub fn new(model: DistributedLayeredModel<M>) -> Self {
        let placement = model.placement().clone();
        Self::from_parts(PipelineParts::new(model.into_stages(), &placement))
    }

    /// A pipeline over `parts`.
    pub fn from_parts(parts: PipelineParts<M>) -> Self {
        Self { parts }
    }

    /// Run each of `batches` as [`DistributedLayeredModel::forward`] does, every part on a thread
    /// of its own. The outputs come back in the order of `batches`.
    pub fn forward(&self, batches: Vec<InputSignal<M>>) -> Vec<OutputSignal<M>> {
        let parts = &self.parts;
        if parts.on_one_device() {
            return batches
                .into_iter()
                .map(|batch| self.forward_sequential(batch))
                .collect();
        }

        thread::scope(|scope| {
            let mut from_previous = spawn_input_part(scope, parts.input.clone(), batches);
            for part in &parts.middle {
                from_previous = spawn_stage(scope, part.clone(), from_previous);
            }
            parts.output.stream.executes(|| {
                from_previous
                    .into_iter()
                    .map(|signal| parts.output.forward(signal))
                    .collect()
            })
        })
    }

    /// One batch through every layer on the calling thread, for parts that share one device and so
    /// have nothing to overlap.
    fn forward_sequential(&self, batch: InputSignal<M>) -> OutputSignal<M> {
        let parts = &self.parts;
        let signal = parts
            .middle
            .iter()
            .fold(parts.input.forward(batch), |signal, part| {
                part.stage.forward(signal)
            });
        parts.output.forward(signal)
    }
}

/// `part` on a thread of its own and on its stream, running each of `batches` in turn. Returns
/// where the next part receives what this one returns.
fn spawn_input_part<'scope, M: LayerParallelism + 'scope>(
    scope: &'scope Scope<'scope, '_>,
    part: InputPart<M>,
    batches: Vec<InputSignal<M>>,
) -> Receiver<HiddenSignal<M>> {
    let (to_next, from_this_stage) = mpsc::sync_channel(BATCHES_AHEAD);
    scope.spawn(move || {
        part.stream.executes(|| {
            for batch in batches {
                if to_next.send(part.forward(batch)).is_err() {
                    break;
                }
            }
        })
    });
    from_this_stage
}

/// `part` on a thread of its own and on its stream, running each signal `from_previous` brings.
/// Returns where the next part receives what this one returns.
fn spawn_stage<'scope, M: LayerParallelism + 'scope>(
    scope: &'scope Scope<'scope, '_>,
    part: MiddlePart<M>,
    from_previous: Receiver<HiddenSignal<M>>,
) -> Receiver<HiddenSignal<M>> {
    let (to_next, from_this_stage) = mpsc::sync_channel(BATCHES_AHEAD);
    scope.spawn(move || {
        part.stream.executes(|| {
            for signal in from_previous {
                if to_next.send(part.stage.forward(signal)).is_err() {
                    break;
                }
            }
        })
    });
    from_this_stage
}
