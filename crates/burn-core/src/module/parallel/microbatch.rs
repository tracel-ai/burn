use alloc::vec::Vec;
use std::{sync::mpsc, thread};

use super::{DistributedLayer, DistributedLayeredModel, LayerParallelism, StagedLayers};
use crate::module::Module;

/// How many microbatches a stage may finish before the next stage takes the first of them. It
/// bounds the activations waiting on each device, and one is enough to keep the next stage fed.
const MICROBATCHES_AHEAD: usize = 1;

impl<M: LayerParallelism> DistributedLayeredModel<M> {
    /// Run each of `inputs` as [`forward`](Self::forward) does, each stage on a thread of its own,
    /// so a stage works on one microbatch while the stage after it works on the one before. The
    /// outputs come back in the order of `inputs`.
    ///
    /// A stage whose hop to the next one blocks its thread, as a copy between runtimes does, then
    /// stalls only itself. Inference only: nothing here runs a backward.
    pub fn forward_microbatched(
        &self,
        inputs: Vec<<M::InputLayer as DistributedLayer>::Input>,
    ) -> Vec<<M::OutputLayer as DistributedLayer>::Output> {
        let input_device = self.placement().input.clone();
        let output_device = &self.placement().output;
        let StagedLayers {
            input,
            stages,
            output,
        } = self.clone().into_stages();
        let mut stages = stages.into_iter();
        let (Some(first), Some(last)) = (stages.next(), stages.next_back()) else {
            return inputs
                .into_iter()
                .map(|input| self.forward(input))
                .collect();
        };

        thread::scope(|scope| {
            let (sender, mut received) = mpsc::sync_channel(MICROBATCHES_AHEAD);
            scope.spawn(move || {
                for microbatch in inputs {
                    let signal = input.forward(microbatch.to_device(&input_device));
                    if sender.send(first.forward(signal)).is_err() {
                        break;
                    }
                }
            });
            for stage in stages {
                let (sender, next) = mpsc::sync_channel(MICROBATCHES_AHEAD);
                scope.spawn(move || {
                    for signal in received {
                        if sender.send(stage.forward(signal)).is_err() {
                            break;
                        }
                    }
                });
                received = next;
            }
            received
                .into_iter()
                .map(|signal| output.forward(last.forward(signal).to_device(output_device)))
                .collect()
        })
    }
}
