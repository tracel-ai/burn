use alloc::vec::Vec;

use burn_tensor::{Device, StreamId};

use super::{
    DistributedLayer, HeldStage, HiddenSignal, InputSignal, LayerParallelism, LayerPlacement,
    OutputSignal, StagedLayers,
};
use crate::module::Module;

/// A split model's layers cut where a pipeline hands work from one thread to the next: the input
/// layer, the stages of hidden layers, and the output layer. An outer layer runs with the stage
/// beside it when they share a device, and as a part of its own otherwise.
#[derive(Clone, Debug)]
pub struct PipelineParts<M: LayerParallelism> {
    /// The input layer, and the first stage when it is on the same device.
    pub input: InputPart<M>,
    /// The stages run by neither outer part, in forward order.
    pub middle: Vec<MiddlePart<M>>,
    /// The output layer, and the last stage when it is on the same device.
    pub output: OutputPart<M>,
}

impl<M: LayerParallelism> PipelineParts<M> {
    /// `layers` cut into parts, their outer layers on the devices `placement` gives them.
    pub fn new(layers: StagedLayers<M>, placement: &LayerPlacement) -> Self {
        let StagedLayers {
            input,
            stages: mut middle,
            output,
        } = layers;
        let first = if middle
            .first()
            .is_some_and(|stage| stage.device == placement.input)
        {
            Some(middle.remove(0))
        } else {
            None
        };
        let last = if middle
            .last()
            .is_some_and(|stage| stage.device == placement.output)
        {
            middle.pop()
        } else {
            None
        };
        Self {
            input: InputPart {
                layer: input,
                stage: first,
                device: placement.input.clone(),
                stream: StreamId::allocate(),
            },
            middle: middle.into_iter().map(MiddlePart::new).collect(),
            output: OutputPart {
                stage: last,
                layer: output,
                device: placement.output.clone(),
                stream: StreamId::allocate(),
            },
        }
    }

    /// The layers again, the hidden ones in their stages.
    pub fn into_layers(self) -> StagedLayers<M> {
        let stages = self
            .input
            .stage
            .into_iter()
            .chain(self.middle.into_iter().map(|part| part.stage))
            .chain(self.output.stage)
            .collect();
        StagedLayers {
            input: self.input.layer,
            stages,
            output: self.output.layer,
        }
    }

    /// Whether every part runs on one device, so a pipeline has nothing to overlap.
    pub fn on_one_device(&self) -> bool {
        let device = &self.input.device;
        self.middle.iter().all(|part| &part.stage.device == device) && &self.output.device == device
    }

    /// A copy whose layers record no graph, for evaluation.
    pub fn valid(&self) -> Self {
        Self {
            input: InputPart {
                layer: self.input.layer.valid(),
                stage: self.input.stage.as_ref().map(HeldStage::valid),
                device: self.input.device.clone(),
                stream: self.input.stream,
            },
            middle: self
                .middle
                .iter()
                .map(|part| MiddlePart {
                    stage: part.stage.valid(),
                    stream: part.stream,
                })
                .collect(),
            output: OutputPart {
                stage: self.output.stage.as_ref().map(HeldStage::valid),
                layer: self.output.layer.valid(),
                device: self.output.device.clone(),
                stream: self.output.stream,
            },
        }
    }
}

/// The input layer of a split model, and the first stage when the two share a device.
#[derive(Clone, Debug)]
pub struct InputPart<M: LayerParallelism> {
    /// The input layer.
    pub layer: M::InputLayer,
    /// The first stage, on [`device`](Self::device).
    pub stage: Option<HeldStage<M>>,
    /// Where the input layer runs.
    pub device: Device,
    /// The stream the part submits on, kept across runs since a runtime keeps a memory pool per
    /// stream.
    pub stream: StreamId,
}

impl<M: LayerParallelism> InputPart<M> {
    /// The input layer on `input`, moved to its device first, then the stage if it holds one.
    pub fn forward(&self, input: InputSignal<M>) -> HiddenSignal<M> {
        let signal = self.layer.forward(input.to_device(&self.device));
        match &self.stage {
            Some(stage) => stage.forward(signal),
            None => signal,
        }
    }
}

/// A stage that runs as a part of its own, between the outer parts.
#[derive(Clone, Debug)]
pub struct MiddlePart<M: LayerParallelism> {
    /// The stage's layers and device.
    pub stage: HeldStage<M>,
    /// The stream the part submits on, kept across runs since a runtime keeps a memory pool per
    /// stream.
    pub stream: StreamId,
}

impl<M: LayerParallelism> MiddlePart<M> {
    /// `stage` on a stream of its own.
    pub fn new(stage: HeldStage<M>) -> Self {
        Self {
            stage,
            stream: StreamId::allocate(),
        }
    }
}

/// The output layer of a split model, and the last stage when the two share a device.
#[derive(Clone, Debug)]
pub struct OutputPart<M: LayerParallelism> {
    /// The last stage, on [`device`](Self::device).
    pub stage: Option<HeldStage<M>>,
    /// The output layer.
    pub layer: M::OutputLayer,
    /// Where the output layer runs.
    pub device: Device,
    /// The stream the part submits on, kept across runs since a runtime keeps a memory pool per
    /// stream.
    pub stream: StreamId,
}

impl<M: LayerParallelism> OutputPart<M> {
    /// The stage if it holds one, then the output layer, `signal` moved to its device first.
    pub fn forward(&self, signal: HiddenSignal<M>) -> OutputSignal<M> {
        let signal = match &self.stage {
            Some(stage) => stage.forward(signal),
            None => signal,
        };
        self.layer.forward(signal.to_device(&self.device))
    }
}
