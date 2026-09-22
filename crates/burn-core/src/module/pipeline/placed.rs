use burn_tensor::Device;

use super::{Pipeline, PipelinePlacement, PipelineStage};
use crate::module::{Devices, Module, ModuleMapper, ModuleVisitor};

/// A [placed](Pipeline::place) model, holding where each of its segments runs so a split forward
/// reads it rather than working it out again on every call. Dereferences to the model, and is
/// itself a [`Module`], so records, devices and training flags work as they do on the model alone.
#[derive(Clone, Debug)]
pub struct PlacedPipeline<M: Pipeline> {
    model: M,
    placement: PipelinePlacement,
}

impl<M: Pipeline> PlacedPipeline<M> {
    /// Only [`place`](Pipeline::place) and the moves below build one, so the placement always
    /// describes this model.
    pub(super) fn new(model: M, placement: PipelinePlacement) -> Self {
        Self { model, placement }
    }

    /// Where each segment ended up, which is where its own parameters are.
    pub fn placement(&self) -> &PipelinePlacement {
        &self.placement
    }

    /// Run each segment on its device, moving the carry between them.
    pub fn forward(&self, input: M::Input) -> M::Output {
        let mut carry = self
            .model
            .forward_input(input.to_device(&self.placement.input));
        for (index, device) in self.placement.blocks.iter().enumerate() {
            carry = self.model.forward_block(index, carry.to_device(device));
        }
        self.model
            .forward_output(carry.to_device(&self.placement.output))
    }

    /// The model on its own, no longer carrying where its segments run.
    pub fn into_inner(self) -> M {
        self.model
    }

    /// A move of the whole model puts every segment on one device, the block count unchanged.
    fn on_one_device(&self, device: &Device) -> PipelinePlacement {
        PipelinePlacement::new(&[PipelineStage {
            device: device.clone(),
            num_blocks: self.placement.blocks.len(),
        }])
    }
}

impl<M: Pipeline> core::ops::Deref for PlacedPipeline<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.model
    }
}

/// A mapper is trusted to leave each parameter where it found it, so only a move of the whole
/// model rewrites the placement.
impl<M: Pipeline> Module for PlacedPipeline<M> {
    fn collect_devices(&self, devices: Devices) -> Devices {
        self.model.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        let placement = self.on_one_device(device);
        Self::new(self.model.fork(device), placement)
    }

    fn to_device(self, device: &Device) -> Self {
        let placement = self.on_one_device(device);
        Self::new(self.model.to_device(device), placement)
    }

    fn train(self) -> Self {
        Self::new(self.model.train(), self.placement)
    }

    fn valid(&self) -> Self {
        Self::new(self.model.valid(), self.placement.clone())
    }

    fn visit<Visitor: ModuleVisitor>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: ModuleMapper>(self, mapper: &mut Mapper) -> Self {
        Self::new(self.model.map(mapper), self.placement)
    }
}
