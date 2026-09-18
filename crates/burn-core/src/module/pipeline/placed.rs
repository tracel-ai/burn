use alloc::vec::Vec;

use burn_tensor::{Bool, Device, Int, Tensor};

use super::{Pipeline, PipelineLayout, PipelinePlacement, PipelineSegment};
use crate::module::{Devices, Module, ModuleMapper, ModuleVisitor, Param, ParamId};

/// A [placed](Pipeline::place) model, holding where each of its segments runs so a split forward
/// reads it rather than working it out again on every call. Dereferences to the model, and is
/// itself a [`Module`], so records, devices and training flags work as they do on the model alone.
#[derive(Clone, Debug)]
pub struct PlacedPipeline<M: Pipeline> {
    model: M,
    placement: PipelinePlacement,
}

impl<M: Pipeline> PlacedPipeline<M> {
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

    /// A mapper may put a parameter anywhere, as burn-optim's flat-vector mapper does, so where the
    /// segments run is read back off them rather than carried over.
    fn resolved(model: M, previous: &PipelinePlacement) -> Self {
        let layout = model.layout();
        let placement = ResolveDevices::of(&model, &layout, previous);

        Self { model, placement }
    }

    fn on_one_device(&self, device: &Device) -> PipelinePlacement {
        PipelinePlacement::even(core::slice::from_ref(device), self.placement.blocks.len())
    }
}

impl<M: Pipeline> core::ops::Deref for PlacedPipeline<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.model
    }
}

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
        let previous = self.placement.clone();
        Self::resolved(self.model.map(mapper), &previous)
    }
}

struct ResolveDevices<'a> {
    layout: &'a PipelineLayout,
    input: Option<Device>,
    blocks: Vec<Option<Device>>,
    output: Option<Device>,
}

impl<'a> ResolveDevices<'a> {
    /// A segment holding no parameters keeps the device of the one before it, which moves nothing.
    fn of<M: Module>(
        model: &M,
        layout: &'a PipelineLayout,
        previous: &PipelinePlacement,
    ) -> PipelinePlacement {
        let mut resolve = Self {
            layout,
            input: None,
            blocks: alloc::vec![None; layout.num_blocks()],
            output: None,
        };
        model.visit(&mut resolve);

        let mut latest = resolve.input.unwrap_or_else(|| previous.input.clone());
        let input = latest.clone();
        let mut blocks = Vec::with_capacity(resolve.blocks.len());
        for device in resolve.blocks {
            if let Some(device) = device {
                latest = device;
            }
            blocks.push(latest.clone());
        }

        PipelinePlacement {
            input,
            blocks,
            output: resolve.output.unwrap_or(latest),
        }
    }

    /// Reading the device must not initialize a parameter that `place` only retargeted.
    fn record(&mut self, id: ParamId, device: Device) {
        let slot = match self.layout.segment(id) {
            Some(PipelineSegment::Input) => &mut self.input,
            Some(PipelineSegment::Block { index }) => &mut self.blocks[index],
            Some(PipelineSegment::Output) => &mut self.output,
            None => return,
        };
        slot.get_or_insert(device);
    }
}

impl ModuleVisitor for ResolveDevices<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.record(param.id, param.lazy_device());
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        self.record(param.id, param.lazy_device());
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        self.record(param.id, param.lazy_device());
    }
}
