use alloc::collections::BTreeMap;

use burn_tensor::{Bool, Int, Tensor};

use super::StageMap;
use crate::module::{Module, ModuleMapper, Param, ParamId, ParameterValue, list_param_ids};

/// A module whose forward pass runs as `forward_input`, then `forward_block` for each block in
/// order, then `forward_output`, so its blocks can live on different devices with the activations
/// moved between them.
///
/// [`PipelineLayout`] says which submodules each segment runs: a module tree does not say in what
/// order forward runs.
pub trait Pipeline: Module {
    /// What `forward_input` consumes.
    type Input: Module;
    /// What `forward_output` produces, left on the output segment's device.
    type Output;
    /// The tensors passed from one segment to the next.
    type Activations: Module;

    /// Which submodules each segment runs.
    fn layout(&self) -> PipelineLayout;

    /// Everything before the first block: embeddings, input projection.
    fn forward_input(&self, input: Self::Input) -> Self::Activations;

    /// One block. `index` is the block's position in the layout.
    fn forward_block(&self, index: usize, activations: Self::Activations) -> Self::Activations;

    /// Everything after the last block: final norm, output head.
    fn forward_output(&self, activations: Self::Activations) -> Self::Output;

    /// Fork every parameter onto the device `stages` gives the segment that owns it.
    ///
    /// Forking rather than moving keeps each parameter a leaf, so the model still trains. A
    /// parameter not initialized yet only takes the device, so it initializes there, and a record
    /// loaded afterwards loads there.
    ///
    /// # Panics
    ///
    /// Panics when `stages` does not give a device to every block of the layout, or when a
    /// parameter belongs to no segment, since nothing would say which device it goes to.
    fn place(self, stages: &StageMap) -> Self {
        let layout = self.layout();
        stages.assert_covers(&layout);
        let unowned = list_param_ids(&self)
            .into_iter()
            .filter(|id| layout.segment(*id).is_none())
            .count();
        assert_eq!(
            unowned, 0,
            "{unowned} parameters belong to no segment of the layout"
        );

        self.map(&mut Fork {
            stages,
            layout: &layout,
        })
    }

    /// Run the forward pass with each segment on its device in `stages`, the map the model was
    /// [placed](Self::place) with, moving the activations to each segment's device first.
    ///
    /// # Panics
    ///
    /// Panics when `stages` does not give a device to every block of the layout.
    fn forward_on(&self, stages: &StageMap, input: Self::Input) -> Self::Output {
        stages.assert_covers(&self.layout());

        let mut activations = self.forward_input(input.to_device(&stages.input));
        for (index, device) in stages.blocks.iter().enumerate() {
            activations = self.forward_block(index, activations.to_device(device));
        }
        self.forward_output(activations.to_device(&stages.output))
    }
}

/// Which segment of a [`Pipeline`] owns each parameter, built from the submodules each segment runs.
///
/// ```rust,ignore
/// PipelineLayout::new()
///     .input(&self.embedding)
///     .blocks(&self.layers)
///     .output(&self.norm)
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PipelineLayout {
    owners: BTreeMap<ParamId, Segment>,
    blocks: usize,
}

/// A part of a [`Pipeline`]'s forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Segment {
    /// `forward_input`.
    Input,
    /// `forward_block` at this position.
    Block(usize),
    /// `forward_output`.
    Output,
}

impl PipelineLayout {
    /// A layout with no segment owning anything.
    pub fn new() -> Self {
        Self::default()
    }

    /// The input segment owns the parameters of `module`.
    ///
    /// # Panics
    ///
    /// Panics when another segment already owns one of them.
    pub fn input<M: Module>(self, module: &M) -> Self {
        self.claim(module, Segment::Input)
    }

    /// The next block owns the parameters of `module`.
    ///
    /// # Panics
    ///
    /// Panics when another segment already owns one of them.
    pub fn block<M: Module>(mut self, module: &M) -> Self {
        let segment = Segment::Block(self.blocks);
        self.blocks += 1;
        self.claim(module, segment)
    }

    /// One block per module, in order.
    ///
    /// # Panics
    ///
    /// Panics when another segment already owns one of their parameters.
    pub fn blocks<'a, M: Module + 'a>(self, modules: impl IntoIterator<Item = &'a M>) -> Self {
        modules
            .into_iter()
            .fold(self, |layout, module| layout.block(module))
    }

    /// The output segment owns the parameters of `module`.
    ///
    /// # Panics
    ///
    /// Panics when another segment already owns one of them.
    pub fn output<M: Module>(self, module: &M) -> Self {
        self.claim(module, Segment::Output)
    }

    /// How many blocks the layout has.
    pub fn num_blocks(&self) -> usize {
        self.blocks
    }

    pub(crate) fn segment(&self, id: ParamId) -> Option<Segment> {
        self.owners.get(&id).copied()
    }

    fn claim<M: Module>(mut self, module: &M, segment: Segment) -> Self {
        for id in list_param_ids(module) {
            if let Some(owner) = self.owners.insert(id, segment)
                && owner != segment
            {
                panic!(
                    "parameter {id} is owned by both {owner:?} and {segment:?}; give it to one, and \
                     have the other move it to its device"
                );
            }
        }
        self
    }
}

struct Fork<'a> {
    stages: &'a StageMap,
    layout: &'a PipelineLayout,
}

impl Fork<'_> {
    fn fork<P: ParameterValue>(&self, param: Param<P>) -> Param<P>
    where
        Param<P>: Module,
    {
        match self.layout.segment(param.id) {
            Some(segment) => param.fork(self.stages.device(segment)),
            None => param,
        }
    }
}

impl ModuleMapper for Fork<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        self.fork(param)
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        self.fork(param)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        self.fork(param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_device, test_utils::SimpleLinear};

    #[test]
    fn each_parameter_belongs_to_the_segment_that_claimed_its_module() {
        let device = test_device();
        let embedding = SimpleLinear::new(3, 4, &device);
        let layers = [
            SimpleLinear::new(4, 4, &device),
            SimpleLinear::new(4, 4, &device),
        ];
        let head = SimpleLinear::new(4, 2, &device);
        let unused = SimpleLinear::new(1, 1, &device);

        let layout = PipelineLayout::new()
            .input(&embedding)
            .blocks(&layers)
            .output(&head);

        assert_eq!(layout.num_blocks(), 2);
        assert_eq!(layout.segment(embedding.weight.id), Some(Segment::Input));
        assert_eq!(layout.segment(layers[1].weight.id), Some(Segment::Block(1)));
        assert_eq!(layout.segment(head.weight.id), Some(Segment::Output));
        assert_eq!(layout.segment(unused.weight.id), None);
    }

    #[test]
    #[should_panic(expected = "owned by both")]
    fn a_module_claimed_by_two_segments_is_refused() {
        let shared = SimpleLinear::new(4, 4, &test_device());
        PipelineLayout::new().input(&shared).output(&shared);
    }
}
