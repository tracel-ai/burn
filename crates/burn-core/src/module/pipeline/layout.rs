use alloc::collections::BTreeMap;

use crate::module::{Module, ParamId, list_param_ids};

/// Which segment of a [`Pipeline`](super::Pipeline) owns each parameter, built from the submodules
/// each segment runs.
#[derive(Debug, Clone, Default)]
pub struct PipelineLayout {
    /// Flattened to parameters, since [`place`](super::Pipeline::place) walks parameters, not
    /// modules.
    owners: BTreeMap<ParamId, PipelineSegment>,
    /// Counted, not read back from `owners`, which a block with no parameters never enters.
    num_blocks: usize,
}

/// A part of a [`Pipeline`](super::Pipeline)'s forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineSegment {
    /// `forward_input`.
    Input,
    /// `forward_block` at `index`.
    Block { index: usize },
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
        self.claim(module, PipelineSegment::Input)
    }

    /// The next block owns the parameters of `module`.
    ///
    /// # Panics
    ///
    /// Panics when another segment already owns one of them.
    pub fn block<M: Module>(mut self, module: &M) -> Self {
        let segment = PipelineSegment::Block {
            index: self.num_blocks,
        };
        self.num_blocks += 1;
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
        self.claim(module, PipelineSegment::Output)
    }

    /// How many blocks the layout has.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// `None` is a parameter outside the layout, which [`place`](super::Pipeline::place)
    /// refuses rather than pick a device for.
    pub(crate) fn segment(&self, id: ParamId) -> Option<PipelineSegment> {
        self.owners.get(&id).copied()
    }

    /// Giving one module to the same segment twice is allowed; giving it to two is the panic.
    fn claim<M: Module>(mut self, module: &M, segment: PipelineSegment) -> Self {
        for id in list_param_ids(module) {
            if let Some(owner) = self.owners.insert(id, segment)
                && owner != segment
            {
                panic!(
                    "a parameter of the module given to {segment:?} already belongs to \
                     {owner:?}; give it to one segment, and have the other move it to its device"
                );
            }
        }
        self
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
        assert_eq!(
            layout.segment(embedding.weight.id),
            Some(PipelineSegment::Input)
        );
        assert_eq!(
            layout.segment(layers[1].weight.id),
            Some(PipelineSegment::Block { index: 1 })
        );
        assert_eq!(
            layout.segment(head.weight.id),
            Some(PipelineSegment::Output)
        );
        assert_eq!(layout.segment(unused.weight.id), None);
    }

    #[test]
    #[should_panic(expected = "already belongs to")]
    fn a_module_claimed_by_two_segments_is_refused() {
        let shared = SimpleLinear::new(4, 4, &test_device());
        PipelineLayout::new().input(&shared).output(&shared);
    }
}
