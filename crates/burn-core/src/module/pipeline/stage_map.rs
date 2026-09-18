use alloc::vec::Vec;

use burn_tensor::Device;

use super::{PipelineLayout, Segment};

/// The device of every segment of a [`Pipeline`](super::Pipeline).
#[derive(Debug, Clone, PartialEq)]
pub struct StageMap {
    /// Where the input segment runs.
    pub input: Device,
    /// Where each block runs, in forward order.
    pub blocks: Vec<Device>,
    /// Where the output segment runs.
    pub output: Device,
}

/// A run of consecutive blocks on one device.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    /// The device of every block in the run.
    pub device: Device,
    /// How many blocks the run holds.
    pub blocks: usize,
}

impl StageMap {
    /// Consecutive stages, the input segment with the first and the output segment with the last.
    ///
    /// # Panics
    ///
    /// Panics when `stages` is empty.
    pub fn new(stages: &[Stage]) -> Self {
        let first = stages
            .first()
            .expect("a stage map needs at least one stage");
        let last = stages.last().expect("a stage map needs at least one stage");

        Self {
            input: first.device.clone(),
            blocks: stages
                .iter()
                .flat_map(|stage| core::iter::repeat_n(stage.device.clone(), stage.blocks))
                .collect(),
            output: last.device.clone(),
        }
    }

    /// Every segment on one device.
    pub fn single(device: &Device, blocks: usize) -> Self {
        Self::new(&[Stage {
            device: device.clone(),
            blocks,
        }])
    }

    pub(crate) fn device(&self, segment: Segment) -> &Device {
        match segment {
            Segment::Input => &self.input,
            Segment::Block(index) => &self.blocks[index],
            Segment::Output => &self.output,
        }
    }

    pub(crate) fn assert_covers(&self, layout: &PipelineLayout) {
        assert_eq!(
            self.blocks.len(),
            layout.num_blocks(),
            "the stage map must give a device to every block of the layout"
        );
    }
}
