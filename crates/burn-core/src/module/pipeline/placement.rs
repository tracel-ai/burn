use alloc::vec::Vec;

use burn_tensor::Device;

use super::{PipelineLayout, PipelineSegment};

/// Where each segment of a [`Pipeline`](super::Pipeline) runs: the device of its input segment, of
/// each block, and of its output segment.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelinePlacement {
    /// Where the input segment runs.
    pub input: Device,
    /// Where each block runs, in forward order.
    pub blocks: Vec<Device>,
    /// Where the output segment runs.
    pub output: Device,
}

/// A run of consecutive blocks on one device.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStage {
    /// The device of every block in the run.
    pub device: Device,
    /// How many blocks the run holds.
    pub num_blocks: usize,
}

impl PipelinePlacement {
    /// One device per block, read off the stages in order. The input segment runs on the first
    /// stage's device and the output segment on the last stage's, so a stage of no blocks gives a
    /// segment a device to itself.
    ///
    /// # Panics
    ///
    /// Panics when `stages` is empty.
    pub fn new(stages: &[PipelineStage]) -> Self {
        let first = stages
            .first()
            .expect("a placement needs at least one stage");
        let last = stages.last().expect("a placement needs at least one stage");

        Self {
            input: first.device.clone(),
            blocks: stages
                .iter()
                .flat_map(|stage| core::iter::repeat_n(stage.device.clone(), stage.num_blocks))
                .collect(),
            output: last.device.clone(),
        }
    }

    /// The blocks shared out in order across `devices`, as evenly as the count allows: when it
    /// does not divide, the first devices take one block more. Devices past the block count are
    /// left out rather than given a segment with nothing to run; [`new`](Self::new) places those
    /// on purpose.
    ///
    /// # Panics
    ///
    /// Panics when `devices` is empty.
    pub fn even(devices: &[Device], num_blocks: usize) -> Self {
        let devices = &devices[..num_blocks.max(1).min(devices.len())];
        let stages: Vec<PipelineStage> = devices
            .iter()
            .enumerate()
            .map(|(index, device)| PipelineStage {
                device: device.clone(),
                num_blocks: num_blocks / devices.len()
                    + usize::from(index < num_blocks % devices.len()),
            })
            .collect();
        Self::new(&stages)
    }

    /// Indexing `blocks` is safe once [`assert_covers`](Self::assert_covers) has passed.
    pub(crate) fn device(&self, segment: PipelineSegment) -> &Device {
        match segment {
            PipelineSegment::Input => &self.input,
            PipelineSegment::Block { index } => &self.blocks[index],
            PipelineSegment::Output => &self.output,
        }
    }

    /// Checked before the walk, so a block with no device is refused before a parameter moves.
    pub(crate) fn assert_covers(&self, layout: &PipelineLayout) {
        assert_eq!(
            self.blocks.len(),
            layout.num_blocks(),
            "the placement must give a device to every block of the layout"
        );
    }
}

#[cfg(all(test, feature = "autodiff"))]
mod tests {
    use super::*;
    use crate::test_device;

    #[test]
    fn an_even_placement_leaves_out_a_device_it_has_no_block_for() {
        let device = test_device();
        let spare = device.clone().autodiff();
        let placement = PipelinePlacement::even(&[device, spare.clone(), spare], 1);

        assert_eq!(placement.blocks.len(), 1);
        assert!(!placement.input.is_autodiff());
        assert!(!placement.output.is_autodiff());
    }

    #[test]
    fn an_even_placement_gives_the_extra_blocks_to_the_first_devices() {
        let device = test_device();
        let placement = PipelinePlacement::even(&[device.clone().autodiff(), device], 3);

        let on_autodiff: Vec<bool> = placement.blocks.iter().map(Device::is_autodiff).collect();
        assert_eq!(on_autodiff, [true, true, false]);
        assert!(placement.input.is_autodiff());
        assert!(!placement.output.is_autodiff());
    }
}
