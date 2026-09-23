use alloc::vec::Vec;

use burn_tensor::Device;

/// Where each layer of a [`LayerParallelism`](super::LayerParallelism) model runs: the device of
/// its input layer, of each hidden layer, and of its output layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerPlacement {
    /// Where the input layer runs.
    pub input: Device,
    /// Where each hidden layer runs, in forward order.
    pub hidden: Vec<Device>,
    /// Where the output layer runs.
    pub output: Device,
}

/// A run of consecutive hidden layers on one device.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStage {
    /// The device of every hidden layer in the run.
    pub device: Device,
    /// How many hidden layers the run holds.
    pub num_hidden_layers: usize,
}

impl LayerPlacement {
    /// One device per hidden layer, read off the stages in order. The input layer runs on the
    /// first stage's device and the output layer on the last stage's, so a stage of no hidden
    /// layers gives one of them a device to itself.
    ///
    /// # Panics
    ///
    /// Panics when `stages` is empty.
    pub fn new(stages: &[LayerStage]) -> Self {
        let first = stages
            .first()
            .expect("a placement needs at least one stage");
        let last = stages.last().expect("a placement needs at least one stage");

        Self {
            input: first.device.clone(),
            hidden: stages
                .iter()
                .flat_map(|stage| {
                    core::iter::repeat_n(stage.device.clone(), stage.num_hidden_layers)
                })
                .collect(),
            output: last.device.clone(),
        }
    }

    /// The hidden layers shared out in order across `devices`, as evenly as the count allows:
    /// when it does not divide, the first devices take one layer more. Devices past the layer
    /// count are left out rather than given a stage with nothing to run; [`new`](Self::new) places
    /// those on purpose.
    ///
    /// # Panics
    ///
    /// Panics when `devices` is empty.
    pub fn even(devices: &[Device], num_hidden_layers: usize) -> Self {
        let devices = &devices[..num_hidden_layers.max(1).min(devices.len())];
        let stages: Vec<LayerStage> = devices
            .iter()
            .enumerate()
            .map(|(index, device)| LayerStage {
                device: device.clone(),
                num_hidden_layers: num_hidden_layers / devices.len()
                    + usize::from(index < num_hidden_layers % devices.len()),
            })
            .collect();
        Self::new(&stages)
    }
}

#[cfg(all(test, feature = "autodiff"))]
mod tests {
    use super::*;
    use crate::test_device;

    #[test]
    fn an_even_placement_leaves_out_a_device_it_has_no_layer_for() {
        let device = test_device();
        let spare = device.clone().autodiff();
        let placement = LayerPlacement::even(&[device, spare.clone(), spare], 1);

        assert_eq!(placement.hidden.len(), 1);
        assert!(!placement.input.is_autodiff());
        assert!(!placement.output.is_autodiff());
    }

    #[test]
    fn an_even_placement_gives_the_extra_layers_to_the_first_devices() {
        let device = test_device();
        let placement = LayerPlacement::even(&[device.clone().autodiff(), device], 3);

        let on_autodiff: Vec<bool> = placement.hidden.iter().map(Device::is_autodiff).collect();
        assert_eq!(on_autodiff, [true, true, false]);
        assert!(placement.input.is_autodiff());
        assert!(!placement.output.is_autodiff());
    }
}
