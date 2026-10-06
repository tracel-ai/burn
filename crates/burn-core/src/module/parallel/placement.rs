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

/// How many bytes each layer of a split model holds on its device.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerMemory {
    /// The input layer's.
    pub input: u64,
    /// Each hidden layer's, in forward order.
    pub hidden: Vec<u64>,
    /// The output layer's, and where it runs.
    pub output: OutputMemory,
}

/// How many bytes the output layer of a split model holds, and which device it runs on.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputMemory {
    /// On the device after the last hidden layer's run.
    AfterHidden(u64),
    /// On the input layer's device, as an output layer sharing the input layer's parameters must.
    WithInput(u64),
}

/// A device and how many bytes it can hold.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceMemory {
    /// The device.
    pub device: Device,
    /// What it can hold, in bytes.
    pub capacity: u64,
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

    /// The layers in order across `devices`, in runs sized to each device's capacity: the device
    /// using the largest share of its capacity uses as small a share as a split in order allows,
    /// so every device keeps about the same fraction free. A layer too large for what a device has
    /// left moves on to the next, and devices past the last run are left out.
    ///
    /// `None` when the layers do not fit on `devices` at all.
    pub fn by_memory(devices: &[DeviceMemory], memory: &LayerMemory) -> Option<Self> {
        let (input, after_hidden) = match memory.output {
            OutputMemory::AfterHidden(bytes) => (memory.input, Some(bytes)),
            OutputMemory::WithInput(bytes) => (memory.input + bytes, None),
        };
        let layers: Vec<u64> = core::iter::once(input)
            .chain(memory.hidden.iter().copied())
            .chain(after_hidden)
            .collect();
        let capacities: Vec<u64> = devices.iter().map(|device| device.capacity).collect();
        let on = runs_by_memory(&capacities, &layers)?;
        let device = |layer: usize| devices[on[layer]].device.clone();
        Some(Self {
            input: device(0),
            hidden: (1..=memory.hidden.len()).map(device).collect(),
            output: match after_hidden {
                Some(_) => device(layers.len() - 1),
                None => device(0),
            },
        })
    }
}

/// The index of the device each of `layers` runs on, in order, for
/// [`LayerPlacement::by_memory`]. The smallest share of every capacity that a fill in order fits
/// under bounds each device; within it, each device takes about its capacity's share of the
/// layers still to place, and more only when the rest would not fit without them.
fn runs_by_memory(capacities: &[u64], layers: &[u64]) -> Option<Vec<usize>> {
    fill_in_order(capacities, layers, 1.0)?;
    let (mut fits, mut short) = (1.0, 0.0);
    for _ in 0..48 {
        let share = (fits + short) / 2.0;
        match fill_in_order(capacities, layers, share) {
            Some(_) => fits = share,
            None => short = share,
        }
    }

    let mut on = Vec::with_capacity(layers.len());
    let mut next = 0;
    for (device, &capacity) in capacities.iter().enumerate() {
        let left: u64 = layers[next..].iter().sum();
        let rest: u64 = capacities[device..].iter().sum();
        let target = match rest {
            0 => 0.0,
            rest => left as f64 * capacity as f64 / rest as f64,
        };
        let limit = (capacity as f64 * fits) as u64;
        let mut load = 0;
        while let Some(&bytes) = layers.get(next) {
            let wanted = load as f64 + bytes as f64 / 2.0 <= target;
            let needed =
                || fill_in_order(&capacities[device + 1..], &layers[next..], fits).is_none();
            if load + bytes > limit || !(wanted || needed()) {
                break;
            }
            load += bytes;
            on.push(device);
            next += 1;
        }
    }
    (next == layers.len()).then_some(on)
}

/// Each device in turn filled with as many of `layers` as fit under `share` of its capacity.
fn fill_in_order(capacities: &[u64], layers: &[u64], share: f64) -> Option<Vec<usize>> {
    let mut device = 0;
    let mut used = 0;
    layers
        .iter()
        .map(|&bytes| {
            while used + bytes > (*capacities.get(device)? as f64 * share) as u64 {
                device += 1;
                used = 0;
            }
            used += bytes;
            Some(device)
        })
        .collect()
}

#[cfg(test)]
mod runs_by_memory_tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    /// The largest share of its capacity any device holds, `None` past a capacity.
    fn worst_share(capacities: &[u64], layers: &[u64], on: &[usize]) -> Option<f64> {
        let mut loads = vec![0u64; capacities.len()];
        for (&bytes, &device) in layers.iter().zip(on) {
            loads[device] += bytes;
        }
        loads
            .iter()
            .zip(capacities)
            .try_fold(0.0f64, |worst, (&load, &capacity)| match load {
                0 => Some(worst),
                _ if load > capacity => None,
                _ => Some(worst.max(load as f64 / capacity as f64)),
            })
    }

    /// Every way to run `layers` across `devices` devices in order, a layer never on an earlier
    /// device than the one before it.
    fn every_split(devices: usize, layers: usize) -> Vec<Vec<usize>> {
        (0..layers).fold(vec![Vec::new()], |splits, _| {
            splits
                .into_iter()
                .flat_map(|split: Vec<usize>| {
                    let from = split.last().copied().unwrap_or(0);
                    (from..devices).map(move |device| {
                        let mut longer = split.clone();
                        longer.push(device);
                        longer
                    })
                })
                .collect()
        })
    }

    /// Small cases drawn from a fixed linear congruential sequence.
    fn cases() -> impl Iterator<Item = (Vec<u64>, Vec<u64>)> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |below: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % below
        };
        (0..600).map(move |_| {
            let devices = 1 + next(4) as usize;
            let layers = 1 + next(7) as usize;
            let capacities = (0..devices).map(|_| next(61)).collect();
            let sizes = (0..layers).map(|_| next(21)).collect();
            (capacities, sizes)
        })
    }

    #[test]
    fn the_split_is_the_least_loaded_any_split_in_order_can_be() {
        for (capacities, layers) in cases() {
            let best = every_split(capacities.len(), layers.len())
                .iter()
                .filter_map(|split| worst_share(&capacities, &layers, split))
                .reduce(f64::min);
            let found = runs_by_memory(&capacities, &layers);

            match (best, found) {
                (None, None) => {}
                (Some(best), Some(on)) => {
                    let share = worst_share(&capacities, &layers, &on)
                        .expect("every device holds what it can");
                    assert!(
                        share <= best + 1e-9,
                        "{layers:?} on {capacities:?}: {on:?} holds {share}, a split holds {best}"
                    );
                    assert!(on.windows(2).all(|pair| pair[0] <= pair[1]), "{on:?}");
                }
                (best, found) => panic!(
                    "{layers:?} on {capacities:?}: the best split is {best:?}, found {found:?}"
                ),
            }
        }
    }

    #[test]
    fn equal_layers_on_equal_devices_split_as_evenly_as_the_count_allows() {
        for devices in 1..=4 {
            for count in devices..=12 {
                let on = runs_by_memory(&vec![1000; devices], &vec![10; count]).expect("they fit");
                let mut held = vec![0; devices];
                for device in on {
                    held[device] += 1;
                }
                let (least, most) = (held.iter().min().unwrap(), held.iter().max().unwrap());
                assert!(most - least <= 1, "{count} layers on {devices}: {held:?}");
            }
        }
    }
}

#[cfg(all(test, feature = "autodiff"))]
mod tests {
    use super::*;
    use crate::test_device;
    use alloc::vec;

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
    fn a_device_with_twice_the_capacity_takes_twice_the_layers() {
        let device = test_device();
        let (small, large) = (device.clone(), device.autodiff());
        let placement = LayerPlacement::by_memory(
            &[
                DeviceMemory {
                    device: small,
                    capacity: 100,
                },
                DeviceMemory {
                    device: large,
                    capacity: 200,
                },
            ],
            &LayerMemory {
                input: 0,
                hidden: vec![10; 6],
                output: OutputMemory::AfterHidden(0),
            },
        )
        .expect("60 bytes fit in 300");

        let on_large: Vec<bool> = placement.hidden.iter().map(Device::is_autodiff).collect();
        assert_eq!(on_large, [false, false, true, true, true, true]);
    }

    #[test]
    fn the_input_layer_counts_against_its_device() {
        let device = test_device();
        let placement = LayerPlacement::by_memory(
            &[
                DeviceMemory {
                    device: device.clone(),
                    capacity: 100,
                },
                DeviceMemory {
                    device: device.autodiff(),
                    capacity: 100,
                },
            ],
            &LayerMemory {
                input: 60,
                hidden: vec![10; 4],
                output: OutputMemory::AfterHidden(0),
            },
        )
        .expect("100 bytes fit in 200");

        assert!(!placement.input.is_autodiff());
        assert!(placement.hidden.iter().all(Device::is_autodiff));
    }

    #[test]
    fn an_output_layer_with_the_input_runs_and_counts_on_its_device() {
        let device = test_device();
        let devices = [device.clone(), device.autodiff()].map(|device| DeviceMemory {
            device,
            capacity: 40,
        });
        let placement = LayerPlacement::by_memory(
            &devices,
            &LayerMemory {
                input: 10,
                hidden: vec![10; 4],
                output: OutputMemory::WithInput(10),
            },
        )
        .expect("60 bytes fit in 80");

        let on_second: Vec<bool> = placement.hidden.iter().map(Device::is_autodiff).collect();
        assert_eq!(on_second, [false, true, true, true]);
        assert!(!placement.input.is_autodiff());
        assert!(!placement.output.is_autodiff());
    }

    #[test]
    fn layers_that_fit_on_no_device_are_refused() {
        let device = test_device();
        let devices = [device.clone(), device.autodiff()].map(|device| DeviceMemory {
            device,
            capacity: 10,
        });
        let memory = LayerMemory {
            input: 0,
            hidden: vec![8; 3],
            output: OutputMemory::AfterHidden(0),
        };

        assert_eq!(LayerPlacement::by_memory(&devices, &memory), None);
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
