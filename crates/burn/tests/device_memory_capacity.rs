#![cfg(feature = "cpu")]

use burn::tensor::Device;

#[test]
fn a_cubecl_device_states_its_capacity_through_autodiff() {
    assert!(Device::cpu().autodiff().memory_capacity().is_some());
}
