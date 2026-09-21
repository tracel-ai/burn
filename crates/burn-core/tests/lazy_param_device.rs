//! Where a lazy parameter initializes once it has moved.
//!
//! Two fixed CPU backends stand in for two cards, so the placement is observable without one.
//!
//! Run with `cargo test -p burn-core --features flex,ndarray --test lazy_param_device`.
#![cfg(all(feature = "flex", feature = "ndarray"))]
#![allow(deprecated)]

use burn_core::module::{Module, Param, ParamId};
use burn_tensor::{Device, Tensor};

fn transfers() -> [(Device, Device); 2] {
    [
        (Device::flex(), Device::ndarray()),
        (Device::ndarray(), Device::flex()),
    ]
}

#[test]
fn a_lazy_parameter_initializes_on_the_device_it_moved_to() {
    for (source, destination) in transfers() {
        let expected = destination.clone();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            move |device, _| {
                assert_eq!(*device, expected, "the initializer ran on the wrong device");
                Tensor::ones([2, 3], device)
            },
            source,
            false,
            [2, 3].into(),
        );

        let param = param.to_device(&destination);

        assert!(!param.is_initialized());
        assert_eq!(param.val().device(), destination);
    }
}

#[test]
fn a_record_loaded_after_a_move_lands_on_the_new_device() {
    for (source, destination) in transfers() {
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |_, _| panic!("the moved parameter initialized before loading"),
            source.clone(),
            false,
            [2, 3].into(),
        );

        let loaded = param
            .to_device(&destination)
            .transform_for_load(Tensor::ones([2, 3], &source), ParamId::new());

        assert_eq!(loaded.val().device(), destination);
    }
}
