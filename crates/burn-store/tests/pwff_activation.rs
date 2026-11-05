//! Stateful PWFF activations and compatibility with checkpoints containing only GELU.

use burn_core as burn;
use burn_core::{
    module::Module,
    tensor::{Device, Tensor},
};
use burn_nn::{
    Dropout, Gelu, Initializer, Linear,
    activation::{Activation, ActivationConfig, PReluConfig, SwiGluConfig},
    transformer::{PositionWiseFeedForward, PositionWiseFeedForwardConfig},
};
use burn_store::{BurnpackStore, ModuleSnapshot, ModuleStore, PathFilter};

fn config(activation: ActivationConfig) -> PositionWiseFeedForwardConfig {
    PositionWiseFeedForwardConfig::new(1, 1)
        .with_dropout(0.0)
        .with_initializer(Initializer::Constant { value: 1.0 })
        .with_activation(activation)
}

fn assert_round_trip(model: PositionWiseFeedForward, config: PositionWiseFeedForwardConfig) {
    let device = Device::flex();
    let input = Tensor::<3>::from_data([[[-2.0], [1.0]]], &device);
    let expected = model.forward(input.clone()).into_data();
    let mut loaded = config.init(&device);
    // Modified activation parameters must change the output from initialization.
    assert_ne!(loaded.forward(input.clone()).into_data(), expected);

    let mut store = BurnpackStore::from_bytes(None);
    model.save_into(&mut store).unwrap();
    let mut store = BurnpackStore::from_bytes(Some(store.get_bytes().unwrap()));
    let result = loaded.load_from(&mut store).unwrap();
    assert!(result.is_success());
    assert!(result.missing.is_empty());
    assert!(result.unused.is_empty());
    loaded.forward(input).into_data().assert_eq(&expected, true);

    let saved = model.collect(None, None, false);
    let restored = loaded.collect(None, None, false);
    assert_eq!(saved.len(), restored.len());
    for (saved, restored) in saved.iter().zip(&restored) {
        assert_eq!(saved.name, restored.name);
        burn_store::bridge::to_data(restored)
            .unwrap()
            .assert_eq(&burn_store::bridge::to_data(saved).unwrap(), true);
    }
}

#[test]
fn pwff_prelu_checkpoint_preserves_modified_slope() {
    let config = config(ActivationConfig::PRelu(PReluConfig::new()));
    let mut model = config.init(&Device::flex());
    let Activation::PRelu(prelu) = &mut model.activation else {
        panic!("expected PReLU");
    };
    prelu.alpha = prelu.alpha.clone().map(|alpha| alpha.mul_scalar(3.0));
    assert_round_trip(model, config);
}

#[test]
fn pwff_swiglu_checkpoint_preserves_modified_projections() {
    let config = config(ActivationConfig::SwiGlu(
        SwiGluConfig::new(1, 1)
            .with_bias(true)
            .with_initializer(Initializer::Constant { value: 0.5 }),
    ));
    let mut model = config.init(&Device::flex());
    let Activation::SwiGlu(swiglu) = &mut model.activation else {
        panic!("expected SwiGLU");
    };
    for linear in [&mut swiglu.linear_inner, &mut swiglu.linear_outer] {
        linear.weight = linear.weight.clone().map(|weight| weight.mul_scalar(3.0));
        linear.bias = linear
            .bias
            .take()
            .map(|bias| bias.map(|value| value.mul_scalar(2.0)));
    }
    assert_round_trip(model, config);
}

// Match the old field layout, including the former `gelu` field name.
#[derive(Module, Debug)]
struct LegacyPositionWiseFeedForward {
    linear_inner: Linear,
    linear_outer: Linear,
    dropout: Dropout,
    gelu: Gelu,
}

#[test]
fn pwff_loads_legacy_gelu_checkpoint_without_partial_loading() {
    let device = Device::flex();
    let config = config(ActivationConfig::Gelu);
    let original = config.init(&device);
    let legacy = LegacyPositionWiseFeedForward {
        linear_inner: original.linear_inner,
        linear_outer: original.linear_outer,
        dropout: original.dropout,
        gelu: Gelu::new(),
    };
    let input = Tensor::<3>::from_data([[[-2.0], [1.0]]], &device);
    let expected = legacy
        .linear_outer
        .forward(
            legacy.dropout.forward(
                legacy
                    .gelu
                    .forward(legacy.linear_inner.forward(input.clone())),
            ),
        )
        .into_data();
    let mut store = BurnpackStore::from_bytes(None);
    legacy.save_into(&mut store).unwrap();
    let bytes = store.get_bytes().unwrap();
    let mut loaded = config
        .with_initializer(Initializer::Constant { value: 0.0 })
        .init(&device);
    let mut store = BurnpackStore::from_bytes(Some(bytes));
    assert_eq!(store.keys().unwrap().len(), 4);
    let result = loaded.load_from(&mut store).unwrap();
    assert!(result.is_success());
    assert!(result.missing.is_empty());
    assert!(result.unused.is_empty());
    loaded.forward(input).into_data().assert_eq(&expected, true);
}

#[test]
fn pwff_legacy_stateful_checkpoint_can_load_only_existing_linear_parameters() {
    let device = Device::flex();
    let config = config(ActivationConfig::PRelu(PReluConfig::new()));
    let original = config.init(&device);
    // Before the fix, PWFF saved only the two linear layers.
    let filter = PathFilter::new().with_predicate(|path, _| {
        path.starts_with("linear_inner.") || path.starts_with("linear_outer.")
    });
    let mut store = BurnpackStore::from_bytes(None).with_filter(filter.clone());
    original.save_into(&mut store).unwrap();
    let bytes = store.get_bytes().unwrap();
    let mut loaded = config
        .with_initializer(Initializer::Constant { value: 0.0 })
        .init(&device);
    // Filter only the historically absent activation, keeping strict checks on the linears.
    let mut store = BurnpackStore::from_bytes(Some(bytes)).with_filter(filter.clone());
    let result = loaded.load_from(&mut store).unwrap();
    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4);
    assert!(result.missing.is_empty());
    assert_eq!(result.skipped, ["activation.PRelu.alpha"]);
    let Activation::PRelu(prelu) = &loaded.activation else {
        panic!("expected PReLU");
    };
    assert_eq!(prelu.alpha.val().into_scalar::<f32>(), 0.25);
    let input = Tensor::<3>::from_data([[[-2.0]]], &device);
    loaded
        .forward(input.clone())
        .into_data()
        .assert_eq(&original.forward(input).into_data(), true);

    // The migration filter must still reject a missing non-activation parameter.
    let mut incomplete = BurnpackStore::from_bytes(None).with_full_path("linear_inner.weight");
    original.save_into(&mut incomplete).unwrap();
    let mut incomplete =
        BurnpackStore::from_bytes(Some(incomplete.get_bytes().unwrap())).with_filter(filter);
    assert!(loaded.load_from(&mut incomplete).is_err());
}
