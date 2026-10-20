use burn_core::tensor::{Device, Tensor, TensorData, Tolerance};
use burn_nn::{
    Initializer,
    activation::{Activation, ActivationConfig, PReluConfig, SwiGluConfig},
    transformer::PositionWiseFeedForwardConfig,
};
use burn_optim::{GradientsParams, SgdConfig};

#[test]
fn pwff_prelu_parameter_receives_gradient_and_optimizer_update() {
    let device = Device::default().autodiff();
    let model = PositionWiseFeedForwardConfig::new(1, 1)
        .with_dropout(0.0)
        .with_initializer(Initializer::Constant { value: 1.0 })
        .with_activation(ActivationConfig::PRelu(PReluConfig::new()))
        .init(&device);
    let Activation::PRelu(prelu) = &model.activation else {
        panic!("expected PReLU");
    };
    let alpha = prelu.alpha.val();
    // The inner linear layer produces -1, so d(output)/d(alpha) is -1.
    let input = Tensor::<3>::from_data([[[-2.0]]], &device);
    let grads = model.forward(input).sum().backward();
    alpha
        .grad(&grads)
        .expect("PReLU must receive a gradient")
        .into_data()
        .assert_eq(&TensorData::from([-1.0f32]), true);

    let grads = GradientsParams::from_grads(grads, &model);
    let model = SgdConfig::new().init().step(0.1, model, grads);
    let Activation::PRelu(prelu) = model.activation else {
        panic!("expected PReLU");
    };
    prelu
        .alpha
        .val()
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::from([0.35f32]), Tolerance::default());
}

#[test]
fn pwff_swiglu_projections_receive_gradients_and_optimizer_updates() {
    let device = Device::default().autodiff();
    let model = PositionWiseFeedForwardConfig::new(1, 1)
        .with_dropout(0.0)
        .with_initializer(Initializer::Constant { value: 1.0 })
        .with_activation(ActivationConfig::SwiGlu(
            SwiGluConfig::new(1, 1)
                .with_bias(true)
                .with_initializer(Initializer::Constant { value: 0.5 }),
        ))
        .init(&device);
    let Activation::SwiGlu(swiglu) = &model.activation else {
        panic!("expected SwiGLU");
    };
    let weights = [
        swiglu.linear_inner.weight.clone(),
        swiglu.linear_outer.weight.clone(),
    ];
    let biases = [
        swiglu.linear_inner.bias.clone().unwrap(),
        swiglu.linear_outer.bias.clone().unwrap(),
    ];
    let input = Tensor::<3>::from_data([[[1.0]]], &device);
    let grads = model.forward(input).sum().backward();
    for weight in &weights {
        assert!(weight.val().grad(&grads).is_some());
    }
    for bias in &biases {
        assert!(bias.val().grad(&grads).is_some());
    }
    let grads = GradientsParams::from_grads(grads, &model);

    let model = SgdConfig::new().init().step(0.1, model, grads);
    let Activation::SwiGlu(swiglu) = model.activation else {
        panic!("expected SwiGLU");
    };
    for (after, before) in [swiglu.linear_inner, swiglu.linear_outer]
        .into_iter()
        .zip(weights.into_iter().zip(biases))
    {
        assert!(
            (after.weight.val() - before.0.val())
                .abs()
                .sum()
                .into_scalar::<f32>()
                > 0.0
        );
        assert!(
            (after.bias.unwrap().val() - before.1.val())
                .abs()
                .sum()
                .into_scalar::<f32>()
                > 0.0
        );
    }
}
