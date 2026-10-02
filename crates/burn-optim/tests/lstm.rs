use burn_core::{Tensor, module::Param, prelude::Device};
use burn_nn::{Initializer, Lstm, LstmConfig, LstmState};
use burn_optim::{GradientsParams, SgdConfig};

#[test]
fn uncoupled_lstm_trains_forget_gate() {
    assert_lstm_training(false);
}

#[test]
fn coupled_lstm_trains_without_forget_gate_parameters() {
    assert_lstm_training(true);
}

fn assert_lstm_training(input_forget: bool) {
    let device = Device::default().autodiff();
    let mut lstm = LstmConfig::new(1, 1, true)
        .with_input_forget(input_forget)
        .with_initializer(Initializer::Constant { value: 0.2 })
        .init(&device);
    let mut optimizer = SgdConfig::new().init();
    let learning_rate = 0.1;

    for _ in 0..2 {
        assert_eq!(lstm.forget_gate.is_none(), input_forget);
        let input = Tensor::from_floats([[[0.5], [0.7]]], &device);
        // Nonzero initial states exercise both forget transformations immediately.
        let initial_state = LstmState::new(
            Tensor::from_floats([[0.3]], &device),
            Tensor::from_floats([[0.4]], &device),
        );
        let (output, _) = lstm.forward(input, Some(initial_state));
        let grads = GradientsParams::from_grads(output.sum().backward(), &lstm);
        // Each active gate has two weights and two biases.
        assert_eq!(grads.len(), if input_forget { 12 } else { 16 });

        let expected_weights: Vec<_> = gate_weights(&lstm)
            .into_iter()
            .map(|weight| {
                let before = weight.val().into_scalar::<f32>();
                let gradient = grads
                    .get::<2>(weight.id)
                    .expect("every active gate weight should have a gradient")
                    .into_scalar::<f32>();
                assert!(gradient.abs() > 1e-6, "gate gradient should be nonzero");
                (before, before - learning_rate as f32 * gradient)
            })
            .collect();

        lstm = optimizer.step(learning_rate, lstm, grads);

        for (weight, (before, expected)) in gate_weights(&lstm).into_iter().zip(expected_weights) {
            let actual = weight.val().into_scalar::<f32>();
            assert!((actual - expected).abs() < 1e-6, "incorrect SGD update");
            assert!((actual - before).abs() > 1e-8, "gate weight should change");
        }
    }
}

fn gate_weights(lstm: &Lstm) -> Vec<&Param<Tensor<2>>> {
    [&lstm.input_gate, &lstm.output_gate, &lstm.cell_gate]
        .into_iter()
        .chain(lstm.forget_gate.as_ref())
        .flat_map(|gate| [&gate.input_transform.weight, &gate.hidden_transform.weight])
        .collect()
}
