use burn_core as burn;

use crate::{GateController, Initializer, Lstm, LstmConfig, LstmState};
use burn::module::Module;
use burn::store::{ModuleRecord, RecordError};
use burn::tensor::{Device, Tensor, Tolerance};

// The old checkpoint layout stored all four gates as concrete modules, even in coupled mode.
#[derive(Module, Debug)]
struct LegacyLstm {
    input_gate: GateController,
    forget_gate: GateController,
    output_gate: GateController,
    cell_gate: GateController,
}

impl LegacyLstm {
    fn new(bias: bool, device: &Device) -> Self {
        let gate = |value| GateController::new(2, 3, bias, Initializer::Constant { value }, device);
        Self {
            input_gate: gate(0.1),
            forget_gate: gate(0.2),
            output_gate: gate(0.3),
            cell_gate: gate(0.4),
        }
    }

    fn as_lstm(&self, input_forget: bool, device: &Device) -> Lstm {
        let bias = self.input_gate.input_transform.bias.is_some();
        let mut lstm = LstmConfig::new(2, 3, bias)
            .with_input_forget(input_forget)
            .with_initializer(Initializer::Zeros)
            .init(device);
        lstm.input_gate = self.input_gate.clone();
        // Keep the redundant gate to reproduce the old coupled model's layout and behavior.
        lstm.forget_gate = Some(self.forget_gate.clone());
        lstm.output_gate = self.output_gate.clone();
        lstm.cell_gate = self.cell_gate.clone();
        lstm
    }
}

fn assert_same_output(expected: &Lstm, actual: &Lstm, device: &Device) {
    let input = Tensor::<3>::from_data([[[0.2, -0.1], [0.4, 0.3]]], device);
    let state = LstmState::new(
        Tensor::from_data([[0.3, -0.2, 0.1]], device),
        Tensor::from_data([[0.1, 0.2, -0.3]], device),
    );
    let (expected_output, expected_state) = expected.forward(input.clone(), Some(state.clone()));
    let (actual_output, actual_state) = actual.forward(input, Some(state));
    let tolerance = Tolerance::default();
    actual_output
        .to_data()
        .assert_approx_eq::<f32>(&expected_output.to_data(), tolerance);
    actual_state
        .cell
        .to_data()
        .assert_approx_eq::<f32>(&expected_state.cell.to_data(), tolerance);
    actual_state
        .hidden
        .to_data()
        .assert_approx_eq::<f32>(&expected_state.hidden.to_data(), tolerance);
}

#[test]
fn both_modes_round_trip_with_only_their_active_gate_tensors() {
    let device = Device::default();
    for input_forget in [false, true] {
        for bias in [false, true] {
            let config = LstmConfig::new(2, 3, bias).with_input_forget(input_forget);
            let source = config
                .clone()
                .with_initializer(Initializer::Constant { value: 0.2 })
                .init(&device);
            let record = source.clone().into_record();
            let num_gates = if input_forget { 3 } else { 4 };
            let tensors_per_gate = if bias { 4 } else { 2 };
            assert_eq!(record.len(), num_gates * tensors_per_gate);

            let record = ModuleRecord::from_bytes(record.into_bytes().unwrap()).unwrap();
            assert_eq!(record.len(), num_gates * tensors_per_gate);
            let loaded = config
                .with_initializer(Initializer::Zeros)
                .init(&device)
                .try_load_record(record)
                .unwrap();

            assert_eq!(loaded.input_forget, input_forget);
            assert_eq!(loaded.forget_gate.is_none(), input_forget);
            assert_eq!(loaded.num_params(), source.num_params());
            assert_same_output(&source, &loaded, &device);
        }
    }
}

#[test]
fn legacy_uncoupled_checkpoint_loads_without_migration() {
    let device = Device::default();
    for bias in [false, true] {
        let legacy = LegacyLstm::new(bias, &device);
        let expected = legacy.as_lstm(false, &device);
        let bytes = legacy.into_record().into_bytes().unwrap();
        let record = ModuleRecord::from_bytes(bytes).unwrap();
        let loaded = LstmConfig::new(2, 3, bias)
            .with_initializer(Initializer::Zeros)
            .init(&device)
            .try_load_record(record)
            .unwrap();

        assert!(!loaded.input_forget);
        assert!(loaded.forget_gate.is_some());
        assert_same_output(&expected, &loaded, &device);
    }
}

#[test]
fn legacy_coupled_checkpoint_requires_allow_unused_and_preserves_outputs() {
    let device = Device::default();
    for bias in [false, true] {
        let legacy = LegacyLstm::new(bias, &device);
        let expected = legacy.as_lstm(true, &device);
        let bytes = legacy.into_record().into_bytes().unwrap();
        let record = ModuleRecord::from_bytes(bytes).unwrap();
        let target = LstmConfig::new(2, 3, bias)
            .with_input_forget(true)
            .with_initializer(Initializer::Zeros)
            .init(&device);

        let Err(RecordError::Validation(message)) = target.clone().try_load_record(record.clone())
        else {
            panic!("legacy forget tensors must require an explicit migration");
        };
        assert!(message.contains("Unused tensors"));
        assert!(message.contains("forget_gate.input_transform.weight"));
        assert!(message.contains("forget_gate.hidden_transform.weight"));

        let loaded = target.try_load_record(record.allow_unused(true)).unwrap();
        assert!(loaded.input_forget);
        assert!(loaded.forget_gate.is_none());
        assert_same_output(&expected, &loaded, &device);
        assert_eq!(loaded.into_record().len(), if bias { 12 } else { 6 });
    }
}

#[test]
fn coupled_checkpoint_cannot_silently_initialize_an_uncoupled_forget_gate() {
    let device = Device::default();
    let record = LstmConfig::new(2, 3, true)
        .with_input_forget(true)
        .init(&device)
        .into_record();
    let result = LstmConfig::new(2, 3, true)
        .init(&device)
        .try_load_record(record);
    let Err(RecordError::Validation(message)) = result else {
        panic!("an uncoupled LSTM requires forget-gate parameters in the checkpoint");
    };
    assert!(message.contains("Missing tensors"));
    assert!(message.contains("forget_gate.input_transform.weight"));
    assert!(message.contains("forget_gate.hidden_transform.weight"));
}
