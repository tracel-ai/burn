use burn_core as burn;
use burn_core::module::{Module, Param};
use burn_nn::{Linear, LinearConfig, Relu};
use burn_onnx_export::OnnxExporter;
use burn_tensor::{Device, Tensor, TensorData, Tolerance};
use ort::{session::Session, value::Tensor as OrtTensor};

const RTOL: f32 = 1.0e-4;
const ATOL: f32 = 1.0e-5;

#[derive(Module, Debug)]
struct AddModule {
    weight: Param<Tensor<1>>,
}

impl AddModule {
    fn forward(&self, input: Tensor<1>) -> Tensor<1> {
        input + self.weight.val()
    }
}

#[derive(Module, Debug)]
struct Mlp {
    first: Linear,
    activation: Relu,
    second: Linear,
}

impl Mlp {
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.second
            .forward(self.activation.forward(self.first.forward(input)))
    }
}

fn run_ort(model: &[u8], shape: impl Into<Vec<i64>>, input: Vec<f32>) -> TensorData {
    let mut session = Session::builder()
        .unwrap()
        .commit_from_memory(model)
        .unwrap();
    let input = OrtTensor::from_array((shape.into(), input)).unwrap();
    let outputs = session.run(ort::inputs![input]).unwrap();
    let (shape, values) = outputs[0].try_extract_tensor::<f32>().unwrap();
    TensorData::new(
        values.to_vec(),
        shape
            .iter()
            .map(|dimension| *dimension as usize)
            .collect::<Vec<_>>(),
    )
}

#[test]
fn add_matches_burn() {
    let device = Device::default();
    let module = AddModule {
        weight: Param::from_data([2.0f32, 3.0], &device),
    };
    let input_values = vec![5.0f32, 7.0];
    let input = Tensor::<1>::from_floats(input_values.as_slice(), &device);
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, AddModule::forward)
        .unwrap();

    let actual = run_ort(&model, [2], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn small_mlp_matches_burn() {
    let device = Device::default();
    let module = Mlp {
        first: LinearConfig::new(4, 3).init(&device),
        activation: Relu::new(),
        second: LinearConfig::new(3, 2).init(&device),
    };
    let input_values = vec![1.0f32, 2.0, 3.0, 4.0, -2.0, 0.5, 1.5, 3.0];
    let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.5, 3.0]], &device);
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, Mlp::forward)
        .unwrap();

    let actual = run_ort(&model, [2, 4], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}
