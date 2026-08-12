use burn_core as burn;
use burn_core::module::{Module, Param};
use burn_nn::{
    Linear, LinearConfig, Relu,
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
};
use burn_onnx_export::{AxisSpec, InputSpec, OnnxExporter};
use burn_tensor::{Device, Tensor, TensorData, Tolerance};
use burn_tensor::{
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use onnx_ir::ModelProto;
use ort::{session::Session, value::Tensor as OrtTensor};
use protobuf::Message;

mod models;
use models::resnet::ResNet18;

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

#[derive(Module, Debug)]
struct Flatten;

impl Flatten {
    fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch, channels, width] = input.dims();
        input.reshape([batch, channels * width])
    }
}

#[derive(Module, Debug)]
struct SmallCnn {
    conv: Conv2d,
    activation: Relu,
    pool: MaxPool2d,
}

#[derive(Module, Debug)]
struct BilinearResize;

impl BilinearResize {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        interpolate(
            input,
            [5, 7],
            InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
        )
    }
}

#[derive(Module, Debug)]
struct NearestResize;

impl NearestResize {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        interpolate(
            input,
            [5, 7],
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    }
}

#[derive(Module, Debug)]
struct AddFull;

impl AddFull {
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let full = Tensor::full([2, 3], 2.5, &input.device());
        input + full
    }
}

impl SmallCnn {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.pool
            .forward(self.activation.forward(self.conv.forward(input)))
    }
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

#[test]
fn dynamic_reshape_matches_burn_at_third_shape() {
    let device = Device::default();
    let module = Flatten;
    let sample_values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let validation_values = (0..60).map(|value| value as f32).collect::<Vec<_>>();
    let sample = Tensor::<3>::from_data(TensorData::new(sample_values, [2, 3, 4]), &device);
    let validation = Tensor::<3>::from_data(TensorData::new(validation_values, [5, 3, 4]), &device);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch"),
        AxisSpec::Static,
        AxisSpec::Static,
    ])];

    let model = OnnxExporter::new()
        .export_dynamic(&module, sample, validation, &specs, Flatten::forward)
        .unwrap();

    let third_values = (0..84).map(|value| value as f32).collect::<Vec<_>>();
    let third = Tensor::<3>::from_data(TensorData::new(third_values.clone(), [7, 3, 4]), &device);
    let expected = module.forward(third).into_data();
    let actual = run_ort(&model, [7, 3, 4], third_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn small_cnn_matches_burn() {
    let device = Device::default();
    let module = SmallCnn {
        conv: Conv2dConfig::new([1, 2], [3, 3]).init(&device),
        activation: Relu::new(),
        pool: MaxPool2dConfig::new([2, 2]).init(),
    };
    let input_values = (0..25).map(|value| value as f32 / 10.0).collect::<Vec<_>>();
    let input =
        Tensor::<4>::from_data(TensorData::new(input_values.clone(), [1, 1, 5, 5]), &device);
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, SmallCnn::forward)
        .unwrap();

    let actual = run_ort(&model, [1, 1, 5, 5], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn interpolate_matches_burn() {
    let device = Device::default();
    let input_values = (0..12).map(|value| value as f32).collect::<Vec<_>>();

    let input =
        Tensor::<4>::from_data(TensorData::new(input_values.clone(), [1, 1, 3, 4]), &device);
    let expected = BilinearResize.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&BilinearResize, input, BilinearResize::forward)
        .unwrap();
    let actual = run_ort(&model, [1, 1, 3, 4], input_values.clone());
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));

    let input =
        Tensor::<4>::from_data(TensorData::new(input_values.clone(), [1, 1, 3, 4]), &device);
    let expected = NearestResize.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&NearestResize, input, NearestResize::forward)
        .unwrap();
    let actual = run_ort(&model, [1, 1, 3, 4], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn full_matches_burn() {
    let device = Device::default();
    let input_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::<2>::from_data(TensorData::new(input_values.clone(), [2, 3]), &device);
    let expected = AddFull.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&AddFull, input, AddFull::forward)
        .unwrap();

    let actual = run_ort(&model, [2, 3], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn dynamic_small_cnn_matches_burn_at_third_shape() {
    let device = Device::default();
    let module = SmallCnn {
        conv: Conv2dConfig::new([1, 2], [3, 3]).init(&device),
        activation: Relu::new(),
        pool: MaxPool2dConfig::new([2, 2]).init(),
    };
    let sample = Tensor::<4>::from_data(TensorData::zeros::<f32, _>([1, 1, 5, 5]), &device);
    let validation = Tensor::<4>::from_data(TensorData::zeros::<f32, _>([2, 1, 7, 7]), &device);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch"),
        AxisSpec::Static,
        AxisSpec::dynamic("height"),
        AxisSpec::dynamic("width"),
    ])];
    let model = OnnxExporter::new()
        .export_dynamic(&module, sample, validation, &specs, SmallCnn::forward)
        .unwrap();

    let third_values = (0..243)
        .map(|value| value as f32 / 100.0)
        .collect::<Vec<_>>();
    let third =
        Tensor::<4>::from_data(TensorData::new(third_values.clone(), [3, 1, 9, 9]), &device);
    let expected = module.forward(third).into_data();
    let actual = run_ort(&model, [3, 1, 9, 9], third_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn resnet18_matches_burn() {
    let device = Device::default();
    let module = ResNet18::new(10, &device);
    let input_values = (0..3 * 64 * 64)
        .map(|value| (value % 251) as f32 / 251.0)
        .collect::<Vec<_>>();
    let input = Tensor::<4>::from_data(
        TensorData::new(input_values.clone(), [1, 3, 64, 64]),
        &device,
    );
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, ResNet18::forward)
        .unwrap();

    let model_proto = ModelProto::parse_from_bytes(&model).unwrap();
    let batch_norm_count = model_proto
        .graph
        .node
        .iter()
        .filter(|node| node.op_type == "BatchNormalization")
        .count();
    assert_eq!(batch_norm_count, 20);

    let actual = run_ort(&model, [1, 3, 64, 64], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(1.0e-3, 1.0e-5));
}
