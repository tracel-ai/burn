use std::{
    io::Write,
    process::{Command, Stdio},
};

use burn_core as burn;
use burn_core::module::{Module, Param};
use burn_onnx_export::{AxisSpec, InputSpec, OnnxExporter};
use burn_tensor::{Device, Tensor, TensorData};

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
struct Flatten;

impl Flatten {
    fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch, channels, width] = input.dims();
        input.reshape([batch, channels * width])
    }
}

fn checker() -> Option<String> {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let available = Command::new(&python)
        .args(["-c", "import onnx"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !available {
        if std::env::var_os("BURN_ONNX_PYTHON_REQUIRED").is_some() {
            panic!("Python package `onnx` is required for this test");
        }
        eprintln!("skipping: Python package `onnx` is unavailable");
        return None;
    }
    Some(python)
}

fn check_model(python: String, model: &[u8]) {
    let mut child = Command::new(python)
        .args([
            "-c",
            "import sys, onnx; onnx.checker.check_model(onnx.load_model_from_string(sys.stdin.buffer.read()), full_check=True)",
        ])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(model).unwrap();
    let status = child.wait().unwrap();
    assert!(status.success(), "ONNX checker rejected exported model");
}

#[test]
fn checker_accepts_exported_forward() {
    let Some(python) = checker() else { return };
    let device = Device::default();
    let module = AddModule {
        weight: Param::from_data([2.0f32, 3.0], &device),
    };
    let input = Tensor::<1>::from_floats([5.0f32, 7.0], &device);
    let model = OnnxExporter::new()
        .export(&module, input, AddModule::forward)
        .unwrap();
    check_model(python, &model);
}

#[test]
fn checker_accepts_dynamic_reshape() {
    let Some(python) = checker() else { return };
    let device = Device::default();
    let sample = Tensor::<3>::from_data(TensorData::zeros::<f32, _>([2, 3, 4]), &device);
    let validation = Tensor::<3>::from_data(TensorData::zeros::<f32, _>([5, 3, 4]), &device);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch", 5),
        AxisSpec::Static,
        AxisSpec::Static,
    ])];
    let model = OnnxExporter::new()
        .export_dynamic(&Flatten, sample, validation, &specs, Flatten::forward)
        .unwrap();
    check_model(python, &model);
}
