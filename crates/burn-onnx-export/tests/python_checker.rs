use std::{
    io::Write,
    process::{Command, Stdio},
};

use burn_core as burn;
use burn_core::module::{Module, Param};
use burn_onnx_export::OnnxExporter;
use burn_tensor::{Device, Tensor};

#[derive(Module, Debug)]
struct AddModule {
    weight: Param<Tensor<1>>,
}

#[test]
fn checker_accepts_exported_forward() {
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
        return;
    }

    let device = Device::default();
    let module = AddModule {
        weight: Param::from_data([2.0f32, 3.0], &device),
    };
    let input = Tensor::<1>::from_floats([5.0f32, 7.0], &device);
    let model = OnnxExporter::new()
        .export(&module, input, |module, input| input + module.weight.val())
        .unwrap();

    let mut child = Command::new(python)
        .args([
            "-c",
            "import sys, onnx; onnx.checker.check_model(onnx.load_model_from_string(sys.stdin.buffer.read()), full_check=True)",
        ])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&model).unwrap();
    let status = child.wait().unwrap();
    assert!(status.success(), "ONNX checker rejected exported model");
}
