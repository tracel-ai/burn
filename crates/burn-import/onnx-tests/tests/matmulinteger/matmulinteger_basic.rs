
use std::path::PathBuf;
// adjust imports to your actual loader/eval API:
use burn_import::onnx::{load_model, run_model}; // placeholder

fn p(rel: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel).to_string_lossy().into_owned()
}

#[test]
fn matmulinteger_basic_works() {
    let model_path = p("tests/matmulinteger/mmi_basic_inputs.onnx"); // or mmi_const_cast.onnx
    let model = load_model(&model_path).expect("load");

    // A, B as u8
    let a = [[1u8,2,3,4],[10,20,30,40]];
    let b = [[1u8,2,3],[4,5,6],[7,8,9],[10,11,12]];
    // zps (only needed for the *inputs* model)
    let a_zp = [0u8]; let b_zp = [0u8];

    // adapt to your tensor creation & runner
    let outputs = run_model(&model, &[
        ("A", a), ("B", b),
        ("a_zp", a_zp), ("b_zp", b_zp), // omit these two if you use the Const→Cast model
    ]).unwrap();
    let y: Vec<Vec<i32>> = outputs["Y"].clone(); // adapt

    assert_eq!(y, vec![vec![70,80,90], vec![700,800,900]]);
}