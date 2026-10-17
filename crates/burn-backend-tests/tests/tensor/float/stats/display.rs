use super::*;
use burn_tensor::{DType, Element, Shape, TensorData};

// Floating point values might not match for other precisions
fn skip_precision_not_f32() -> bool {
    core::any::TypeId::of::<FloatElem>() != core::any::TypeId::of::<f32>()
}

#[test]
fn test_display_2d_int_tensor() {
    let int_data = TensorData::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let tensor_int = TestTensorInt::<2>::from_data(int_data, &Default::default());

    let output = format!("{}", tensor_int);
    let expected = format!(
        r#"Tensor {{
  data:
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]],
  shape:  [3, 3],
  device:  {:?},
  kind:  "Int",
  dtype:  "{dtype}",
}}"#,
        tensor_int.device(),
        dtype = core::any::type_name::<IntElem>(),
    );
    assert_eq!(output, expected);
}

#[test]
fn test_display_2d_float_tensor() {
    if skip_precision_not_f32() {
        return;
    }

    let float_data = TensorData::from([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]);
    let tensor_float = TestTensor::<2>::from_data(float_data, &Default::default());

    let output = format!("{}", tensor_float);
    let expected = format!(
        r#"Tensor {{
  data:
[[1.1, 2.2, 3.3],
 [4.4, 5.5, 6.6],
 [7.7, 8.8, 9.9]],
  shape:  [3, 3],
  device:  {:?},
  kind:  "Float",
  dtype:  "f32",
}}"#,
        tensor_float.device(),
    );
    assert_eq!(output, expected);
}

#[test]
fn test_display_2d_bool_tensor() {
    let device = Default::default();
    let bool_data = TensorData::from([
        [true, false, true],
        [false, true, false],
        [false, true, true],
    ]);
    let tensor_bool = TestTensorBool::<2>::from_data(bool_data, &device);

    let output = format!("{}", tensor_bool);
    // TODO: remove once backends no longer rely on generics for default elem types
    let expected_dtype_name = match device.settings().bool_dtype {
        burn_tensor::BoolDType::Native => DType::Bool(burn_tensor::BoolStore::Native).name(),
        burn_tensor::BoolDType::U8 => DType::Bool(burn_tensor::BoolStore::U8).name(),
        burn_tensor::BoolDType::U32 => DType::Bool(burn_tensor::BoolStore::U32).name(),
    };
    let expected = format!(
        r#"Tensor {{
  data:
[[true, false, true],
 [false, true, false],
 [false, true, true]],
  shape:  [3, 3],
  device:  {:?},
  kind:  "Bool",
  dtype:  {:?},
}}"#,
        tensor_bool.device(),
        expected_dtype_name,
    );
    assert_eq!(output, expected);
}

#[test]
fn test_display_3d_tensor() {
    let data = TensorData::from([
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ]);
    let tensor = TestTensorInt::<3>::from_data(data, &Default::default());

    let output = format!("{}", tensor);
    let expected = format!(
        r#"Tensor {{
  data:
[[[1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12]],
 [[13, 14, 15, 16],
  [17, 18, 19, 20],
  [21, 22, 23, 24]]],
  shape:  [2, 3, 4],
  device:  {:?},
  kind:  "Int",
  dtype:  "{dtype}",
}}"#,
        tensor.device(),
        dtype = core::any::type_name::<IntElem>(),
    );
    assert_eq!(output, expected);
}

#[test]
fn test_display_4d_tensor() {
    let data = TensorData::from([
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
    ]);

    let tensor = TestTensorInt::<4>::from_data(data, &Default::default());

    let output = format!("{}", tensor);
    let expected = format!(
        r#"Tensor {{
  data:
[[[[1, 2, 3],
   [4, 5, 6]],
  [[7, 8, 9],
   [10, 11, 12]]],
 [[[13, 14, 15],
   [16, 17, 18]],
  [[19, 20, 21],
   [22, 23, 24]]]],
  shape:  [2, 2, 2, 3],
  device:  {:?},
  kind:  "Int",
  dtype:  "{dtype}",
}}"#,
        tensor.device(),
        dtype = core::any::type_name::<IntElem>(),
    );
    assert_eq!(output, expected);
}

#[test]
fn test_display_tensor_summarize_1() {
    let tensor = TestTensor::<4>::zeros(Shape::new([2, 2, 2, 1000]), &Default::default());

    let output = format!("{}", tensor);
    let expected = format!(
        r#"Tensor {{
  data:
[[[[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]]],
 [[[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]]]],
  shape:  [2, 2, 2, 1000],
  device:  {:?},
  kind:  "Float",
  dtype:  "{dtype}",
}}"#,
        tensor.device(),
        dtype = FloatElem::dtype().name(),
    );
    assert_eq!(output, expected);
}

#[test]
fn test_display_tensor_summarize_2() {
    let tensor = TestTensor::<4>::zeros(Shape::new([2, 2, 20, 100]), &Default::default());

    let output = format!("{}", tensor);
    let expected = format!(
        r#"Tensor {{
  data:
[[[[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]]],
 [[[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]]]],
  shape:  [2, 2, 20, 100],
  device:  {:?},
  kind:  "Float",
  dtype:  "{dtype}",
}}"#,
        tensor.device(),
        dtype = FloatElem::dtype().name(),
    );
    assert_eq!(output, expected);
}

#[test]
fn test_display_tensor_summarize_3() {
    let tensor = TestTensor::<4>::zeros(Shape::new([2, 2, 200, 6]), &Default::default());

    let output = format!("{}", tensor);
    let expected = format!(
        r#"Tensor {{
  data:
[[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
 [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   ...
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]],
  shape:  [2, 2, 200, 6],
  device:  {:?},
  kind:  "Float",
  dtype:  "{dtype}",
}}"#,
        tensor.device(),
        dtype = FloatElem::dtype().name(),
    );
    assert_eq!(output, expected);
}
#[test]
fn test_display_precision() {
    if skip_precision_not_f32() {
        return;
    }

    let tensor = TestTensor::<2>::full([1, 1], 0.123456789, &Default::default());

    let output = format!("{}", tensor);
    let expected = format!(
        r#"Tensor {{
  data:
[[0.12345679]],
  shape:  [1, 1],
  device:  {:?},
  kind:  "Float",
  dtype:  "f32",
}}"#,
        tensor.device(),
    );
    assert_eq!(output, expected);

    // CAN'T DO THIS BECAUSE OF GLOBAL STATE
    // let print_options = PrintOptions {
    //     precision: Some(3),
    //     ..Default::default()
    // };
    // set_print_options(print_options);

    let tensor = TestTensor::<2>::full([3, 2], 0.123456789, &Default::default());

    // Set precision to 3
    let output = format!("{:.3}", tensor);

    let expected = format!(
        r#"Tensor {{
  data:
[[0.123, 0.123],
 [0.123, 0.123],
 [0.123, 0.123]],
  shape:  [3, 2],
  device:  {:?},
  kind:  "Float",
  dtype:  "f32",
}}"#,
        tensor.device(),
    );
    assert_eq!(output, expected);
}

// Quantized tensors display identity metadata only: values are stored as codes with scales,
// so `Display` points to `dequantize()` / `to_data()` instead of printing them.
#[cfg(feature = "quantization")]
mod quantized {
    use super::super::super::quantization::qtensor::QTensor;
    use super::*;
    use burn_tensor::Device;
    use burn_tensor::quantization::QuantValue;

    const BLOCK_DATA: [[f32; 16]; 2] = [
        [
            -1.8, -1.0, 0.0, 0.5, -1.8, -1.0, 0.0, 0.5, 0.01, 0.025, 0.03, 0.04, 0.01, 0.025, 0.03,
            0.04,
        ],
        [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ],
    ];

    /// Asserts the full metadata-only display output of a quantized tensor.
    fn assert_display<const D: usize>(tensor: &TestTensor<D>, shape: &str) {
        let scheme = match tensor.dtype() {
            DType::QFloat(scheme) => scheme,
            _ => panic!("Expected a quantized dtype"),
        };
        let expected = format!(
            r#"Tensor {{
  data:  <quantized, use .dequantize() to view values>,
  shape:  {shape},
  device:  {:?},
  kind:  "Float",
  dtype:  "qfloat",
  scheme:  {scheme:?},
}}"#,
            tensor.device(),
        );
        assert_eq!(format!("{}", tensor), expected);
    }

    #[test]
    fn test_display_quantized_tensor() {
        let tensor = QTensor::<2>::int8([[-127.0, 0.0, 64.0, 127.0], [1.0, -2.0, 3.0, -4.0]]);
        assert_display(&tensor, "[2, 4]");
    }

    #[test]
    fn test_display_quantized_block_tensor() {
        let tensor = QTensor::<2>::int8_block(BLOCK_DATA);
        assert_display(&tensor, "[2, 16]");
    }

    #[test]
    fn test_display_quantized_large_tensor() {
        // Above the print threshold: metadata-only display reads no data and never summarizes.
        let tensor = QTensor::<2>::int8_block(TensorData::new(vec![0.5f32; 1024], [64usize, 16]));
        assert_display(&tensor, "[64, 16]");
    }

    #[test]
    fn test_display_quantized_data_default_scheme() {
        // Built from quantized data since `quantize_dynamic` doesn't support every scheme on
        // every backend.
        let device = Device::default();
        let data = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            device.settings().quantization.scheme,
            &[0.014_173_228],
            None,
        );
        assert_display(&TestTensor::<1>::from_data(data, &device), "[4]");
    }

    #[test]
    fn test_display_quantized_precision() {
        // Trailing dim must be divisible by 4 for packed storage on cubecl backends.
        let tensor = QTensor::<1>::int8([1.0, 2.0, 3.0, 4.0]);
        // Precision only affects float formatting; metadata display is unchanged.
        assert_eq!(format!("{:.3}", tensor), format!("{}", tensor));
    }

    #[test]
    fn test_display_quantized_data_float_codes() {
        // Float-code schemes have no readable codes; `TensorData` display must not panic.
        let device = Device::default();
        let data = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            device
                .settings()
                .quantization
                .scheme
                .with_value(QuantValue::E4M3),
            &[0.014_173_228],
            None,
        );
        let scheme = match data.dtype() {
            DType::QFloat(scheme) => scheme,
            _ => panic!("Expected a quantized dtype"),
        };
        assert_eq!(
            format!("{}", data),
            format!("<float-quantized> {:?}", scheme)
        );
    }
}
