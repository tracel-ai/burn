#[burn_tensor_testgen::testgen(display)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Shape, Tensor};

    type FloatElem = <TestBackend as Backend>::FloatElem;
    type IntElem = <TestBackend as Backend>::IntElem;

    #[test]
    fn test_display_2d_int_tensor() {
        let int_data = Data::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let tensor_int: burn_tensor::Tensor<TestBackend, 2, burn_tensor::Int> =
            Tensor::from_data(int_data);

        let output = format!("{}", tensor_int);
        let expected = format!(
            r#"Tensor {{
  data:
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]],
  shape:  [3, 3],
  device:  {:?},
  backend:  {:?},
  kind:  "Int",
  dtype:  "{dtype}",
}}"#,
            tensor_int.device(),
            TestBackend::name(),
            dtype = core::any::type_name::<IntElem>(),
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_2d_float_tensor() {
        let float_data = Data::from([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]);
        let tensor_float: burn_tensor::Tensor<TestBackend, 2, burn_tensor::Float> =
            Tensor::from_data(float_data);

        let output = format!("{}", tensor_float);
        let expected = format!(
            r#"Tensor {{
  data:
[[1.1, 2.2, 3.3],
 [4.4, 5.5, 6.6],
 [7.7, 8.8, 9.9]],
  shape:  [3, 3],
  device:  {:?},
  backend:  {:?},
  kind:  "Float",
  dtype:  "{dtype}",
}}"#,
            tensor_float.device(),
            TestBackend::name(),
            dtype = core::any::type_name::<FloatElem>(),
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_2d_bool_tensor() {
        let bool_data = Data::from([
            [true, false, true],
            [false, true, false],
            [false, true, true],
        ]);
        let tensor_bool: burn_tensor::Tensor<TestBackend, 2, burn_tensor::Bool> =
            Tensor::from_data(bool_data);

        let output = format!("{}", tensor_bool);
        let expected = format!(
            r#"Tensor {{
  data:
[[true, false, true],
 [false, true, false],
 [false, true, true]],
  shape:  [3, 3],
  device:  {:?},
  backend:  {:?},
  kind:  "Bool",
  dtype:  "bool",
}}"#,
            tensor_bool.device(),
            TestBackend::name(),
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_3d_tensor() {
        let data = Data::from([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ]);
        let tensor: burn_tensor::Tensor<TestBackend, 3, burn_tensor::Int> = Tensor::from_data(data);

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
  backend:  {:?},
  kind:  "Int",
  dtype:  "{dtype}",
}}"#,
            tensor.device(),
            TestBackend::name(),
            dtype = core::any::type_name::<IntElem>(),
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_4d_tensor() {
        let data = Data::from([
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ]);

        let tensor: burn_tensor::Tensor<TestBackend, 4, burn_tensor::Int> = Tensor::from_data(data);

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
  backend:  {:?},
  kind:  "Int",
  dtype:  "{dtype}",
}}"#,
            tensor.device(),
            TestBackend::name(),
            dtype = core::any::type_name::<IntElem>(),
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_tensor_summarize_1() {
        let tensor: burn_tensor::Tensor<TestBackend, 4, burn_tensor::Float> =
            Tensor::zeros(Shape::new([2, 2, 2, 1000]));

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
  backend:  {:?},
  kind:  "Float",
  dtype:  "f32",
}}"#,
            tensor.device(),
            TestBackend::name(),
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_tensor_summarize_2() {
        let tensor: burn_tensor::Tensor<TestBackend, 4, burn_tensor::Float> =
            Tensor::zeros(Shape::new([2, 2, 20, 100]));

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
  backend:  {:?},
  kind:  "Float",
  dtype:  "f32",
}}"#,
            tensor.device(),
            TestBackend::name(),
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_tensor_summarize_3() {
        let tensor: burn_tensor::Tensor<TestBackend, 4, burn_tensor::Float> =
            Tensor::zeros(Shape::new([2, 2, 200, 6]));

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
  backend:  {:?},
  kind:  "Float",
  dtype:  "f32",
}}"#,
            tensor.device(),
            TestBackend::name(),
        );
        assert_eq!(output, expected);
    }
}
