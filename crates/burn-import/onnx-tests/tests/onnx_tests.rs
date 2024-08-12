#![no_std]

/// Include generated models in the `model` directory in the target directory.
macro_rules! include_models {
    ($($model:ident),*) => {
        $(
            // Allow type complexity for generated code
            #[allow(clippy::type_complexity)]
            pub mod $model {
                include!(concat!(env!("OUT_DIR"), concat!("/model/", stringify!($model), ".rs")));
            }
        )*
    };
}

// ATTENTION: Modify this macro to include all models in the `model` directory.
include_models!(
    add,
    add_int,
    argmax,
    avg_pool1d,
    avg_pool2d,
    batch_norm,
    cast,
    clip_opset16,
    clip_opset7,
    concat,
    constant_of_shape,
    constant_of_shape_full_like,
    conv1d,
    conv2d,
    conv3d,
    conv_transpose2d,
    conv_transpose3d,
    cos,
    div,
    dropout_opset16,
    dropout_opset7,
    equal,
    erf,
    exp,
    expand,
    flatten,
    gather,
    gather_scalar,
    gather_elements,
    gelu,
    global_avr_pool,
    greater,
    greater_scalar,
    greater_or_equal,
    greater_or_equal_scalar,
    hard_sigmoid,
    layer_norm,
    leaky_relu,
    less,
    less_scalar,
    less_or_equal,
    less_or_equal_scalar,
    linear,
    log,
    log_softmax,
    mask_where,
    matmul,
    max,
    maxpool1d,
    maxpool2d,
    min,
    mean,
    mul,
    neg,
    not,
    pad,
    pow,
    pow_int,
    prelu,
    random_normal,
    random_uniform,
    range,
    recip,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_prod,
    reduce_sum_opset11,
    reduce_sum_opset13,
    relu,
    reshape,
    resize_with_sizes,
    resize_1d_linear_scale,
    resize_1d_nearest_scale,
    resize_2d_bicubic_scale,
    resize_2d_bilinear_scale,
    resize_2d_nearest_scale,
    shape,
    sigmoid,
    sign,
    sin,
    slice,
    softmax,
    sqrt,
    squeeze_multiple,
    squeeze_opset13,
    squeeze_opset16,
    sub,
    sub_int,
    sum,
    sum_int,
    tanh,
    tile,
    transpose,
    unsqueeze,
    unsqueeze_opset11,
    unsqueeze_opset16
);

#[cfg(test)]
mod tests {
    use core::f64::consts;

    use super::*;

    use burn::tensor::{Bool, Int, Shape, Tensor, TensorData};

    use float_cmp::ApproxEq;

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn add_scalar_to_tensor_and_tensor_to_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: add::Model<Backend> = add::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 2f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[9f32, 10., 11., 12.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_scalar_to_int_tensor_and_int_tensor_to_int_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: add_int::Model<Backend> = add_int::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let scalar = 2;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[9i64, 11, 13, 15]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sub_scalar_from_tensor_and_tensor_from_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: sub::Model<Backend> = sub::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 3.0f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[-12f32, -13., -14., -15.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sub_scalar_from_int_tensor_and_int_tensor_from_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: sub_int::Model<Backend> = sub_int::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let scalar = 3;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[-12i64, -12, -12, -12]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sum_tensor_and_tensor() {
        let device = Default::default();
        let model: sum::Model<Backend> = sum::Model::default();

        let input1 = Tensor::<Backend, 1>::from_floats([1., 2., 3., 4.], &device);
        let input2 = Tensor::<Backend, 1>::from_floats([1., 2., 3., 4.], &device);
        let input3 = Tensor::<Backend, 1>::from_floats([1., 2., 3., 4.], &device);

        let output = model.forward(input1, input2, input3);
        let expected = TensorData::from([3f32, 6., 9., 12.]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sum_int_tensor_and_int_tensor() {
        let device = Default::default();
        let model: sum_int::Model<Backend> = sum_int::Model::default();

        let input1 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input2 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input3 = Tensor::<Backend, 1, Int>::from_ints([1, 2, 3, 4], &device);

        let output = model.forward(input1, input2, input3);
        let expected = TensorData::from([3i64, 6, 9, 12]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mean_tensor_and_tensor() {
        let device = Default::default();
        let model: mean::Model<Backend> = mean::Model::default();

        let input1 = Tensor::<Backend, 1>::from_floats([1., 2., 3., 4.], &device);
        let input2 = Tensor::<Backend, 1>::from_floats([2., 2., 4., 0.], &device);
        let input3 = Tensor::<Backend, 1>::from_floats([3., 2., 5., -4.], &device);

        let output = model.forward(input1, input2, input3);
        let expected = TensorData::from([2.0f32, 2., 4., 0.]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mul_scalar_with_tensor_and_tensor_with_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: mul::Model<Backend> = mul::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 6.0f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[126f32, 252., 378., 504.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn div_tensor_by_scalar_and_tensor_by_tensor() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: div::Model<Backend> = div::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[3., 6., 6., 9.]]]], &device);
        let scalar1 = 9.0f64;
        let scalar2 = 3.0f64;
        let output = model.forward(input, scalar1, scalar2);
        let expected = TensorData::from([[[[1f32, 2., 2., 3.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn matmul() {
        // Initialize the model with weights (loaded from the exported file)
        let model: matmul::Model<Backend> = matmul::Model::default();

        let device = Default::default();
        let a = Tensor::<Backend, 1, Int>::arange(0..24, &device)
            .reshape([1, 2, 3, 4])
            .float();
        let b = Tensor::<Backend, 1, Int>::arange(0..16, &device)
            .reshape([1, 2, 4, 2])
            .float();
        let c = Tensor::<Backend, 1, Int>::arange(0..96, &device)
            .reshape([2, 3, 4, 4])
            .float();
        let d = Tensor::<Backend, 1, Int>::arange(0..4, &device).float();

        let (output_mm, output_mv, output_vm) = model.forward(a, b, c, d);
        // matrix-matrix `a @ b`
        let expected_mm = TensorData::from([[
            [[28f32, 34.], [76., 98.], [124., 162.]],
            [[604., 658.], [780., 850.], [956., 1042.]],
        ]]);
        // matrix-vector `c @ d` where the lhs vector is expanded and broadcasted to the correct dims
        let expected_mv = TensorData::from([
            [
                [14f32, 38., 62., 86.],
                [110., 134., 158., 182.],
                [206., 230., 254., 278.],
            ],
            [
                [302., 326., 350., 374.],
                [398., 422., 446., 470.],
                [494., 518., 542., 566.],
            ],
        ]);
        // vector-matrix `d @ c` where the rhs vector is expanded and broadcasted to the correct dims
        let expected_vm = TensorData::from([
            [
                [56f32, 62., 68., 74.],
                [152., 158., 164., 170.],
                [248., 254., 260., 266.],
            ],
            [
                [344., 350., 356., 362.],
                [440., 446., 452., 458.],
                [536., 542., 548., 554.],
            ],
        ]);

        output_mm.to_data().assert_eq(&expected_mm, true);
        output_vm.to_data().assert_eq(&expected_vm, true);
        output_mv.to_data().assert_eq(&expected_mv, true);
    }

    #[test]
    fn concat_tensors() {
        // Initialize the model
        let device = Default::default();
        let model: concat::Model<Backend> = concat::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::zeros([1, 2, 3, 5], &device);

        let output = model.forward(input);

        let expected = Shape::from([1, 18, 3, 5]);

        assert_eq!(output.shape(), expected);
    }

    #[test]
    fn conv1d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv1d::Model<Backend> = conv1d::Model::default();

        // Run the model with pi as input for easier testing
        let input = Tensor::<Backend, 3>::full([6, 4, 10], consts::PI, &Default::default());

        let output = model.forward(input);

        // test the output shape
        let expected_shape: Shape<3> = Shape::from([6, 2, 7]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv1d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();
        let expected_sum = -54.549_243; // from pytorch
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv2d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv2d::Model<Backend> = conv2d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 6, 15]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv2d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = -113.869_99; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv3d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv3d::Model<Backend> = conv3d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 5>::ones([2, 4, 4, 5, 7], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 3, 5, 5]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv3d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = 48.494_262; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn dropout_opset16() {
        let model: dropout_opset16::Model<Backend> = dropout_opset16::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 4, 10, 15]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();

        let expected_sum = 1200.0; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn dropout_opset7() {
        let model: dropout_opset7::Model<Backend> = dropout_opset7::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 4, 10, 15]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();

        let expected_sum = 1200.0; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn erf() {
        let model: erf::Model<Backend> = erf::Model::default();

        let device = Default::default();
        let input = Tensor::<Backend, 4>::from_data([[[[1.0, 2.0, 3.0, 4.0]]]], &device);
        let output = model.forward(input);
        let expected =
            Tensor::<Backend, 4>::from_data([[[[0.8427f32, 0.9953, 1.0000, 1.0000]]]], &device);

        output.to_data().assert_approx_eq(&expected.to_data(), 4);
    }

    #[test]
    fn gather() {
        let model: gather::Model<Backend> = gather::Model::default();

        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = Tensor::<Backend, 1, Int>::from_ints([0, 2], &device);
        let output = model.forward(input, index);
        let expected = TensorData::from([[1f32, 3.], [4., 6.]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_scalar() {
        let model: gather_scalar::Model<Backend> = gather_scalar::Model::default();

        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = 0;
        let output = model.forward(input, index);
        let expected = TensorData::from([1f32, 2., 3.]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_elements() {
        // Initialize the model with weights (loaded from the exported file)
        let model: gather_elements::Model<Backend> = gather_elements::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 2>::from_floats([[1., 2.], [3., 4.]], &device);
        let index = Tensor::<Backend, 2, Int>::from_ints([[0, 0], [1, 0]], &device);
        let output = model.forward(input, index);
        let expected = TensorData::from([[1f32, 1.], [4., 3.]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn argmax() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmax::Model<Backend> = argmax::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input);
        let expected = TensorData::from([[2i64], [2]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn globalavrpool_1d_2d() {
        // The model contains 1d and 2d global average pooling nodes
        let model: global_avr_pool::Model<Backend> = global_avr_pool::Model::default();

        let device = Default::default();
        // Run the model with ones as input for easier testing
        let input_1d = Tensor::<Backend, 3>::ones([2, 4, 10], &device);
        let input_2d = Tensor::<Backend, 4>::ones([3, 10, 3, 15], &device);

        let (output_1d, output_2d) = model.forward(input_1d, input_2d);

        let expected_shape_1d = Shape::from([2, 4, 1]);
        let expected_shape_2d = Shape::from([3, 10, 1, 1]);
        assert_eq!(output_1d.shape(), expected_shape_1d);
        assert_eq!(output_2d.shape(), expected_shape_2d);

        let output_sum_1d = output_1d.sum().into_scalar();
        let output_sum_2d = output_2d.sum().into_scalar();

        let expected_sum_1d = 8.0; // from pytorch
        let expected_sum_2d = 30.0; // from pytorch

        assert!(expected_sum_1d.approx_eq(output_sum_1d, (1.0e-4, 2)));
        assert!(expected_sum_2d.approx_eq(output_sum_2d, (1.0e-4, 2)));
    }

    #[test]
    fn slice() {
        let model: slice::Model<Backend> = slice::Model::default();
        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats(
            [
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                [11., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
                [21., 22., 23., 24., 25., 26., 27., 28., 29., 30.],
                [31., 32., 33., 34., 35., 36., 37., 38., 39., 40.],
                [41., 42., 43., 44., 45., 46., 47., 48., 49., 50.],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [1f32, 2., 3., 4., 5.],
            [11f32, 12., 13., 14., 15.],
            [21., 22., 23., 24., 25.],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn softmax() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: softmax::Model<Backend> = softmax::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [
                [0.33669037, 0.128_809_4, 0.23446237],
                [0.23033303, -1.122_856_4, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.36830685f32, 0.29917702, 0.33251613],
            [0.521_469_2, 0.13475533, 0.343_775_5],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn log_softmax() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: log_softmax::Model<Backend> = log_softmax::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [
                [0.33669037, 0.128_809_4, 0.23446237],
                [0.23033303, -1.122_856_4, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [-0.998_838_9f32, -1.206_719_9, -1.101_067],
            [-0.651_105_1, -2.004_294_6, -1.067_766_4],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sqrt() {
        let device = Default::default();
        let model: sqrt::Model<Backend> = sqrt::Model::new(&device);

        let input1 = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let input2 = 36f64;

        let (output1, output2) = model.forward(input1, input2);
        let expected1 = TensorData::from([[[[1.0f32, 2.0, 3.0, 5.0]]]]);
        let expected2 = 6.0;

        output1.to_data().assert_eq(&expected1, true);
        assert_eq!(output2, expected2);
    }

    #[test]
    fn min() {
        let device = Default::default();

        let model: min::Model<Backend> = min::Model::new(&device);
        let input1 = Tensor::<Backend, 2>::from_floats([[-1.0, 42.0, 0.0, 42.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[2.0, 4.0, 42.0, 25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[-1.0f32, 4.0, 0.0, 25.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn max() {
        let device = Default::default();

        let model: max::Model<Backend> = max::Model::new(&device);
        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 42.0, 9.0, 42.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[42.0, 4.0, 42.0, 25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[42.0f32, 42.0, 42.0, 42.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool1d() {
        let device = Default::default();

        let model: maxpool1d::Model<Backend> = maxpool1d::Model::new(&device);
        let input = Tensor::<Backend, 3>::from_floats(
            [[
                [1.927, 1.487, 0.901, -2.106, 0.678],
                [-1.235, -0.043, -1.605, -0.752, -0.687],
                [-0.493, 0.241, -1.111, 0.092, -2.317],
                [-0.217, -1.385, -0.396, 0.803, -0.622],
                [-0.592, -0.063, -0.829, 0.331, -1.558],
            ]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[
            [1.927f32, 1.927, 0.901],
            [-0.043, -0.043, -0.687],
            [0.241, 0.241, 0.092],
            [-0.217, 0.803, 0.803],
            [-0.063, 0.331, 0.331],
        ]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool2d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: maxpool2d::Model<Backend> = maxpool2d::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats(
            [[[
                [1.927, 1.487, 0.901, -2.106, 0.678],
                [-1.235, -0.043, -1.605, -0.752, -0.687],
                [-0.493, 0.241, -1.111, 0.092, -2.317],
                [-0.217, -1.385, -0.396, 0.803, -0.622],
                [-0.592, -0.063, -0.829, 0.331, -1.558],
            ]]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[[
            [0.901f32, 1.927, 1.487, 0.901],
            [0.901, 1.927, 1.487, 0.901],
            [-0.396, 0.803, 0.241, -0.396],
        ]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn avg_pool1d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: avg_pool1d::Model<Backend> = avg_pool1d::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 3>::from_floats(
            [[
                [-1.526, -0.750, -0.654, -1.609, -0.100],
                [-0.609, -0.980, -1.609, -0.712, 1.171],
                [1.767, -0.095, 0.139, -1.579, -0.321],
                [-0.299, 1.879, 0.336, 0.275, 1.716],
                [-0.056, 0.911, -1.392, 2.689, -0.111],
            ]],
            &device,
        );
        let (output1, output2, output3) = model.forward(input.clone(), input.clone(), input);
        let expected1 = TensorData::from([[[-1.135f32], [-0.978], [0.058], [0.548], [0.538]]]);
        let expected2 = TensorData::from([[
            [-0.569f32, -1.135, -0.591],
            [-0.397, -0.978, -0.288],
            [0.418, 0.058, -0.440],
            [0.395, 0.548, 0.582],
            [0.214, 0.538, 0.296],
        ]]);
        let expected3 = TensorData::from([[
            [-1.138f32, -1.135, -0.788],
            [-0.794, -0.978, -0.383],
            [0.836, 0.058, -0.587],
            [0.790, 0.548, 0.776],
            [0.427, 0.538, 0.395],
        ]]);

        let expected_shape1 = Shape::from([1, 5, 1]);
        let expected_shape2 = Shape::from([1, 5, 3]);
        let expected_shape3 = Shape::from([1, 5, 3]);

        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        output1.to_data().assert_approx_eq(&expected1, 3);
        output2.to_data().assert_approx_eq(&expected2, 3);
        output3.to_data().assert_approx_eq(&expected3, 3);
    }

    #[test]
    fn avg_pool2d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: avg_pool2d::Model<Backend> = avg_pool2d::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats(
            [[[
                [-0.077, 0.360, -0.782, 0.072, 0.665],
                [-0.287, 1.621, -1.597, -0.052, 0.611],
                [0.760, -0.034, -0.345, 0.494, -0.078],
                [-1.805, -0.476, 0.205, 0.338, 1.353],
                [0.374, 0.013, 0.774, -0.109, -0.271],
            ]]],
            &device,
        );
        let (output1, output2, output3) = model.forward(input.clone(), input.clone(), input);
        let expected1 = TensorData::from([[[[0.008f32, -0.131, -0.208, 0.425]]]]);
        let expected2 = TensorData::from([[[
            [-0.045f32, 0.202, -0.050, -0.295, 0.162, 0.160],
            [-0.176, 0.008, -0.131, -0.208, 0.425, 0.319],
            [-0.084, -0.146, 0.017, 0.170, 0.216, 0.125],
        ]]]);
        let expected3 = TensorData::from([[[
            [-0.182f32, 0.404, -0.100, -0.590, 0.324, 0.638],
            [-0.352, 0.008, -0.131, -0.208, 0.425, 0.638],
            [-0.224, -0.195, 0.023, 0.226, 0.288, 0.335],
        ]]]);

        let expected_shape1 = Shape::from([1, 1, 1, 4]);
        let expected_shape2 = Shape::from([1, 1, 3, 6]);
        let expected_shape3 = Shape::from([1, 1, 3, 6]);

        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        output1.to_data().assert_approx_eq(&expected1, 3);
        output2.to_data().assert_approx_eq(&expected2, 3);
        output3.to_data().assert_approx_eq(&expected3, 3);
    }

    #[test]
    fn reduce_max() {
        let device = Default::default();
        let model: reduce_max::Model<Backend> = reduce_max::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([25f32]);
        let expected = TensorData::from([[[[25f32]]]]);

        assert_eq!(output_scalar.to_data(), expected_scalar);
        assert_eq!(output_tensor.to_data(), input.to_data());
        assert_eq!(output_value.to_data(), expected);
    }

    #[test]
    fn reduce_min() {
        let device = Default::default();
        let model: reduce_min::Model<Backend> = reduce_min::Model::new(&device);

        // Run the models
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([1f32]);
        let expected = TensorData::from([[[[1f32]]]]);

        assert_eq!(output_scalar.to_data(), expected_scalar);
        assert_eq!(output_tensor.to_data(), input.to_data());
        assert_eq!(output_value.to_data(), expected);
    }

    #[test]
    fn reduce_mean() {
        let device = Default::default();
        let model: reduce_mean::Model<Backend> = reduce_mean::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([9.75f32]);
        let expected = TensorData::from([[[[9.75f32]]]]);

        output_scalar.to_data().assert_eq(&expected_scalar, true);
        output_tensor.to_data().assert_eq(&input.to_data(), true);
        output_value.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn reduce_prod() {
        let device = Default::default();
        let model: reduce_prod::Model<Backend> = reduce_prod::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([900f32]);
        let expected = TensorData::from([[[[900f32]]]]);

        // Tolerance of 0.001 since floating-point multiplication won't be perfect
        output_scalar
            .to_data()
            .assert_approx_eq(&expected_scalar, 3);
        output_tensor
            .to_data()
            .assert_approx_eq(&input.to_data(), 3);
        output_value.to_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn reduce_sum_opset11() {
        let device = Default::default();
        let model: reduce_sum_opset11::Model<Backend> = reduce_sum_opset11::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([39f32]);
        let expected = TensorData::from([[[[39f32]]]]);

        output_scalar.to_data().assert_eq(&expected_scalar, true);
        output_tensor.to_data().assert_eq(&input.to_data(), true);
        output_value.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn reduce_sum_opset13() {
        let device = Default::default();
        let model: reduce_sum_opset13::Model<Backend> = reduce_sum_opset13::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([39f32]);
        let expected = TensorData::from([[[[39f32]]]]);

        output_scalar.to_data().assert_eq(&expected_scalar, true);
        output_tensor.to_data().assert_eq(&input.to_data(), true);
        output_value.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn reshape() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: reshape::Model<Backend> = reshape::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 1>::from_floats([0., 1., 2., 3.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([[0f32, 1., 2., 3.]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn resize_with_sizes() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_with_sizes::Model<Backend> = resize_with_sizes::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats(
            [[[
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]]],
            &device,
        );

        // The sizes are [1, 1, 2, 3]
        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0f32, 1.5, 3.0], [12.0, 13.5, 15.0]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    #[ignore = "https://github.com/tracel-ai/burn/issues/2080"]
    fn resize_with_scales_1d_linear() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_1d_linear_scale::Model<Backend> =
            resize_1d_linear_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 3>::from_floats(
            [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
            &device,
        );

        // The scales are 1.5
        let output = model.forward(input);

        let output_sum = output.sum().into_scalar();
        let expected_sum = -4.568_224; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_2d_bilinear() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_bilinear_scale::Model<Backend> =
            resize_2d_bilinear_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        let output_sum = output.sum().into_scalar();
        let expected_sum = -3.401_126_6; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_2d_nearest() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_nearest_scale::Model<Backend> =
            resize_2d_nearest_scale::Model::<Backend>::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9, 9]);

        let output_sum = output.sum().into_scalar();
        let expected_sum = -0.812_227_7; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_1d_nearest() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_1d_nearest_scale::Model<Backend> =
            resize_1d_nearest_scale::Model::<Backend>::new(&device);

        // Run the model
        let input = Tensor::<Backend, 3>::from_floats(
            [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9]);

        let output_sum = output.sum().into_scalar();
        let expected_sum = -4.568_224; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_2d_bicubic() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_bicubic_scale::Model<Backend> =
            resize_2d_bicubic_scale::Model::<Backend>::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9, 9]);

        let output_sum = output.sum().into_scalar();

        let expected_sum = -3.515_921; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-3, 2)));
    }

    #[test]
    fn shape() {
        let device = Default::default();
        let model: shape::Model<Backend> = shape::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::ones([4, 2], &device);
        let output = model.forward(input);
        let expected = [4, 2];
        assert_eq!(output, expected);
    }

    #[test]
    fn flatten() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: flatten::Model<Backend> = flatten::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 3>::ones([1, 5, 15], &device);
        let output = model.forward(input);

        let expected_shape = Shape::from([1, 75]);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn batch_norm() {
        let model: batch_norm::Model<Backend> = batch_norm::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 3>::ones([1, 20, 1], &Default::default());
        let output = model.forward(input);

        let expected_shape = Shape::from([1, 5, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();
        let expected_sum = 19.999_802; // from pytorch
        assert!(expected_sum.approx_eq(output_sum, (1.0e-8, 2)));
    }

    #[test]
    fn layer_norm() {
        let device = Default::default();
        let model: layer_norm::Model<Backend> = layer_norm::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 3>::from_floats(
            [
                [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                [
                    [12., 13., 14., 15.],
                    [16., 17., 18., 19.],
                    [20., 21., 22., 23.],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [
                [-1.3416f32, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
            [
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
        ]);

        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn leaky_relu() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: leaky_relu::Model<Backend> = leaky_relu::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [
                [0.33669037, 0.0, 0.23446237],
                [0.23033303, -1.122_856, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.33669037f32, 0.0, 0.23446237],
            [0.23033303, -0.01122_856, -0.0018632829],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn prelu() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: prelu::Model<Backend> = prelu::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [
                [0.33669037, 0.0, 0.23446237],
                [0.23033303, -1.122_856, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.33669037f32, 0.0, 0.23446237],
            [0.23033303, -0.280714, -0.046582073],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn relu() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: relu::Model<Backend> = relu::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [
                [0.33669037, 0.128_809_4, 0.23446237],
                [0.23033303, -1.122_856_4, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.33669037f32, 0.128_809_4, 0.23446237],
            [0.23033303, 0.00000000, 0.00000000],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sigmoid() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: sigmoid::Model<Backend> = sigmoid::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [
                [0.33669037, 0.128_809_4, 0.23446237],
                [0.23033303, -1.122_856_4, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.58338636f32, 0.532_157_9, 0.55834854],
            [0.557_33, 0.24548186, 0.45355222],
        ]);

        output.to_data().assert_approx_eq(&expected, 7);
    }

    #[test]
    fn hard_sigmoid() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: hard_sigmoid::Model<Backend> = hard_sigmoid::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [
                [0.33669037, 0.12880941, 0.23446237],
                [0.23033303, -1.12285638, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.55611509, 0.52146822, 0.53907704],
            [0.53838885, 0.31285727, 0.46894526],
        ]);

        output.to_data().assert_approx_eq(&expected, 7);
    }

    #[test]
    fn sin() {
        let device = Default::default();
        let model: sin::Model<Backend> = sin::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.8415f32, -0.7568, 0.4121, -0.1324]]]]);

        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn transpose() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: transpose::Model<Backend> = transpose::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 3>::from_floats(
            [
                [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                [
                    [12., 13., 14., 15.],
                    [16., 17., 18., 19.],
                    [20., 21., 22., 23.],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [[0f32, 4., 8.], [12., 16., 20.]],
            [[1., 5., 9.], [13., 17., 21.]],
            [[2., 6., 10.], [14., 18., 22.]],
            [[3., 7., 11.], [15., 19., 23.]],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn equal_scalar_to_scalar_and_tensor_to_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: equal::Model<Backend> = equal::Model::default();

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 1., 1., 1.]]]], &Default::default());

        let scalar = 2f64;
        let (tensor_out, scalar_out) = model.forward(input, scalar);
        let expected_tensor = TensorData::from([[[[true, true, true, true]]]]);
        let expected_scalar = false;

        tensor_out.to_data().assert_eq(&expected_tensor, true);
        assert_eq!(scalar_out, expected_scalar);
    }

    #[test]
    fn clip_opset16() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: clip_opset16::Model<Backend> = clip_opset16::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 1>::from_floats(
            [
                0.88226926,
                0.91500396,
                0.38286376,
                0.95930564,
                0.390_448_2,
                0.60089535,
            ],
            &device,
        );
        let (output1, output2, output3) = model.forward(input);
        let expected1 = TensorData::from([
            0.88226926f32,
            0.91500396,
            0.38286376,
            0.95930564,
            0.390_448_2,
            0.60089535,
        ]);
        let expected2 = TensorData::from([0.7f32, 0.7, 0.5, 0.7, 0.5, 0.60089535]);
        let expected3 = TensorData::from([0.8f32, 0.8, 0.38286376, 0.8, 0.390_448_2, 0.60089535]);

        output1.to_data().assert_eq(&expected1, true);
        output2.to_data().assert_eq(&expected2, true);
        output3.to_data().assert_eq(&expected3, true);
    }

    #[test]
    fn clip_opset7() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: clip_opset7::Model<Backend> = clip_opset7::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 1>::from_floats(
            [
                0.88226926f32,
                0.91500396,
                0.38286376,
                0.95930564,
                0.390_448_2,
                0.60089535,
            ],
            &device,
        );
        let (output1, output2, output3) = model.forward(input);
        let expected1 = TensorData::from([
            0.88226926f32,
            0.91500396,
            0.38286376,
            0.95930564,
            0.390_448_2,
            0.60089535,
        ]);
        let expected2 = TensorData::from([0.7f32, 0.7, 0.5, 0.7, 0.5, 0.60089535]);
        let expected3 = TensorData::from([0.8f32, 0.8, 0.38286376, 0.8, 0.390_448_2, 0.60089535]);

        output1.to_data().assert_eq(&expected1, true);
        output2.to_data().assert_eq(&expected2, true);
        output3.to_data().assert_eq(&expected3, true);
    }

    #[test]
    fn linear() {
        let device = Default::default();
        // Initialize the model with weights (loaded from the exported file)
        let model: linear::Model<Backend> = linear::Model::default();
        #[allow(clippy::approx_constant)]
        let input1 = Tensor::<Backend, 2>::full([4, 3], 3.14, &device);
        #[allow(clippy::approx_constant)]
        let input2 = Tensor::<Backend, 2>::full([2, 5], 3.14, &device);
        #[allow(clippy::approx_constant)]
        let input3 = Tensor::<Backend, 3>::full([3, 2, 7], 3.14, &device);

        let (output1, output2, output3) = model.forward(input1, input2, input3);

        // test the output shape
        let expected_shape1: Shape<2> = Shape::from([4, 4]);
        let expected_shape2: Shape<2> = Shape::from([2, 6]);
        let expected_shape3: Shape<3> = Shape::from([3, 2, 8]);
        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        // We are using the sum of the output tensor to test the correctness of the conv1d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum1 = output1.sum().into_scalar();
        let output_sum2 = output2.sum().into_scalar();
        let output_sum3 = output3.sum().into_scalar();

        let expected_sum1 = -9.655_477; // from pytorch
        let expected_sum2 = -8.053_822; // from pytorch
        let expected_sum3 = 27.575_281; // from pytorch

        assert!(expected_sum1.approx_eq(output_sum1, (1.0e-6, 2)));
        assert!(expected_sum2.approx_eq(output_sum2, (1.0e-6, 2)));
        assert!(expected_sum3.approx_eq(output_sum3, (1.0e-6, 2)));
    }

    #[test]
    fn tanh() {
        // Initialize the model
        let device = Default::default();
        let model = tanh::Model::<Backend>::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let output = model.forward(input);
        // data from pyTorch
        let expected = TensorData::from([[[[0.7616f32, 0.9640, 0.9951, 0.9993]]]]);
        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn range() {
        let device = Default::default();
        let model: range::Model<Backend> = range::Model::new(&device);

        // Run the model
        let start = 0i64;
        let limit = 10i64;
        let delta = 2i64;
        let output = model.forward(start, limit, delta);

        let expected = TensorData::from([0i64, 2, 4, 6, 8]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn recip() {
        // Initialize the model
        let device = Default::default();
        let model = recip::Model::<Backend>::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let output = model.forward(input);
        // data from pyTorch
        let expected = TensorData::from([[[[1.0000f32, 0.5000, 0.3333, 0.2500]]]]);
        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn conv_transpose2d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv_transpose2d::Model<Backend> = conv_transpose2d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 17, 15]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv_transpose2d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = -120.070_15; // result running pytorch model (conv_transpose2d.py)

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv_transpose3d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv_transpose3d::Model<Backend> = conv_transpose3d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 5>::ones([2, 4, 4, 5, 7], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 5, 5, 9]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv_transpose3d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = -67.267_15; // result running pytorch model (conv_transpose3d.py)

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn cos() {
        let device = Default::default();
        let model: cos::Model<Backend> = cos::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.5403f32, -0.6536, -0.9111, 0.9912]]]]);

        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn exp() {
        let device = Default::default();
        let model: exp::Model<Backend> = exp::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[0.0000, 0.6931]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1f32, 2.]]]]);

        output.to_data().assert_approx_eq(&expected, 2);
    }

    #[test]
    fn expand() {
        let device = Default::default();
        let model: expand::Model<Backend> = expand::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[-1.0], [1.0]], &device);

        let output = model.forward(input1);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn gelu() {
        let device = Default::default();
        let model: gelu::Model<Backend> = gelu::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.8413f32, 3.9999, 9.0000, 25.0000]]]]);

        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn log() {
        let device = Default::default();
        let model: log::Model<Backend> = log::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0000f32, 1.3863, 2.1972, 3.2189]]]]);

        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn neg() {
        let device = Default::default();
        let model: neg::Model<Backend> = neg::Model::new(&device);

        let input1 = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let input2 = 99f64;

        let (output1, output2) = model.forward(input1, input2);
        let expected1 = TensorData::from([[[[-1.0f32, -4.0, -9.0, -25.0]]]]);
        let expected2 = -99f64;

        output1.to_data().assert_approx_eq(&expected1, 4);

        assert_eq!(output2, expected2);
    }

    #[test]
    fn not() {
        let device = Default::default();
        let model: not::Model<Backend> = not::Model::new(&device);

        let input = Tensor::<Backend, 4, Bool>::from_bool(
            TensorData::from([[[[true, false, true, false]]]]),
            &device,
        );

        let output = model.forward(input).to_data();
        let expected = TensorData::from([[[[false, true, false, true]]]]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad() {
        let device = Default::default();
        let model: pad::Model<Backend> = pad::Model::new(&device);

        let input = Tensor::<Backend, 2>::from_floats([[1., 2.], [3., 4.], [5., 6.]], &device);
        let output = model.forward(input).to_data();
        let expected = TensorData::from([
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 1., 2., 0., 0., 0., 0.],
            [0.0_f32, 0., 3., 4., 0., 0., 0., 0.],
            [0.0_f32, 0., 5., 6., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn greater() {
        let device = Default::default();
        let model: greater::Model<Backend> = greater::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, false, true, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_scalar() {
        let device = Default::default();
        let model: greater_scalar::Model<Backend> = greater_scalar::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, true, true, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less() {
        let device = Default::default();
        let model: less::Model<Backend> = less::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, true, false, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less_scalar() {
        let device = Default::default();
        let model: less_scalar::Model<Backend> = less_scalar::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, false, false, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_or_equal() {
        let device = Default::default();
        let model: greater_or_equal::Model<Backend> = greater_or_equal::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, false, true, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_or_equal_scalar() {
        let device = Default::default();
        let model: greater_or_equal_scalar::Model<Backend> =
            greater_or_equal_scalar::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, true, true, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less_or_equal() {
        let device = Default::default();
        let model: less_or_equal::Model<Backend> = less_or_equal::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, true, false, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less_or_equal_scalar() {
        let device = Default::default();
        let model: less_or_equal_scalar::Model<Backend> = less_or_equal_scalar::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, false, false, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_model_creation_with_a_default_device() {
        let device = Default::default();
        let model: neg::Model<Backend> = neg::Model::new(&device);

        let input1 = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let input2 = 99f64;

        let (output1, output2) = model.forward(input1, input2);
        let expected1 = TensorData::from([[[[-1.0f32, -4.0, -9.0, -25.0]]]]);
        let expected2 = -99f64;

        output1.to_data().assert_approx_eq(&expected1, 4);

        assert_eq!(output2, expected2);
    }
    #[test]
    fn pow_int_with_tensor_and_scalar() {
        let device = Default::default();
        let model: pow_int::Model<Backend> = pow_int::Model::new(&device);

        let input1 = Tensor::<Backend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let input2 = 2;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[[[1i64, 16, 729, 65536]]]]);

        output.to_data().assert_eq(&expected, true);
    }
    #[test]
    fn pow_with_tensor_and_scalar() {
        let device = Default::default();
        let model: pow::Model<Backend> = pow::Model::new(&device);

        let input1 = Tensor::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let input2 = 2f64;

        let output = model.forward(input1, input2);

        let expected = TensorData::from([[[[1.0000f32, 1.6000e+01, 7.2900e+02, 6.5536e+04]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn tile() {
        let device = Default::default();
        let model: tile::Model<Backend> = tile::Model::new(&device);

        let input = Tensor::<Backend, 2>::from_floats([[1., 2.], [3., 4.]], &device);
        let output = model.forward(input).to_data();
        let expected = TensorData::from([
            [1.0f32, 2.0f32, 1.0f32, 2.0f32],
            [3.0f32, 4.0f32, 3.0f32, 4.0f32],
            [1.0f32, 2.0f32, 1.0f32, 2.0f32],
            [3.0f32, 4.0f32, 3.0f32, 4.0f32],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn unsqueeze() {
        let device = Default::default();
        let model: unsqueeze::Model<Backend> = unsqueeze::Model::new(&device);
        let input_shape = Shape::from([3, 4, 5]);
        let expected_shape = Shape::from([1, 1, 3, 4, 5, 1]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn unsqueeze_opset16() {
        let device = Default::default();
        let model = unsqueeze_opset16::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 5]);
        let expected_shape = Shape::from([3, 4, 5, 1]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input, 1.0);
        assert_eq!(expected_shape, output.0.shape());
        assert_eq!(Shape::from([1]), output.1.shape());
    }

    #[test]
    fn unsqueeze_opset11() {
        let device = Default::default();
        let model = unsqueeze_opset11::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 5]);
        let expected_shape = Shape::from([3, 4, 5, 1]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input, 1.0);
        assert_eq!(expected_shape, output.0.shape());
        assert_eq!(Shape::from([1]), output.1.shape());
    }

    #[test]
    fn cast() {
        let device = Default::default();
        let model: cast::Model<Backend> = cast::Model::new(&device);

        let input_bool =
            Tensor::<Backend, 2, Bool>::from_bool(TensorData::from([[true], [true]]), &device);
        let input_int = Tensor::<Backend, 2, Int>::from_ints([[1], [1]], &device);
        let input_float = Tensor::<Backend, 2>::from_floats([[1f32], [1.]], &device);
        let input_scalar = 1f32;

        let (
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output_scalar,
        ) = model.forward(
            input_bool.clone(),
            input_int.clone(),
            input_float.clone(),
            input_scalar,
        );
        let expected_bool = input_bool.to_data();
        let expected_int = input_int.to_data();
        let expected_float = input_float.to_data();
        let expected_scalar = 1;

        output1.to_data().assert_eq(&expected_bool, true);
        output2.to_data().assert_eq(&expected_int, true);
        output3.to_data().assert_approx_eq(&expected_float, 4);

        output4.to_data().assert_eq(&expected_bool, true);
        output5.to_data().assert_eq(&expected_int, true);
        output6.to_data().assert_approx_eq(&expected_float, 4);

        output7.to_data().assert_eq(&expected_bool, true);
        output8.to_data().assert_eq(&expected_int, true);
        output9.to_data().assert_approx_eq(&expected_float, 4);

        assert_eq!(output_scalar, expected_scalar);
    }

    #[test]
    fn mask_where() {
        let device = Default::default();
        let model: mask_where::Model<Backend> = mask_where::Model::new(&device);

        let x1 = Tensor::ones([2, 2], &device);
        let y1 = Tensor::zeros([2, 2], &device);
        let x2 = Tensor::ones([2], &device);
        let y2 = Tensor::zeros([2], &device);
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let (output, output_broadcasted) = model.forward(mask, x1, y1, x2, y2);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
        output_broadcasted.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sign() {
        let device = Default::default();
        let model: sign::Model<Backend> = sign::Model::new(&device);

        let input = Tensor::<Backend, 4>::from_floats([[[[-1.0, 2.0, 0.0, -4.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-1.0f32, 1.0, 0.0, -1.0]]]]);

        output.to_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn squeeze_opset16() {
        let device = Default::default();
        let model = squeeze_opset16::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_opset13() {
        let device = Default::default();
        let model = squeeze_opset13::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_multiple() {
        let device = Default::default();
        let model = squeeze_multiple::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5, 1]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn random_uniform() {
        let device = Default::default();
        let model = random_uniform::Model::<Backend>::new(&device);
        let expected_shape = Shape::from([2, 3]);
        let output = model.forward();
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn random_normal() {
        let device = Default::default();
        let model = random_normal::Model::<Backend>::new(&device);
        let expected_shape = Shape::from([2, 3]);
        let output = model.forward();
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn constant_of_shape() {
        // This tests shape is being passed directly to the model
        let device = Default::default();
        let model = constant_of_shape::Model::<Backend>::new(&device);
        let input_shape = [2, 3, 2];
        let expected = Tensor::<Backend, 3>::full(input_shape, 1.125, &device).to_data();

        let output = model.forward(input_shape);

        output.to_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn constant_of_shape_full_like() {
        // This tests shape is being passed from the input tensor

        let device = Default::default();
        let model = constant_of_shape_full_like::Model::<Backend>::new(&device);
        let shape = [2, 3, 2];
        let f_expected = Tensor::<Backend, 3>::full(shape, 3.0, &device);
        let i_expected = Tensor::<Backend, 3, Int>::full(shape, 5, &device);
        let b_expected = Tensor::<Backend, 3, Int>::ones(shape, &device).bool();

        let input = Tensor::ones(shape, &device);
        let (f_output, i_output, b_output) = model.forward(input);

        assert!(f_output.equal(f_expected).all().into_scalar());
        assert!(i_output.equal(i_expected).all().into_scalar());
        assert!(b_output.equal(b_expected).all().into_scalar());
    }
}
