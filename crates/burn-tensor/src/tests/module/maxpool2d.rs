#[burn_tensor_testgen::testgen(module_max_pool2d)]
mod tests {
    use super::*;
    use burn_tensor::module::{max_pool2d, max_pool2d_with_indices};
    use burn_tensor::{Tensor, TensorData, backend::Backend};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_max_pool2d_simple() {
        let batch_size = 2;
        let channels_in = 2;
        let kernel_size_1 = 3;
        let kernel_size_2 = 3;
        let padding_1 = 1;
        let padding_2 = 1;
        let stride_1 = 1;
        let stride_2 = 1;
        let dilation_1 = 1;
        let dilation_2 = 1;

        let x = TestTensor::from([
            [
                [
                    [0.9861, 0.5474, 0.4477, 0.0732, 0.3548, 0.8221],
                    [0.8148, 0.5474, 0.9490, 0.7890, 0.5537, 0.5689],
                    [0.5986, 0.2059, 0.4897, 0.6136, 0.2965, 0.6182],
                    [0.1485, 0.9540, 0.4023, 0.6176, 0.7111, 0.3392],
                    [0.3703, 0.0472, 0.2771, 0.1868, 0.8855, 0.5605],
                    [0.5063, 0.1638, 0.9432, 0.7836, 0.8696, 0.1068],
                ],
                [
                    [0.8872, 0.0137, 0.1652, 0.5505, 0.6127, 0.6473],
                    [0.1128, 0.0888, 0.1152, 0.5456, 0.6199, 0.7947],
                    [0.5911, 0.7781, 0.7256, 0.6578, 0.0989, 0.9149],
                    [0.5879, 0.5189, 0.6561, 0.0578, 0.7025, 0.6426],
                    [0.9590, 0.0325, 0.6455, 0.6248, 0.2009, 0.1544],
                    [0.7339, 0.1369, 0.6598, 0.5528, 0.6775, 0.1572],
                ],
            ],
            [
                [
                    [0.6853, 0.6439, 0.4639, 0.5573, 0.2723, 0.5910],
                    [0.5419, 0.7729, 0.6743, 0.8956, 0.2997, 0.9546],
                    [0.0334, 0.2178, 0.6917, 0.4958, 0.3357, 0.6584],
                    [0.7358, 0.9074, 0.2462, 0.5159, 0.6420, 0.2441],
                    [0.7602, 0.6297, 0.6073, 0.5937, 0.8037, 0.4881],
                    [0.8859, 0.0974, 0.3954, 0.6763, 0.1078, 0.7467],
                ],
                [
                    [0.2991, 0.5012, 0.8024, 0.7653, 0.9378, 0.7952],
                    [0.7393, 0.2336, 0.9521, 0.2719, 0.8445, 0.0454],
                    [0.6479, 0.9822, 0.7905, 0.0318, 0.2474, 0.0628],
                    [0.9955, 0.7591, 0.4140, 0.3215, 0.4349, 0.1527],
                    [0.8064, 0.0164, 0.4002, 0.2024, 0.6128, 0.5827],
                    [0.5368, 0.7895, 0.8727, 0.7793, 0.0910, 0.3421],
                ],
            ],
        ]);
        let y = TestTensor::<4>::from([
            [
                [
                    [0.9861, 0.9861, 0.9490, 0.9490, 0.8221, 0.8221],
                    [0.9861, 0.9861, 0.9490, 0.9490, 0.8221, 0.8221],
                    [0.9540, 0.9540, 0.9540, 0.9490, 0.7890, 0.7111],
                    [0.9540, 0.9540, 0.9540, 0.8855, 0.8855, 0.8855],
                    [0.9540, 0.9540, 0.9540, 0.9432, 0.8855, 0.8855],
                    [0.5063, 0.9432, 0.9432, 0.9432, 0.8855, 0.8855],
                ],
                [
                    [0.8872, 0.8872, 0.5505, 0.6199, 0.7947, 0.7947],
                    [0.8872, 0.8872, 0.7781, 0.7256, 0.9149, 0.9149],
                    [0.7781, 0.7781, 0.7781, 0.7256, 0.9149, 0.9149],
                    [0.9590, 0.9590, 0.7781, 0.7256, 0.9149, 0.9149],
                    [0.9590, 0.9590, 0.6598, 0.7025, 0.7025, 0.7025],
                    [0.9590, 0.9590, 0.6598, 0.6775, 0.6775, 0.6775],
                ],
            ],
            [
                [
                    [0.7729, 0.7729, 0.8956, 0.8956, 0.9546, 0.9546],
                    [0.7729, 0.7729, 0.8956, 0.8956, 0.9546, 0.9546],
                    [0.9074, 0.9074, 0.9074, 0.8956, 0.9546, 0.9546],
                    [0.9074, 0.9074, 0.9074, 0.8037, 0.8037, 0.8037],
                    [0.9074, 0.9074, 0.9074, 0.8037, 0.8037, 0.8037],
                    [0.8859, 0.8859, 0.6763, 0.8037, 0.8037, 0.8037],
                ],
                [
                    [0.7393, 0.9521, 0.9521, 0.9521, 0.9378, 0.9378],
                    [0.9822, 0.9822, 0.9822, 0.9521, 0.9378, 0.9378],
                    [0.9955, 0.9955, 0.9822, 0.9521, 0.8445, 0.8445],
                    [0.9955, 0.9955, 0.9822, 0.7905, 0.6128, 0.6128],
                    [0.9955, 0.9955, 0.8727, 0.8727, 0.7793, 0.6128],
                    [0.8064, 0.8727, 0.8727, 0.8727, 0.7793, 0.6128],
                ],
            ],
        ]);

        let output = max_pool2d(
            x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            false,
        );

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_max_pool2d_different_padding_stride_kernel() {
        let batch_size = 1;
        let channels_in = 1;
        let kernel_size_1 = 3;
        let kernel_size_2 = 1;
        let padding_1 = 1;
        let padding_2 = 0;
        let stride_1 = 1;
        let stride_2 = 2;
        let dilation_1 = 1;
        let dilation_2 = 1;

        let x = TestTensor::from([[[
            [0.6309, 0.6112, 0.6998],
            [0.4708, 0.9161, 0.5402],
            [0.4577, 0.7397, 0.9870],
            [0.6380, 0.4352, 0.5884],
            [0.6277, 0.5139, 0.4525],
            [0.9333, 0.9846, 0.5006],
        ]]]);
        let y = TestTensor::<4>::from([[[
            [0.6309, 0.6998],
            [0.6309, 0.9870],
            [0.6380, 0.9870],
            [0.6380, 0.9870],
            [0.9333, 0.5884],
            [0.9333, 0.5006],
        ]]]);

        let output = max_pool2d(
            x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            false,
        );

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_max_pool2d_with_neg() {
        let batch_size = 1;
        let channels_in = 1;
        let kernel_size_1 = 3;
        let kernel_size_2 = 3;
        let padding_1 = 1;
        let padding_2 = 1;
        let stride_1 = 1;
        let stride_2 = 1;
        let dilation_1 = 1;
        let dilation_2 = 1;

        let x = TestTensor::from([[[
            [0.6309, 0.6112, 0.6998],
            [0.4708, 0.9161, 0.5402],
            [0.4577, 0.7397, 0.9870],
            [0.6380, 0.4352, 0.5884],
            [0.6277, 0.5139, 0.4525],
            [0.9333, 0.9846, 0.5006],
        ]]])
        .neg();
        let y = TestTensor::<4>::from([[[
            [-0.4708, -0.4708, -0.5402],
            [-0.4577, -0.4577, -0.5402],
            [-0.4352, -0.4352, -0.4352],
            [-0.4352, -0.4352, -0.4352],
            [-0.4352, -0.4352, -0.4352],
            [-0.5139, -0.4525, -0.4525],
        ]]]);

        let output = max_pool2d(
            x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            false,
        );

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_max_pool2d_with_dilation() {
        let batch_size = 1;
        let channels_in = 1;
        let kernel_size_1 = 2;
        let kernel_size_2 = 2;
        let padding_1 = 0;
        let padding_2 = 0;
        let stride_1 = 1;
        let stride_2 = 1;
        let dilation_1 = 2;
        let dilation_2 = 2;

        let x = TestTensor::from([[[
            [0.9861, 0.9861, 0.9490, 0.9490, 0.8221, 0.8221],
            [0.9861, 0.9861, 0.9490, 0.9490, 0.8221, 0.8221],
            [0.9540, 0.9540, 0.9540, 0.9490, 0.7890, 0.7111],
            [0.9540, 0.9540, 0.9540, 0.8855, 0.8855, 0.8855],
            [0.9540, 0.9540, 0.9540, 0.9432, 0.8855, 0.8855],
            [0.5063, 0.9432, 0.9432, 0.9432, 0.8855, 0.8855],
        ]]]);
        let y = TestTensor::<4>::from([[[
            [0.9861, 0.9861, 0.9540, 0.9490],
            [0.9861, 0.9861, 0.9540, 0.9490],
            [0.9540, 0.9540, 0.9540, 0.9490],
            [0.9540, 0.9540, 0.9540, 0.9432],
        ]]]);

        let output = max_pool2d(
            x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            false,
        );

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    fn test_max_pool2d_with_indices() {
        let batch_size = 1;
        let channels_in = 1;
        let kernel_size_1 = 2;
        let kernel_size_2 = 2;
        let padding_1 = 1;
        let padding_2 = 1;
        let stride_1 = 1;
        let stride_2 = 1;
        let dilation_1 = 1;
        let dilation_2 = 1;

        let x = TestTensor::from([[[
            [0.2479, 0.6386, 0.3166, 0.5742],
            [0.7065, 0.1940, 0.6305, 0.8959],
            [0.5416, 0.8602, 0.8129, 0.1662],
            [0.3358, 0.3059, 0.8293, 0.0990],
        ]]]);
        let indices = TensorData::from([[[
            [0, 1, 1, 3, 3],
            [4, 4, 1, 7, 7],
            [4, 9, 9, 7, 7],
            [8, 9, 9, 14, 11],
            [12, 12, 14, 14, 15],
        ]]]);
        let y = TestTensor::<4>::from([[[
            [0.2479, 0.6386, 0.6386, 0.5742, 0.5742],
            [0.7065, 0.7065, 0.6386, 0.8959, 0.8959],
            [0.7065, 0.8602, 0.8602, 0.8959, 0.8959],
            [0.5416, 0.8602, 0.8602, 0.8293, 0.1662],
            [0.3358, 0.3358, 0.8293, 0.8293, 0.0990],
        ]]]);

        let (output, output_indices) = max_pool2d_with_indices(
            x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            false,
        );

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
        output_indices.into_data().assert_eq(&indices, false);
    }

    #[test]
    fn test_max_pool2d_complex() {
        let batch_size = 1;
        let channels_in = 1;
        let kernel_size_1 = 4;
        let kernel_size_2 = 2;
        let padding_1 = 2;
        let padding_2 = 1;
        let stride_1 = 1;
        let stride_2 = 2;
        let dilation_1 = 1;
        let dilation_2 = 1;

        let x = TestTensor::from([[[
            [0.5388, 0.0676, 0.7122, 0.8316, 0.0653],
            [0.9154, 0.1536, 0.9089, 0.8016, 0.7518],
            [0.2073, 0.0501, 0.8811, 0.5604, 0.5075],
            [0.4384, 0.9963, 0.9698, 0.4988, 0.2609],
            [0.3391, 0.2230, 0.4610, 0.5365, 0.6880],
        ]]]);
        let indices = TensorData::from([[[
            [5, 7, 3],
            [5, 7, 3],
            [5, 16, 3],
            [5, 16, 8],
            [15, 16, 24],
            [15, 16, 24],
        ]]]);
        let y = TestTensor::<4>::from([[[
            [0.9154, 0.9089, 0.8316],
            [0.9154, 0.9089, 0.8316],
            [0.9154, 0.9963, 0.8316],
            [0.9154, 0.9963, 0.8016],
            [0.4384, 0.9963, 0.688],
            [0.4384, 0.9963, 0.688],
        ]]]);
        let (output, output_indices) = max_pool2d_with_indices(
            x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            false,
        );

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
        output_indices.into_data().assert_eq(&indices, false);
    }

    #[test]
    fn test_max_pool2d_ceil_mode() {
        // Test ceil_mode=true which produces larger output when input doesn't divide evenly by stride
        // Using 1x1x6x6 with kernel 3x3, stride 2x2, padding 0:
        // Floor mode: output = (6+0-1*(3-1)-1)/2+1 = 3/2+1 = 2 x 2
        // Ceil mode: output = ceil(3/2)+1 = 2+1 = 3 x 3
        let kernel_size_1 = 3;
        let kernel_size_2 = 3;
        let padding_1 = 0;
        let padding_2 = 0;
        let stride_1 = 2;
        let stride_2 = 2;
        let dilation_1 = 1;
        let dilation_2 = 1;

        // Input (values 1-36 arranged row by row):
        // col:    0  1  2  3  4  5
        // row 0:  1  2  3  4  5  6
        // row 1:  7  8  9  10 11 12
        // row 2: 13 14 15 16 17 18
        // row 3: 19 20 21 22 23 24
        // row 4: 25 26 27 28 29 30
        // row 5: 31 32 33 34 35 36
        let x = TestTensor::from([[[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        ]]]);

        // With ceil_mode=false (floor): output is 2x2
        // (0,0): rows 0-2, cols 0-2 -> max(1,2,3,7,8,9,13,14,15) = 15
        // (0,1): rows 0-2, cols 2-4 -> max(3,4,5,9,10,11,15,16,17) = 17
        // (1,0): rows 2-4, cols 0-2 -> max(13,14,15,19,20,21,25,26,27) = 27
        // (1,1): rows 2-4, cols 2-4 -> max(15,16,17,21,22,23,27,28,29) = 29
        let y_floor = TestTensor::<4>::from([[[[15.0, 17.0], [27.0, 29.0]]]]);

        let output_floor = max_pool2d(
            x.clone(),
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            false,
        );

        y_floor
            .to_data()
            .assert_approx_eq::<FT>(&output_floor.into_data(), Tolerance::default());

        // With ceil_mode=true: output is 3x3
        // Extra windows at edges use only available input values (padded with -inf for max pooling)
        // (0,0): rows 0-2, cols 0-2 -> max = 15
        // (0,1): rows 0-2, cols 2-4 -> max = 17
        // (0,2): rows 0-2, cols 4-5 -> max(5,6,11,12,17,18) = 18
        // (1,0): rows 2-4, cols 0-2 -> max = 27
        // (1,1): rows 2-4, cols 2-4 -> max = 29
        // (1,2): rows 2-4, cols 4-5 -> max(17,18,23,24,29,30) = 30
        // (2,0): rows 4-5, cols 0-2 -> max(25,26,27,31,32,33) = 33
        // (2,1): rows 4-5, cols 2-4 -> max(27,28,29,33,34,35) = 35
        // (2,2): rows 4-5, cols 4-5 -> max(29,30,35,36) = 36
        let y_ceil =
            TestTensor::<4>::from([[[[15.0, 17.0, 18.0], [27.0, 29.0, 30.0], [33.0, 35.0, 36.0]]]]);

        let output_ceil = max_pool2d(
            x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
            [dilation_1, dilation_2],
            true,
        );

        y_ceil
            .to_data()
            .assert_approx_eq::<FT>(&output_ceil.into_data(), Tolerance::default());
    }
}
