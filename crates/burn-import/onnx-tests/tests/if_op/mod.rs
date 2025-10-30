// Tests for ONNX If operator

use crate::include_models;
include_models!(
    if_conv2d, if_linear,
    nested_if // TODO: Enable when Loop and Scan support is added
              // nested_if_loop_if,
              // nested_if_loop_if_scan
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_if_conv2d_then_branch() {
        // Test If operator with Conv2d - then branch (condition=true)
        // Values generated from if_conv2d.py using ONNX ReferenceEvaluator
        let device = Default::default();
        let model: if_conv2d::Model<TestBackend> = Default::default();

        // Input shape: [1, 2, 4, 4] (batch=1, channels=2, height=4, width=4)
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [
                    [
                        0.18085071444511414,
                        2.6312596797943115,
                        1.5169763565063477,
                        0.39920559525489807,
                    ],
                    [
                        -1.3128409385681152,
                        0.7649950981140137,
                        0.28677263855934143,
                        1.2184350490570068,
                    ],
                    [
                        -0.5875476598739624,
                        -1.7047046422958374,
                        0.120703786611557,
                        -0.2267778515815735,
                    ],
                    [
                        -0.1574159562587738,
                        1.4494507312774658,
                        0.41282033920288086,
                        0.010600652545690536,
                    ],
                ],
                [
                    [
                        -0.5540851354598999,
                        1.0953863859176636,
                        -0.8105842471122742,
                        0.9001023173332214,
                    ],
                    [
                        -0.3472646474838257,
                        -0.22818411886692047,
                        -0.9487846493721008,
                        -1.4585974216461182,
                    ],
                    [
                        -0.5947825312614441,
                        -0.5502279996871948,
                        0.5772436857223511,
                        0.6150617599487305,
                    ],
                    [
                        1.9114148616790771,
                        1.9370934963226318,
                        -2.256314277648926,
                        -0.9484795928001404,
                    ],
                ],
            ]]),
            &device,
        );
        let condition = true;

        let output = model.forward(input, condition);

        // Expected output from ONNX ReferenceEvaluator - then branch (shape: [1, 4, 4, 4])
        let expected = TensorData::from([[
            [
                [
                    0.9263929724693298,
                    1.6761226654052734,
                    1.646040678024292,
                    0.968873918056488,
                ],
                [
                    0.5332736968994141,
                    1.237782597541809,
                    0.8174108266830444,
                    1.3897026777267456,
                ],
                [
                    0.6048609018325806,
                    1.266493558883667,
                    1.0557849407196045,
                    0.6272218227386475,
                ],
                [
                    0.7798487544059753,
                    1.5006515979766846,
                    1.5222256183624268,
                    0.8282109498977661,
                ],
            ],
            [
                [
                    0.8426209688186646,
                    0.8884355425834656,
                    0.656369149684906,
                    1.1384658813476562,
                ],
                [
                    1.2442083358764648,
                    1.1762200593948364,
                    1.3494484424591064,
                    0.8862987160682678,
                ],
                [
                    0.5478358268737793,
                    -0.17960917949676514,
                    0.5264441967010498,
                    1.2296161651611328,
                ],
                [
                    0.8600679636001587,
                    0.8178144693374634,
                    0.8499643802642822,
                    1.1286100149154663,
                ],
            ],
            [
                [
                    0.8299998044967651,
                    0.6678880453109741,
                    1.4803285598754883,
                    0.808264970779419,
                ],
                [
                    0.43443775177001953,
                    0.9290375113487244,
                    0.5278126001358032,
                    1.1974546909332275,
                ],
                [
                    1.7791976928710938,
                    0.48859381675720215,
                    1.1486115455627441,
                    0.6964491009712219,
                ],
                [
                    0.8043434023857117,
                    1.581263542175293,
                    1.3846330642700195,
                    0.7583174705505371,
                ],
            ],
            [
                [
                    0.8284513354301453,
                    0.9873950481414795,
                    1.2118624448776245,
                    0.5708482265472412,
                ],
                [
                    0.35558998584747314,
                    0.9581461548805237,
                    1.1113965511322021,
                    1.2119866609573364,
                ],
                [
                    1.7028021812438965,
                    1.116762638092041,
                    0.9485116004943848,
                    0.9732359647750854,
                ],
                [
                    0.9091841578483582,
                    1.448109745979309,
                    1.2588164806365967,
                    0.6174608469009399,
                ],
            ],
        ]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_if_conv2d_else_branch() {
        // Test If operator with Conv2d - else branch (condition=false)
        // Values generated from if_conv2d.py using ONNX ReferenceEvaluator
        let device = Default::default();
        let model: if_conv2d::Model<TestBackend> = Default::default();

        // Input shape: [1, 2, 4, 4] (batch=1, channels=2, height=4, width=4)
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [
                    [
                        0.18085071444511414,
                        2.6312596797943115,
                        1.5169763565063477,
                        0.39920559525489807,
                    ],
                    [
                        -1.3128409385681152,
                        0.7649950981140137,
                        0.28677263855934143,
                        1.2184350490570068,
                    ],
                    [
                        -0.5875476598739624,
                        -1.7047046422958374,
                        0.120703786611557,
                        -0.2267778515815735,
                    ],
                    [
                        -0.1574159562587738,
                        1.4494507312774658,
                        0.41282033920288086,
                        0.010600652545690536,
                    ],
                ],
                [
                    [
                        -0.5540851354598999,
                        1.0953863859176636,
                        -0.8105842471122742,
                        0.9001023173332214,
                    ],
                    [
                        -0.3472646474838257,
                        -0.22818411886692047,
                        -0.9487846493721008,
                        -1.4585974216461182,
                    ],
                    [
                        -0.5947825312614441,
                        -0.5502279996871948,
                        0.5772436857223511,
                        0.6150617599487305,
                    ],
                    [
                        1.9114148616790771,
                        1.9370934963226318,
                        -2.256314277648926,
                        -0.9484795928001404,
                    ],
                ],
            ]]),
            &device,
        );
        let condition = false;

        let output = model.forward(input, condition);

        // Expected output from ONNX ReferenceEvaluator - else branch (shape: [1, 3, 4, 4])
        let expected = TensorData::from([[
            [
                [
                    -0.048462554812431335,
                    -0.06931453943252563,
                    -0.13530631363391876,
                    0.0350324772298336,
                ],
                [
                    0.04340369999408722,
                    -0.057828404009342194,
                    -0.0797911211848259,
                    -0.16195590794086456,
                ],
                [
                    -0.010808168910443783,
                    0.05070926994085312,
                    0.028562359511852264,
                    0.049261245876550674,
                ],
                [
                    0.13026703894138336,
                    0.047644298523664474,
                    -0.17177824676036835,
                    -0.0652826651930809,
                ],
            ],
            [
                [
                    0.1354195773601532,
                    -0.34662774205207825,
                    0.15019887685775757,
                    -0.22149889171123505,
                ],
                [
                    0.1380942016839981,
                    0.03667265549302101,
                    0.2265060842037201,
                    0.3163011372089386,
                ],
                [
                    0.17212392389774323,
                    0.20057526230812073,
                    -0.13419438898563385,
                    -0.13109949231147766,
                ],
                [
                    -0.4448876976966858,
                    -0.5073702335357666,
                    0.5361302495002747,
                    0.23611167073249817,
                ],
            ],
            [
                [
                    0.1679793745279312,
                    -0.24954521656036377,
                    0.27333715558052063,
                    -0.24583707749843597,
                ],
                [
                    0.07322432100772858,
                    0.08783966302871704,
                    0.2841879427433014,
                    0.45300331711769104,
                ],
                [
                    0.16161242127418518,
                    0.12247245013713837,
                    -0.15937748551368713,
                    -0.17845454812049866,
                ],
                [
                    -0.5503063201904297,
                    -0.5198705196380615,
                    0.6638606786727905,
                    0.2775975465774536,
                ],
            ],
        ]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_if_linear_then_branch() {
        // Test If operator with Linear - then branch (condition=true)
        // Values generated from if_linear.py using ONNX ReferenceEvaluator
        let device = Default::default();
        let model: if_linear::Model<TestBackend> = Default::default();

        // Input shape: [2, 5] (batch=2, features=5)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    -2.178354501724243,
                    0.3916308879852295,
                    -1.2458105087280273,
                    -1.8448173999786377,
                    0.4595855176448822,
                ],
                [
                    0.05172090232372284,
                    0.6706469655036926,
                    1.1630100011825562,
                    -0.2599570155143738,
                    0.16391819715499878,
                ],
            ]),
            &device,
        );
        let condition = true;

        let output = model.forward(input, condition);

        // Expected output from ONNX ReferenceEvaluator - then branch (shape: [2, 8])
        let expected = TensorData::from([
            [
                -0.05453979969024658,
                0.5893748998641968,
                0.9069600105285645,
                0.8564329147338867,
                1.6499955654144287,
                -0.1498846411705017,
                0.10434189438819885,
                0.7201496362686157,
            ],
            [
                0.45539796352386475,
                0.2825457453727722,
                0.7953628301620483,
                0.13400855660438538,
                0.4214242398738861,
                0.4132808744907379,
                0.6058053970336914,
                0.47209885716438293,
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_if_linear_else_branch() {
        // Test If operator with Linear - else branch (condition=false)
        // Values generated from if_linear.py using ONNX ReferenceEvaluator
        let device = Default::default();
        let model: if_linear::Model<TestBackend> = Default::default();

        // Input shape: [2, 5] (batch=2, features=5)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    -2.178354501724243,
                    0.3916308879852295,
                    -1.2458105087280273,
                    -1.8448173999786377,
                    0.4595855176448822,
                ],
                [
                    0.05172090232372284,
                    0.6706469655036926,
                    1.1630100011825562,
                    -0.2599570155143738,
                    0.16391819715499878,
                ],
            ]),
            &device,
        );
        let condition = false;

        let output = model.forward(input, condition);

        // Expected output from ONNX ReferenceEvaluator - else branch (shape: [2, 6])
        let expected = TensorData::from([
            [
                0.5174261331558228,
                -0.4233645796775818,
                -0.34426870942115784,
                -0.7503882646560669,
                0.09909266978502274,
                0.40197938680648804,
            ],
            [
                -0.15882353484630585,
                -0.31122398376464844,
                -0.0277077816426754,
                0.40166664123535156,
                -0.045064087957143784,
                -0.2151859849691391,
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
    #[test]
    fn test_nested_if_then_then_then() {
        // Test nested If with scoped variables: then->then->then
        // Path: ((x + 10) - 0.5) + 1.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, true, true, true);

        let expected = TensorData::from([[11.5, 12.5, 13.5], [14.5, 15.5, 16.5]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_then_then_else() {
        // Test nested If with scoped variables: then->then->else
        // Path: ((x + 10) - 0.5) * 2.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, true, true, false);

        let expected = TensorData::from([[21.0, 23.0, 25.0], [27.0, 29.0, 31.0]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_then_else_then() {
        // Test nested If with scoped variables: then->else->then
        // Path: ((x + 10) / 3.0) + 1.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, true, false, true);

        let expected = TensorData::from([
            [4.6666669845581055, 5.0, 5.333333492279053],
            [5.666666507720947, 6.0, 6.333333492279053],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_then_else_else() {
        // Test nested If with scoped variables: then->else->else
        // Path: ((x + 10) / 3.0) * 2.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, true, false, false);

        let expected = TensorData::from([
            [7.333333492279053, 8.0, 8.666666984558105],
            [9.333333015441895, 10.0, 10.666666984558105],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_else_then_then() {
        // Test nested If with scoped variables: else->then->then
        // Path: ((-x) - 0.5) + 1.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, false, true, true);

        let expected = TensorData::from([[-0.5, -1.5, -2.5], [-3.5, -4.5, -5.5]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_else_then_else() {
        // Test nested If with scoped variables: else->then->else
        // Path: ((-x) - 0.5) * 2.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, false, true, false);

        let expected = TensorData::from([[-3.0, -5.0, -7.0], [-9.0, -11.0, -13.0]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_else_else_then() {
        // Test nested If with scoped variables: else->else->then
        // Path: ((-x) / 3.0) + 1.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, false, false, true);

        let expected = TensorData::from([
            [0.6666666269302368, 0.3333333134651184, 0.0],
            [-0.3333333730697632, -0.6666666269302368, -1.0],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_else_else_else() {
        // Test nested If with scoped variables: else->else->else
        // Path: ((-x) / 3.0) * 2.0
        let device = Default::default();
        let model: nested_if::Model<TestBackend> = Default::default();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input, false, false, false);

        let expected = TensorData::from([
            [-0.6666666865348816, -1.3333333730697632, -2.0],
            [-2.6666667461395264, -3.3333332538604736, -4.0],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    // Note: Expected values should be copied from Python script output
    // Run: cd tests/subgraph && uv run if_conv2d.py
    // Then copy the printed values into the test assertions below
}
