// Tests for ONNX subgraph operators (If, Loop, Scan)

use crate::include_models;
include_models!(
    if_conv2d, if_linear,
    nested_if // TODO: Enable when Loop support is added
              // loop_simple,
              // loop_multi_deps,
              // TODO: Enable when Scan support is added
              // scan_cumsum,
              // TODO: Enable when nested subgraph support is complete
              // nested_if_loop_if,
              // nested_if_loop_if_scan
);

#[cfg(test)]
mod tests {
    use super::*; // Import the models from parent scope
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

    // TODO: Enable when Loop support is added
    /*
    #[test]
    fn test_loop_simple_3_iterations() {
        // Test Loop operator with 3 iterations - verifies constant loading from .mpk
        // Values generated from loop_simple.py using ONNX ReferenceEvaluator
        let model: loop_simple::Model<TestBackend> = Default::default();
        let device = Default::default();

        let m = 3i64;
        let cond = true;
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [-1.7024059295654297, -0.8983973264694214, 1.4876478910446167],
                [
                    -0.9882321953773499,
                    -0.1252097636461258,
                    -0.3502468466758728,
                ],
            ]),
            &device,
        );
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    -0.10187239944934845,
                    1.1868587732315063,
                    -0.40321722626686096,
                ],
                [
                    -0.004976058378815651,
                    -0.019503045827150345,
                    -0.21081003546714783,
                ],
            ]),
            &device,
        );

        let output = model.forward(m, cond, initial_accum, x);

        // Loop body computes: result = ((accum + x) + 1.0) * 2.0
        // Expected output from ONNX ReferenceEvaluator after 3 iterations
        let expected = TensorData::from([
            [-1.045461654663086, 23.428844451904297, 20.256141662597656],
            [6.024477958679199, 12.725279808044434, 8.246685028076172],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
    */

    /*
    #[test]
    fn test_loop_simple_5_iterations() {
        // Test Loop operator with 5 iterations - verifies constant loading from .mpk
        // Values generated from loop_simple.py using ONNX ReferenceEvaluator
        let model: loop_simple::Model<TestBackend> = Default::default();
        let device = Default::default();

        let m = 5i64;
        let cond = true;
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.7864915132522583, 0.5976946949958801, 0.5340687036514282],
                [
                    -1.688645601272583,
                    -0.40642204880714417,
                    -0.27859559655189514,
                ],
            ]),
            &device,
        );
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.6873053908348083, 0.20610758662223816, 1.87429940700531],
                [-0.40776318311691284, 1.0868241786956787, 0.3541634678840637],
            ]),
            &device,
        );

        let output = model.forward(m, cond, initial_accum, x);

        // Loop body computes: result = ((accum + x) + 1.0) * 2.0
        // Expected output from ONNX ReferenceEvaluator after 5 iterations
        let expected = TensorData::from([
            [129.78067016601562, 93.90489959716797, 195.2967529296875],
            [-17.31797981262207, 116.37759399414062, 75.04307556152344],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_loop_simple_0_iterations() {
        // Test Loop operator with 0 iterations (should return initial accumulator)
        // Values generated from loop_simple.py using ONNX ReferenceEvaluator
        let device = Default::default();
        let model: loop_simple::Model<TestBackend> = Default::default();

        let m = 0i64;
        let cond = true;
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.7991490960121155, 0.79441899061203, -1.5536589622497559],
                [-1.460789680480957, -0.4079414904117584, 1.7659316062927246],
            ]),
            &device,
        );
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.7554057240486145, -1.1071630716323853, -1.0971266031265259],
                [
                    0.5415622591972351,
                    -0.008862749673426151,
                    -0.8675622344017029,
                ],
            ]),
            &device,
        );

        let output = model.forward(m, cond, initial_accum.clone(), x);

        // With 0 iterations, output should equal initial_accum
        output
            .to_data()
            .assert_approx_eq::<f32>(&initial_accum.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_scan_cumsum_random() {
        // Test Scan operator for cumulative sum with random data
        // Values generated from scan_cumsum.py using ONNX ReferenceEvaluator
        let device = Default::default();
        let model: scan_cumsum::Model<TestBackend> = Default::default();

        // Test data from scan_cumsum.py (random test case)
        let initial_sum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    0.6668040156364441f32,
                    0.8698066473007202,
                    1.3719428777694702,
                ],
                [0.6295738816261292, 0.5616223216056824, -1.4157276153564453],
            ]),
            &device,
        );

        let input_sequence = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [
                    [
                        -0.5013001561164856f32,
                        1.8008418083190918,
                        -0.008793800137937069,
                    ],
                    [
                        -0.8269177079200745,
                        -0.35227257013320923,
                        -0.13253618776798248,
                    ],
                ],
                [
                    [
                        -1.3441442251205444,
                        -1.0407260656356812,
                        -0.9849160313606262,
                    ],
                    [2.2742502689361572, 1.2131147384643555, 0.4437800645828247],
                ],
                [
                    [0.6664568781852722, 0.06628644466400146, 1.693333625793457],
                    [
                        -0.012491798959672451,
                        0.005913415923714638,
                        0.6656486392021179,
                    ],
                ],
                [
                    [-0.7180415391921997, 0.4855446219444275, 0.12954597175121307],
                    [-1.1406440734863281, 1.059289813041687, -0.3055904805660248],
                ],
            ]),
            &device,
        );

        let (final_sum, cumsum_sequence) = model.forward(initial_sum, input_sequence);

        let expected_final = TensorData::from([
            [
                -1.2302250862121582f32,
                2.181753635406494,
                2.2011125087738037,
            ],
            [0.9237706661224365, 2.4876675605773926, -0.7444255352020264],
        ]);

        let expected_cumsum = TensorData::from([
            [
                [
                    0.1655038595199585f32,
                    2.6706485748291016,
                    1.3631490468978882,
                ],
                [
                    -0.1973438262939453,
                    0.20934975147247314,
                    -1.5482637882232666,
                ],
            ],
            [
                [-1.178640365600586, 1.6299225091934204, 0.37823301553726196],
                [2.076906442642212, 1.4224644899368286, -1.104483723640442],
            ],
            [
                [-0.5121834874153137, 1.6962089538574219, 2.071566581726074],
                [2.0644147396087646, 1.4283778667449951, -0.438835084438324],
            ],
            [
                [-1.2302250862121582, 2.181753635406494, 2.2011125087738037],
                [0.9237706661224365, 2.4876675605773926, -0.7444255352020264],
            ],
        ]);

        final_sum
            .to_data()
            .assert_approx_eq::<f32>(&expected_final, burn::tensor::Tolerance::default());
        cumsum_sequence
            .to_data()
            .assert_approx_eq::<f32>(&expected_cumsum, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_loop_multi_deps_4_iterations() {
        // Test Loop operator with 3 loop-carried dependencies updated independently
        // Values generated from loop_multi_deps.py using ONNX ReferenceEvaluator
        let model: loop_multi_deps::Model<TestBackend> = Default::default();
        let device = Default::default();

        let m = 4i64;
        let cond = true;

        let accum1_init = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.09672398120164871f32, 0.011854584328830242, 0.26395124197006226],
                [0.6290820837020874, 0.268655925989151, -1.2466931343078613],
            ]),
            &device,
        );

        let accum2_init = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [-0.09221914410591125f32, 0.5988513231277466, 0.734139084815979],
                [0.9589252471923828, -1.17632257938385, 0.10864276438951492],
            ]),
            &device,
        );

        let accum3_init = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.6031635999679565f32, 2.3189985752105713, 0.6109723448753357],
                [-0.03926624357700348, 0.5098816752433777, 0.04807303100824356],
            ]),
            &device,
        );

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [1.3561112880706787f32, -1.069679617881775, -0.25508126616477966],
                [0.3487488031387329, 0.6476014256477356, -1.322524070739746],
            ]),
            &device,
        );

        let (accum1_final, accum2_final, accum3_final) =
            model.forward(m, cond, accum1_init, accum2_init, accum3_init, x);

        // Expected outputs from ONNX ReferenceEvaluator after 4 iterations
        // accum1: adds x each iteration -> accum1_init + (4 * x)
        let expected_accum1 = TensorData::from([
            [5.521169662475586f32, -4.26686429977417, -0.7563738822937012],
            [2.0240774154663086, 2.8590614795684814, -6.536789417266846],
        ]);

        // accum2: multiplies by 2 each iteration -> accum2_init * (2^4)
        let expected_accum2 = TensorData::from([
            [-1.47550630569458f32, 9.581621170043945, 11.746225357055664],
            [15.342803955078125, -18.8211612701416, 1.7382842302322388],
        ]);

        // accum3: subtracts 0.5 each iteration -> accum3_init - (4 * 0.5)
        let expected_accum3 = TensorData::from([
            [-1.3968364000320435f32, 0.3189985752105713, -1.3890275955200195],
            [-2.0392661094665527, -1.4901182651519775, -1.9519269466400146],
        ]);

        accum1_final
            .to_data()
            .assert_approx_eq::<f32>(&expected_accum1, burn::tensor::Tolerance::default());
        accum2_final
            .to_data()
            .assert_approx_eq::<f32>(&expected_accum2, burn::tensor::Tolerance::default());
        accum3_final
            .to_data()
            .assert_approx_eq::<f32>(&expected_accum3, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_loop_if_3_levels() {
        // Test 3-level nested subgraph: If -> Loop -> If
        // Values generated from nested_if_loop_if.py using ONNX ReferenceEvaluator
        let model: nested_if_loop_if::Model<TestBackend> = Default::default();
        let device = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    0.07552595436573029f32,
                    0.1409377157688141,
                    -1.145248532295227,
                ],
                [2.0610644817352295, 1.177796483039856, -1.4510904550552368],
            ]),
            &device,
        );

        let m = 3i64;
        let cond = true;

        // Test 1: condition1=True, condition2=True (then->loop->then: x+1, 3 times)
        let condition1 = true;
        let condition2 = true;

        let output = model.forward(x.clone(), m, cond, condition1, condition2);

        let expected = TensorData::from([
            [3.075525999069214f32, 3.1409378051757812, 1.854751467704773],
            [5.061064720153809, 4.177796363830566, 1.5489095449447632],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());

        // Test 2: condition1=True, condition2=False (then->loop->else: x-1, 3 times)
        let condition2_false = false;

        let output2 = model.forward(x.clone(), m, cond, condition1, condition2_false);

        let expected2 = TensorData::from([
            [-2.924474000930786f32, -2.8590621948242188, -4.1452484130859375],
            [-0.9389355182647705, -1.822203516960144, -4.451090335845947],
        ]);

        output2
            .to_data()
            .assert_approx_eq::<f32>(&expected2, burn::tensor::Tolerance::default());

        // Test 3: condition1=False (else: x*2)
        let condition1_false = false;

        let output3 = model.forward(x.clone(), m, cond, condition1_false, condition2);

        let expected3 = TensorData::from([
            [0.15105190873146057f32, 0.2818754315376282, -2.290497064590454],
            [4.122128963470459, 2.355592966079712, -2.9021809101104736],
        ]);

        output3
            .to_data()
            .assert_approx_eq::<f32>(&expected3, burn::tensor::Tolerance::default());
    }

    #[test]
    #[ignore] // TODO: Fix Scan nested in If nested in Loop nested in If
    fn test_nested_if_loop_if_scan_4_levels() {
        // Test 4-level nested subgraph: If -> Loop -> If -> Scan
        // Values generated from nested_if_loop_if_scan.py using ONNX ReferenceEvaluator
        let model: nested_if_loop_if_scan::Model<TestBackend> = Default::default();
        let device = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [-0.5215373635292053f32, 2.5796539783477783, 1.1229078769683838],
                [2.3792245388031006, 0.4096767008304596, -2.2666780948638916],
            ]),
            &device,
        );

        let m = 2i64;
        let cond = true;
        let sum_init = Tensor::<TestBackend, 1>::zeros([3], &device);

        // Test 1: condition1=True, condition2=True (then->loop->then->scan)
        let condition1 = true;
        let condition2 = true;

        let output = model.forward(x.clone(), m, cond, condition1, condition2, sum_init.clone());

        let expected = TensorData::from([
            [3.71537446975708f32, 5.97866153717041, -2.2875404357910156],
            [3.71537446975708, 5.97866153717041, -2.2875404357910156],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());

        // Test 2: condition1=True, condition2=False (then->loop->else)
        let condition2_false = false;

        let output2 =
            model.forward(x.clone(), m, cond, condition1, condition2_false, sum_init.clone());

        let expected2 = TensorData::from([
            [-2.28462553024292f32, -0.021338701248168945, -8.287540435791016],
            [-2.28462553024292, -0.021338701248168945, -8.287540435791016],
        ]);

        output2
            .to_data()
            .assert_approx_eq::<f32>(&expected2, burn::tensor::Tolerance::default());

        // Test 3: condition1=False (else: x*2)
        let condition1_false = false;

        let output3 = model.forward(x.clone(), m, cond, condition1_false, condition2, sum_init);

        let expected3 = TensorData::from([
            [-1.0430747270584106f32, 5.159307956695557, 2.2458157539367676],
            [4.758449077606201, 0.8193534016609192, -4.533356189727783],
        ]);

        output3
            .to_data()
            .assert_approx_eq::<f32>(&expected3, burn::tensor::Tolerance::default());
    }
    */

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
