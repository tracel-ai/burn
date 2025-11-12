// Tests for ONNX Loop operator

use crate::include_models;
include_models!(
    loop_simple,
    loop_dynamic_cond,
    loop_multi_deps,
    loop_nested,
    loop_scan_outputs
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn loop_simple_3_iterations() {
        // Test with M=3 iterations
        let device = Default::default();
        let model: loop_simple::Model<TestBackend> = Default::default();

        // M = 3
        let m = 3i64;
        let cond = true;

        // Initial accumulator
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [1.5427501201629639, 0.01986491121351719, 0.6051717400550842],
                [-0.1073952168226242, 1.2809762954711914, -1.9534032344818115],
            ]),
            &device,
        );

        // X tensor
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [1.571418046951294, 2.1232078075408936, 0.008226290345191956],
                [0.16664840281009674, -0.9279910326004028, 0.4096680283546448],
            ]),
            &device,
        );

        let output = model.forward(m, cond, initial_accum, x);

        let expected = TensorData::from([
            [48.341854095458984, 43.883827209472656, 18.95654296875],
            [15.473916053771973, 11.255935668945312, 4.108126163482666],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_simple_5_iterations() {
        // Test with M=5 iterations
        let device = Default::default();
        let model: loop_simple::Model<TestBackend> = Default::default();

        // M = 5
        let m = 5i64;
        let cond = true;

        // Initial accumulator
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [1.1227790117263794, 0.9607051610946655, -1.2476868629455566],
                [0.0905989333987236, 0.4496997892856598, -0.4249286651611328],
            ]),
            &device,
        );

        // X tensor
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    -1.5024702548980713,
                    -0.9480401873588562,
                    -0.37771573662757874,
                ],
                [1.2145531177520752, -1.4656798839569092, 0.1872921586036682],
            ]),
            &device,
        );

        let output = model.forward(m, cond, initial_accum, x);

        let expected = TensorData::from([
            [4.775772571563721, 33.964073181152344, -1.344353437423706],
            [140.20144653320312, -14.481760025024414, 60.01439666748047],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_simple_0_iterations() {
        // Test with M=0 (no iterations - should return initial accumulator unchanged)
        let device = Default::default();
        let model: loop_simple::Model<TestBackend> = Default::default();

        // M = 0
        let m = 0i64;
        let cond = true;

        // Initial accumulator
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.08663585782051086, -1.081067442893982, -1.0119551420211792],
                [0.35541200637817383, 0.308596134185791, 0.8396860361099243],
            ]),
            &device,
        );

        // X tensor (won't be used but needed as input)
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.9175063967704773, -2.750408411026001, 0.5216607451438904],
                [
                    -0.18170106410980225,
                    -0.9019166231155396,
                    0.17441026866436005,
                ],
            ]),
            &device,
        );

        let output = model.forward(m, cond, initial_accum.clone(), x);

        // Expected output should be the same as initial accumulator
        output
            .to_data()
            .assert_approx_eq::<f32>(&initial_accum.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_dynamic_cond_counter_5() {
        // Test with counter starting at 5.0
        // Loop decrements counter until it reaches 0 (dynamic condition)
        let device = Default::default();
        let model: loop_dynamic_cond::Model<TestBackend> = Default::default();

        // M = 100 (max iterations, but loop will stop early)
        let m = 100i64;
        let cond_init = true;

        // Counter starts at 5.0
        let counter_init = Tensor::<TestBackend, 1>::from_data(TensorData::from([5.0]), &device);

        let output = model.forward(m, cond_init, counter_init);

        // Expected: counter reaches 0.0 after 5 iterations
        let expected = TensorData::from([0.0]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_dynamic_cond_counter_3() {
        // Test with counter starting at 3.0
        let device = Default::default();
        let model: loop_dynamic_cond::Model<TestBackend> = Default::default();

        let m = 100i64;
        let cond_init = true;

        // Counter starts at 3.0
        let counter_init = Tensor::<TestBackend, 1>::from_data(TensorData::from([3.0]), &device);

        let output = model.forward(m, cond_init, counter_init);

        // Expected: counter reaches 0.0 after 3 iterations
        let expected = TensorData::from([0.0]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_multi_deps_4_iterations() {
        // Test with 3 loop-carried dependencies and 4 iterations
        let device = Default::default();
        let model: loop_multi_deps::Model<TestBackend> = Default::default();

        // M = 4
        let m = 4i64;
        let cond = true;

        // accum1: adds x each iteration
        let accum1_init = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    -0.9759665131568909,
                    -0.6875132322311401,
                    -0.06939402967691422,
                ],
                [
                    0.018175272271037102,
                    -0.37374961376190186,
                    1.3255383968353271,
                ],
            ]),
            &device,
        );

        // accum2: multiplies by 2 each iteration
        let accum2_init = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [1.0530825853347778, 1.3591053485870361, 1.4057459831237793],
                [2.684823989868164, -1.2770761251449585, 1.3482451438903809],
            ]),
            &device,
        );

        // accum3: subtracts 0.5 each iteration
        let accum3_init = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    0.3927198052406311,
                    -0.01692209206521511,
                    -0.33274683356285095,
                ],
                [-0.759807288646698, 1.6555919647216797, -0.6027320027351379],
            ]),
            &device,
        );

        // x: read-only input used in accum1 calculation
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [
                    -0.2069942057132721,
                    0.5069937109947205,
                    -0.39182549715042114,
                ],
                [-0.9965894222259521, -0.6853103041648865, -0.295870840549469],
            ]),
            &device,
        );

        let (out1, out2, out3) = model.forward(m, cond, accum1_init, accum2_init, accum3_init, x);

        // accum1: Each iteration adds x, so final = accum1_init + (4 * x)
        let expected1 = TensorData::from([
            [-1.8039432764053345, 1.3404616117477417, -1.6366958618164062],
            [-3.968182325363159, -3.1149909496307373, 0.14205509424209595],
        ]);

        // accum2: Each iteration multiplies by 2, so final = accum2_init * (2^4)
        let expected2 = TensorData::from([
            [16.849321365356445, 21.745685577392578, 22.49193572998047],
            [42.957183837890625, -20.433218002319336, 21.571922302246094],
        ]);

        // accum3: Each iteration subtracts 0.5, so final = accum3_init - (4 * 0.5)
        let expected3 = TensorData::from([
            [-1.6072802543640137, -2.0169219970703125, -2.332746982574463],
            [-2.7598073482513428, -0.3444080352783203, -2.602731943130493],
        ]);

        out1.to_data()
            .assert_approx_eq::<f32>(&expected1, burn::tensor::Tolerance::default());
        out2.to_data()
            .assert_approx_eq::<f32>(&expected2, burn::tensor::Tolerance::default());
        out3.to_data()
            .assert_approx_eq::<f32>(&expected3, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_nested_test1() {
        // Test nested loops: outer=2, inner=3
        // Outer loop runs 2 times, inner loop runs 3 times per outer iteration
        let device = Default::default();
        let model: loop_nested::Model<TestBackend> = Default::default();

        let m_outer = 2i64;
        let m_inner = 3i64;
        let cond_init = true;
        let sum_init = Tensor::<TestBackend, 1>::from_data(TensorData::from([0.0]), &device);

        let (sum_output, _m_inner_output) = model.forward(m_outer, m_inner, cond_init, sum_init);

        // Expected: Each outer iteration runs inner loop 3 times (adds 3*3=9), then adds 1
        // Iteration 1: 0 + 9 + 1 = 10
        // Iteration 2: 10 + 9 + 1 = 20
        let expected = TensorData::from([20.0]);

        sum_output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_nested_test2() {
        // Test nested loops: outer=3, inner=2, starting sum=5.0
        let device = Default::default();
        let model: loop_nested::Model<TestBackend> = Default::default();

        let m_outer = 3i64;
        let m_inner = 2i64;
        let cond_init = true;
        let sum_init = Tensor::<TestBackend, 1>::from_data(TensorData::from([5.0]), &device);

        let (sum_output, _m_inner_output) = model.forward(m_outer, m_inner, cond_init, sum_init);

        // Expected: Each outer iteration runs inner loop 2 times (adds 2*3=6), then adds 1
        // Iteration 1: 5 + 6 + 1 = 12
        // Iteration 2: 12 + 6 + 1 = 19
        // Iteration 3: 19 + 6 + 1 = 26
        let expected = TensorData::from([26.0]);

        sum_output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_scan_outputs_3_iterations() {
        // Test loop with scan outputs (M=3 iterations)
        // Collects intermediate accumulator values and iteration numbers
        let device = Default::default();
        let model: loop_scan_outputs::Model<TestBackend> = Default::default();

        let m = 3i64;
        let cond = true;

        // Initial accumulator [2, 3]
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let (final_accum, accumulated_values, iteration_numbers) =
            model.forward(m, cond, initial_accum);

        // Final accumulator after 3 iterations (each iteration adds 1.0)
        let expected_final = TensorData::from([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        // Scan output: accumulated_values concatenated along axis 0
        // Shape: [6, 3] = [3 iters * 2 batch, 3 features]
        // Iteration 0: [[1,2,3], [4,5,6]]
        // Iteration 1: [[2,3,4], [5,6,7]]
        // Iteration 2: [[3,4,5], [6,7,8]]
        let expected_accumulated = TensorData::from([
            [1.0, 2.0, 3.0], // iter 0, batch 0
            [4.0, 5.0, 6.0], // iter 0, batch 1
            [2.0, 3.0, 4.0], // iter 1, batch 0
            [5.0, 6.0, 7.0], // iter 1, batch 1
            [3.0, 4.0, 5.0], // iter 2, batch 0
            [6.0, 7.0, 8.0], // iter 2, batch 1
        ]);

        // Scan output: iteration numbers
        // Shape: [3, 1] (ONNX adds dimension for scalars)
        let expected_iterations = TensorData::from([[0.0], [1.0], [2.0]]);

        final_accum
            .to_data()
            .assert_approx_eq::<f32>(&expected_final, burn::tensor::Tolerance::default());
        accumulated_values
            .to_data()
            .assert_approx_eq::<f32>(&expected_accumulated, burn::tensor::Tolerance::default());
        iteration_numbers
            .to_data()
            .assert_approx_eq::<f32>(&expected_iterations, burn::tensor::Tolerance::default());
    }

    #[test]
    fn loop_scan_outputs_1_iteration() {
        // Test loop with scan outputs (M=1 iteration - edge case)
        let device = Default::default();
        let model: loop_scan_outputs::Model<TestBackend> = Default::default();

        let m = 1i64;
        let cond = true;

        // Initial accumulator [2, 3]
        let initial_accum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            &device,
        );

        let (final_accum, accumulated_values, iteration_numbers) =
            model.forward(m, cond, initial_accum.clone());

        // Final accumulator after 1 iteration
        let expected_final = TensorData::from([[11.0, 21.0, 31.0], [41.0, 51.0, 61.0]]);

        // Scan output: only 1 iteration, so shape [2, 3] (same as input)
        let expected_accumulated = TensorData::from([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]);

        // Iteration numbers: [1, 1] for single iteration
        let expected_iterations = TensorData::from([[0.0]]);

        final_accum
            .to_data()
            .assert_approx_eq::<f32>(&expected_final, burn::tensor::Tolerance::default());
        accumulated_values
            .to_data()
            .assert_approx_eq::<f32>(&expected_accumulated, burn::tensor::Tolerance::default());
        iteration_numbers
            .to_data()
            .assert_approx_eq::<f32>(&expected_iterations, burn::tensor::Tolerance::default());
    }
}
