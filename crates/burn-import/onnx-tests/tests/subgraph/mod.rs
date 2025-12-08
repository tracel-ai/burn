// Tests for subgraph operations
//
// This module tests:
// 1. Deeply nested subgraphs (If, Loop, Scan nested to multiple levels)
// 2. Outer-scope variable references (subgraphs referencing values from parent graph)

use crate::include_models;
include_models!(
    nested_if_loop_if,
    nested_if_loop_if_scan,
    outer_scope_ref,
    outer_scope_multi_var,
    outer_scope_loop,
    outer_scope_scan,
    outer_scope_constant
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_nested_if_loop_if_c1_true_c2_true() {
        // Test 1: condition1=True, condition2=True
        // Path: Outer Then -> Loop(3x) -> Inner Then (x+1 each iteration)
        // Expected: x + 3
        let device = Default::default();
        let model: nested_if_loop_if::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.8520069718360901, 1.4743412733078003, -0.3827274441719055],
                [0.473310261964798, 1.878186821937561, -1.341393232345581],
            ]),
            &device,
        );

        let m = 3i64;
        let cond = true;
        let condition1 = true;
        let condition2 = true;

        let output = model.forward(x, m, cond, condition1, condition2);

        let expected = TensorData::from([
            [3.8520069122314453, 4.47434139251709, 2.6172726154327393],
            [3.4733102321624756, 4.8781867027282715, 1.658606767654419],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_loop_if_c1_true_c2_false() {
        // Test 2: condition1=True, condition2=False
        // Path: Outer Then -> Loop(3x) -> Inner Else (x-1 each iteration)
        // Expected: x - 3
        let device = Default::default();
        let model: nested_if_loop_if::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.8520069718360901, 1.4743412733078003, -0.3827274441719055],
                [0.473310261964798, 1.878186821937561, -1.341393232345581],
            ]),
            &device,
        );

        let m = 3i64;
        let cond = true;
        let condition1 = true;
        let condition2 = false;

        let output = model.forward(x, m, cond, condition1, condition2);

        let expected = TensorData::from([
            [
                -2.1479930877685547,
                -1.5256587266921997,
                -3.3827273845672607,
            ],
            [-2.5266897678375244, -1.121813178062439, -4.34139347076416],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_loop_if_c1_false() {
        // Test 3: condition1=False
        // Path: Outer Else (x*2)
        // Expected: x * 2
        let device = Default::default();
        let model: nested_if_loop_if::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.8520069718360901, 1.4743412733078003, -0.3827274441719055],
                [0.473310261964798, 1.878186821937561, -1.341393232345581],
            ]),
            &device,
        );

        let m = 3i64;
        let cond = true;
        let condition1 = false;
        let condition2 = true;

        let output = model.forward(x, m, cond, condition1, condition2);

        let expected = TensorData::from([
            [1.7040139436721802, 2.9486825466156006, -0.765454888343811],
            [0.946620523929596, 3.756373643875122, -2.682786464691162],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_loop_if_scan_c1_true_c2_true() {
        // Test 1: condition1=True, condition2=True
        // Path: Outer Then -> Loop(2x) -> Inner Then -> Scan (cumsum)
        let device = Default::default();
        let model: nested_if_loop_if_scan::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [-0.6080955266952515, 2.022130012512207, -0.14551113545894623],
                [-0.5326734781265259, 0.9188717007637024, -1.9797388315200806],
            ]),
            &device,
        );

        let m = 2i64;
        let cond = true;
        let condition1 = true;
        let condition2 = true;

        let output = model.forward(x, m, cond, condition1, condition2);

        let expected = TensorData::from([
            [-2.2815380096435547, 5.882003307342529, -4.250499725341797],
            [-2.2815380096435547, 5.882003307342529, -4.250499725341797],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_loop_if_scan_c1_true_c2_false() {
        // Test 2: condition1=True, condition2=False
        // Path: Outer Then -> Loop(2x) -> Inner Else (sum(x-1))
        let device = Default::default();
        let model: nested_if_loop_if_scan::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [-0.6080955266952515, 2.022130012512207, -0.14551113545894623],
                [-0.5326734781265259, 0.9188717007637024, -1.9797388315200806],
            ]),
            &device,
        );

        let m = 2i64;
        let cond = true;
        let condition1 = true;
        let condition2 = false;

        let output = model.forward(x, m, cond, condition1, condition2);

        let expected = TensorData::from([
            [
                -8.281538009643555,
                -0.11799657344818115,
                -10.250499725341797,
            ],
            [
                -8.281538009643555,
                -0.11799657344818115,
                -10.250499725341797,
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_nested_if_loop_if_scan_c1_false() {
        // Test 3: condition1=False
        // Path: Outer Else (x*2)
        // Expected: x * 2
        let device = Default::default();
        let model: nested_if_loop_if_scan::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [-0.6080955266952515, 2.022130012512207, -0.14551113545894623],
                [-0.5326734781265259, 0.9188717007637024, -1.9797388315200806],
            ]),
            &device,
        );

        let m = 2i64;
        let cond = true;
        let condition1 = false;
        let condition2 = true;

        let output = model.forward(x, m, cond, condition1, condition2);

        let expected = TensorData::from([
            [-1.216191053390503, 4.044260025024414, -0.29102227091789246],
            [-1.0653469562530518, 1.8377434015274048, -3.959477663040161],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    // ==========================================================================
    // Outer-scope reference tests
    //
    // These test the DeferredGraph lazy building pattern where a subgraph
    // references a value computed in the parent graph (not passed as explicit input).
    //
    // Pattern being tested:
    //   x = input
    //   y = Relu(x)           // y is computed in parent graph
    //   z = If(condition) {
    //       then_branch: Add(y, bias)   // References 'y' from parent scope
    //       else_branch: Mul(y, scale)  // References 'y' from parent scope
    //   }
    // ==========================================================================

    #[test]
    fn test_outer_scope_ref_then_branch() {
        // condition=True -> then branch -> Relu(x) + 10
        let device = Default::default();
        let model: outer_scope_ref::Model<TestBackend> = Default::default();

        // Input with some negative values (to test Relu)
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-1.5_f32, 2.0, -0.5], [3.0, -2.0, 1.0]]),
            &device,
        );

        let condition = true;
        let output = model.forward(x, condition);

        // y = Relu(x) = [[0, 2, 0], [3, 0, 1]]
        // output = y + 10 = [[10, 12, 10], [13, 10, 11]]
        let expected = TensorData::from([[10.0_f32, 12.0, 10.0], [13.0, 10.0, 11.0]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_outer_scope_ref_else_branch() {
        // condition=False -> else branch -> Relu(x) * 2
        let device = Default::default();
        let model: outer_scope_ref::Model<TestBackend> = Default::default();

        // Input with some negative values (to test Relu)
        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-1.5_f32, 2.0, -0.5], [3.0, -2.0, 1.0]]),
            &device,
        );

        let condition = false;
        let output = model.forward(x, condition);

        // y = Relu(x) = [[0, 2, 0], [3, 0, 1]]
        // output = y * 2 = [[0, 4, 0], [6, 0, 2]]
        let expected = TensorData::from([[0.0_f32, 4.0, 0.0], [6.0, 0.0, 2.0]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    // ==========================================================================
    // Multi-variable outer-scope reference tests
    //
    // These test that multiple variables from parent scope are correctly passed
    // to subgraphs. Pattern:
    //   y1 = Relu(x)
    //   y2 = Sigmoid(x)
    //   y3 = Tanh(x)
    //   z = If(condition) {
    //       then: y1 + y2 + y3  // References 3 vars from parent
    //       else: y1 * y2 * y3  // References 3 vars from parent
    //   }
    // ==========================================================================

    #[test]
    fn test_outer_scope_multi_var_then_branch() {
        // condition=True -> then branch -> y1 + y2 + y3
        let device = Default::default();
        let model: outer_scope_multi_var::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-1.5_f32, 2.0, -0.5], [3.0, -2.0, 1.0]]),
            &device,
        );

        let condition = true;
        let output = model.forward(x, condition);

        // y1 = Relu(x), y2 = Sigmoid(x), y3 = Tanh(x)
        // output = y1 + y2 + y3
        //
        // For x = [[-1.5, 2.0, -0.5], [3.0, -2.0, 1.0]]:
        //   y1 = Relu(x)    = [[0, 2, 0], [3, 0, 1]]
        //   y2 = Sigmoid(x) = [[0.182, 0.881, 0.378], [0.953, 0.119, 0.731]]
        //   y3 = Tanh(x)    = [[-0.905, 0.964, -0.462], [0.995, -0.964, 0.762]]
        //   y1 + y2 + y3    = [[-0.723, 3.845, -0.085], [4.948, -0.845, 2.493]]
        let expected = TensorData::from([
            [-0.72272265_f32, 3.8448248, -0.08457646],
            [4.947629, -0.8448247, 2.492653],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_outer_scope_multi_var_else_branch() {
        // condition=False -> else branch -> y1 * y2 * y3
        let device = Default::default();
        let model: outer_scope_multi_var::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-1.5_f32, 2.0, -0.5], [3.0, -2.0, 1.0]]),
            &device,
        );

        let condition = false;
        let output = model.forward(x, condition);

        // y1 = Relu(x), y2 = Sigmoid(x), y3 = Tanh(x)
        // output = y1 * y2 * y3
        //
        // For x = [[-1.5, 2.0, -0.5], [3.0, -2.0, 1.0]]:
        //   y1 = Relu(x)    = [[0, 2, 0], [3, 0, 1]]
        //   y2 = Sigmoid(x) = [[0.182, 0.881, 0.378], [0.953, 0.119, 0.731]]
        //   y3 = Tanh(x)    = [[-0.905, 0.964, -0.462], [0.995, -0.964, 0.762]]
        //   y1 * y2 * y3    = [[0, 1.698, 0], [2.844, 0, 0.557]]
        let expected = TensorData::from([[0.0_f32, 1.6982254, 0.0], [2.84359, 0.0, 0.55676997]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    // ==========================================================================
    // Loop outer-scope reference test
    //
    // Pattern:
    //   y = Relu(x)
    //   z = Loop(3) {
    //       body: accum = Add(accum, y)  // References y from parent
    //   }
    //   Result: 0 + y + y + y = 3*y
    // ==========================================================================

    #[test]
    fn test_outer_scope_loop() {
        let device = Default::default();
        let model: outer_scope_loop::Model<TestBackend> = Default::default();

        let x = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-1.5_f32, 2.0, -0.5], [3.0, -2.0, 1.0]]),
            &device,
        );
        let max_iter = 3i64;
        let cond_init = true;
        let accum_init = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            &device,
        );

        let output = model.forward(x, max_iter, cond_init, accum_init);

        // y = Relu(x) = [[0, 2, 0], [3, 0, 1]]
        // output = 0 + y + y + y = 3*y = [[0, 6, 0], [9, 0, 3]]
        let expected = TensorData::from([[0.0_f32, 6.0, 0.0], [9.0, 0.0, 3.0]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    // ==========================================================================
    // Scan outer-scope reference test
    //
    // Pattern:
    //   y = Relu(x)           # Shape [3]
    //   z = Scan(sequence) {  # sequence shape [4, 3]
    //       body: scan_out = Add(elem, y)  // References y from parent
    //   }
    //   Result: each row of sequence + y
    // ==========================================================================

    #[test]
    fn test_outer_scope_scan() {
        let device = Default::default();
        let model: outer_scope_scan::Model<TestBackend> = Default::default();

        let x =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([-1.0_f32, 2.0, 0.5]), &device);
        let sequence = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [1.0_f32, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]),
            &device,
        );

        let output = model.forward(x, sequence);

        // y = Relu(x) = [0, 2, 0.5]
        // scan_out[i] = sequence[i] + y
        let expected = TensorData::from([
            [1.0_f32, 4.0, 3.5],
            [4.0, 7.0, 6.5],
            [7.0, 10.0, 9.5],
            [10.0, 13.0, 12.5],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    // ==========================================================================
    // Outer-scope CONSTANT/INITIALIZER reference tests
    //
    // These test that constants/initializers defined in the parent graph can be
    // accessed from subgraphs (If branches). This is different from outer_scope_ref
    // which tests computed values.
    //
    // Pattern:
    //   weight = parent_initializer [3, 2]  # Defined in parent graph
    //   bias = parent_initializer [2]       # Defined in parent graph
    //   z = If(condition) {
    //       then: MatMul(x, weight) + bias  # Uses parent's weight/bias
    //       else: x[:, :2] * 2              # Simple fallback
    //   }
    //
    // This pattern is common in real models like Silero VAD where Conv layers
    // inside If branches use weights from the parent graph.
    // ==========================================================================

    #[test]
    fn test_outer_scope_constant_then_branch() {
        // condition=True -> then branch -> MatMul(x, weight) + bias
        // Uses weight and bias initializers from parent graph
        let device = Default::default();
        let model: outer_scope_constant::Model<TestBackend> = Default::default();

        let x =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0_f32, 2.0, 3.0]]), &device);

        let condition = true;
        let output = model.forward(x, condition);

        // x @ weight + bias
        // [[1, 2, 3]] @ [[1, 0.5], [2, 1], [0.5, 2]] + [0.1, 0.2]
        // = [[1*1 + 2*2 + 3*0.5, 1*0.5 + 2*1 + 3*2]] + [0.1, 0.2]
        // = [[6.5, 8.5]] + [0.1, 0.2] = [[6.6, 8.7]]
        let expected = TensorData::from([[6.6_f32, 8.7]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn test_outer_scope_constant_else_branch() {
        // condition=False -> else branch -> x[:, :2] * 2
        let device = Default::default();
        let model: outer_scope_constant::Model<TestBackend> = Default::default();

        let x =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0_f32, 2.0, 3.0]]), &device);

        let condition = false;
        let output = model.forward(x, condition);

        // x[:, :2] * 2 = [[1, 2]] * 2 = [[2, 4]]
        let expected = TensorData::from([[2.0_f32, 4.0]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
