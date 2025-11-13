// Tests for deeply nested subgraph operations
// These tests verify that If, Loop, and Scan operators can be nested to multiple levels

use crate::include_models;
include_models!(nested_if_loop_if, nested_if_loop_if_scan);

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
}
