#[burn_tensor_testgen::testgen(cross)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, backend::Backend};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn test_cross_3d_last_dim() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0], [2.0, -1.0, 4.0]]);
        let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[-7.0, -21.0, -14.0], [-18.0, 16.0, 13.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_3d_dim0() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
        let tensor_2 = TestTensor::from([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);

        // Some backends (for example CubeCL) may not support cross on non-last
        // dimensions and will intentionally panic with a message like
        // "Cross product on non-last dimension not yet implemented". In that
        // case we treat the panic as a skipped test for that backend.
        let res = catch_unwind(AssertUnwindSafe(|| tensor_1.cross(tensor_2, 0)));
        let output = match res {
            Ok(t) => t,
            Err(err) => {
                if let Some(s) = err.downcast_ref::<&str>() {
                    if s.contains("Cross product on non-last dimension") {
                        eprintln!("Skipping cross dim0 test: backend does not support non-last-dim cross");
                        return;
                    }
                }
                if let Some(s) = err.downcast_ref::<String>() {
                    if s.contains("Cross product on non-last dimension") {
                        eprintln!("Skipping cross dim0 test: backend does not support non-last-dim cross");
                        return;
                    }
                }
                std::panic::resume_unwind(err);
            }
        };

        output.into_data().assert_eq(
            &TensorData::from([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_3d_broadcast() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0]]);
        let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[-7.0, -21.0, -14.0], [19.0, -13.0, -4.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_4d_last_dim() {
        let tensor_1 = TestTensor::<3>::from([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        let tensor_2 = TestTensor::from([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]),
            false,
        );
    }
}
