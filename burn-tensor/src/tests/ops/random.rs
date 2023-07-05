#[burn_tensor_testgen::testgen(random)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn rand_standard() {
        let tensor = Tensor::<TestBackend, 1>::random([20], Distribution::Standard);

        // check that the tensor is within the range of [0..1) (1 is exclusive)
        // (we subtract f32::EPSILON to make sure that 1.0 is not included)
        tensor
            .into_data()
            .assert_within_range(0.0..1.0_f32 - f32::EPSILON);
    }
}
