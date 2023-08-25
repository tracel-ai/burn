#[burn_tensor_testgen::testgen(random)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn rand_default() {
        let tensor = Tensor::<TestBackend, 1>::random([20], Distribution::Default);

        // check that the tensor is within the range of [0..1) (1 is exclusive)
        tensor.into_data().assert_within_range(0.0..1.0);
    }
}
