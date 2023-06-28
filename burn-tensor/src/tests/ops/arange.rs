#[burn_tensor_testgen::testgen(arange)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_arange() {
        let tensor = Tensor::<TestBackend, 1, Int>::arange(2..5);
        assert_eq!(tensor.into_data(), Data::from([2, 3, 4]));
    }
}
