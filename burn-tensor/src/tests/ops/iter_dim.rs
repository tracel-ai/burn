#[burn_tensor_testgen::testgen(iter_dim)]
mod test {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_1d_iter_last_item() {
        let data = [1, 2, 3, 4];
        let tensor = Tensor::<TestBackend, 1, Int>::from_ints(data);
        assert_eq!(
            Tensor::<TestBackend, 1, Int>::from_ints([4]).into_data(),
            tensor.iter_dim(0).last().unwrap().into_data()
        )
    }

    #[test]
    #[should_panic]
    fn test_too_high_dimension() {
        Tensor::<TestBackend, 1>::zeros([10]).iter_dim(1);
    }

    #[test]
    fn test_transposed() {
        let data = [
            [1., 2., 3., 1., 2.],
            [4., 5., 6., 1., 2.],
            [7., 8., 9., 1., 2.],
        ];
        let tensor = Tensor::<TestBackend, 2>::from_floats(data);
        let lhs = tensor.clone().slice([1..2, 0..5]);
        let rhs = tensor.transpose().iter_dim(1).nth(1).unwrap();
        assert_eq!(lhs.into_data().value, rhs.into_data().value);
    }

    fn test_iteration_over_low_dim() {
        let data = [[
            [1., 2., 3., 1., 2.],
            [4., 5., 6., 1., 2.],
            [7., 8., 9., 1., 2.],
        ]; 5];
        let tensor = Tensor::<TestBackend, 3>::from_floats(data);
        let lhs = tensor.iter_dim(2).nth(1).unwrap();
        let rhs = Data::from([2., 5., 8.]);
        assert_eq!(lhs.into_data().value, rhs.value);
    }
}
