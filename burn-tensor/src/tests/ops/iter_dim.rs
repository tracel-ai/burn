#[burn_tensor_testgen::testgen(iter_dim)]
mod test {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_1d_iter_last_item() {
        let data = [1, 2, 3, 4];
        let tensor = Tensor::<TestBackend, 1, Int>::from_ints_devauto(data);
        assert_eq!(
            Tensor::<TestBackend, 1, Int>::from_ints_devauto([4]).into_data(),
            tensor.iter_dim(0).last().unwrap().into_data()
        )
    }

    #[test]
    #[should_panic]
    fn test_too_high_dimension() {
        Tensor::<TestBackend, 1>::zeros_devauto([10]).iter_dim(1);
    }

    #[test]
    fn test_transposed() {
        let data = [
            [1., 2., 3., 1., 2.],
            [4., 5., 6., 1., 2.],
            [7., 8., 9., 1., 2.],
        ];
        let tensor = Tensor::<TestBackend, 2>::from_floats_devauto(data);
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
        let tensor = Tensor::<TestBackend, 3>::from_floats_devauto(data);
        let lhs = tensor.iter_dim(2).nth(1).unwrap();
        let rhs = Data::from([2., 5., 8.]);
        assert_eq!(lhs.into_data().value, rhs.value);
    }

    #[test]
    fn test_iter_dim_double_end() {
        let input =
            Tensor::<TestBackend, 1, Int>::arange_devauto(0..(4 * 6 * 3)).reshape([4, 6, 3]);
        let mut iter = input.iter_dim(1);

        let ele0 = Data::from([[[0, 1, 2]], [[18, 19, 20]], [[36, 37, 38]], [[54, 55, 56]]]);
        let ele1 = Data::from([[[3, 4, 5]], [[21, 22, 23]], [[39, 40, 41]], [[57, 58, 59]]]);
        let ele2 = Data::from([[[6, 7, 8]], [[24, 25, 26]], [[42, 43, 44]], [[60, 61, 62]]]);
        let ele3 = Data::from([
            [[9, 10, 11]],
            [[27, 28, 29]],
            [[45, 46, 47]],
            [[63, 64, 65]],
        ]);
        let ele4 = Data::from([
            [[12, 13, 14]],
            [[30, 31, 32]],
            [[48, 49, 50]],
            [[66, 67, 68]],
        ]);
        let ele5 = Data::from([
            [[15, 16, 17]],
            [[33, 34, 35]],
            [[51, 52, 53]],
            [[69, 70, 71]],
        ]);

        assert_eq!(iter.next().unwrap().into_data(), ele0);
        assert_eq!(iter.next_back().unwrap().into_data(), ele5);
        assert_eq!(iter.next_back().unwrap().into_data(), ele4);
        assert_eq!(iter.next().unwrap().into_data(), ele1);
        assert_eq!(iter.next().unwrap().into_data(), ele2);
        assert_eq!(iter.next().unwrap().into_data(), ele3);
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_iter_dim_single_element() {
        let input =
            Tensor::<TestBackend, 1, Int>::arange_devauto(0..(4 * 1 * 3)).reshape([4, 1, 3]);

        let mut iter = input.clone().iter_dim(1);
        assert_eq!(iter.next().unwrap().into_data(), input.clone().into_data());
        assert!(iter.next_back().is_none());
        assert!(iter.next().is_none());

        let mut iter = input.clone().iter_dim(1);
        assert_eq!(
            iter.next_back().unwrap().into_data(),
            input.clone().into_data()
        );
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}
