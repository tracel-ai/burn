#[burn_tensor_testgen::testgen(iter_dim)]
mod test {
    use super::*;
    use burn_tensor::{backend::Backend, Int, Tensor, TensorData};

    #[test]
    fn test_1d_iter_last_item() {
        let data = [1, 2, 3, 4];
        let device = Default::default();
        let tensor = TestTensorInt::<1>::from_ints(data, &device);
        assert_eq!(
            TestTensorInt::<1>::from_ints([4], &device).into_data(),
            tensor.iter_dim(0).last().unwrap().into_data()
        )
    }

    #[test]
    #[should_panic]
    fn test_too_high_dimension() {
        TestTensor::<1>::zeros([10], &Default::default()).iter_dim(1);
    }

    #[test]
    fn test_transposed() {
        let data = [
            [1., 2., 3., 1., 2.],
            [4., 5., 6., 1., 2.],
            [7., 8., 9., 1., 2.],
        ];
        let tensor = TestTensor::<2>::from_floats(data, &Default::default());
        let lhs = tensor.clone().slice([1..2, 0..5]);
        let rhs = tensor.transpose().iter_dim(1).nth(1).unwrap();
        assert_eq!(
            lhs.into_data()
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap(),
            rhs.into_data()
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap()
        );
    }

    fn test_iteration_over_low_dim() {
        let data = [[
            [1., 2., 3., 1., 2.],
            [4., 5., 6., 1., 2.],
            [7., 8., 9., 1., 2.],
        ]; 5];
        let tensor = TestTensor::<3>::from_floats(data, &Default::default());
        let lhs = tensor.iter_dim(2).nth(1).unwrap();
        let rhs = TensorData::from([2., 5., 8.]);
        assert_eq!(
            lhs.into_data()
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap(),
            rhs.as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap()
        );
    }

    #[test]
    fn test_iter_dim_double_end() {
        let input =
            TestTensorInt::<1>::arange(0..(4 * 6 * 3), &Default::default()).reshape([4, 6, 3]);
        let mut iter = input.iter_dim(1);

        let ele0 = TensorData::from([[[0, 1, 2]], [[18, 19, 20]], [[36, 37, 38]], [[54, 55, 56]]]);
        let ele1 = TensorData::from([[[3, 4, 5]], [[21, 22, 23]], [[39, 40, 41]], [[57, 58, 59]]]);
        let ele2 = TensorData::from([[[6, 7, 8]], [[24, 25, 26]], [[42, 43, 44]], [[60, 61, 62]]]);
        let ele3 = TensorData::from([
            [[9, 10, 11]],
            [[27, 28, 29]],
            [[45, 46, 47]],
            [[63, 64, 65]],
        ]);
        let ele4 = TensorData::from([
            [[12, 13, 14]],
            [[30, 31, 32]],
            [[48, 49, 50]],
            [[66, 67, 68]],
        ]);
        let ele5 = TensorData::from([
            [[15, 16, 17]],
            [[33, 34, 35]],
            [[51, 52, 53]],
            [[69, 70, 71]],
        ]);

        iter.next().unwrap().into_data().assert_eq(&ele0, false);
        iter.next_back()
            .unwrap()
            .into_data()
            .assert_eq(&ele5, false);
        iter.next_back()
            .unwrap()
            .into_data()
            .assert_eq(&ele4, false);
        iter.next().unwrap().into_data().assert_eq(&ele1, false);
        iter.next().unwrap().into_data().assert_eq(&ele2, false);
        iter.next().unwrap().into_data().assert_eq(&ele3, false);
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_iter_dim_single_element() {
        let input =
            TestTensorInt::<1>::arange(0..(4 * 1 * 3), &Default::default()).reshape([4, 1, 3]);

        let mut iter = input.clone().iter_dim(1);
        iter.next()
            .unwrap()
            .into_data()
            .assert_eq(&input.clone().into_data(), false);
        assert!(iter.next_back().is_none());
        assert!(iter.next().is_none());

        let mut iter = input.clone().iter_dim(1);
        iter.next_back()
            .unwrap()
            .into_data()
            .assert_eq(&input.clone().into_data(), false);
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}
