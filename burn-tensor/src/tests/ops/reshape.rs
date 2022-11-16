#[macro_export]
macro_rules! test_reshape {
    () => {
        #[test]
        fn should_support_reshape_1d() {
            let data = Data::from([0.0, 1.0, 2.0]);
            let tensor = Tensor::<TestBackend, 1>::from_data(data);

            let data_actual = tensor.reshape(Shape::new([1, 3])).into_data();

            let data_expected = Data::from([[0.0, 1.0, 2.0]]);
            assert_eq!(data_expected, data_actual);
        }

        #[test]
        fn should_support_reshape_2d() {
            let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
            let tensor = Tensor::<TestBackend, 2>::from_data(data);

            let data_actual = tensor.reshape(Shape::new([6])).into_data();

            let data_expected = Data::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(data_expected, data_actual);
        }
    };
}
