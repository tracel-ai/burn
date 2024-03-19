#[burn_tensor_testgen::testgen(reduction)]
mod reduction {
    use super::*;
    use burn_jit::kernel::reduce::{
        argmax, argmin, mean_dim, prod, prod_dim, sum, sum_dim, ReduceStrategy,
    };
    use burn_tensor::{ops::IntTensorOps, Data, Distribution, Int, Shape, Tensor};

    #[test]
    fn reduction_sum_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val = Tensor::<TestBackend, 2>::from_primitive(sum_dim::<TestRuntime, f32, f32, 2>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::Naive,
        ));
        let val_ref = tensor_ref.sum_dim(1);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_prod_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val = Tensor::<TestBackend, 2>::from_primitive(prod_dim::<TestRuntime, f32, f32, 2>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::Naive,
        ));
        let val_ref = tensor_ref.prod_dim(1);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_argmin_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val =
            Tensor::<TestBackend, 2, Int>::from_primitive(argmin::<TestRuntime, f32, i32, 2>(
                tensor.into_primitive(),
                reduce_dim,
                ReduceStrategy::Naive,
            ));
        let val_ref = tensor_ref.argmin(reduce_dim);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }

    #[test]
    fn reduction_argmax_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val =
            Tensor::<TestBackend, 2, Int>::from_primitive(argmax::<TestRuntime, f32, i32, 2>(
                tensor.into_primitive(),
                reduce_dim,
                ReduceStrategy::Naive,
            ));
        let val_ref = tensor_ref.argmax(reduce_dim);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }

    #[test]
    fn sum_dim_should_work_with_int() {
        let summed_shape = Shape::new([1]);
        let data = Data::from([1, 2, 3, 4]);
        let tensor = TestBackend::int_from_data(data, &Default::default());

        let val = Tensor::<TestBackend, 1, Int>::from_primitive(
            sum_dim::<TestRuntime, i32, i32, 1>(tensor, 0, ReduceStrategy::Naive),
        );

        let sum_as_data = Data::from([10]);
        val.into_data().assert_approx_eq(&sum_as_data, 1);
    }

    #[test]
    fn mean_dim_should_work_with_int() {
        let mean_shape = Shape::new([1]);
        let data = Data::from([1, 2, 3, 4]);
        let tensor = TestBackend::int_from_data(data, &Default::default());

        let val = Tensor::<TestBackend, 1, Int>::from_primitive(
            mean_dim::<TestRuntime, i32, i32, 1>(tensor, 0, ReduceStrategy::Naive),
        );

        // Mean calculation truncates to an integer
        let mean_as_data = Data::from([2]);
        val.into_data().assert_approx_eq(&mean_as_data, 1);
    }

    #[test]
    fn reduction_sum_dim_shared_memory_small() {
        let tensor =
            Tensor::<TestBackend, 1>::random([700], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 1>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 0;

        let val = Tensor::<TestBackend, 1>::from_primitive(sum_dim::<TestRuntime, f32, f32, 1>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::SharedMemory,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_sum_dim_shared_memory_medium_divisible() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val = Tensor::<TestBackend, 2>::from_primitive(sum_dim::<TestRuntime, f32, f32, 2>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::SharedMemory,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_sum_dim_shared_memory_medium_not_divisible() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1025], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val = Tensor::<TestBackend, 2>::from_primitive(sum_dim::<TestRuntime, f32, f32, 2>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::SharedMemory,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_sum_dim_shared_memory_large() {
        let tensor = Tensor::<TestBackend, 3>::random(
            [4, 1024, 50],
            Distribution::Default,
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val = Tensor::<TestBackend, 3>::from_primitive(sum_dim::<TestRuntime, f32, f32, 3>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::SharedMemory,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_mean_dim_shared_memory_medium() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 0;

        let val = Tensor::<TestBackend, 2>::from_primitive(mean_dim::<TestRuntime, f32, f32, 2>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::SharedMemory,
        ));
        let val_ref = tensor_ref.mean_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_argmin_shared_memory_medium() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val = Tensor::<TestBackend, 2>::from_primitive(argmin::<TestRuntime, f32, f32, 2>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::SharedMemory,
        ));
        let val_ref = tensor_ref.argmin(reduce_dim);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }

    #[test]
    fn reduction_argmax_shared_memory_medium() {
        let tensor = Tensor::<TestBackend, 2>::random(
            [10, 3000],
            Distribution::Default,
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;

        let val = Tensor::<TestBackend, 2>::from_primitive(argmax::<TestRuntime, f32, f32, 2>(
            tensor.into_primitive(),
            reduce_dim,
            ReduceStrategy::SharedMemory,
        ));
        let val_ref = tensor_ref.argmax(reduce_dim);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }

    #[test]
    fn reduction_sum_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val = Tensor::<TestBackend, 1>::from_primitive(sum(
            tensor.into_primitive(),
            ReduceStrategy::default(),
        ));
        let val_ref = tensor_ref.sum();

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_prod_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val = Tensor::<TestBackend, 1>::from_primitive(prod(
            tensor.into_primitive(),
            ReduceStrategy::default(),
        ));
        let val_ref = tensor_ref.prod();

        val_ref.into_data().assert_approx_eq(&val.into_data(), 2);
    }

    #[test]
    fn reduction_argmax_shared_memory_extreme_values_float() {
        let data: Data<f32, 1> = Data::from([-999999., -999997., -999998.]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &Default::default());

        let val_shared =
            Tensor::<TestBackend, 1, Int>::from_primitive(argmax::<TestRuntime, f32, i32, 1>(
                tensor.into_primitive(),
                0,
                ReduceStrategy::SharedMemory,
            ));

        assert_eq!(1, val_shared.into_data().value[0]);
    }

    #[test]
    fn reduction_argmin_shared_memory_extreme_values_float() {
        let data: Data<f32, 1> = Data::from([999999., 999998., 999997.]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &Default::default());

        let val_shared =
            Tensor::<TestBackend, 1, Int>::from_primitive(argmin::<TestRuntime, f32, i32, 1>(
                tensor.into_primitive(),
                0,
                ReduceStrategy::SharedMemory,
            ));

        assert_eq!(2, val_shared.into_data().value[0]);
    }

    #[test]
    fn reduction_argmin_shared_memory_extreme_values_i32() {
        let data: Data<i32, 1> = Data::from([999999, 999998, 999997]);
        let tensor = Tensor::<TestBackend, 1, Int>::from_data(data, &Default::default());

        let val_shared =
            Tensor::<TestBackend, 1, Int>::from_primitive(argmin::<TestRuntime, i32, i32, 1>(
                tensor.into_primitive(),
                0,
                ReduceStrategy::SharedMemory,
            ));

        assert_eq!(2, val_shared.into_data().value[0]);
    }

    #[test]
    fn reduction_argmax_shared_memory_extreme_values_i32() {
        let data: Data<i32, 1> = Data::from([-999999, -999997, -999998]);
        let tensor = Tensor::<TestBackend, 1, Int>::from_data(data, &Default::default());

        let val_shared =
            Tensor::<TestBackend, 1, Int>::from_primitive(argmax::<TestRuntime, i32, i32, 1>(
                tensor.into_primitive(),
                0,
                ReduceStrategy::SharedMemory,
            ));

        assert_eq!(1, val_shared.into_data().value[0]);
    }
}
