#[burn_tensor_testgen::testgen(reduce)]
mod reduce {
    use super::*;
    use burn_cubecl::kernel::reduce::{
        reduce, reduce_dim, ArgMax, ArgMin, Mean, Prod, ReduceStrategy, Sum,
    };
    use burn_tensor::{
        backend::Backend, ops::IntTensorOps, Distribution, Int, Shape, Tensor, TensorData,
        TensorPrimitive,
    };

    const RANK: usize = 4;
    const SHAPE: [usize; RANK] = [2, 4, 8, 16];

    #[test]
    fn reduction_argmax_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        for dim in 0..RANK {
            tensor
                .clone()
                .argmax(dim)
                .into_data()
                .assert_eq(&tensor_ref.clone().argmax(dim).into_data(), false);
        }
    }

    #[test]
    fn reduction_argmin_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        for dim in 0..RANK {
            tensor
                .clone()
                .argmin(dim)
                .into_data()
                .assert_eq(&tensor_ref.clone().argmin(dim).into_data(), false);
        }
    }

    #[test]
    fn reduction_mean_dim_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        for dim in 0..RANK {
            tensor
                .clone()
                .mean_dim(dim)
                .into_data()
                .assert_approx_eq_diff(&tensor_ref.clone().mean_dim(dim).into_data(), 1e-6);
        }
    }

    #[test]
    fn reduction_mean_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        tensor
            .clone()
            .mean()
            .into_data()
            .assert_approx_eq_diff(&tensor_ref.clone().mean().into_data(), 1e-6);
    }

    #[test]
    fn reduction_prod_dim_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        for dim in 0..RANK {
            tensor
                .clone()
                .prod_dim(dim)
                .into_data()
                .assert_approx_eq_diff(&tensor_ref.clone().prod_dim(dim).into_data(), 1e-6);
        }
    }

    #[test]
    fn reduction_prod_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        tensor
            .clone()
            .prod()
            .into_data()
            .assert_approx_eq_diff(&tensor_ref.clone().prod().into_data(), 1e-6);
    }

    #[test]
    fn reduction_sum_dim_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        for dim in 0..RANK {
            tensor
                .clone()
                .sum_dim(dim)
                .into_data()
                .assert_approx_eq_diff(&tensor_ref.clone().sum_dim(dim).into_data(), 1e-6);
        }
    }

    #[test]
    fn reduction_sum_should_match_reference_backend() {
        let tensor =
            Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
        tensor
            .clone()
            .sum()
            .into_data()
            .assert_approx_eq_diff(&tensor_ref.clone().sum().into_data(), 1e-6);
    }
}
