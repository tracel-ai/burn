#[burn_tensor_testgen::testgen(scatter)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Distribution, Int, Tensor};

    #[test]
    fn scatter_should_work_with_multiple_workgroups_2d_dim0() {
        same_as_reference_same_shape(0, [256, 32]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_2d_dim1() {
        same_as_reference_same_shape(1, [32, 256]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim0() {
        same_as_reference_same_shape(0, [256, 6, 6]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim1() {
        same_as_reference_same_shape(1, [6, 256, 6]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim2() {
        same_as_reference_same_shape(2, [6, 6, 256]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_diff_shapes() {
        same_as_reference_diff_shape(1, [32, 128], [32, 1]);
    }

    fn same_as_reference_diff_shape<const D: usize>(
        dim: usize,
        shape1: [usize; D],
        shape2: [usize; D],
    ) {
        TestBackend::seed(0);
        let test_device = Default::default();
        let tensor = Tensor::<TestBackend, D>::random(shape1, Distribution::Default, &test_device);
        let value = Tensor::<TestBackend, D>::random(shape2, Distribution::Default, &test_device);
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random(
                [shape2.iter().product()],
                Distribution::Uniform(0., shape2[dim] as f64),
                &test_device,
            )
            .into_data(),
            &test_device,
        )
        .reshape(shape2);
        let ref_device = Default::default();
        let tensor_ref = Tensor::<ReferenceBackend, D>::from_data(tensor.to_data(), &ref_device);
        let value_ref = Tensor::<ReferenceBackend, D>::from_data(value.to_data(), &ref_device);
        let indices_ref =
            Tensor::<ReferenceBackend, D, Int>::from_data(indices.to_data(), &ref_device);

        let actual = tensor.scatter(dim, indices, value);
        let expected = tensor_ref.scatter(dim, indices_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    fn same_as_reference_same_shape<const D: usize>(dim: usize, shape: [usize; D]) {
        same_as_reference_diff_shape(dim, shape, shape);
    }
}
