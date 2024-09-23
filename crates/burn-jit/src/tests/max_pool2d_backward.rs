#[burn_tensor_testgen::testgen(max_pool2d_backward)]
mod tests {
    use super::*;
    use burn_tensor::{module, ops::ModuleOps, Distribution, Tensor, TensorPrimitive};

    #[test]
    pub fn max_pool2d_with_indices_backward_should_match_reference_backend() {
        let test_device = Default::default();
        let tensor =
            Tensor::<TestBackend, 4>::random([32, 32, 32, 32], Distribution::Default, &test_device);
        let grad_output =
            Tensor::<TestBackend, 4>::random([32, 32, 16, 16], Distribution::Default, &test_device);
        let ref_device = Default::default();
        let tensor_ref = Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &ref_device);
        let grad_output_ref =
            Tensor::<ReferenceBackend, 4>::from_data(grad_output.to_data(), &ref_device);
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding = [1, 1];
        let dilation = [1, 1];

        let (_, indices) =
            module::max_pool2d_with_indices(tensor.clone(), kernel_size, stride, padding, dilation);
        let (_, indices_ref) = module::max_pool2d_with_indices(
            tensor_ref.clone(),
            kernel_size,
            stride,
            padding,
            dilation,
        );
        let grad = TestBackend::max_pool2d_with_indices_backward(
            tensor.into_primitive().tensor(),
            kernel_size,
            stride,
            padding,
            dilation,
            grad_output.into_primitive().tensor(),
            indices.into_primitive(),
        )
        .x_grad;
        let grad_ref = ReferenceBackend::max_pool2d_with_indices_backward(
            tensor_ref.into_primitive().tensor(),
            kernel_size,
            stride,
            padding,
            dilation,
            grad_output_ref.into_primitive().tensor(),
            indices_ref.into_primitive(),
        )
        .x_grad;

        Tensor::<TestBackend, 4>::from_primitive(TensorPrimitive::Float(grad))
            .into_data()
            .assert_approx_eq(
                &Tensor::<ReferenceBackend, 4>::from_primitive(TensorPrimitive::Float(grad_ref))
                    .into_data(),
                3,
            );
    }
}
