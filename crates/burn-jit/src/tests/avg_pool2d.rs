#[burn_tensor_testgen::testgen(avg_pool2d)]
mod tests {
    use super::*;
    use burn_tensor::{
        backend::Backend, module, ops::ModuleOps, Distribution, Tensor, TensorPrimitive,
    };

    #[test]
    fn avg_pool2d_should_match_reference_backend() {
        let tensor = Tensor::<TestBackend, 4>::random(
            [32, 32, 32, 32],
            Distribution::Default,
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &Default::default());
        let kernel_size = [3, 4];
        let stride = [1, 2];
        let padding = [1, 2];
        let count_include_pad = true;

        let pooled = module::avg_pool2d(tensor, kernel_size, stride, padding, count_include_pad);
        let pooled_ref =
            module::avg_pool2d(tensor_ref, kernel_size, stride, padding, count_include_pad);

        pooled
            .into_data()
            .assert_approx_eq(&pooled_ref.into_data(), 3);
    }

    #[test]
    fn avg_pool2d_backward_should_match_reference_backend() {
        TestBackend::seed(0);
        ReferenceBackend::seed(0);
        let tensor = Tensor::<TestBackend, 4>::random(
            [32, 32, 32, 32],
            Distribution::Default,
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &Default::default());
        let kernel_size = [3, 3];
        let stride = [1, 1];
        let padding = [1, 1];
        let count_include_pad = true;

        let shape_out = module::avg_pool2d(
            tensor.clone(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
        )
        .shape();
        let grad_output =
            Tensor::<TestBackend, 4>::random(shape_out, Distribution::Default, &Default::default());
        let grad_output_ref =
            Tensor::<ReferenceBackend, 4>::from_data(grad_output.to_data(), &Default::default());

        let grad: Tensor<TestBackend, 4> =
            Tensor::from_primitive(TensorPrimitive::Float(TestBackend::avg_pool2d_backward(
                tensor.into_primitive().tensor(),
                grad_output.into_primitive().tensor(),
                kernel_size,
                stride,
                padding,
                count_include_pad,
            )));
        let grad_ref: Tensor<ReferenceBackend, 4> = Tensor::from_primitive(TensorPrimitive::Float(
            ReferenceBackend::avg_pool2d_backward(
                tensor_ref.into_primitive().tensor(),
                grad_output_ref.into_primitive().tensor(),
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
        ));

        grad.into_data().assert_approx_eq(&grad_ref.into_data(), 3);
    }
}
