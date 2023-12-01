use crate::{
    fusion::codegen::{Elem, Operator},
    fusion::kernel::FusionKernel,
    FloatElement, GraphicsApi, IntElement, Wgpu,
};
use burn_fusion::{graph::Context, Optimization, TensorDescription};
use burn_tensor::Device;

#[derive(Clone)]
pub(crate) struct FloatElementWise<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    pub(crate) inputs: Vec<(TensorDescription, Elem)>,
    pub(crate) outputs: Vec<(TensorDescription, Elem)>,
    pub(crate) locals: Vec<u16>,
    pub(crate) operators: Vec<Operator>,
    pub(crate) scalars_f32: usize,
    pub(crate) device: Device<Wgpu<G, F, I>>,
}

impl<G, F, I> Optimization<Wgpu<G, F, I>> for FloatElementWise<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    fn execute(&self, context: &mut Context<'_, Wgpu<G, F, I>>) {
        let inputs = self
            .inputs
            .iter()
            .map(|(tensor, elem)| (context.tensors.get(&tensor.id).unwrap(), *elem))
            .collect::<Vec<_>>();

        let outputs = self
            .outputs
            .iter()
            .map(|(tensor, elem)| (context.tensors.get(&tensor.id).unwrap(), *elem))
            .collect::<Vec<_>>();

        // The context may contain scalars for the end condition, which may vary.
        let scalars_f32 = &context.scalar_floats[0..self.scalars_f32];

        FusionKernel::new(&self.device)
            .inputs(&inputs, scalars_f32)
            .body(&self.operators)
            .outputs(&outputs, &self.locals)
            .execute(context.handles);
    }

    fn len(&self) -> usize {
        self.operators.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_fusion::graph::Ops;
    use burn_fusion::{Fusion, FusionBackend};
    use burn_tensor::{backend::Backend, Data, Tensor};

    #[test]
    fn test_fusion_same_behavior() {
        type Backend = Wgpu;
        type FusedBackend = Fusion<Wgpu>;

        let data_1 = Tensor::<FusedBackend, 2>::random([1, 32], burn_tensor::Distribution::Default)
            .into_data();
        let data_2 =
            Tensor::<Backend, 2>::random([32, 32], burn_tensor::Distribution::Default).into_data();

        let result_ref = execute::<Backend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant1,
        );
        let result_fused = execute::<FusedBackend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant1,
        );

        result_ref.assert_approx_eq(&result_fused, 3);
    }

    #[test]
    fn test_fusion_same_behavior_different_variant() {
        type Backend = Wgpu;
        type FusedBackend = Fusion<Wgpu>;

        let data_1 = Tensor::<FusedBackend, 2>::random([1, 32], burn_tensor::Distribution::Default)
            .into_data();
        let data_2 =
            Tensor::<Backend, 2>::random([32, 32], burn_tensor::Distribution::Default).into_data();

        let result_ref = execute::<Backend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant2,
        );
        let result_fused_variant1 = execute::<FusedBackend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant1,
        );
        let result_fused_variant2 = execute::<FusedBackend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant2,
        );

        result_ref.assert_approx_eq(&result_fused_variant1, 3);
        result_ref.assert_approx_eq(&result_fused_variant2, 3);
    }

    #[test]
    fn test_end_condition_scalar_ops() {
        type Backend = Fusion<Wgpu>;
        let tensor1 = Tensor::<Backend, 2>::ones([32, 32]);
        let tensor2 = Tensor::<Backend, 2>::ones([32, 42]);
        let output = tensor1.exp().log();

        // This will add a scalar to the context, even if the actual operation can't be fused with
        // the preceding ones because of the shape difference.
        let _ = tensor2 + 2;

        // When we try to execute the operations, the number of bindings can be different if we are
        // not careful.
        Backend::sync(&output.device());
    }

    struct FakeAddOps;

    impl<B: FusionBackend> Ops<B> for FakeAddOps {
        fn execute(self: Box<Self>, _: &mut burn_fusion::HandleContainer<B>) {
            panic!("Should always fused during tests.")
        }
    }

    enum ImplementationDetails {
        Variant1,
        Variant2,
    }
    fn execute<B: Backend>(
        data_1: Data<f32, 2>,
        data_2: Data<f32, 2>,
        variant: ImplementationDetails,
    ) -> Data<f32, 2> {
        let tensor_1 = Tensor::<B, 2>::from_data(data_1.convert());
        let tensor_2 = Tensor::<B, 2>::from_data(data_2.convert());
        let tensor_3 = tensor_1.clone() + tensor_2;
        let tensor_4 = tensor_3.clone() - tensor_1;
        let mut tensor_5 = tensor_4.clone() + 5.0;
        match variant {
            ImplementationDetails::Variant1 => {}
            ImplementationDetails::Variant2 => {
                tensor_5 = tensor_5 + 1;
                tensor_5 = tensor_5 - 1;
            }
        }
        let tensor_6 = burn_tensor::activation::gelu(tensor_5 + tensor_3.clone());
        let mask = tensor_4.lower_equal(tensor_3);
        let tmp = tensor_6.mask_fill(mask, 0.3);

        tmp.into_data().convert()
    }
}
