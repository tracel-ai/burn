use crate::tensor::backend::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{ops::*, Element},
};
use rand::distributions::Standard;

register_ops!(
    ops UnaryOps,
    name ADTensorMaskFillOps state B::BoolTensorPrimitive<D>,
    partial |mask: &B::BoolTensorPrimitive<D>, state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        state.output.grad().mask_fill(&mask, B::Elem::zeros(&B::Elem::default()))
    },
);

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty
    ) => {
        impl<E: Element, const D: usize> TensorOpsMask<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            fn mask_fill(
                &self,
                mask: &<$backend as Backend>::BoolTensorPrimitive<D>,
                value: E,
            ) -> Self {
                let node = execute_ops!(
                    input self.node.clone(),
                    out TensorOpsMask::mask_fill(&self.tensor(), &mask, value),
                    ops ADTensorMaskFillOps::<$backend_inner, D>::new(mask.clone()),
                );
                self.from_existing(node)
            }
        }
    };
}

#[cfg(feature = "ndarray")]
define_impl!(
    crate::tensor::backend::autodiff::ADBackendNdArray::<E>,
    crate::tensor::backend::ndarray::NdArrayBackend::<E>
);

#[cfg(feature = "tch")]
define_impl!(
    crate::tensor::backend::autodiff::ADBackendTch::<E>,
    crate::tensor::backend::tch::TchBackend::<E>
);

#[cfg(test)]
mod tests {
    use crate::tensor::BoolTensor;
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_mask() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, 7.0], [2.0, 3.0]]);
        let mask = Data::<bool, 2>::from([[true, false], [false, true]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let mask = BoolTensor::from_data(mask);

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_3.mask_fill(&mask, 2.0);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[7.0, 3.0], [4.0, 2.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 1.0], [3.0, 7.0]]));
    }
}
