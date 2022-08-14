use crate::graph::node::{ForwardNode, ForwardNodeState};
use crate::graph::ops::ForwardUnaryRecordedOps;
use crate::{back::Backend, tensor::ops::*};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::ADTensor,
};
use rand::distributions::Standard;
use std::sync::Arc;

#[derive(Debug)]
struct ADTensorOpsMean<B: Backend, const D1: usize> {
    _b: B,
}

impl<B: Backend, const D1: usize> ADTensorOpsMean<B, D1> {
    pub fn new() -> Self {
        Self { _b: B::default() }
    }
}

impl<B: Backend, const D1: usize> UnaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<1>>
    for ADTensorOpsMean<B, D1>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D1>, B::TensorPrimitive<1>>,
    ) -> B::TensorPrimitive<D1> {
        println!("HERE");
        println!("{:?}", state.input.value());
        state.input.value().ones()
    }
}

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty,
        $element:ident
    ) => {
        impl<E: $element, const D: usize> TensorOpsAggregation<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            fn mean(&self) -> <$backend as Backend>::TensorPrimitive<1> {
                let input = self.tensor();
                let out = TensorOpsAggregation::mean(&input);
                let shape = out.shape.clone();

                let state = ForwardNodeState::new(out);

                let ops = ADTensorOpsMean::<$backend_inner, D>::new();
                let ops = Arc::new(ops);
                let ops = ForwardUnaryRecordedOps::new(self.node.clone(), ops);
                let ops = Arc::new(ops);

                let node = ForwardNode::from_unary(&self.node, state, ops);
                let node = Arc::new(node);

                ADTensor { node, shape }
            }

            fn sum(&self) -> <$backend as Backend>::TensorPrimitive<1> {
                todo!()
            }

            fn mean_dim<const D2: usize>(
                &self,
                dim: usize,
            ) -> <$backend as Backend>::TensorPrimitive<D2> {
                todo!()
            }

            fn sum_dim<const D2: usize>(
                &self,
                dim: usize,
            ) -> <$backend as Backend>::TensorPrimitive<D2> {
                todo!()
            }

            fn mean_dim_keepdim(&self, dim: usize) -> <$backend as Backend>::TensorPrimitive<D> {
                todo!()
            }

            fn sum_dim_keepdim(&self, dim: usize) -> <$backend as Backend>::TensorPrimitive<D> {
                todo!()
            }
        }
    };
}

crate::register_tch!();
crate::register_ndarray!();

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_mean() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_2.mul(&tensor_1.mean().unsqueeze());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[54.5991, 27.4746], [54.5991, 27.4746]]), 3);
        grad_2.to_data().assert_approx_eq(
            &Data::from([[-5.4598e+01, -9.1188e-04], [2.9556e+01, 8.0342e+01]]),
            3,
        );
    }
}
