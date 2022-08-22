use crate::graph::node::{ForwardNode, ForwardNodeState};
use crate::graph::ops::ForwardUnaryRecordedOps;
use crate::tensor::backend::backend::Backend;
use crate::tensor::{ops::*, Shape};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::ADTensor,
};
use std::sync::Arc;

#[derive(Debug)]
struct ADTensorOpsReshape<B: Backend, const D1: usize, const D2: usize> {
    shape: Shape<D1>,
    _b: B,
}

impl<B: Backend, const D1: usize, const D2: usize> ADTensorOpsReshape<B, D1, D2> {
    pub fn new(shape: Shape<D1>) -> Self {
        Self {
            shape,
            _b: B::default(),
        }
    }
}

impl<B: Backend, const D1: usize, const D2: usize>
    UnaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D2>> for ADTensorOpsReshape<B, D1, D2>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D1>, B::TensorPrimitive<D2>>,
    ) -> B::TensorPrimitive<D1> {
        let mut grad = state.output.grad();
        let value = state.output.value();

        let shape_grad = grad.shape().clone();
        let shape_value = value.shape().clone();

        if shape_value == shape_grad {
            return grad.reshape(self.shape.clone());
        }

        for i in 0..D2 {
            if shape_value.dims[i] == 1 && shape_grad.dims[i] != 1 {
                grad = grad.sum_dim(i);
            }
        }

        grad.reshape(self.shape.clone())
    }
}

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty,
        $element:ident
    ) => {
        impl<E: $element, const D1: usize> TensorOpsReshape<$backend, D1>
            for <$backend as Backend>::TensorPrimitive<D1>
        {
            fn reshape<const D2: usize>(
                &self,
                shape: Shape<D2>,
            ) -> <$backend as Backend>::TensorPrimitive<D2> {
                let input = self.tensor();
                let out = TensorOpsReshape::reshape(&input, shape.clone());

                let state = ForwardNodeState::new(out);

                let ops = ADTensorOpsReshape::<$backend_inner, D1, D2>::new(self.shape.clone());
                let ops = Arc::new(ops);
                let ops = ForwardUnaryRecordedOps::new(self.node.clone(), ops);
                let ops = Arc::new(ops);

                let node = ForwardNode::from_unary(&self.node, state, ops);
                let node = Arc::new(node);

                let shape = shape.clone();

                ADTensor { node, shape }
            }
        }
    };
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;
    use crate::NdArrayElement;

    define_impl!(
        crate::tensor::backend::autodiff::ADBackendNdArray::<E>,
        crate::tensor::backend::ndarray::NdArrayBackend::<E>,
        NdArrayElement
    );
}

#[cfg(feature = "tch")]
mod tch_impl {
    use super::*;
    use crate::TchElement;

    define_impl!(
        crate::tensor::backend::autodiff::ADBackendTch::<E>,
        crate::tensor::backend::tch::TchBackend::<E>,
        TchElement
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_mul() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 1> = Data::from([4.0, 7.0, 2.0, 3.0]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_2.reshape(Shape::new([2, 2]));
        let tensor_4 = &tensor_1.matmul(&tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([3.0, 3.0, 10.0, 10.0]));
    }
}
