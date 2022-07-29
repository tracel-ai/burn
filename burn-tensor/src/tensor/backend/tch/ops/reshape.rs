use rand::distributions::{uniform::SampleUniform, Standard};

use crate::tensor::{
    backend::tch::{TchShape, TchTensor, TchTensorCPUBackend},
    ops::*,
    Element, Shape, Tensor,
};

impl<P, const D1: usize> TensorOpsReshape<P, D1, TchTensorCPUBackend<P>> for TchTensor<P, D1>
where
    Standard: rand::distributions::Distribution<P>,
    P: Element + tch::kind::Element + Into<f64> + SampleUniform,
{
    fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, TchTensorCPUBackend<P>> {
        let shape_tch: TchShape<D2> = shape.clone().into();
        let tensor = self.tensor.reshape(&shape_tch.dims);
        let kind = self.kind.clone();

        TchTensor {
            tensor,
            kind,
            shape,
        }
    }
}
