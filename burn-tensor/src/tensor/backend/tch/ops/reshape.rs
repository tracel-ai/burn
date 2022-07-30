use crate::tensor::{
    backend::tch::{TchShape, TchTensor},
    ops::*,
    Element, Shape,
};
use rand::distributions::uniform::SampleUniform;

impl<P, const D1: usize, const D2: usize> TensorOpsReshape<P, D1, D2, TchTensor<P, D2>>
    for TchTensor<P, D1>
where
    P: Element + tch::kind::Element + Into<f64> + SampleUniform,
{
    fn reshape(&self, shape: Shape<D2>) -> TchTensor<P, D2> {
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
