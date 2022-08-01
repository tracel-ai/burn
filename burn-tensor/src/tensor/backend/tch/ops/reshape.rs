use crate::tensor::{
    backend::tch::{TchBackend, TchShape, TchTensor},
    ops::*,
    Element, Shape,
};
use rand::distributions::Standard;

impl<P: Element, const D: usize> TensorOpsReshape<TchBackend<P>, D> for TchTensor<P, D>
where
    Standard: rand::distributions::Distribution<P>,
{
    fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> TchTensor<P, D2> {
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
