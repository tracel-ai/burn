use crate::tensor::{
    backend::tch::{TchBackend, TchShape, TchTensor},
    ops::*,
    Shape, TchElement,
};

impl<P: TchElement, const D: usize> TensorOpsReshape<TchBackend<P>, D> for TchTensor<P, D> {
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
