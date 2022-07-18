use crate::{
    backend::tch::{TchShape, TchTensor},
    Shape, TensorOpsReshape,
};

impl<
        P: tch::kind::Element + std::fmt::Debug + Copy + Default,
        const D1: usize,
        const D2: usize,
    > TensorOpsReshape<P, D1, D2, TchTensor<P, D2>> for TchTensor<P, D1>
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
