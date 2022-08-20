use crate::tensor::{
    backend::tch::{TchBackend, TchShape, TchTensor},
    ops::*,
    Shape, TchElement,
};
use rand::distributions::Standard;

impl<P: TchElement, const D: usize> TensorOpsReshape<TchBackend<P>, D> for TchTensor<P, D>
where
    Standard: rand::distributions::Distribution<P>,
{
    fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> TchTensor<P, D2> {
        println!("Reshape to {:?} from {:?}", shape, self.shape);
        println!("{:?}", self.to_data());
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
