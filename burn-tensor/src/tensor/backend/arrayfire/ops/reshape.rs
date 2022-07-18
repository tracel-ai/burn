use crate::{backend::arrayfire::ArrayfireTensor, Shape, TensorOpsReshape};
use arrayfire::HasAfEnum;

impl<P: HasAfEnum + std::fmt::Debug + Copy + Default, const D1: usize, const D2: usize>
    TensorOpsReshape<P, D1, D2, ArrayfireTensor<P, D2>> for ArrayfireTensor<P, D1>
{
    fn reshape(&self, shape: Shape<D2>) -> ArrayfireTensor<P, D2> {
        self.set_backend_single_ops();

        let array = arrayfire::moddims(&self.array, shape.clone().into());
        let device = self.device;

        ArrayfireTensor {
            array,
            shape,
            device,
        }
    }
}
