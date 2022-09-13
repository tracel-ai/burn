use crate::tensor::Shape;
use ndarray::Dim;

macro_rules! define_convertion {
    (
        $n:expr
    ) => {
        impl From<Shape<$n>> for Dim<[usize; $n]> {
            fn from(shape: Shape<$n>) -> Dim<[usize; $n]> {
                Dim(shape.dims)
            }
        }
    };
}

define_convertion!(0);
define_convertion!(1);
define_convertion!(2);
define_convertion!(3);
define_convertion!(4);
define_convertion!(5);
define_convertion!(6);
