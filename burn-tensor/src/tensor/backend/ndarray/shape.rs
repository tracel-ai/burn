use crate::Shape;
use ndarray::Dim;

macro_rules! define_convertion {
    (
        $n:expr
    ) => {
        impl Into<Dim<[usize; $n]>> for Shape<$n> {
            fn into(self) -> Dim<[usize; $n]> {
                Dim(self.dims)
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
