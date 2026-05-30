use burn_tensor::{Complex, Float, Tensor, split::SplitTensor};
pub type FloatTensor<const D: usize> = Tensor<D, Float>;

macro_rules! gen_tests {
    ($variant:ident, $($ty:ty),*) => {
        $(
            paste::paste! {
                mod [<$variant:snake>] {
                    
                    mod basic {
                        use crate::FloatTensor;
                        type TestTensor<const D: usize> = $ty;
                        include!("compat/basic.rs");
                    }
                    mod numeric {
                        use crate::FloatTensor;
                        type TestTensor<const D: usize> = $ty;
                        include!("compat/numeric.rs");
                    }
                    mod ops {
                        use crate::FloatTensor;
                        type TestTensor<const D: usize> = $ty;
                        include!("compat/ops.rs");
                    }
                }
            }
        )*
    };
}

gen_tests!(split, burn_tensor::split::SplitTensor<D,burn_tensor::Complex> );
gen_tests!(interleaved, burn_tensor::Tensor<D,burn_tensor::Complex>);
