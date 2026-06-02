use burn_tensor::{Complex, Float, SplitTensor, Tensor};
pub type FloatTensor<const D: usize> = Tensor<D, Float>;

macro_rules! gen_tests {
    ($variant:ident, $($ty:ty),*) => {
        $(
            paste::paste! {
                mod [<$variant:snake>] {

                    mod basic {
                        type TestTensor<const D: usize> = $ty;
                        include!("compat/basic.rs");
                    }
                    mod numeric {
                        type TestTensor<const D: usize> = $ty;
                        include!("compat/numeric.rs");
                    }
                    mod ops {
                        type TestTensor<const D: usize> = $ty;
                        include!("compat/ops.rs");
                    }
                }
            }
        )*
    };
}

//gen_tests!(split, burn_tensor::SplitTensor<D,burn_tensor::Complex> );
gen_tests!(interleaved, burn_tensor::Tensor<D,burn_tensor::Complex>);
