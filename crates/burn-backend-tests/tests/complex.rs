use burn_tensor::{Float, Tensor};
pub type FloatTensor<const D: usize> = Tensor<D, Float>;
#[allow(unused)]
macro_rules! gen_tests {
    ($variant:ident, $($ty:ty),*) => {
        $(
            paste::paste! {
                mod [<$variant:snake>] {

                    mod basic {
                        type TestTensor<const D: usize> = $ty;
                        include!("complex/basic.rs");
                    }
                    mod numeric {
                        type TestTensor<const D: usize> = $ty;
                        include!("complex/numeric.rs");
                    }
                    mod ops {
                        type TestTensor<const D: usize> = $ty;
                        include!("complex/ops.rs");
                    }
                }
            }
        )*
    };
}

//gen_tests!(split, burn_tensor::SplitTensor<D,burn_tensor::Complex> );
#[cfg(feature = "flex")]
gen_tests!(interleaved, burn_tensor::Tensor<D,burn_tensor::Complex>);
