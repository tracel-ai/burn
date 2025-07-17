type FloatType = f32;

#[cfg(feature = "int-i32")]
type IntType = i32;

#[cfg(not(feature = "int-i32"))]
type IntType = i64; // default int type

#[cfg(feature = "bool-u32")]
type BoolType = u32;

#[cfg(feature = "backend-autodiff-wgpu")]
pub type Backend = burn_autodiff::Autodiff<burn_wgpu::Wgpu<FloatType, IntType, BoolType>>;

#[cfg(all(feature = "bool-u32", not(feature = "backend-autodiff-wgpu")))]
compile_error!(
    "`bool-u32` has no effect when using the `NdArray` backend, which always uses native `bool`."
);
#[cfg(not(feature = "backend-autodiff-wgpu"))]
pub type Backend = burn_ndarray::NdArray<FloatType, IntType>;
