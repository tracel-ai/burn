#![cfg_attr(docsrs, feature(doc_auto_cfg))]
extern crate alloc;

#[cfg(target_os = "linux")]
use burn_cubecl::CubeBackend;

#[cfg(target_os = "linux")]
pub use cubecl::hip::HipDevice;

#[cfg(target_os = "linux")]
use cubecl::hip::HipRuntime;

#[cfg(target_os = "linux")]
#[cfg(not(feature = "fusion"))]
pub type Hip<F = f32, I = i32, B = u8> = CubeBackend<HipRuntime, F, I, B>;

#[cfg(target_os = "linux")]
#[cfg(feature = "fusion")]
pub type Hip<F = f32, I = i32, B = u8> = burn_fusion::Fusion<CubeBackend<HipRuntime, F, I, B>>;

// TODO: Hang the computer when AMD isn't available.
//
// #[cfg(target_os = "linux")]
// #[cfg(test)]
// mod tests {
//     use burn_cubecl::CubeBackend;
//
//     pub type TestRuntime = cubecl::hip::HipRuntime;
//     pub use half::f16;
//
//     // TODO: Add tests for bf16
//     burn_cubecl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
// }
