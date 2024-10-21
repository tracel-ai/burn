#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

use burn_jit::JitBackend;
pub use cubecl::hip::HipDevice;
use cubecl::hip::HipRuntime;

#[cfg(not(feature = "fusion"))]
pub type Hip<F = f32, I = i32> = JitBackend<HipRuntime, F, I>;

#[cfg(feature = "fusion")]
pub type Hip<F = f32, I = i32> = burn_fusion::Fusion<JitBackend<HipRuntime, F, I>>;

#[cfg(test)]
mod tests {
    use burn_jit::JitBackend;

    pub type TestRuntime = cubecl::hip::HipRuntime;

    burn_jit::testgen_all!();
}
