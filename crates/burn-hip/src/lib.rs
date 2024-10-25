#![cfg_attr(docsrs, feature(doc_auto_cfg))]
extern crate alloc;

#[cfg(target_os = "linux")]
use burn_jit::JitBackend;

#[cfg(target_os = "linux")]
pub use cubecl::hip::HipDevice;

#[cfg(target_os = "linux")]
use cubecl::hip::HipRuntime;

#[cfg(target_os = "linux")]
#[cfg(not(feature = "fusion"))]
pub type Hip<F = f32, I = i32> = JitBackend<HipRuntime, F, I>;

#[cfg(target_os = "linux")]
#[cfg(feature = "fusion")]
pub type Hip<F = f32, I = i32> = burn_fusion::Fusion<JitBackend<HipRuntime, F, I>>;

#[cfg(target_os = "linux")]
#[cfg(test)]
mod tests {
    use burn_jit::JitBackend;

    pub type TestRuntime = cubecl::hip::HipRuntime;

    burn_jit::testgen_all!();
}
