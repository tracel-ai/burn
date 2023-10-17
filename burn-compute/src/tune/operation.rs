use core::fmt::Debug;

use crate::server::{ComputeServer, Handle};

/// Type of operation for the kernel
pub trait AutotuneKernel<S>: Debug + Send
where
    S: ComputeServer,
{
    fn autotune_key(&self) -> String;
    fn autotune_kernels(&self) -> Vec<S::Kernel>;
    fn autotune_handles(&self) -> &[&Handle<S>];
    fn fastest_kernel(&self, fastest_kernel_index: usize) -> S::Kernel;
}
