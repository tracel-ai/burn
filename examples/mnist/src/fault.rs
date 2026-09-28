//! Deliberate compute failures, to see what a training or inference loop gets
//! back when the device fails under it.
//!
//! Two kinds, which a caller has to tell apart:
//!
//! - [`Fault::Refused`] — the kernel is rejected before it runs. Its output is
//!   never written, so reading it fails, but the device is fine: dropping the
//!   work and redoing it succeeds.
//! - [`Fault::DevicePoisoned`] — the kernel writes far outside its buffer, and the
//!   device raises an illegal address. On CUDA that poisons the whole context:
//!   every later read fails too, and only a new process recovers.
//!
//! Either way the failure surfaces where the program reads a result back
//! (`try_into_data`, `try_into_scalar`, `Device::sync`), never at the call
//! that injected it.

use burn::{
    backend::{
        AutodiffBackend, Dispatch,
        autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
        backend_extension,
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{Int, Tensor},
};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, prelude::*};

/// Which failure to inject.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// A kernel the compiler refuses: recoverable.
    Refused,
    /// A kernel that faults on the device: the device is poisoned.
    DevicePoisoned,
}

impl core::str::FromStr for Fault {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "refused" => Ok(Self::Refused),
            "poisoned" => Ok(Self::DevicePoisoned),
            other => Err(format!(
                "unknown fault `{other}`, expected `refused` or `poisoned`"
            )),
        }
    }
}

#[backend_extension(Autodiff, Cube, Fusion)]
pub trait FaultOps: burn::backend::Backend {
    #[fusion(dtype = x, shape = x)]
    fn fault_float(x: FloatTensor<Self>, poison_device: bool) -> FloatTensor<Self>;

    #[fusion(dtype = x, shape = x)]
    fn fault_int(x: IntTensor<Self>, poison_device: bool) -> IntTensor<Self>;
}

/// A tensor shaped like `x` whose computation fails with `fault`.
pub fn inject_float<const D: usize>(x: Tensor<D>, fault: Fault) -> Tensor<D> {
    let poison_device = fault == Fault::DevicePoisoned;
    Tensor::from_dispatch(Dispatch::fault_float(x.into_dispatch(), poison_device))
}

/// A tensor shaped like `x` whose computation fails with `fault`.
pub fn inject_int<const D: usize>(x: Tensor<D, Int>, fault: Fault) -> Tensor<D, Int> {
    let poison_device = fault == Fault::DevicePoisoned;
    Tensor::from_dispatch(Dispatch::fault_int(x.into_dispatch(), poison_device))
}

impl FaultOps for CubeBackend {
    fn fault_float(x: FloatTensor<Self>, poison_device: bool) -> FloatTensor<Self> {
        launch_fault(x, poison_device)
    }

    fn fault_int(x: IntTensor<Self>, poison_device: bool) -> IntTensor<Self> {
        launch_fault(x, poison_device)
    }
}

/// A fault has no gradient: the autodiff layer passes the tensor through to
/// the backend underneath, untracked.
impl<B: FaultOps, C: CheckpointStrategy> FaultOps for Autodiff<B, C> {
    fn fault_float(x: FloatTensor<Self>, poison_device: bool) -> FloatTensor<Self> {
        Self::from_inner(B::fault_float(Self::inner(x), poison_device))
    }

    fn fault_int(x: IntTensor<Self>, poison_device: bool) -> IntTensor<Self> {
        Self::int_from_inner(B::fault_int(Self::int_inner(x), poison_device))
    }
}

/// A launch the compiler is guaranteed to refuse, before anything runs.
#[cube(launch_unchecked)]
fn refused(out: &mut [u32], #[comptime] reason: String) {
    push_validation_error(reason);
    out[0] = 1u32;
}

/// Writes a gigabyte past the end of its buffer: nothing is mapped there, so
/// the device raises an illegal address.
#[cube(launch_unchecked)]
fn out_of_bounds(out: &mut [u32]) {
    out[ABSOLUTE_POS + 268_435_456] = 1u32;
}

fn launch_fault(x: CubeTensor, poison_device: bool) -> CubeTensor {
    let bytes = x.meta.shape().num_elements() * x.dtype.size();
    // The kernels address the buffer as words; give them at least one.
    let words = bytes.div_ceil(4).max(1);
    let buffer = x.client.empty(words * 4);
    let output = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        x.meta.shape().clone(),
        buffer.clone(),
        x.dtype,
    );

    let arg = unsafe { BufferArg::from_raw_parts(buffer, words) };
    match poison_device {
        // SAFETY: deliberately not — this is the fault being injected.
        true => unsafe {
            out_of_bounds::launch_unchecked(
                &x.client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                arg,
            )
        },
        // SAFETY: the kernel is refused before it runs, so it touches nothing.
        false => unsafe {
            refused::launch_unchecked(
                &x.client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                arg,
                "injected fault: this kernel is refused on purpose".to_string(),
            )
        },
    }

    output
}

/// A fault the training loop injects at a given step, see [`schedule`].
static SCHEDULE: std::sync::Mutex<Option<(Fault, usize)>> = std::sync::Mutex::new(None);
static STEP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Make the `step`-th call to [`scheduled`] inject `fault`.
pub fn schedule(fault: Fault, step: usize) {
    *SCHEDULE.lock().unwrap() = Some((fault, step));
    STEP.store(0, std::sync::atomic::Ordering::SeqCst);
}

/// `targets` as they are, or failing with the scheduled fault when this is
/// the step it was scheduled for.
pub fn scheduled(targets: Tensor<1, Int>) -> Tensor<1, Int> {
    let step = STEP.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    match *SCHEDULE.lock().unwrap() {
        Some((fault, at)) if at == step => {
            log::warn!("injecting {fault:?} into the targets at step {step}");
            inject_int(targets, fault)
        }
        _ => targets,
    }
}
