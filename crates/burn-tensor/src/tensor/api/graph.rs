use crate::Device;
use burn_backend::{Backend, BackendGraph};
use burn_dispatch::Dispatch;

/// A captured computation.
///
/// [`capture`] records a closure once; [`replay`](Graph::replay) then re-runs it
/// as a single dispatch on backends with hardware graph support (CUDA/HIP —
/// collapsing hundreds of kernel launches into one), or simply re-executes the
/// closure everywhere else. The API is identical either way; only the speed
/// differs.
///
/// The recorded graph replays against the exact device buffers used during
/// capture, so the closure must read its inputs from, and write its outputs to,
/// **stable** buffers held across the loop. Refresh those inputs in place with
/// [`Tensor::inplace`](crate::Tensor::inplace)-style writes before each replay.
pub struct Graph<T, F> {
    device: Device,
    output: T,
    closure: F,
    /// The captured hardware graph, or `None` to re-run the closure (a backend
    /// without graph support, or a capture that failed).
    hardware: Option<BackendGraph>,
}

/// Capture `closure` for repeated replay (see [`Graph`]).
///
/// Runs `closure` twice: once to warm up (trigger autotuning and allocate every
/// buffer, so the capture itself needs no fresh device allocation — which is
/// illegal mid-capture), then once under capture. On a backend without graph
/// support the warmup/capture collapses to a single run and replay just re-runs
/// the closure.
pub fn capture<T, F>(device: &Device, mut closure: F) -> Graph<T, F>
where
    F: FnMut() -> T,
{
    let dispatch = device.as_dispatch();
    // Prepare the allocator for capture, then warm up: this triggers all
    // autotuning and allocates every buffer the capture will reuse.
    //
    // Several warmup iterations, not one: with fusion the first run builds and
    // autotunes the fused optimization (a different execution path, allocating
    // different scratch/metadata buffers than steady state). Only from the
    // second run on does the closure take the cached fused path whose buffers
    // the capture must reuse. Warming up a few times lets those steady-state
    // allocations populate the persistent pool before capture opens — otherwise
    // the capture run allocates them fresh, which faults mid-capture.
    const WARMUP_ITERS: usize = 3;
    let _ = Dispatch::graph_prepare(dispatch);
    for _ in 0..WARMUP_ITERS {
        // Hold the output alive across the sync. A lazy backend (fusion) elides
        // a computation whose result is dropped before it drains — so dropping
        // the warmup output would skip the very kernels (and allocations) the
        // capture must reuse, and the capture run would then be the first to
        // allocate them, faulting mid-capture. Keeping `out` until after `sync`
        // forces the closure to actually execute and populate the pool.
        let out = closure();
        let _ = Dispatch::sync(dispatch);
        drop(out);
    }

    let (hardware, output) = match Dispatch::graph_start_capture(dispatch) {
        // Record the closure's launches into a graph.
        Ok(()) => {
            let output = closure();
            (Dispatch::graph_stop_capture(dispatch).ok(), output)
        }
        // No hardware graph support: fall back to re-running the closure.
        Err(_) => (None, closure()),
    };

    Graph {
        device: device.clone(),
        output,
        closure,
        hardware,
    }
}

impl<T, F> Graph<T, F>
where
    F: FnMut() -> T,
{
    /// Re-run the captured computation and return its output.
    ///
    /// On a captured graph this is one dispatch replaying the recorded launches
    /// against their original buffers — so write fresh inputs into those
    /// buffers first. On the fallback path it re-executes the closure. The
    /// returned reference is the output produced during capture (whose buffer
    /// the replay just overwrote), or the fresh output on the fallback path.
    pub fn replay(&mut self) -> &T {
        match &self.hardware {
            Some(graph) => {
                Dispatch::graph_replay(self.device.as_dispatch(), graph)
                    .expect("graph replay should succeed");
            }
            None => {
                self.output = (self.closure)();
            }
        }
        &self.output
    }

    /// The output tensor(s) the graph writes to — stable across replays on the
    /// hardware path (the same buffer is overwritten each time).
    pub fn output(&self) -> &T {
        &self.output
    }

    /// Whether this graph replays as a hardware dispatch (`true`) or by
    /// re-running the closure (`false`).
    pub fn is_hardware(&self) -> bool {
        self.hardware.is_some()
    }
}
