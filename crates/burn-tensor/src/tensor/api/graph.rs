use crate::{Device, ExecutionError};
use burn_backend::{Backend, BackendGraph};
use burn_dispatch::{Dispatch, DispatchDevice};

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
///
/// # Safety
///
/// The graph has no explicit input/state/output signature: whatever buffers the
/// closure touched during capture are what every replay reads and writes, with
/// nothing tracking them afterwards. [`replay`](Graph::replay) is therefore
/// `unsafe` — see its safety contract. Safe, structured APIs can be layered on
/// top of this mechanism for specific workloads (e.g. a decode step with pinned
/// KV-cache and token buffers).
pub struct Graph<T, F> {
    device: Device,
    output: T,
    closure: F,
    /// The captured hardware graph, or why replay re-runs the closure instead
    /// (a backend without graph support, or a capture that failed).
    hardware: Result<BackendGraph<Dispatch>, ExecutionError>,
}

/// Capture `closure` for repeated replay (see [`Graph`]).
///
/// Runs `closure` several times. First as rehearsals, whose launches do everything short of
/// running: kernels compile, tunes measure, and the memory grows to what the capture needs, since
/// allocating mid-capture is illegal. The rehearsals repeat until two runs in a row allocate
/// alike, since a first run can take another path than the steady state: with fusion it builds
/// and tunes the fused optimization, which allocates other buffers than the cached path does.
/// Then once more, recorded. What the last rehearsal left allocated is the closure's output, and
/// the recording places it on memory no other graph of the device writes, so it outlives every
/// replay but its own graph's.
///
/// The closure must therefore allocate the same buffers, and keep the same ones, on every run,
/// and must not read a tensor's data: a rehearsal's launches never ran. On a backend without
/// graph support the rehearsal is one ordinary run, and replay re-runs the closure.
pub fn capture<T, F>(device: &Device, mut closure: F) -> Graph<T, F>
where
    F: FnMut() -> T,
{
    let dispatch = device.as_dispatch();
    let hardware = rehearse(dispatch, &mut closure);
    let (hardware, output) = match hardware.and_then(|()| Dispatch::graph_start_capture(dispatch)) {
        // Record the closure's launches into a graph.
        Ok(()) => {
            let output = closure();
            // A failed stop still resets the stream out of capture mode (the
            // backend guarantees this even on error), so re-running the closure
            // on replay stays correct; we lose the graph, not stream health.
            (Dispatch::graph_stop_capture(dispatch), output)
        }
        // No hardware graph support, or a rehearsal that failed: fall back to re-running the
        // closure.
        Err(error) => (Err(error), closure()),
    };

    Graph {
        device: device.clone(),
        output,
        closure,
        hardware,
    }
}

/// Rehearse `closure` on `device` until two runs allocate alike, leaving the capture prepared
/// for its recording. A rehearsal that fails ends the capture, so the device may capture again.
fn rehearse<T>(
    device: &DispatchDevice,
    closure: &mut impl FnMut() -> T,
) -> Result<(), ExecutionError> {
    /// Runs rehearsed before recording whatever the last one found: a cold fused path settles
    /// by its second run, and a closure that never settles is refused when the capture stops.
    const REHEARSALS_MAX: usize = 4;

    Dispatch::graph_rehearse(device)?;
    for _ in 0..REHEARSALS_MAX {
        // The output is held across the end of the run: what the run still holds when it ends
        // is what the capture keeps off the memory other graphs share.
        let output = closure();
        let settled = Dispatch::graph_rehearsed(device);
        drop(output);
        match settled {
            Ok(true) => return Ok(()),
            Ok(false) => continue,
            Err(error) => {
                let _ = Dispatch::graph_stop_capture(device);
                return Err(error);
            }
        }
    }
    Ok(())
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
    ///
    /// # Safety
    ///
    /// On the hardware path this dispatches the recorded kernels against the raw
    /// device buffers the closure touched during capture, with nothing checking
    /// they are still valid. The caller must guarantee, until the replay's work
    /// completes (e.g. it is followed by a read of the output or a device sync):
    ///
    /// - **Liveness** — every tensor the captured closure read or wrote still
    ///   exists. Dropping one frees its buffer for reuse by other allocations,
    ///   and a later replay would read or overwrite whatever now lives there.
    ///   Tensors owned by the closure (or by `self`, like the output) are kept
    ///   alive automatically; tensors the closure only borrowed must outlive
    ///   the replays.
    /// - **No concurrent use** — no other stream or thread reads or writes a
    ///   tensor shared with the graph while the replay executes; the replay is
    ///   only ordered against work on its own capture stream.
    /// - **Same-stream refreshes** — input refreshes and output reads are
    ///   issued on the stream the graph was captured on (the same device
    ///   thread/client), so they order correctly against the replay rather
    ///   than racing it with stale or torn data.
    ///
    /// On the fallback path (no hardware graph) this simply re-runs the closure
    /// and is trivially safe.
    pub unsafe fn replay(&mut self) -> &T {
        match &self.hardware {
            Ok(graph) => {
                // Safety: forwarded verbatim from this method's own contract.
                unsafe { Dispatch::graph_replay(self.device.as_dispatch(), graph) }
                    .expect("graph replay should succeed");
            }
            Err(_) => {
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
        self.hardware.is_ok()
    }

    /// Why this graph replays by re-running the closure: the backend's error
    /// from starting or stopping the capture, which on a backend without graph
    /// support is its "not supported" error. `None` if it replays as a hardware
    /// dispatch.
    pub fn capture_error(&self) -> Option<&ExecutionError> {
        self.hardware.as_ref().err()
    }
}
