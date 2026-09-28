use crate::{CubeDevice, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    Backend, BackendGraph, BackendTypes, DTypeUsage, DTypeUsageSet, ExecutionError,
    MemoryPoolUsage, ProfileDuration, ProfileOptions, ProfileToken, SlicedPoolReport, TensorData,
    profile_with_tokens,
};
use burn_std::{BoolStore, DType, id::StreamId, quantization::quantizable};
use cubecl::device::DeviceId;
use cubecl::{
    MemoryPoolKind, MemoryScope,
    client::{Client, ProfileWindow},
    features::{MmaConfig, TypeUsage},
    ir::ElemType,
    server::{ProfileError, ProfilingToken},
};

#[cfg(not(feature = "fusion"))]
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
#[cfg(not(feature = "fusion"))]
use burn_ir::{BackendIr, TensorHandle};

/// Whether the runtime can hold a quantized dtype's scales, which `dtype_to_storage_type` misses
/// because it doesn't see the scheme's scale levels. Non-quantized dtypes always pass.
fn qfloat_params_usable(client: &Client, dtype: DType) -> bool {
    let DType::QFloat(scheme) = dtype else {
        return true;
    };

    quantizable(&scheme)
        && client
            .properties()
            .type_usage(ElemType::from_scale_dtype(scheme.scale_dtype()))
            .is_superset(TypeUsage::Buffer | TypeUsage::Conversion)
}

/// Turn a cubecl graph-capture error into a backend [`ExecutionError`].
fn graph_err(err: impl core::fmt::Display) -> ExecutionError {
    ExecutionError::WithContext {
        reason: format!("{err}"),
    }
}

/// Turn a cubecl profiling error into a backend [`ExecutionError`].
fn profile_err(err: ProfileError) -> ExecutionError {
    ExecutionError::WithContext {
        reason: format!("{err}"),
    }
}

/// A window a kernel-stamping runtime could not measure.
///
/// A runtime that stamps the stream (CUDA, HIP) answers a window nothing ran
/// in with two stamps and nothing between them. One that stamps kernels (wgpu)
/// has no query set for it and refuses it as [`ProfileError::NotMeasured`] —
/// but it refuses **two** cases with one error, and cubecl says so where the
/// refusal is raised: a window that dispatched nothing, and a window whose
/// work never landed in a timestamped pass. The second is a kernel that ran.
///
/// So this resolves to no measurement rather than to a zero. Zero is the
/// fastest duration there is, and a caller comparing two windows — which is
/// what a profiling scope is for — would take the one that could not be
/// measured as the quicker of the two. `None` is already how every reader of a
/// [`ProfileDuration`] spells an absence, and the error it replaces carried
/// exactly that meaning.
fn empty_window() -> ProfileDuration {
    ProfileDuration::new_device_time_maybe(async move { None })
}

/// A captured launch sequence, tagged with the device it was captured on.
///
/// `cubecl::client::Graph` is self-contained: it owns a handle to the device it recorded on and
/// replays there no matter what device the caller names. Every cubecl runtime is one backend now,
/// so a graph captured on CUDA and replayed with a wgpu device is no longer a variant mismatch the
/// dispatch layer catches — without this tag it would replay, silently, on the wrong device.
#[derive(Clone, Debug)]
pub struct CubeGraph {
    graph: cubecl::client::Graph,
    device: CubeDevice,
}

/// Tensor backend that compiles just-in-time for whichever runtime its device
/// names.
#[derive(new)]
pub struct CubeBackend;

impl BackendTypes for CubeBackend {
    type Device = CubeDevice;

    type FloatTensorPrimitive = CubeTensor;
    type IntTensorPrimitive = CubeTensor;
    type BoolTensorPrimitive = CubeTensor;
    type QuantizedTensorPrimitive = CubeTensor;

    type GraphPrimitive = CubeGraph;
}

impl Backend for CubeBackend {
    fn name(device: &Self::Device) -> String {
        let client = device.client();
        format!("cubecl<{}>", client.name())
    }

    fn seed(_device: &Self::Device, seed: u64) {
        cubek::random::seed(seed);
    }

    fn ad_enabled(_device: &Self::Device) -> bool {
        false
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = device.client();
        // A barrier plus the device's own fault, and nothing more: a launch
        // failure lives on the buffers the launch never wrote and surfaces on
        // the read of one of them, so it is not this sync's to report.
        // `client.sync_buffers` is the same barrier plus a check of named
        // tensors, for a caller that wants both.
        futures_lite::future::block_on(client.sync()).map_err(|err| ExecutionError::WithContext {
            reason: format!("{err}"),
        })
    }

    fn profile<O: Send + 'static>(
        device: &Self::Device,
        options: ProfileOptions,
        func: impl FnOnce() -> O + Send,
    ) -> Result<(O, ProfileDuration), ExecutionError> {
        // Not cubecl's bracketed `profile`: that one runs the closure on the
        // device's runner while holding the device, so a closure waiting on
        // another thread's call to the same device — a data loader building
        // its batch there, say — would never get it back. The split window
        // opens and closes on the stream without holding anything between.
        profile_with_tokens::<Self, O>(device, options, func)
    }

    fn profile_start(device: &Self::Device) -> Result<Option<ProfileToken>, ExecutionError> {
        let client = device.client();
        client
            .profile_start()
            .map(|window| {
                Some(ProfileToken {
                    id: window.token.id,
                    opened_on: window.stream_id.value,
                })
            })
            .map_err(profile_err)
    }

    fn profile_end(
        device: &Self::Device,
        token: ProfileToken,
        _options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        // Nothing is queued past the window here: every launch reaches the
        // stream as it is made, so there is nothing for the flush option to
        // force out.
        let client = device.client();
        // Closed on the stream it was opened on, which the token carries —
        // not on the calling thread's, which need not be the same one.
        let window = ProfileWindow {
            stream_id: StreamId {
                value: token.opened_on,
            },
            token: ProfilingToken { id: token.id },
        };
        match client.profile_end(window) {
            Ok(duration) => Ok(duration),
            Err(ProfileError::NotMeasured { .. }) => Ok(empty_window()),
            Err(err) => Err(profile_err(err)),
        }
    }

    /// Dropped on the stream it was opened on, without recording an end —
    /// cubecl returns the start event to its pool and nothing is measured.
    fn profile_abandon(device: &Self::Device, token: ProfileToken) {
        let window = ProfileWindow {
            stream_id: StreamId {
                value: token.opened_on,
            },
            token: ProfilingToken { id: token.id },
        };
        device.client().profile_abandon(window);
    }

    fn graph_prepare(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = device.client();
        client.graph_prepare().map_err(graph_err)
    }

    fn graph_start_capture(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = device.client();
        client.start_capture().map_err(graph_err)
    }

    fn graph_stop_capture(device: &Self::Device) -> Result<BackendGraph<Self>, ExecutionError> {
        let client = device.client();
        let graph = client.stop_capture().map_err(graph_err)?;

        Ok(CubeGraph {
            graph,
            device: device.clone(),
        })
    }

    unsafe fn graph_replay(
        device: &Self::Device,
        graph: &BackendGraph<Self>,
    ) -> Result<(), ExecutionError> {
        // The replay goes to the device the graph was captured on whatever is passed here, so a
        // caller naming a different one gets told rather than silently served the other device.
        if &graph.device != device {
            return Err(ExecutionError::WithContext {
                reason: format!(
                    "The graph was captured on {:?} and cannot replay on {device:?}",
                    graph.device
                ),
            });
        }

        // cubecl's `Graph::replay` blocks on the enqueue and reports what the
        // enqueue said; a failure also leaves the graph's write set carrying
        // it, so a read of those buffers keeps failing until a replay lands.
        //
        // Safety: the buffer-liveness and stream-ordering obligations are the
        // caller's, forwarded verbatim from this method's own contract.
        unsafe { graph.graph.replay() }.map_err(graph_err)
    }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        let client = device.client();
        client.memory_persistent_allocation(input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        let client = device.client();
        // Refused only while a stream records a graph, which keeps its pages
        // for the replay. The memory is released by the next cleanup instead.
        let _ = client.memory_cleanup();
    }

    fn memory_pool_report(device: &Self::Device) -> Option<Vec<SlicedPoolReport>> {
        let report = device.client().memory_report(MemoryScope::Device);

        // One entry per pool that carves pages — the pool sized to what it
        // serves, the pools a growth left behind, and the metadata pool —
        // in the order allocations are routed through them. Pools whose
        // allocations own their page have no page size to report.
        Some(
            report
                .streams
                .iter()
                .flat_map(|stream| &stream.pools.dynamic)
                .filter_map(|pool| {
                    let page_size = match pool.kind {
                        MemoryPoolKind::Sliced { page_size, .. } => page_size,
                        MemoryPoolKind::Adaptive { page_size, .. } => page_size,
                        _ => return None,
                    };
                    Some(SlicedPoolReport {
                        page_size,
                        pages: pool.pages,
                        pages_peak: pool.pages_peak,
                        largest_alloc: pool.largest_alloc,
                    })
                })
                .collect(),
        )
    }

    fn memory_pool_usage(device: &Self::Device) -> Option<MemoryPoolUsage> {
        let usage = device.client().memory_report(MemoryScope::Device).usage();

        Some(MemoryPoolUsage {
            number_allocs: usage.number_allocs,
            bytes_in_use: usage.bytes_in_use,
            bytes_padding: usage.bytes_padding,
            bytes_reserved: usage.bytes_reserved,
        })
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        let client = device.client();
        TensorData::with_bytes_mut(data, |bytes| client.staging(bytes.into_iter(), false));
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        // Right now no cubecl backend actually works with native bool, even if
        // the `TypeUsage` might indicate otherwise.
        if let DType::Bool(BoolStore::Native) = dtype {
            return false;
        }
        let client = device.client();

        if !qfloat_params_usable(&client, dtype) {
            return false;
        }

        let type_usage = client.properties().type_usage(dtype_to_storage_type(dtype));
        // Same as `TypeUsage::all_scalar()`, but we make the usage explicit here
        type_usage.is_superset(
            TypeUsage::Buffer
                | TypeUsage::Conversion
                | TypeUsage::Arithmetic
                | TypeUsage::DotProduct,
        )
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        // Right now no cubecl backend actually works with native bool, even if
        // the `TypeUsage` might indicate otherwise.
        if let DType::Bool(BoolStore::Native) = dtype {
            return DTypeUsageSet::empty();
        }
        let client = device.client();

        if !qfloat_params_usable(&client, dtype) {
            return DTypeUsageSet::empty();
        }

        let props = client.properties();
        let storage = dtype_to_storage_type(dtype);
        let usage = props.type_usage(storage);

        let mut out = DTypeUsageSet::new();

        if usage.is_superset(TypeUsage::Buffer | TypeUsage::Conversion) {
            out |= DTypeUsage::Storage;
        }

        if usage.contains(TypeUsage::Arithmetic) {
            out |= DTypeUsage::Arithmetic;
        }

        let has_mma = |cfg: &MmaConfig| {
            cfg.a_type == storage || cfg.b_type == storage || cfg.cd_type == storage
        };
        if props.features.matmul.cmma.iter().any(has_mma)
            || props.features.matmul.mma.iter().any(has_mma)
        {
            out |= DTypeUsage::Accelerated;
        }

        out
    }

    fn device_count(type_id: u16) -> usize {
        CubeDevice::enumerate(DeviceId::new(type_id, 0)).len()
    }

    fn flush(device: &Self::Device) {
        let client = device.client();
        client.flush().unwrap();
    }
}

impl core::fmt::Debug for CubeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CubeCLBackend")
    }
}

impl Clone for CubeBackend {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Default for CubeBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "fusion"))]
impl BackendIr for CubeBackend {
    type Handle = CubeTensor;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        handle.handle
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        handle.handle
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        handle.handle
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        handle.handle
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor
    }
}
