use crate::{CubeRuntime, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    Backend, BackendGraph, BackendTypes, DTypeUsage, DTypeUsageSet, DeviceOps, ExecutionError,
    InstallMemoryPoolsError, MemoryPoolLayout, MemoryPoolUsage, SlicedPool, SlicedPoolReport,
    TensorData,
};
use burn_std::{BoolStore, DType, quantization::quantizable};
use cubecl::{
    MemoryConfiguration, MemoryPoolKind,
    client::ComputeClient,
    config::memory::{MemoryPoolConfig, MemoryPoolsConfig, MemoryPoolsPreset},
    config::size::MemorySize,
    features::{MmaConfig, TypeUsage},
    ir::ElemType,
    server::ComputeServer,
};
use std::marker::PhantomData;

#[cfg(not(feature = "fusion"))]
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
#[cfg(not(feature = "fusion"))]
use burn_ir::{BackendIr, TensorHandle};

/// Whether the runtime can hold a quantized dtype's scales, which `dtype_to_storage_type` misses
/// because it doesn't see the scheme's scale levels. Non-quantized dtypes always pass.
fn qfloat_params_usable(client: &ComputeClient, dtype: DType) -> bool {
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

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct CubeBackend<R: CubeRuntime> {
    _runtime: PhantomData<R>,
}

impl<R> BackendTypes for CubeBackend<R>
where
    R: CubeRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
{
    type Device = R::Device;

    type FloatTensorPrimitive = CubeTensor<R>;
    type IntTensorPrimitive = CubeTensor<R>;
    type BoolTensorPrimitive = CubeTensor<R>;
    type QuantizedTensorPrimitive = CubeTensor<R>;

    type GraphPrimitive = cubecl::client::Graph;
}

impl<R> Backend for CubeBackend<R>
where
    R: CubeRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
{
    fn name(device: &Self::Device) -> String {
        let client = R::client(device);
        format!("cubecl<{}>", client.name())
    }

    fn seed(_device: &Self::Device, seed: u64) {
        cubek::random::seed(seed);
    }

    fn ad_enabled(_device: &Self::Device) -> bool {
        false
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = R::client(device);
        // A barrier plus the device's own fault, and nothing more: a launch
        // failure lives on the buffers the launch never wrote and surfaces on
        // the read of one of them, so it is not this sync's to report.
        // `client.sync_buffers` is the same barrier plus a check of named
        // tensors, for a caller that wants both.
        futures_lite::future::block_on(client.sync()).map_err(|err| ExecutionError::WithContext {
            reason: format!("{err}"),
        })
    }

    fn graph_prepare(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = R::client(device);
        client.graph_prepare().map_err(graph_err)
    }

    fn graph_start_capture(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = R::client(device);
        client.start_capture().map_err(graph_err)
    }

    fn graph_stop_capture(device: &Self::Device) -> Result<BackendGraph<Self>, ExecutionError> {
        let client = R::client(device);
        client.stop_capture().map_err(graph_err)
    }

    unsafe fn graph_replay(
        _device: &Self::Device,
        graph: &BackendGraph<Self>,
    ) -> Result<(), ExecutionError> {
        // cubecl's `Graph::replay` blocks on the enqueue and reports what the
        // enqueue said; a failure also leaves the graph's write set carrying
        // it, so a read of those buffers keeps failing until a replay lands.
        //
        // Safety: the buffer-liveness and stream-ordering obligations are the
        // caller's, forwarded verbatim from this method's own contract.
        unsafe { graph.replay() }.map_err(graph_err)
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
        let client = R::client(device);
        client.memory_persistent_allocation(input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        let client = R::client(device);
        client.memory_cleanup();
    }

    fn memory_install_pools(
        device: &Self::Device,
        layout: MemoryPoolLayout,
    ) -> Result<(), InstallMemoryPoolsError> {
        let client = R::client(device);
        let properties = &client.properties().memory;
        let config = pool_config(layout, properties.alignment.max(1))?;

        // The runtime panics on an unhonourable layout, on a device thread the
        // caller cannot catch and taking the device with it. Resolving it here
        // first turns that into this method's error, by the runtime's own rules
        // rather than a second copy of them.
        MemoryConfiguration::default()
            .resolve(Some(&config), properties)
            .map_err(|err| InstallMemoryPoolsError::InvalidLayout {
                reason: err.to_string(),
            })?;

        client
            .install_memory_pools(&config)
            .map_err(runtime_install_error)
    }

    fn memory_pool_report(device: &Self::Device) -> Option<Vec<SlicedPoolReport>> {
        let report = R::client(device).memory_report();

        // The pools a layout can be paired with, in the order allocations are
        // routed through them: a `Sliced` layout maps onto them one to one, and
        // a `Direct` layout is the single entry with no page size. The presets
        // mix in pools of other kinds, dropped here — so their entries keep the
        // routing order but not the positions of any layout.
        Some(
            report
                .dynamic
                .iter()
                .filter_map(|pool| match pool.kind {
                    MemoryPoolKind::Sliced { page_size, .. } => Some(SlicedPoolReport {
                        page_size,
                        pages: pool.pages,
                        pages_peak: pool.pages_peak,
                        largest_alloc: pool.largest_alloc,
                    }),
                    MemoryPoolKind::Direct => Some(SlicedPoolReport {
                        page_size: 0,
                        pages: pool.pages,
                        pages_peak: pool.pages_peak,
                        largest_alloc: pool.largest_alloc,
                    }),
                    _ => None,
                })
                .collect(),
        )
    }

    fn memory_pool_usage(device: &Self::Device) -> Option<MemoryPoolUsage> {
        let usage = R::client(device).memory_usage();

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
        let client = R::client(device);
        client.staging(data.map(|td| &mut td.bytes), false);
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        // Right now no cubecl backend actually works with native bool, even if
        // the `TypeUsage` might indicate otherwise.
        if let DType::Bool(BoolStore::Native) = dtype {
            return false;
        }
        let client = R::client(device);

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
        let client = R::client(device);

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
        R::enumerate_devices(type_id).len()
    }

    fn flush(device: &Self::Device) {
        let client = R::client(device);
        client.flush().unwrap();
    }
}

impl<R: CubeRuntime> core::fmt::Debug for CubeBackend<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CubeCLBackend")
    }
}

impl<R: CubeRuntime> Clone for CubeBackend<R> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<R: CubeRuntime> Default for CubeBackend<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: cubecl::Runtime> CubeRuntime for R
where
    R::Device: DeviceOps,
{
    type CubeDevice = R::Device;
    type CubeServer = R::Server;
}

#[cfg(not(feature = "fusion"))]
impl<R: CubeRuntime> BackendIr for CubeBackend<R> {
    type Handle = CubeTensor<R>;

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

/// A pool layout in the runtime's own vocabulary, with sizes aligned to
/// `alignment` — the rounding the runtime applies anyway, done here because the
/// cap has to be counted in the pages it will actually build. Multiplying the
/// *requested* page size instead buys fewer aligned pages than were asked for,
/// and for a single-page pool a cap that cannot fit its page at all.
fn pool_config(
    layout: MemoryPoolLayout,
    alignment: u64,
) -> Result<MemoryPoolsConfig, InstallMemoryPoolsError> {
    let config = match layout {
        MemoryPoolLayout::Sliced(pools) => MemoryPoolsConfig::Explicit(
            pools
                .into_iter()
                .map(
                    |SlicedPool {
                         page_size,
                         pages,
                         max_slice,
                     }| {
                        let page_size = align_up(page_size, alignment)?;
                        let max_pool_size = pages
                            .map(|pages| {
                                page_size.checked_mul(pages).ok_or_else(|| {
                                    InstallMemoryPoolsError::InvalidLayout {
                                        reason: format!(
                                            "a cap of {pages} pages of {page_size} B overflows"
                                        ),
                                    }
                                })
                            })
                            .transpose()?;

                        Ok(MemoryPoolConfig::Sliced {
                            page_size: MemorySize(page_size),
                            max_slice_size: max_slice
                                .map(|size| align_up(size, alignment))
                                .transpose()?
                                .map(MemorySize),
                            max_pool_size: max_pool_size.map(MemorySize),
                            dealloc_period: None,
                        })
                    },
                )
                .collect::<Result<Vec<_>, _>>()?,
        ),
        // Never reclaiming on its own: a direct pool is installed to measure
        // what a workload allocates, which is a short window with an explicit
        // cleanup on either side of it.
        MemoryPoolLayout::Direct => {
            MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Direct { reclaim_at: None }])
        }
        MemoryPoolLayout::SubSlices => MemoryPoolsConfig::Preset(MemoryPoolsPreset::SubSlices),
        MemoryPoolLayout::ExclusivePages => {
            MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages)
        }
    };

    Ok(config)
}

/// A size rounded up to the device's alignment. Zero stays zero, for the
/// layout's own validation to reject by field.
fn align_up(size: u64, alignment: u64) -> Result<u64, InstallMemoryPoolsError> {
    size.checked_next_multiple_of(alignment)
        .ok_or_else(|| InstallMemoryPoolsError::InvalidLayout {
            reason: format!("{size} B cannot be aligned up to {alignment} B"),
        })
}

/// The runtime's refusal, in the backend's vocabulary.
fn runtime_install_error(err: cubecl::InstallMemoryPoolsError) -> InstallMemoryPoolsError {
    match err {
        cubecl::InstallMemoryPoolsError::PoolsInUse { bytes_in_use } => {
            InstallMemoryPoolsError::PoolsInUse { bytes_in_use }
        }
        cubecl::InstallMemoryPoolsError::Unsupported => InstallMemoryPoolsError::Unsupported,
    }
}
