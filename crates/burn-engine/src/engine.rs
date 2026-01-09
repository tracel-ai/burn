// TODO:
// -[ ] rename `RunnerChannel` -> `RouterChannel`, `RunnerClient` -> `RouterClient`, `Runner` -> `RouterEndpoint`

use alloc::format;
use alloc::string::String;

use burn_backend::Backend;
use burn_backend::TensorData;
use burn_backend::{DeviceId, DeviceOps, ExecutionError};
use burn_ir::BackendIr;
use burn_ir::TensorHandle;
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_std::future::DynFut;
use burn_std::{DType, Shape};

#[cfg(feature = "cpu")]
use burn_cpu::{Cpu, CpuDevice};
#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "ndarray")]
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "rocm")]
use burn_rocm::{Rocm, RocmDevice};
#[cfg(feature = "metal")]
use burn_wgpu::Metal;
#[cfg(feature = "vulkan")]
use burn_wgpu::Vulkan;
#[cfg(feature = "webgpu")]
use burn_wgpu::WebGpu;
#[cfg(feature = "wgpu")]
use burn_wgpu::WgpuDevice;

use burn_router::{
    BackendRouter, MultiBackendBridge, RouterTensor, Runner, RunnerChannel, RunnerClient,
};

/// The main execution engine in Burn.
///
/// [`Engine`] acts as a global backend that can manage multiple underlying
/// backends (e.g., `Cpu`, `Cuda`, `Wgpu`, `Metal`, etc.).  
/// It is responsible for:
/// - Dispatching tensor operations to the appropriate backend.
/// - Managing cross-backend tensor transfers.
///
/// Essentially, [`Engine`] is the single entry point for executing tensor operations
/// in a backend-agnostic way. It allows Burn to provide a unified, global backend
/// for users while still leveraging multiple specialized backends under the hood.
///
/// # Example
///
/// ```ignore
/// use burn::Engine;
/// use burn::EngineDevice;
///
/// // Select the device to execute operations on
/// let device = EngineDevice::Cuda(Default::default());
///
/// // Create a tensor using the global engine
/// let t = Tensor::<Engine, 2>::zeros([128, 128], &device);
/// ```
pub type Engine = BackendRouter<EngineChannel>;
// pub struct Engine;

/// Represents a device for the [`Engine`].
///
/// Each variant corresponds to a backend that the [`Engine`] can dispatch operations to.
///
/// # Example
///
/// ```ignore
/// use burn::EngineDevice;
///
/// #[cfg(feature = "cpu")]
/// let cpu_device = EngineDevice::Cpu(Default::default());
///
/// #[cfg(feature = "cuda")]
/// let cuda_device = EngineDevice::Cuda(Default::default());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineDevice {
    /// The [CPU backend](Cpu) device.
    #[cfg(feature = "cpu")]
    Cpu(CpuDevice),

    /// The [CUDA backend](Cuda) device.
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),

    /// The [Metal backend](Metal) device (via WGPU runtime).
    #[cfg(feature = "metal")]
    Metal(WgpuDevice),

    /// The [ROCm backend](Rocm) device.
    #[cfg(feature = "rocm")]
    Rocm(RocmDevice),

    /// The [Vulkan backend](Vulkan) device.
    #[cfg(feature = "vulkan")]
    Vulkan(WgpuDevice),

    /// The [WebGPU backend](WebGpu) device (via WGPU runtime).
    #[cfg(feature = "webgpu")]
    WebGpu(WgpuDevice),

    /// The [NdArray backend](NdArray) device (CPU-only).
    #[cfg(feature = "ndarray")]
    NdArray(NdArrayDevice),
    // Missing: tch, candle (?), and possibly autodiff
}

/// Global engine backend with feature-gated backend support
// pub type Engine = BackendRouter<EngineChannel>;

impl Default for EngineDevice {
    #[allow(unreachable_code)]
    fn default() -> Self {
        // TODO: which priority?

        #[cfg(feature = "cpu")]
        return Self::Cpu(CpuDevice::default());

        #[cfg(feature = "cuda")]
        return Self::Cuda(CudaDevice::default());

        #[cfg(feature = "metal")]
        return Self::Metal(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "rocm")]
        return Self::Rocm(RocmDevice::default());

        #[cfg(feature = "vulkan")]
        return Self::Vulkan(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "webgpu")]
        return Self::WebGpu(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "ndarray")]
        return Self::NdArray(NdArrayDevice::default());
    }
}

/// Base multiplier to avoid type_id clashes between backends.
/// Limits the number of device types per backend, but this is a sensible limit.
const TYPE_ID_BASE: u16 = 10;

impl EngineDevice {
    /// Returns a unique number per variant to encode into type_id.
    fn variant_id(&self) -> EngineId {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => EngineId::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => EngineId::Cuda,
            #[cfg(feature = "metal")]
            Self::Metal(_) => EngineId::Metal,
            #[cfg(feature = "rocm")]
            Self::Rocm(_) => EngineId::Rocm,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => EngineId::Vulkan,
            #[cfg(feature = "webgpu")]
            Self::WebGpu(_) => EngineId::WebGpu,
            #[cfg(feature = "ndarray")]
            Self::NdArray(_) => EngineId::NdArray,
        }
    }

    /// Encode variant ID and backend type ID into a unique `type_id`.
    fn encode_type_id(&self, backend_type_id: u16) -> u16 {
        u16::from(self.variant_id()) * TYPE_ID_BASE + backend_type_id
    }

    /// Decode an encoded `type_id` into variant ID and backend type ID.
    fn decode_type_id(type_id: u16) -> (EngineId, u16) {
        let variant = type_id / TYPE_ID_BASE;
        let backend_type_id = type_id % TYPE_ID_BASE;
        (
            EngineId::try_from(variant).expect("Unknown EngineDevice variant"),
            backend_type_id,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
enum EngineId {
    #[cfg(feature = "cpu")]
    Cpu = 0,
    #[cfg(feature = "cuda")]
    Cuda = 1,
    #[cfg(feature = "metal")]
    Metal = 2,
    #[cfg(feature = "rocm")]
    Rocm = 3,
    #[cfg(feature = "vulkan")]
    Vulkan = 4,
    #[cfg(feature = "webgpu")]
    WebGpu = 5,
    #[cfg(feature = "ndarray")]
    NdArray = 6,
}

impl From<EngineId> for u16 {
    fn from(variant: EngineId) -> Self {
        variant as u16
    }
}

impl TryFrom<u16> for EngineId {
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            #[cfg(feature = "cpu")]
            0 => Ok(Self::Cpu),
            #[cfg(feature = "cuda")]
            1 => Ok(Self::Cuda),
            #[cfg(feature = "metal")]
            2 => Ok(Self::Metal),
            #[cfg(feature = "rocm")]
            3 => Ok(Self::Rocm),
            #[cfg(feature = "vulkan")]
            4 => Ok(Self::Vulkan),
            #[cfg(feature = "webgpu")]
            5 => Ok(Self::WebGpu),
            #[cfg(feature = "ndarray")]
            6 => Ok(Self::NdArray),
            _ => Err(()),
        }
    }
}

impl DeviceOps for EngineDevice {}

impl burn_std::device::Device for EngineDevice {
    fn from_id(mut device_id: DeviceId) -> Self {
        let (engine_id, backend_type_id) = Self::decode_type_id(device_id.type_id);
        device_id.type_id = backend_type_id;

        match engine_id {
            #[cfg(feature = "cpu")]
            EngineId::Cpu => Self::Cpu(CpuDevice::from_id(device_id)),
            #[cfg(feature = "cuda")]
            EngineId::Cuda => Self::Cuda(CudaDevice::from_id(device_id)),
            #[cfg(feature = "metal")]
            EngineId::Metal => Self::Metal(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "rocm")]
            EngineId::Rocm => Self::Rocm(RocmDevice::from_id(device_id)),
            #[cfg(feature = "vulkan")]
            EngineId::Vulkan => Self::Vulkan(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "webgpu")]
            EngineId::WebGpu => Self::WebGpu(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "ndarray")]
            EngineId::NdArray => Self::NdArray(NdArrayDevice::from_id(device_id)),
        }
    }

    fn to_id(&self) -> DeviceId {
        let mut device_id = match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => device.to_id(),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.to_id(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.to_id(),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => device.to_id(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.to_id(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(device) => device.to_id(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => device.to_id(),
        };
        device_id.type_id = self.encode_type_id(device_id.type_id);
        device_id
    }

    fn device_count(type_id: u16) -> usize {
        let (engine_id, backend_type_id) = Self::decode_type_id(type_id);
        match engine_id {
            #[cfg(feature = "cpu")]
            EngineId::Cpu => CpuDevice::device_count(backend_type_id),
            #[cfg(feature = "cuda")]
            EngineId::Cuda => CudaDevice::device_count(backend_type_id),
            #[cfg(feature = "metal")]
            EngineId::Metal => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "rocm")]
            EngineId::Rocm => RocmDevice::device_count(backend_type_id),
            #[cfg(feature = "vulkan")]
            EngineId::Vulkan => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "webgpu")]
            EngineId::WebGpu => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "ndarray")]
            EngineId::NdArray => NdArrayDevice::device_count(backend_type_id),
        }
    }
}

/// Tensor handle for the different engine backends.
#[derive(Debug)]
pub enum EngineHandle {
    /// Cpu handle.
    #[cfg(feature = "cpu")]
    Cpu(<Cpu as BackendIr>::Handle),
    /// Cuda handle.
    #[cfg(feature = "cuda")]
    Cuda(<Cuda as BackendIr>::Handle),
    /// Metal handle.
    #[cfg(feature = "metal")]
    Metal(<Metal as BackendIr>::Handle),
    /// Vulkan handle.
    #[cfg(feature = "vulkan")]
    Vulkan(<Vulkan as BackendIr>::Handle),
    /// WebGpu handle.
    #[cfg(feature = "webgpu")]
    WebGpu(<WebGpu as BackendIr>::Handle),
    /// NdArray handle.
    #[cfg(feature = "ndarray")]
    NdArray(<NdArray as BackendIr>::Handle),
}

/// Client that routes operations to the appropriate backend.
#[derive(Debug, Clone)]
pub enum EngineClient {
    /// Cpu client.
    #[cfg(feature = "cpu")]
    Cpu(Runner<Cpu>),
    /// Cuda client.
    #[cfg(feature = "cuda")]
    Cuda(Runner<Cuda>),
    /// Metal client.
    #[cfg(feature = "metal")]
    Metal(Runner<Metal>),
    /// Vulkan client.
    #[cfg(feature = "vulkan")]
    Vulkan(Runner<Vulkan>),
    /// WebGpu client.
    #[cfg(feature = "webgpu")]
    WebGpu(Runner<WebGpu>),
    /// NdArray client.
    #[cfg(feature = "ndarray")]
    NdArray(Runner<NdArray>),
}

impl RunnerClient for EngineClient {
    type Device = EngineDevice;

    fn register_op(&self, op: OperationIr) {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(runner) => runner.register_op(op),
            #[cfg(feature = "metal")]
            Self::Metal(runner) => runner.register_op(op),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(runner) => runner.register_op(op),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(runner) => runner.register_op(op),
            #[cfg(feature = "cuda")]
            Self::Cuda(runner) => runner.register_op(op),
            #[cfg(feature = "ndarray")]
            Self::NdArray(runner) => runner.register_op(op),
        }
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(runner) => runner.read_tensor_async(tensor),
            #[cfg(feature = "metal")]
            Self::Metal(runner) => runner.read_tensor_async(tensor),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(runner) => runner.read_tensor_async(tensor),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(runner) => runner.read_tensor_async(tensor),
            #[cfg(feature = "cuda")]
            Self::Cuda(runner) => runner.read_tensor_async(tensor),
            #[cfg(feature = "ndarray")]
            Self::NdArray(runner) => runner.read_tensor_async(tensor),
        }
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        match self {
            #[cfg(feature = "cpu")]
            EngineClient::Cpu(runner) => runner.sync(),
            #[cfg(feature = "cuda")]
            EngineClient::Cuda(runner) => runner.sync(),
            #[cfg(feature = "metal")]
            EngineClient::Metal(runner) => runner.sync(),
            #[cfg(feature = "vulkan")]
            EngineClient::Vulkan(runner) => runner.sync(),
            #[cfg(feature = "webgpu")]
            EngineClient::WebGpu(runner) => runner.sync(),
            #[cfg(feature = "ndarray")]
            EngineClient::NdArray(runner) => runner.sync(),
        }
    }

    fn create_empty_handle(&self) -> TensorId {
        match self {
            #[cfg(feature = "cpu")]
            EngineClient::Cpu(runner) => runner.create_empty_handle(),
            #[cfg(feature = "cuda")]
            EngineClient::Cuda(runner) => runner.create_empty_handle(),
            #[cfg(feature = "metal")]
            EngineClient::Metal(runner) => runner.create_empty_handle(),
            #[cfg(feature = "vulkan")]
            EngineClient::Vulkan(runner) => runner.create_empty_handle(),
            #[cfg(feature = "webgpu")]
            EngineClient::WebGpu(runner) => runner.create_empty_handle(),
            #[cfg(feature = "ndarray")]
            EngineClient::NdArray(runner) => runner.create_empty_handle(),
        }
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        match self {
            #[cfg(feature = "cpu")]
            EngineClient::Cpu(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
            }
            #[cfg(feature = "cuda")]
            EngineClient::Cuda(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
            }
            #[cfg(feature = "metal")]
            EngineClient::Metal(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
            }
            #[cfg(feature = "vulkan")]
            EngineClient::Vulkan(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
            }
            #[cfg(feature = "webgpu")]
            EngineClient::WebGpu(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
            }
            #[cfg(feature = "ndarray")]
            EngineClient::NdArray(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
            }
        }
    }

    fn device(&self) -> Self::Device {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(runner) => EngineDevice::Cpu(runner.device()),
            #[cfg(feature = "cuda")]
            Self::Cuda(runner) => EngineDevice::Cuda(runner.device()),
            #[cfg(feature = "metal")]
            Self::Metal(runner) => EngineDevice::Metal(runner.device()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(runner) => EngineDevice::Vulkan(runner.device()),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(runner) => EngineDevice::WebGpu(runner.device()),
            #[cfg(feature = "ndarray")]
            Self::NdArray(runner) => EngineDevice::NdArray(runner.device()),
        }
    }

    fn seed(&self, seed: u64) {
        match self {
            #[cfg(feature = "cpu")]
            EngineClient::Cpu(runner) => runner.seed(seed),
            #[cfg(feature = "cuda")]
            EngineClient::Cuda(runner) => runner.seed(seed),
            #[cfg(feature = "metal")]
            EngineClient::Metal(runner) => runner.seed(seed),
            #[cfg(feature = "vulkan")]
            EngineClient::Vulkan(runner) => runner.seed(seed),
            #[cfg(feature = "webgpu")]
            EngineClient::WebGpu(runner) => runner.seed(seed),
            #[cfg(feature = "ndarray")]
            EngineClient::NdArray(runner) => runner.seed(seed),
        }
    }

    fn supports_dtype(&self, dtype: DType) -> bool {
        match self {
            #[cfg(feature = "cpu")]
            EngineClient::Cpu(runner) => runner.supports_dtype(dtype),
            #[cfg(feature = "cuda")]
            EngineClient::Cuda(runner) => runner.supports_dtype(dtype),
            #[cfg(feature = "metal")]
            EngineClient::Metal(runner) => runner.supports_dtype(dtype),
            #[cfg(feature = "vulkan")]
            EngineClient::Vulkan(runner) => runner.supports_dtype(dtype),
            #[cfg(feature = "webgpu")]
            EngineClient::WebGpu(runner) => runner.supports_dtype(dtype),
            #[cfg(feature = "ndarray")]
            EngineClient::NdArray(runner) => runner.supports_dtype(dtype),
        }
    }
}

/// Channel for the engine backend
#[derive(Clone)]
pub struct EngineChannel;

impl RunnerChannel for EngineChannel {
    type Device = EngineDevice;
    type Bridge = BackendBridge;
    type FloatElem = f32; // or make configurable
    type IntElem = i32;
    type BoolElem = bool;
    type Client = EngineClient;

    fn init_client(device: &Self::Device) -> Self::Client {
        match device {
            #[cfg(feature = "cpu")]
            EngineDevice::Cpu(device) => EngineClient::Cpu(Runner::new(device.clone())),
            #[cfg(feature = "cuda")]
            EngineDevice::Cuda(device) => EngineClient::Cuda(Runner::new(device.clone())),
            #[cfg(feature = "metal")]
            EngineDevice::Metal(device) => EngineClient::Metal(Runner::new(device.clone())),
            #[cfg(feature = "vulkan")]
            EngineDevice::Vulkan(device) => EngineClient::Vulkan(Runner::new(device.clone())),
            #[cfg(feature = "webgpu")]
            EngineDevice::WebGpu(device) => EngineClient::WebGpu(Runner::new(device.clone())),
            #[cfg(feature = "ndarray")]
            EngineDevice::NdArray(device) => EngineClient::NdArray(Runner::new(device.clone())),
        }
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> EngineHandle {
        match client {
            #[cfg(feature = "cpu")]
            EngineClient::Cpu(runner) => EngineHandle::Cpu(runner.get_tensor_handle(tensor)),
            #[cfg(feature = "cuda")]
            EngineClient::Cuda(runner) => EngineHandle::Cuda(runner.get_tensor_handle(tensor)),
            #[cfg(feature = "metal")]
            EngineClient::Metal(runner) => EngineHandle::Metal(runner.get_tensor_handle(tensor)),
            #[cfg(feature = "vulkan")]
            EngineClient::Vulkan(runner) => EngineHandle::Vulkan(runner.get_tensor_handle(tensor)),
            #[cfg(feature = "webgpu")]
            EngineClient::WebGpu(runner) => EngineHandle::WebGpu(runner.get_tensor_handle(tensor)),
            #[cfg(feature = "ndarray")]
            EngineClient::NdArray(runner) => {
                EngineHandle::NdArray(runner.get_tensor_handle(tensor))
            }
        }
    }

    fn register_tensor(
        client: &Self::Client,
        handle: EngineHandle,
        shape: Shape,
        dtype: DType,
    ) -> RouterTensor<Self::Client> {
        match (client, handle) {
            #[cfg(feature = "cpu")]
            (EngineClient::Cpu(runner), EngineHandle::Cpu(handle)) => {
                runner.register_tensor(handle, shape, dtype, client.clone())
            }
            #[cfg(feature = "cuda")]
            (EngineClient::Cuda(runner), EngineHandle::Cuda(handle)) => {
                runner.register_tensor(handle, shape, dtype, client.clone())
            }
            #[cfg(feature = "metal")]
            (EngineClient::Metal(runner), EngineHandle::Metal(handle)) => {
                runner.register_tensor(handle, shape, dtype, client.clone())
            }
            #[cfg(feature = "vulkan")]
            (EngineClient::Vulkan(runner), EngineHandle::Vulkan(handle)) => {
                runner.register_tensor(handle, shape, dtype, client.clone())
            }
            #[cfg(feature = "webgpu")]
            (EngineClient::WebGpu(runner), EngineHandle::WebGpu(handle)) => {
                runner.register_tensor(handle, shape, dtype, client.clone())
            }
            #[cfg(feature = "ndarray")]
            (EngineClient::NdArray(runner), EngineHandle::NdArray(handle)) => {
                runner.register_tensor(handle, shape, dtype, client.clone())
            }
            (c, h) => panic!("Handle and client backend mismatch: ({c:?}, {h:?}"),
        }
    }

    fn name(device: &Self::Device) -> String {
        let device_name = match device {
            #[cfg(feature = "cpu")]
            EngineDevice::Cpu(device) => <Cpu as Backend>::name(device),
            #[cfg(feature = "cuda")]
            EngineDevice::Cuda(device) => <Cuda as Backend>::name(device),
            #[cfg(feature = "metal")]
            EngineDevice::Metal(device) => <Metal as Backend>::name(device),
            #[cfg(feature = "vulkan")]
            EngineDevice::Vulkan(device) => <Vulkan as Backend>::name(device),
            #[cfg(feature = "webgpu")]
            EngineDevice::WebGpu(device) => <WebGpu as Backend>::name(device),
            #[cfg(feature = "ndarray")]
            EngineDevice::NdArray(device) => <NdArray as Backend>::name(device),
        };
        format!("engine<{device_name}>")
    }
}

/// Bridge for transferring tensors between backends.
pub struct BackendBridge;

/// Move the tensor handle to the given device (on the same backend).
fn to_device<B1: BackendIr>(handle: B1::Handle, shape: Shape, device: &B1::Device) -> B1::Handle {
    let tensor = B1::float_tensor(TensorHandle {
        handle: handle,
        shape: shape,
    });
    let tensor = B1::float_to_device(tensor, device);
    B1::float_tensor_handle(tensor)
}

/// Move the tensor handle to the given backend device.
///
/// # NOTE
/// The data transfer is not direct from `B1` -> `B2`. The data is read in-memory, which is not optimal.
fn to_backend<B1: BackendIr, B2: BackendIr>(
    handle: B1::Handle,
    shape: Shape,
    device: &B2::Device,
) -> B2::Handle {
    let tensor = B1::float_tensor(TensorHandle {
        handle: handle,
        shape: shape,
    });
    let data = burn_backend::try_read_sync(B1::float_into_data(tensor)).unwrap().expect("Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.");
    let tensor = B2::float_from_data(data, device);
    B2::float_tensor_handle(tensor)
}

impl MultiBackendBridge for BackendBridge {
    type TensorHandle = EngineHandle;
    type Device = EngineDevice;

    fn change_backend_float(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        // NOTE: default backend dtypes are ignored for these, but it might make sense anyway to
        // have dtype set for a device instead: https://github.com/tracel-ai/burn/issues/3642
        match (tensor, target_device) {
            // Change device only
            #[cfg(feature = "cpu")]
            (EngineHandle::Cpu(handle), EngineDevice::Cpu(device)) => {
                EngineHandle::Cpu(to_device::<Cpu>(handle, shape, device))
            }
            #[cfg(feature = "cuda")]
            (EngineHandle::Cuda(handle), EngineDevice::Cuda(device)) => {
                EngineHandle::Cuda(to_device::<Cuda>(handle, shape, device))
            }
            #[cfg(feature = "metal")]
            (EngineHandle::Metal(handle), EngineDevice::Metal(device)) => {
                EngineHandle::Metal(to_device::<Metal>(handle, shape, device))
            }
            #[cfg(feature = "vulkan")]
            (EngineHandle::Vulkan(handle), EngineDevice::Vulkan(device)) => {
                EngineHandle::Vulkan(to_device::<Vulkan>(handle, shape, device))
            }
            #[cfg(feature = "webgpu")]
            (EngineHandle::WebGpu(handle), EngineDevice::WebGpu(device)) => {
                EngineHandle::WebGpu(to_device::<WebGpu>(handle, shape, device))
            }
            #[cfg(feature = "ndarray")]
            (EngineHandle::NdArray(handle), EngineDevice::NdArray(device)) => {
                EngineHandle::NdArray(to_device::<NdArray>(handle, shape, device))
            }
            // Change backends: Cpu -> Other
            #[cfg(all(feature = "cpu", feature = "cuda"))]
            (EngineHandle::Cpu(handle), EngineDevice::Cuda(device)) => {
                EngineHandle::Cuda(to_backend::<Cpu, Cuda>(handle, shape, device))
            }
            #[cfg(all(feature = "cpu", feature = "metal"))]
            (EngineHandle::Cpu(handle), EngineDevice::Metal(device)) => {
                EngineHandle::Metal(to_backend::<Cpu, Metal>(handle, shape, device))
            }
            #[cfg(all(feature = "cpu", feature = "vulkan"))]
            (EngineHandle::Cpu(handle), EngineDevice::Vulkan(device)) => {
                EngineHandle::Vulkan(to_backend::<Cpu, Vulkan>(handle, shape, device))
            }
            #[cfg(all(feature = "cpu", feature = "webgpu"))]
            (EngineHandle::Cpu(handle), EngineDevice::WebGpu(device)) => {
                EngineHandle::WebGpu(to_backend::<Cpu, WebGpu>(handle, shape, device))
            }
            #[cfg(all(feature = "cpu", feature = "ndarray"))]
            (EngineHandle::Cpu(handle), EngineDevice::NdArray(device)) => {
                EngineHandle::NdArray(to_backend::<Cpu, NdArray>(handle, shape, device))
            }
            // Change backends: Cuda -> Other
            #[cfg(all(feature = "cuda", feature = "cpu"))]
            (EngineHandle::Cuda(handle), EngineDevice::Cpu(device)) => {
                EngineHandle::Cpu(to_backend::<Cuda, Cpu>(handle, shape, device))
            }
            #[cfg(all(feature = "cuda", feature = "metal"))]
            (EngineHandle::Cuda(handle), EngineDevice::Metal(device)) => {
                EngineHandle::Metal(to_backend::<Cuda, Metal>(handle, shape, device))
            }
            #[cfg(all(feature = "cuda", feature = "vulkan"))]
            (EngineHandle::Cuda(handle), EngineDevice::Vulkan(device)) => {
                EngineHandle::Vulkan(to_backend::<Cuda, Vulkan>(handle, shape, device))
            }
            #[cfg(all(feature = "cuda", feature = "webgpu"))]
            (EngineHandle::Cuda(handle), EngineDevice::WebGpu(device)) => {
                EngineHandle::WebGpu(to_backend::<Cuda, WebGpu>(handle, shape, device))
            }
            #[cfg(all(feature = "cuda", feature = "ndarray"))]
            (EngineHandle::Cuda(handle), EngineDevice::NdArray(device)) => {
                EngineHandle::NdArray(to_backend::<Cuda, NdArray>(handle, shape, device))
            }
            // Change backends: Metal -> Other
            #[cfg(all(feature = "metal", feature = "cpu"))]
            (EngineHandle::Metal(handle), EngineDevice::Cpu(device)) => {
                EngineHandle::Cpu(to_backend::<Metal, Cpu>(handle, shape, device))
            }
            #[cfg(all(feature = "metal", feature = "cuda"))]
            (EngineHandle::Metal(handle), EngineDevice::Cuda(device)) => {
                EngineHandle::Cuda(to_backend::<Metal, Cuda>(handle, shape, device))
            }
            #[cfg(all(feature = "metal", feature = "ndarray"))]
            (EngineHandle::Metal(handle), EngineDevice::NdArray(device)) => {
                EngineHandle::NdArray(to_backend::<Metal, NdArray>(handle, shape, device))
            }
            // Change backends: Vulkan -> Other
            #[cfg(all(feature = "vulkan", feature = "cpu"))]
            (EngineHandle::Vulkan(handle), EngineDevice::Cpu(device)) => {
                EngineHandle::Cpu(to_backend::<Vulkan, Cpu>(handle, shape, device))
            }
            #[cfg(all(feature = "vulkan", feature = "cuda"))]
            (EngineHandle::Vulkan(handle), EngineDevice::Cuda(device)) => {
                EngineHandle::Cuda(to_backend::<Vulkan, Cuda>(handle, shape, device))
            }
            #[cfg(all(feature = "vulkan", feature = "ndarray"))]
            (EngineHandle::Vulkan(handle), EngineDevice::NdArray(device)) => {
                EngineHandle::NdArray(to_backend::<Vulkan, NdArray>(handle, shape, device))
            }
            // Change backends: WebGpu -> Other
            #[cfg(all(feature = "webgpu", feature = "cpu"))]
            (EngineHandle::WebGpu(handle), EngineDevice::Cpu(device)) => {
                EngineHandle::Cpu(to_backend::<WebGpu, Cpu>(handle, shape, device))
            }
            #[cfg(all(feature = "webgpu", feature = "cuda"))]
            (EngineHandle::WebGpu(handle), EngineDevice::Cuda(device)) => {
                EngineHandle::Cuda(to_backend::<WebGpu, Cuda>(handle, shape, device))
            }
            #[cfg(all(feature = "webgpu", feature = "ndarray"))]
            (EngineHandle::WebGpu(handle), EngineDevice::NdArray(device)) => {
                EngineHandle::NdArray(to_backend::<WebGpu, NdArray>(handle, shape, device))
            }
            // Change backends: NdArray -> Other
            #[cfg(all(feature = "ndarray", feature = "cpu"))]
            (EngineHandle::NdArray(handle), EngineDevice::Cpu(device)) => {
                EngineHandle::Cpu(to_backend::<NdArray, Cpu>(handle, shape, device))
            }
            #[cfg(all(feature = "ndarray", feature = "cuda"))]
            (EngineHandle::NdArray(handle), EngineDevice::Cuda(device)) => {
                EngineHandle::Cuda(to_backend::<NdArray, Cuda>(handle, shape, device))
            }
            #[cfg(all(feature = "ndarray", feature = "metal"))]
            (EngineHandle::NdArray(handle), EngineDevice::Metal(device)) => {
                EngineHandle::Metal(to_backend::<NdArray, Metal>(handle, shape, device))
            }
            #[cfg(all(feature = "ndarray", feature = "vulkan"))]
            (EngineHandle::NdArray(handle), EngineDevice::Vulkan(device)) => {
                EngineHandle::Vulkan(to_backend::<NdArray, Vulkan>(handle, shape, device))
            }
            #[cfg(all(feature = "ndarray", feature = "webgpu"))]
            (EngineHandle::NdArray(handle), EngineDevice::WebGpu(device)) => {
                EngineHandle::WebGpu(to_backend::<NdArray, WebGpu>(handle, shape, device))
            }
            _ => unreachable!(), // Metal <> Vulkan <> WebGpu
        }
    }

    fn change_backend_int(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        todo!()
    }

    fn change_backend_bool(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        todo!()
    }
}
