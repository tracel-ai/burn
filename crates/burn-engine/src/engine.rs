// TODO:
// -[ ] move to separate crate e.g. `burn-engine`
// -[ ] rename `RunnerChannel` -> `RouterChannel`, `RunnerClient` -> `RouterClient`, `Runner` -> `RouterEndpoint`

use burn_backend::Backend;
use burn_backend::TensorData;
use burn_backend::{DeviceId, DeviceOps, ExecutionError};
use burn_ir::BackendIr;
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
use burn_wgpu::{Wgpu, WgpuDevice};

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
/// let device = EngineDevice::Cuda(0);
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
        // sum of all device counts for each backend device
        todo!()
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
    /// Wgpu handle.
    // Same for all Wgpu runtimes
    #[cfg(feature = "wgpu")]
    Wgpu(<Wgpu as BackendIr>::Handle),
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
    /// Wgpu client.
    #[cfg(feature = "wgpu")]
    Wgpu(Runner<Wgpu>),
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
            #[cfg(feature = "wgpu")]
            Self::Wgpu(runner) => runner.register_op(op),
            #[cfg(feature = "cuda")]
            Self::Cuda(runner) => runner.register_op(op),
        }
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(runner) => runner.read_tensor_async(tensor),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(runner) => runner.read_tensor_async(tensor),
            #[cfg(feature = "cuda")]
            Self::Cuda(runner) => runner.read_tensor_async(tensor),
        }
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        match self {
            #[cfg(feature = "cpu")]
            EngineClient::Cpu(runner) => runner.sync(),
            #[cfg(feature = "cuda")]
            EngineClient::Cuda(runner) => runner.sync(),
            #[cfg(feature = "wgpu")]
            EngineClient::Wgpu(runner) => runner.sync(),
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
            #[cfg(feature = "wgpu")]
            EngineClient::Wgpu(runner) => runner.create_empty_handle(),
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
            #[cfg(feature = "wgpu")]
            EngineClient::Wgpu(runner) => {
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
            Self::Wgpu(runner) => EngineDevice::Metal(runner.device()),
            #[cfg(feature = "vulkan")]
            Self::Wgpu(runner) => EngineDevice::Vulkan(runner.device()),
            #[cfg(feature = "webgpu")]
            Self::Wgpu(runner) => EngineDevice::WebGpu(runner.device()),
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
            #[cfg(feature = "wgpu")]
            EngineClient::Wgpu(runner) => runner.seed(seed),
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
            #[cfg(feature = "wgpu")]
            EngineClient::Wgpu(runner) => runner.supports_dtype(dtype),
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
    type Bridge = EngineBridge;
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
            EngineDevice::Metal(device) => EngineClient::Wgpu(Runner::new(device.clone())),
            #[cfg(feature = "vulkan")]
            EngineDevice::Vulkan(device) => EngineClient::Wgpu(Runner::new(device.clone())),
            #[cfg(feature = "webgpu")]
            EngineDevice::WebGpu(device) => EngineClient::Wgpu(Runner::new(device.clone())),
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
            #[cfg(feature = "wgpu")]
            EngineClient::Wgpu(runner) => EngineHandle::Wgpu(runner.get_tensor_handle(tensor)),
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
            #[cfg(feature = "wgpu")]
            (EngineClient::Wgpu(runner), EngineHandle::Wgpu(handle)) => {
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
            #[cfg(feature = "metal")]
            EngineDevice::Metal(device) => <Metal as Backend>::name(device),
            #[cfg(feature = "cuda")]
            EngineDevice::Cuda(device) => <Cuda as Backend>::name(device),
            #[cfg(feature = "vulkan")]
            EngineDevice::Vulkan(device) => <Vulkan as Backend>::name(device),
        };
        format!("engine<{device_name}>")
    }
}

/// Bridge for transferring tensors between backends
pub struct EngineBridge;

impl MultiBackendBridge for EngineBridge {
    type TensorHandle = EngineHandle;
    type Device = EngineDevice;

    fn change_backend_float(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        todo!()
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
