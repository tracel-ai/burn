# Backend and Device

Burn's user-facing API is centered on `Tensor`, `Module`, and `Device`. These types are not generic
over a backend. Instead, every tensor carries a runtime `Device` that identifies where and how its
operations execute.

```rust, ignore
use burn::tensor::{Device, Tensor};

let device = Device::wgpu(Default::default());
let tensor = Tensor::<2>::ones([2, 3], &device);

// The same model and tensor API can target another backend.
let other_device = Device::cuda(0);
let tensor = tensor.to_device(&other_device);
```

The corresponding Cargo feature must be enabled for each device constructor. For example, the `wgpu`
and `cuda` features make `Device::wgpu` and `Device::cuda` available.

## Selecting a Device

`Device` provides constructors for the backends enabled in your build. Common choices include:

| Constructor                                | Target                                          |
| ------------------------------------------ | ----------------------------------------------- |
| `Device::wgpu(Default::default())`         | Best WGPU adapter available                     |
| `Device::wgpu(DeviceKind::DiscreteGpu(0))` | First discrete GPU through WGPU                 |
| `Device::vulkan(Default::default())`       | Best Vulkan adapter                             |
| `Device::metal(Default::default())`        | Best Metal adapter                              |
| `Device::webgpu(Default::default())`       | Browser WebGPU device                           |
| `Device::cuda(0)`                          | CUDA GPU at index 0                             |
| `Device::cuda(DeviceIndex::Default)`       | Backend-selected CUDA GPU                       |
| `Device::rocm(0)`                          | ROCm/HIP GPU at index 0                         |
| `Device::cpu()`                            | CubeCL CPU backend                              |
| `Device::flex()`                           | Flex CPU backend                                |
| `Device::ndarray()`                        | NdArray CPU backend (deprecated)                |
| `Device::libtorch()`                       | LibTorch CPU backend (deprecated)               |
| `Device::libtorch_cuda(0)`                 | LibTorch CUDA GPU at index 0 (deprecated)       |
| `Device::libtorch_mps()`                   | LibTorch Metal Performance Shaders (deprecated) |
| `Device::libtorch_vulkan()`                | LibTorch Vulkan device (deprecated)             |

Indexed devices accept either an integer or `DeviceIndex`. WGPU-family constructors accept a
`DeviceKind`, which can select a discrete, integrated, or virtual GPU, a CPU adapter, or the best
available device:

```rust, ignore
use burn::tensor::{Device, DeviceIndex, DeviceKind};

let default_wgpu = Device::wgpu(Default::default());
let discrete_wgpu = Device::wgpu(DeviceKind::DiscreteGpu(0));
let integrated_wgpu = Device::wgpu(DeviceKind::IntegratedGpu(0));
let default_cuda = Device::cuda(DeviceIndex::Default);
let second_cuda = Device::cuda(1);
```

Burn also supports remote devices when the corresponding remote feature is enabled. Constructors
include `Device::remote_websocket` for WebSocket connections and `Device::remote_iroh` for
peer-to-peer remote execution.

## Using a Device

Tensor creation methods take `&Device`. Modules receive the same device when their parameters are
initialized:

```rust, ignore
let device = Device::cuda(0);
let input = Tensor::<2>::zeros([32, 128], &device);
let model = ModelConfig::new().init(&device);
let output = model.forward(input);
```

Use `Tensor::to_device` or `Module::to_device` to move existing values. Operations involving
multiple tensors require compatible devices, so move them explicitly before combining them.

```rust, ignore
let cpu = Device::flex();
let gpu = Device::cuda(0);

let tensor = Tensor::<2>::ones([2, 3], &cpu);
let tensor = tensor.to_device(&gpu);
let model = model.to_device(&gpu);
```

## Autodiff and Execution

Automatic differentiation is configured on a device. Tensors created on an autodiff device inherit
that context and can later change it independently. Calling `autodiff` returns such a device:

```rust, ignore
let device = Device::wgpu(Default::default());
let training_device = device.autodiff();

assert!(training_device.is_autodiff());
let inference_device = training_device.without_autodiff();
assert!(!inference_device.is_autodiff());
```

`autodiff()` and `without_autodiff()` are idempotent. The historical `inner()` method is equivalent
to `without_autodiff()`. Chain `autodiff().gradient_checkpointing()` to enable autodiff with the
balanced checkpointing strategy.

The following methods are also useful when coordinating execution:

- `seed(seed)` seeds random operations on the device.
- `sync()` waits for queued work and reports an execution error if one occurred.
- `flush()` submits queued work without waiting for completion.
- `is_autodiff()` reports whether autodiff is associated with the device.
- `gradient_checkpointing_strategy()` returns the active strategy, or `None` without autodiff.
- `supports_dtype(dtype)` reports whether the device supports a dtype.
- `memory_cleanup()` asks the backend to release unused cached allocations.

## Device Settings

Each device has their own runtime settings, including its default float, integer, and boolean
dtypes. Inspect them with `settings()` and set them with `configure()`:

```rust, ignore
use burn::tensor::{Device, DeviceConfig, FloatDType, IntDType};

let mut device = Device::cuda(0);
device.configure(
    DeviceConfig::default()
        .float_dtype(FloatDType::F16)
        .int_dtype(IntDType::I32),
)?;

let settings = device.settings();
```

Configure defaults before the first tensor operation on that device. Once initialized, the default
dtypes are locked; a later incompatible configuration returns an error.

## Enumerating Devices

`Device::enumerate` discovers enabled devices matching a filter. This is useful for selecting
hardware dynamically or setting up multi-device training:

```rust, ignore
use burn::tensor::{Device, DeviceFilter, DeviceType};

let devices = Device::enumerate(
    DeviceFilter::new()
        .with(DeviceType::Cuda)
        .with(DeviceType::Wgpu),
);
let devices = devices.into_vec();
```

The exact `DeviceType` variants available depend on the backend features enabled for the
application.

## Execution Stack

Under the hood, an operation flows through the **Tensor → Bridge → Dispatch → Backend** stack:

- `Tensor` provides the stable, backend-independent API used by applications.
- A bridge converts tensor handles into dispatch values.
- `Dispatch` selects the implementation associated with the tensor's device.
- The backend executes the primitive operation.

The `Backend` and `AutodiffBackend` traits still define the low-level implementation contract, but
ordinary application code does not need bounds such as `B: Backend`. You will mainly encounter those
traits when implementing or extending a backend; see
[Backend Extension](../advanced/backend-extension/README.md).
