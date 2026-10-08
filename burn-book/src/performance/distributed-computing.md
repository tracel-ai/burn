# Distributed Computing

Burn supports data-parallel training across multiple devices and transparent execution on devices
hosted by another process. These capabilities can be used independently or together:

- The types in `burn::tensor::distributed` provide collective tensor operations across a group of
  devices.
- `burn::train::ExecutionStrategy::ddp` uses those collectives to synchronize gradients during
  distributed data-parallel (DDP) training.
- A remote `Device` sends normal tensor operations to a Burn compute server. A set of remote devices
  can also participate in DDP.

## Distributed Tensor Operations

Collective support depends on the execution runtime. The current CubeCL implementation supplies
all-reduce on CUDA. Remote DDP also requires collective support on the server's devices. Use the
non-DDP multi-device strategy when the selected runtimes do not provide collectives.

The distributed tensor API currently centers on all-reduce:

| Type or function     | Purpose                                                                           |
| -------------------- | --------------------------------------------------------------------------------- |
| `DistributedContext` | Starts and owns communication resources for a device group                        |
| `DistributedConfig`  | Configures gradient aggregation for the group                                     |
| `ReduceOperation`    | Selects `Sum` or `Mean` reduction                                                 |
| `all_reduce`         | Reduces a tensor across every participating device and returns the result to each |
| `CollectiveTensor`   | Represents a collective result that must be synchronized before normal use        |

Create a context before issuing collectives. Dropping the context closes its communication server,
so keep it alive for as long as the device group is active:

```rust, ignore
use burn::tensor::{
    Device, DeviceType, Tensor,
    distributed::{
        CollectiveTensor, DistributedConfig, DistributedContext, ReduceOperation, all_reduce,
    },
};

let devices = Device::enumerate(DeviceType::Cuda).into_vec();
let _context = DistributedContext::init(
    devices.clone(),
    DistributedConfig {
        all_reduce_op: ReduceOperation::Mean,
    },
);

// Every participant submits its local tensor with the same device list.
let local_tensors: Vec<Tensor<2>> = devices
    .iter()
    .map(compute_local_value)
    .collect();
let collectives: Vec<_> = local_tensors
    .into_iter()
    .map(|tensor| all_reduce(tensor, ReduceOperation::Sum, devices.clone()))
    .collect();
let reduced: Vec<Tensor<2>> = collectives
    .into_iter()
    .map(CollectiveTensor::resolve)
    .collect();
```

`all_reduce` returns a `CollectiveTensor` because collective communication can be asynchronous. Call
`resolve()` before using its result; it synchronizes the collective and returns a regular `Tensor`.
The unsafe `assume_resolved()` method is reserved for code that arranges synchronization itself.

Every participant must invoke collectives in a compatible order with the same device group.
Application code will usually use the DDP training strategy instead of calling `all_reduce`
directly.

## Distributed Data Parallel Training

DDP keeps one model replica on each device and splits training input across them. Each replica
computes a forward and backward pass locally, then Burn all-reduces the gradients before applying
the optimizer update. With `ReduceOperation::Mean`, every replica receives the mean gradient.

```rust, ignore
use burn::{
    tensor::{Device, DeviceType, distributed::{DistributedConfig, ReduceOperation}},
    train::{ExecutionStrategy, Learner, SupervisedTraining},
};

// List all available CUDA devices
let devices = Device::enumerate(DeviceType::Cuda).into_vec();
let strategy = ExecutionStrategy::ddp(
    devices,
    DistributedConfig {
        all_reduce_op: ReduceOperation::Mean,
    },
);

// Init the model on the main device with autodiff
let model = ModelConfig::new().init(&strategy.main_device().clone().autodiff());

// Launch DDP training
let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
    .with_training_strategy(strategy.into())
    .num_epochs(config.num_epochs);
let result = training.launch(Learner::new(model, optimizer, lr_scheduler));
```

This keeps model construction independent of whether the selected strategy is single-device,
multi-device, or DDP. The learner manages model replicas, data distribution, collective gradient
synchronization, and the lifetime of the `DistributedContext`.

DDP differs from `ExecutionStrategy::MultiDevice`: DDP gives each device a model replica and uses
collectives to synchronize gradients, whereas the multi-device strategy coordinates optimization
through Burn's non-DDP multi-device training path.

## Remote Devices

A remote device implements the same `Device` interface as a local CUDA, WGPU, or CPU device. Tensor
creation and operations use the normal API, but execution happens on a device a Burn server hosts.
A `RemoteHost` names the server, and `Device::remote_options` connects one of its devices:

```rust, ignore
let host = RemoteHost::iroh(server_id).with_credential(token);
let device = Device::remote_options(&host).init()?;

let tensor = Tensor::<2>::ones([32, 128], &device);
let output = model.to_device(&device).forward(tensor); // Executed by the remote server.
```

The server hosts its devices on a transport:

```rust, ignore
let transport = IrohTransport::new(IrohIdentity::load_or_create("server.key")?);
println!("server id: {}", transport.id());

RemoteServer::new([Device::cuda(0)])
    .with_authorizer(TokenAuthorizer::new(token)?)
    .serve(transport)?;
```

Iroh, the default transport, identifies a server by its id rather than a fixed address, and works
across any network, authenticated and encrypted. A system should generate a random `IrohIdentity`
and distribute its id through a trusted channel. A server opens every session unless given an
authorizer, such as a `TokenAuthorizer` checking the credential its clients set.

WebSocket is the simplest setup on a trusted network: `WebSocketTransport::new(3000)` on the server
and `RemoteHost::websocket("ws://gpu:3000")` on the client. It is unencrypted, so a token stops stray
clients but not someone reading the traffic.

An application that already runs an Iroh endpoint dials from it with `IrohHost::with_endpoint`, and
serves on it with `RemoteServer::into_protocol`. Endpoints Burn binds send no segmentation-offloaded
(GSO) batches because of [iroh#4555](https://github.com/n0-computer/iroh/issues/4555);
`iroh_segmentation_offload` under `[remote]` in `burn.toml` turns them on. An application's own
endpoint keeps its own setting.

`Device::enumerate(DeviceType::Remote(host))` lists every device the server hosts, beside any
local device type, and each connects on first use; `host.devices()` does the same and returns an
error where `enumerate` panics.
`init_async().await` connects from async code, and is the only form in a browser, where a
synchronous connection cannot be established.

### DDP on Remote Devices

Remote execution and DDP compose naturally. The
[`text-classification` example](https://github.com/tracel-ai/burn/tree/main/examples/text-classification/examples/ag-news-train.rs)
lists every device a remote WebSocket server hosts and passes them to the same DDP
strategy:

```rust, ignore
pub fn run() {
    let devices = Device::enumerate(DeviceType::Remote(RemoteHost::websocket(ADDRESS)));

    crate::launch(ExecutionStrategy::ddp(
        devices.into_vec(),
        DistributedConfig {
            all_reduce_op: ReduceOperation::Mean,
        },
    ));
}
```

From the learner's perspective, local and remote DDP use the same `Vec<Device>`. The remote devices
forward computation to the server, while the distributed context coordinates gradient collectives
across the selected server devices.

## Choosing an Approach

- Use a single remote device when computation should run elsewhere but does not need data-parallel
  synchronization.
- Use local DDP when several devices are directly available to the training process.
- Use remote devices with DDP when a Burn server exposes several accelerators to a client.

Distributed execution assumes that participating devices support the required collective operations.
