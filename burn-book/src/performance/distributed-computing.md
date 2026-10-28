# Distributed Computing

Burn supports data-parallel training across multiple devices, splitting one model across several
devices, and transparent execution on devices hosted by another process. These capabilities can be
used independently or together:

- The types in `burn::tensor::distributed` configure gradient synchronization across a group of
  devices.
- `burn::train::ExecutionStrategy::ddp` synchronizes gradients with the backends' collectives
  during distributed data-parallel (DDP) training.
- A remote `Device` sends normal tensor operations to a Burn compute server. A set of remote devices
  can also participate in DDP.
- `burn::module::parallel::LayerParallelism` splits one model by whole layers across several
  devices, so a model too large for one device runs as a sequence of stages.

## Distributed Tensor Operations

`Tensor` runs five collectives over a group of devices, each called once with every member's
tensor. Member `i` is position `i`, on its own member's device and of the same shape and dtype as
every other member, and one output comes back per member, on that member's device. A chunk is a
member's share of a dimension, as even as possible with the first members holding one more.

| Operation                                    | Member `i` receives                                      |
| -------------------------------------------- | -------------------------------------------------------- |
| `Tensor::all_reduce(tensors, op)`            | `op` over every member's tensor                          |
| `Tensor::all_gather(tensors, dim)`           | every member's tensor, concatenated in member order      |
| `Tensor::reduce_scatter(tensors, dim, op)`   | chunk `i` of `op` over every member's tensor             |
| `Tensor::all_to_all(tensors, split, concat)` | chunk `i` of every member's tensor, concatenated         |
| `Tensor::broadcast(tensor, devices)`         | the tensor, on each of `devices`                         |

The reductions take a numeric tensor, float or int; the others take any kind, bool included. No
backend has a quantized collective, so a quantized tensor is refused.

```rust, ignore
let devices = Device::enumerate(DeviceType::Cuda).into_vec();
let ones = Tensor::<2>::ones([32, 128], &devices[0]);

// One tensor per device, then their sum on every device.
let members = Tensor::broadcast(ones, &devices);
let summed = Tensor::all_reduce(members, ReduceOperation::Sum);
```

A runtime with a transport between its devices runs what it can natively, NCCL on CUDA, and
broadcast is always emulated; any other runtime, and a group mixing runtimes, emulates them with
transfers between devices. DDP synchronizes gradients with the all-reduce, so it runs on every
CubeCL runtime and on remote devices, at full speed where the devices share a transport.

`burn::tensor::distributed` holds what DDP is configured with:

| Type                 | Purpose                                                   |
| -------------------- | --------------------------------------------------------- |
| `DistributedContext` | Starts and owns the gradient synchronization of a group   |
| `DistributedConfig`  | Configures gradient aggregation for the group             |
| `ReduceOperation`    | Selects `Sum` or `Mean` reduction                         |

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

## Layer Parallelism

DDP and the multi-device strategy keep a whole copy of the model on each device. When the model does
not fit on one, layer parallelism splits it by whole layers instead: each device holds a run of
consecutive layers, and only what one layer returns moves to the next layer's device.

A model splits by implementing `LayerParallelism`, which names its input layer, its hidden layers and
its output layer, each a `DistributedLayer`. What passes between the layers is any `Module`, such as
a tensor, or a struct carrying a transformer's hidden states and padding mask:

```rust, ignore
impl LayerParallelism for Model {
    type InputLayer = Embedding;
    type HiddenLayer = Block;
    type OutputLayer = Head;

    fn layer_input(&self) -> &Embedding {
        &self.embedding
    }

    fn layers_hidden(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    fn layer_output(&self) -> &Head {
        &self.head
    }

    fn into_layers(self) -> LayerSequence<Self> {
        LayerSequence {
            input: self.embedding,
            hidden: self.blocks,
            output: self.head,
        }
    }
}
```

A `LayerPlacement` gives every layer a device, evenly, by stages of chosen sizes, or by device
memory. Build each layer on its device, then wrap the model: `DistributedLayeredModel::new` moves
nothing, and refuses a layer built anywhere else. The `burn-nn` layers initialize lazily, so each
weight then loads straight onto its layer's device and the model never exists whole on one device.
Weights trained on another struct load by remapping their keys:

```rust, ignore
let placement = LayerPlacement::even(&devices, config.num_blocks);
let model = Model::new(&config, &placement); // Each layer built on `placement`'s device for it.
let mut model = DistributedLayeredModel::new(model, &placement);
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_key_remapping(r"^layers\.", "blocks.");
model.load_from(&mut store)?;

let predictions = model.forward(features);
```

It trains as on one device: build it on autodiff devices, and each gradient lands on its parameter's
device, where the optimizer updates it. The loss runs on the output layer's device.

- **A weight shared by two layers must be on one device.** A head tied to the input embedding only
  splits when the placement puts both on the same device; otherwise `new` refuses it.
- **Split an existing model with `fork`, not `to_device`.** A moved parameter is no longer a leaf, so
  it gets no gradient and the optimizer skips it.

Layers run one after another, so this buys capacity rather than speed. The `layer-parallelism-text`
example splits the transformer the `text-classification` example trains, checks it against the
single-device model, and fine-tunes it through the split.

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
- Use layer parallelism when one copy of the model does not fit on a device, or to run a model
  across devices of different backends.

DDP runs on the CubeCL backends and on remote devices.
CUDA devices with NCCL, local or on one remote server, synchronize gradients natively, and the
others with transfers between devices. Flex and Capture do not run DDP.
