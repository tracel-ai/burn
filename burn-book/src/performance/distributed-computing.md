# Distributed Computing

Burn supports data-parallel training across multiple devices, splitting one model across several
devices, and transparent execution on devices hosted by another process. These capabilities can be
used independently or together:

- The types in `burn::tensor::distributed` provide collective tensor operations across a group of
  devices.
- `burn::train::ExecutionStrategy::ddp` uses those collectives to synchronize gradients during
  distributed data-parallel (DDP) training.
- A remote `Device` sends normal tensor operations to a Burn compute server. A set of remote devices
  can also participate in DDP.
- `burn::module::parallel::LayerParallelism` splits one model by whole layers across several
  devices, so a model too large for one device runs as a sequence of stages.

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

## Layer Parallelism

DDP and the multi-device strategy keep a whole copy of the model on each device. When the model does
not fit on one, layer parallelism splits it by whole layers instead: each device holds a run of
consecutive layers, and what one layer returns moves to the next layer's device.

The split is described by a struct built for it, usually not the one the model trained with, since
a split layer may be represented differently. Each layer implements
`burn::module::parallel::DistributedLayer`, and the model implements `LayerParallelism`, naming its
input layer, its hidden layers and its output layer:

```rust, ignore
impl DistributedLayer for Block {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn forward(&self, hidden: Tensor<2>) -> Tensor<2> {
        relu(self.linear.forward(hidden.clone())) + hidden
    }
}

impl LayerParallelism for Model {
    type InputLayer = Embedding;
    type HiddenLayer = Block;
    type OutputLayer = Head;

    fn layer_input(&self) -> &Embedding {
        &self.embedding
    }

    fn layer_hidden(&self, index: usize) -> Option<&Block> {
        self.blocks.get(index)
    }

    fn layer_output(&self) -> &Head {
        &self.head
    }
}
```

What the input layer returns is what every hidden layer takes and returns, `HiddenLayerSignal`. It
is a module, so it can move between devices: tensors, tuples, arrays, `Vec` and `Option` already
are, and a struct of tensors with a transformer's masks can derive `Module`.

A `LayerPlacement` gives every layer a device, either evenly across some devices or as stages of
chosen sizes, and each layer is built on its device:

```rust, ignore
impl Model {
    pub fn new(config: &ModelConfig, placement: &LayerPlacement) -> Self {
        Self {
            embedding: Embedding::new(config, &placement.input),
            blocks: placement.hidden.iter().map(|device| Block::new(config, device)).collect(),
            head: Head::new(config, &placement.output),
        }
    }
}
```

`DistributedLayeredModel::new` moves nothing. It checks the placement has a device for every hidden
layer and that every layer's parameters and held tensors are on that device. The result runs each
layer where it is, dereferences to the model and is itself a `Module`.

The `burn-nn` layers built from a config initialize lazily, so the model allocates nothing until its
weights load, and each weight then loads onto its own layer's device. The model never lands whole on
one device, which is how one too large for a device loads. Weights trained on another struct load by
remapping their keys:

```rust, ignore
let placement = LayerPlacement::even(&devices, config.num_blocks);
let mut model = DistributedLayeredModel::new(Model::new(&config, &placement), &placement);
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_key_remapping(r"^layers\.", "blocks.");
model.load_from(&mut store)?;
```

Two things to watch:

- **A weight shared by two layers is on one device.** An output head tied to the input embedding
  can only be split when the placement puts both on the same device; otherwise `new` refuses it.
- **A model already built moves with `fork`, not `to_device`.** To split one that exists on a single
  device, fork each layer onto its device before `new`. A moved parameter is no longer a leaf, so it
  gets no gradient and the optimizer skips it.

Layers run one after another, so what this buys today is capacity rather than speed: each device
waits for the one before it, and every move costs the size of the signal. Overlapping them takes a
schedule that splits a batch into microbatches, such as GPipe or 1F1B, which is not implemented yet.
The `layer-parallelism` example splits a small model across one device per GPU, loading a
checkpoint of its single-device shape, and checks the predictions against that shape.

### Training a Split Model

A split model trains as it would on one device. Every parameter is built on its layer's device, so
its gradient lands there and the optimizer updates it there. Build the model on autodiff devices,
and put the targets on the output layer's device, where the loss runs:

```rust, ignore
let devices = [Device::cuda(0).autodiff(), Device::cuda(1).autodiff()];
let placement = LayerPlacement::even(&devices, 8);
let mut model = DistributedLayeredModel::new(Model::new(&config, &placement), &placement);

let predictions = model.forward(features);
let loss = loss_fn.forward(predictions, targets.to_device(&placement.output), Reduction::Mean);
let grads = GradientsParams::from_grads(loss.backward(), &model);
model = optim.step(lr, model, grads);
```

## Remote Devices

A remote device implements the same `Device` interface as a local CUDA, WGPU, or CPU device. Tensor
creation and operations use the normal API, but execution happens on a device exposed by a Burn
server:

```rust, ignore
let device = Device::remote_websocket("ws://localhost:3000", 0);
let tensor = Tensor::<2>::ones([32, 128], &device);
let output = model.to_device(&device).forward(tensor);
```

WebSocket remote devices are retained for existing deployments. New native integrations should
prefer the Iroh transport, which identifies a server by its peer identity instead of requiring a
fixed WebSocket address. The server exposes a local device with `Channel::Iroh` and a
`RemoteSecret`; clients connect through an Iroh endpoint and receive the same unified `Device`:

```rust, ignore
let transport = QuicTransportConfig::builder()
    .enable_segmentation_offload(false)
    .build();
let endpoint = Endpoint::builder(presets::N0)
    .transport_config(transport)
    .bind()
    .await?;
let device = Device::remote_iroh(&endpoint, server_id, 0);

let tensor = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
let output = tensor.square().sum(); // Executed by the remote server.
```

Segmentation offload (GSO) is off because of an Iroh bug
([iroh#4555](https://github.com/n0-computer/iroh/issues/4555)). On Linux before 6.11, a network card
without TX checksum offload, such as most MediaTek wifi cards, refuses GSO sends, and Iroh keeps
sending them on connections that are already open until those connections time out. The built-in
server turns GSO off for the same reason.

A system should generate a random `RemoteSecret` and distribute its public identity through a
trusted channel. `Device::remote_iroh_authorized` also sends an application-defined credential to
servers that enforce peer authorization. Async constructors are available for browser targets, where
a synchronous connection cannot be established.

### DDP on Remote Devices

Remote execution and DDP compose naturally. The
[`text-classification` example](https://github.com/tracel-ai/burn/tree/main/examples/text-classification/examples/ag-news-train.rs)
enumerates every device hosted by a remote WebSocket server and passes them to the same DDP
strategy:

```rust, ignore
pub fn run() {
    let devices = Device::enumerate(DeviceType::remote_websocket(ADDRESS));

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

Distributed execution assumes that participating devices support the required collective operations.
