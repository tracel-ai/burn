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
- `burn::module::pipeline::Pipeline` splits one model by whole layers across several devices, so a
  model too large for one device runs as a sequence of stages.

## Distributed Tensor Operations

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

## Pipeline Parallelism

DDP and the multi-device strategy keep a whole copy of the model on each device. When the model does
not fit on one device, pipeline parallelism cuts it by whole layers instead: each device holds a run
of consecutive layers, a stage, and the activations move from one device to the next. A model opts
in by implementing `burn::module::pipeline::Pipeline`, which splits its forward pass into segments:
`forward_input`, then `forward_block` for each block in order, then `forward_output`. The model also
states which submodules each segment runs, in a `PipelineLayout`:

```rust, ignore
impl Pipeline for Model {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type Carry = Tensor<2>;

    fn layout(&self) -> PipelineLayout {
        PipelineLayout::new()
            .input(&self.input)
            .blocks(&self.blocks)
            .output(&self.output)
    }

    fn forward_input(&self, features: Tensor<2>) -> Tensor<2> {
        relu(self.input.forward(features))
    }

    fn forward_block(&self, index: usize, hidden: Tensor<2>) -> Tensor<2> {
        relu(self.blocks[index].forward(hidden.clone())) + hidden
    }

    fn forward_output(&self, hidden: Tensor<2>) -> Tensor<2> {
        self.output.forward(hidden)
    }
}
```

What a segment passes to the next is its `Carry`: the activations, and whatever rides along with
them. A transformer, for instance, passes its hidden state along with its masks. The input and the
carry are modules, so they can move to another device: tensors, tuples of tensors, arrays, `Vec` and
`Option` already are, and a struct of tensors can derive `Module`.

A `PipelinePlacement` gives every segment a device: `PipelinePlacement::even` shares the blocks out
in order across some devices, and `PipelinePlacement::new` takes stages of chosen sizes. `place`
forks each parameter onto the device of the segment that owns it and returns a `PlacedPipeline`,
whose `forward` runs each segment where its own parameters ended up, moving the input and the
carry along. The placement is named once, at `place`, so the carry can never be sent somewhere the
weights are not, and where the segments run is resolved once rather than on every forward. A
`PlacedPipeline` dereferences to the model and is itself a `Module`, so records and the model's own
methods work as they do on one device:

```rust, ignore
// One device per card: `enumerate_physical` returns each card once, with every runtime that
// reaches it, so a GPU both CUDA and Vulkan see is not used twice.
let devices: Vec<Device> = Device::enumerate_physical()
    .into_iter()
    .map(|gpu| gpu.devices[0].clone())
    .collect();

let placement = PipelinePlacement::even(&devices, 8);
let model = model.place(&placement);

let predictions = model.forward(features);
```

A parameter that has not initialized yet is not copied: it takes its segment's device and
initializes there on first use. The `burn-nn` layers built from a config initialize lazily, so a
model built on any device never lands on it whole, and weights loaded after `place` go straight to
their segment's device. That is how a model too large for one device loads:

```rust, ignore
let model = ModelConfig::new().init(&device).place(&placement).load_record(record);
```

A lazy parameter that a live clone of the model shares is the exception: it initializes where it is
and is then copied, since every clone must see one value. Drop other clones before placing.

Only parameters move. A tensor a module holds directly, such as a precomputed positional table,
stays where the model was built, so the segment that reads it moves it to its own device. Likewise,
create any tensor inside a segment on the device of the tensors you were handed rather than on a
device of the model: once placed, the model has several.

Every parameter belongs to exactly one segment: `PipelineLayout` refuses a submodule claimed by two
segments, and `place` refuses a parameter no segment claims. A segment that reads a parameter
another segment owns, such as an output head tied to the input embedding, moves it to its own
device:

```rust, ignore
// In a model whose carry is the hidden state alone.
fn forward_output(&self, hidden: Tensor<3>) -> Tensor<3> {
    let embedding = self.embedding.weight.val().to_device(&hidden.device());
    hidden.matmul(embedding.transpose().unsqueeze())
}
```

That copies the whole embedding on every forward pass, so every generated token. When the parameter
does not train, put the output segment on the input segment's device instead:

```rust, ignore
let placement = PipelinePlacement { output: placement.input.clone(), ..placement };
```

Stages run one after another, so what this buys today is capacity rather than speed: each device
waits for the previous one, and every move between devices costs the size of the carry. Overlapping
the stages takes a schedule that splits a batch into microbatches, such as GPipe or 1F1B; the layout
and placement here are what such a schedule would drive, but none is implemented yet.

The `pipeline` example splits a small model this way across one device per GPU, loads its weights
onto each stage, and checks its predictions against the same model on one device.

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
let endpoint = Endpoint::builder(presets::N0).bind().await?;
let device = Device::remote_iroh(&endpoint, server_id, 0);

let tensor = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
let output = tensor.square().sum(); // Executed by the remote server.
```

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
- Use a pipeline when one copy of the model does not fit on a device, or to run a model across
  devices of different backends.

Distributed execution assumes that participating devices support the required collective operations.
