# Distributed Computing

Burn supports data-parallel training across multiple devices and transparent execution on devices
hosted by another process. These capabilities can be used independently or together:

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

DDP copies the whole model onto every device. When the model does not fit on one device, pipeline
parallelism cuts it by whole layers instead: each device holds a run of consecutive layers, a stage,
and the activations move from one device to the next. A model opts in by implementing
`burn::module::pipeline::Pipeline`, which splits its forward pass into segments: `forward_input`,
then `forward_block` for each block in order, then `forward_output`.

The input and the activations passed between segments are modules, so they can move to another
device: tensors, tuples of tensors, arrays, `Vec` and `Option` already are, and a struct of tensors
can derive `Module`. The model also states which submodules each segment runs, in a
`PipelineLayout`. A transformer, for instance, passes its hidden state along with its masks:

```rust, ignore
impl Pipeline for Transformer {
    type Input = (Tensor<2, Int>, Tensor<2, Bool>);
    type Output = Tensor<3>;
    type Activations = (Tensor<3>, Tensor<2, Bool>, Tensor<3, Bool>);

    fn layout(&self) -> PipelineLayout {
        PipelineLayout::new()
            .input(&self.embedding_token)
            .input(&self.embedding_pos)
            .blocks(&self.layers)
            .output(&self.output)
    }

    fn forward_input(&self, (tokens, mask_pad): Self::Input) -> Self::Activations { /* embeddings and masks */ }
    fn forward_block(&self, index: usize, activations: Self::Activations) -> Self::Activations { /* one layer */ }
    fn forward_output(&self, activations: Self::Activations) -> Self::Output { /* output projection */ }
}
```

A `StageMap` gives every segment a device, usually built from stages. `place` forks each parameter
onto the device of the segment that owns it, and `forward_on` runs the segments on their devices,
moving the input and the activations along. The model keeps its type, so the optimizer, records and
checkpoints work as they do on one device:

```rust, ignore
let stages = StageMap::new(&[
    Stage { device: Device::cuda(0).autodiff(), blocks: 6 },
    Stage { device: Device::cuda(1).autodiff(), blocks: 6 },
]);
let model = model.place(&stages);

let logits = model.forward_on(&stages, (tokens, mask_pad));
let loss = loss_fn.forward(logits.flatten(0, 1), targets.flatten(0, 1));
let grads = GradientsParams::from_grads(loss.backward(), &model);
let model = optim.step(lr, model, grads);
```

A parameter that has not initialized yet is not copied: it takes its segment's device and
initializes there on first use. The `burn-nn` layers built from a config initialize lazily, so a
model built on any device never lands on it whole, and weights loaded after `place` go straight to
their segment's device. That is how a model too large for one device loads:

```rust, ignore
let model = ModelConfig::new().init(&device).place(&stages).load_record(record);
```

Inside the segments, create any tensor on the device of the tensors you were handed rather than on a
device of the model: once placed, the model has several.

Every parameter belongs to exactly one segment: `PipelineLayout` refuses a submodule claimed by two
segments, and `place` refuses a parameter no segment claims. A segment that reads a parameter another
segment owns, such as an output head tied to the input embedding, moves it to its own device, and the
gradient flows back to the owner:

```rust, ignore
// In a model whose activations are the hidden state alone.
fn forward_output(&self, hidden: Tensor<3>) -> Tensor<3> {
    let embedding = self.embedding.weight.val().to_device(&hidden.device());
    hidden.matmul(embedding.transpose().unsqueeze())
}
```

Stages run one after another, so pipeline parallelism buys capacity, not speed: each device waits for
the previous one, and every move between devices costs the size of the activations.

The `pipeline-parallel` example trains a small model split this way across one device per GPU.

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
