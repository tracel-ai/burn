# Profiling

`burn::instrument` breaks a run of a model down by component: which parts of the model the device
spent its time in, which kernels each one launched, and how far each is from what the hardware
allows. It measures the graph a build actually runs. Nothing is drained at a component boundary, so
with fusion on, the kernels it times are the fused ones a clean run launches.

## Declaring components

Add `#[compute]` to a method to include it in the profile. Annotated calls form a component tree.
This works for custom modules built from layers or tensor operations, without defining bounds.

Enable the `instrument` feature on your `burn` dependency to activate the annotations.
Without it, they compile away.

### Custom modules

The classifier below uses named methods to keep its two `Linear` layers separate in the report.
Without those labels, both calls would appear as one `linear` component under `classifier`.

```rust,ignore
use burn::instrument::compute;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, activation::relu};

#[derive(Module, Debug)]
struct Classifier {
    hidden: Linear,
    output: Linear,
}

impl Classifier {
    fn new(device: &Device) -> Self {
        Self {
            hidden: LinearConfig::new(32, 64).init(device),
            output: LinearConfig::new(64, 10).init(device),
        }
    }

    #[compute]
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let hidden = relu(self.project_hidden(input));
        self.project_output(hidden)
    }

    #[compute(label = "hidden projection")]
    fn project_hidden(&self, input: Tensor<2>) -> Tensor<2> {
        self.hidden.forward(input)
    }

    #[compute(label = "output projection")]
    fn project_output(&self, input: Tensor<2>) -> Tensor<2> {
        self.output.forward(input)
    }
}
```

Calling `model.forward(input)` during a measurement produces this tree:

```text
classifier
├── hidden projection
│   └── linear
└── output projection
    └── linear
```

The linear layers already define their bounds. The `relu` call has no annotation, so its work
belongs to the classifier. When a kernel fuses work from several components, its time is counted
under their closest common parent.

### The layers of `burn::nn`

Built-in layers are already annotated. Linear, embedding, normalization, convolution, pooling,
interpolation and positional encoding layers declare bounds. Feed-forward blocks and transformer
layers and stacks also declare bounds for the whole component, so the report can show how much
time fusion could save.

Attention and recurrent layers declare bounds for their parts. They use labels to distinguish
layers of the same type, such as an attention's query and key projections.

### Component options and bounds

Bounds describe the minimum traffic and arithmetic for a component running alone.
This projection assumes floating-point inputs and weights of the same dtype:

```rust,ignore
use burn::instrument::{Bounds, compute, tensor_bytes};
use burn::tensor::Tensor;

struct Projection {
    role: &'static str,
    weight: Tensor<2>,
}

impl Projection {
    #[compute(label = self.role, bounds = self.bounds(&x))]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.weight.clone())
    }

    /// Read the input and the weight once, write the output once.
    fn bounds(&self, x: &Tensor<2>) -> Bounds {
        let [rows, input] = x.dims();
        let [_, output] = self.weight.dims();
        let element = x.dtype().size() as u64;
        Bounds {
            read: tensor_bytes(x) + tensor_bytes(&self.weight),
            write: (rows * output) as u64 * element,
            flops: 2 * (rows * input * output) as u64,
        }
    }
}
```

- **`label`** names the component. It defaults to the type's name in words (`RmsNorm` becomes
  `rms norm`). Components with the same label under the same parent merge.
- **`variant`** names the execution chosen at run time. The report counts calls per variant.
- **`bounds`** is the floor for running the component alone: the bytes it must read and write
  through global memory at the widths its tensors are stored in, and its multiply-add FLOPs. Leave
  it off for a component that will be fused into its neighbour. A floor nothing can reach is worse
  than none.

## Measuring a model

To measure the classifier on CUDA, enable `instrument` and `cuda` on your `burn` dependency
and add this entry point. If your backend uses fusion, enable `fusion` too.
You can use another CubeCL backend by changing the backend feature and device constructor.

```rust,ignore
use burn::instrument::Instrument;
use burn::tensor::DeviceIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda(DeviceIndex::Default);
    let model = Classifier::new(&device);
    let input = Tensor::<2>::ones([128, 32], &device);

    let measurement = Instrument::new(&device).measure(|| model.forward(input.clone()));
    measurement.write(std::fs::File::create("model.json")?)?;
    Ok(())
}
```

`measure` runs the closure several times, holding each output until its run has ended, since an
output dropped inside the run lets the fuser skip the work that wrote it:

- **warmup** (5 runs by default, `.warmup(n)`): the first run compiles and autotunes, which is
  not part of the model's time.
- **peaks**, probed before anything is timed, at the device's default float dtype (`.dtype(..)`):
  the probes are saturating kernels that move clocks and thermals for whatever runs after them.
- **clean wall**, the median of 10 clean runs (`.samples(n)`): what the report splits into device
  time and the time the device waited on the host. An instrumented run times every launch, so its
  own wall time describes the instrumentation, not the model.
- **pricing**, one untimed run that evaluates component bounds, including any value readbacks.
- **the instrumented run**, whose breakdown the file holds.

### Replay requirements

The pricing and instrumented runs must enter the same annotated components in the same order,
with equivalent shapes and values wherever they affect bounds. Reuse the same inputs and ensure
relevant model state is equivalent for each invocation. Advancing a batch iterator, growing a
KV cache, or changing routing decisions can invalidate the result.

The device is reseeded before pricing and timing. This can repeat device random draws, but does
not reset application state, model state, or host-side random generators. Bounds calculations
must also preserve state and randomness needed by the forward pass. The instrument does not
check that the runs match: a mismatch can silently assign bounds to the wrong components or
score a component using bounds calculated for different inputs.

Read a batch before profiling, rather than advancing the iterator inside the measured closure:

```rust,ignore
let batch = batches.next().unwrap();
let measurement = Instrument::new(&device).measure(|| model.forward(batch.clone()));
```

The model must also satisfy the state requirements above. `Instrument::breakdown` runs the
closure twice, for pricing and timing, and has the same replay requirements.

### Profiling the text classifier

The text-classification `profile` example measures the AG News classifier and prints the report.
Pass a path to save the measurement as well:

```sh
cargo run --release -p text-classification --example profile --features cuda,profile -- model.json
```

## Reading the report

`burn_instrument_report::Report` renders a measurement as the report, and `burn-instrument-report`
prints a saved one:

```sh
cargo run --release -p burn-instrument-report -- model.json
```

The report reads the file on any machine, with or without a GPU. It prints:

- **the device and its peaks**, and anything a probe reported that cannot be true, which is then
  left out of every ratio
- **how it ran**: fused or unfused, and whether the kernels were timed in place or one at a
  time
- **the component tree**, each component with its time, its share of the run, the time the device
  waited before its kernels, the wall it is bound by (`bytes` or `flops`), how close it got to that
  wall's peak, its floor, and what it could still give back. A kernel that carried out several
  components' work is listed on their closest common parent and names them (`· fuses ↑`), and a
  component whose work all ran in such a kernel reads `fused ↓`.
- **the ladder**: of the run's time, what faster kernels could recover and what fusing inside
  components could recover past that
- **a worklist**, ranked by what each lever recovers, counting each millisecond once

Hold a measurement against a saved one with `--against`:

```sh
cargo run --release -p burn-instrument-report -- model.json --against before.json
```

Components are matched by their path from the root, and each delta carries the executions that
served it. A component whose work fused into another's in one measurement and not the other is
named rather than subtracted. Two measurements on a different device, graph or timing are announced
before any number.
