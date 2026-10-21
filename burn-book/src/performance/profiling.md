# Profiling

`burn::instrument` breaks a run of a model down by component: which parts of the model the device
spent its time in, which kernels each one launched, and how far each is from what the hardware
allows. It measures the graph a build actually runs. Nothing is drained at a component boundary, so
with fusion on, the kernels it times are the fused ones a clean run launches.

## Declaring components

A component is a method annotated with `#[compute]`. Every call opens a tag that names it, and
components nest the way the methods call each other:

```rust,ignore
use burn::instrument::{Bounds, compute, tensor_bytes};

impl Projection {
    #[compute(label = self.role, bounds = self.bounds(&x))]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.weight.clone())
    }

    /// Read the input and the weight once, write the output once.
    fn bounds(&self, x: &Tensor<2>) -> Bounds {
        let [rows, input] = x.dims();
        let [_, output] = self.weight.dims();
        let element = tensor_bytes(x) / (rows * input) as u64;
        Bounds {
            read: tensor_bytes(x) + tensor_bytes(&self.weight),
            write: (rows * output) as u64 * element,
            flops: 2 * (rows * input * output) as u64,
        }
    }
}
```

- **`label`** names the component. It defaults to the type's name in words (`RmsNorm` is
  `rms norm`). Components merge on their label among siblings, so a type used in several roles
  under one parent, like the projections of a block, needs one label per role.
- **`variant`** names which execution served a call, where that is chosen at run time. The report
  prints how often each one ran.
- **`bounds`** is the floor for running the component alone: the bytes it must read and write
  through global memory at the widths its tensors are stored in, and its multiply-add FLOPs. Leave
  it off for a component that will be fused into its neighbour. A floor nothing can reach is worse
  than none.

The annotations are armed by burn's `instrument` feature. Without it they compile away, so a library
can annotate its components at no cost to a build that never measures; with it and no measurement
running, an annotated method costs one thread-local read.

### The layers of `burn::nn`

The layers in `burn::nn` are components already, so a model built from them is broken down without
annotating anything. The layers that run one operation (linear, embedding, the norms, the
convolutions, the pools, the interpolations and the positional encodings) declare their floors.
The feed-forward and the transformer's encoder and decoder, their layers and their stacks are priced
whole as well, so the report can say what fusing inside them would recover. Attention and the
recurrent layers are timed, and their parts are priced: their projections and gates, and the scaled
dot product inside each attention. Where one
type serves several roles in a layer, each runs under its role, so an attention's `query` and `key`
projections are two components and not one `linear` entered twice. Annotate your own modules to
name what composes them.

## Measuring a model

Measuring needs burn's `instrument` feature, a CubeCL runtime and, on a fused build, burn's `fusion`
feature:

```rust,ignore
use burn::instrument::Instrument;

let measurement = Instrument::new(&device).measure(|| model.forward(input.clone()));
measurement.write(std::fs::File::create("model.json")?)?;
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
- **the instrumented run**, whose breakdown the file holds.

The text-classification example's `profile` does all of this for the AG News classifier, a
transformer encoder built from `burn::nn`, and prints the report; with a path, it saves the
measurement there too:

```sh
cargo run --release -p text-classification --example profile --features cuda -- model.json
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
