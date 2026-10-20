# Profiling

`burn::instrument` breaks a pass down by component: which parts of a model the device spent its
time in, which kernels each one launched, and how far each is from what the hardware allows. It
measures the graph a build actually runs. Nothing is drained at a component boundary, so with
fusion on, the kernels it times are the fused ones a clean pass launches.

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

With no collection running, an annotated method costs one thread-local read.

## Collecting a pass

Collection needs a CubeCL runtime and, on a fused build, burn's `fusion` feature. Collect a pass
the device has already run, so autotuning and compilation are not part of it:

```rust,ignore
use burn::instrument::{CollectedPass, PassCollector, Peaks};

// Warm up first: the first pass compiles and autotunes.
let peaks = Peaks::measure(&device, FloatDType::F32);
let clean_wall = /* the median wall time of a few uncollected passes */;

let mut output = None;
let breakdown = PassCollector::new(&device).collect(|| output = Some(model.forward(input.clone())));

let pass = CollectedPass { device: format!("{device:?}"), peaks: Some(peaks), clean_wall: Some(clean_wall), breakdown };
pass.write(std::fs::File::create("pass.json")?)?;
```

Measure the peaks before the pass: the probes are saturating kernels that move clocks and
thermals for whatever runs after them. The clean wall is what the report splits into device time
and the time the device waited on the host. A collected pass times every launch, so its own wall
time describes the instrumentation, not the model. Hold the outputs past the collection, because an
output dropped inside the pass lets the fuser skip the work that wrote it.

The `instrument` example does all of this for a small model:

```sh
cargo run --release -p instrument --features cuda -- pass.json
```

## Reading the report

```sh
cargo run --release -p burn-instrument-report -- pass.json
```

The report reads the file on any machine, with or without a GPU. It prints:

- **the device and its peaks**, and anything a probe reported that cannot be true, which is then
  left out of every ratio
- **how the pass ran**: fused or unfused, and whether the kernels were timed in place or one at a
  time
- **the component tree**, each component with its time, its share of the pass, the time the device
  waited before its kernels, the wall it is bound by (`bytes` or `flops`), how close it got to that
  wall's peak, its floor, and what it could still give back. A kernel that carried out several
  components' work is listed on their closest common parent and names them (`· fuses ↑`), and a
  component whose work all ran in such a kernel reads `fused ↓`.
- **the ladder**: of the pass's time, what faster kernels could recover and what fusing inside
  components could recover past that
- **a worklist**, ranked by what each lever recovers, counting each millisecond once

Hold a pass against a saved one with `--against`:

```sh
cargo run --release -p burn-instrument-report -- pass.json --against before.json
```

Components are matched by their path from the root, and each delta carries the executions that
served it. A component whose work fused into another's in one pass and not the other is named
rather than subtracted. Two passes on a different device, graph or timing are announced before any
number.
