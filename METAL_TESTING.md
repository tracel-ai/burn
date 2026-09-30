# Testing device-error handling on Metal

These branches change how Burn and CubeCL handle GPU errors: a failed kernel or a
crashed GPU should come back as an error the program can handle, and training should
stop cleanly with the cause, instead of panicking. It has been tested on CUDA, wgpu
(Vulkan) and AMD, but not on a Mac. Please run the three parts below on an Apple
Silicon Mac and send back the outputs listed at the end of each part.

Everything needs a recent stable Rust toolchain (1.95 or newer).

## 1. Fault injection example (Burn)

```sh
git clone https://github.com/tracel-ai/burn && cd burn
git checkout feat/sync-point-errors-fault-example
cargo build -p mnist --example mnist-fault --release --features metal,fault-injection
```

The first build takes a while. The first run downloads MNIST (about 12 MB).

Run each scenario, saving everything it prints:

```sh
B=./target/release/examples/mnist-fault
$B inference refused 5  > inference-refused.txt  2>&1
$B inference poisoned 5 > inference-poisoned.txt 2>&1
$B training refused 20  > training-refused.txt   2>&1; cp /tmp/burn-example-mnist/experiment.log training-refused.log
$B training poisoned 20 > training-poisoned.txt  2>&1; cp /tmp/burn-example-mnist/experiment.log training-poisoned.log
```

What each scenario does:

- **`refused`**: the kernel is rejected by the compiler before it runs. The GPU stays
  healthy.
- **`poisoned`**: the kernel writes 1 GB past the end of its buffer, which on CUDA
  crashes the GPU for the rest of the process.

What we expect:

| Scenario | Expected result |
|---|---|
| `inference refused 5` | Batch 5 prints `failed, retrying it.`, the retry succeeds, and the run ends with `done: every batch was classified` (exit code 0). |
| `inference poisoned 5` | Unknown on Metal. The Metal backend may block the out-of-bounds write, in which case every batch succeeds. If the GPU does fault, we want to see how it is reported. |
| `training refused 20` | Training stops a few steps after step 20. `experiment.log` contains `Stopped by an error: Event processing failed ...`. The program then panics while saving the model: that part is expected. |
| `training poisoned 20` | Same as above if the GPU faults; otherwise training runs normally. You can stop it with Ctrl-C after a minute or so. |

**Please send:** the four `.txt` files and the two `.log` files. The `.log` files can
be large; if so, `grep -E "ERROR|WARN|Stopped by" training-*.log` is enough.

## 2. CubeCL tests

```sh
git clone https://github.com/tracel-ai/cubecl && cd cubecl
git checkout feat/sync-point-errors
cargo test -p cubecl-metal          > cubecl-metal.txt 2>&1
cargo test -p cubecl-wgpu --features msl > cubecl-wgpu-msl.txt 2>&1
cargo test -p cubecl-server -p cubecl-runtime > cubecl-server.txt 2>&1
```

- `cubecl-metal` is the native Metal runtime. Part of this branch's changes to it
  could not be compiled on Linux, so a compile error here is a real finding.
- `cubecl-wgpu --features msl` is the runtime Burn uses on a Mac.

**Please send:** the three `.txt` files.

## 3. Burn tests

In the `burn` checkout from part 1:

```sh
cargo xtask test --ci github-mac-runner > burn-mac.txt 2>&1
cargo test -p burn-train --all-features > burn-train.txt 2>&1
```

The first command runs the same Metal suite as Burn's macOS CI, and can take a long
time.

**Please send:** both `.txt` files, or the summary lines at the end of each if they
are too large (`grep -E "test result|FAILED|panicked" burn-mac.txt`).
