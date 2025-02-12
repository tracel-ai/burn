# Burn Benchmark

This crate allows to compare backend computation times, from tensor operations
to complex models.

## burnbench CLI

This crate comes with a CLI binary called `burnbench` which can be executed via
`cargo run --release --bin burnbench`.

Note that you need to run the `release` target of `burnbench` otherwise you won't
be able to share your benchmark results.

The end of options argument `--` is used to pass arguments to the `burnbench`
application. For instance `cargo run --bin burnbench -- list` passes the `list`
argument to `burnbench` effectively calling `burnbench list`.

There is also a cargo alias `cargo bb` which simplifies the command line.
The example command above then becomes: `cargo bb list`.

### Commands

#### List benches and backends

To list all the available benches and backends use the `list` command:

```sh
> cargo run --release --bin burnbench -- list
    Finished dev [unoptimized] target(s) in 0.10s
     Running `target/debug/burnbench list`
Available Backends:
- candle-cpu
- candle-cuda
- candle-metal
- ndarray
- ndarray-blas-accelerate
- ndarray-blas-netlib
- ndarray-blas-openblas
- tch-cpu
- tch-gpu
- wgpu
- wgpu-fusion
- wgpu-spirv
- wgpu-spirv-fusion

Available Benchmarks:
- binary
- custom-gelu
- data
- matmul
- unary
- max-pool2d
- resnet50
- load-record
- autodiff
- conv-transpose2d
- conv-transpose3d
- conv2d
- conv3d
- reduce
```

#### Run benchmarks

To run a given benchmark against a specific backend we use the `run` command
with the arguments `--benches` and `--backends` respectively. In the following
example we execute the `unary` benchmark against the `wgpu-fusion` backend:

```sh
> cargo run --release --bin burnbench -- run --benches unary --backends wgpu-fusion
```

Shorthands can be used, the following command line is the same:

```sh
> cargo run --release --bin burnbench -- run -b unary -B wgpu-fusion
```

Multiple benchmarks and backends can be passed on the same command line. In this
case, all the combinations of benchmarks with backends will be executed.

```sh
> cargo run --bin burnbench -- run --benches unary binary --backends wgpu-fusion tch-gpu
     Running `target/release/burnbench run --benches unary binary --backends wgpu-fusion wgpu`
Executing the following benchmark and backend combinations (Total: 4):
- Benchmark: unary, Backend: wgpu-fusion
- Benchmark: binary, Backend: wgpu-fusion
- Benchmark: unary, Backend: tch-gpu
- Benchmark: binary, Backend: tch-gpu
Running benchmarks...
```

By default `burnbench` uses a compact output with a progress bar which hides the
compilation logs and benchmarks results as they are executed. If a benchmark
failed to run, the `--verbose` flag can be used to investigate the error.

#### Authentication and benchmarks sharing

Burnbench can upload benchmark results to our servers so that users can share
their results with the community and we can use this information to drive the
development of Burn. The results can be explored on [Burn website][1].

Sharing results is opt-in and it is enabled with the `--share` arguments passed
to the `run` command:

```sh
> cargo run --release --bin burnbench -- run --share --benches unary --backends wgpu-fusion
```

To be able to upload results you must be authenticated. We only support GitHub
authentication. To authenticate run the `auth` command, then follow the URL
to enter your device code and authorize the Burnbench application:

```sh
> cargo run --release --bin burnbench -- auth
```

If everything is fine you should get a confirmation in the terminal that your
token has been saved to the burn cache directory.

We don't store any of your personal information. An anonymized user name will
be attributed to you and displayed in the terminal once you are authenticated.
For instance:

```
🔑 Your username is: CuteFlame
```

You can now use the `--share` argument to upload and share your benchmarks.
A URL to the results will displayed at the end of the report table.

Note that your access token will be refreshed automatically so you should not
need to reauthorize the application again except if your refresh token itself
becomes invalid.

## Execute benchmarks with cargo

To execute a benchmark against a given backend using only cargo is done with the
`bench` command. In this case the backend is a feature of this crate.

```sh
> cargo bench --features wgpu-fusion
```

## Add a new benchmark

To add a new benchmark it must be first declared in the `Cargo.toml` file of this
crate:

```toml
[[bench]]
name = "mybench"
harness = false
```

Then it must be registered in the `BenchmarkValues` enumeration:

```rs

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
pub(crate) enum BenchmarkValues {
    // ...
    #[strum(to_string = "mybench")]
    MyBench,
    // ...
}
```

Create a new file `mybench.rs` in the `benches` directory and implement the
`Benchmark` trait over your benchmark structure. Then implement the `bench`
function. At last call the macro `backend_comparison::bench_on_backend!()` in
the `main` function.

## Add a new backend

You can easily register and new backend in the `BackendValues` enumeration:

```rs

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
pub(crate) enum BackendValues {
    // ...
    #[strum(to_string = "mybackend")]
    MyBackend,
    // ...
}
```

Then update the macro `bench_on_backend` to support the newly registered
backend.

[1]: https://burn.dev/benchmarks/community-benchmarks
