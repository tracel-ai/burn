# Burn Benchmark

This crate allows to compare backend computation times, from tensor operations
to complex models.

Note: in order to compare different backend-specific tensor operation
implementations (for autotuning purposes, for instance), this should be done
within the corresponding backend crate.

## burnbench CLI

This crate comes with a CLI binary called `burnbench` which can be executed via
`cargo run --bin burnbench`.

The end of options argument `--` is used to pass arguments to the `burnbench`
application. For instance `cargo run --bin burnbench -- list` passes the `list`
argument to `burnbench` effectively calling `burnbench list`.

To list all the available benches and backends use the `list` command:

```sh
> cargo run --bin burnbench -- list
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

Available Benchmarks:
- binary
- custom-gelu
- data
- matmul
- unary
```

To execute a given benchmark against a specific backend we use the `run` command
with the arguments `--bench` and `--backend` respectively. In the following
example we execute the `unary` benchmark against the `wgpu-fusion` backend:

```sh
> cargo run --bin burnbench -- run --bench unary --backend wgpu-fusion
```

Shorthands can be used, the following command line is the same:

```sh
> cargo run --bin burnbench -- run -b unary -B wgpu-fusion
```

Multiple benchmarks and backends can be passed on the same command line. In this
case, all the combinations of benchmarks with backends will be executed.

```sh
> cargo run --bin burnbench -- run --bench unary --bench binary --backend wgpu-fusion --backend tch-gpu
    Finished dev [unoptimized] target(s) in 0.09s
     Running `target/debug/burnbench run --bench unary --bench binary --backend wgpu-fusion --backend wgpu`
Executing the following benchmark and backend combinations (Total: 4):
- Benchmark: unary, Backend: wgpu-fusion
- Benchmark: binary, Backend: wgpu-fusion
- Benchmark: unary, Backend: tch-gpu
- Benchmark: binary, Backend: tch-gpu
Running benchmarks...
```

### Terminal UI

This is a work in progress.

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
pub(crate) enum BackendValues {
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

