# LibTorch Setup

The `burn-tch` backend relies on the C++ PyTorch runtime (LibTorch).

## Feature behavior

- `burn` with `tch` expects LibTorch to be available on your system.
- `burn` with `tch,download-libtorch` asks `tch` to download a compatible LibTorch distribution during build.
- Without `download-libtorch`, the `tch` crate does not automatically download a CPU LibTorch distribution.

```console
cargo add burn --features tch
cargo add burn --features "tch,download-libtorch"
```

## Manual setup options

If you use `tch` without `download-libtorch`, provide LibTorch manually with one of these common options:

- Set `LIBTORCH` to a local LibTorch installation.
- Set `LIBTORCH_USE_PYTORCH=1` to reuse a Python PyTorch installation.

For guaranteed compatibility, use LibTorch/PyTorch v2.9.0 with `tch` v0.22.0. To try another
version anyway, set `LIBTORCH_BYPASS_VERSION_CHECK=1`.

### Linux shell examples (`bash`)

You can pass `LIBTORCH_USE_PYTORCH` for a single command:

```sh
LIBTORCH_USE_PYTORCH=1 cargo run
```

You can also export it, then run commands normally:

```sh
export LIBTORCH_USE_PYTORCH=1
cargo run
```

Equivalent split form:

```sh
LIBTORCH_USE_PYTORCH=1
export LIBTORCH_USE_PYTORCH
cargo run
```

Why this does **not** work:

```sh
LIBTORCH_USE_PYTORCH=1
cargo run
```

In this form, the first line only creates a shell variable. It is not exported to child processes,
so `cargo run` does not receive `LIBTORCH_USE_PYTORCH`.

For more setup options, see the environment variables accepted by `tch`/`torch-sys`.
