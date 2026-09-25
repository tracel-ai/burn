# Runtime Configuration

Burn uses `burn.toml` to configure logging, operation fusion, and remote-backend batching.
For CubeCL backends, the same file can also configure autotuning, profiling, compilation,
and memory management through `[cubecl.*]` sections.

## Overview

On platforms with filesystem support and the `std` feature enabled, Burn looks for `burn.toml`
or `Burn.toml` in the process's current working directory, then searches parent directories.
The first successfully parsed file is used; settings from multiple files are not merged.
Missing fields use their defaults, and if no valid file is found, all settings use defaults.
Malformed files produce a warning and are skipped.

Configuration is loaded on first use and kept for the lifetime of the process. Set environment
variables before starting the application, and restart it after editing the file. Changing these
settings does not require recompilation.

## Configuration File Structure

A `burn.toml` file can contain both Burn and CubeCL settings:

```toml
[fusion]
logger = { level = "basic", stderr = true }

[autodiff]
logger = { level = "disabled" }

[remote]
flush_threshold = 4
flush_bytes_threshold = 1048576

[cubecl.autotune]
level = "balanced"

[cubecl.profiling]
logger = { level = "basic", stdout = true }
```

Burn reads the top-level `fusion`, `autodiff`, and `remote` sections. CubeCL reads the `cubecl`
section independently when a CubeCL backend is used. Options apply to the corresponding
subsystem when it is enabled.

## Burn Configuration Options

### Fusion

The `[fusion]` section controls operation-fusion logging and how cached execution graphs grow.
The `[fusion.beam_search]` subsection controls exploration of fusion opportunities.

| Setting | Default | Effect |
| --- | --- | --- |
| `fusion.max_graph_size` | No limit | Caps the number of operations in a client-cached graph. |
| `fusion.growth_patience` | `32` | Closes a graph after this many consecutive operations fail to improve its best fusion score. |
| `fusion.beam_search.max_blocks` | `5` | Limits the number of independent blocks explored during fusion search. |
| `fusion.beam_search.max_explorations` | No limit | Caps optimization explorations per stream. Once reached, cache misses execute unfused; existing cache hits still replay. |

Omit optional limits to leave them unlimited. For example, to bound exploration for a workload
whose execution graphs keep changing:

```toml
[fusion]
logger = { level = "medium", stderr = true }

[fusion.beam_search]
max_explorations = 1000
```

The exploration limit trades potential fusion opportunities for less time building new
optimizations. Leave the default unless measurements show that exploration is costly for
your workload.

Fusion log levels are `disabled` (default), `basic`, `medium`, and `full`. They range from
execution-strategy summaries to cache and merge decisions, then detailed registration logs.

### Autodiff

The `[autodiff]` section controls logging for automatic differentiation:

```toml
[autodiff]
logger = { level = "basic", stderr = true }
```

Log levels are `disabled` (default), `basic`, `medium`, and `full`. Higher levels add detail
about checkpointing and recomputation. This section configures logging; the checkpoint strategy
is selected through the autodiff API.

### Remote Backend

The `[remote]` section controls outgoing message batching and remote-backend logging:

| Setting | Default | Effect |
| --- | --- | --- |
| `flush_threshold` | `4` | Flushes when this many tasks have accumulated. |
| `flush_bytes_threshold` | `1048576` (1 MiB) | Flushes when buffered tensor data reaches this size, independently of the task-count threshold. |

Larger thresholds allow more batching; smaller thresholds reduce the delay before buffered work
is sent. Remote log levels are `disabled` (default), `basic`, and `full`.

```toml
[remote]
logger = { level = "basic", stderr = true }
flush_threshold = 4
flush_bytes_threshold = 1048576
```

## Logging

Each subsystem has a `logger` field. Choose a verbosity level and at least one destination:

```toml
[fusion.logger]
level = "medium"
stderr = true
file = "logs/fusion.log"
append = true
```

This is equivalent to specifying `logger = { ... }` inside `[fusion]`. Destinations can be
combined:

- `stdout = true` or `stderr = true` writes directly to the corresponding stream.
- `file = "path/to/file.log"` writes to a file. `append` defaults to `true`; relative paths
  are resolved from the process's working directory.
- `log = "info"`, `"debug"`, or `"trace"` forwards messages to Rust's `log` crate. Initialize
  a compatible logger in your application to receive them.

Logging is disabled by default, and no destinations are enabled by default. The subsystem's
`level` controls which messages are generated; `log` selects the level used when forwarding
those messages to the `log` crate.

## CubeCL Configuration

For CUDA, HIP, WGPU, Metal, and CPU backends using CubeCL, refer to the
[CubeCL Book's configuration reference](https://burn.dev/books/cubecl/advanced-usage/config.html)
for supported options, defaults, and environment variables.

To use an example from that reference in `burn.toml`, prefix each section with `cubecl.`:

| In `cubecl.toml` | In `burn.toml` |
| --- | --- |
| `[autotune]` | `[cubecl.autotune]` |
| `[compilation]` | `[cubecl.compilation]` |
| `[memory]` | `[cubecl.memory]` |
| `[streaming]` | `[cubecl.streaming]` |

The same prefix applies to nested sections: `[profiling.logger]` becomes
`[cubecl.profiling.logger]`. Available settings depend on the CubeCL version used by Burn.

CubeCL can also read a separate `cubecl.toml` or `CubeCL.toml`. At each directory level, it
checks those files before looking for the `[cubecl]` section in `burn.toml` or `Burn.toml`,
then continues upward if none is valid. The first matching configuration is used as a whole;
the separate files and embedded sections are not merged. Keeping CubeCL settings in one place
makes their source easier to track.

## Environment Variable Overrides

These Burn variables override the corresponding file settings when configuration is first loaded:

| Variable | Setting | Accepted values |
| --- | --- | --- |
| `BURN_FUSION_LOG` | `fusion.logger.level` | `disabled`, `basic`, `medium`, `full` |
| `BURN_FUSION_MAX_EXPLORATIONS` | `fusion.beam_search.max_explorations` | Non-negative integer |
| `BURN_REMOTE_LOG` | `remote.logger.level` | `disabled`, `basic`, `full` |

The log-level variables are case-insensitive. Both accept `off` or `0` for `disabled`.
`BURN_FUSION_LOG=1` selects `full`; `BURN_REMOTE_LOG=1` selects `basic`, and `2` selects `full`.
Enabling logging through these variables also enables stderr output. There is currently no
`BURN_AUTODIFF_LOG` override; configure autodiff logging in the file.

For example, on Linux or macOS:

```sh
BURN_FUSION_LOG=medium BURN_FUSION_MAX_EXPLORATIONS=1000 cargo run --release
```

CubeCL's `CUBECL_*` overrides also apply when its settings come from `[cubecl.*]` in `burn.toml`;
see the [CubeCL environment variable reference](https://burn.dev/books/cubecl/advanced-usage/config.html#environment-variable-overrides).

## Inspecting Configuration

To inspect Burn's resolved configuration from Rust:

```rust,ignore
let config = burn::runtime_config();
println!("Fusion settings: {:?}", config.fusion());
println!("Autodiff settings: {:?}", config.autodiff());
println!("Remote settings: {:?}", config.remote());
```

This call initializes the configuration if it has not already been loaded. CubeCL keeps its own
configuration, so its settings are not included in this value.
