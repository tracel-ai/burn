# Runtime Configuration

Burn uses `burn.toml` to configure logging, operation fusion, and remote-backend batching. For
CubeCL backends, the same file can also configure autotuning, profiling, compilation, and memory
management through `[cubecl.*]` sections.

## Overview

With filesystem support and the `std` feature enabled, Burn searches the current directory and its
parents for `burn.toml` or `Burn.toml`. It uses the first valid file without merging files. Missing
settings use defaults; malformed files are skipped with a warning.

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
section independently when a CubeCL backend is used.

## Burn Configuration Options

### Fusion

The `[fusion]` section controls operation-fusion logging and how cached execution graphs grow. The
`[fusion.beam_search]` subsection controls exploration of fusion opportunities.

**Log Levels:**

- `disabled` (default): No fusion logs.
- `basic`: Logs the execution strategy selected for each stream.
- `medium`: Adds cache hits and misses, and block merge/split decisions.
- `full`: Adds every registration, rejection, and scoring decision.

**Settings:**

| Setting                               | Default  | Effect                                                                                                                   |
| ------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `fusion.max_graph_size`               | No limit | Caps the number of operations in a client-cached graph.                                                                  |
| `fusion.growth_patience`              | `32`     | Closes a graph after this many consecutive operations fail to improve its best fusion score.                             |
| `fusion.beam_search.max_blocks`       | `5`      | Limits the number of independent blocks explored during fusion search.                                                   |
| `fusion.beam_search.max_explorations` | No limit | Caps optimization explorations per stream. Once reached, cache misses execute unfused; existing cache hits still replay. |

Omit optional limits to leave them unlimited. Limiting exploration reduces optimization work but can
miss fusion opportunities.

**Example:**

```toml
[fusion]
logger = { level = "medium", stderr = true }

[fusion.beam_search]
max_explorations = 1000
```

### Autodiff

The `[autodiff]` section controls logging for automatic differentiation.

**Log Levels:**

- `disabled` (default): No autodiff logs.
- `basic`: Logs backward graph size and the checkpoint strategy.
- `medium`: Adds which tensors are checkpointed or recomputed.
- `full`: Adds every graph node traversal and recomputation event.

**Example:**

```toml
[autodiff]
logger = { level = "basic", stderr = true }
```

### Remote Backend

The `[remote]` section controls outgoing message batching and remote-backend logging.

**Log Levels:**

- `disabled` (default): No remote-backend logs.
- `basic`: Logs periodic summaries of network bytes saved by graph caching.
- `full`: Adds every optimization registration and replay, with message sizes.

**Settings:**

| Setting                 | Default           | Effect                                                                                          |
| ----------------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| `flush_threshold`       | `4`               | Flushes when this many tasks have accumulated.                                                  |
| `flush_bytes_threshold` | `1048576` (1 MiB) | Flushes when buffered tensor data reaches this size, independently of the task-count threshold. |

Larger thresholds allow more batching; smaller thresholds reduce the delay before buffered work is
sent.

**Example:**

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

This is equivalent to specifying `logger = { ... }` inside `[fusion]`. Destinations can be combined:

- `stdout = true` or `stderr = true` writes directly to the corresponding stream.
- `file = "path/to/file.log"` writes to a file. `append` defaults to `true`; relative paths are
  resolved from the process's working directory.
- `log = "info"`, `"debug"`, or `"trace"` forwards messages to Rust's `log` crate. Initialize a
  compatible logger in your application to receive them.

No destinations are enabled by default. The subsystem's `level` controls which messages are
generated; `log` selects the level used when forwarding those messages to the `log` crate.

## CubeCL Configuration

For CubeCL backends, refer to the
[CubeCL Book's configuration reference](https://github.com/tracel-ai/cubecl/blob/main/cubecl-book/src/advanced-usage/config.md)
for supported options, defaults, and environment variables.

In `burn.toml`, prefix CubeCL sections with `cubecl.`, for example `[autotune]` becomes
`[cubecl.autotune]`. This also applies to nested sections. Available settings depend on the CubeCL
version used by Burn.

At each directory level, CubeCL checks `cubecl.toml` and `CubeCL.toml` before the `[cubecl]` section
in `burn.toml` or `Burn.toml`, then searches parent directories. It uses the first valid
configuration; separate files and embedded sections are not merged.

## Environment Variable Overrides

These Burn variables override the corresponding file settings when configuration is first loaded:

| Variable                       | Setting                               | Accepted values                       |
| ------------------------------ | ------------------------------------- | ------------------------------------- |
| `BURN_FUSION_LOG`              | `fusion.logger.level`                 | `disabled`, `basic`, `medium`, `full` |
| `BURN_FUSION_MAX_EXPLORATIONS` | `fusion.beam_search.max_explorations` | Non-negative integer                  |
| `BURN_REMOTE_LOG`              | `remote.logger.level`                 | `disabled`, `basic`, `full`           |

The log-level variables are case-insensitive. Both accept `off` or `0` for `disabled`.
`BURN_FUSION_LOG=1` selects `full`; `BURN_REMOTE_LOG=1` selects `basic`, and `2` selects `full`.
Enabling logging through these variables also enables stderr output. There is currently no
`BURN_AUTODIFF_LOG` override; configure autodiff logging in the file.

For example, on Linux or macOS:

```sh
BURN_FUSION_LOG=medium BURN_FUSION_MAX_EXPLORATIONS=1000 cargo run --release
```

CubeCL's `CUBECL_*` overrides also apply when its settings come from `[cubecl.*]` in `burn.toml`;
see the
[CubeCL environment variable reference](https://github.com/tracel-ai/cubecl/blob/main/cubecl-book/src/advanced-usage/config.md#environment-variable-overrides).
