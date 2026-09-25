# Runtime Configuration

Burn provides runtime configuration for autotuning, profiling, logging, operation fusion, and
remote-backend batching.

## Overview

By default, Burn loads its configuration from a TOML file (`burn.toml` or `Burn.toml`) in your
current directory or a parent directory. Missing settings use defaults. If no valid file is found,
Burn uses the default configuration.

You can also override configuration options using environment variables, which is useful for
debugging, CI, and deployment.

File loading and environment overrides require the `std` feature and a supported platform with
filesystem access. Burn uses the first valid file without merging files; malformed files are skipped
with a warning.

> **Note:** Configuration is loaded on first use. Set environment variables before starting your
> application, and restart it after editing the file. No recompilation is needed.

## Configuration File Structure

A typical `burn.toml` file might look like this:

```toml
[cubecl.autotune]
level = "balanced"

[cubecl.profiling]
logger = { level = "basic", stdout = true }

[fusion]
logger = { level = "basic", stderr = true }

[autodiff]
logger = { level = "disabled" }

[remote]
flush_threshold = 4
flush_bytes_threshold = 1048576
```

Each section configures a different aspect of Burn:

- **cubecl**: Configures autotuning, profiling, compilation, and memory for CubeCL backends.
- **fusion**: Controls operation fusion and its logging.
- **autodiff**: Controls automatic differentiation logging.
- **remote**: Configures remote-backend batching and logging.

## Configuration Options

### CubeCL

The `[cubecl]` section configures the CubeCL runtime. You can use the options from the
[CubeCL configuration guide](https://github.com/tracel-ai/cubecl/blob/main/cubecl-book/src/advanced-usage/config.md)
by prefixing each section with `cubecl.`. For example, `[autotune]` becomes `[cubecl.autotune]`. The
same prefix applies to nested sections.

See the guide for available options, defaults, and examples. Supported settings depend on the CubeCL
version used by Burn.

At each directory level, CubeCL checks `cubecl.toml` and `CubeCL.toml` before the `[cubecl]` section
in `burn.toml` or `Burn.toml`, then searches parent directories. It uses the first valid
configuration; separate files and embedded sections are not merged.

### Fusion

The `[fusion]` section controls how Burn combines operations and logs fusion activity.

**Log Levels:**

- `disabled` (default): No fusion logs.
- `basic`: Logs the execution strategy selected for each stream.
- `medium`: Adds cache hits and misses, and block merge/split decisions.
- `full`: Adds every registration, rejection, and scoring decision.

**Graph Settings:**

- `max_graph_size`: Maximum operations in a client-cached graph (default: no limit).
- `growth_patience`: Closes a graph after this many consecutive operations fail to improve its best
  fusion score (default: `32`).

**Fusion Search** (`[fusion.beam_search]`):

- `max_blocks`: Maximum independent blocks explored during fusion search (default: `5`).
- `max_explorations`: Maximum optimization explorations per stream (default: no limit). Once
  reached, cache misses execute unfused; cached optimizations still run.

Leave optional limits unset to keep them unlimited.

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

**Batching Settings:**

- `flush_threshold`: Sends buffered tasks when this many have accumulated (default: `4`).
- `flush_bytes_threshold`: Sends buffered tasks when their tensor data reaches this size in bytes
  (default: `1048576`, or 1 MiB). Either threshold triggers a flush.

Larger thresholds allow more batching; smaller thresholds reduce the delay before buffered work is
sent.

**Example:**

```toml
[remote]
logger = { level = "basic", stderr = true }
flush_threshold = 4
flush_bytes_threshold = 1048576
```

## Environment Variable Overrides

Burn supports the following environment variables to override configuration at runtime:

- `BURN_FUSION_LOG`: Sets fusion log verbosity. Accepts `disabled`, `basic`, `medium`, or `full`;
  `off` and `0` disable logging, and `1` selects `full`.
- `BURN_FUSION_MAX_EXPLORATIONS`: Sets `fusion.beam_search.max_explorations` to a non-negative
  integer.
- `BURN_REMOTE_LOG`: Sets remote-backend log verbosity. Accepts `disabled`, `basic`, or `full`;
  `off` and `0` disable logging, `1` selects `basic`, and `2` selects `full`.

Log levels are case-insensitive. Enabling logging through these variables also enables stderr
output. Configure autodiff logging in the file; there is no `BURN_AUTODIFF_LOG` override.

**Example (Linux/macOS):**

```sh
export BURN_FUSION_LOG=medium
export BURN_FUSION_MAX_EXPLORATIONS=1000
```

CubeCL's `CUBECL_*` overrides also apply when its settings come from `[cubecl.*]` in `burn.toml`;
see the
[CubeCL environment variable reference](https://github.com/tracel-ai/cubecl/blob/main/cubecl-book/src/advanced-usage/config.md#environment-variable-overrides).

## Logging

Burn can log to multiple destinations at once. Configure them in each section's `logger` field:

- `stdout = true`: Writes to stdout.
- `stderr = true`: Writes to stderr.
- `file = "burn.log"`: Writes to a file. `append` defaults to `true`; relative paths start from your
  application's working directory.
- `log = "info"`, `"debug"`, or `"trace"`: Forwards messages to Rust's `log` crate. Initialize a
  compatible logger in your application to receive them.

Choose a verbosity `level` and at least one destination. No destinations are enabled by default. The
`level` setting controls which messages are generated; `log` sets their level in the `log` crate.

**Example:**

```toml
[fusion]
logger = { level = "medium", stderr = true, file = "burn.log", append = true }
```
