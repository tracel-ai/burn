# dop_timer

This binary exists to time the behavior of distributed (local, global) collective operations.

## Example

1. Setup an OTEL Collector

2. Run the binary, with the OTEL Collector endpoint as an argument:

```bash
$ cargo run -p dop_timer --features cuda -- --tracing otel
```