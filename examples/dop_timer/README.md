# dop_timer

This binary exists to time the behavior of distributed (local, global) collective operations.

This binary uses the `gRPC` OTEL exporter to send traces to an OTEL Collector on port `4317`.

## Example

1. Setup an OTEL Collector

There are many ways to do this; one of the simplest is to use the `jaegertracing/all-in-one:latest` docker image:

```bash
$ docker run -e OTEL_TRACES_SAMPLER=always_off -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317-4318:4317-4318 -p 14250:14250 -p 14268:14268 -p 14269:14269 jaegertracing/all-in-one:latest
```

Then navigate to `localhost:16686` to view traces.

2. Run the binary, with the OTEL Collector endpoint as an argument:

```bash
$ cargo run -p dop_timer --features cuda -- --tracing otel
```