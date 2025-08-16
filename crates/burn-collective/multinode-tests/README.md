# Integration test for burn collective operations with multiple nodes and devices.

Run `cargo run --bin test_launcher`

There are 3 binaries:

## node.rs

Launches `n` threads each simulating a different device. Currently the backend is NdArray,
so everything is CPU. The program takes a file with configurations and input data.

## global.rs

Runs the global orchestrator, who is responsible for responding to global collective operation
requests. In the case of an all-reduce, the orchestrator responds with a strategy for reducing,
and the node can do the reduction independently.

## test_launcher.rs

Generates input data, calculates the expected results, and launches the nodes each with their
own inputs in a separate file.

The topology is [4, 4, 4, 4]. This means 4 nodes are launched,
each with 4 threads (for each device).

The global orchestrator (`global.rs`) is also launched.

## Output

The outputs and inputs for each node and the orchestrator are written to the `target/test_files` folder

If the nodes or orchestrator stall, there is a timeout.
