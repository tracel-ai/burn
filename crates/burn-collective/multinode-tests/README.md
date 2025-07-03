# Integration test for burn collective operations with multiple nodes and devices.

Run `cargo build` then 
`CARGO_TARGET_DIR=../../../target/ cargo run --bin main` from the `multinode-tests` directory.

There are 3 binaries:

## client.rs
Launches `n` threads each simulating a different device. Currently the backend is NdArray, 
so everything is CPU. The program takes a file with configurations and input data.

## server.rs
Runs the global collective server

## main.rs
Generates input data, calculates the expected results, and launches the clients each with their 
own inputs in a seperate file.

The tolopogy is [5, 5, 5, 5, 5]. This means 5 clients (nodes) are launched, 
each with 5 threads (devices).

The global collective server (`server.rs`) is also launched.

## Output
The outputs and inputs for each client and server are written to the `target/test` folder
