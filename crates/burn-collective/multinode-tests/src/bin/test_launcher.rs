use burn_communication::Address;
use std::{
    fs::{self, File},
    process::{self, ExitStatus},
    str::FromStr,
};
use tokio::time::{Duration, Instant};

use tokio::process::{Child, Command};

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Shape, Tensor, TensorData},
};
use burn_collective::{AllReduceStrategy, ReduceOperation};
use burn_collective_multinode_tests::shared::NodeTestData;
use burn_common::rand::{SeedableRng, StdRng};

use serde_json::to_writer_pretty;

/// Main function to run the multi-node all-reduce test.
/// Launches a orchestrator and multiple nodes based on the provided topology.
#[tokio::main]
async fn main() {
    let test_files_dir = "target/test_files";
    fs::create_dir_all(test_files_dir).expect("Couldn't create test_files directory");

    let topology = vec![5, 5, 5, 5, 5];
    let tensor_shape = Shape { dims: vec![4] };

    let mut orchestrator = launch_orchestrator(test_files_dir);

    println!(
        "Launching {} nodes with topology: {:?}",
        topology.len(),
        topology
    );
    let nodes = launch_nodes(
        topology,
        tensor_shape,
        ReduceOperation::Mean,
        AllReduceStrategy::Ring,
        AllReduceStrategy::Tree(2),
    );

    // Await results with timeout
    let result = await_results_with_timeout(nodes).await;

    // Shutdown orchestrator
    let _ = orchestrator.kill().await;
    let _ = orchestrator.wait().await;

    match result {
        Ok(_) => {
            println!("All tests successful");
            process::exit(0);
        }
        Err(msg) => {
            println!("{msg}");
            process::exit(1);
        }
    }
}

/// Awaits the results of each child, with a timeout.
/// Returns true if all children exited successfully
async fn await_results_with_timeout(
    mut node_processes: Vec<(String, Child)>,
) -> Result<(), String> {
    let i = 0;
    let timeout = Duration::from_secs(30);
    let start = Instant::now();
    while i < node_processes.len() {
        // Timeout?
        if start.elapsed() > timeout {
            return Err(format!(
                "Test timed out after {} seconds",
                start.elapsed().as_secs()
            ));
        }

        // Get process' name and exit status
        let (name, status) = {
            let (node_name, process) = &mut node_processes[i];
            let mut status: Option<ExitStatus> = None;
            if let Ok(mut s) = process.try_wait() {
                status = s.take();
            }

            (node_name.clone(), status)
        };

        // If node has finished
        if let Some(status) = status {
            node_processes.remove(i);
            if !status.success() {
                // Node failed! Close other node processes
                for (_, mut child) in node_processes {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                }
                return Err(format!("Node {name} failed: {status:?}"));
            }
        }
    }

    // Success
    Ok(())
}

/// Launch a global orchestrator with an output file in the given directory.
/// Necessary for global collective operations
///
/// Server listens on localhost port 3000
fn launch_orchestrator(test_files_dir: &str) -> Child {
    let out_path = format!("{test_files_dir}/orchestrator_out.txt");
    let out = File::create(out_path).expect("Could't create orchestrator output file");

    Command::new("cargo")
        .args(["run", "--bin", "global", "--", "3000"])
        .stdout(out.try_clone().unwrap())
        .stderr(out)
        .spawn()
        .expect("failed to launch orchestrator")
}

/// Launch nodes based on the provided topology.
/// Each node will connect to the global orchestrator and run an all-reduce operation.
/// The topology is a vector where each element represents the number of devices in that node.
fn launch_nodes(
    topology: Vec<u32>,
    tensor_shape: Shape,
    reduce_op: ReduceOperation,
    global_strategy: AllReduceStrategy,
    local_strategy: AllReduceStrategy,
) -> Vec<(String, Child)> {
    let total_device_count = topology.iter().sum();
    let (inputs, expected) =
        generate_random_input(tensor_shape, reduce_op, total_device_count, 42);

    // URL for the global orchestrator on port 3000
    let global_url = "ws://localhost:3000";
    let global_address = Address::from_str(global_url).unwrap();

    let node_count = topology.len() as u32;
    let mut nodes = vec![];
    let mut total_device_count = 0;
    for (node_idx, &device_count) in topology.iter().enumerate() {
        // Generate the input file for this node
        let input_filename = write_node_input(
            node_idx as u32,
            device_count,
            &mut total_device_count,
            &inputs,
            expected.clone(),
            node_count,
            global_address.clone(),
            reduce_op,
            global_strategy,
            local_strategy,
        );

        // Create output file
        let output_filename = format!("target/test_files/node_{}_out.txt", node_idx + 1);
        let out = File::create(output_filename).expect("Could't create node output file");

        // Start a process for each node.
        let node: Child = Command::new("cargo")
            .args(["run", "--bin", "node", "--", &input_filename])
            .stdout(out.try_clone().unwrap())
            .stderr(out)
            .spawn()
            .expect("node failed");

        let node_name = format!("Node {}", node_idx + 1);
        nodes.push((node_name, node));
    }

    nodes
}

/// Write test inputs for a node to a file, return the filename
#[allow(clippy::too_many_arguments)]
fn write_node_input(
    node_idx: u32,
    device_count: u32,
    total_device_count: &mut usize,
    all_inputs: &[TensorData],
    expected: TensorData,
    node_count: u32,
    global_address: Address,
    reduce_op: ReduceOperation,
    global_strategy: AllReduceStrategy,
    local_strategy: AllReduceStrategy,
) -> String {
    // Construct URL for node
    // Ports 3001... are for each node
    let data_service_port = node_idx as u16 + 3001;
    let node_url = format!("ws://localhost:{data_service_port}");
    let node_address = Address::from_str(&node_url).unwrap();

    // Copy a slice of inputs: one input tensor for each device
    let upper = *total_device_count + device_count as usize;
    let inputs = all_inputs[*total_device_count..upper].to_vec();
    *total_device_count += device_count as usize;

    // Input struct
    let data = NodeTestData {
        device_count,
        node_id: node_idx.into(),
        node_count,
        global_address,
        node_address,
        data_service_port,
        all_reduce_op: reduce_op,
        global_strategy,
        local_strategy,
        inputs,
        expected,
    };

    // Writing inputs to the file
    let input_filename = format!("target/test_files/node_{}_in.txt", node_idx + 1);
    let file = File::create(&input_filename).expect("Failed to create file");
    to_writer_pretty(file, &data).expect("Failed to write JSON");

    input_filename
}

/// Generates random input tensors and expected output based on the provided shape and reduce kind.
fn generate_random_input(
    shape: Shape,
    reduce_kind: ReduceOperation,
    input_count: u32,
    seed: u64,
) -> (Vec<TensorData>, TensorData) {
    let mut rng = StdRng::seed_from_u64(seed);

    // A random tensor for each device
    let input: Vec<TensorData> = (0..input_count)
        .map(|_| {
            TensorData::random::<f32, _, _>(
                shape.clone(),
                burn::tensor::Distribution::Default,
                &mut rng,
            )
        })
        .collect();

    // Sum up the inputs
    let device = <NdArray as Backend>::Device::default();
    let mut expected_tensor = Tensor::<NdArray, 1>::zeros(shape, &device);
    for item in input.iter().take(input_count as usize) {
        let input_tensor = Tensor::<NdArray, 1>::from_data(item.clone(), &device);
        expected_tensor = expected_tensor.add(input_tensor);
    }

    if reduce_kind == ReduceOperation::Mean {
        expected_tensor = expected_tensor.div_scalar(input_count);
    }

    // All-Reduce results should have this value
    let expected = expected_tensor.to_data();

    (input, expected)
}
