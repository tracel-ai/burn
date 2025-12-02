use burn::tensor::TensorData;
use burn_communication::Address;
use futures::{SinkExt, StreamExt};
use std::{
    fmt::Display,
    fs::{self, File},
    str::FromStr,
    time::{Duration, Instant},
    vec,
};
use tokio::net::TcpListener;
use tokio_serde::formats::MessagePack;
use tokio_util::codec::LengthDelimitedCodec;

use burn::{backend::NdArray, prelude::Backend, tensor::Tensor};
use burn_collective::{AllReduceStrategy, ReduceOperation};
use burn_collective_multinode_tests::shared::{NodeTest, NodeTestResult, TENSOR_RANK};
use burn_std::rand::{SeedableRng, StdRng};
use tokio::process::{Child, Command};

#[derive(Clone)]
struct AllReduceTest {
    shape: [usize; TENSOR_RANK],
    op: ReduceOperation,
    local_strategy: AllReduceStrategy,
    global_strategy: AllReduceStrategy,
}

impl Display for AllReduceTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op_str = match self.op {
            ReduceOperation::Sum => "sum",
            ReduceOperation::Mean => "mean",
        };
        let local_strategy_str = match self.local_strategy {
            AllReduceStrategy::Centralized => "local_centralized",
            AllReduceStrategy::Tree(n) => &format!("local_tree_{n}"),
            AllReduceStrategy::Ring => "local_ring",
        };
        let global_strategy_str = match self.global_strategy {
            AllReduceStrategy::Centralized => "global_centralized",
            AllReduceStrategy::Tree(n) => &format!("global_tree_{n}"),
            AllReduceStrategy::Ring => "global_ring",
        };

        write!(f, "{op_str}_{local_strategy_str}_{global_strategy_str}")
    }
}

/// Framed TCP connection for sending tests and receiving results
type TestChannel = tokio_serde::Framed<
    tokio_util::codec::Framed<tokio::net::TcpStream, LengthDelimitedCodec>,
    NodeTestResult,
    NodeTest,
    MessagePack<NodeTestResult, NodeTest>,
>;

/// Handle for each node process
struct NodeProcessHandle {
    process: Child,
    channel: TestChannel,
}

/// Main function to run the multi-node all-reduce test.
/// Launches a orchestrator and multiple nodes based on the provided topology.
#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() {
    let all_reduce_tests = vec![
        AllReduceTest {
            shape: [4, 64, 512],
            op: ReduceOperation::Mean,
            local_strategy: AllReduceStrategy::Tree(2),
            global_strategy: AllReduceStrategy::Tree(2),
        },
        AllReduceTest {
            shape: [4, 64, 512],
            op: ReduceOperation::Mean,
            local_strategy: AllReduceStrategy::Tree(2),
            global_strategy: AllReduceStrategy::Ring,
        },
        AllReduceTest {
            shape: [4, 64, 512],
            op: ReduceOperation::Mean,
            local_strategy: AllReduceStrategy::Centralized,
            global_strategy: AllReduceStrategy::Centralized,
        },
    ];

    let test_files_dir = "target/test_files";
    fs::create_dir_all(test_files_dir).expect("Couldn't create test_files directory");

    let topology: Vec<usize> = vec![4; 4];

    let mut orchestrator = launch_orchestrator(test_files_dir);

    let launcher_endpoint = "127.0.0.1:4000";

    // Build and run node processes
    let mut all_tests_durations = vec![];
    if let Ok(mut nodes) = launch_nodes(&topology, launcher_endpoint).await {
        // Run one test
        for test in all_reduce_tests.clone() {
            let test_name = test.to_string();

            let time =
                test_all_reduce_centralized_no_collective::<NdArray>(&topology, test.clone());
            println!(
                "{test_name}: Benchmark (no collective, centralized, single-threaded): {} secs",
                time.as_secs_f32()
            );

            match test_all_reduce(&topology, test, &mut nodes).await {
                Err(node_idx) => {
                    println!("{test_name}: Node with index {node_idx} failed!");
                    // Kill other node processes
                    for mut node in nodes.drain(..) {
                        node.process.kill().await.unwrap();
                        node.process.wait().await.unwrap();
                    }
                    break;
                }
                Ok(durations) => {
                    all_tests_durations.append(&mut durations.clone());
                    let avg = durations.iter().map(|dur| dur.as_secs_f32()).sum::<f32>()
                        / durations.len() as f32;
                    println!("{test_name}: Success in {avg} secs");
                }
            }
        }
    }

    if !all_tests_durations.is_empty() {
        let avg = all_tests_durations
            .iter()
            .map(|dur| dur.as_secs_f32())
            .sum::<f32>()
            / all_tests_durations.len() as f32;
        println!("Average for all tests: {avg} secs");
    }

    // Shutdown orchestrator
    orchestrator.kill().await.unwrap();
    orchestrator.wait().await.unwrap();
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

/// Launch nodes for a all_reduce test
/// Each node will connect to the global orchestrator and run an all-reduce operation.
/// The topology is a vector where each element represents the number of devices in that node.
async fn launch_nodes(
    topology: &[usize],
    launcher_endpoint: &str,
) -> Result<Vec<NodeProcessHandle>, ()> {
    println!(
        "Launching {} nodes with topology: {:?}",
        topology.len(),
        topology
    );

    // Listen for node connections
    let listener = TcpListener::bind(launcher_endpoint).await.unwrap();
    println!("Server listening on {launcher_endpoint}");

    let mut nodes = vec![];

    for node_idx in 0..topology.len() {
        // Create log file
        let output_filename = format!("target/test_files/node_{}_log.txt", node_idx + 1);
        let out = File::create(output_filename).expect("Could't open node log file");

        // Start a process for each node. Pass on our feature flags
        let node_process: Child = Command::new("cargo")
            .args([
                "run",
                "--release",
                "--features",
                #[cfg(feature = "ndarray")]
                "ndarray",
                "--bin",
                "node",
                "--",
                launcher_endpoint,
                &node_idx.to_string(),
            ])
            .stdout(out.try_clone().unwrap())
            .stderr(out)
            .spawn()
            .expect("node failed");

        // Wait for child to connect for io
        let (socket, _peer_addr) = listener.accept().await.unwrap();
        let length_delimited = tokio_util::codec::Framed::new(socket, LengthDelimitedCodec::new());
        let channel: TestChannel = tokio_serde::Framed::new(
            length_delimited,
            MessagePack::<NodeTestResult, NodeTest>::default(),
        );

        nodes.push(NodeProcessHandle {
            process: node_process,
            channel,
        });
    }

    Ok(nodes)
}

async fn test_all_reduce(
    topology: &[usize],
    test: AllReduceTest,
    nodes: &mut [NodeProcessHandle],
) -> Result<Vec<Duration>, usize> {
    dispatch_all_reduce_test(topology, test, nodes).await;

    let mut all_durations = vec![];
    for (idx, handle) in nodes.iter_mut().enumerate() {
        match handle.channel.next().await {
            Some(Ok(mut result)) => {
                if !result.success {
                    return Err(idx);
                }
                all_durations.append(&mut result.durations);
            }
            _ => {
                return Err(idx);
            }
        }
    }

    Ok(all_durations)
}

async fn dispatch_all_reduce_test(
    topology: &[usize],
    test: AllReduceTest,
    nodes: &mut [NodeProcessHandle],
) {
    let total_device_count: usize = topology.iter().sum();
    let (mut all_inputs, expected) =
        generate_random_input(test.shape, test.op, total_device_count, 42);

    // URL for the global orchestrator on port 3000
    let global_url = "ws://localhost:3000";
    let global_address = Address::from_str(global_url).unwrap();

    for (node_idx, &device_count) in topology.iter().enumerate() {
        // Construct URL for node
        // Ports 3001... are for each node
        let data_service_port = node_idx as u16 + 3001;
        let node_url = format!("ws://localhost:{data_service_port}");
        let node_address = Address::from_str(&node_url).unwrap();

        // take input tensors for each device
        let inputs = all_inputs[0..device_count].to_vec();
        all_inputs = all_inputs[device_count..].to_vec();

        let test = NodeTest {
            device_count,
            node_id: node_idx.into(),
            node_count: topology.len() as u32,
            global_address: global_address.clone(),
            node_address,
            data_service_port,
            all_reduce_op: test.op,
            global_strategy: test.global_strategy,
            local_strategy: test.local_strategy,
            inputs,
            expected: expected.clone(),
        };
        let handle = &mut nodes[node_idx];

        handle.channel.send(test).await.unwrap();
    }

    assert!(
        all_inputs.is_empty(),
        "Not all inputs have been sent to tests"
    );
}

/// Run the test sequentially with no collective operations to get the optimal single-threaded speed
fn test_all_reduce_centralized_no_collective<B: Backend>(
    topology: &[usize],
    test: AllReduceTest,
) -> Duration {
    let total_device_count: usize = topology.iter().sum();
    let (all_inputs, _expected) =
        generate_random_input(test.shape, test.op, total_device_count, 42);

    let mut all_inputs = all_inputs
        .into_iter()
        .map(|data| Tensor::<B, 3>::from_data(data, &B::Device::default()));

    let start = Instant::now();

    // Sequential test
    let mut result = all_inputs.next().unwrap();
    for other in all_inputs {
        result = result.add(other);
    }
    if test.op == ReduceOperation::Mean {
        result.div_scalar(total_device_count as u32);
    }

    start.elapsed()
}

/// Generates random input tensors and expected output based on the provided shape and reduce kind.
fn generate_random_input(
    shape: [usize; 3],
    reduce_kind: ReduceOperation,
    input_count: usize,
    seed: u64,
) -> (Vec<TensorData>, TensorData) {
    let mut rng = StdRng::seed_from_u64(seed);

    // A random tensor for each device
    let input: Vec<TensorData> = (0..input_count)
        .map(|_| {
            TensorData::random::<f32, _, _>(shape, burn::tensor::Distribution::Default, &mut rng)
        })
        .collect();

    // Sum up the inputs
    let device = <NdArray as Backend>::Device::default();
    let mut expected_tensor = Tensor::<NdArray, TENSOR_RANK>::zeros(shape, &device);
    for item in input.iter().take(input_count) {
        let input_tensor = Tensor::<NdArray, TENSOR_RANK>::from_data(item.clone(), &device);
        expected_tensor = expected_tensor.add(input_tensor);
    }

    if reduce_kind == ReduceOperation::Mean {
        expected_tensor = expected_tensor.div_scalar(input_count as u32);
    }

    // All-Reduce results should have this value
    let expected = expected_tensor.to_data();

    (input, expected)
}
