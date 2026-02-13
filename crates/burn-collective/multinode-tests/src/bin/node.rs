use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{
        Tensor, TensorPrimitive, Tolerance,
        backend::{PeerId, ReduceOperation},
    },
};
use burn_collective::{
    CollectiveConfig, all_reduce, finish_collective, register, reset_collective,
};
use burn_collective_multinode_tests::shared::{NodeTest, NodeTestResult, TENSOR_RANK};
use std::{
    env,
    sync::mpsc::SyncSender,
    time::{Duration, Instant},
};
use tokio::net::TcpStream;

use futures::{SinkExt, StreamExt};
use std::thread::JoinHandle;
use tokio_serde::formats::MessagePack;
use tokio_util::codec::LengthDelimitedCodec;

type TestBackend = NdArray;

/// Framed TCP connection channel
type TestChannel = tokio_serde::Framed<
    tokio_util::codec::Framed<tokio::net::TcpStream, LengthDelimitedCodec>,
    NodeTest,
    NodeTestResult,
    MessagePack<NodeTest, NodeTestResult>,
>;

/// Start a node that will test all-reduce
/// Args are the following:
/// - launcher endpoint
#[tokio::main]
pub async fn main() {
    let args: Vec<String> = env::args().collect();

    let launcher_addr = args[1].clone();

    let socket = TcpStream::connect(launcher_addr).await.unwrap();
    let length_delimited = tokio_util::codec::Framed::new(socket, LengthDelimitedCodec::new());
    let mut socket: TestChannel = tokio_serde::Framed::new(
        length_delimited,
        MessagePack::<NodeTest, NodeTestResult>::default(),
    );

    // Loop: receive, do test, send result
    while let Some(Ok(test)) = socket.next().await {
        println!("Received test: {test:?}");

        let result = run_test::<NdArray>(&test);

        // send the result back
        socket.send(result).await.expect("failed to send Result");
    }

    println!("Server closed connection; exiting.");
}

/// Runs a test for one node
fn run_test<B: Backend>(test_input: &NodeTest) -> NodeTestResult {
    reset_collective::<TestBackend>();

    // Channel for results
    let (result_send, result_recv) = std::sync::mpsc::sync_channel(32);

    // Launch a thread for each "device"
    let handles = launch_threads::<B>(test_input.clone(), result_send);

    // Receive results
    let mut durations = vec![];
    let tol: Tolerance<f32> = Tolerance::balanced();
    for _ in 0..test_input.device_count {
        // Assert all results are equal to each other as well as expected result
        let (tensor, duration) = result_recv.recv().unwrap();
        test_input.expected.assert_approx_eq(&tensor.to_data(), tol);

        durations.push(duration);
    }

    // Threads finish
    for handle in handles {
        let _ = handle.join();
    }

    NodeTestResult {
        success: true,
        durations,
    }
}

/// Launch a thread for each device, and run the all-reduce
fn launch_threads<B: Backend>(
    test_input: NodeTest,
    result_send: SyncSender<(Tensor<B, TENSOR_RANK>, Duration)>,
) -> Vec<JoinHandle<()>> {
    let mut handles = vec![];
    for id in 0..test_input.device_count {
        // Launch a thread to test

        // Put all the parameters in the config
        let config = CollectiveConfig::default()
            .with_num_devices(test_input.device_count)
            .with_global_address(test_input.global_address.clone())
            .with_node_address(test_input.node_address.clone())
            .with_data_service_port(test_input.data_service_port)
            .with_num_nodes(test_input.node_count)
            .with_global_all_reduce_strategy(test_input.global_strategy)
            .with_local_all_reduce_strategy(test_input.local_strategy);

        // Inputs and outputs for the test
        let tensor_data = test_input.inputs[id].clone();
        let tensor = Tensor::<B, TENSOR_RANK>::from_data(tensor_data, &B::Device::default());
        let result_send = result_send.clone();

        let handle = std::thread::spawn(move || {
            run_peer::<B>(
                id.into(),
                config,
                tensor,
                result_send,
                test_input.all_reduce_op,
            )
        });
        handles.push(handle);
    }

    handles
}

/// Runs a thread in the all-reduce test.
pub fn run_peer<B: Backend>(
    id: PeerId,
    config: CollectiveConfig,
    input: Tensor<B, TENSOR_RANK>,
    output: SyncSender<(Tensor<B, TENSOR_RANK>, Duration)>,
    all_reduce_op: ReduceOperation,
) {
    // Register the device
    register::<B>(id, input.device(), config).unwrap();

    let start = Instant::now();

    // All-reduce
    let input = input.into_primitive().tensor();
    let tensor = all_reduce::<B>(id, input, all_reduce_op).unwrap();
    let tensor = Tensor::<B, TENSOR_RANK>::from_primitive(TensorPrimitive::Float(tensor));

    let duration = start.elapsed();

    // Send result
    output.send((tensor, duration)).unwrap();

    finish_collective::<B>(id).unwrap();
}
