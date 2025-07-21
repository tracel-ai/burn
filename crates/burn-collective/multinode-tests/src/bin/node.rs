use std::{env, sync::mpsc::SyncSender};

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Tensor, Tolerance},
};
use burn_collective::{
    AllReduceStrategy, CollectiveConfig, all_reduce, finish_collective, register, reset_collective,
};
use burn_collective_multinode_tests::shared::NodeTestData;

use serde_json::from_reader;
use std::fs::File;
use std::thread::JoinHandle;

type TestBackend = NdArray;

/// Start a node that will test all-reduce
/// Args are the following:
/// - name of test config file
pub fn main() {
    let args: Vec<String> = env::args().collect();

    let inputs_filename = args[1].clone();

    let file = File::open(inputs_filename).expect("Failed to open file");
    let test_data = from_reader(file).expect("Failed to parse JSON");

    test_all_reduce::<NdArray>(test_data);
}

/// Runs the all-reduce test for one node
fn test_all_reduce<B: Backend>(test_input: NodeTestData) {
    reset_collective::<TestBackend>();

    // Channel for results
    let (result_send, result_recv) = std::sync::mpsc::sync_channel(32);

    // Launch a thread for each "device"
    let handles = launch_threads::<B, 1>(test_input.clone(), result_send);

    // Receive results
    let first = result_recv.recv().unwrap().to_data();
    let tol: Tolerance<f32> = Tolerance::balanced();
    for _ in 1..test_input.device_count {
        // Assert all results are equal to each other as well as expected result
        let tensor = result_recv.recv().unwrap();
        tensor.to_data().assert_eq(&first, true);
    }

    test_input.expected.assert_approx_eq(&first, tol);

    // Threads finish
    for handle in handles {
        let _ = handle.join();
    }

    println!(
        "Test success: {:?} and {:?}",
        first.to_vec::<f32>().unwrap(),
        test_input.expected.to_vec::<f32>().unwrap()
    );
}

/// Launch a thread for each device, and run the all-reduce
fn launch_threads<B: Backend, const D: usize>(
    test_input: NodeTestData,
    result_send: SyncSender<Tensor<B, D>>,
) -> Vec<JoinHandle<()>> {
    let mut handles = vec![];
    for id in 0..test_input.device_count {
        // Launch a thread to test

        // Put all the parameters in the config
        let config = CollectiveConfig::default()
            .with_all_reduce_kind(test_input.all_reduce_kind)
            .with_num_devices(test_input.device_count)
            .with_device_id(id.into())
            .with_node_id(test_input.node_id)
            .with_global_address(test_input.global_address.clone())
            .with_node_address(test_input.node_address.clone())
            .with_data_service_port(test_input.data_service_port)
            .with_num_nodes(test_input.node_count)
            .with_global_strategy(test_input.global_strategy)
            .with_local_strategy(test_input.local_strategy);

        // Inputs and outputs for the test
        let tensor_data = test_input.inputs[id as usize].clone();
        let tensor = Tensor::<B, D>::from_data(tensor_data, &B::Device::default());
        let result_send = result_send.clone();

        let handle = std::thread::spawn(move || run_peer::<B, D>(config, tensor, result_send));
        handles.push(handle);
    }

    handles
}

/// Runs a thread in the all-reduce test.
pub fn run_peer<B: Backend, const D: usize>(
    config: CollectiveConfig,
    input: Tensor<B, D>,
    output: SyncSender<Tensor<B, D>>,
) {
    // Register the device
    register::<B>(&config).unwrap();

    // All-reduce
    let tensor = all_reduce(input, &config).unwrap();

    // Send result
    output.send(tensor).unwrap();

    finish_collective::<B>(&config).unwrap();
}
