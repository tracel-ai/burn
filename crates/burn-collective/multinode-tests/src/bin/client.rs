use std::{env, sync::mpsc::SyncSender};

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Tensor, TensorData, Tolerance},
};
use burn_collective::{
    GlobalRegisterParams, RegisterParams, SharedAllReduceParams, SharedGlobalRegisterParams,
    SharedRegisterParams, all_reduce, finish_collective, register, reset_collective,
};
use burn_collective_multinode_tests::shared::NodeTestData;

use serde_json::from_reader;
use std::fs::File;

type TestBackend = NdArray;

/// Start a client that will test all-reduce
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

    let device_count = test_input.device_count;

    let global = Some(GlobalRegisterParams {
        node_id: test_input.node_id,
        server_address: test_input.server_address,
        client_address: test_input.client_address,
        client_data_port: test_input.client_data_port,
        shared_params: SharedGlobalRegisterParams {
            num_nodes: test_input.node_count,
        },
    });
    let reg_params = RegisterParams {
        device_id: 0.into(),
        shared: SharedRegisterParams {
            num_devices: test_input.device_count,
        },
        global,
    };

    let (send, recv) = std::sync::mpsc::sync_channel(32);

    let mut handles = vec![];
    for id in 0..device_count {
        let send = send.clone();
        let mut reg_params = reg_params.clone();
        reg_params.device_id = id.into();
        let params = test_input.aggregate_params.clone();
        let input = test_input.inputs[id as usize].clone();
        let handle = std::thread::spawn(move || run_peer::<B>(reg_params, params, input, send));
        handles.push(handle);
    }

    let first = recv.recv().unwrap().to_data();
    for _ in 1..device_count {
        let tensor = recv.recv().unwrap();
        tensor.to_data().assert_eq(&first, true);
    }

    let tol: Tolerance<f32> = Tolerance::balanced();
    test_input.expected.assert_approx_eq(&first, tol);

    for handle in handles {
        let _ = handle.join();
    }

    println!(
        "Test success: {:?} and {:?}",
        first.to_vec::<f32>().unwrap(),
        test_input.expected.to_vec::<f32>().unwrap()
    );
}

/// Runs a thread in the all-reduce test.
pub fn run_peer<B: Backend>(
    reg_params: RegisterParams,
    all_reduce_params: SharedAllReduceParams,
    input: TensorData,
    output: SyncSender<Tensor<B, 1>>,
) {
    let device = B::Device::default();
    let id = reg_params.device_id;

    register::<B>(reg_params).unwrap();

    let tensor = Tensor::<B, 1>::from_data(input, &device);

    let tensor = all_reduce(id, tensor, &all_reduce_params).unwrap();

    output.send(tensor).unwrap();

    finish_collective::<B>(id).unwrap();
}
