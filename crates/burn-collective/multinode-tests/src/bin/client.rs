use std::{env, sync::mpsc::SyncSender};

use burn::{
    backend::{
        NdArray,
        collective::{
            AllReduceParams, GlobalRegisterParams, RegisterParams, all_reduce, finish_collective,
            register, reset_collective,
        },
    },
    prelude::Backend,
    tensor::{Tensor, TensorData, Tolerance},
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

fn test_all_reduce<B: Backend>(test_input: NodeTestData) {
    reset_collective::<TestBackend>();

    let global_server_url = test_input.server_url;
    let global_client_url = test_input.client_url;
    let global_client_data_port = test_input.client_data_port;
    let node_id = test_input.node_id;
    let num_nodes = test_input.node_count;
    let device_count = test_input.device_count;

    let global_params = Some(GlobalRegisterParams {
        node_id,
        num_nodes,
        server_url: global_server_url,
        client_url: global_client_url,
        client_data_port: global_client_data_port,
    });
    let reg_params = RegisterParams {
        num_devices: device_count,
        global_params,
    };

    let (send, recv) = std::sync::mpsc::sync_channel(32);

    let mut global_idx: usize = 0;
    for id in 0..test_input.device_count {
        let send = send.clone();
        let reg_params = reg_params.clone();
        let params = test_input.aggregate_params.clone();
        let input = test_input.inputs[global_idx].clone();
        std::thread::spawn(move || run_peer::<B>(id, reg_params, params, input, send));

        global_idx += 1;
    }

    let first = recv.recv().unwrap().to_data();
    for _ in 1..device_count {
        let tensor = recv.recv().unwrap();
        tensor.to_data().assert_eq(&first, true);
    }

    let tol: Tolerance<f32> = Tolerance::balanced();
    test_input.expected.assert_approx_eq(&first, tol);

    println!("Test success");
}

pub fn run_peer<B: Backend>(
    id: u32,
    reg_params: RegisterParams,
    params: AllReduceParams,
    input: TensorData,
    output: SyncSender<Tensor<B, 1>>,
) {
    let device = B::Device::default();

    register::<B>(id, reg_params);

    let tensor = Tensor::<B, 1>::from_data(input, &device);

    let tensor = all_reduce(tensor, params);

    output.send(tensor).unwrap();

    finish_collective::<B>();
}
