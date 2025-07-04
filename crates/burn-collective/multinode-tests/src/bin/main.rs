use std::{
    fs::{self, File},
    process::{Child, Command},
};

use burn::{
    backend::{
        NdArray,
        collective::{AllReduceParams, AllReduceStrategy, ReduceKind},
    },
    prelude::Backend,
    tensor::{Shape, Tensor, TensorData},
};
use burn_collective_multinode_tests::shared::NodeTestData;
use burn_common::rand::{SeedableRng, StdRng};

use serde_json::to_writer_pretty;

/// Main function to run the multinode all-reduce test.
/// Launches a server and multiple clients based on the provided topology.
fn main() {
    let test_files_dir = "target/test_files";
    fs::create_dir_all(test_files_dir).expect("Couldn't create test_files directory");

    let server_out_path = format!("{}/server_out.txt", test_files_dir);
    let server_out = File::create(server_out_path).expect("Could't create sever ouput file");

    let topology = vec![5, 5, 5, 5, 5];
    let tensor_shape = Shape { dims: vec![4] };
    let aggregate_params = AllReduceParams {
        kind: ReduceKind::Sum,
        local_strategy: AllReduceStrategy::Tree(2),
        global_strategy: Some(AllReduceStrategy::Tree(2)),
    };

    let mut server: Child = Command::new("cargo")
        .args(["run", "--bin", "server", "--", "3000"])
        .stdout(server_out.try_clone().unwrap())
        .stderr(server_out)
        .spawn()
        .expect("failed to launch server");

    let clients = launch_clients(topology, tensor_shape, aggregate_params);

    // Wait on every client
    for mut client in clients {
        let _ = client.wait();
    }

    // Shutdown server
    let _ = server.kill();
    let _ = server.wait();
}

/// Launch clients based on the provided topology.
/// Each client will connect to the server and run an all-reduce operation.
/// The topology is a vector where each element represents the number of devices in that node.
fn launch_clients(
    topology: Vec<u32>,
    tensor_shape: Shape,
    aggregate_params: AllReduceParams,
) -> Vec<Child> {
    let total_device_count = topology.iter().sum();
    let (inputs, expected) =
        generate_random_input(tensor_shape, aggregate_params.kind, total_device_count, 42);

    let server_url = "ws://localhost:3000";

    let node_count = topology.len();
    let mut clients = vec![];
    let mut launched_devices: usize = 0;
    for (node_idx, &device_count) in topology.iter().enumerate() {
        let client_data_port = node_idx as u16 + 3001;
        let client_url = format!("ws://localhost:{}", client_data_port);
        let input_filename = format!("target/test_files/client_{}_in.txt", node_idx + 1);
        let output_filename = format!("target/test_files/client_{}_out.txt", node_idx + 1);

        let upper = launched_devices + device_count as usize;
        let inputs = inputs[launched_devices..upper].to_vec();
        launched_devices += device_count as usize;

        let data = NodeTestData {
            device_count,
            node_id: node_idx as u32,
            node_count: node_count as u32,
            server_url: server_url.to_owned(),
            client_url: client_url.to_owned(),
            client_data_port,
            aggregate_params: aggregate_params.clone(),
            inputs,
            expected: expected.clone(),
        };

        let file = File::create(&input_filename).expect("Failed to create file");
        to_writer_pretty(file, &data).expect("Failed to write JSON");

        let client_out = File::create(output_filename).expect("Could't create client ouput file");

        let client: Child = Command::new("cargo")
            .args(["run", "--bin", "client", "--", &input_filename])
            .stdout(client_out.try_clone().unwrap())
            .stderr(client_out)
            .spawn()
            .expect("client failed");

        clients.push(client);
    }

    clients
}

/// Generates random input tensors based on the provided shape and reduce kind.
fn generate_random_input(
    shape: Shape,
    reduce_kind: ReduceKind,
    input_count: u32,
    seed: u64,
) -> (Vec<TensorData>, TensorData) {
    let mut rng = StdRng::seed_from_u64(seed);

    let input: Vec<TensorData> = (0..input_count)
        .map(|_| {
            TensorData::random::<f32, _, _>(
                shape.clone(),
                burn::tensor::Distribution::Default,
                &mut rng,
            )
        })
        .collect();

    let device = <NdArray as Backend>::Device::default();

    let mut expected_tensor = Tensor::<NdArray, 1>::zeros(shape, &device);
    for item in input.iter().take(input_count as usize) {
        let input_tensor = Tensor::<NdArray, 1>::from_data(item.clone(), &device);
        expected_tensor = expected_tensor.add(input_tensor);
    }

    if reduce_kind == ReduceKind::Mean {
        expected_tensor = expected_tensor.div_scalar(input_count);
    }

    let expected = expected_tensor.to_data();

    (input, expected)
}
