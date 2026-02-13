use std::sync::mpsc::SyncSender;

use burn_autodiff::Autodiff;
use burn_collective::{AllReduceStrategy, CollectiveConfig, register, reset_collective};
use burn_tensor::{
    Tensor, TensorData, Tolerance,
    backend::{AutodiffBackend, Backend, PeerId, ReduceOperation},
};
use rand::rngs::StdRng;
use rand::{SeedableRng, rngs::SysRng};
use serial_test::serial;

pub type TestBackend = burn_ndarray::NdArray<f32>;
pub type TestAutodiffBackend = Autodiff<TestBackend>;

pub fn run_peer_sharded<B>(
    id: PeerId,
    config: CollectiveConfig,
    input: TensorData,
    op: ReduceOperation,
    output: SyncSender<Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    transformation: fn(Tensor<B, 1>) -> Tensor<B, 1>,
) where
    B: AutodiffBackend,
{
    let device = B::Device::default();

    register::<<B as AutodiffBackend>::InnerBackend>(id, device.clone(), config).unwrap();

    let input_tensor = Tensor::<B, 1>::from_data(input, &device)
        .require_grad()
        .set_sharded_params(id, op);
    let out_tensor = transformation(input_tensor.clone());
    let grads = out_tensor.backward();

    let tensor_grad = input_tensor.grad(&grads).unwrap();

    output.send(tensor_grad).unwrap();
}

fn generate_random_input_autodiff<B>(
    shape: Vec<usize>,
    op: ReduceOperation,
    thread_count: usize,
    transformation: fn(Tensor<B, 1>) -> Tensor<B, 1>,
) -> (Vec<TensorData>, TensorData)
where
    B: AutodiffBackend,
{
    let input: Vec<TensorData> = (0..thread_count)
        .map(|_| {
            TensorData::random::<f32, _, _>(
                shape.clone(),
                burn_tensor::Distribution::Default,
                &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
            )
        })
        .collect();

    let device = <B as Backend>::Device::default();

    let mut expected_tensor =
        Tensor::<<B as AutodiffBackend>::InnerBackend, 1>::zeros(shape, &device);
    for item in input.iter().take(thread_count as usize) {
        let input_tensor = Tensor::<B, 1>::from_data(item.clone(), &device).require_grad();
        let out_tensor = transformation(input_tensor.clone());
        let grads = out_tensor.backward();
        let grads = input_tensor.grad(&grads).unwrap();
        expected_tensor = expected_tensor.add(grads);
    }
    if op == ReduceOperation::Mean {
        expected_tensor = expected_tensor.div_scalar(thread_count as u32);
    }

    let expected = expected_tensor.to_data();

    (input, expected)
}

fn test_all_reduce<B>(
    device_count: usize,
    op: ReduceOperation,
    strategy: AllReduceStrategy,
    tensor_size: usize,
    transformation: fn(Tensor<B, 1>) -> Tensor<B, 1>,
) where
    B: AutodiffBackend,
{
    reset_collective::<<TestAutodiffBackend as AutodiffBackend>::InnerBackend>();

    let (send, recv) = std::sync::mpsc::sync_channel(32);

    let shape = vec![tensor_size];

    let (input, expected) = generate_random_input_autodiff(shape, op, device_count, transformation);

    let config = CollectiveConfig::default()
        .with_num_devices(device_count)
        .with_local_all_reduce_strategy(strategy);

    for id in 0..device_count {
        let send = send.clone();
        let input = input[id as usize].clone();

        std::thread::spawn({
            let config = config.clone();
            move || run_peer_sharded::<B>(id.into(), config, input, op, send, transformation)
        });
    }

    let first = recv.recv().unwrap().to_data();
    for _ in 1..device_count {
        let tensor = recv.recv().unwrap();
        tensor.to_data().assert_eq(&first, true);
    }

    let tol: Tolerance<f32> = Tolerance::balanced();
    expected.assert_approx_eq(&first, tol);
}

#[test]
#[serial]
pub fn test_all_reduce_centralized_sum() {
    test_all_reduce::<TestAutodiffBackend>(
        4,
        ReduceOperation::Sum,
        AllReduceStrategy::Centralized,
        4,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_centralized_mean() {
    test_all_reduce::<TestAutodiffBackend>(
        4,
        ReduceOperation::Mean,
        AllReduceStrategy::Centralized,
        4,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_binary_tree_sum() {
    test_all_reduce::<TestAutodiffBackend>(
        4,
        ReduceOperation::Sum,
        AllReduceStrategy::Tree(2),
        4,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_binary_tree_mean() {
    test_all_reduce::<TestAutodiffBackend>(
        4,
        ReduceOperation::Mean,
        AllReduceStrategy::Tree(2),
        4,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_5_tree_sum() {
    test_all_reduce::<TestAutodiffBackend>(
        4,
        ReduceOperation::Sum,
        AllReduceStrategy::Tree(5),
        4,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_5_tree_mean() {
    test_all_reduce::<TestAutodiffBackend>(
        4,
        ReduceOperation::Mean,
        AllReduceStrategy::Tree(5),
        4,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_ring_sum() {
    test_all_reduce::<TestAutodiffBackend>(
        3,
        ReduceOperation::Sum,
        AllReduceStrategy::Ring,
        3,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_ring_mean() {
    test_all_reduce::<TestAutodiffBackend>(
        3,
        ReduceOperation::Mean,
        AllReduceStrategy::Ring,
        3,
        |tensor| tensor,
    );
}

#[test]
#[serial]
pub fn test_all_reduce_ring_irregular_sum() {
    // this should trigger the fallback algorithm when the tensor is too small.
    test_all_reduce::<TestAutodiffBackend>(
        4,
        ReduceOperation::Sum,
        AllReduceStrategy::Ring,
        3,
        |tensor| tensor,
    );
}
