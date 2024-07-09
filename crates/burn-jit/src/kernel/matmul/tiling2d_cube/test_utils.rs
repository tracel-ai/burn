use burn_compute::server::Handle;
use burn_cube::CubeElement;

use crate::{
    kernel::matmul::config::{CubeTiling2dConfig, Tiling2dConfig},
    tensor::JitTensor,
    JitBackend, JitRuntime,
};

pub(crate) const TILE_SIZE: usize = 4;

pub(crate) fn range_tensor<R: JitRuntime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> JitTensor<R, f32, 2> {
    type B<R> = JitBackend<R, f32, i32>;

    let n_elements = (x * y) as i64;
    burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..n_elements, device)
        .reshape([x, y])
        .float()
        .into_primitive()
        .tensor()
}

pub(crate) fn range_tensor_transposed<R: JitRuntime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> JitTensor<R, f32, 2> {
    type B<R> = JitBackend<R, f32, i32>;

    let n_elements = (x * y) as i64;

    burn_tensor::Tensor::<B<R>, 2>::from_data(
        burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..n_elements, device)
            .reshape([x, y])
            .float()
            .transpose()
            .into_data(),
        device,
    )
    .into_primitive()
    .tensor()
}

pub(crate) fn zeros_tensor<R: JitRuntime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> JitTensor<R, f32, 2> {
    type B<R> = JitBackend<R, f32, i32>;
    burn_tensor::Tensor::<B<R>, 2>::zeros([x, y], device)
        .into_primitive()
        .tensor()
}

pub(crate) fn create_empty<R: JitRuntime>(
    x: usize,
    y: usize,
    device: &R::Device,
) -> Handle<<R as JitRuntime>::JitServer> {
    let client = R::client(device);
    client.empty(x * y * core::mem::size_of::<f32>())
}

pub(crate) fn assert_equals<R: JitRuntime>(
    output: Handle<<R as JitRuntime>::JitServer>,
    expected: &[f32],
    device: &R::Device,
) {
    let client = R::client(device);

    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, expected);
}

pub(crate) fn make_config(m: usize, k: usize, n: usize) -> CubeTiling2dConfig {
    let tiling2d_config = Tiling2dConfig {
        block_size_m: 8,
        block_size_k: 8,
        block_size_n: 8,
        ..Default::default()
    };
    CubeTiling2dConfig::new(&tiling2d_config, m, k, n, false, false)
}
