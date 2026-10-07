//! Fused trailing update and panel Cholesky for CubeCL.
//!
//! Each launch applies the preceding panel's rank-B update, factors the next
//! panel, and leaves the trailing Schur complement in a separate workspace.
//! Panel workgroups independently factor the small diagonal tile. They write
//! only the factor buffer; other workgroups update disjoint workspace tiles.
//! This avoids inter-workgroup barriers and needs only ceil(n / B) launches.
use alloc::vec::Vec;
use burn_core as burn;
use burn_core::backend::{
    Backend, TensorMetadata, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn_cubecl::{CubeBackend, cubecl, ops::numeric::empty_device_contiguous_dtype};
use burn_std::{DType, Shape};
use cubecl::prelude::*;

const BLOCK: usize = 32;

/// The subgroup kernel requires a fixed 32-lane plane. Other hardware keeps
/// the portable tensor path; never assume a subgroup width on variable-width
/// devices. Bound launch dimensions and the default 32-bit element addresses.
pub(crate) fn supported(device: &burn_cubecl::CubeDevice, n: usize, batch: usize) -> bool {
    let client = device.client();
    let properties = client.properties();
    let hardware = &properties.hardware;
    let blocks = n.div_ceil(BLOCK);
    hardware.plane_size_min == 32
        && hardware.plane_size_max == 32
        && properties
            .features
            .plane
            .contains(cubecl::features::Plane::Ops)
        && hardware.max_shared_memory_size >= 4 * BLOCK * BLOCK * size_of::<f32>()
        && hardware.max_units_per_cube >= 128
        && hardware.max_cube_dim.0 >= 128
        && blocks <= hardware.max_cube_count.0.min(hardware.max_cube_count.1) as usize
        && batch <= hardware.max_cube_count.1.min(hardware.max_cube_count.2) as usize
        && n.checked_mul(n)
            .and_then(|v| v.checked_mul(batch))
            .is_some_and(|v| v <= u32::MAX as usize)
}

#[backend_extension(
    Cube: cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan",
        feature = "metal", feature = "cuda", feature = "rocm", feature = "cpu")),
    Fusion: cfg(feature = "fusion")
)]
pub(crate) trait CholeskyGpuOps: Backend {
    #[fusion(meta = |tensor, _upper| {
        let n = tensor.shape[tensor.shape.num_dims() - 1];
        let batch = tensor.shape.num_elements() / (n * n);
        (tensor.clone(), burn::backend::fusion::custom::TensorSpec::new(
            Shape::new([batch, n.div_ceil(BLOCK)]), DType::I32))
    })]
    fn cholesky_gpu(tensor: FloatTensor<Self>, upper: bool)
    -> (FloatTensor<Self>, IntTensor<Self>);
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn panel(
    input: &Tensor<f32>,
    output: &mut Tensor<f32>,
    workspace: &mut Tensor<f32>,
    info: &mut Tensor<i32>,
    n: usize,
    start: usize,
    #[comptime] block: usize,
    #[comptime] upper: bool,
) {
    let batch = CUBE_POS_Z as usize;
    let tile = CUBE_POS_X as usize;
    let tile_col = CUBE_POS_Y as usize;
    if tile < tile_col {
        terminate!();
    }
    let col_start = start + tile_col * block;
    let row_start = start + tile * block;
    let tid = UNIT_POS as usize;
    let threads = CUBE_DIM as usize;
    let base = batch * n * n;
    let rank = input.rank();
    let mut input_base = 0;
    let mut remaining = batch;
    for d in 0..rank - 2 {
        let axis = rank - 3 - d;
        input_base += (remaining % input.shape(axis)) * input.stride(axis);
        remaining /= input.shape(axis);
    }
    let rs = input.stride(rank - 2);
    let cs = input.stride(rank - 1);
    let mut diagonal = Shared::<[f32]>::new_slice(block * block);
    let mut below = Shared::<[f32]>::new_slice(block * block);
    let mut previous_diagonal = Shared::<[f32]>::new_slice(block * block);
    let mut previous_below = Shared::<[f32]>::new_slice(block * block);
    let mut failure = 0i32;
    let mut index = tid;
    while index < block * block {
        let r = index % block;
        let c = index / block;
        let i = start + r;
        let j = col_start + c;
        let row = row_start + r;
        let mut d = 0.0f32;
        let mut b = 0.0f32;
        if tile_col == 0 && i < n && j < n && r >= c {
            if start > block {
                d = workspace[base + j * n + i];
            } else if upper {
                d = input[input_base + j * rs + i * cs];
            } else {
                d = input[input_base + i * rs + j * cs];
            }
        }
        if tile > 0 && row < n && j < n && row >= j {
            if start > block {
                b = workspace[base + j * n + row];
            } else if upper {
                b = input[input_base + j * rs + row * cs];
            } else {
                b = input[input_base + row * rs + j * cs];
            }
        }
        diagonal[index] = d;
        below[index] = b;
        index += threads;
    }
    sync_cube();
    if start > 0 {
        let previous = start - block;
        let mut index = tid;
        while index < block * block {
            let r = index % block;
            let k = previous + index / block;
            let mut d = 0.0f32;
            let mut b = 0.0f32;
            if col_start + r < n {
                d = output[base + k * n + col_start + r];
            }
            if tile > 0 && row_start + r < n {
                b = output[base + k * n + row_start + r];
            }
            previous_diagonal[index] = d;
            previous_below[index] = b;
            index += threads;
        }
        sync_cube();
        let mut index = tid;
        while index < block * block {
            let r = index % block;
            let c = index / block;
            let mut d = 0.0f32;
            let mut b = 0.0f32;
            #[unroll]
            for k in 0..block {
                let right = previous_diagonal[k * block + c];
                if tile_col == 0 {
                    d += previous_diagonal[k * block + r] * right;
                }
                if tile > 0 {
                    b += previous_below[k * block + r] * right;
                }
            }
            diagonal[index] -= d;
            below[index] -= b;
            index += threads;
        }
        sync_cube();
    }
    if tile_col > 0 {
        let mut index = tid;
        while index < block * block {
            let row = row_start + index % block;
            let col = col_start + index / block;
            if row < n && col < n && row >= col {
                workspace[base + col * n + row] = below[index];
            }
            index += threads;
        }
        terminate!();
    }
    // One lane owns each row; subgroup broadcasts publish pivot columns.
    if tid < block {
        let mut d = Array::<f32>::new(block);
        let mut b = Array::<f32>::new(block);
        #[unroll]
        for c in 0..block {
            d[c] = diagonal[c * block + tid];
            b[c] = below[c * block + tid];
        }
        #[unroll]
        for j in 0..block {
            if start + j < n {
                let pivot = plane_broadcast(d[j], j as u32);
                let valid = pivot > 0.0 && pivot <= 3.402_823_5e38_f32;
                let mut root = 1.0f32;
                if valid {
                    root = f32::sqrt(pivot);
                } else if failure == 0 {
                    failure = (start + j + 1) as i32;
                }
                let inverse = 1.0 / root;
                let mut v = d[j] * inverse;
                if tid == j {
                    v = root;
                }
                let w = b[j] * inverse;
                #[unroll]
                for c in 0..block {
                    let coefficient = plane_broadcast(v, c as u32);
                    if c > j {
                        d[c] -= v * coefficient;
                        b[c] -= w * coefficient;
                    }
                }
                d[j] = v;
                b[j] = w;
            }
        }
        #[unroll]
        for c in 0..block {
            let row = row_start + tid;
            let col = start + c;
            if row < n && col < n && row >= col {
                let mut value = d[c];
                if tile > 0 {
                    value = b[c];
                }
                output[base + col * n + row] = value;
                if row != col {
                    output[base + row * n + col] = 0.0;
                }
            }
        }
        if tile == 0 && tid == 0 {
            let info_index = batch * info.shape(1) + start / block;
            info[info_index] = failure;
        }
    }
}

impl CholeskyGpuOps for CubeBackend {
    fn cholesky_gpu(
        tensor: FloatTensor<Self>,
        upper: bool,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let tensor = burn_cubecl::kernel::untile(tensor);
        assert_eq!(tensor.dtype, DType::F32);
        let shape = tensor.shape();
        let rank = shape.num_dims();
        let n = shape[rank - 1];
        let batch = shape.num_elements() / (n * n);
        assert!(
            supported(&tensor.device, n, batch),
            "unsupported Cholesky GPU launch"
        );
        let client = tensor.client.clone();
        let output =
            empty_device_contiguous_dtype(client.clone(), tensor.device.clone(), shape, DType::F32);
        let workspace = empty_device_contiguous_dtype(
            client.clone(),
            tensor.device.clone(),
            tensor.shape(),
            DType::F32,
        );
        let info = empty_device_contiguous_dtype(
            client.clone(),
            tensor.device.clone(),
            Shape::new([batch, n.div_ceil(BLOCK)]),
            DType::I32,
        );
        // Views may refer to a larger allocation than their logical shape.
        let address_type = tensor
            .required_address_type()
            .max(output.required_address_type())
            .max(workspace.required_address_type())
            .max(info.required_address_type());
        for start in (0..n).step_by(BLOCK) {
            // SAFETY: supported() bounds grid and address sizes. Every global
            // row/column access guards the matrix tails. Shared indices are
            // below BLOCK²; previous factor columns exist when start > 0.
            // All threads reach workgroup barriers uniformly. Broadcasts use
            // one complete 32-lane subgroup and source lanes 0..32.
            // Panel groups read workspace tiles that this launch never writes.
            // Other groups read/write only their own workspace tile. Factor
            // writes are disjoint, and reads use only completed prior columns.
            unsafe {
                panel::launch_unchecked(
                    &client,
                    CubeCount::Static(
                        (n - start).div_ceil(BLOCK) as u32,
                        if start == 0 {
                            1
                        } else {
                            (n - start).div_ceil(BLOCK) as u32
                        },
                        batch as u32,
                    ),
                    CubeDim::new_1d(128),
                    address_type,
                    tensor.clone().into_tensor_arg(),
                    output.clone().into_tensor_arg(),
                    workspace.clone().into_tensor_arg(),
                    info.clone().into_tensor_arg(),
                    n,
                    start,
                    BLOCK,
                    upper,
                );
            }
        }
        let output = if upper {
            output
        } else {
            let mut axes: Vec<_> = (0..rank).collect();
            axes.swap(rank - 2, rank - 1);
            burn_cubecl::ops::permute(output, &axes)
        };
        (output, info)
    }
}
