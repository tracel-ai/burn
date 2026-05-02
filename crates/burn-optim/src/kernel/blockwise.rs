////! Blockwise quantization #[cube] helpers, callable from a kernel.

//use cubecl::prelude::*;

//use crate::kernel::signed;
//use crate::kernel::unsigned;

//const PACKING_AMOUNT: u32 = 2;
//const PACK_SHIFT: u32 = 256;

//#[derive(Debug, Clone, Copy, PartialEq, Eq)]
//#[repr(u32)]
//pub enum Scheme {
//    Signed = 0,
//    Unsigned = 1,
//}

//#[cube]
//fn encode_dispatch(normalized: f32, #[comptime] scheme: u32) -> u32 {
//    // If/Else because of poor Cubecl support for match.
//    if comptime!(scheme == Scheme::Signed as u32) {
//        signed::encode(normalized)
//    } else if comptime!(scheme == Scheme::Unsigned as u32) {
//        unsigned::encode(normalized)
//    } else {
//        comptime! { panic!("Scheme not a proper value."); }
//    }
//}

//#[cube]
//fn decode_dispatch(code: u32, #[comptime] scheme: u32) -> f32 {
//    // If/Else because of poor Cubecl support for match.
//    if comptime!(scheme == Scheme::Signed as u32) {
//        signed::decode(code)
//    } else if comptime!(scheme == Scheme::Unsigned as u32) {
//        unsigned::decode(code)
//    } else {
//        comptime! { panic!("Scheme not a proper value."); }
//    }
//}

///// Quantize one block. Each unit in the cube processes its own element.
///// Cooperatively finds the block's absmax via plane_max, then each unit
///// normalizes and encodes; pairs of units pack their codes together.
/////
///// Inputs:
/////   - `value`: this unit's f32 value (already loaded from wherever)
/////   - `unit`: this unit's position within the block (0..block_size)
/////   - `block`: this unit's block index
/////   - `packed_out`, `scales_out`: output arrays
/////
///// Limitation: assumes `block_size <= plane_size` for the reduction.
//// #[cube]
//// pub fn quantize_blockwise(
////     value: f32,
////     neighbor_value: f32,
////     unit: u32,
////     block: u32,
////     packed_out: &mut Array<u32>,
////     scales_out: &mut Array<f32>,
////     #[comptime] block_size: u32,
////     #[comptime] scheme: u32,
//// ) {
////     let block_absmax = plane_max(value.abs());
////     let safe_scale = if block_absmax > 0.0f32 {
////         block_absmax
////     } else {
////         1.0f32.into()
////     };

////     if unit == 0 {
////         scales_out[block as usize] = safe_scale;
////     }

////     let normalized = value / safe_scale;
////     let code = encode_dispatch(normalized, scheme);

////     if unit % PACKING_AMOUNT == 0 {
////         let neighbor_normalized = neighbor_value / safe_scale;
////         let neighbor_code = encode_dispatch(neighbor_normalized, scheme);

////         let i = block * block_size + unit;
////         let pack_idx = i / PACKING_AMOUNT;
////         packed_out[pack_idx as usize] = code * PACK_SHIFT + neighbor_code;
////     }
//// }

//// #[cube]
//// pub fn quantize_blockwise(
////     value: f32,
////     neighbor_value: f32,
////     unit: u32,
////     block: u32,
////     packed_out: &mut Array<u32>,
////     scales_out: &mut Array<f32>,
////     #[comptime] block_size: u32,
////     #[comptime] scheme: u32,
//// ) {
////     // Pad inactive lanes with the identity for max-of-absolute-value (zero),
////     // so plane_max gives the correct block absmax even when the cube is
////     // larger than block_size. Inactive lanes contribute 0.0 and are
////     // ignored for everything downstream.
////     let abs_for_reduction = if unit < block_size {
////         value.abs()
////     } else {
////         0.0f32.into()
////     };
////     let block_absmax = plane_max(abs_for_reduction);
////     let safe_scale = if block_absmax > 0.0f32 {
////         block_absmax
////     } else {
////         1.0f32.into()
////     };

////     if unit < block_size {
////         if unit == 0 {
////             scales_out[block as usize] = safe_scale;
////         }

////         let normalized = value / safe_scale;
////         let code = encode_dispatch(normalized, scheme);

////         if unit % PACKING_AMOUNT == 0 {
////             let neighbor_normalized = neighbor_value / safe_scale;
////             let neighbor_code = encode_dispatch(neighbor_normalized, scheme);
////             let i = block * block_size + unit;
////             let pack_idx = i / PACKING_AMOUNT;
////             packed_out[pack_idx as usize] = code * PACK_SHIFT + neighbor_code;
////         }
////     }
//// }

//#[cube]
//fn cube_max_abs(value: f32, #[comptime] cube_dim: u32) -> f32 {
//    PLANE_DIM
//    plane_max

//    // const PLANE_SIZE: u32 = 64; // DO NOT USE ROCM

//    let num_planes = comptime!(cube_dim / PLANE_SIZE);

//    let plane_local_max = plane_max(value.abs());

//    let mut plane_maxes = SharedMemory::<f32>::new(num_planes as usize);
//    let plane_id = UNIT_POS_X / PLANE_SIZE;
//    let lane_id = UNIT_POS_X % PLANE_SIZE;

//    if lane_id == 0 {
//        plane_maxes[plane_id as usize] = plane_local_max;
//    }
//    sync_cube();

//    let across_planes = if plane_id == 0 && lane_id < num_planes {
//        plane_maxes[lane_id as usize]
//    } else {
//        0.0f32.into()
//    };
//    let block_max = plane_max(across_planes);

//    let mut broadcast = SharedMemory::<f32>::new(1usize);
//    if UNIT_POS_X == 0 {
//        broadcast[0] = block_max;
//    }
//    sync_cube();
//    broadcast[0]
//}

//#[cube]
//pub fn quantize_blockwise(
//    block: u32,
//    input: &Array<f32>,
//    packed_out: &mut Array<u32>,
//    scales_out: &mut Array<f32>,
//    #[comptime] block_size: u32,
//    // #[comptime] cube_dim: u32,
//    #[comptime] scheme: u32,
//) {
//    let cube_dim = 64;

//    let unit = UNIT_POS_X;
//    let block_start = block * block_size;
//    let elements_per_thread = comptime!(block_size / cube_dim);

//    // ---- Pass 1: local max ----
//    let mut local_max = 0.0f32;
//    #[unroll]
//    for iter in 0..elements_per_thread {
//        let element = unit + iter * cube_dim;
//        let i = block_start + element;
//        local_max = max(local_max, input[i as usize].abs());
//    }

//    // ---- Cross-thread reduction ----
//    let block_absmax = cube_max_abs(local_max, cube_dim);
//    let safe_scale = if block_absmax > 0.0f32 {
//        block_absmax
//    } else {
//        1.0f32.into()
//    };

//    if unit == 0 {
//        scales_out[block as usize] = safe_scale;
//    }

//    // ---- Pass 2: encode and pack (Strategy A: thread owns pairs) ----
//    let pairs_per_thread = comptime!(elements_per_thread / 2);
//    #[unroll]
//    for iter in 0..pairs_per_thread {
//        let element = unit * 2 + iter * cube_dim * 2;
//        let i = block_start + element;
//        let v0 = input[i as usize];
//        let v1 = input[(i + 1) as usize];
//        let code0 = encode_dispatch(v0 / safe_scale, scheme);
//        let code1 = encode_dispatch(v1 / safe_scale, scheme);

//        let pack_idx = i / PACKING_AMOUNT;
//        packed_out[pack_idx as usize] = code0 * PACK_SHIFT + code1;
//    }
//}

///// Dequantize one element of a packed blockwise tensor.
/////
///// Returns the f32 value at element index `i`. The caller decides
///// what to do with it (write to output, feed into a moment update, etc.).
//#[cube]
//pub fn dequantize_blockwise(
//    packed: &Array<u32>,
//    scales: &Array<f32>,
//    i: u32,
//    #[comptime] block_size: u32,
//    #[comptime] scheme: u32,
//) -> f32 {
//    let pack_idx = i / PACKING_AMOUNT;
//    let pack_pos = i % PACKING_AMOUNT;
//    let packed_val = packed[pack_idx as usize];

//    let code = if pack_pos == 0 {
//        packed_val / PACK_SHIFT
//    } else {
//        packed_val - (packed_val / PACK_SHIFT) * PACK_SHIFT
//    };

//    let normalized = decode_dispatch(code, scheme);
//    let block = i / block_size;
//    let scale = scales[block as usize];

//    normalized * scale
//}

//#[cfg(test)]
//mod tests {
//    use super::*;
//    use cubecl::bytes::Bytes;
//    use cubecl::cuda::CudaRuntime;
//    use cubecl::hip::HipRuntime;
//    use cubecl::prelude::*;
//    use cubecl::wgpu::WgpuRuntime;

//    type TestRuntime = WgpuRuntime;

//    const BLOCK_SIZE: u32 = 64;

//    // =========================================================================
//    //  Test-only kernels that wrap the #[cube] helpers
//    // =========================================================================

//    /// Drives `quantize_blockwise` from a launchable kernel.
//    /// One cube per block; each unit handles its own element.
//    #[cube(launch_unchecked)]
//    fn quantize_kernel(
//        input: &Array<f32>,
//        packed_out: &mut Array<u32>,
//        scales_out: &mut Array<f32>,
//        #[comptime] block_size: u32,
//        #[comptime] scheme: u32,
//    ) {
//        let block = CUBE_POS_X;
//        let unit = UNIT_POS_X;

//        // Inactive lanes pass dummy values; the helper ignores them anyway.
//        let value = if unit < block_size {
//            let i = block * block_size + unit;
//            input[i as usize]
//        } else {
//            0.0f32.into()
//        };
//        let neighbor = if unit + 1 < block_size {
//            let i = block * block_size + unit;
//            input[(i + 1) as usize]
//        } else {
//            0.0f32.into()
//        };

//        let block = CUBE_POS_X;
//        quantize_blockwise(block, input, packed_out, scales_out, block_size, scheme);

//        //         quantize_blockwise(
//        //             value, neighbor, unit, block, packed_out, scales_out, block_size, scheme,
//        //         );
//    }

//    /// Drives `dequantize_blockwise` from a launchable kernel.
//    /// One thread per output element.
//    #[cube(launch_unchecked)]
//    fn dequantize_kernel(
//        packed: &Array<u32>,
//        scales: &Array<f32>,
//        output: &mut Array<f32>,
//        #[comptime] block_size: u32,
//        #[comptime] scheme: u32,
//    ) {
//        let i = ABSOLUTE_POS;
//        if i < output.len() {
//            let i = i as u32;
//            output[i as usize] = dequantize_blockwise(packed, scales, i, block_size, scheme);
//        }
//    }

//    // =========================================================================
//    //  Host-side roundtrip harness
//    // =========================================================================

//    /// Quantize then dequantize a slice, returning the recovered values.
//    /// Block size must divide the input length evenly.
//    fn roundtrip_via_kernel<R: Runtime>(
//        client: &ComputeClient<R>,
//        input: &[f32],
//        block_size: u32,
//        scheme: Scheme,
//    ) -> Vec<f32> {
//        let n = input.len();
//        assert!(
//            n % block_size as usize == 0,
//            "input length {} must divide evenly by block_size {}",
//            n,
//            block_size,
//        );
//        let num_blocks = n as u32 / block_size;
//        let packed_count = n / PACKING_AMOUNT as usize;

//        // Upload input.
//        let input_bytes = f32::as_bytes(input).to_vec();
//        let input_handle = client.create(Bytes::from_bytes_vec(input_bytes));

//        // Allocate intermediate buffers (codes, scales) and output.
//        let packed_handle = client.empty(packed_count * core::mem::size_of::<u32>());
//        let scales_handle = client.empty(num_blocks as usize * core::mem::size_of::<f32>());
//        let output_handle = client.empty(n * core::mem::size_of::<f32>());

//        // let props = client.properties();
//        // let plane_size = props.hardware.plane_size_max;
//        let plane_size = 64;

//        // Quantize: one cube per block, block_size units per cube.
//        unsafe {
//            quantize_kernel::launch_unchecked::<R>(
//                client,
//                CubeCount::Static(num_blocks, 1, 1),
//                CubeDim::new(client, plane_size as usize),
//                ArrayArg::from_raw_parts(input_handle, n),
//                ArrayArg::from_raw_parts(packed_handle.clone(), packed_count),
//                ArrayArg::from_raw_parts(scales_handle.clone(), num_blocks as usize),
//                block_size,
//                scheme as u32,
//            );
//        }

//        // Dequantize: one thread per output element.
//        let dequant_dim = CubeDim::new(client, n);
//        let units_per_cube = dequant_dim.x * dequant_dim.y * dequant_dim.z;
//        let dequant_cubes = (n as u32).div_ceil(units_per_cube);

//        unsafe {
//            dequantize_kernel::launch_unchecked::<R>(
//                client,
//                CubeCount::Static(dequant_cubes, 1, 1),
//                dequant_dim,
//                ArrayArg::from_raw_parts(packed_handle, packed_count),
//                ArrayArg::from_raw_parts(scales_handle, num_blocks as usize),
//                ArrayArg::from_raw_parts(output_handle.clone(), n),
//                block_size,
//                scheme as u32,
//            );
//        }

//        // Read back recovered values.
//        let bytes: Bytes = client.read_one_unchecked(output_handle);
//        f32::from_bytes(&bytes).to_vec()
//    }

//    // =========================================================================
//    //  Tests — same scenarios as the tensor-ops version
//    // =========================================================================

//    #[test]
//    fn test_blockwise_roundtrip_uniform() {
//        let client = TestRuntime::client(&Default::default());

//        // All same value — trivial case.
//        let input: Vec<f32> = vec![0.5; 512];
//        let recovered =
//            roundtrip_via_kernel::<TestRuntime>(&client, &input, BLOCK_SIZE, Scheme::Signed);

//        let max_err = input
//            .iter()
//            .zip(recovered.iter())
//            .map(|(a, b)| ((a - b) / a).abs())
//            .fold(0.0f32, f32::max);

//        println!(
//            "Uniform 0.5 roundtrip max relative error: {:.4}%",
//            max_err * 100.0
//        );
//        assert!(max_err < 0.02, "max relative error too high: {}", max_err);
//    }

//    #[test]
//    fn test_blockwise_roundtrip_mixed() {
//        let client = TestRuntime::client(&Default::default());

//        // Mixed positive and negative values, -0.9..0.9.
//        let input: Vec<f32> = (0..512)
//            .map(|i| {
//                let t = i as f32 / 511.0;
//                (t * 2.0 - 1.0) * 0.9
//            })
//            .collect();

//        let recovered =
//            roundtrip_via_kernel::<TestRuntime>(&client, &input, BLOCK_SIZE, Scheme::Signed);

//        // Print outliers — same diagnostic as the tensor-ops test.
//        input
//            .iter()
//            .zip(recovered.iter())
//            .enumerate()
//            .filter(|(_, (a, _))| a.abs() > 0.01)
//            .for_each(|(i, (a, b))| {
//                let rel_err = ((a - b) / a).abs() * 100.0;
//                if rel_err > 5.0 {
//                    println!("  [{i}] {a:.7} -> {b:.7}  (err: {rel_err:.2}%)");
//                }
//            });

//        let max_err = input
//            .iter()
//            .zip(recovered.iter())
//            .filter(|(a, _)| a.abs() > 0.01)
//            .map(|(a, b)| ((a - b) / a).abs())
//            .fold(0.0f32, f32::max);

//        println!(
//            "Mixed roundtrip max relative error: {:.4}%",
//            max_err * 100.0
//        );
//        assert!(max_err < 0.10, "max relative error too high: {}", max_err);
//    }

//    #[test]
//    fn test_blockwise_roundtrip_unsigned() {
//        let client = TestRuntime::client(&Default::default());

//        // Non-negative, simulating moment 2.
//        let input: Vec<f32> = (0..512).map(|i| (i as f32 / 511.0) * 0.9 + 0.001).collect();

//        let recovered =
//            roundtrip_via_kernel::<TestRuntime>(&client, &input, BLOCK_SIZE, Scheme::Unsigned);

//        let max_err = input
//            .iter()
//            .zip(recovered.iter())
//            .map(|(a, b)| ((a - b) / a).abs())
//            .fold(0.0f32, f32::max);

//        println!(
//            "Unsigned roundtrip max relative error: {:.4}%",
//            max_err * 100.0
//        );
//        assert!(max_err < 0.05, "max relative error too high: {}", max_err);
//    }

//    #[test]
//    fn test_blockwise_minimal() {
//        let client = TestRuntime::client(&Default::default());
//        let block_size: u32 = 4; // tiny block, fits in any plane
//        let input: Vec<f32> = vec![
//            0.5, -0.5, 0.25, -0.25, // block 0
//            0.1, -0.1, 0.05, -0.05, // block 1
//        ];
//        let recovered =
//            roundtrip_via_kernel::<TestRuntime>(&client, &input, block_size, Scheme::Signed);

//        println!("input:     {:?}", input);
//        println!("recovered: {:?}", recovered);
//    }

//    #[test]
//    fn what_is_my_plane_size() {
//        let client = TestRuntime::client(&Default::default());
//        let props = client.properties();
//        println!("plane size: {}", props.hardware.plane_size_max);
//        println!("max units per cube: {}", props.hardware.max_units_per_cube);
//    }
//}

//! Blockwise quantization #[cube] helpers, callable from a kernel.
//!
//! Current scope: block_size == PLANE_SIZE only. One element per thread,
//! single-plane reduction. Larger block sizes require hierarchical reduction
//! and per-thread loops; those are deferred.

use cubecl::prelude::*;

use crate::kernel::signed;
use crate::kernel::unsigned;

const PACKING_AMOUNT: u32 = 2;
const PACK_SHIFT: u32 = 256;

/// Plane size for the target hardware. Hardcoded for AMD RDNA 3 (7900 XTX).
/// On Apple/NVIDIA/older AMD this would be 32. To support both, promote to
/// a #[comptime] kernel parameter sourced from `client.properties()`.
const PLANE_SIZE: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Scheme {
    Signed = 0,
    Unsigned = 1,
}

#[cube]
fn encode_dispatch(normalized: f32, #[comptime] scheme: u32) -> u32 {
    if comptime!(scheme == Scheme::Signed as u32) {
        signed::encode(normalized)
    } else if comptime!(scheme == Scheme::Unsigned as u32) {
        unsigned::encode(normalized)
    } else {
        comptime! { panic!("Scheme not a proper value."); }
    }
}

#[cube]
fn decode_dispatch(code: u32, #[comptime] scheme: u32) -> f32 {
    if comptime!(scheme == Scheme::Signed as u32) {
        signed::decode(code)
    } else if comptime!(scheme == Scheme::Unsigned as u32) {
        unsigned::decode(code)
    } else {
        comptime! { panic!("Scheme not a proper value."); }
    }
}

#[cube]
pub fn quantize_blockwise(
    block: u32,
    input: &Array<f32>,
    packed_out: &mut Array<u32>,
    scales_out: &mut Array<f32>,
    #[comptime] block_size: u32,
    #[comptime] scheme: u32,
) {
    let unit = UNIT_POS_X;
    let block_start = block * block_size;
    let elements_per_thread = comptime!(block_size / PLANE_SIZE);

    // ---- Pass 1: each thread finds its local max across its elements ----
    let mut local_max = 0.0f32;
    #[unroll]
    for iter in 0..elements_per_thread {
        let element = unit + iter * PLANE_SIZE;
        let i = block_start + element;
        local_max = max(local_max, input[i as usize].abs());
    }

    // ---- Cross-thread reduction (single plane) ----
    let block_absmax = plane_max(local_max);
    let safe_scale = if block_absmax > 0.0f32 {
        block_absmax
    } else {
        1.0f32.into()
    };

    if unit == 0 {
        scales_out[block as usize] = safe_scale;
    }

    // ---- Pass 2: each thread encodes its elements and packs pairs ----
    // Strategy A: each thread handles consecutive pairs, so packing is
    // local — no inter-thread communication needed.
    let pairs_per_thread = comptime!(elements_per_thread / 2);
    #[unroll]
    for iter in 0..pairs_per_thread {
        // Each thread `unit` owns elements at positions
        // (unit*2 + iter * PLANE_SIZE * 2) and (... + 1) within the block.
        let element = unit * 2 + iter * PLANE_SIZE * 2;
        let i = block_start + element;

        let v0 = input[i as usize];
        let v1 = input[i as usize + 1];
        let code0 = encode_dispatch(v0 / safe_scale, scheme);
        let code1 = encode_dispatch(v1 / safe_scale, scheme);

        let pack_idx = i / PACKING_AMOUNT;
        packed_out[pack_idx as usize] = code0 * PACK_SHIFT + code1;
    }
}

#[cube]
pub fn dequantize_blockwise(
    packed: &Array<u32>,
    scales: &Array<f32>,
    i: u32,
    #[comptime] block_size: u32,
    #[comptime] scheme: u32,
) -> f32 {
    let pack_idx = i / PACKING_AMOUNT;
    let pack_pos = i % PACKING_AMOUNT;
    let packed_val = packed[pack_idx as usize];

    let code = if pack_pos == 0 {
        packed_val / PACK_SHIFT
    } else {
        packed_val - (packed_val / PACK_SHIFT) * PACK_SHIFT
    };

    let normalized = decode_dispatch(code, scheme);
    let block = i / block_size;
    let scale = scales[block as usize];

    normalized * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::bytes::Bytes;
    use cubecl::hip::HipRuntime; // reports 32 plane size
    use cubecl::wgpu::WgpuRuntime;

    type TestRuntime = WgpuRuntime;

    /// Block size for tests. Must equal PLANE_SIZE (64) until larger blocks
    /// are supported via hierarchical reduction.
    const BLOCK_SIZE: u32 = 256;

    #[cube(launch_unchecked)]
    fn quantize_kernel(
        input: &Array<f32>,
        packed_out: &mut Array<u32>,
        scales_out: &mut Array<f32>,
        #[comptime] block_size: u32,
        #[comptime] scheme: u32,
    ) {
        let block = CUBE_POS_X;
        quantize_blockwise(block, input, packed_out, scales_out, block_size, scheme);
    }

    #[cube(launch_unchecked)]
    fn dequantize_kernel(
        packed: &Array<u32>,
        scales: &Array<f32>,
        output: &mut Array<f32>,
        #[comptime] block_size: u32,
        #[comptime] scheme: u32,
    ) {
        let i = ABSOLUTE_POS as u32;
        if (i as usize) < output.len() {
            output[i as usize] = dequantize_blockwise(packed, scales, i, block_size, scheme);
        }
    }

    fn roundtrip_via_kernel<R: Runtime>(
        client: &ComputeClient<R>,
        input: &[f32],
        block_size: u32,
        scheme: Scheme,
    ) -> Vec<f32> {
        let n = input.len();
        assert_eq!(
            n % block_size as usize,
            0,
            "input length {} must divide evenly by block_size {}",
            n,
            block_size,
        );
        let num_blocks = n as u32 / block_size;
        let packed_count = n / PACKING_AMOUNT as usize;

        let input_bytes = f32::as_bytes(input).to_vec();
        let input_handle = client.create(Bytes::from_bytes_vec(input_bytes));
        let packed_handle = client.empty(packed_count * core::mem::size_of::<u32>());
        let scales_handle = client.empty(num_blocks as usize * core::mem::size_of::<f32>());
        let output_handle = client.empty(n * core::mem::size_of::<f32>());

        // Quantize: one cube per block, block_size units per cube.
        unsafe {
            quantize_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_blocks, 1, 1),
                CubeDim::new(client, PLANE_SIZE as usize), // ← always 64
                ArrayArg::from_raw_parts(input_handle, n),
                ArrayArg::from_raw_parts(packed_handle.clone(), packed_count),
                ArrayArg::from_raw_parts(scales_handle.clone(), num_blocks as usize),
                block_size,
                scheme as u32,
            );
        }

        // Dequantize: one thread per output element.
        let dequant_dim = CubeDim::new(client, n);
        let units_per_cube = dequant_dim.x * dequant_dim.y * dequant_dim.z;
        let dequant_cubes = (n as u32).div_ceil(units_per_cube);

        unsafe {
            dequantize_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(dequant_cubes, 1, 1),
                dequant_dim,
                ArrayArg::from_raw_parts(packed_handle, packed_count),
                ArrayArg::from_raw_parts(scales_handle, num_blocks as usize),
                ArrayArg::from_raw_parts(output_handle.clone(), n),
                block_size,
                scheme as u32,
            );
        }

        let bytes: Bytes = client.read_one_unchecked(output_handle);
        f32::from_bytes(&bytes).to_vec()
    }

    #[test]
    fn test_blockwise_roundtrip_uniform() {
        let client = TestRuntime::client(&Default::default());
        let input: Vec<f32> = vec![0.5; 512];
        let recovered =
            roundtrip_via_kernel::<TestRuntime>(&client, &input, BLOCK_SIZE, Scheme::Signed);

        let max_err = input
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f32, f32::max);

        println!(
            "Uniform 0.5 roundtrip max relative error: {:.4}%",
            max_err * 100.0
        );
        assert!(max_err < 0.02, "max relative error too high: {}", max_err);
    }

    #[test]
    fn test_blockwise_roundtrip_mixed() {
        let client = TestRuntime::client(&Default::default());
        let input: Vec<f32> = (0..512)
            .map(|i| (i as f32 / 511.0 * 2.0 - 1.0) * 0.9)
            .collect();

        let recovered =
            roundtrip_via_kernel::<TestRuntime>(&client, &input, BLOCK_SIZE, Scheme::Signed);

        input
            .iter()
            .zip(recovered.iter())
            .enumerate()
            .filter(|(_, (a, _))| a.abs() > 0.01)
            .for_each(|(i, (a, b))| {
                let rel_err = ((a - b) / a).abs() * 100.0;
                if rel_err > 5.0 {
                    println!("  [{i}] {a:.7} -> {b:.7}  (err: {rel_err:.2}%)");
                }
            });

        let max_err = input
            .iter()
            .zip(recovered.iter())
            .filter(|(a, _)| a.abs() > 0.01)
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f32, f32::max);

        println!(
            "Mixed roundtrip max relative error: {:.4}%",
            max_err * 100.0
        );
        assert!(max_err < 0.10, "max relative error too high: {}", max_err);
    }

    #[test]
    fn test_blockwise_roundtrip_unsigned() {
        let client = TestRuntime::client(&Default::default());
        let input: Vec<f32> = (0..512).map(|i| (i as f32 / 511.0) * 0.9 + 0.001).collect();

        let recovered =
            roundtrip_via_kernel::<TestRuntime>(&client, &input, BLOCK_SIZE, Scheme::Unsigned);

        let max_err = input
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f32, f32::max);

        println!(
            "Unsigned roundtrip max relative error: {:.4}%",
            max_err * 100.0
        );
        assert!(max_err < 0.05, "max relative error too high: {}", max_err);
    }
}
