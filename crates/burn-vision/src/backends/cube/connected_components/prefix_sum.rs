use burn_tensor::Shape;
use cubecl::prelude::*;

use burn_cubecl::{
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::CubeTensor,
    CubeRuntime, IntElement,
};

const CUBE_SIZE: u32 = 256;
const MIN_SUBGROUP_SIZE: u32 = 4;
const MAX_REDUCE_SIZE: u32 = CUBE_SIZE / MIN_SUBGROUP_SIZE;

const PART_SIZE: u32 = 4096;

#[cube(launch_unchecked)]
fn prefix_sum_kernel<I: Int>(
    scan_in: &Tensor<Line<I>>,
    scan_out: &mut Tensor<Line<I>>,
    scan_bump: &Tensor<Atomic<I>>,
    reduction: &Tensor<Atomic<I>>,
    cube_count_x: u32,
) {
    let mut broadcast = SharedMemory::<I>::new(1);
    let mut reduce = SharedMemory::<I>::new(MAX_REDUCE_SIZE);
    let batch = CUBE_POS_Z;
    let line_spt = comptime!(PART_SIZE / CUBE_SIZE / scan_in.line_size());
    let nums_per_cube = CUBE_SIZE * line_spt;
    let v_last = comptime!(scan_in.line_size() - 1);

    //acquire partition index
    if UNIT_POS_X == 0 {
        broadcast[0] = Atomic::add(&scan_bump[batch], I::new(1));
    }
    sync_units();
    let part_id = u32::cast_from(broadcast[0]);

    let plane_id = UNIT_POS_X / PLANE_DIM;
    let dev_offs = part_id * nums_per_cube;
    let plane_offs = plane_id * PLANE_DIM * line_spt;

    // Exit if full plane is out of bounds
    if dev_offs + plane_offs >= scan_in.shape(1) {
        terminate!();
    }

    let zero = I::new(0);

    let flag_reduction = I::new(1);
    let flag_inclusive = I::new(2);
    let flag_mask = I::new(3);

    let red_offs = batch * reduction.stride(0);
    let scan_offs = batch * scan_in.stride(0);

    let mut t_scan = Array::<Line<I>>::vectorized(line_spt, scan_in.line_size());
    {
        let mut i = dev_offs + plane_offs + UNIT_POS_PLANE;

        if part_id < cube_count_x - 1 {
            for k in 0..line_spt {
                // Manually fuse not_equal and cast
                let mut scan = Line::cast_from(scan_in[i + scan_offs].not_equal(Line::new(zero)));
                #[unroll]
                for v in 1..scan_in.line_size() {
                    let prev = scan[v - 1];
                    scan[v] += prev;
                }
                t_scan[k] = scan;
                i += PLANE_DIM;
            }
        }

        if part_id == cube_count_x - 1 {
            for k in 0..line_spt {
                if i < scan_in.shape(1) {
                    // Manually fuse not_equal and cast
                    let mut scan =
                        Line::cast_from(scan_in[i + scan_offs].not_equal(Line::new(zero)));
                    #[unroll]
                    for v in 1..scan_in.line_size() {
                        let prev = scan[v - 1];
                        scan[v] += prev;
                    }
                    t_scan[k] = scan;
                }
                i += PLANE_DIM;
            }
        }

        let mut prev = zero;
        let plane_mask = PLANE_DIM - 1;
        let circular_shift = (UNIT_POS_PLANE + plane_mask) & plane_mask;
        for k in 0..line_spt {
            let t = plane_broadcast(plane_inclusive_sum(t_scan[k][v_last]), circular_shift);
            t_scan[k] += Line::cast_from(select(UNIT_POS_PLANE != 0, t, zero) + prev);
            prev += plane_broadcast(t, 0);
        }

        if UNIT_POS_PLANE == 0 {
            reduce[plane_id] = prev;
        }
    }
    sync_units();

    //Non-divergent subgroup agnostic inclusive scan across subgroup reductions
    let lane_log = count_trailing_zeros(PLANE_DIM);
    let spine_size = CUBE_DIM >> lane_log;
    {
        let mut offset_0 = 0;
        let mut offset_1 = 0;
        let aligned_size =
            1 << ((count_trailing_zeros(spine_size) + lane_log + 1) / lane_log * lane_log);
        let mut j = PLANE_DIM;
        while j <= aligned_size {
            let i_0 = ((UNIT_POS_X + offset_0) << offset_1) - offset_0;
            let pred_0 = i_0 < spine_size;
            let t_0 = plane_inclusive_sum(select(pred_0, reduce[i_0], zero));
            if pred_0 {
                reduce[i_0] = t_0;
            }
            sync_units();

            if j != PLANE_DIM {
                let rshift = j >> lane_log;
                let i_1 = UNIT_POS_X + rshift;
                if (i_1 & (j - 1)) >= rshift {
                    let pred_1 = i_1 < spine_size;
                    let t_1 = select(pred_1, reduce[((i_1 >> offset_1) << offset_1) - 1], zero);
                    if pred_1 && ((i_1 + 1) & (rshift - 1)) != 0 {
                        reduce[i_1] += t_1;
                    }
                }
            } else {
                offset_0 += 1;
            }
            offset_1 += lane_log;

            j <<= lane_log;
        }
    }
    sync_units();

    //Device broadcast
    if UNIT_POS_X == 0 {
        Atomic::store(
            &reduction[part_id + red_offs],
            (reduce[spine_size - 1] << I::new(2))
                | select(part_id != 0, flag_reduction, flag_inclusive),
        )
    }

    //Lookback, single thread
    if part_id != 0 {
        if UNIT_POS_X == 0 {
            let mut lookback_id = part_id - 1;
            let mut prev_reduction = zero;
            loop {
                let flag_payload = Atomic::load(&reduction[lookback_id + red_offs]);
                if (flag_payload & flag_mask) == flag_inclusive {
                    prev_reduction += flag_payload >> I::new(2);
                    Atomic::store(
                        &reduction[part_id + red_offs],
                        ((prev_reduction + reduce[spine_size - 1]) << I::new(2)) | flag_inclusive,
                    );
                    broadcast[0] = prev_reduction;
                    break;
                }

                if (flag_payload & flag_mask) == flag_reduction {
                    prev_reduction += flag_payload >> I::new(2);
                    lookback_id -= 1;
                }
            }
        }
        sync_units();
    }

    {
        let prev = if plane_id != 0 {
            reduce[plane_id - 1]
        } else {
            zero
        };
        let prev = Line::cast_from(broadcast[0] + prev);
        let s_offset = UNIT_POS_PLANE + plane_id * PLANE_DIM * line_spt;
        let dev_offset = part_id * nums_per_cube;
        let mut i = s_offset + dev_offset;

        if part_id < cube_count_x - 1 {
            for k in 0..line_spt {
                scan_out[i + scan_offs] = t_scan[k] + prev;
                i += PLANE_DIM;
            }
        }

        if part_id == cube_count_x - 1 {
            for k in 0..line_spt {
                if i < scan_out.shape(1) {
                    scan_out[i + scan_offs] = t_scan[k] + prev;
                }
                i += PLANE_DIM;
            }
        }
    }
}

#[cube]
fn count_trailing_zeros(num: u32) -> u32 {
    u32::find_first_set(num) - 1
}

/// Compute the prefix sum of a tensor
pub fn prefix_sum<R: CubeRuntime, I: IntElement>(input: CubeTensor<R>) -> CubeTensor<R> {
    let client = input.client.clone();
    let device = input.device.clone();
    let num_elems = input.shape.num_elements() as u32;
    let numbers = *input.shape.dims.last().unwrap() as u32;
    let batches = num_elems / numbers;

    let input = reshape(input, Shape::new([batches as usize, numbers as usize]));
    let out = empty_device::<R, I>(client.clone(), device.clone(), input.shape.clone());

    let cubes = numbers.div_ceil(PART_SIZE);
    let cube_dim = CubeDim::new_1d(CUBE_SIZE);
    let cube_count = CubeCount::new_3d(cubes, 1, batches);

    let bump = zeros_device::<R, I>(
        client.clone(),
        device.clone(),
        Shape::new([batches as usize]),
    );
    let reduction = zeros_device::<R, I>(
        client.clone(),
        device.clone(),
        Shape::new([batches as usize, cubes as usize]),
    );

    unsafe {
        prefix_sum_kernel::launch_unchecked::<I, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<I>(4),
            out.as_tensor_arg::<I>(4),
            bump.as_tensor_arg::<I>(1),
            reduction.as_tensor_arg::<I>(1),
            ScalarArg::new(cubes),
        )
    };

    out
}
