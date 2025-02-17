//! Hardware Accelerated 4-connected, adapted from
//! A. Hennequin, L. Lacassagne, L. Cabaret, Q. Meunier,
//! "A new Direct Connected Component Labeling and Analysis Algorithms for GPUs",
//! DASIP, 2018

use crate::{
    backends::cube::connected_components::stats_from_opts, ConnectedStatsOptions,
    ConnectedStatsPrimitive, Connectivity,
};
use burn_cubecl::{
    kernel,
    ops::{into_data_sync, numeric::zeros_device},
    tensor::CubeTensor,
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
};
use burn_tensor::{cast::ToElement, ops::IntTensorOps, Shape};
use cubecl::{prelude::*, Feature};

use super::prefix_sum::prefix_sum;

const BLOCK_H: u32 = 4;

#[cube]
fn merge<I: Int>(labels: &Tensor<Atomic<I>>, label_1: u32, label_2: u32) {
    let mut label_1 = label_1;
    let mut label_2 = label_2;

    while label_1 != label_2 && (label_1 != u32::cast_from(Atomic::load(&labels[label_1])) - 1) {
        label_1 = u32::cast_from(Atomic::load(&labels[label_1])) - 1;
    }
    while label_1 != label_2 && (label_2 != u32::cast_from(Atomic::load(&labels[label_2])) - 1) {
        label_2 = u32::cast_from(Atomic::load(&labels[label_2])) - 1;
    }
    while label_1 != label_2 {
        #[allow(clippy::manual_swap)]
        if label_1 < label_2 {
            let tmp = label_1;
            label_1 = label_2;
            label_2 = tmp;
        }
        let label_3 = u32::cast_from(Atomic::min(&labels[label_1], I::cast_from(label_2 + 1))) - 1;
        if label_1 == label_3 {
            label_1 = label_2;
        } else {
            label_1 = label_3;
        }
    }
}

#[cube]
fn start_distance(pixels: u32, tx: u32) -> u32 {
    u32::leading_zeros(u32::bitwise_not(pixels << (32 - tx)))
}

#[cube]
fn end_distance(pixels: u32, tx: u32) -> u32 {
    u32::find_first_set(u32::bitwise_not(pixels >> (tx + 1)))
}

#[cube]
#[allow(unconditional_panic, reason = "clippy thinks PLANE_DIM is always 2")]
fn ballot_dyn(y: u32, pred: bool) -> u32 {
    let index = y % (PLANE_DIM / 32);
    plane_ballot(pred)[index]
}

#[cube(launch_unchecked)]
fn strip_labeling<I: Int, BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &Tensor<Atomic<I>>,
    #[comptime] connectivity: Connectivity,
) {
    let mut shared_pixels = SharedMemory::<u32>::new(BLOCK_H);

    let y = ABSOLUTE_POS_Y;
    let rows = labels.shape(0);
    let cols = labels.shape(1);

    if y >= rows {
        terminate!();
    }

    let img_stride = img.stride(0);
    let labels_stride = labels.stride(0);

    let img_line_base = y * img_stride + UNIT_POS_X;
    let labels_line_base = y * labels.stride(0) + UNIT_POS_X;

    let mut distance_y = 0;
    let mut distance_y_1 = 0;

    for i in range_stepped(0, img.shape(1), PLANE_DIM) {
        let x = UNIT_POS_X + i;

        if x < cols {
            let mut mask = 0xffffffffu32;
            let involved_cols = cols - i;
            if involved_cols < 32 {
                mask >>= 32 - involved_cols;
            }

            let img_index = img_line_base + i;
            let labels_index = labels_line_base + i;

            let p_y = bool::cast_from(img[img_index]);

            let pixels_y = ballot_dyn(UNIT_POS_Y, p_y) & mask;
            let mut s_dist_y = start_distance(pixels_y, UNIT_POS_X);

            if p_y && s_dist_y == 0 {
                Atomic::store(
                    &labels[labels_index],
                    I::cast_from(labels_index - select(UNIT_POS_X == 0, distance_y, 0) + 1),
                );
            }

            // Only needed pre-Volta, but we can't check that at present
            sync_units();

            if UNIT_POS_X == 0 {
                shared_pixels[UNIT_POS_Y] = pixels_y;
            }

            sync_units();

            // Requires if and not select, because `select` may execute the then branch even if the
            // condition is false (on non-CUDA backends), which can lead to OOB reads.
            let pixels_y_1 = if UNIT_POS_Y > 0 {
                shared_pixels[UNIT_POS_Y - 1]
            } else {
                0u32
            };

            let p_y_1 = (pixels_y_1 >> UNIT_POS_X) & 1 != 0;
            let mut s_dist_y_1 = start_distance(pixels_y_1, UNIT_POS_X);

            if UNIT_POS_X == 0 {
                s_dist_y = distance_y;
                s_dist_y_1 = distance_y_1;
            }

            match connectivity {
                Connectivity::Four => {
                    if p_y && p_y_1 && (s_dist_y == 0 || s_dist_y_1 == 0) {
                        let label_1 = labels_index - s_dist_y;
                        let label_2 = labels_index - s_dist_y_1 - labels_stride;
                        merge(labels, label_1, label_2);
                    }
                }
                Connectivity::Eight => {
                    let pixels_y_shifted = (pixels_y << 1) | (distance_y > 0) as u32;
                    let pixels_y_1_shifted = (pixels_y_1 << 1) | (distance_y_1 > 0) as u32;

                    if p_y && p_y_1 && (s_dist_y == 0 || s_dist_y_1 == 0) {
                        let label_1 = labels_index - s_dist_y;
                        let label_2 = labels_index - s_dist_y_1 - labels_stride;
                        merge(labels, label_1, label_2);
                    } else if p_y && s_dist_y == 0 && (pixels_y_1_shifted >> UNIT_POS_X) & 1 != 0 {
                        let s_dist_y_1_prev = select(
                            UNIT_POS_X == 0,
                            distance_y_1 - 1,
                            start_distance(pixels_y_1, UNIT_POS_X - 1),
                        );
                        let label_1 = labels_index;
                        let label_2 = labels_index - labels_stride - 1 - s_dist_y_1_prev;
                        merge(labels, label_1, label_2);
                    } else if p_y_1 && s_dist_y_1 == 0 && (pixels_y_shifted >> UNIT_POS_X) & 1 != 0
                    {
                        let s_dist_y_prev = select(
                            UNIT_POS_X == 0,
                            distance_y - 1,
                            start_distance(pixels_y, UNIT_POS_X - 1),
                        );
                        let label_1 = labels_index - 1 - s_dist_y_prev;
                        let label_2 = labels_index - labels_stride;
                        merge(labels, label_1, label_2);
                    }
                }
            }

            if p_y && p_y_1 && (s_dist_y == 0 || s_dist_y_1 == 0) {
                let label_1 = labels_index - s_dist_y;
                let label_2 = labels_index - s_dist_y_1 - labels_stride;
                merge(labels, label_1, label_2);
            }

            let mut d = start_distance(pixels_y_1, 32);
            distance_y_1 = d + select(d == 32, distance_y_1, 0);
            d = start_distance(pixels_y, 32);
            distance_y = d + select(d == 32, distance_y, 0);
        }
    }
}

#[cube(launch_unchecked)]
fn strip_merge<I: Int, BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &Tensor<Atomic<I>>,
    #[comptime] connectivity: Connectivity,
) {
    let plane_start_x = CUBE_POS_X * (CUBE_DIM_X * CUBE_DIM_Z - PLANE_DIM) + UNIT_POS_Z * PLANE_DIM;
    let y = (CUBE_POS_Y + 1) * BLOCK_H;
    let x = plane_start_x + UNIT_POS_X;

    let img_step = img.stride(0);
    let labels_step = labels.stride(0);
    let cols = img.shape(1);

    if y < labels.shape(0) && x < labels.shape(1) {
        let mut mask = 0xffffffffu32;
        if cols - plane_start_x < 32 {
            mask >>= 32 - (cols - plane_start_x);
        }

        let img_index = y * img_step + x;
        let labels_index = y * labels_step + x;

        let img_index_up = img_index - img_step;
        let labels_index_up = labels_index - labels_step;

        let p = bool::cast_from(img[img_index]);
        let p_up = bool::cast_from(img[img_index_up]);

        let pixels = ballot_dyn(UNIT_POS_Z, p) & mask;
        let pixels_up = ballot_dyn(UNIT_POS_Z, p_up) & mask;

        match connectivity {
            Connectivity::Four => {
                if p && p_up {
                    let s_dist = start_distance(pixels, UNIT_POS_X);
                    let s_dist_up = start_distance(pixels_up, UNIT_POS_X);
                    if s_dist == 0 || s_dist_up == 0 {
                        merge(labels, labels_index - s_dist, labels_index_up - s_dist_up);
                    }
                }
            }
            Connectivity::Eight => {
                let mut last_dist_vec = SharedMemory::<u32>::new(32);
                let mut last_dist_up_vec = SharedMemory::<u32>::new(32);

                let s_dist = start_distance(pixels, UNIT_POS_X);
                let s_dist_up = start_distance(pixels_up, UNIT_POS_X);

                if UNIT_POS_PLANE == PLANE_DIM - 1 {
                    last_dist_vec[UNIT_POS_Z] = start_distance(pixels, 32);
                    last_dist_up_vec[UNIT_POS_Z] = start_distance(pixels_up, 32);
                }

                sync_units();

                if CUBE_POS_X == 0 || UNIT_POS_Z > 0 {
                    let last_dist = if UNIT_POS_Z > 0 {
                        last_dist_vec[UNIT_POS_Z - 1]
                    } else {
                        0u32
                    };
                    let last_dist_up = if UNIT_POS_Z > 0 {
                        last_dist_up_vec[UNIT_POS_Z - 1]
                    } else {
                        0u32
                    };

                    let p_prev =
                        select(UNIT_POS_X > 0, (pixels >> (UNIT_POS_X - 1)) & 1, last_dist) != 0;
                    let p_up_prev = select(
                        UNIT_POS_X > 0,
                        (pixels_up >> (UNIT_POS_X - 1)) & 1,
                        last_dist_up,
                    ) != 0;

                    if p && p_up {
                        let s_dist = start_distance(pixels, UNIT_POS_X);
                        let s_dist_up = start_distance(pixels_up, UNIT_POS_X);
                        if s_dist == 0 || s_dist_up == 0 {
                            merge(labels, labels_index - s_dist, labels_index_up - s_dist_up);
                        }
                    } else if p && p_up_prev && s_dist == 0 {
                        let s_dist_up_prev = select(
                            UNIT_POS_X == 0,
                            last_dist_up - 1,
                            start_distance(pixels_up, UNIT_POS_X - 1),
                        );
                        merge(labels, labels_index, labels_index_up - 1 - s_dist_up_prev);
                    } else if p_prev && p_up && s_dist_up == 0 {
                        let s_dist_prev = select(
                            UNIT_POS_X == 0,
                            last_dist - 1,
                            start_distance(pixels, UNIT_POS_X - 1),
                        );
                        merge(labels, labels_index - 1 - s_dist_prev, labels_index_up);
                    }
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
fn relabeling<I: Int, BT: CubePrimitive>(img: &Tensor<BT>, labels: &mut Tensor<I>) {
    let plane_start_x = CUBE_POS_X * CUBE_DIM_X;
    let y = ABSOLUTE_POS_Y;
    let x = plane_start_x + UNIT_POS_X;

    let cols = labels.shape(1);
    let rows = labels.shape(0);
    let img_step = img.stride(0);
    let labels_step = labels.stride(0);

    if x < cols && y < rows {
        let mut mask = 0xffffffffu32;
        if cols - plane_start_x < 32 {
            mask >>= 32 - (cols - plane_start_x);
        }

        let img_index = y * img_step + x;
        let labels_index = y * labels_step + x;

        let p = bool::cast_from(img[img_index]);
        let pixels = ballot_dyn(UNIT_POS_Y, p) & mask;
        let s_dist = start_distance(pixels, UNIT_POS_X);
        let mut label = 0u32;

        if p && s_dist == 0 {
            label = u32::cast_from(labels[labels_index]) - 1;
            while label != u32::cast_from(labels[label]) - 1 {
                label = u32::cast_from(labels[label]) - 1;
            }
        }

        label = plane_broadcast(label, UNIT_POS_X - s_dist);

        if p {
            labels[labels_index] = I::cast_from(label + 1);
        }
    }
}

#[cube(launch_unchecked)]
fn analysis<I: Int, BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &mut Tensor<I>,
    area: &mut Tensor<Atomic<I>>,
    top: &mut Tensor<Atomic<I>>,
    left: &mut Tensor<Atomic<I>>,
    right: &mut Tensor<Atomic<I>>,
    bottom: &mut Tensor<Atomic<I>>,
    max_label: &mut Tensor<Atomic<I>>,
    #[comptime] opts: ConnectedStatsOptions,
) {
    let y = ABSOLUTE_POS_Y;
    let x = ABSOLUTE_POS_X;

    let cols = labels.shape(1);
    let rows = labels.shape(0);
    let img_step = img.stride(0);
    let labels_step = labels.stride(0);

    if x < cols && y < rows {
        let mut mask = 0xffffffffu32;
        if cols - CUBE_POS_X * CUBE_DIM_X < 32 {
            mask >>= 32 - (cols - CUBE_POS_X * CUBE_DIM_X);
        }

        let img_index = y * img_step + x;
        let labels_index = y * labels_step + x;

        let p = bool::cast_from(img[img_index]);
        let pixels = ballot_dyn(UNIT_POS_Y, p) & mask;
        let s_dist = start_distance(pixels, UNIT_POS_X);
        let count = end_distance(pixels, UNIT_POS_X);
        let max_x = x + count - 1;

        let mut label = 0u32;

        if p && s_dist == 0 {
            label = u32::cast_from(labels[labels_index]) - 1;
            while label != u32::cast_from(labels[label]) - 1 {
                label = u32::cast_from(labels[label]) - 1;
            }
            label += 1;

            Atomic::add(&area[label], I::cast_from(count));

            if opts.bounds_enabled {
                Atomic::min(&left[label], I::cast_from(x));
                Atomic::min(&top[label], I::cast_from(y));
                Atomic::max(&right[label], I::cast_from(max_x));
                Atomic::max(&bottom[label], I::cast_from(y));
            }
            if comptime!(opts.max_label_enabled || opts.compact_labels) {
                Atomic::max(&max_label[0], I::cast_from(label));
            }
        }

        label = plane_broadcast(label, UNIT_POS_X - s_dist);

        if p {
            labels[labels_index] = I::cast_from(label);
        }
    }
}

#[cube(launch_unchecked)]
fn compact_labels<I: Int>(
    labels: &mut Tensor<I>,
    remap: &Tensor<I>,
    max_label: &Tensor<Atomic<I>>,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let labels_pos = y * labels.stride(0) + x;

    if labels_pos >= labels.len() {
        terminate!();
    }

    let label = u32::cast_from(labels[labels_pos]);
    if label != 0 {
        let new_label = remap[label];
        labels[labels_pos] = new_label;
        Atomic::max(&max_label[0], new_label);
    }
}

#[cube(launch_unchecked)]
fn compact_stats<I: Int>(
    area: &Tensor<I>,
    area_new: &mut Tensor<I>,
    top: &Tensor<I>,
    top_new: &mut Tensor<I>,
    left: &Tensor<I>,
    left_new: &mut Tensor<I>,
    right: &Tensor<I>,
    right_new: &mut Tensor<I>,
    bottom: &Tensor<I>,
    bottom_new: &mut Tensor<I>,
    remap: &Tensor<I>,
) {
    let label = ABSOLUTE_POS_X;
    if label >= remap.len() {
        terminate!();
    }

    let area = area[label];
    if area == I::new(0) {
        terminate!();
    }
    let new_label = u32::cast_from(remap[label]);

    area_new[new_label] = area;
    // This should be gated but there's a problem with the Eq bound only being implemented for tuples
    // up to 12 elems, so I can't pass the opts. It's not unsafe, but potentially unnecessary work.
    top_new[new_label] = top[label];
    left_new[new_label] = left[label];
    right_new[new_label] = right[label];
    bottom_new[new_label] = bottom[label];
}

#[allow(clippy::type_complexity)]
pub fn hardware_accelerated<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    img: CubeTensor<R>,
    stats_opt: ConnectedStatsOptions,
    connectivity: Connectivity,
) -> Result<
    (
        CubeTensor<R>,
        ConnectedStatsPrimitive<CubeBackend<R, F, I, BT>>,
    ),
    String,
> {
    let client = img.client.clone();
    let device = img.device.clone();

    if !client.properties().feature_enabled(Feature::Plane) {
        return Err("Requires plane instructions".into());
    }

    let props = client.properties().hardware_properties();

    if props.plane_size_min < 32 {
        return Err("Requires plane size of at least 32".into());
    }

    let [rows, cols] = img.shape.dims();

    let labels = zeros_device::<R, I>(client.clone(), device.clone(), img.shape.clone());

    // Assume 32 wide warp. Currently, larger warps are handled by just exiting everything past 32.
    // This isn't ideal but we require CUBE_DIM_X == warp_size, and we can't query the actual warp
    // size at compile time. `REQUIRE_FULL_SUBGROUPS` or subgroup size controls are not supported
    // in wgpu.
    let warp_size = 32;
    let cube_dim = CubeDim::new_2d(warp_size, BLOCK_H);
    let cube_count = CubeCount::new_2d(1, (rows as u32).div_ceil(cube_dim.y));

    unsafe {
        strip_labeling::launch_unchecked::<I, BT, R>(
            &client,
            cube_count,
            cube_dim,
            img.as_tensor_arg::<BT>(1),
            labels.as_tensor_arg::<I>(1),
            connectivity,
        )
    };

    let horizontal_warps = Ord::min((cols as u32).div_ceil(warp_size), 32);
    let cube_dim_merge = CubeDim::new_3d(warp_size, 1, horizontal_warps);
    let cube_count = CubeCount::new_2d(
        Ord::max((cols as u32 + warp_size * 30 - 1) / (warp_size * 31), 1),
        (rows as u32 - 1) / BLOCK_H,
    );

    unsafe {
        strip_merge::launch_unchecked::<I, BT, R>(
            &client,
            cube_count,
            cube_dim_merge,
            img.as_tensor_arg::<BT>(1),
            labels.as_tensor_arg::<I>(1),
            connectivity,
        )
    };

    let cube_count = CubeCount::new_2d(
        (cols as u32).div_ceil(cube_dim.x),
        (rows as u32).div_ceil(cube_dim.y),
    );

    let mut stats = stats_from_opts(labels.clone(), stats_opt);

    if stats_opt == ConnectedStatsOptions::none() {
        unsafe {
            relabeling::launch_unchecked::<I, BT, R>(
                &client,
                cube_count,
                cube_dim,
                img.as_tensor_arg::<BT>(1),
                labels.as_tensor_arg::<I>(1),
            )
        };
    } else {
        unsafe {
            analysis::launch_unchecked::<I, BT, R>(
                &client,
                cube_count,
                cube_dim,
                img.as_tensor_arg::<BT>(1),
                labels.as_tensor_arg::<I>(1),
                stats.area.as_tensor_arg::<I>(1),
                stats.top.as_tensor_arg::<I>(1),
                stats.left.as_tensor_arg::<I>(1),
                stats.right.as_tensor_arg::<I>(1),
                stats.bottom.as_tensor_arg::<I>(1),
                stats.max_label.as_tensor_arg::<I>(1),
                stats_opt,
            )
        };
        if stats_opt.compact_labels {
            let max_label = CubeBackend::<R, F, I, BT>::int_max(stats.max_label);
            let max_label = into_data_sync::<R, I>(max_label);
            let max_label = ToElement::to_usize(&max_label.as_slice::<I>().unwrap()[0]);
            let sliced = kernel::slice::<R, I>(
                stats.area.clone(),
                #[allow(clippy::single_range_in_vec_init)]
                &[0..(max_label + 1).next_multiple_of(4)],
            );
            let relabel = prefix_sum::<R, I>(sliced);

            let cube_dim = CubeDim::default();
            let cube_count = CubeCount::new_2d(
                (cols as u32).div_ceil(cube_dim.x),
                (rows as u32).div_ceil(cube_dim.y),
            );
            stats.max_label = zeros_device::<R, I>(client.clone(), device.clone(), Shape::new([1]));
            unsafe {
                compact_labels::launch_unchecked::<I, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    labels.as_tensor_arg::<I>(1),
                    relabel.as_tensor_arg::<I>(1),
                    stats.max_label.as_tensor_arg::<I>(1),
                )
            };

            let cube_dim = CubeDim::new_1d(256);
            let cube_count = CubeCount::new_1d((rows * cols).div_ceil(256) as u32);
            unsafe {
                compact_stats::launch_unchecked::<I, R>(
                    &client,
                    cube_count,
                    cube_dim,
                    stats.area.copy().as_tensor_arg::<I>(1),
                    stats.area.as_tensor_arg::<I>(1),
                    stats.top.copy().as_tensor_arg::<I>(1),
                    stats.top.as_tensor_arg::<I>(1),
                    stats.left.copy().as_tensor_arg::<I>(1),
                    stats.left.as_tensor_arg::<I>(1),
                    stats.right.copy().as_tensor_arg::<I>(1),
                    stats.right.as_tensor_arg::<I>(1),
                    stats.bottom.copy().as_tensor_arg::<I>(1),
                    stats.bottom.as_tensor_arg::<I>(1),
                    relabel.as_tensor_arg::<I>(1),
                )
            };
        }
    }

    Ok((labels, stats))
}
