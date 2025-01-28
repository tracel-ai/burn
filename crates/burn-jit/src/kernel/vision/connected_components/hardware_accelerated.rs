//! Hardware Accelerated 4-connected, adapted from
//! A. Hennequin, L. Lacassagne, L. Cabaret, Q. Meunier,
//! "A new Direct Connected Component Labeling and Analysis Algorithms for GPUs",
//! DASIP, 2018

use crate::{
    kernel::vision::connected_components::stats_from_opts, ops::numeric::zeros_device,
    tensor::JitTensor, BoolElement, FloatElement, IntElement, JitBackend, JitRuntime,
};
use burn_tensor::Shape;
use burn_vision::{ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity};
use cubecl::{prelude::*, Feature};

const BLOCK_H: u32 = 4;

#[cube]
fn merge(labels: &Tensor<Atomic<u32>>, label_1: u32, label_2: u32) {
    let mut label_1 = label_1;
    let mut label_2 = label_2;

    while label_1 != label_2 && (label_1 != Atomic::load(&labels[label_1]) - 1) {
        label_1 = Atomic::load(&labels[label_1]) - 1;
    }
    while label_1 != label_2 && (label_2 != Atomic::load(&labels[label_2]) - 1) {
        label_2 = Atomic::load(&labels[label_2]) - 1;
    }
    while label_1 != label_2 {
        #[allow(clippy::manual_swap)]
        if label_1 < label_2 {
            let tmp = label_1;
            label_1 = label_2;
            label_2 = tmp;
        }
        let label_3 = Atomic::min(&labels[label_1], label_2 + 1) - 1;
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

#[cube(launch)]
fn strip_labeling<BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &Tensor<Atomic<u32>>,
    #[comptime] connectivity: Connectivity,
) {
    let mut shared_pixels = SharedMemory::<u32>::new(BLOCK_H);

    let batch = ABSOLUTE_POS_Z;
    let y = ABSOLUTE_POS_Y;
    let rows = labels.shape(1);
    let cols = labels.shape(2);

    if y >= rows {
        terminate!();
    }

    let img_stride = img.stride(2);
    let labels_stride = labels.stride(1);

    let img_line_base = batch * img.stride(0) + y * img_stride + UNIT_POS_X;
    let labels_line_base = batch * labels.stride(0) + y * labels.stride(1) + UNIT_POS_X;

    let mut distance_y = 0;
    let mut distance_y_1 = 0;

    for i in range_stepped(0, img.shape(3), PLANE_DIM) {
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

            let pixels_y = plane_ballot(p_y)[0] & mask;
            let mut s_dist_y = start_distance(pixels_y, UNIT_POS_X);

            if p_y && s_dist_y == 0 {
                Atomic::store(
                    &labels[labels_index],
                    labels_index - select(UNIT_POS_X == 0, distance_y, 0) + 1,
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

#[cube(launch)]
fn strip_merge<BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &Tensor<Atomic<u32>>,
    #[comptime] connectivity: Connectivity,
) {
    let batch = CUBE_POS_Z;
    let plane_start_x = CUBE_POS_X * (CUBE_DIM_X * CUBE_DIM_Z - PLANE_DIM) + UNIT_POS_Z * PLANE_DIM;
    let y = (CUBE_POS_Y + 1) * BLOCK_H;
    let x = plane_start_x + UNIT_POS_X;

    let img_step = img.stride(2);
    let labels_step = labels.stride(1);
    let cols = img.shape(3);

    if y < labels.shape(1) && x < labels.shape(2) {
        let mut mask = 0xffffffffu32;
        if cols - plane_start_x < 32 {
            mask >>= 32 - (cols - plane_start_x);
        }

        let img_index = batch * img.stride(0) + y * img_step + x;
        let labels_index = batch * labels.stride(0) + y * labels_step + x;

        let img_index_up = img_index - img_step;
        let labels_index_up = labels_index - labels_step;

        let p = bool::cast_from(img[img_index]);
        let p_up = bool::cast_from(img[img_index_up]);

        let pixels = plane_ballot(p)[0] & mask;
        let pixels_up = plane_ballot(p_up)[0] & mask;

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

#[cube(launch)]
fn relabeling<BT: CubePrimitive>(img: &Tensor<BT>, labels: &mut Tensor<u32>) {
    let batch = ABSOLUTE_POS_Z;
    let plane_start_x = CUBE_POS_X * CUBE_DIM_X;
    let y = ABSOLUTE_POS_Y;
    let x = plane_start_x + UNIT_POS_X;

    let cols = labels.shape(2);
    let rows = labels.shape(1);
    let img_step = img.stride(2);
    let labels_step = labels.stride(1);

    if x < cols && y < rows {
        let mut mask = 0xffffffffu32;
        if cols - plane_start_x < 32 {
            mask >>= 32 - (cols - plane_start_x);
        }

        let img_index = batch * img.stride(0) + y * img_step + x;
        let labels_index = batch * labels.stride(0) + y * labels_step + x;

        let p = bool::cast_from(img[img_index]);
        let pixels = plane_ballot(p)[0] & mask;
        let s_dist = start_distance(pixels, UNIT_POS_X);
        let mut label = 0u32;

        if p && s_dist == 0 {
            label = labels[labels_index] - 1;
            while label != labels[label] - 1 {
                label = labels[label] - 1;
            }
        }

        label = plane_broadcast(label, UNIT_POS_X - s_dist);

        if p {
            labels[labels_index] = label + 1;
        }
    }
}

#[cube(launch)]
fn analysis<BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &mut Tensor<u32>,
    area: &mut Tensor<Atomic<u32>>,
    top: &mut Tensor<Atomic<u32>>,
    left: &mut Tensor<Atomic<u32>>,
    right: &mut Tensor<Atomic<u32>>,
    bottom: &mut Tensor<Atomic<u32>>,
    #[comptime] opts: ConnectedStatsOptions,
) {
    let batch = ABSOLUTE_POS_Z;
    let y = ABSOLUTE_POS_Y;
    let x = ABSOLUTE_POS_X;

    let cols = labels.shape(2);
    let rows = labels.shape(1);
    let img_step = img.stride(2);
    let labels_step = labels.stride(1);

    if x < cols && y < rows {
        let mut mask = 0xffffffffu32;
        if cols - CUBE_POS_X * CUBE_DIM_X < 32 {
            mask >>= 32 - (cols - CUBE_POS_X * CUBE_DIM_X);
        }

        let img_index = batch * img.stride(0) + y * img_step + x;
        let labels_index = batch * labels.stride(0) + y * labels_step + x;

        let p = bool::cast_from(img[img_index]);
        let pixels = plane_ballot(p)[0] & mask;
        let s_dist = start_distance(pixels, UNIT_POS_X);
        let count = end_distance(pixels, UNIT_POS_X);
        let max_x = x + count - 1;

        let mut label = 0u32;

        if p && s_dist == 0 {
            label = labels[labels_index] - 1;
            while label != labels[label] - 1 {
                label = labels[label] - 1;
            }

            if opts.area_enabled {
                Atomic::add(&area[label], count);
            }
            if opts.left_enabled {
                Atomic::min(&left[label], x);
            }
            if opts.top_enabled {
                Atomic::min(&top[label], y);
            }
            if opts.right_enabled {
                Atomic::max(&right[label], max_x);
            }
            if opts.bottom_enabled {
                Atomic::max(&bottom[label], y);
            }
        }

        label = plane_broadcast(label, UNIT_POS_X - s_dist);

        if p {
            labels[labels_index] = label + 1;
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn hardware_accelerated<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    img: JitTensor<R>,
    stats_opt: ConnectedStatsOptions,
    connectivity: Connectivity,
) -> Result<
    (
        JitTensor<R>,
        ConnectedStatsPrimitive<JitBackend<R, F, I, BT>>,
    ),
    String,
> {
    let client = img.client.clone();
    let device = img.device.clone();

    if !client.properties().feature_enabled(Feature::Plane) {
        return Err("Requires plane instructions".into());
    }

    let props = client.properties().hardware_properties();

    if props.plane_size_min != 32 || props.plane_size_min != props.plane_size_max {
        return Err(
            "Currently only supports 32 wide planes because it's heavily tied to plane op width"
                .into(),
        );
    }

    let [batches, channels, rows, cols] = img.shape.dims();
    assert_eq!(channels, 1);

    let shape = Shape::new([batches, rows, cols]);
    let labels = zeros_device::<R, u32>(client.clone(), device.clone(), shape);

    let warp_size = 32;
    let cube_dim = CubeDim::new_2d(warp_size, BLOCK_H);
    let cube_count = CubeCount::Static(1, (rows as u32).div_ceil(cube_dim.y), batches as u32);

    strip_labeling::launch::<BT, R>(
        &client,
        cube_count,
        cube_dim,
        img.as_tensor_arg::<u8>(1),
        labels.as_tensor_arg::<u32>(1),
        connectivity,
    );

    let horizontal_warps = Ord::min((cols as u32).div_ceil(warp_size), 32);
    let cube_dim_merge = CubeDim::new_3d(warp_size, 1, horizontal_warps);
    let cube_count = CubeCount::Static(
        Ord::max((cols as u32 + warp_size * 30 - 1) / (warp_size * 31), 1),
        (rows as u32 - 1) / BLOCK_H,
        batches as u32,
    );

    strip_merge::launch::<BT, R>(
        &client,
        cube_count,
        cube_dim_merge,
        img.as_tensor_arg::<u8>(1),
        labels.as_tensor_arg::<u32>(1),
        connectivity,
    );

    let cube_count = CubeCount::Static(
        (cols as u32).div_ceil(cube_dim.x),
        (rows as u32).div_ceil(cube_dim.y),
        batches as u32,
    );

    let stats = stats_from_opts(labels.clone(), stats_opt);

    if stats_opt == ConnectedStatsOptions::none() {
        relabeling::launch::<BT, R>(
            &client,
            cube_count,
            cube_dim,
            img.as_tensor_arg::<u8>(1),
            labels.as_tensor_arg::<u32>(1),
        );
    } else {
        analysis::launch::<BT, R>(
            &client,
            cube_count,
            cube_dim,
            img.as_tensor_arg::<u8>(1),
            labels.as_tensor_arg::<u32>(1),
            stats.area.as_tensor_arg::<u32>(1),
            stats.top.as_tensor_arg::<u32>(1),
            stats.left.as_tensor_arg::<u32>(1),
            stats.right.as_tensor_arg::<u32>(1),
            stats.bottom.as_tensor_arg::<u32>(1),
            stats_opt,
        );
    }

    Ok((labels, stats))
}
