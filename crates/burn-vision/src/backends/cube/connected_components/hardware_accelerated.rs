//! Hardware Accelerated 4-connected, adapted from
//! A. Hennequin, L. Lacassagne, L. Cabaret, Q. Meunier,
//! "A new Direct Connected Component Labeling and Analysis Algorithms for GPUs",
//! DASIP, 2018

use crate::ConnectedStatsPrimitive;
use crate::{
    ConnectedStatsOptions, Connectivity, backends::cube::connected_components::stats_from_opts,
};
use burn_core::backend::cubecl::dtype_to_storage_type;
use burn_core::backend::{TensorMetadata, ops::IntTensorOps};
use burn_core::tensor::DType;
use burn_core::tensor::{Shape, cast::ToElement};
use burn_cubecl::{
    CubeBackend, kernel,
    ops::{into_data_sync, numeric::zeros_client},
    tensor::CubeTensor,
};
use cubecl::{
    features::{AtomicUsage, Plane},
    ir::Type,
    prelude::*,
};

use super::prefix_sum::prefix_sum;

const BLOCK_H: usize = 4;

#[cube]
fn merge<I: Int>(labels: &Tensor<Atomic<I>>, label_1: u32, label_2: u32) {
    let mut label_1 = label_1 as usize;
    let mut label_2 = label_2 as usize;

    // Keep atomic loads out of short-circuit loop conditions: the current CubeCL lowering
    // carries the condition through a loop phi before initializing it.
    while label_1 != label_2 {
        let parent = usize::cast_from(labels[label_1].load()) - 1;
        if parent == label_1 {
            break;
        }
        label_1 = parent;
    }
    while label_1 != label_2 {
        let parent = usize::cast_from(labels[label_2].load()) - 1;
        if parent == label_2 {
            break;
        }
        label_2 = parent;
    }
    while label_1 != label_2 {
        #[allow(clippy::manual_swap)]
        if label_1 < label_2 {
            let tmp = label_1;
            label_1 = label_2;
            label_2 = tmp;
        }
        let label_3 = usize::cast_from(labels[label_1].fetch_min(I::cast_from(label_2 + 1))) - 1;
        if label_1 == label_3 {
            label_1 = label_2;
        } else {
            label_1 = label_3;
        }
    }
}

#[cube]
fn start_distance(pixels: u32, tx: u32) -> u32 {
    // tx ranges from 0 through 32. Mask the shift to avoid shifting by 32;
    // clamping to tx gives zero at the first lane and preserves the tx == 32 carry.
    (!(pixels << ((32 - tx) & 31))).leading_zeros().min(tx)
}

#[cube]
fn end_distance(pixels: u32, tx: u32) -> u32 {
    // Separate shifts keep both counts below 32, including at the last lane.
    (!((pixels >> tx) >> 1)).find_first_set()
}

#[cube]
fn ballot(pred: bool) -> u32 {
    plane_ballot(pred).extract(0usize)
}

#[cube(launch_unchecked)]
fn strip_labeling<I: Int, BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &Tensor<Atomic<I>>,
    #[comptime] connectivity: Connectivity,
    #[define(I, BT)] _dtypes: [ElemType; 2],
) {
    let mut shared_pixels = Shared::new_slice(BLOCK_H);

    let y = ABSOLUTE_POS_Y;
    let rows = labels.shape(0) as u32;
    let cols = labels.shape(1) as u32;

    let img_stride = img.stride(0) as u32;
    let labels_stride = labels.stride(0) as u32;

    let img_line_base = y * img_stride + UNIT_POS_X;
    let labels_line_base = y * labels_stride + UNIT_POS_X;

    let mut distance_y = 0u32;
    let mut distance_y_1 = 0;

    for i in range_stepped(0, img.shape(1) as u32, PLANE_DIM) {
        let x = UNIT_POS_X + i;

        let img_index = img_line_base + i;
        let labels_index = labels_line_base + i;

        let p_y = if x < cols && y < rows {
            bool::cast_from(img[img_index as usize])
        } else {
            false
        };

        let pixels_y = ballot(p_y);
        let mut s_dist_y = start_distance(pixels_y, UNIT_POS_X);

        if p_y && s_dist_y == 0 {
            labels[labels_index as usize].store(I::cast_from(
                labels_index - select(UNIT_POS_X == 0, distance_y, 0) + 1,
            ));
        }

        // Initialize every run before another row follows its parent pointer.
        sync_cube();

        if UNIT_POS_X == 0 {
            shared_pixels[UNIT_POS_Y as usize] = pixels_y;
        }

        sync_cube();

        // Requires if and not select, because `select` may execute the then branch even if the
        // condition is false (on non-CUDA backends), which can lead to OOB reads.
        let pixels_y_1 = if UNIT_POS_Y > 0 {
            shared_pixels[(UNIT_POS_Y - 1) as usize]
        } else {
            0u32.runtime()
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
                    let s_dist_y_1_prev = if UNIT_POS_X == 0 {
                        distance_y_1 - 1
                    } else {
                        start_distance(pixels_y_1, UNIT_POS_X - 1)
                    };
                    let label_1 = labels_index;
                    let label_2 = labels_index - labels_stride - 1 - s_dist_y_1_prev;
                    merge(labels, label_1, label_2);
                } else if p_y_1 && s_dist_y_1 == 0 && (pixels_y_shifted >> UNIT_POS_X) & 1 != 0 {
                    let s_dist_y_prev = if UNIT_POS_X == 0 {
                        distance_y - 1
                    } else {
                        start_distance(pixels_y, UNIT_POS_X - 1)
                    };
                    let label_1 = labels_index - 1 - s_dist_y_prev;
                    let label_2 = labels_index - labels_stride;
                    merge(labels, label_1, label_2);
                }
            }
        }

        let mut d = start_distance(pixels_y_1, 32);
        distance_y_1 = d + select(d == 32, distance_y_1, 0);
        d = start_distance(pixels_y, 32);
        distance_y = d + select(d == 32, distance_y, 0);
        sync_cube();
    }
}

#[cube(launch_unchecked)]
fn strip_merge<I: Int, BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &Tensor<Atomic<I>>,
    #[comptime] connectivity: Connectivity,
    #[define(I, BT)] _dtypes: [ElemType; 2],
) {
    let plane_start_x = CUBE_POS_X * (CUBE_DIM_X * CUBE_DIM_Z - PLANE_DIM) + UNIT_POS_Z * PLANE_DIM;
    let y = (CUBE_POS_Y + 1) * BLOCK_H as u32;
    let x = plane_start_x + UNIT_POS_X;

    let img_step = img.stride(0) as u32;
    let labels_step = labels.stride(0) as u32;
    let cols = img.shape(1) as u32;

    let img_index = y * img_step + x;
    let labels_index = y * labels_step + x;

    let img_index_up = img_index - img_step;
    let labels_index_up = labels_index - labels_step;

    let p = if x < cols {
        bool::cast_from(img[img_index as usize])
    } else {
        false
    };
    let p_up = if x < cols {
        bool::cast_from(img[img_index_up as usize])
    } else {
        false
    };

    let pixels = ballot(p);
    let pixels_up = ballot(p_up);

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
            let mut last_dist_vec = Shared::new_slice(32usize);
            let mut last_dist_up_vec = Shared::new_slice(32usize);

            let s_dist = start_distance(pixels, UNIT_POS_X);
            let s_dist_up = start_distance(pixels_up, UNIT_POS_X);

            if UNIT_POS_PLANE == PLANE_DIM - 1 {
                last_dist_vec[UNIT_POS_Z as usize] = start_distance(pixels, 32);
                last_dist_up_vec[UNIT_POS_Z as usize] = start_distance(pixels_up, 32);
            }

            sync_cube();

            if CUBE_POS_X == 0 || UNIT_POS_Z > 0 {
                let last_dist = if UNIT_POS_Z > 0 {
                    last_dist_vec[(UNIT_POS_Z - 1) as usize]
                } else {
                    0u32.runtime()
                };
                let last_dist_up = if UNIT_POS_Z > 0 {
                    last_dist_up_vec[(UNIT_POS_Z - 1) as usize]
                } else {
                    0u32.runtime()
                };

                let p_prev = if UNIT_POS_X > 0 {
                    (pixels >> (UNIT_POS_X - 1)) & 1
                } else {
                    last_dist
                } != 0;
                let p_up_prev = if UNIT_POS_X > 0 {
                    (pixels_up >> (UNIT_POS_X - 1)) & 1
                } else {
                    last_dist_up
                } != 0;

                if p && p_up {
                    let s_dist = start_distance(pixels, UNIT_POS_X);
                    let s_dist_up = start_distance(pixels_up, UNIT_POS_X);
                    if s_dist == 0 || s_dist_up == 0 {
                        merge(labels, labels_index - s_dist, labels_index_up - s_dist_up);
                    }
                } else if p && p_up_prev && s_dist == 0 {
                    let s_dist_up_prev = if UNIT_POS_X == 0 {
                        last_dist_up - 1
                    } else {
                        start_distance(pixels_up, UNIT_POS_X - 1)
                    };
                    merge(labels, labels_index, labels_index_up - 1 - s_dist_up_prev);
                } else if p_prev && p_up && s_dist_up == 0 {
                    let s_dist_prev = if UNIT_POS_X == 0 {
                        last_dist - 1
                    } else {
                        start_distance(pixels, UNIT_POS_X - 1)
                    };
                    merge(labels, labels_index - 1 - s_dist_prev, labels_index_up);
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
fn relabeling<I: Int, BT: CubePrimitive>(
    img: &Tensor<BT>,
    labels: &mut Tensor<I>,
    #[define(I, BT)] _dtypes: [ElemType; 2],
) {
    let plane_start_x = CUBE_POS_X * CUBE_DIM_X;
    let y = ABSOLUTE_POS_Y;
    let x = plane_start_x + UNIT_POS_X;

    let cols = labels.shape(1) as u32;
    let rows = labels.shape(0) as u32;
    let img_step = img.stride(0) as u32;
    let labels_step = labels.stride(0) as u32;

    let img_index = y * img_step + x;
    let labels_index = y * labels_step + x;

    let p = if x < cols && y < rows {
        bool::cast_from(img[img_index as usize])
    } else {
        false
    };
    let pixels = ballot(p);
    let s_dist = start_distance(pixels, UNIT_POS_X);
    let mut label = 0u32;

    if p && s_dist == 0 {
        label = u32::cast_from(labels[labels_index as usize]) - 1;
        while label != u32::cast_from(labels[label as usize]) - 1 {
            label = u32::cast_from(labels[label as usize]) - 1;
        }
    }

    label = plane_shuffle(label, UNIT_POS_X - s_dist);

    if p {
        labels[labels_index as usize] = I::cast_from(label + 1);
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
    #[define(I, BT)] _dtypes: [ElemType; 2],
) {
    let y = ABSOLUTE_POS_Y;
    let x = ABSOLUTE_POS_X;

    // Background has no bounds; foreground entries retain MAX for fetch_min.
    if opts.bounds_enabled && x == 0 && y == 0 {
        left[0].store(I::new(0));
        top[0].store(I::new(0));
    }

    let cols = labels.shape(1) as u32;
    let rows = labels.shape(0) as u32;
    let img_step = img.stride(0) as u32;
    let labels_step = labels.stride(0) as u32;

    let img_index = y * img_step + x;
    let labels_index = y * labels_step + x;

    let p = if x < cols && y < rows {
        bool::cast_from(img[img_index as usize])
    } else {
        false
    };
    let pixels = ballot(p);
    let s_dist = start_distance(pixels, UNIT_POS_X);
    let count = end_distance(pixels, UNIT_POS_X);
    let max_x = x + count - 1;

    let mut label = 0u32;

    if p && s_dist == 0 {
        label = u32::cast_from(labels[labels_index as usize]) - 1;
        while label != u32::cast_from(labels[label as usize]) - 1 {
            label = u32::cast_from(labels[label as usize]) - 1;
        }
        label += 1;

        area[label as usize].fetch_add(I::cast_from(count));

        if opts.bounds_enabled {
            left[label as usize].fetch_min(I::cast_from(x));
            top[label as usize].fetch_min(I::cast_from(y));
            right[label as usize].fetch_max(I::cast_from(max_x));
            bottom[label as usize].fetch_max(I::cast_from(y));
        }
        if comptime!(opts.max_label_enabled || opts.compact_labels) {
            max_label[0].fetch_max(I::cast_from(label));
        }
    }

    label = plane_shuffle(label, UNIT_POS_X - s_dist);

    if p {
        labels[labels_index as usize] = I::cast_from(label);
    }
}

#[cube(launch_unchecked)]
fn compact_labels<I: Int>(
    labels: &mut Tensor<I>,
    remap: &Tensor<I>,
    max_label: &Tensor<Atomic<I>>,
    #[define(I)] _dtype: ElemType,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let labels_pos = y * labels.stride(0) as u32 + x;

    if x >= labels.shape(1) as u32 || y >= labels.shape(0) as u32 {
        terminate!();
    }

    let label = u32::cast_from(labels[labels_pos as usize]);
    if label != 0 {
        let new_label = remap[label as usize];
        labels[labels_pos as usize] = new_label;
        max_label[0].fetch_max(new_label);
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
    #[comptime] bounds_enabled: bool,
    #[define(I)] _dtype: ElemType,
) {
    let label = ABSOLUTE_POS_X;
    if label as usize >= remap.len() {
        terminate!();
    }

    let area = area[label as usize];
    // Preserve background statistics; only unused foreground labels are skipped.
    if label != 0 && area == I::new(0) {
        terminate!();
    }
    let new_label = u32::cast_from(remap[label as usize]);

    area_new[new_label as usize] = area;
    if bounds_enabled {
        top_new[new_label as usize] = top[label as usize];
        left_new[new_label as usize] = left[label as usize];
        right_new[new_label as usize] = right[label as usize];
        bottom_new[new_label as usize] = bottom[label as usize];
    }
}

pub fn hardware_accelerated(
    img: CubeTensor,
    stats_opt: ConnectedStatsOptions,
    connectivity: Connectivity,
    int_dtype: DType,
) -> Result<(CubeTensor, ConnectedStatsPrimitive<CubeBackend>), String> {
    if img.meta.shape().num_elements() <= 1 {
        return Err("Small images use the CPU fallback".into());
    }
    let client = img.client.clone();
    let device = img.device.clone();
    if int_dtype != DType::I32 {
        return Err("Requires i32 output labels".into());
    }
    let dtypes = [
        dtype_to_storage_type(int_dtype),
        dtype_to_storage_type(img.dtype),
    ];
    let int_storage = dtype_to_storage_type(int_dtype);

    if !client.properties().features.plane.contains(Plane::Ops) {
        return Err("Requires plane instructions".into());
    }

    let props = &client.properties().hardware;

    if props.plane_size_min != 32 || props.plane_size_max != 32 {
        return Err("Requires a fixed plane size of 32".into());
    }

    let mut atomics = AtomicUsage::LoadStore | AtomicUsage::MinMax;
    if stats_opt != ConnectedStatsOptions::none() {
        atomics |= AtomicUsage::Add;
    }
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(int_storage))
        .is_superset(atomics)
    {
        return Err("Requires i32 load/store, min/max and add atomics".into());
    }
    let [rows, cols] = img.meta.shape().dims();
    let compact = stats_opt.compact_labels;
    let shared_bytes = if compact {
        // The scan uses 64 i32 subgroup totals and one shared broadcast value.
        260
    } else if connectivity == Connectivity::Eight && rows > BLOCK_H {
        256
    } else {
        16
    };
    if props.max_units_per_cube < if compact { 256 } else { 128 }
        || props.max_cube_dim.0 < if compact { 256 } else { 32 }
        || props.max_cube_dim.1 < if compact { 8 } else { 4 }
        || props.max_cube_dim.2 == 0
        || props.max_shared_memory_size < shared_bytes
    {
        return Err("Insufficient workgroup dimensions or shared memory".into());
    }
    // The lookback scan reserves two bits in its i32 payload for partition status.
    let max_elements = if compact {
        (i32::MAX >> 2) as usize
    } else {
        i32::MAX as usize
    };
    if img.meta.num_elements() > max_elements {
        return Err("Image exceeds the label or scan-payload index limits".into());
    }

    let warp_size = 32;
    let max_warps = (props.max_units_per_cube / warp_size)
        .min(props.max_cube_dim.2)
        .min(32);
    if rows > BLOCK_H && cols > warp_size as usize && max_warps < 2 {
        return Err("Wide images require at least two planes per strip-merge workgroup".into());
    }
    if (cols as u32).div_ceil(32) > props.max_cube_count.0
        || (rows as u32).div_ceil(4) > props.max_cube_count.1
        || (compact
            && (img.meta.num_elements() + 1).div_ceil(256) > props.max_cube_count.0 as usize)
    {
        return Err("Image exceeds the device's workgroup count limits".into());
    }
    // The kernels honor the row stride, but require adjacent columns.
    let img = if img.meta.strides()[1] != 1 {
        kernel::into_contiguous(img)
    } else {
        img
    };

    // Parent indices address the labels buffer directly, so its rows must not be padded.
    // Reserve one extra element so disabled N + 1 statistics can safely alias this buffer.
    let labels = zeros_client(
        client.clone(),
        device.clone(),
        Shape::new([rows * cols + 1]),
        int_dtype,
    );
    let labels = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        img.shape(),
        labels.handle,
        int_dtype,
    );

    // Each 32-wide plane owns one row, as guaranteed by the fixed-plane gate above.
    let cube_dim = CubeDim::new_2d(warp_size, BLOCK_H as u32);
    let cube_count = CubeCount::new_2d(1, (rows as u32).div_ceil(cube_dim.y));

    unsafe {
        strip_labeling::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            img.clone().into_tensor_arg(),
            labels.clone().into_tensor_arg(),
            connectivity,
            dtypes,
        )
    };

    let horizontal_warps = (cols as u32).div_ceil(warp_size).min(max_warps);
    let cube_dim_merge = CubeDim::new_3d(warp_size, 1, horizontal_warps);
    let merge_blocks = if horizontal_warps == 1 {
        1
    } else {
        (cols as u32)
            .saturating_sub(warp_size)
            .div_ceil(warp_size * (horizontal_warps - 1))
    };
    let cube_count = CubeCount::new_2d(merge_blocks, (rows as u32 - 1) / BLOCK_H as u32);

    if rows > BLOCK_H {
        unsafe {
            strip_merge::launch_unchecked(
                &client,
                cube_count,
                cube_dim_merge,
                img.clone().into_tensor_arg(),
                labels.clone().into_tensor_arg(),
                connectivity,
                dtypes,
            )
        };
    }

    let cube_count = CubeCount::new_2d(
        (cols as u32).div_ceil(cube_dim.x),
        (rows as u32).div_ceil(cube_dim.y),
    );

    let mut stats = stats_from_opts(labels.clone(), stats_opt, int_dtype);

    if stats_opt == ConnectedStatsOptions::none() {
        unsafe {
            relabeling::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                img.into_tensor_arg(),
                labels.clone().into_tensor_arg(),
                dtypes,
            )
        };
    } else {
        unsafe {
            analysis::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                img.clone().into_tensor_arg(),
                labels.clone().into_tensor_arg(),
                stats.area.clone().into_tensor_arg(),
                stats.top.clone().into_tensor_arg(),
                stats.left.clone().into_tensor_arg(),
                stats.right.clone().into_tensor_arg(),
                stats.bottom.clone().into_tensor_arg(),
                stats.max_label.clone().into_tensor_arg(),
                stats_opt,
                dtypes,
            )
        };
        if stats_opt.compact_labels {
            let max_label = CubeBackend::int_max(stats.max_label.clone());
            let max_label = into_data_sync(max_label);
            let max_label = ToElement::to_usize(&max_label.iter::<i32>().next().unwrap());
            let sliced = kernel::slice(
                stats.area.clone(),
                #[allow(clippy::single_range_in_vec_init)]
                &[0..max_label + 1],
            );
            let relabel = prefix_sum(sliced, int_dtype);

            let cube_dim = CubeDim::new_2d(32, 8);
            let cube_count = CubeCount::new_2d(
                (cols as u32).div_ceil(cube_dim.x),
                (rows as u32).div_ceil(cube_dim.y),
            );
            // Fresh destinations keep obsolete labels from retaining duplicate statistics.
            let source_stats = stats;
            stats = stats_from_opts(labels.clone(), stats_opt, int_dtype);
            unsafe {
                compact_labels::launch_unchecked(
                    &client,
                    cube_count,
                    cube_dim,
                    labels.clone().into_tensor_arg(),
                    relabel.clone().into_tensor_arg(),
                    stats.max_label.clone().into_tensor_arg(),
                    int_storage,
                )
            };

            let cube_dim = CubeDim::new_1d(256);
            let cube_count = CubeCount::new_1d((max_label + 1).div_ceil(256) as u32);
            unsafe {
                compact_stats::launch_unchecked(
                    &client,
                    cube_count,
                    cube_dim,
                    source_stats.area.into_tensor_arg(),
                    stats.area.clone().into_tensor_arg(),
                    source_stats.top.into_tensor_arg(),
                    stats.top.clone().into_tensor_arg(),
                    source_stats.left.into_tensor_arg(),
                    stats.left.clone().into_tensor_arg(),
                    source_stats.right.into_tensor_arg(),
                    stats.right.clone().into_tensor_arg(),
                    source_stats.bottom.into_tensor_arg(),
                    stats.bottom.clone().into_tensor_arg(),
                    relabel.into_tensor_arg(),
                    stats_opt.bounds_enabled,
                    int_storage,
                )
            };
        }
    }

    Ok((labels, stats))
}
