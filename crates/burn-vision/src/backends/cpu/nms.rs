use crate::NmsOptions;
use aligned_vec::{AVec, ConstAlign};
use alloc::vec::Vec;
use burn_tensor::{Int, Shape, Tensor, TensorData, backend::Backend};
use macerator::{Scalar, Simd, Vector, vload};

/// Perform NMS on CPU using SIMD acceleration.
///
/// This implementation:
/// 1. Sorts boxes by score (descending)
/// 2. Iteratively selects the highest-scoring non-suppressed box
/// 3. Suppresses all boxes with IoU > threshold using SIMD
pub fn nms<B: Backend>(
    boxes: Tensor<B, 2>,
    scores: Tensor<B, 1>,
    options: NmsOptions,
) -> Tensor<B, 1, Int> {
    let device = boxes.device();
    let [n_boxes, _] = boxes.shape().dims();
    if n_boxes == 0 {
        return Tensor::<B, 1, Int>::empty([0], &device);
    }

    // Get raw data
    let boxes_data = boxes.to_data();
    let boxes_vec: Vec<f32> = boxes_data.to_vec().unwrap();

    let scores_data = scores.to_data();
    let scores_vec: Vec<f32> = scores_data.to_vec().unwrap();

    let keep = nms_vec(boxes_vec, scores_vec, options);
    let n_kept = keep.len();
    let indices_data = TensorData::new(keep, Shape::new([n_kept]));
    Tensor::<B, 1, Int>::from_data(indices_data, &device)
}

/// Perform NMS on CPU using SIMD acceleration.
fn nms_vec(boxes_vec: Vec<f32>, scores_vec: Vec<f32>, options: NmsOptions) -> Vec<i32> {
    let n_boxes = scores_vec.len();

    if n_boxes == 0 {
        return vec![];
    }

    // Filter by score threshold first
    let mut filtered_indices = Vec::with_capacity(n_boxes);

    for i in 0..n_boxes {
        let score = scores_vec[i];
        if score >= options.score_threshold {
            filtered_indices.push(i); // original index
        }
    }

    let n_filtered = filtered_indices.len();
    if n_filtered == 0 {
        return vec![];
    }

    // Sort by score descending
    filtered_indices.sort_by(|&a, &b| scores_vec[b].total_cmp(&scores_vec[a]));

    const ALIGN: usize = 64;
    const FLOATS_PER_ALIGN: usize = ALIGN / size_of::<f32>(); // 16
    let stride = (n_filtered + FLOATS_PER_ALIGN - 1) / FLOATS_PER_ALIGN * FLOATS_PER_ALIGN;
    let mut buf: AVec<f32, ConstAlign<64>> = AVec::with_capacity(ALIGN, stride * 5);
    buf.resize(stride * 5, 0.0);

    let (x1s, rest) = buf.split_at_mut(stride);
    let (y1s, rest) = rest.split_at_mut(stride);
    let (x2s, rest) = rest.split_at_mut(stride);
    let (y2s, areas) = rest.split_at_mut(stride);

    // Convert filtered boxes to SoA format
    for (j, &orig_idx) in filtered_indices.iter().enumerate() {
        let x1 = boxes_vec[orig_idx * 4];
        let y1 = boxes_vec[orig_idx * 4 + 1];
        let x2 = boxes_vec[orig_idx * 4 + 2];
        let y2 = boxes_vec[orig_idx * 4 + 3];
        x1s[j] = x1;
        y1s[j] = y1;
        x2s[j] = x2;
        y2s[j] = y2;
        areas[j] = (x2 - x1) * (y2 - y1);
    }

    // Apply NMS with SIMD dispatch
    let mut suppressed = vec![false; stride];
    let mut keep = Vec::new();

    for i in 0..n_filtered {
        if suppressed[i] {
            continue;
        }

        // Optimization to reduce inner loop comparisons
        suppressed[i] = true;
        keep.push(filtered_indices[i] as i32); // original index

        if options.max_output_boxes > 0 && keep.len() >= options.max_output_boxes {
            break;
        }

        // Suppress overlapping boxes using SIMD
        suppress_overlapping(
            x1s[i],
            y1s[i],
            x2s[i],
            y2s[i],
            areas[i],
            &x1s,
            &y1s,
            &x2s,
            &y2s,
            &areas,
            &mut suppressed,
            stride,
            options.iou_threshold,
        );
    }

    keep
}

/// SIMD-accelerated suppression of overlapping boxes.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
#[macerator::with_simd]
fn suppress_overlapping<'a, S: Simd>(
    ref_x1: f32,
    ref_y1: f32,
    ref_x2: f32,
    ref_y2: f32,
    ref_area: f32,
    x1s: &'a [f32],
    y1s: &'a [f32],
    x2s: &'a [f32],
    y2s: &'a [f32],
    areas: &'a [f32],
    suppressed: &'a mut [bool],
    n_boxes: usize, // stride, always multiple of lanes
    threshold: f32,
) where
    'a: 'a,
{
    let lanes = f32::lanes::<S>();

    // Splat reference values
    let ref_x1_v: Vector<S, f32> = ref_x1.splat();
    let ref_y1_v: Vector<S, f32> = ref_y1.splat();
    let ref_x2_v: Vector<S, f32> = ref_x2.splat();
    let ref_y2_v: Vector<S, f32> = ref_y2.splat();
    let ref_area_v: Vector<S, f32> = ref_area.splat();
    let thresh_v: Vector<S, f32> = threshold.splat();
    let zero_v: Vector<S, f32> = 0.0f32.splat();

    let mut i = 0;

    let mut mask_buf = core::mem::MaybeUninit::<[bool; 16]>::uninit();
    // Process lanes boxes at a time with SIMD
    while i + lanes <= n_boxes {
        // Skip if all boxes in this chunk are already suppressed
        let mut any_active = false;
        for k in 0..lanes {
            if !suppressed[i + k] {
                any_active = true;
                break;
            }
        }

        if any_active {
            let x1_v: Vector<S, f32> = unsafe { vload(x1s.as_ptr().add(i)) };
            let y1_v: Vector<S, f32> = unsafe { vload(y1s.as_ptr().add(i)) };
            let x2_v: Vector<S, f32> = unsafe { vload(x2s.as_ptr().add(i)) };
            let y2_v: Vector<S, f32> = unsafe { vload(y2s.as_ptr().add(i)) };
            let area_v: Vector<S, f32> = unsafe { vload(areas.as_ptr().add(i)) };

            // Compute intersection coordinates
            let xx1 = ref_x1_v.max(x1_v);
            let yy1 = ref_y1_v.max(y1_v);
            let xx2 = ref_x2_v.min(x2_v);
            let yy2 = ref_y2_v.min(y2_v);

            // Compute intersection area (clamp to 0 for non-overlapping)
            let w = (xx2 - xx1).max(zero_v);
            let h = (yy2 - yy1).max(zero_v);
            let inter = w * h;

            // Compute IoU
            let union = ref_area_v + area_v - inter;
            let iou = inter / union;

            // Get suppression mask (IoU > threshold)
            let suppress_mask = iou.gt(thresh_v);

            // Extract mask to bool array and apply to suppressed
            // SAFETY: mask_store_as_bool writes exactly `lanes` bools, we only read 0..lanes
            unsafe { f32::mask_store_as_bool::<S>(mask_buf.as_mut_ptr().cast(), suppress_mask) };
            let mask_buf = unsafe { mask_buf.assume_init() };

            for k in 0..lanes {
                if mask_buf[k] {
                    suppressed[i + k] = true;
                }
            }
        }

        i += lanes;
    }
}
