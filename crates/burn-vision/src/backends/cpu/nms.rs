use crate::NmsOptions;
use alloc::vec::Vec;
use burn_tensor::{Int, Shape, Tensor, TensorData, backend::Backend};
use macerator::{Scalar, Simd, Vector, vload_unaligned};

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

    // Filter by score threshold and convert to SoA format in one pass
    let score_thresh = options.score_threshold;
    let mut x1s = Vec::with_capacity(n_boxes);
    let mut y1s = Vec::with_capacity(n_boxes);
    let mut x2s = Vec::with_capacity(n_boxes);
    let mut y2s = Vec::with_capacity(n_boxes);
    let mut areas = Vec::with_capacity(n_boxes);
    let mut filtered_scores = Vec::with_capacity(n_boxes);
    let mut original_indices = Vec::with_capacity(n_boxes);

    for i in 0..n_boxes {
        let score = scores_vec[i];
        if score < score_thresh {
            continue;
        }

        let x1 = boxes_vec[i * 4];
        let y1 = boxes_vec[i * 4 + 1];
        let x2 = boxes_vec[i * 4 + 2];
        let y2 = boxes_vec[i * 4 + 3];
        x1s.push(x1);
        y1s.push(y1);
        x2s.push(x2);
        y2s.push(y2);
        areas.push((x2 - x1 + 1.0) * (y2 - y1 + 1.0));
        filtered_scores.push(score);
        original_indices.push(i);
    }

    let n_filtered = x1s.len();
    if n_filtered == 0 {
        return vec![];
    }

    // Sort by score descending
    let mut order: Vec<usize> = (0..n_filtered).collect();
    order.sort_by(|&a, &b| filtered_scores[b].partial_cmp(&filtered_scores[a]).unwrap());

    // Apply NMS with SIMD dispatch
    let mut suppressed = vec![false; n_filtered];
    let mut keep = Vec::new();

    for &i in &order {
        if suppressed[i] {
            continue;
        }

        // Optimization to reduce inner loop comparisons
        suppressed[i] = true;
        // Store original index, not filtered index
        keep.push(original_indices[i] as i32);

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
            i,
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
    ref_idx: usize,
    threshold: f32,
) where
    'a: 'a,
{
    let lanes = f32::lanes::<S>();
    let n_boxes = x1s.len();

    // Splat reference values
    let ref_x1_v: Vector<S, f32> = ref_x1.splat();
    let ref_y1_v: Vector<S, f32> = ref_y1.splat();
    let ref_x2_v: Vector<S, f32> = ref_x2.splat();
    let ref_y2_v: Vector<S, f32> = ref_y2.splat();
    let ref_area_v: Vector<S, f32> = ref_area.splat();
    let thresh_v: Vector<S, f32> = threshold.splat();
    let zero_v: Vector<S, f32> = 0.0f32.splat();
    let one_v: Vector<S, f32> = 1.0f32.splat();

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
            // Load box coordinates (unaligned since data may not be SIMD-aligned)
            let x1_v: Vector<S, f32> = unsafe { vload_unaligned(x1s.as_ptr().add(i)) };
            let y1_v: Vector<S, f32> = unsafe { vload_unaligned(y1s.as_ptr().add(i)) };
            let x2_v: Vector<S, f32> = unsafe { vload_unaligned(x2s.as_ptr().add(i)) };
            let y2_v: Vector<S, f32> = unsafe { vload_unaligned(y2s.as_ptr().add(i)) };
            let area_v: Vector<S, f32> = unsafe { vload_unaligned(areas.as_ptr().add(i)) };

            // Compute intersection coordinates
            let xx1 = ref_x1_v.max(x1_v);
            let yy1 = ref_y1_v.max(y1_v);
            let xx2 = ref_x2_v.min(x2_v);
            let yy2 = ref_y2_v.min(y2_v);

            // Compute intersection area (clamp to 0 for non-overlapping)
            let w = (xx2 - xx1 + one_v).max(zero_v);
            let h = (yy2 - yy1 + one_v).max(zero_v);
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

    // Scalar remainder
    while i < n_boxes {
        if !suppressed[i] && i != ref_idx {
            let xx1 = ref_x1.max(x1s[i]);
            let yy1 = ref_y1.max(y1s[i]);
            let xx2 = ref_x2.min(x2s[i]);
            let yy2 = ref_y2.min(y2s[i]);

            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;

            let iou = inter / (ref_area + areas[i] - inter);

            if iou > threshold {
                suppressed[i] = true;
            }
        }
        i += 1;
    }
}
