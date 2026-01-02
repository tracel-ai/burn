//! Spaghetti algorithm for connected component labeling
//! F. Bolelli, S. Allegretti, L. Baraldi, and C. Grana,
//! "Spaghetti Labeling: Directed Acyclic Graphs for Block-Based Bonnected Components Labeling,"
//! IEEE Transactions on Image Processing, vol. 29, no. 1, pp. 1999-2012, 2019.
//!
//! Decision forests are generated using a modified [GRAPHGEN](https://github.com/wingertge/GRAPHGEN)
//! as described in
//!
//! F. Bolelli, S. Allegretti, C. Grana.
//! "One DAG to Rule Them All."
//! IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021

#![allow(
    unreachable_code,
    clippy::collapsible_else_if,
    clippy::if_same_then_else
)]

use std::cmp::Ordering;

use burn_tensor::{Element, ElementComparison, ElementConversion};
use ndarray::{Array2, Axis, s};

#[allow(non_snake_case)]
mod Spaghetti_forest_labels;
pub(crate) use Spaghetti_forest_labels::*;

use crate::Connectivity;

use super::{Solver, StatsOp, max_labels};

pub fn process<I: Element + ElementComparison, B: Element, LabelsSolver: Solver<I>>(
    img_arr: Array2<B>,
    stats: &mut impl StatsOp<I>,
) -> Array2<I> {
    let (h, w) = img_arr.dim();
    let mut img_labels_arr = Array2::<I>::default(img_arr.raw_dim());

    let img = img_arr.as_ptr();

    let e_rows = h as u32 & 0xfffffffe;
    let o_rows = h % 2 == 1;
    let e_cols = w as u32 & 0xfffffffe;
    let o_cols = w % 2 == 1;

    let img_labels = img_labels_arr.as_mut_ptr();

    let mut solver = LabelsSolver::init(max_labels(h, w, Connectivity::Eight));

    let solver = &mut solver;

    let w = w as i32;

    // SAFETY:
    // Generated code includes mathematically proven bounds checks, so raw pointers are a safe speed
    // boost.
    unsafe {
        if h == 1 {
            // Single line
            let r = 0;
            //Pointers:
            // Row pointers for the input image
            let img_row00 = img.add(r * w as usize);

            // Row pointers for the output image
            let img_labels_row00 = img_labels.add(r * w as usize);

            let mut c = -2i32;
            let entry = singleLabels::sl_tree_0;

            include!("Spaghetti_single_line_forest_code.rs");
        } else {
            // More than one line

            // First couple of lines
            {
                let r = 0;
                //Pointers:
                // Row pointers for the input image
                let img_row00 = img.add(r * w as usize);
                let img_row01 = img.add((r + 1) * w as usize);

                // Row pointers for the output image
                let img_labels_row00 = img_labels.add(r * w as usize);

                let mut c = -2i32;
                let entry = firstLabels::fl_tree_0;

                include!("Spaghetti_first_line_forest_code.rs");
            }

            // Every other line but the last one if image has an odd number of rows
            for r in (2..e_rows as usize).step_by(2) {
                //Pointers:
                // Row pointers for the input image
                let img_row00 = img.add(r * w as usize);
                let img_row12 = img.add((r - 2) * w as usize);
                let img_row11 = img.add((r - 1) * w as usize);
                let img_row01 = img.add((r + 1) * w as usize);

                // Row pointers for the output image
                let img_labels_row00 = img_labels.add(r * w as usize);
                let img_labels_row12 = img_labels.add((r - 2) * w as usize);

                let mut c = -2;
                let entry = centerLabels::cl_tree_0;

                include!("Spaghetti_center_line_forest_code.rs");
            }

            if o_rows {
                let r = h - 1;
                //Pointers:
                // Row pointers for the input image
                let img_row00 = img.add(r * w as usize);
                let img_row12 = img.add((r - 2) * w as usize);
                let img_row11 = img.add((r - 1) * w as usize);

                // Row pointers for the output image
                let img_labels_row00 = img_labels.add(r * w as usize);
                let img_labels_row12 = img_labels.add((r - 2) * w as usize);

                let mut c = -2;
                let entry = lastLabels::ll_tree_0;

                include!("Spaghetti_last_line_forest_code.rs");
            }
        }
    }

    let n_labels = solver.flatten();
    stats.init(n_labels.to_usize());

    let img = img_arr;
    let mut img_labels = img_labels_arr;

    for r in (0..e_rows as usize).step_by(2) {
        //Pointers:
        // Row pointers for the input image
        let img_row00 = img.index_axis(Axis(0), r);
        let img_row01 = img.index_axis(Axis(0), r + 1);

        // Row pointers for the output image
        let (mut img_labels_row00, mut img_labels_row01) =
            img_labels.multi_slice_mut((s![r, ..], s![r + 1, ..]));

        for c in (0..e_cols as usize).step_by(2) {
            let mut i_label = img_labels_row00[c];
            if matches!(i_label.cmp(&0.elem()), Ordering::Greater) {
                i_label = solver.get_label(i_label);
                if img_row00[c].to_u8() > 0 {
                    img_labels_row00[c] = i_label;
                    stats.update(r, c, i_label);
                } else {
                    img_labels_row00[c] = 0.elem();
                    stats.update(r, c, 0.elem());
                }
                if img_row00[c + 1].to_u8() > 0 {
                    img_labels_row00[c + 1] = i_label;
                    stats.update(r, c + 1, i_label);
                } else {
                    img_labels_row00[c + 1] = 0.elem();
                    stats.update(r, c + 1, 0.elem());
                }
                if img_row01[c].to_u8() > 0 {
                    img_labels_row01[c] = i_label;
                    stats.update(r + 1, c, i_label);
                } else {
                    img_labels_row01[c] = 0.elem();
                    stats.update(r + 1, c, 0.elem());
                }
                if img_row01[c + 1].to_u8() > 0 {
                    img_labels_row01[c + 1] = i_label;
                    stats.update(r + 1, c + 1, i_label);
                } else {
                    img_labels_row01[c + 1] = 0.elem();
                    stats.update(r + 1, c + 1, 0.elem());
                }
            } else {
                img_labels_row00[c] = 0.elem();
                stats.update(r, c, 0.elem());
                img_labels_row00[c + 1] = 0.elem();
                stats.update(r, c + 1, 0.elem());
                img_labels_row01[c] = 0.elem();
                stats.update(r + 1, c, 0.elem());
                img_labels_row01[c + 1] = 0.elem();
                stats.update(r + 1, c + 1, 0.elem());
            }
        }
        if o_cols {
            let c = e_cols as usize;
            let mut i_label = img_labels_row00[c];
            if matches!(i_label.cmp(&0.elem()), Ordering::Greater) {
                i_label = solver.get_label(i_label);
                if img_row00[c].to_u8() > 0 {
                    img_labels_row00[c] = i_label;
                    stats.update(r, c, i_label);
                } else {
                    img_labels_row00[c] = 0.elem();
                    stats.update(r, c, 0.elem());
                }
                if img_row01[c].to_u8() > 0 {
                    img_labels_row01[c] = i_label;
                    stats.update(r + 1, c, i_label);
                } else {
                    img_labels_row01[c] = 0.elem();
                    stats.update(r + 1, c, 0.elem());
                }
            } else {
                img_labels_row00[c] = 0.elem();
                stats.update(r, c, 0.elem());
                img_labels_row01[c] = 0.elem();
                stats.update(r + 1, c, 0.elem());
            }
        }
    }

    if o_rows {
        let r = e_rows as usize;

        // Row pointers for the input image
        let img_row00 = img.index_axis(Axis(0), r);

        // Row pointers for the output image
        let mut img_labels_row00 = img_labels.slice_mut(s![r, ..]);

        for c in (0..e_cols as usize).step_by(2) {
            let mut i_label = img_labels_row00[c];
            if matches!(i_label.cmp(&0.elem()), Ordering::Greater) {
                i_label = solver.get_label(i_label);
                if img_row00[c].to_u8() > 0 {
                    img_labels_row00[c] = i_label;
                    stats.update(r, c, i_label);
                } else {
                    img_labels_row00[c] = 0.elem();
                    stats.update(r, c, 0.elem());
                }
                if img_row00[c + 1].to_u8() > 0 {
                    img_labels_row00[c + 1] = i_label;
                    stats.update(r, c + 1, i_label);
                } else {
                    img_labels_row00[c + 1] = 0.elem();
                    stats.update(r, c + 1, 0.elem());
                }
            } else {
                img_labels_row00[c] = 0.elem();
                stats.update(r, c, 0.elem());
                img_labels_row00[c + 1] = 0.elem();
                stats.update(r, c + 1, 0.elem());
            }
        }
        if o_cols {
            let c = e_cols as usize;
            let mut i_label = img_labels_row00[c];
            if matches!(i_label.cmp(&0.elem()), Ordering::Greater) {
                i_label = solver.get_label(i_label);
                if img_row00[c].to_u8() > 0 {
                    img_labels_row00[c] = i_label;
                    stats.update(r, c, i_label);
                } else {
                    img_labels_row00[c] = 0.elem();
                    stats.update(r, c, 0.elem());
                }
            } else {
                img_labels_row00[c] = 0.elem();
                stats.update(r, c, i_label);
            }
        }
    }

    stats.finish();
    img_labels
}
