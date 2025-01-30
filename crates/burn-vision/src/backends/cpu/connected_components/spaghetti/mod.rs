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
//! IEEE Transactions on Pattern Analisys and Machine Intelligence, 2021

#![allow(
    unreachable_code,
    clippy::collapsible_else_if,
    clippy::if_same_then_else
)]

use ndarray::{s, Array2, ArrayView2, Axis};

#[allow(non_snake_case)]
mod Spaghetti_forest_labels;
pub(crate) use Spaghetti_forest_labels::*;

use super::Solver;

pub fn process<LabelsSolver: Solver>(img: ArrayView2<u8>) -> Array2<u32> {
    let (h, w) = img.dim();

    let e_rows = h as u32 & 0xfffffffe;
    let o_rows = h % 2 == 1;
    let e_cols = w as u32 & 0xfffffffe;
    let o_cols = w % 2 == 1;

    let mut img_labels = Array2::default(img.raw_dim());

    let mut solver = LabelsSolver::init(((h + 1) / 2) * ((w + 1) / 2) + 1);

    let solver = &mut solver;

    let w = w as i32;

    if h == 1 {
        // Single line
        let r = 0;
        // Row pointers for the input image
        let img_row00 = img.index_axis(Axis(0), r);

        // Row pointers for the output image
        let mut img_labels_row00 = img_labels.slice_mut(s![r, ..]);
        let mut c = -2i32;
        let entry = singleLabels::sl_tree_0;

        include!("Spaghetti_single_line_forest_code.rs");
    } else {
        // More than one line

        // First couple of lines
        {
            let img_row00 = img.index_axis(Axis(0), 0);
            let img_row01 = img.index_axis(Axis(0), 1);
            let mut img_labels_row00 = img_labels.slice_mut(s![0, ..]);
            let mut c = -2i32;
            let entry = firstLabels::fl_tree_0;

            include!("Spaghetti_first_line_forest_code.rs");
        }

        // Every other line but the last one if image has an odd number of rows
        for r in (2..e_rows as usize).step_by(2) {
            // Row pointers for the input image
            let img_row00 = img.index_axis(Axis(0), r);
            let img_row12 = img.index_axis(Axis(0), r - 2);
            let img_row11 = img.index_axis(Axis(0), r - 1);
            let img_row01 = img.index_axis(Axis(0), r + 1);

            // Row pointers for the output image
            let (mut img_labels_row00, img_labels_row12) =
                img_labels.multi_slice_mut((s![r, ..], s![r - 2, ..]));

            let mut c = -2;
            let entry = centerLabels::cl_tree_0;

            include!("Spaghetti_center_line_forest_code.rs");
        }

        if o_rows {
            let r = h - 1;
            // Row pointers for the input image
            let img_row00 = img.index_axis(Axis(0), r);
            let img_row12 = img.index_axis(Axis(0), r - 2);
            let img_row11 = img.index_axis(Axis(0), r - 1);

            // Row pointers for the output image
            let (mut img_labels_row00, img_labels_row12) =
                img_labels.multi_slice_mut((s![r, ..], s![r - 2, ..]));
            let mut c = -2;
            let entry = lastLabels::ll_tree_0;

            include!("Spaghetti_last_line_forest_code.rs");
        }
    }

    solver.flatten();

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
            if i_label > 0 {
                i_label = solver.get_label(i_label);
                if img_row00[c] > 0 {
                    img_labels_row00[c] = i_label;
                } else {
                    img_labels_row00[c] = 0;
                }
                if img_row00[c + 1] > 0 {
                    img_labels_row00[c + 1] = i_label;
                } else {
                    img_labels_row00[c + 1] = 0;
                }
                if img_row01[c] > 0 {
                    img_labels_row01[c] = i_label;
                } else {
                    img_labels_row01[c] = 0;
                }
                if img_row01[c + 1] > 0 {
                    img_labels_row01[c + 1] = i_label;
                } else {
                    img_labels_row01[c + 1] = 0;
                }
            } else {
                img_labels_row00[c] = 0;
                img_labels_row00[c + 1] = 0;
                img_labels_row01[c] = 0;
                img_labels_row01[c + 1] = 0;
            }
        }
        if o_cols {
            let c = e_cols as usize;
            let mut i_label = img_labels_row00[c];
            if i_label > 0 {
                i_label = solver.get_label(i_label);
                if img_row00[c] > 0 {
                    img_labels_row00[c] = i_label;
                } else {
                    img_labels_row00[c] = 0;
                }
                if img_row01[c] > 0 {
                    img_labels_row01[c] = i_label;
                } else {
                    img_labels_row01[c] = 0;
                }
            } else {
                img_labels_row00[c] = 0;
                img_labels_row01[c] = 0;
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
            if i_label > 0 {
                i_label = solver.get_label(i_label);
                if img_row00[c] > 0 {
                    img_labels_row00[c] = i_label;
                } else {
                    img_labels_row00[c] = 0;
                }
                if img_row00[c + 1] > 0 {
                    img_labels_row00[c + 1] = i_label;
                } else {
                    img_labels_row00[c + 1] = 0;
                }
            } else {
                img_labels_row00[c] = 0;
                img_labels_row00[c + 1] = 0;
            }
        }
        if o_cols {
            let c = e_cols as usize;
            let mut i_label = img_labels_row00[c];
            if i_label > 0 {
                i_label = solver.get_label(i_label);
                if img_row00[c] > 0 {
                    img_labels_row00[c] = i_label;
                } else {
                    img_labels_row00[c] = 0;
                }
            } else {
                img_labels_row00[c] = 0;
            }
        }
    }

    img_labels
}
