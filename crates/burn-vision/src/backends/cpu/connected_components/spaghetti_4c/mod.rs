//! Spaghetti algorithm for connected component labeling, modified for 4-connectivity using the
//! 4-connected Rosenfeld mask.
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

#![allow(unreachable_code)]

use burn_tensor::{Element, ElementConversion};
use ndarray::Array2;

use crate::Connectivity;

use super::{max_labels, Solver, StatsOp};

#[allow(non_snake_case)]
mod Spaghetti4C_forest_labels;
pub(crate) use Spaghetti4C_forest_labels::*;

pub fn process<I: Element, B: Element, LabelsSolver: Solver<I>>(
    img: Vec<B>,
    h: usize,
    w: usize,
    stats: &mut impl StatsOp<I>,
) -> Array2<I> {
    let img = img.as_ptr();

    let mut img_labels: Vec<I> = vec![0.elem(); h * w];

    // A quick and dirty upper bound for the maximum number of labels.
    // Following formula comes from the fact that a 2x2 block in 4-connectivity case
    // can never have more than 2 new labels and 1 label for background.
    // Worst case image example pattern:
    // 1 0 1 0 1...
    // 0 1 0 1 0...
    // 1 0 1 0 1...
    // ............
    let max_labels = max_labels(h, w, Connectivity::Four);

    let mut solver = LabelsSolver::init(max_labels);
    let solver = &mut solver;

    let w = w as i32;
    // SAFETY:
    // This code is generated from constraints and includes manual bounds checks, so unchecked pointer
    // indexes are always safe.
    unsafe {
        // First row
        {
            let r = 0;
            //Pointers:
            // Row pointers for the input image
            let img_row00 = img.add(r * w as usize);

            // Row pointers for the output image
            let img_labels_row00 = img_labels.as_mut_ptr().add(r * w as usize);
            let mut c = -1i32;

            let entry = firstLabels::fl_tree_0;

            include!("Spaghetti4C_first_line_forest_code.rs");
        }

        for r in 1..h {
            //Pointers:
            // Row pointers for the input image
            let img_row00 = img.add(r * w as usize);
            let img_row11 = img.add((r - 1) * w as usize);

            // Row pointers for the output image
            let img_labels_row00 = img_labels.as_mut_ptr().add(r * w as usize);
            let img_labels_row11 = img_labels.as_mut_ptr().add((r - 1) * w as usize);
            let mut c = -1i32;

            let entry = centerLabels::cl_tree_0;

            include!("Spaghetti4C_center_line_forest_code.rs");
        }
    }

    let n_labels = solver.flatten();
    stats.init(n_labels.to_usize());

    // SAFETY: This is always valid
    let mut img_labels = unsafe { Array2::from_shape_vec_unchecked((h, w as usize), img_labels) };

    img_labels.indexed_iter_mut().for_each(|((r, c), label)| {
        *label = solver.get_label(*label);
        stats.update(r, c, *label);
    });

    stats.finish();

    img_labels
}
