use alloc::vec::Vec;
use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, IntTensor},
    Bool, Int, Shape, Tensor, TensorData,
};
use ndarray::{Array3, Axis};

use crate::{ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity};

mod spaghetti;
mod spaghetti_4c;

pub fn connected_components<B: Backend>(
    img: BoolTensor<B>,
    connectivity: Connectivity,
) -> IntTensor<B> {
    let device = B::bool_device(&img);
    let img = Tensor::<B, 4, Bool>::from_primitive(img);
    let [batches, _, height, width] = img.shape().dims();
    let img = img.into_data().convert::<u8>().to_vec::<u8>().unwrap();
    let img = Array3::from_shape_vec((batches, height, width), img).unwrap();

    let process = match connectivity {
        Connectivity::Four => spaghetti_4c::process::<UnionFind>,
        Connectivity::Eight => spaghetti::process::<UnionFind>,
    };

    let mut out = process(img.index_axis(Axis(0), 0));
    for i in 1..batches {
        let batch = process(img.index_axis(Axis(0), i));
        out.append(Axis(0), batch.view()).unwrap();
    }
    println!("{out:?}");
    let (data, _) = out.into_raw_vec_and_offset();
    let data = TensorData::new(data, Shape::new([batches, height, width]));
    Tensor::<B, 3, Int>::from_data(data, &device).into_primitive()
}

pub fn connected_components_with_stats<B: Backend>(
    _img: BoolTensor<B>,
    _connectivity: Connectivity,
    _options: ConnectedStatsOptions,
) -> (IntTensor<B>, ConnectedStatsPrimitive<B>) {
    todo!()
}

pub trait Solver {
    fn init(max_labels: usize) -> Self;
    /// Hack to get around mutable borrow limitations on methods
    fn merge(label_1: u32, label_2: u32, solver: &mut Self) -> u32;
    fn new_label(&mut self) -> u32;
    fn flatten(&mut self);
    fn get_label(&mut self, i_label: u32) -> u32;
}

pub(crate) struct UnionFind {
    labels: Vec<u32>,
}

impl Solver for UnionFind {
    fn init(max_labels: usize) -> Self {
        let mut labels = Vec::with_capacity(max_labels);
        labels.push(0);
        Self { labels }
    }

    fn merge(mut label_1: u32, mut label_2: u32, solver: &mut Self) -> u32 {
        while solver.labels[label_1 as usize] < label_1 {
            label_1 = solver.labels[label_1 as usize];
        }

        while solver.labels[label_2 as usize] < label_2 {
            label_2 = solver.labels[label_2 as usize];
        }

        if label_1 < label_2 {
            solver.labels[label_2 as usize] = label_1;
            label_1
        } else {
            solver.labels[label_1 as usize] = label_2;
            label_2
        }
    }

    fn new_label(&mut self) -> u32 {
        let len = self.labels.len() as u32;
        self.labels.push(len);
        len
    }

    fn flatten(&mut self) {
        let mut k = 1;
        for i in 1..self.labels.len() {
            if self.labels[i] < i as u32 {
                self.labels[i] = self.labels[self.labels[i] as usize];
            } else {
                self.labels[i] = k;
                k += 1;
            }
        }
    }

    fn get_label(&mut self, i_label: u32) -> u32 {
        self.labels[i_label as usize]
    }
}
