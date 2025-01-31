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
    run::<B, NoOp>(img, connectivity, || NoOp).0
}

pub fn connected_components_with_stats<B: Backend>(
    img: BoolTensor<B>,
    connectivity: Connectivity,
    _options: ConnectedStatsOptions,
) -> (IntTensor<B>, ConnectedStatsPrimitive<B>) {
    let device = B::bool_device(&img);
    let (labels, stats) = run::<B, ConnectedStatsOp>(img, connectivity, ConnectedStatsOp::default);
    println!("{stats:?}");
    let stats = finalize_stats(&device, stats);
    (labels, stats)
}

fn run<B: Backend, Stats: StatsOp>(
    img: BoolTensor<B>,
    connectivity: Connectivity,
    stats: impl Fn() -> Stats,
) -> (IntTensor<B>, Vec<Stats>) {
    let device = B::bool_device(&img);
    let img = Tensor::<B, 4, Bool>::from_primitive(img);
    let [batches, _, height, width] = img.shape().dims();
    let img = img.into_data().convert::<u8>().to_vec::<u8>().unwrap();
    let img = Array3::from_shape_vec((batches, height, width), img).unwrap();
    let mut stats_res = Vec::with_capacity(batches);

    let process = match connectivity {
        Connectivity::Four => spaghetti_4c::process::<UnionFind>,
        Connectivity::Eight => spaghetti::process::<UnionFind>,
    };

    let mut stats_0 = stats();
    let mut out = process(img.index_axis(Axis(0), 0), &mut stats_0);
    stats_res.push(stats_0);
    for i in 1..batches {
        let mut stats_i = stats();
        let batch = process(img.index_axis(Axis(0), i), &mut stats_i);
        out.append(Axis(0), batch.view()).unwrap();
        stats_res.push(stats_i);
    }
    let (data, _) = out.into_raw_vec_and_offset();
    let data = TensorData::new(data, Shape::new([batches, height, width]));
    let labels = Tensor::<B, 3, Int>::from_data(data, &device).into_primitive();
    (labels, stats_res)
}

pub trait Solver {
    fn init(max_labels: usize) -> Self;
    /// Hack to get around mutable borrow limitations on methods
    fn merge(label_1: u32, label_2: u32, solver: &mut Self) -> u32;
    fn new_label(&mut self) -> u32;
    fn flatten(&mut self) -> u32;
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

    fn flatten(&mut self) -> u32 {
        let mut k = 1;
        for i in 1..self.labels.len() {
            if self.labels[i] < i as u32 {
                self.labels[i] = self.labels[self.labels[i] as usize];
            } else {
                self.labels[i] = k;
                k += 1;
            }
        }
        k
    }

    fn get_label(&mut self, i_label: u32) -> u32 {
        self.labels[i_label as usize]
    }
}

pub trait StatsOp {
    fn init(&mut self, num_labels: u32);
    fn update(&mut self, row: usize, column: usize, label: u32);
    fn finish(&mut self);
}

struct NoOp;

impl StatsOp for NoOp {
    fn init(&mut self, _num_labels: u32) {}

    fn update(&mut self, _row: usize, _column: usize, _label: u32) {}

    fn finish(&mut self) {}
}

#[derive(Default, Debug)]
struct ConnectedStatsOp {
    pub area: Vec<u32>,
    pub left: Vec<u32>,
    pub top: Vec<u32>,
    pub right: Vec<u32>,
    pub bottom: Vec<u32>,
}

impl StatsOp for ConnectedStatsOp {
    fn init(&mut self, num_labels: u32) {
        let num_labels = num_labels as usize;
        self.area = vec![0; num_labels];
        self.left = vec![u32::MAX; num_labels];
        self.top = vec![u32::MAX; num_labels];
        self.right = vec![0; num_labels];
        self.bottom = vec![0; num_labels];
    }

    fn update(&mut self, row: usize, column: usize, label: u32) {
        let l = label as usize;
        self.area[l] += 1;
        self.left[l] = self.left[l].min(column as u32);
        self.top[l] = self.top[l].min(row as u32);
        self.right[l] = self.right[l].max(column as u32);
        self.bottom[l] = self.bottom[l].max(row as u32);
    }

    fn finish(&mut self) {
        // Background shouldn't have stats
        self.area[0] = 0;
        self.left[0] = 0;
        self.right[0] = 0;
        self.top[0] = 0;
        self.bottom[0] = 0;
    }
}

fn finalize_stats<B: Backend>(
    device: &B::Device,
    stats: Vec<ConnectedStatsOp>,
) -> ConnectedStatsPrimitive<B> {
    let batches = stats.len();
    let max_len = stats.iter().map(|it| it.area.len()).max().unwrap_or(1);
    let mut area = Vec::with_capacity(batches * max_len);
    let mut left = Vec::with_capacity(batches * max_len);
    let mut top = Vec::with_capacity(batches * max_len);
    let mut right = Vec::with_capacity(batches * max_len);
    let mut bottom = Vec::with_capacity(batches * max_len);

    for mut stats in stats {
        stats.area.resize(max_len, 0);
        stats.left.resize(max_len, 0);
        stats.top.resize(max_len, 0);
        stats.right.resize(max_len, 0);
        stats.bottom.resize(max_len, 0);

        area.extend(stats.area);
        left.extend(stats.left);
        top.extend(stats.top);
        right.extend(stats.right);
        bottom.extend(stats.bottom);
    }

    let into_prim = |data: Vec<u32>| {
        let data = TensorData::new(data, Shape::new([batches, max_len]));
        Tensor::<B, 2, Int>::from_data(data, device).into_primitive()
    };

    ConnectedStatsPrimitive {
        area: into_prim(area),
        left: into_prim(left),
        top: into_prim(top),
        right: into_prim(right),
        bottom: into_prim(bottom),
    }
}
