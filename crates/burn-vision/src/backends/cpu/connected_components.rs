use std::cmp::Ordering;

use alloc::vec::Vec;
use burn_tensor::{
    Bool, Element, ElementComparison, ElementConversion, Int, Shape, Tensor, TensorData, backend::Backend, ops::{BoolTensor, IntTensor}
};
use ndarray::Array2;

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
    let (labels, stats) =
        run::<B, ConnectedStatsOp<B::IntElem>>(img, connectivity, ConnectedStatsOp::default);
    let stats = finalize_stats(&device, stats);
    (labels, stats)
}

fn run<B: Backend, Stats: StatsOp<B::IntElem>>(
    img: BoolTensor<B>,
    connectivity: Connectivity,
    stats: impl Fn() -> Stats,
) -> (IntTensor<B>, Stats) {
    let device = B::bool_device(&img);
    let img = Tensor::<B, 2, Bool>::from_primitive(img);
    let [height, width] = img.shape().dims();
    let img = img.into_data();
    let img = img.into_vec::<B::BoolElem>().unwrap();

    let mut stats = stats();

    let out = match connectivity {
        Connectivity::Four => spaghetti_4c::process::<B::IntElem, B::BoolElem, UnionFind<_>>(
            img, height, width, &mut stats,
        ),
        Connectivity::Eight => {
            // SAFETY: This is validated by `TensorData`
            let img = unsafe { Array2::from_shape_vec_unchecked((height, width), img) };
            spaghetti::process::<B::IntElem, B::BoolElem, UnionFind<_>>(img, &mut stats)
        }
    };

    let (data, _) = out.into_raw_vec_and_offset();
    let data = TensorData::new(data, Shape::new([height, width]));
    let labels = Tensor::<B, 2, Int>::from_data(data, &device).into_primitive();
    (labels, stats)
}

pub trait Solver<I: Element> {
    fn init(max_labels: usize) -> Self;
    /// Hack to get around mutable borrow limitations on methods
    fn merge(label_1: I, label_2: I, solver: &mut Self) -> I;
    fn new_label(&mut self) -> I;
    fn flatten(&mut self) -> I;
    fn get_label(&self, i_label: I) -> I;
}

pub(crate) struct UnionFind<I: Element> {
    labels: Vec<I>,
}

impl<I: Element + ElementComparison> Solver<I> for UnionFind<I> {
    fn init(max_labels: usize) -> Self {
        let mut labels = Vec::with_capacity(max_labels);
        labels.push(0.elem());
        Self { labels }
    }

    fn merge(mut label_1: I, mut label_2: I, solver: &mut Self) -> I {
        use Ordering::Less;

        while matches!(solver.labels[label_1.to_usize()].cmp(&label_1), Less) {
            label_1 = solver.labels[label_1.to_usize()];
        }

        while matches!(solver.labels[label_2.to_usize()].cmp(&label_2), Less) {
            label_2 = solver.labels[label_2.to_usize()];
        }

        if matches!(label_1.cmp(&label_2), Less) {
            solver.labels[label_2.to_usize()] = label_1;
            label_1
        } else {
            solver.labels[label_1.to_usize()] = label_2;
            label_2
        }
    }

    fn new_label(&mut self) -> I {
        let len = I::from_elem(self.labels.len());
        self.labels.push(len);
        len
    }

    fn flatten(&mut self) -> I {
        let mut k = 1;
        for i in 1..self.labels.len() {
            if matches!(self.labels[i].cmp(&I::from_elem(i)), Ordering::Less) {
                self.labels[i] = self.labels[self.labels[i].to_usize()];
            } else {
                self.labels[i] = k.elem();
                k += 1;
            }
        }
        k.elem()
    }

    fn get_label(&self, i_label: I) -> I {
        self.labels[i_label.to_usize()]
    }
}

pub trait StatsOp<I: Element> {
    fn init(&mut self, num_labels: usize);
    fn update(&mut self, row: usize, column: usize, label: I);
    fn finish(&mut self);
}

struct NoOp;

impl<I: Element> StatsOp<I> for NoOp {
    fn init(&mut self, _num_labels: usize) {}

    fn update(&mut self, _row: usize, _column: usize, _label: I) {}

    fn finish(&mut self) {}
}

#[derive(Default, Debug)]
struct ConnectedStatsOp<I: Element> {
    pub area: Vec<I>,
    pub left: Vec<I>,
    pub top: Vec<I>,
    pub right: Vec<I>,
    pub bottom: Vec<I>,
}

impl<I: Element> StatsOp<I> for ConnectedStatsOp<I> {
    fn init(&mut self, num_labels: usize) {
        self.area = vec![0.elem(); num_labels];
        self.left = vec![I::MAX; num_labels];
        self.top = vec![I::MAX; num_labels];
        self.right = vec![0.elem(); num_labels];
        self.bottom = vec![0.elem(); num_labels];
    }

    fn update(&mut self, row: usize, column: usize, label: I) {
        let l = label.to_usize();
        unsafe {
            *self.area.get_unchecked_mut(l) =
                I::from_elem((*self.area.get_unchecked(l)).to_usize() + 1);
            *self.left.get_unchecked_mut(l) =
                I::from_elem((*self.left.get_unchecked(l)).to_usize().min(column));
            *self.top.get_unchecked_mut(l) =
                I::from_elem((*self.top.get_unchecked(l)).to_usize().min(row));
            *self.right.get_unchecked_mut(l) =
                I::from_elem((*self.right.get_unchecked(l)).to_usize().max(column));
            *self.bottom.get_unchecked_mut(l) =
                I::from_elem((*self.bottom.get_unchecked(l)).to_usize().max(row));
        }
    }

    fn finish(&mut self) {
        // Background shouldn't have stats
        self.area[0] = 0.elem();
        self.left[0] = 0.elem();
        self.right[0] = 0.elem();
        self.top[0] = 0.elem();
        self.bottom[0] = 0.elem();
    }
}

fn finalize_stats<B: Backend>(
    device: &B::Device,
    stats: ConnectedStatsOp<B::IntElem>,
) -> ConnectedStatsPrimitive<B> {
    let labels = stats.area.len();

    let into_prim = |data: Vec<B::IntElem>| {
        let data = TensorData::new(data, Shape::new([labels]));
        Tensor::<B, 1, Int>::from_data(data, device).into_primitive()
    };

    let max_label = {
        let data = TensorData::new(vec![B::IntElem::from_elem(labels - 1)], Shape::new([1]));
        Tensor::<B, 1, Int>::from_data(data, device).into_primitive()
    };

    ConnectedStatsPrimitive {
        area: into_prim(stats.area),
        left: into_prim(stats.left),
        top: into_prim(stats.top),
        right: into_prim(stats.right),
        bottom: into_prim(stats.bottom),
        max_label,
    }
}

pub fn max_labels(h: usize, w: usize, conn: Connectivity) -> usize {
    match conn {
        Connectivity::Four => (h * w).div_ceil(2) + 1,
        Connectivity::Eight => h.div_ceil(2) * w.div_ceil(2) + 1,
    }
}
