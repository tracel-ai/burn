use std::{
    cmp::{self},
    ops::Range,
    sync::mpsc::SyncSender,
};

use burn_tensor::{
    ElementConversion, Shape, TensorMetadata,
    backend::{Backend, DeviceOps},
};

pub struct Aggregator {}

pub type AggregationId = u32;

#[derive(Debug, PartialEq, Clone)]
pub enum AggregateStrategy {
    Centralized,
    Tree(u32),
    Ring,
}

#[derive(Debug, PartialEq, Clone)]
pub enum AggregateKind {
    Sum,
    Mean,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AggregateParams {
    pub kind: AggregateKind,
    pub strategy: AggregateStrategy,
}

#[derive(Debug)]
pub enum Message<B: Backend> {
    Aggregate {
        tensor: B::FloatTensorPrimitive,
        params: AggregateParams,
        callback: SyncSender<B::FloatTensorPrimitive>,
    },
    Register {
        id: u32,
        num_nodes: u32,
        callback: SyncSender<()>,
    },
    Reset,
}

#[derive(Clone)]
pub struct AggregatorClient<B: Backend> {
    channel: SyncSender<Message<B>>,
}

impl<B: Backend> AggregatorClient<B> {
    pub fn reset(&self) {
        self.channel.send(Message::Reset).unwrap();
    }

    pub fn register(&self, id: u32, num_nodes: u32) {
        let (callback, rec) = std::sync::mpsc::sync_channel::<()>(1);

        self.channel
            .send(Message::Register {
                id,
                num_nodes,
                callback,
            })
            .unwrap();

        rec.recv().unwrap();
    }

    pub fn aggregate(
        &self,
        tensor: B::FloatTensorPrimitive,
        kind: AggregateKind,
        strategy: AggregateStrategy,
    ) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<B::FloatTensorPrimitive>(1);
        let params = AggregateParams { kind, strategy };

        self.channel
            .send(Message::Aggregate {
                tensor,
                params,
                callback,
            })
            .unwrap();

        // returns a tensor primitive that may or may not be on the correct device,
        // depending on the strategy used.
        rec.recv()
            .expect("Failed to receive callback from aggregator")
    }
}

impl Aggregator {
    pub fn start<B: Backend>() -> AggregatorClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let client = AggregatorClient { channel: sender };

        let _handle = std::thread::spawn(move || {
            let mut registered_nodes = vec![];
            let mut cur_params = None;

            let mut tensors = Vec::new();
            let mut callbacks_register = Vec::new();
            let mut callbacks_aggregate = Vec::new();

            while let Ok(message) = rec.recv() {
                match message {
                    Message::Aggregate {
                        tensor,
                        params,
                        callback,
                    } => {
                        if tensors.is_empty() || cur_params.is_none() {
                            cur_params = Some(params);
                        } else if *cur_params.as_ref().unwrap() != params {
                            panic!(
                                "Trying to aggregate a different way ({:?}) than is currently
                                    being done ({:?})",
                                params, cur_params,
                            );
                        }

                        tensors.push(tensor);
                        callbacks_aggregate.push(callback);
                    }
                    Message::Register {
                        id,
                        num_nodes,
                        callback,
                    } => {
                        if registered_nodes.contains(&id) {
                            panic!("Cannot register a node twice!");
                        }
                        registered_nodes.push(id);
                        callbacks_register.push(callback);
                        if registered_nodes.len() == num_nodes as usize {
                            for callback in callbacks_register.drain(..) {
                                callback.send(()).unwrap();
                            }
                        }
                    }
                    Message::Reset => {
                        registered_nodes.clear();
                        tensors.clear();
                        cur_params = None;
                    }
                }

                let tensor_count = tensors.len();
                if tensor_count > 0 && tensor_count == registered_nodes.len() {
                    let kind = &cur_params.as_ref().unwrap().kind;
                    let strategy = &cur_params.as_ref().unwrap().strategy;
                    match &strategy {
                        &strategy => {
                            let mut outs = match strategy {
                                AggregateStrategy::Centralized => {
                                    let out = aggregate_centralized::<B>(&mut tensors, kind);
                                    vec![out; tensor_count]
                                }
                                AggregateStrategy::Tree(arity) => {
                                    let out = aggregate_tree::<B>(&mut tensors, kind, *arity);
                                    vec![out; tensor_count]
                                }
                                AggregateStrategy::Ring => aggregate_ring::<B>(&mut tensors, kind),
                            };

                            for callback in callbacks_aggregate.drain(..) {
                                let out = outs.remove(0);
                                callback.send(out).unwrap();
                            }
                        }
                    };
                }
            }

            log::debug!("Aggregator message failed");
        });

        client
    }
}

fn aggregate_centralized<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &AggregateKind,
) -> B::FloatTensorPrimitive {
    let tensor_count = tensors.len();
    let mut base = tensors.pop().unwrap();

    for tensor in tensors.drain(..) {
        let target_device = B::float_device(&base);
        let tensor = B::float_to_device(tensor, &target_device);
        base = B::float_add(base, tensor);
    }

    if *kind == AggregateKind::Mean {
        base = B::float_div_scalar(base, (tensor_count as f32).elem());
    }

    base
}

fn aggregate_tree<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &AggregateKind,
    arity: u32,
) -> B::FloatTensorPrimitive {
    // Sort by device id
    tensors.sort_by(|a, b| {
        let dev_a = B::float_device(a).id();
        let dev_b = B::float_device(b).id();

        dev_a.cmp(&dev_b)
    });

    let tensor_count = tensors.len() as u32;
    let mut result = if tensor_count > arity {
        // Split tensor vec into chunks
        let chunk_count = cmp::min(arity, tensor_count);
        let chunk_size = tensor_count / chunk_count;
        let chunks: Vec<Vec<B::FloatTensorPrimitive>> = tensors
            .chunks(chunk_size as usize)
            .map(|s| s.into())
            .collect();

        // Recursive reduce
        let mut new_tensors = vec![];
        for mut chunk in chunks {
            new_tensors.push(aggregate_tree::<B>(&mut chunk, kind, arity));
        }
        aggregate_centralized::<B>(&mut new_tensors, &AggregateKind::Sum)
    } else {
        aggregate_centralized::<B>(tensors, &AggregateKind::Sum)
    };

    if *kind == AggregateKind::Mean {
        result = B::float_div_scalar(result, (tensor_count as f32).elem());
    }

    result
}

// TODO the algo used here is definitely not in-place.
// In theory this is a completely in-place algorithm.
fn aggregate_ring<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &AggregateKind,
) -> Vec<B::FloatTensorPrimitive> {
    // https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model
    fn get_slice_dim(shape: &Shape) -> usize {
        // get dimension with greatest size
        shape
            .dims
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(index, _)| index)
            .unwrap()
    }

    let mut shape = None;
    let mut ranges: Vec<Range<usize>> = vec![];

    let tensor_count = tensors.len();
    // verify all shapes are the same
    for tensor in tensors.as_slice() {
        if shape.is_none() {
            shape = Some(tensor.shape());
        } else if tensor.shape() != *shape.as_ref().unwrap() {
            panic!("Cannot aggregate tensors with different sizes");
        }
    }
    let shape = shape.unwrap();

    // Chose and axis and build the slice ranges
    let slice_dim = get_slice_dim(&shape);
    let dim_size = shape.dims[slice_dim];
    let slice_size = if dim_size < tensor_count {
        unimplemented!("Small tensors unsupported for now");
    } else {
        dim_size / tensor_count
    };

    for i in 0..tensor_count {
        let start = i * slice_size;
        let end = start + slice_size;
        ranges.push(Range { start, end });
    }
    ranges.last_mut().unwrap().end = dim_size;

    // split tensors into slices
    let mut sliced_tensors = vec![];
    for tensor in tensors {
        let mut slices = vec![];
        for range in &ranges {
            // TODO could be cached
            let mut range_vec: Vec<Range<usize>> = shape
                .dims
                .iter()
                .map(|d| Range { start: 0, end: *d }.clone())
                .collect();
            range_vec[slice_dim] = range.clone();
            let slice = B::float_slice(tensor.clone(), &range_vec);
            slices.push(slice);
        }
        sliced_tensors.push(slices);
    }

    // phase 1: aggregate in ring N-1 times (Reduce-Scatter)
    for cycle in 0..(tensor_count - 1) {
        // aggregate slices in a ring
        for i in 0..tensor_count {
            let src_tensor_idx = i;
            let dest_tensor_idx = (i + 1) % tensor_count;
            let slice_idx = (i + (tensor_count - 1) * cycle) % tensor_count;

            let src_slice = sliced_tensors[src_tensor_idx].remove(slice_idx);
            let mut dest_slice = sliced_tensors[dest_tensor_idx].remove(slice_idx);

            let dest_device = B::float_device(&dest_slice);
            let src_slice_on_dest = B::float_to_device(src_slice.clone(), &dest_device);
            dest_slice = B::float_add(dest_slice, src_slice_on_dest);

            sliced_tensors[src_tensor_idx].insert(slice_idx, src_slice);
            sliced_tensors[dest_tensor_idx].insert(slice_idx, dest_slice);
        }
    }

    // phase 2: share (overwrite) in a ring N-1 times (All-Gather)
    for cycle in 0..(tensor_count - 1) {
        // aggregate slices in a ring
        for i in 0..tensor_count {
            let src_tensor_idx = i;
            let dest_tensor_idx = (i + 1) % tensor_count;
            // +1 because we're on the slice *after* the one for the last phase (see the graphs)
            let slice_idx = (i + (tensor_count - 1) * cycle + 1) % tensor_count;

            let src_slice = sliced_tensors[src_tensor_idx].remove(slice_idx);
            let mut dest_slice = sliced_tensors[dest_tensor_idx].remove(slice_idx);

            let dest_device = B::float_device(&dest_slice);
            let src_slice_on_dest = B::float_to_device(src_slice.clone(), &dest_device);
            dest_slice = src_slice_on_dest;

            sliced_tensors[src_tensor_idx].insert(slice_idx, src_slice);
            sliced_tensors[dest_tensor_idx].insert(slice_idx, dest_slice);
        }
    }

    // merge new slices
    let mut results = vec![];
    while let Some(slices) = sliced_tensors.pop() {
        let mut result = B::float_cat(slices, slice_dim);
        if *kind == AggregateKind::Mean {
            result = B::float_div_scalar(result, (tensor_count as f32).elem());
        }

        results.insert(0, result)
    }

    results
}
