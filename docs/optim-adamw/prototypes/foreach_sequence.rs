//! A bounded PR3 prerequisite proof: independent strided views in one chunked launch.
//! This exercises argument registration and addressing, not the AdamW algorithm.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_server::launched::LaunchedKernels;
use cubecl_server::logging::{LaunchObservation, LaunchObserver};
use cubecl_server::runtime::Runtime;
use cubecl_server::server::Handle;
use cubecl_std::tensor::layout::linear::{
    LinearView, LinearViewLaunch, LinearViewLayout, LinearViewLayoutLaunch, LinearViewMut,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Chunk {
    tensor: usize,
    start: usize,
    len: usize,
}

#[cube(launch_unchecked)]
fn chunked_views(
    inputs: Sequence<LinearView<'_, Vector<f32, Const<1>>>>,
    history: ComptimeOption<Sequence<LinearView<'_, Vector<f32, Const<1>>>>>,
    mut outputs: Sequence<LinearViewMut<'_, Vector<f32, Const<1>>>>,
    #[comptime] chunks: Vec<Chunk>,
) {
    #[unroll]
    for chunk in 0..comptime![chunks.len()] {
        if CUBE_POS_X == chunk as u32 {
            let tensor = comptime![chunks[chunk].tensor];
            let start = comptime![chunks[chunk].start];
            let len = comptime![chunks[chunk].len];
            let local = UNIT_POS_X as usize;
            if local < len {
                let index = start + local;
                let mut value = inputs.index(tensor).read(index) * Vector::new(2.0f32);
                #[comptime]
                if let ComptimeOption::Some(history) = history.clone() {
                    value += history.index(tensor).read(index);
                }
                outputs.index_mut(tensor).write(index, value);
            }
        }
    }
}

#[derive(Default)]
struct CountLaunches(AtomicUsize);

impl LaunchObserver for CountLaunches {
    fn launched(&self, kernel: &'static str) {
        if kernel.contains("foreach_sequence::chunked_views::") {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn view(handle: Handle, shape: &[usize], strides: &[usize]) -> LinearViewLaunch {
    // Callers allocate enough for the full span of these explicit test layouts.
    let arg = unsafe { TensorArg::from_raw_parts(handle, strides.into(), shape.into()) };
    LinearViewLaunch::new_tensor::<LinearViewLayout>(arg, LinearViewLayoutLaunch::new())
}

#[test]
fn sequence_views_cover_mixed_rank_layouts_and_optional_state_in_one_launch() {
    let client = CpuRuntime::client(&Default::default());
    let layouts: &[(&[usize], &[usize])] = &[
        (&[21], &[1]),
        // Column-major input; the output below is row-major.
        (&[3, 5], &[1, 3]),
        (&[1, 2, 2, 3], &[12, 6, 3, 1]),
    ];
    let counts = Arc::new(CountLaunches::default());
    let _observation = LaunchObservation::new(counts.clone());

    let mut variant_keys = Vec::new();
    for (with_history, reuse_outputs) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        for repetition in 0..2 {
            let mut inputs = SequenceArg::new();
            let mut history = SequenceArg::new();
            let mut outputs = SequenceArg::new();
            let mut expected = Vec::new();
            let mut output_handles = Vec::new();
            let mut input_handles = Vec::new();
            let mut chunks = Vec::new();
            let chunk_len = 8usize;

            for (tensor, &(shape, strides)) in layouts.iter().enumerate() {
                let len: usize = shape.iter().product();
                let physical: Vec<f32> = (0..len).map(|i| (tensor * 100 + i) as f32).collect();
                let input = client.create_from_slice(f32::as_bytes(&physical));
                inputs.push(view(input.clone(), shape, strides));

                let mut out_strides = vec![1; shape.len()];
                for dim in (0..shape.len() - 1).rev() {
                    out_strides[dim] = out_strides[dim + 1] * shape[dim + 1];
                }
                let previous: Vec<f32> = (0..len).map(|i| (tensor * 1000 + i + 1) as f32).collect();
                history.push(view(
                    client.create_from_slice(f32::as_bytes(&previous)),
                    shape,
                    &out_strides,
                ));
                // Only the already contiguous roles alias their original input. The transposed
                // role keeps a fresh output, as the production COW rules require.
                let reused = reuse_outputs && tensor != 1;
                let output = if reused {
                    input.clone()
                } else {
                    client.create_from_slice(f32::as_bytes(&vec![-999.0; len]))
                };
                outputs.push(if reused {
                    LinearViewLaunch::new_tensor::<LinearViewLayout>(
                        TensorArg::Alias {
                            input_pos: tensor,
                            strides: out_strides.clone().into(),
                            shape: shape.into(),
                        },
                        LinearViewLayoutLaunch::new(),
                    )
                } else {
                    view(output.clone(), shape, &out_strides)
                });
                output_handles.push(output);

                let values: Vec<f32> = (0..len)
                    .map(|i| {
                        let physical_index: usize = shape
                            .iter()
                            .zip(strides)
                            .zip(&out_strides)
                            .map(|((&size, &stride), &logical)| (i / logical % size) * stride)
                            .sum();
                        physical[physical_index] * 2.0
                            + if with_history { previous[i] } else { 0.0 }
                    })
                    .collect();
                input_handles.push((input, if reused { values.clone() } else { physical }));
                expected.push(values);
                for start in (0..len).step_by(chunk_len) {
                    chunks.push(Chunk {
                        tensor,
                        start,
                        len: chunk_len.min(len - start),
                    });
                }
            }
            let before = counts.0.load(Ordering::Relaxed);
            let compiled = LaunchedKernels::new();
            // Each cube owns one descriptor; guards exclude threads outside that chunk.
            unsafe {
                chunked_views::launch_unchecked(
                    &client,
                    CubeCount::Static(chunks.len() as u32, 1, 1),
                    CubeDim::new_1d(chunk_len as u32),
                    inputs,
                    if with_history { Some(history) } else { None }.into(),
                    outputs,
                    chunks,
                );
            }
            assert_eq!(counts.0.load(Ordering::Relaxed) - before, 1);
            let keys = compiled.finish();
            assert_eq!(keys.len(), 1);
            if repetition == 0 {
                assert!(variant_keys.iter().all(|previous| *previous != keys));
                variant_keys.push(keys);
            } else {
                assert_eq!(
                    &keys,
                    variant_keys.last().unwrap(),
                    "new handles must use the same kernel key"
                );
            }
            for (handle, expected) in output_handles.into_iter().zip(expected) {
                let bytes = client.read_one(handle).unwrap();
                assert_eq!(f32::from_bytes(&bytes), expected);
            }
            for (handle, expected) in input_handles {
                let bytes = client.read_one(handle).unwrap();
                assert_eq!(f32::from_bytes(&bytes), expected);
            }
        }
    }
}
