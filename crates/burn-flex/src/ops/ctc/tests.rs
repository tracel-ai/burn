use super::*;
use burn_backend::{
    TensorData,
    ops::{FloatTensorOps, IntTensorOps},
};
use burn_std::{FloatDType, IntDType};
use std::panic::{AssertUnwindSafe, catch_unwind};

type Inputs = [FlexTensor; 4];

const FLOAT_DTYPES: [FloatDType; 4] = [
    FloatDType::F32,
    FloatDType::F64,
    FloatDType::F16,
    FloatDType::BF16,
];

fn with_float_dtype(inputs: &Inputs, dtype: FloatDType) -> Inputs {
    let mut inputs = inputs.clone();
    inputs[0] = Flex::float_cast(inputs[0].clone(), dtype);
    inputs
}

fn run(inputs: &Inputs, native: bool, blank: usize) -> FlexTensor {
    let [scores, targets, input_lengths, target_lengths] = inputs.clone();
    if native {
        ctc_loss(scores, targets, input_lengths, target_lengths, blank)
    } else {
        ctc_loss_default::<Flex>(scores, targets, input_lengths, target_lengths, blank)
    }
}

fn optimized(inputs: &Inputs, blank: usize) -> FlexTensor {
    let [scores, targets, input_lengths, target_lengths] = inputs;
    try_ctc_loss(scores, targets, input_lengths, target_lengths, blank)
        .expect("fixture must use native CTC")
}

fn assert_close(actual: FlexTensor, expected: &FlexTensor) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    let epsilon = expected.dtype().finfo().unwrap().epsilon;
    let tolerance = match expected.dtype() {
        DType::F32 => 2e-5,
        DType::F64 => 16.0 * epsilon,
        _ => 2.0 * epsilon,
    };
    for (actual, expected) in actual
        .into_data()
        .iter::<f64>()
        .zip(expected.clone().into_data().iter::<f64>())
    {
        if expected.is_finite() {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "{actual} != {expected}"
            );
            if actual == 0.0 && expected == 0.0 {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        } else if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual, expected);
        }
    }
}

fn small_inputs() -> Inputs {
    [
        FlexTensor::from_data(TensorData::new(
            (0..24).map(|i| -0.4f32 - (i % 7) as f32 * 0.2).collect(),
            [4, 2, 3],
        )),
        FlexTensor::from_data(TensorData::from([[1i32, 1], [2, 0]])),
        FlexTensor::from_data(TensorData::from([4i32, 3])),
        FlexTensor::from_data(TensorData::from([2i32, 1])),
    ]
}

#[test]
fn rolling_rows_match_default_with_reused_and_parallel_scratch() {
    let (time, batch, classes, width) = (256, 16, 8, 64);
    let inputs = [
        FlexTensor::from_data(TensorData::new(
            (0..time * batch * classes)
                .map(|i| -0.2 - (i % 29) as f32 * 0.13)
                .collect(),
            [time, batch, classes],
        )),
        FlexTensor::from_data(TensorData::new(
            (0..batch * width).map(|i| (i / 2 % 6) as i32).collect(),
            [batch, width],
        )),
        FlexTensor::from_data(TensorData::new(
            (0..batch).map(|i| (time - i) as i32).collect(),
            [batch],
        )),
        FlexTensor::from_data(TensorData::new(
            (0..batch).map(|i| [64i32, 48, 0, 64][i % 4]).collect(),
            [batch],
        )),
    ];
    for dtype in FLOAT_DTYPES {
        let inputs = with_float_dtype(&inputs, dtype);
        let snapshots = inputs.each_ref().map(|input| input.bytes().to_vec());
        let expected = run(&inputs, false, 7);
        assert_close(optimized(&inputs, 7), &expected);
        #[cfg(feature = "rayon")]
        for workers in [1, 2] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| assert_close(optimized(&inputs, 7), &expected));
        }
        for (input, before) in inputs.iter().zip(snapshots) {
            assert_eq!(input.bytes(), before);
        }
    }
}

#[test]
fn rolling_rows_read_logical_views_and_empty_target_storage() {
    for dtype in FLOAT_DTYPES {
        let mut inputs = with_float_dtype(&small_inputs(), dtype);
        let layout = inputs[0].layout().flip(&[0, 2]).narrow(0, 1, 3);
        inputs[0] = inputs[0].clone().with_layout(layout);
        for input in &mut inputs[1..] {
            *input = input.clone().with_layout(input.layout().flip(&[0]));
        }
        inputs[2] = FlexTensor::from_data(TensorData::from([3i32, 3, -1]));
        inputs[2] = inputs[2]
            .clone()
            .with_layout(inputs[2].layout().narrow(0, 0, 2));
        assert_close(optimized(&inputs, 0), &run(&inputs, false, 0));

        let mut inputs = small_inputs();
        inputs[1] = FlexTensor::from_data(TensorData::new(Vec::<i32>::new(), [2, 0]));
        inputs[3] = FlexTensor::from_data(TensorData::from([0i32, 0]));
        for value in [-0.0f32, -1e30, -f32::MAX] {
            inputs[0] = Flex::float_cast(
                FlexTensor::from_data(TensorData::new(vec![value; 24], [4, 2, 3])),
                dtype,
            );
            assert_close(optimized(&inputs, 0), &run(&inputs, false, 0));
        }
    }
}

#[test]
fn rolling_rows_preserve_half_rounding_and_f64_precision() {
    let mut inputs = [
        FlexTensor::from_data(TensorData::new(vec![-0.1f32; 64], [32, 1, 2])),
        FlexTensor::from_data(TensorData::new(Vec::<i32>::new(), [1, 0])),
        FlexTensor::from_data(TensorData::from([32i32])),
        FlexTensor::from_data(TensorData::from([0i32])),
    ];
    for dtype in [FloatDType::F16, FloatDType::BF16] {
        let half = with_float_dtype(&inputs, dtype);
        let expected = run(&half, false, 0);
        assert_eq!(optimized(&half, 0).bytes(), expected.bytes());
        // Promoting the recurrence and casting only its result loses timestep rounding.
        let promoted = with_float_dtype(&half, FloatDType::F32);
        let promoted = Flex::float_cast(run(&promoted, false, 0), dtype);
        assert_ne!(promoted.bytes(), expected.bytes());

        let half = with_float_dtype(&small_inputs(), dtype);
        assert_eq!(optimized(&half, 0).bytes(), run(&half, false, 0).bytes());
    }

    let emission = -0.123456789012345f64;
    inputs[0] = FlexTensor::from_data(TensorData::new(vec![emission; 6], [3, 1, 2]));
    inputs[2] = FlexTensor::from_data(TensorData::from([3i32]));
    let actual = optimized(&inputs, 0);
    assert_close(actual.clone(), &run(&inputs, false, 0));
    assert!((actual.storage::<f64>()[0] + 3.0 * emission).abs() < 1e-14);
}

#[test]
fn rolling_rows_support_matching_integer_dtypes() {
    for dtype in [
        IntDType::I8,
        IntDType::I16,
        IntDType::I32,
        IntDType::I64,
        IntDType::U8,
        IntDType::U16,
        IntDType::U32,
        IntDType::U64,
    ] {
        let mut inputs = small_inputs();
        for tensor in &mut inputs[1..] {
            *tensor = Flex::int_cast(tensor.clone(), dtype);
        }
        assert_close(optimized(&inputs, 0), &run(&inputs, false, 0));
    }

    // Classes need not fit the index dtype when the labels and blank do.
    let mut inputs = small_inputs();
    inputs[0] = FlexTensor::from_data(TensorData::new(
        (0..8000).map(|i| -0.4f32 - (i % 29) as f32 * 0.1).collect(),
        [4, 2, 1000],
    ));
    for tensor in &mut inputs[1..] {
        *tensor = Flex::int_cast(tensor.clone(), IntDType::I8);
    }
    assert_close(optimized(&inputs, 0), &run(&inputs, false, 0));
}

#[test]
fn unsupported_inputs_preserve_default_results_and_panics() {
    let base = small_inputs();
    let mut cases = Vec::new();
    let mut inputs = base.clone();
    inputs[1] = FlexTensor::from_data(TensorData::from([[1i64, 1], [2, 0]]));
    cases.push(inputs);
    // Unsigned endpoint subtraction wraps for empty targets in the default path.
    for dtype in [IntDType::U8, IntDType::U16, IntDType::U32, IntDType::U64] {
        let mut inputs = base.clone();
        inputs[3] = FlexTensor::from_data(TensorData::from([0i32, 1]));
        for tensor in &mut inputs[1..] {
            *tensor = Flex::int_cast(tensor.clone(), dtype);
        }
        cases.push(inputs);
    }
    for score in [f32::NAN, f32::INFINITY, 1.0] {
        let mut inputs = base.clone();
        inputs[0] = FlexTensor::from_data(TensorData::new(vec![score; 24], [4, 2, 3]));
        cases.push(inputs);
    }
    let mut inputs = base.clone();
    inputs[1] = FlexTensor::from_data(TensorData::from([[1i32, 1], [2, 99]]));
    cases.push(inputs);
    for lengths in [[0i32, 3], [5, 3]] {
        let mut inputs = base.clone();
        inputs[2] = FlexTensor::from_data(TensorData::from(lengths));
        cases.push(inputs);
    }
    let mut inputs = base;
    inputs[0] = FlexTensor::from_data(TensorData::new(Vec::<f32>::new(), [0, 2, 3]));
    cases.push(inputs);
    for inputs in cases {
        let [scores, targets, input_lengths, target_lengths] = &inputs;
        assert!(try_ctc_loss(scores, targets, input_lengths, target_lengths, 0).is_none());
        let expected = catch_unwind(AssertUnwindSafe(|| run(&inputs, false, 0)));
        let actual = catch_unwind(AssertUnwindSafe(|| run(&inputs, true, 0)));
        match (actual, expected) {
            (Ok(actual), Ok(expected)) => {
                assert_eq!(actual.dtype(), expected.dtype());
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(actual.bytes(), expected.bytes());
            }
            (Err(_), Err(_)) => {}
            _ => panic!("native fallback changed the default outcome"),
        }
    }
}
