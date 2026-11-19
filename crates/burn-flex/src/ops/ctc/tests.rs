use super::*;
use burn_backend::{
    TensorData,
    ops::{FloatTensorOps, IntTensorOps, ModuleOps},
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

fn optimized_backward(inputs: &Inputs, grad_loss: &FlexTensor, blank: usize) -> FlexTensor {
    let [scores, targets, input_lengths, target_lengths] = inputs;
    try_ctc(
        scores,
        targets,
        input_lengths,
        target_lengths,
        Some(grad_loss),
        blank,
    )
    .expect("fixture must use native CTC backward")
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
fn native_scratch_is_reused_without_mutating_inputs() {
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
        let grad_loss = Flex::float_cast(
            FlexTensor::from_data(TensorData::new(
                (0..batch)
                    .map(|sample| [-0.5f32, 0.0, 2.0][sample % 3])
                    .collect(),
                [batch],
            )),
            dtype,
        );
        let grad_snapshot = grad_loss.bytes().to_vec();
        let expected = optimized(&inputs, 7);
        let expected_grad = optimized_backward(&inputs, &grad_loss, 7);
        assert_eq!(optimized(&inputs, 7).bytes(), expected.bytes());
        assert_eq!(
            optimized_backward(&inputs, &grad_loss, 7).bytes(),
            expected_grad.bytes()
        );
        #[cfg(feature = "rayon")]
        for workers in [1, 2] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    assert_eq!(optimized(&inputs, 7).bytes(), expected.bytes());
                    assert_eq!(
                        optimized_backward(&inputs, &grad_loss, 7).bytes(),
                        expected_grad.bytes()
                    );
                });
        }
        assert_eq!(grad_loss.bytes(), grad_snapshot);
        for (input, before) in inputs.iter().zip(snapshots) {
            assert_eq!(input.bytes(), before);
        }
    }
}

#[test]
fn native_dispatch_accepts_supported_storage_dtypes() {
    assert!(Flex::has_ctc_loss_backward());
    for dtype in FLOAT_DTYPES {
        let output = optimized(&with_float_dtype(&small_inputs(), dtype), 0);
        assert_eq!(output.dtype(), DType::from(dtype));
        assert_eq!(output.shape(), Shape::new([2]));
        let grad_loss = Flex::float_cast(
            FlexTensor::from_data(TensorData::from([1.0f32, 2.0])),
            dtype,
        );
        let grad = optimized_backward(&with_float_dtype(&small_inputs(), dtype), &grad_loss, 0);
        assert_eq!(grad.dtype(), DType::from(dtype));
        assert_eq!(grad.shape(), Shape::new([4, 2, 3]));
    }

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
        assert!(try_ctc_loss(&inputs[0], &inputs[1], &inputs[2], &inputs[3], 0).is_some());
        let grad_loss = FlexTensor::from_data(TensorData::from([1.0f32, 2.0]));
        assert!(
            try_ctc(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                Some(&grad_loss),
                0
            )
            .is_some()
        );
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
    assert!(try_ctc_loss(&inputs[0], &inputs[1], &inputs[2], &inputs[3], 0).is_some());
}

#[test]
fn unsupported_inputs_preserve_default_results_and_panics() {
    let base = small_inputs();
    for grad_loss in [
        FlexTensor::from_data(TensorData::from([1.0f32])),
        FlexTensor::from_data(TensorData::from([1.0f64, 2.0])),
    ] {
        assert!(try_ctc(&base[0], &base[1], &base[2], &base[3], Some(&grad_loss), 0).is_none());
    }
    let mut cases = Vec::new();
    let mut inputs = base.clone();
    inputs[1] = FlexTensor::from_data(TensorData::from([[1i64, 1], [2, 0]]));
    let grad_loss = FlexTensor::from_data(TensorData::from([0.5f32, -2.0]));
    assert!(
        try_ctc(
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            Some(&grad_loss),
            0
        )
        .is_none()
    );
    let backward = |native| {
        let [scores, targets, input_lengths, target_lengths] = inputs.clone();
        if native {
            ctc_loss_backward(
                scores,
                targets,
                input_lengths,
                target_lengths,
                grad_loss.clone(),
                0,
            )
        } else {
            ctc_loss_backward_default::<Flex>(
                scores,
                targets,
                input_lengths,
                target_lengths,
                grad_loss.clone(),
                0,
            )
        }
    };
    let expected = catch_unwind(AssertUnwindSafe(|| backward(false)));
    let actual = catch_unwind(AssertUnwindSafe(|| backward(true)));
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual.dtype(), expected.dtype());
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.bytes(), expected.bytes());
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(
                actual.downcast_ref::<String>(),
                expected.downcast_ref::<String>()
            );
        }
        _ => panic!("native backward fallback changed the default outcome"),
    }
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
