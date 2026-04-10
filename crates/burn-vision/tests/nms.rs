use burn_vision::{Nms, NmsOptions};

mod common;
use common::*;

#[test]
fn should_suppress_non_maximum() {
    let boxes = Tensor::<2>::from([
        [0, 0, 100, 100],
        [0, 1, 100, 100],
        [0, 101, 200, 200],
        [0, 100, 200, 200],
        [0, 170, 300, 300],
    ]);
    let scores = Tensor::<1>::from([0.1, 0.2, 0.4, 0.3, 0.5]);
    let options = NmsOptions {
        iou_threshold: 0.5,
        score_threshold: 0.0,
        max_output_boxes: 0,
    };

    let output = boxes.nms(scores, options);

    let expected = TestTensorInt::<1>::from([4, 2, 1]);
    output.into_data().assert_eq(&expected.into_data(), true);
}

#[test]
fn should_apply_score_threshold() {
    let boxes = Tensor::<2>::from([
        [0, 0, 100, 100],
        [0, 1, 100, 100],
        [0, 101, 200, 200],
        [0, 100, 200, 200],
        [0, 170, 300, 300],
    ]);
    let scores = Tensor::<1>::from([0.1, 0.2, 0.4, 0.3, 0.5]);
    let options = NmsOptions {
        iou_threshold: 0.5,
        score_threshold: 0.3,
        max_output_boxes: 0,
    };

    let output = boxes.nms(scores, options);

    let expected = TestTensorInt::<1>::from([4, 2]);
    output.into_data().assert_eq(&expected.into_data(), true);
}

#[test]
fn should_apply_iou_threshold() {
    let boxes = Tensor::<2>::from([
        [0, 0, 100, 100],
        [0, 1, 100, 100],
        [0, 101, 200, 200],
        [0, 100, 200, 200],
        [0, 170, 300, 300],
    ]);
    let scores = Tensor::<1>::from([0.1, 0.2, 0.4, 0.3, 0.5]);
    let options = NmsOptions {
        iou_threshold: 0.1,
        score_threshold: 0.0,
        max_output_boxes: 0,
    };

    let output = boxes.nms(scores, options);

    let expected = TestTensorInt::<1>::from([4, 1]);
    output.into_data().assert_eq(&expected.into_data(), true);
}

#[test]
fn should_apply_max_output_boxes() {
    let boxes = Tensor::<2>::from([
        [0, 0, 100, 100],
        [0, 1, 100, 100],
        [0, 101, 200, 200],
        [0, 100, 200, 200],
        [0, 170, 300, 300],
    ]);
    let scores = Tensor::<1>::from([0.1, 0.2, 0.4, 0.3, 0.5]);
    let options = NmsOptions {
        iou_threshold: 0.5,
        score_threshold: 0.0,
        max_output_boxes: 1,
    };

    let output = boxes.nms(scores, options);

    let expected = TestTensorInt::<1>::from([4]);
    output.into_data().assert_eq(&expected.into_data(), true);
}
