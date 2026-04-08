#![cfg(all(feature = "vision", feature = "ndarray"))]

use burn::{
    Tensor,
    backend::NdArray,
    tensor::TensorData,
    vision::{Nms, NmsOptions},
};

type TestBackend = NdArray<f32, i32>;

#[test]
fn nms_is_available_with_vision_and_ndarray_features() {
    let boxes = Tensor::<TestBackend, 2>::from([
        [0.0, 0.0, 100.0, 100.0],
        [0.0, 1.0, 100.0, 100.0],
        [0.0, 101.0, 200.0, 200.0],
        [0.0, 100.0, 200.0, 200.0],
        [0.0, 170.0, 300.0, 300.0],
    ]);
    let scores = Tensor::<TestBackend, 1>::from([0.1, 0.2, 0.4, 0.3, 0.5]);
    let options = NmsOptions {
        iou_threshold: 0.5,
        score_threshold: 0.0,
        max_output_boxes: 0,
    };

    let output = boxes.nms(scores, options);

    output
        .into_data()
        .assert_eq(&TensorData::from([4, 2, 1]), false);
}
