use std::{env, path::Path};

use burn::record::{
    FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder, NamedMpkGzFileRecorder,
    PrettyJsonFileRecorder, Recorder,
};

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig,
    },
    tensor::{
        activation::{log_softmax, relu},
        backend::Backend,
        Tensor,
    },
};

#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
struct Net<B: Backend> {
    conv_blocks: Vec<ConvBlock<B>>,
    norm1: BatchNorm<B, 2>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        let conv_blocks = vec![
            ConvBlock {
                conv: Conv2dConfig::new([2, 4], [3, 2])
                    .init_with(record.conv_blocks[0].conv.clone()),
                norm: BatchNormConfig::new(2).init_with(record.conv_blocks[0].norm.clone()),
            },
            ConvBlock {
                conv: Conv2dConfig::new([4, 6], [3, 2])
                    .init_with(record.conv_blocks[1].conv.clone()),
                norm: BatchNormConfig::new(4).init_with(record.conv_blocks[1].norm.clone()),
            },
        ];
        let norm1 = BatchNormConfig::new(6).init_with(record.norm1);
        let fc1 = LinearConfig::new(120, 12).init_with(record.fc1);
        let fc2 = LinearConfig::new(12, 10).init_with(record.fc2);
        Self {
            conv_blocks,
            norm1,
            fc1,
            fc2,
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv_blocks[0].forward(x);
        let x = self.conv_blocks[1].forward(x);
        let x = self.norm1.forward(x);
        let x = x.reshape([0, -1]);
        let x = self.fc1.forward(x);
        let x = relu(x);
        let x = self.fc2.forward(x);

        log_softmax(x, 1)
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);

        self.norm.forward(x)
    }
}

type TestBackend = burn_ndarray::NdArray<f32>;

fn model_test(record: NetRecord<TestBackend>, precision: usize) {
    let model = Net::<TestBackend>::new_with(record);

    let input = Tensor::<TestBackend, 4>::ones([1, 2, 9, 6]) - 0.5;

    let output = model.forward(input);

    let expected = Tensor::<TestBackend, 2>::from_data([[
        -2.306_613,
        -2.058_945_4,
        -2.298_372_7,
        -2.358_294,
        -2.296_395_5,
        -2.416_090_5,
        -2.107_669,
        -2.428_420_8,
        -2.526_469,
        -2.319_918_6,
    ]]);

    output
        .to_data()
        .assert_approx_eq(&expected.to_data(), precision);
}

#[test]
fn json_full_record() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/full/complex_nested");

    let record = PrettyJsonFileRecorder::<FullPrecisionSettings>::default()
        .load(file_path)
        .expect("Failed to decode state");

    model_test(record, 8);
}

#[test]
fn json_half_record() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/half/complex_nested");

    let record = PrettyJsonFileRecorder::<HalfPrecisionSettings>::default()
        .load(file_path)
        .expect("Failed to decode state");

    model_test(record, 4);
}

#[test]
fn mpk_full_record() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/full/complex_nested");

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(file_path)
        .expect("Failed to decode state");

    model_test(record, 8);
}

#[test]
fn mpk_half_record() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/half/complex_nested");

    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::default()
        .load(file_path)
        .expect("Failed to decode state");

    model_test(record, 4);
}

#[test]
fn mpk_gz_full_record() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/full/complex_nested");

    let record = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default()
        .load(file_path)
        .expect("Failed to decode state");

    model_test(record, 8);
}

#[test]
fn mpk_gz_half_record() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("model/half/complex_nested");

    let record = NamedMpkGzFileRecorder::<HalfPrecisionSettings>::default()
        .load(file_path)
        .expect("Failed to decode state");

    model_test(record, 4);
}
