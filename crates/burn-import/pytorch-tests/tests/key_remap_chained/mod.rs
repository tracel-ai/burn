use std::marker::PhantomData;

use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Tensor, backend::Backend},
};

/// Some module that implements a specific method so it can be used in a sequential block.
pub trait ForwardModule<B: Backend> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4>;
}

/// Conv2d + BatchNorm block.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> ForwardModule<B> for ConvBlock<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        self.bn.forward(out)
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: &Device<B>) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_bias(false)
            .init(device);
        let bn = BatchNormConfig::new(out_channels).init(device);

        Self { conv, bn }
    }
}

/// Collection of sequential blocks.
#[derive(Module, Debug)]
pub struct ModuleBlock<B: Backend, M> {
    blocks: Vec<M>,
    _backend: PhantomData<B>,
}

impl<B: Backend, M: ForwardModule<B>> ModuleBlock<B, M> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = input;
        for block in &self.blocks {
            out = block.forward(out);
        }
        out
    }
}

impl<B: Backend> ModuleBlock<B, ConvBlock<B>> {
    pub fn new(device: &Device<B>) -> Self {
        let blocks = vec![ConvBlock::new(6, 6, device), ConvBlock::new(6, 6, device)];

        Self {
            blocks,
            _backend: PhantomData,
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend, M> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    layer: ModuleBlock<B, M>,
}

impl<B: Backend> Model<B, ConvBlock<B>> {
    pub fn new(device: &Device<B>) -> Self {
        let conv = Conv2dConfig::new([3, 6], [3, 3])
            .with_bias(false)
            .init(device);
        let bn = BatchNormConfig::new(6).init(device);

        let layer = ModuleBlock::new(device);

        Self { conv, bn, layer }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        let out = self.bn.forward(out);
        self.layer.forward(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    use super::*;

    #[test]
    #[should_panic]
    fn key_remap_chained_missing_pattern() {
        // Loading record should fail due to missing pattern to map the layer.blocks
        let device = Default::default();
        let load_args = LoadArgs::new("tests/key_remap_chained/key_remap.pt".into())
            // Map *.block.0.* -> *.conv.*
            .with_key_remap("(.+)\\.block\\.0\\.(.+)", "$1.conv.$2")
            // Map *.block.1.* -> *.bn.*
            .with_key_remap("(.+)\\.block\\.1\\.(.+)", "$1.bn.$2");

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model: Model<TestBackend, _> = Model::new(&device);

        model.load_record(record);
    }

    #[test]
    fn key_remap_chained() {
        let device = Default::default();
        let load_args = LoadArgs::new("tests/key_remap_chained/key_remap.pt".into())
            // Map *.block.0.* -> *.conv.*
            .with_key_remap("(.+)\\.block\\.0\\.(.+)", "$1.conv.$2")
            // Map *.block.1.* -> *.bn.*
            .with_key_remap("(.+)\\.block\\.1\\.(.+)", "$1.bn.$2")
            // Map layer.[i].* -> layer.blocks.[i].*
            .with_key_remap("layer\\.([0-9])\\.(.+)", "layer.blocks.$1.$2");

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model: Model<TestBackend, _> = Model::new(&device);

        let model = model.load_record(record);

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.76193494, 0.626_546_1, 0.49510366, 0.11974698],
                    [0.07161391, 0.03232569, 0.704_681, 0.254_516],
                    [0.399_373_7, 0.21224737, 0.40888822, 0.14808255],
                    [0.17329216, 0.665_855_4, 0.351_401_8, 0.808_671_6],
                ],
                [
                    [0.33959562, 0.13321638, 0.41178054, 0.257_626_3],
                    [0.347_029_2, 0.02400219, 0.77974546, 0.15189773],
                    [0.75130886, 0.726_892_1, 0.85721636, 0.11647397],
                    [0.859_598_4, 0.263_624_2, 0.685_534_6, 0.96955734],
                ],
                [
                    [0.42948407, 0.49613327, 0.38488472, 0.08250773],
                    [0.73995143, 0.00364107, 0.81039995, 0.87411255],
                    [0.972_853_2, 0.38206023, 0.08917904, 0.61241513],
                    [0.77621365, 0.00234562, 0.38650817, 0.20027226],
                ],
            ]],
            &device,
        );
        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [[0.198_967_1, 0.17847246], [0.06883702, 0.20012866]],
                [[0.17582723, 0.11344293], [0.05444185, 0.13307181]],
                [[0.192_229_5, 0.20391327], [0.06150475, 0.22688155]],
                [[0.00230906, -0.02177845], [0.01129148, 0.00925517]],
                [[0.14751078, 0.14433631], [0.05498439, 0.29049855]],
                [[0.16868964, 0.133_269_3], [0.06917118, 0.35094324]],
            ]],
            &device,
        );

        let output = model.forward(input);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
