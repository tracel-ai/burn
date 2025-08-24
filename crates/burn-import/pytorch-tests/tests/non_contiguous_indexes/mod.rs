use burn::{
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Tensor, activation::relu, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    fc: Vec<Conv2d<B>>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        let device = Default::default();
        let conv2d_config = Conv2dConfig::new([2, 2], [3, 3]).with_padding(PaddingConfig2d::Same);
        let mut fc = vec![];
        for fc_record in record.fc.into_iter() {
            fc.push(conv2d_config.init(&device).load_record(fc_record));
        }
        Net { fc }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.fc.iter().fold(x, |x_i, conv| relu(conv.forward(x_i)))
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::record::{FullPrecisionSettings, Recorder};
    use burn::tensor::{Tolerance, ops::FloatElem};
    use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
    type FT = FloatElem<TestBackend>;

    use super::*;

    #[test]
    fn key_remap() {
        let device = Default::default();
        let load_args =
            LoadArgs::new("tests/non_contiguous_indexes/non_contiguous_indexes.pt".into())
                .with_debug_print();

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = Net::<TestBackend>::new_with(record);

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [
                        0.67890584,
                        0.307_537_2,
                        0.265_156_2,
                        0.528_318_8,
                        0.86194897,
                    ],
                    [0.14828813, 0.73480314, 0.821_220_7, 0.989_098_6, 0.15003455],
                    [0.62109494, 0.13028657, 0.926_875_1, 0.30604684, 0.80117637],
                    [0.514_885_7, 0.46105868, 0.484_046_1, 0.58499724, 0.73569804],
                    [0.58018994, 0.65252745, 0.05023766, 0.864_268_7, 0.935_932],
                ],
                [
                    [0.913_302_9, 0.869_611_3, 0.139_184_3, 0.314_65, 0.94086266],
                    [0.11917073, 0.953_610_6, 0.10675198, 0.14779574, 0.744_439],
                    [0.14075547, 0.38544965, 0.863_745_9, 0.89604443, 0.97287786],
                    [0.39854127, 0.11136961, 0.99230546, 0.39348692, 0.29428244],
                    [0.621_886_9, 0.15033776, 0.828_640_1, 0.81336635, 0.10325938],
                ],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.04485746, 0.03582812, 0.03432692, 0.02892298, 0.013_844_3],
                ],
                [
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                ],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::absolute(1e-7));
    }
}
