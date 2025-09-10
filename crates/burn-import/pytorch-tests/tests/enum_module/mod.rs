use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Conv<B: Backend> {
    DwsConv(DwsConv<B>),
    Conv(Conv2d<B>),
}

#[derive(Module, Debug)]
pub struct DwsConv<B: Backend> {
    dconv: Conv2d<B>,
    pconv: Conv2d<B>,
}

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv: Conv<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        let device = Default::default();

        let conv = match record.conv {
            ConvRecord::DwsConv(dws_conv) => {
                let dconv = Conv2dConfig::new([2, 2], [3, 3])
                    .with_groups(2)
                    .init(&device)
                    .load_record(dws_conv.dconv);
                let pconv = Conv2dConfig::new([2, 2], [1, 1])
                    .with_groups(1)
                    .init(&device)
                    .load_record(dws_conv.pconv);
                Conv::DwsConv(DwsConv { dconv, pconv })
            }
            ConvRecord::Conv(conv) => {
                let conv2d_config = Conv2dConfig::new([2, 2], [3, 3]);
                Conv::Conv(conv2d_config.init(&device).load_record(conv))
            }
        };
        Net { conv }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.conv {
            Conv::DwsConv(dws_conv) => {
                let x = dws_conv.dconv.forward(x);
                dws_conv.pconv.forward(x)
            }
            Conv::Conv(conv) => conv.forward(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::{
        record::{FullPrecisionSettings, Recorder},
        tensor::{Tolerance, ops::FloatElem},
    };
    use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
    type FT = FloatElem<TestBackend>;

    use super::*;

    #[test]
    fn depthwise_false() {
        let device = Default::default();
        let load_args =
            LoadArgs::new("tests/enum_module/enum_depthwise_false.pt".into()).with_debug_print();

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = Net::<TestBackend>::new_with(record);
        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.713_979_7, 0.267_644_3, 0.990_609, 0.28845078, 0.874_962_4],
                    [0.505_920_8, 0.23659128, 0.757_007_4, 0.23458993, 0.64705235],
                    [0.355_621_4, 0.445_182_8, 0.01930594, 0.26160914, 0.771_317],
                    [0.37846136, 0.99802476, 0.900_794_2, 0.476_588_2, 0.16625845],
                    [
                        0.804_481_1,
                        0.65517855,
                        0.17679012,
                        0.824_772_3,
                        0.803_550_9,
                    ],
                ],
                [
                    [0.943_447_5, 0.21972018, 0.417_697, 0.49031407, 0.57302874],
                    [0.12054086, 0.14518881, 0.772_002_3, 0.38275403, 0.744_236_7],
                    [0.52850497, 0.664_172_4, 0.60994434, 0.681_799_7, 0.74785537],
                    [
                        0.03694397,
                        0.751_675_7,
                        0.148_438_4,
                        0.12274551,
                        0.530_407_2,
                    ],
                    [0.414_796_4, 0.793_662, 0.21043217, 0.05550903, 0.863_884_4],
                ],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.35449377, -0.02832414, 0.490_976_1],
                    [0.29709217, 0.332_586_3, 0.30594018],
                    [0.18101373, 0.30932188, 0.30558896],
                ],
                [
                    [-0.17683622, -0.13244139, -0.05608707],
                    [0.23467252, -0.07038684, 0.255_044_1],
                    [-0.241_931_3, -0.20476191, -0.14468731],
                ],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn depthwise_true() {
        let device = Default::default();
        let load_args =
            LoadArgs::new("tests/enum_module/enum_depthwise_true.pt".into()).with_debug_print();

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = Net::<TestBackend>::new_with(record);

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.713_979_7, 0.267_644_3, 0.990_609, 0.28845078, 0.874_962_4],
                    [0.505_920_8, 0.23659128, 0.757_007_4, 0.23458993, 0.64705235],
                    [0.355_621_4, 0.445_182_8, 0.01930594, 0.26160914, 0.771_317],
                    [0.37846136, 0.99802476, 0.900_794_2, 0.476_588_2, 0.16625845],
                    [
                        0.804_481_1,
                        0.65517855,
                        0.17679012,
                        0.824_772_3,
                        0.803_550_9,
                    ],
                ],
                [
                    [0.943_447_5, 0.21972018, 0.417_697, 0.49031407, 0.57302874],
                    [0.12054086, 0.14518881, 0.772_002_3, 0.38275403, 0.744_236_7],
                    [0.52850497, 0.664_172_4, 0.60994434, 0.681_799_7, 0.74785537],
                    [
                        0.03694397,
                        0.751_675_7,
                        0.148_438_4,
                        0.12274551,
                        0.530_407_2,
                    ],
                    [0.414_796_4, 0.793_662, 0.21043217, 0.05550903, 0.863_884_4],
                ],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.77874625, 0.859_017_6, 0.834_283_5],
                    [0.773_056_4, 0.73817325, 0.78292674],
                    [0.710_775_2, 0.747_187_2, 0.733_264_4],
                ],
                [
                    [-0.44891885, -0.49027523, -0.394_170_7],
                    [-0.43836114, -0.33961445, -0.387_311_5],
                    [-0.581_134_3, -0.34197026, -0.535_035_7],
                ],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
