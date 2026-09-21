use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Device, Tensor, activation::relu},
};

/// Mirrors the PyTorch model from issue #4716: `flows` is a list whose odd entries are
/// parameter-free (a Flip), so its indices 0, 2, 4 are real on the Burn side too and the
/// list is longer than its highest index in the file. `fc` is a Sequential with a ReLU
/// at 1, so its 0, 2 must still collapse to 0, 1.
#[derive(Module, Debug)]
pub struct Net {
    flows: Vec<Option<Conv1d>>,
    fc: Vec<Linear>,
}

impl Net {
    /// Create a new model with placeholder values.
    pub fn init(device: &Device) -> Self {
        let conv = Conv1dConfig::new(2, 2, 1);
        let flows = vec![
            Some(conv.init(device)),
            None,
            Some(conv.init(device)),
            None,
            Some(conv.init(device)),
            None,
        ];
        let linear = LinearConfig::new(2, 2);
        let fc = vec![linear.init(device), linear.init(device)];
        Net { flows, fc }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.flows.iter().fold(x, |x, flow| match flow {
            Some(conv) => conv.forward(x),
            None => x.flip([1]),
        });
        let x = x.swap_dims(1, 2);
        let x = relu(self.fc[0].forward(x));
        let x = self.fc[1].forward(x);
        x.swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {

    use burn::tensor::Tolerance;
    use burn_store::{ModuleSnapshot, PytorchStore};
    type FT = f32;

    use super::*;

    const PATH: &str = "tests/keep_indices/keep_indices.pt";

    #[test]
    fn keep_indices_for_flows() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file(PATH).map_indices_contiguous_except(r"^flows$");

        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        let input = Tensor::<3>::from_data(
            [[
                [0.04579777, 0.17550886, 0.61767447],
                [0.82907301, 0.52463114, 0.27080512],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<3>::from_data(
            [[
                [-0.27289051, -0.27018493, -0.26568484],
                [-0.66148949, -0.66561544, -0.67231065],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::absolute(1e-7));
    }

    #[test]
    fn default_mapping_renumbers_flows() {
        // Without the exception, flows.{2,4} become flows.{1,2} and no longer line up
        // with the Burn side, which is the failure reported in issue #4716.
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file(PATH);

        let err = model
            .load_from(&mut store)
            .expect_err("flows.2 and flows.4 should be missing");
        // flows.2 gets flows.4's weights, flows.4 is missing and the renumbered
        // flows.1 goes unused.
        let msg = err.to_string();
        assert!(msg.contains("flows.4.weight"), "{msg}");
        assert!(msg.contains("flows.1.weight"), "{msg}");
    }
}
