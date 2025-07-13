// Import the shared macro
use crate::include_models;
include_models!(constant_of_shape, constant_of_shape_full_like);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, Tolerance, ops::FloatElem};

    use crate::backend::Backend;
    type FT = FloatElem<Backend>;

    #[test]
    fn constant_of_shape() {
        // This tests shape is being passed directly to the model
        let device = Default::default();
        let model = constant_of_shape::Model::<Backend>::new(&device);
        let input_shape = [2, 3, 2];
        let expected = Tensor::<Backend, 3>::full(input_shape, 1.125, &device).to_data();

        let output = model.forward(input_shape);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn constant_of_shape_full_like() {
        // This tests shape is being passed from the input tensor
        let device = Default::default();
        let model = constant_of_shape_full_like::Model::<Backend>::new(&device);
        let shape = [2, 3, 2];
        let f_expected = Tensor::<Backend, 3>::full(shape, 3.0, &device);
        let i_expected = Tensor::<Backend, 3, Int>::full(shape, 5, &device);
        let b_expected = Tensor::<Backend, 3, Int>::ones(shape, &device).bool();

        let input = Tensor::ones(shape, &device);
        let (f_output, i_output, b_output) = model.forward(input);

        assert!(f_output.equal(f_expected).all().into_scalar());
        assert!(i_output.equal(i_expected).all().into_scalar());
        assert!(b_output.equal(b_expected).all().into_scalar());
    }
}
