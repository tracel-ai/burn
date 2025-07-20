// Import the shared macro
use crate::include_models;
include_models!(slice, slice_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn slice() {
        let model: slice::Model<Backend> = slice::Model::default();
        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats(
            [
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                [11., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
                [21., 22., 23., 24., 25., 26., 27., 28., 29., 30.],
                [31., 32., 33., 34., 35., 36., 37., 38., 39., 40.],
                [41., 42., 43., 44., 45., 46., 47., 48., 49., 50.],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [1f32, 2., 3., 4., 5.],
            [11f32, 12., 13., 14., 15.],
            [21., 22., 23., 24., 25.],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn slice_shape() {
        let model: slice_shape::Model<Backend> = slice_shape::Model::default();
        let device = Default::default();

        let input = Tensor::<Backend, 4>::zeros([1, 2, 3, 1], &device);

        // Slice Start == 1, End == 3
        let output = model.forward(input);

        assert_eq!(output, [2, 3]);
    }
}
