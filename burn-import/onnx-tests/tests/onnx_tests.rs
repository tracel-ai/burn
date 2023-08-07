pub mod add {
    include!(concat!(env!("OUT_DIR"), "/model/add.rs"));
}

pub mod sub {
    include!(concat!(env!("OUT_DIR"), "/model/sub.rs"));
}

pub mod mul {
    include!(concat!(env!("OUT_DIR"), "/model/mul.rs"));
}

pub mod div {
    include!(concat!(env!("OUT_DIR"), "/model/div.rs"));
}

pub mod concat {
    include!(concat!(env!("OUT_DIR"), "/model/concat.rs"));
}

pub mod conv2d {
    include!(concat!(env!("OUT_DIR"), "/model/conv2d.rs"));
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::tensor::{Data, Shape, Tensor};

    use float_cmp::ApproxEq;

    type Backend = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn add_scalar_to_tensor_and_tensor_to_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: add::Model<Backend> = add::Model::default();

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]]);
        let scalar = 2f64;
        let output = model.forward(input, scalar);
        let expected = Data::from([[[[9., 10., 11., 12.]]]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn sub_scalar_from_tensor_and_tensor_from_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: sub::Model<Backend> = sub::Model::default();

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]]);
        let scalar = 3.0f64;
        let output = model.forward(input, scalar);
        let expected = Data::from([[[[6., 7., 8., 9.]]]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn mul_scalar_with_tensor_and_tensor_with_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: mul::Model<Backend> = mul::Model::default();

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]]);
        let scalar = 6.0f64;
        let output = model.forward(input, scalar);
        let expected = Data::from([[[[126., 252., 378., 504.]]]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn div_tensor_by_scalar_and_tensor_by_tensor() {
        // Initialize the model without weights (because the exported file does not contain them)
        let model: div::Model<Backend> = div::Model::new();

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[3., 6., 6., 9.]]]]);
        let scalar1 = 9.0f64;
        let scalar2 = 3.0f64;
        let output = model.forward(input, scalar1, scalar2);
        let expected = Data::from([[[[1., 2., 2., 3.]]]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn concat_tensors() {
        // Initialize the model
        let model: concat::Model<Backend> = concat::Model::new();

        // Run the model
        let input = Tensor::<Backend, 4>::zeros([1, 2, 3, 5]);

        let output = model.forward(input);

        let expected = Shape::from([1, 18, 3, 5]);

        assert_eq!(output.shape(), expected);
    }

    #[test]
    fn conv2d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv2d::Model<Backend> = conv2d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<Backend, 4>::ones([2, 4, 10, 15]);

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 6, 15]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv2d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = 24.004_995; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }
}
