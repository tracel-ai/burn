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

#[cfg(test)]
mod tests {
    use super::*;

    use burn::tensor::{Data, Shape, Tensor};

    type Backend = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn add() {
        // Tests the add node with matrix additions and scalar additions

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
    fn sub() {
        // Tests the sub node with matrix subtractions and scalar subtractions

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
    fn mul() {
        // Tests the mul node with matrix multiplications and scalar multiplications

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
    fn div() {
        // Tests the div node with matrix divisions and scalar divisions

        // Initialize the model
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
    fn concat() {
        // Tests the concat node with matrix concatenations

        // Initialize the model
        let model: concat::Model<Backend> = concat::Model::new();

        // Run the model
        let input = Tensor::<Backend, 4>::zeros([1, 2, 3, 5]);

        let output = model.forward(input);

        let expected = Shape::from([1, 18, 3, 5]);

        assert_eq!(output.shape(), expected);
    }
}
