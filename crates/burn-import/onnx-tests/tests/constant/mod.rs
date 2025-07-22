// Import the shared macro
use crate::include_models;
include_models!(constant_f32, constant_f64, constant_i32, constant_i64);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Shape, Tensor};

    use crate::backend::Backend;

    #[test]
    fn add_constant_f32() {
        let device = Default::default();
        let model = constant_f32::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_f64() {
        let device = Default::default();
        let model = constant_f64::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_i32() {
        let device = Default::default();
        let model = constant_i32::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3, Int>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3, Int>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_i64() {
        let device = Default::default();
        let model = constant_i64::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3, Int>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3, Int>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }
}
