use crate::Shape;
use rand::{distributions::Standard, prelude::Distribution};

#[derive(new, Debug, Clone, PartialEq)]
pub struct Data<P, const D: usize> {
    pub value: Vec<P>,
    pub shape: Shape<D>,
}

impl<P: std::fmt::Debug, const D: usize> Data<P, D>
where
    Standard: Distribution<P>,
{
    pub fn random(shape: Shape<D>) -> Data<P, D> {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(rand::random());
        }

        Data::new(data, shape)
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize> From<[P; A]> for Data<P, 1> {
    fn from(elems: [P; A]) -> Self {
        let mut data = Vec::with_capacity(2 * A);
        for i in 0..A {
            data.push(elems[i]);
        }

        Data::new(data, Shape::new([A]))
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize, const B: usize> From<[[P; B]; A]> for Data<P, 2> {
    fn from(elems: [[P; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B);
        for i in 0..A {
            for j in 0..B {
                data.push(elems[i][j]);
            }
        }

        Data::new(data, Shape::new([A, B]))
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize, const B: usize, const C: usize>
    From<[[[P; C]; B]; A]> for Data<P, 3>
{
    fn from(elems: [[[P; C]; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B * C);
        for i in 0..A {
            for j in 0..B {
                for k in 0..C {
                    data.push(elems[i][j][k]);
                }
            }
        }

        Data::new(data, Shape::new([A, B, C]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_have_right_num_elements() {
        let shape = Shape::new([3, 5, 6]);
        let data = Data::<f32, 3>::random(shape.clone());
        assert_eq!(shape.num_elements(), data.value.len());
    }

    #[test]
    fn should_have_right_shape() {
        let data = Data::from([[3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, Shape::new([1, 3]));

        let data = Data::from([[4.0, 5.0, 8.0], [3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, Shape::new([2, 3]));

        let data = Data::from([3.0, 5.0, 6.0]);
        assert_eq!(data.shape, Shape::new([3]));
    }
}
