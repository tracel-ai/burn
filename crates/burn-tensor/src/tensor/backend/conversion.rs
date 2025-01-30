use crate::tensor::Shape;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Convertor {
    indices_from: HashMap<usize, Vec<usize>>,
    indices_to: HashMap<Vec<usize>, usize>,
}

impl Convertor {
    pub fn new<const D: usize>(shape: &Shape<D>, order_from: Order, order_to: Order) -> Self {
        Self {
            indices_from: revert_indices_map(build_indices(shape, order_from)),
            indices_to: build_indices(shape, order_to),
        }
    }

    pub fn convert<P: Copy>(&self, data: &Vec<P>) -> Vec<P> {
        let mut data_converted = data.clone();

        for (i, val) in data.iter().enumerate() {
            let indices = self.indices_from.get(&i).unwrap();
            let i_converted = self.indices_to.get(indices).unwrap();
            data_converted[*i_converted] = *val;
        }
        data_converted
    }
}

pub enum Order {
    Left,
    Right,
}
pub fn build_indices<const D: usize>(shape: &Shape<D>, order: Order) -> HashMap<Vec<usize>, usize> {
    let num_elements = shape.num_elements();
    let mut indices = init_indices::<D>();
    let mut num_repeat_next = 1;

    for i in 0..D {
        let index = match order {
            Order::Left => D - i - 1,
            Order::Right => i,
        };
        let size = shape.dims[index];
        let num_repeat = num_repeat_next;
        let times = num_elements / (num_repeat * size);
        num_repeat_next *= size;

        let dim = IndicesDimGenerator::new(size, num_repeat, times);
        indices[index] = dim.generate();
    }

    build_map_from(shape, indices)
}

fn init_indices<const D: usize>() -> Vec<Vec<usize>> {
    let mut indices: Vec<Vec<usize>> = Vec::with_capacity(D);
    for _ in 0..D {
        indices.push(Vec::new());
    }
    indices
}

fn build_map_from<const D: usize>(
    shape: &Shape<D>,
    indices: Vec<Vec<usize>>,
) -> HashMap<Vec<usize>, usize> {
    let num_elements = shape.num_elements();
    let mut map = HashMap::new();

    for e in 0..num_elements {
        let mut index = Vec::with_capacity(D);
        for d in 0..D {
            let arr = &indices[d];
            let num = arr[e];
            index.push(num);
        }
        map.insert(index, e);
    }
    map
}

fn revert_indices_map(map: HashMap<Vec<usize>, usize>) -> HashMap<usize, Vec<usize>> {
    let mut map_revert = HashMap::with_capacity(map.len());
    for (key, value) in map.into_iter() {
        map_revert.insert(value, key);
    }
    map_revert
}

#[derive(new, Debug)]
struct IndicesDimGenerator {
    size: usize,
    num_repeat: usize,
    times: usize,
}

impl IndicesDimGenerator {
    fn generate(self) -> Vec<usize> {
        let mut vec = Vec::new();

        for _ in 0..self.times {
            for i in 0..self.size {
                for _ in 0..self.num_repeat {
                    vec.push(i);
                }
            }
        }
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_convert_data_2d_from_left_to_right() {
        let shape = Shape::new([2, 3]);
        let data_ij = data_ij(&shape);
        let data_ji = data_ji(&shape);
        let convertor = Convertor::new(&shape, Order::Left, Order::Right);

        let data_ij_converted = convertor.convert(&data_ij);

        assert_eq!(data_ji, data_ij_converted);
    }

    #[test]
    fn should_convert_data_2d_from_right_to_left() {
        let shape = Shape::new([2, 3]);
        let data_ij = data_ij(&shape);
        let data_ji = data_ji(&shape);
        let convertor = Convertor::new(&shape, Order::Right, Order::Left);

        let data_ji_converted = convertor.convert(&data_ji);

        assert_eq!(data_ij, data_ji_converted);
    }

    fn data_ij(shape: &Shape<2>) -> Vec<usize> {
        let mut data = Vec::new();

        for i in 0..shape.dims[0] {
            for j in 0..shape.dims[1] {
                data.push(i + j);
            }
        }
        data
    }

    fn data_ji(shape: &Shape<2>) -> Vec<usize> {
        let mut data = Vec::new();

        for i in 0..shape.dims[1] {
            for j in 0..shape.dims[0] {
                data.push(i + j);
            }
        }
        data
    }

    #[test]
    fn should_build_indices_1d_simple() {
        let shape = Shape::new([2]);

        let indices = build_indices(&shape, Order::Left);

        let expected = HashMap::from([(vec![0], 0), (vec![1], 1)]);

        assert_eq!(expected, indices);
    }

    #[test]
    fn should_build_indices_2d_simple() {
        let shape = Shape::new([2, 2]);

        let indices = build_indices(&shape, Order::Left);

        let expected = HashMap::from([
            (vec![0, 0], 0),
            (vec![0, 1], 1),
            (vec![1, 0], 2),
            (vec![1, 1], 3),
        ]);

        assert_eq!(expected, indices);
    }

    #[test]
    fn should_build_indices_2d_complex() {
        let shape = Shape::new([2, 3]);

        let indices = build_indices(&shape, Order::Left);

        let expected = HashMap::from([
            (vec![0, 0], 0),
            (vec![0, 1], 1),
            (vec![0, 2], 2),
            (vec![1, 0], 3),
            (vec![1, 1], 4),
            (vec![1, 2], 5),
        ]);

        assert_eq!(expected, indices);
    }

    #[test]
    fn should_build_indices_3d_complex() {
        let shape = Shape::new([2, 5, 3]);

        let indices = build_indices(&shape, Order::Left);

        let expected = HashMap::from([
            (vec![0, 0, 0], 0),
            (vec![0, 0, 1], 1),
            (vec![0, 0, 2], 2),
            (vec![0, 1, 0], 3),
            (vec![0, 1, 1], 4),
            (vec![0, 1, 2], 5),
            (vec![0, 2, 0], 6),
            (vec![0, 2, 1], 7),
            (vec![0, 2, 2], 8),
            (vec![0, 3, 0], 9),
            (vec![0, 3, 1], 10),
            (vec![0, 3, 2], 11),
            (vec![0, 4, 0], 12),
            (vec![0, 4, 1], 13),
            (vec![0, 4, 2], 14),
            (vec![1, 0, 0], 15),
            (vec![1, 0, 1], 16),
            (vec![1, 0, 2], 17),
            (vec![1, 1, 0], 18),
            (vec![1, 1, 1], 19),
            (vec![1, 1, 2], 20),
            (vec![1, 2, 0], 21),
            (vec![1, 2, 1], 22),
            (vec![1, 2, 2], 23),
            (vec![1, 3, 0], 24),
            (vec![1, 3, 1], 25),
            (vec![1, 3, 2], 26),
            (vec![1, 4, 0], 27),
            (vec![1, 4, 1], 28),
            (vec![1, 4, 2], 29),
        ]);

        assert_eq!(expected, indices);
    }

    #[test]
    fn should_build_indices_4d_weird() {
        let shape = Shape::new([2, 1, 2, 1]);

        let indices = build_indices(&shape, Order::Left);

        let expected = HashMap::from([
            (vec![0, 0, 0, 0], 0),
            (vec![0, 0, 1, 0], 1),
            (vec![1, 0, 0, 0], 2),
            (vec![1, 0, 1, 0], 3),
        ]);

        assert_eq!(expected, indices);
    }
}
