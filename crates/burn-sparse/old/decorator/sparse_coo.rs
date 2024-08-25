use crate::backend::SparseBackend;
use crate::backend::SparseTensor;
use crate::decorator::SparseCOO;
use crate::decorator::SparseDecorator;
use burn_tensor::ops::FloatElem;
use burn_tensor::ops::FloatTensor;
use burn_tensor::ops::FloatTensorOps;

use burn_tensor::Device;
use burn_tensor::{
    backend::Backend, Bool, ElementConversion, Float, Int, Shape, Tensor, TensorData,
    TensorPrimitive,
};

#[derive(Clone, Debug)]
pub struct SparseCOOTensor<B: Backend, const D: usize> {
    pub coordinates: Option<Tensor<B, 2, Int>>,
    pub values: Option<Tensor<B, 1, Float>>,
    pub shape: Shape<D>,
    pub device: Device<B>,
}

fn flatten_coordinates<B: Backend, const D: usize, const S: usize>(
    coordinates: Tensor<B, 2, Int>,
    shape: Shape<D>,
    device: &Device<B>,
) -> Tensor<B, 2, Int> {
    let mut strides_data = [[1]; D];
    for i in (0..D).rev() {
        if D - 1 - i == S {
            strides_data[i] = [1];
        } else if D - 1 - i < S {
            strides_data[i] = [0];
        } else {
            strides_data[i] = [strides_data[i + 1][0] * shape.dims[i + 1] as i64];
        }
    }
    let strides_data: TensorData = TensorData::from(strides_data);
    let strides: Tensor<B, 2, Int> = Tensor::from_data(strides_data, device);
    let flat_coordinates: Tensor<B, 1, Int> = strides.mul(coordinates).sum_dim(0).flatten(0, 1);

    flat_coordinates.unsqueeze_dim(0)
}

fn unflatten_coordinates<B: Backend, const D: usize>(
    flat_coordinates: Tensor<B, 2, Int>,
    new_shape: Shape<D>,
) -> Tensor<B, 2, Int> {
    let flat_coordinates = flat_coordinates.squeeze::<1>(0);
    let mut remaining_flat_coordinates = flat_coordinates.clone();
    let mut new_coordinates = Vec::with_capacity(D);

    for &dim_size in new_shape.dims.iter().rev() {
        let size = dim_size as i64;
        let new_coord = remaining_flat_coordinates.clone().remainder_scalar(size);
        new_coordinates.push(new_coord.clone());
        remaining_flat_coordinates = remaining_flat_coordinates.div_scalar(size);
    }

    new_coordinates.reverse();

    Tensor::stack(new_coordinates, 0)
}

impl<B> SparseBackend for SparseDecorator<B, SparseCOO>
where
    B: Backend,
{
    type SparseTensorPrimitive<const D: usize> = SparseCOOTensor<B, D>;

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        let dense: Tensor<B, D> = Tensor::from_primitive(TensorPrimitive::Float(dense));

        let shape = dense.shape();
        let device = dense.device();

        let significant = dense.clone().not_equal_elem(0.0);
        if !significant.clone().any().into_scalar() {
            return Self::sparse_empty(dense.shape(), &device);
        };

        let coordinates = significant
            .clone()
            .nonzero()
            .into_iter()
            .map(|tensor| {
                let length = tensor.shape().dims[0];
                let shape = Shape::new([1, length]);
                tensor.reshape(shape)
            })
            .collect();

        let coordinates = Tensor::cat(coordinates, 0);

        let dense = dense.flatten(0, D - 1);

        let dims = significant.dims();
        let values = dense.gather(
            0,
            significant
                .flatten::<1>(0, dims.len() - 1)
                .nonzero()
                .remove(0),
        );

        let coordinates = Some(coordinates);
        let values = Some(values);

        Self::SparseTensorPrimitive {
            coordinates,
            values,
            shape,
            device,
        }
    }

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = sparse;

        let (Some(coordinates), Some(values)) = (coordinates, values) else {
            return Tensor::<B, D>::zeros(shape, &device)
                .into_primitive()
                .tensor();
        };

        let dense: Tensor<B, 1, Float> = Tensor::zeros(Shape::new([shape.num_elements()]), &device);
        let flat_coordinates =
            flatten_coordinates::<B, D, 0>(coordinates, shape.clone(), &device).squeeze(0);
        let dense = dense.select_assign(0, flat_coordinates, values);

        dense.reshape(shape).into_primitive().tensor()
    }

    fn sparse_sddmm<const D: usize>(
        lhs: Self::FloatTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        if sparse.coordinates.is_none() || sparse.values.is_none() {
            return sparse;
        }

        // Flatten the lhs and rhs into a tensor of rows and cols respectively
        let lhs = Tensor::<B, D>::new(burn_tensor::TensorPrimitive::Float(lhs));
        let rhs = Tensor::<B, D>::new(burn_tensor::TensorPrimitive::Float(rhs)).transpose();
        let lhs_dims = lhs.shape().dims;
        let rhs_dims = rhs.shape().dims;

        if lhs_dims[D - 1] != rhs_dims[D - 1]
            || lhs_dims[D - 2] != sparse.shape.dims[D - 2]
            || rhs_dims[D - 2] != sparse.shape.dims[D - 1]
        {
            panic!("invalid dimensions for sddmm. lhs and rhs must have compatible shapes for matmul, and sparse must have the correct shape for output of matmul between lhs and rhs.");
        }

        let lhs = lhs.reshape([-1, lhs_dims[D - 1] as i32]);
        let rhs = rhs.reshape([-1, rhs_dims[D - 1] as i32]);

        // Flatten the sparse tensor into
        let device = sparse.device.clone();
        let mut shape = sparse.shape.clone();
        let lhs_coordinates = sparse
            .coordinates
            .clone()
            .expect("Expected non-empty sparse tensor");

        // swap the last two dims so its column-first
        let swizzle = Tensor::<B, 1, Int>::arange(0..D as i64, &device)
            .slice_assign(
                [D - 2..D],
                Tensor::<B, 1, Int>::from_ints([D - 1, D - 2], &device),
            )
            .unsqueeze_dim(1)
            .repeat(1, lhs_coordinates.shape().dims[1]);
        let rhs_coordinates = lhs_coordinates.clone().gather(0, swizzle);

        let row_indices = flatten_coordinates::<B, D, 1>(lhs_coordinates, shape.clone(), &device);

        shape.dims.swap(D - 1, D - 2);
        let col_indices = flatten_coordinates::<B, D, 1>(rhs_coordinates, shape.clone(), &device);

        let row_indices = row_indices.transpose().repeat(1, lhs_dims[D - 1]);
        let col_indices = col_indices.transpose().repeat(1, rhs_dims[D - 1]);

        let lhs = lhs.gather(0, row_indices);
        let rhs = rhs.gather(0, col_indices);

        let dotted = lhs.mul(rhs).sum_dim(1).squeeze(1);

        SparseCOOTensor {
            coordinates: sparse.coordinates,
            values: Some(dotted),
            shape: sparse.shape,
            device,
        }
    }

    fn sparse_spmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = lhs;

        let rhs: Tensor<B, D, Float> = Tensor::from_primitive(TensorPrimitive::Float(rhs));
        let rhs_shape = rhs.shape();
        let mut out_shape = shape.clone();
        out_shape.dims[D - 1] = rhs_shape.dims[D - 1];

        let (Some(coordinates), Some(values)) = (coordinates, values) else {
            // All zeros, exit early
            return Tensor::<B, D>::zeros(out_shape, &device)
                .into_primitive()
                .tensor();
        };

        let nnz = coordinates.shape().dims[1];

        // Ensure they are of the correct shape to multiply
        if shape.dims[D - 1] != rhs_shape.dims[D - 2] {
            panic!("Invalid shape for matrix multiplication");
        }

        // Ensure batches are the same
        if D > 2 && rhs_shape.dims[0..D - 2] != shape.dims[0..D - 2] {
            panic!("Batches must be of the same shape");
        }

        // Compute strides for the dense tensor to match the flattened shape
        let mut strides_data = [1; D];
        for i in (0..D - 1).rev() {
            strides_data[i] = strides_data[i + 1] * shape.dims[i + 1] as i32;
        }
        let strides: Tensor<B, 2, Int> =
            Tensor::<B, 1, Int>::from_ints(strides_data, &device).unsqueeze_dim(1);

        let column_index = coordinates.clone().slice([D - 1..D, 0..nnz]);

        // the indices into the flat row vector at which the containing matrix starts
        let matrix_starts: Tensor<B, 2, Int> = if D > 2 {
            coordinates
                .clone()
                .slice([0..D - 2, 0..nnz])
                .mul(strides.clone().slice([0..D - 2]))
                .div_scalar((shape.dims[D - 1]) as i32)
                .sum_dim(0)
        } else {
            Tensor::<B, 2, Int>::zeros(column_index.shape(), &device)
        };

        let row_index = coordinates.slice([D - 2..D - 1, 0..nnz]);

        let gather_index = matrix_starts.clone() + column_index;
        let scatter_index = matrix_starts + row_index;

        let gather_index = gather_index.transpose().repeat(1, rhs_shape.dims[D - 1]);
        let scatter_index = scatter_index.transpose().repeat(1, rhs_shape.dims[D - 1]);
        let values = values.unsqueeze_dim(1).repeat(1, rhs_shape.dims[D - 1]);

        // Flatten the rhs similarly into 2 dimensions
        let rhs: Tensor<B, 2> = rhs.reshape([-1, rhs_shape.dims[D - 1] as i32]);

        // Do the matmul using gather/scatter
        let output: Tensor<B, 2, Float> =
            Tensor::zeros([out_shape.dims[0], rhs.shape().dims[1]], &device);
        let gathered = rhs.gather(0, gather_index);

        let multiplied = gathered.mul(values);

        let scattered = output.scatter(0, scatter_index, multiplied);

        scattered.reshape(out_shape).into_primitive().tensor()
    }

    fn sparse_device<const D: usize>(tensor: &SparseTensor<Self, D>) -> burn_tensor::Device<Self> {
        tensor.device.clone()
    }

    fn sparse_to_device<const D: usize>(
        tensor: SparseTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        SparseCOOTensor {
            coordinates: tensor.coordinates.map(|t| t.to_device(device)),
            values: tensor.values.map(|t| t.to_device(device)),
            shape: tensor.shape,
            device: device.clone(),
        }
    }

    fn sparse_shape<const D: usize>(
        tensor: &Self::SparseTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        tensor.shape.clone()
    }

    fn sparse_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> SparseTensor<Self, D> {
        SparseCOOTensor {
            coordinates: None,
            values: None,
            shape,
            device: device.clone(),
        }
    }

    fn sparse_slice<const D1: usize, const D2: usize>(
        tensor: Self::SparseTensorPrimitive<D1>,
        indices: [core::ops::Range<usize>; D2],
    ) -> SparseTensor<Self, D1> {
        let mut indices = Vec::from(indices);
        indices.extend(tensor.shape.dims[indices.len()..D1].iter().map(|&l| 0..l));
        let indices: [core::ops::Range<usize>; D1] = indices.try_into().expect("D2 must be <= D1");
        let out_shape = Shape::new(indices.clone().map(|r| r.end));

        if tensor.coordinates.is_none() && tensor.values.is_none() {
            return SparseCOOTensor {
                coordinates: None,
                values: None,
                shape: out_shape,
                device: tensor.device,
            };
        }

        let coordinates = tensor
            .coordinates
            .expect("Mismatch between coordinates and values");
        let values = tensor
            .values
            .expect("Mismatch between coordinates and values");
        let device = tensor.device;

        let number_nonzero = coordinates.shape().dims[1];

        let mut mask: Tensor<B, 1, Int> = Tensor::ones(Shape::new([number_nonzero]), &device);

        for (dim, bound) in indices.iter().enumerate() {
            let coords = coordinates.clone().slice([dim..dim + 1, 0..number_nonzero]);
            let coords = coords.reshape(Shape::new([number_nonzero]));

            let mask_lower = coords
                .clone()
                .lower_elem(B::IntElem::from_elem(bound.end))
                .int();

            let mask_upper = coords
                .clone()
                .greater_equal_elem(B::IntElem::from_elem(bound.start))
                .int();

            mask = mask.mul(mask_lower).mul(mask_upper);
        }

        let nonzero = mask.not_equal_elem(B::IntElem::from_elem(0));
        if !nonzero.clone().any().into_scalar() {
            // no existing values were in the slice, so return an empty tensor
            return SparseCOOTensor {
                coordinates: None,
                values: None,
                shape: out_shape,
                device,
            };
        }

        let nonzero = nonzero.nonzero();

        let indices_dim1 = nonzero
            .first()
            .cloned()
            .expect("Expected dimension to exist");

        let coordinates = coordinates.select(1, indices_dim1.clone());
        let values = values.select(0, indices_dim1);

        let coordinates = Some(coordinates);
        let values = Some(values);

        SparseCOOTensor {
            coordinates,
            values,
            shape: out_shape,
            device,
        }
    }

    fn sparse_from_data<const D: usize>(
        data: TensorData,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        let dense = B::float_from_data(data, device);
        Self::sparse_to_sparse(dense)
    }

    fn sparse_into_data<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> impl core::future::Future<Output = TensorData> + Send {
        B::float_into_data(Self::sparse_to_dense(tensor))
    }

    fn sparse_reshape<const D1: usize, const D2: usize>(
        tensor: SparseCOOTensor<B, D1>,
        out_shape: Shape<D2>,
    ) -> SparseCOOTensor<B, D2> {
        if tensor.coordinates.is_none() && tensor.values.is_none() {
            return SparseCOOTensor {
                coordinates: None,
                values: None,
                shape: out_shape,
                device: tensor.device,
            };
        }

        let coordinates = tensor
            .coordinates
            .expect("Mismatch between coordinates and values");
        let values = tensor
            .values
            .expect("Mismatch between coordinates and values");
        let shape = tensor.shape;
        let device = tensor.device;

        // Flatten the coordinates
        let flat_coordinates = flatten_coordinates::<B, D1, 0>(coordinates, shape, &device);

        // Unflatten the coordinates to the new shape
        let new_coordinates = unflatten_coordinates(flat_coordinates, out_shape.clone());

        SparseCOOTensor {
            coordinates: Some(new_coordinates),
            values: Some(values),
            shape: out_shape,
            device,
        }
    }

    fn sparse_transpose<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        let d = tensor.shape.dims.len();
        let mut axes: Vec<usize> = (0..d).collect();
        axes.swap(d - 1, d - 2);
        Self::sparse_permute(tensor, &axes)
    }

    fn sparse_swap_dims<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> SparseTensor<Self, D> {
        let d = tensor.shape.dims.len();
        let mut axes: Vec<usize> = (0..d).collect();
        axes.swap(dim1, dim2);
        Self::sparse_permute(tensor, &axes)
    }

    fn sparse_permute<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D> {
        let SparseCOOTensor {
            coordinates,
            values,
            mut shape,
            device,
        } = tensor;

        for (i, &j) in (0..D).zip(axes).filter(|(i, j)| i < j) {
            shape.dims.swap(i, j);
        }

        let axes = Tensor::from(axes);
        let coordinates = coordinates.map(|coordinates| coordinates.select(0, axes));

        SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        }
    }

    fn sparse_flip<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = tensor;

        let (Some(coordinates), Some(values)) = (coordinates, values) else {
            // All zeros, exit early
            return SparseCOOTensor {
                coordinates: None,
                values: None,
                shape,
                device,
            };
        };

        let nnz = coordinates.shape().dims[1];

        let mut mask = [0; D];
        for &axis in axes {
            mask[axis] = 1;
        }
        let mask: Tensor<B, 2, Bool> = Tensor::<_, 1, _>::from_ints(mask, &device)
            .unsqueeze_dim(1)
            .repeat(1, nnz)
            .bool();

        let flipped: Tensor<B, 2, Int> = Tensor::<_, 1, _>::from_ints(shape.dims, &device)
            .unsqueeze_dim(1)
            .repeat(1, nnz)
            .sub(coordinates.clone())
            .sub_scalar(1);

        let coordinates = coordinates.mask_where(mask, flipped);

        let coordinates = Some(coordinates);
        let values = Some(values);

        SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        }
    }

    fn sparse_slice_assign<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        ranges: [core::ops::Range<usize>; D2],
        mut value: SparseTensor<Self, D1>,
    ) -> SparseTensor<Self, D1> {
        let value_nnz = value
            .coordinates
            .as_ref()
            .map(|coords| coords.shape().dims[1])
            .unwrap_or(0);

        let mut ranges = Vec::from(ranges);
        ranges.extend(tensor.shape.dims[ranges.len()..D1].iter().map(|&l| 0..l));
        let ranges: [core::ops::Range<usize>; D1] = ranges.try_into().expect("D2 must be <= D1");

        let shape = tensor.shape.clone();
        let sliced = Self::sparse_reshape(
            Self::sparse_slice(tensor.clone(), ranges.clone()),
            shape.clone(),
        );
        let tensor = Self::sparse_sub(tensor, sliced);
        let offset = Tensor::<B, 1, Int>::from_ints(ranges.map(|r| r.start), &tensor.device);
        let offset = offset.unsqueeze_dim::<2>(1).repeat(1, value_nnz);

        value.shape = shape;
        value.coordinates = value.coordinates.map(|coords| coords + offset);

        Self::sparse_add(tensor, value)
    }

    fn sparse_repeat<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> SparseTensor<Self, D> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = tensor;

        let mut out_shape = shape.clone();
        out_shape.dims[dim] *= times;

        let (Some(coordinates), Some(values)) = (coordinates, values) else {
            // All zeros, exit early
            return SparseCOOTensor {
                coordinates: None,
                values: None,
                shape,
                device,
            };
        };

        let device = coordinates.device();
        let nnz = coordinates.shape().dims[1];

        let values = values.repeat(0, times);

        let coordinates_mask: Tensor<B, 2, Int> = Tensor::zeros(coordinates.shape(), &device);
        let ones: Tensor<B, 2, Int> = Tensor::ones(Shape::new([1, nnz]), &device);
        let coordinates_mask = coordinates_mask.slice_assign([dim..dim + 1, 0..nnz], ones);
        let coordinates = Tensor::cat(
            (0..times)
                .map(|n| {
                    coordinates.clone()
                        + coordinates_mask.clone() * (n as i32) * (shape.dims[dim] as i32)
                })
                .collect::<Vec<_>>(),
            1,
        );

        let coordinates = Some(coordinates);
        let values = Some(values);

        SparseCOOTensor {
            coordinates,
            values,
            shape: out_shape,
            device,
        }
    }

    fn sparse_cat<const D: usize>(
        _tensors: Vec<SparseTensor<Self, D>>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        let _offset = 0;
        todo!()
    }

    fn sparse_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("elementwise equal is unsupported for SparseCOO until scatter supports other reduction methods");
    }

    fn sparse_not_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("elementwise not_equal is unsupported for SparseCOO until scatter supports other reduction methods");
    }

    fn sparse_any<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        let SparseCOOTensor {
            coordinates,
            values: _,
            shape: _,
            device: _,
        } = tensor;
        let any = coordinates.is_some();
        Tensor::<B, 1, Bool>::from([any]).into_primitive()
    }

    fn sparse_any_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("any_dim is unsupported for the SparseCOO Decorator due to performance issues, convert to dense explicitly to ensure you understand");
    }

    fn sparse_all<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        let SparseCOOTensor {
            coordinates,
            values: _,
            shape,
            device: _,
        } = tensor;
        let all = match coordinates {
            Some(coordinates) => shape.num_elements() == coordinates.shape().dims[1],
            None => false,
        };
        Tensor::<B, 1, Bool>::from([all]).into_primitive()
    }

    fn sparse_all_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("all_dim is unsupported for the SparseCOO Decorator due to performance issues, convert to dense explicitly to ensure you understand");
    }

    fn sparse_expand<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        _shape: Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        let SparseCOOTensor {
            coordinates: _,
            values: _,
            shape: _,
            device: _,
        } = tensor;
        todo!()
    }

    fn sparse_coalesce_sum<const D: usize>(
        tensor: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        if tensor.coordinates.as_ref().map(|c| c.shape().dims[1] <= 1) == Some(true) {
            return tensor;
        }
        let original_shape = tensor.shape.clone();

        if tensor.coordinates.is_none() && tensor.values.is_none() {
            return SparseCOOTensor {
                coordinates: None,
                values: None,
                shape: original_shape,
                device: tensor.device,
            };
        }

        let coordinates = tensor
            .coordinates
            .expect("Mismatch between coordinates and values");
        let values = tensor
            .values
            .expect("Mismatch between coordinates and values");
        let device = tensor.device;
        let nnz = coordinates.shape().dims[1];

        let coordinates =
            flatten_coordinates::<B, D, 0>(coordinates, original_shape.clone(), &device);
        let _flat_shape = Shape::new([original_shape.num_elements()]);

        let (coordinates, indices) = coordinates.sort_with_indices(1);
        let values = values.select(0, indices.squeeze(0));
        let range = Tensor::<B, 1, Int>::arange(0..nnz as i64, &device).unsqueeze::<2>();

        // Get the diff of coordinates, diff[i] = coordinates[i]-coordinates[i-1]
        let left_slice = coordinates.clone().slice([0..1, 0..nnz - 1]);
        let right_slice = coordinates.clone().slice([0..1, 1..nnz]);
        let diff = right_slice - left_slice;
        let ones = Tensor::<B, 2, Int>::ones(Shape::new([1, 1]), &device);
        let diff = Tensor::cat(vec![ones, diff], 1);

        // TODO this all would be way cleaner with cumsum/max, but that is waiting on a pull request as of writing
        // inspiration could be taken from pytorch_scatter for better implementations
        let unique_mask = diff.not_equal_elem(0);
        let unique_indices = unique_mask.clone().nonzero().remove(1);
        let steps = Tensor::cat(
            vec![unique_indices.clone(), Tensor::from_data([nnz], &device)],
            0,
        );
        let unique = steps.shape().dims[0];
        let steps = steps
            .clone()
            .slice([1..unique])
            .sub(steps.slice([0..unique - 1]))
            .max()
            // .sub_scalar(1)
            .into_scalar()
            .elem::<i32>();

        let mut scatter_indices = range.mul(unique_mask.int());

        for _ in 0..steps {
            scatter_indices = scatter_indices
                .clone()
                .slice([0..1, 1..nnz])
                .max_pair(scatter_indices.slice([0..1, 0..nnz - 1]));
            scatter_indices = Tensor::cat(
                vec![Tensor::zeros(Shape::new([1, 1]), &device), scatter_indices],
                1,
            );
        }

        // Scatter/Gather everything into place
        let zeroed = Tensor::<B, 1>::zeros(Shape::new([nnz]), &device);
        let values = zeroed.scatter(0, scatter_indices.squeeze(0), values);
        let values = values.gather(0, unique_indices.clone());
        let coordinates = coordinates.gather(1, unique_indices.unsqueeze::<2>());
        let coordinates = unflatten_coordinates(coordinates, original_shape.clone());

        let coordinates = Some(coordinates);
        let values = Some(values);

        // reshape back into the original shape and send it!
        SparseCOOTensor {
            coordinates,
            values,
            shape: original_shape,
            device,
        }
    }

    fn sparse_nonzero<const D: usize>(tensor: Self::SparseTensorPrimitive<D>) -> usize {
        match tensor.coordinates {
            Some(coordinates) => coordinates.shape().dims[1],
            None => 0,
        }
    }

    fn sparse_density<const D: usize>(sparse: Self::SparseTensorPrimitive<D>) -> f32 {
        match sparse.coordinates {
            Some(coordinates) => {
                coordinates.shape().dims[1] as f32 / sparse.shape.num_elements() as f32
            }
            None => 0.0,
        }
    }

    fn sparse_add<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        let SparseCOOTensor {
            coordinates: lhs_coordinates,
            values: lhs_values,
            shape: lhs_shape,
            device: lhs_device,
        } = lhs;
        let (Some(lhs_coordinates), Some(lhs_values)) = (lhs_coordinates, lhs_values) else {
            return rhs;
        };

        let SparseCOOTensor {
            coordinates: rhs_coordinates,
            values: rhs_values,
            shape: rhs_shape,
            device: rhs_device,
        } = rhs;
        let (Some(rhs_coordinates), Some(rhs_values)) = (rhs_coordinates, rhs_values) else {
            return SparseCOOTensor {
                coordinates: Some(lhs_coordinates),
                values: Some(lhs_values),
                shape: lhs_shape,
                device: lhs_device,
            };
        };

        assert_eq!(lhs_shape, rhs_shape);
        assert_eq!(lhs_device, rhs_device);

        let coordinates = Some(Tensor::cat(vec![lhs_coordinates, rhs_coordinates], 1));
        let values = Some(Tensor::cat(vec![lhs_values, rhs_values], 0));
        let shape = lhs_shape;
        let device = lhs_device;

        let result = SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        };

        Self::sparse_coalesce_sum(result)
    }

    fn sparse_add_scalar<const D: usize>(
        _: SparseTensor<Self, D>,
        _: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        panic!("Cannot add scalar to sparse, only zero preserving operations are permitted");
    }

    fn sparse_add_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        if lhs.shape != B::float_shape(&rhs) {
            panic!("lhs and rhs must have the same shape for sparse_add_dense");
        }

        if lhs.coordinates.is_none() && lhs.values.is_none() {
            return rhs;
        }

        let coordinates = lhs
            .coordinates
            .expect("Mismatch between coordinates and values");
        let values = lhs.values.expect("Mismatch between coordinates and values");
        let device = lhs.device;
        let shape = lhs.shape;

        let coordinates = flatten_coordinates::<B, D, 0>(coordinates, shape.clone(), &device);
        let dense = B::float_reshape(rhs, Shape::new([shape.num_elements()]));

        let dense = B::float_scatter(
            0,
            dense,
            coordinates.squeeze(0).into_primitive(),
            values.into_primitive().tensor(),
        );

        B::float_reshape(dense, shape)
    }

    fn sparse_sub<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        Self::sparse_add(
            lhs,
            Self::sparse_mul_scalar(rhs, FloatElem::<Self>::from_elem(-1.0)),
        )
    }

    fn sparse_sub_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        Self::sparse_add_dense(
            lhs,
            B::float_mul_scalar(rhs, FloatElem::<Self>::from_elem(-1.0)),
        )
    }

    fn sparse_sub_scalar<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        panic!("Cannot add scalar to sparse, only zero preserving operations are permitted");
    }

    fn sparse_mul<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        panic!("sparse_mul is unsupported until scatter supports multiplication based reduction");
    }

    fn sparse_mul_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        if lhs.shape != B::float_shape(&rhs) {
            panic!("lhs and rhs must have the same shape for sparse_add_dense");
        }

        if lhs.coordinates.is_none() && lhs.values.is_none() {
            return Self::float_zeros(lhs.shape, &lhs.device);
        }

        // TODO: this could potentially be optimized if/when scatter gets other reduction strategies
        let lhs = Self::sparse_to_dense(lhs);
        Self::float_mul(lhs, rhs)
    }

    fn sparse_mul_scalar<const D: usize>(
        mut lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        lhs.values = lhs.values.map(|values| values.mul_scalar(rhs));
        lhs
    }

    fn sparse_div<const D: usize>(
        _: SparseTensor<Self, D>,
        _: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        panic!("sparse_div is unsupported until scatter supports multiplication based reduction");
    }

    fn sparse_div_dense<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        if lhs.shape != B::float_shape(&rhs) {
            panic!("lhs and rhs must have the same shape for sparse_add_dense");
        }

        if lhs.coordinates.is_none() && lhs.values.is_none() {
            return Self::float_zeros(lhs.shape, &lhs.device);
        }

        // TODO: this could potentially be optimized if/when scatter gets other reduction strategies
        let lhs = Self::sparse_to_dense(lhs);
        Self::float_div(lhs, rhs)
    }

    fn sparse_div_scalar<const D: usize>(
        mut lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        lhs.values = lhs.values.map(|values| values.div_scalar(rhs));
        lhs
    }

    fn sparse_max<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        panic!("max is unsupported for SparseCOO until scatter supports other reduction methods");
    }

    fn sparse_max_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        panic!(
            "max_dim is unsupported for SparseCOO until scatter supports other reduction methods"
        );
    }

    fn sparse_min<const D: usize>(_tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        panic!("min is unsupported for SparseCOO until scatter supports other reduction methods");
    }

    fn sparse_min_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        panic!(
            "min_dim is unsupported for SparseCOO until scatter supports other reduction methods"
        );
    }

    fn sparse_greater<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_greater is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_greater_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_greater_elem is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_greater_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_greater_equal is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_greater_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!(
            "sparse_greater_equal_elem is not supported for SparseCOO as it outputs a dense tensor"
        );
    }

    fn sparse_lower<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_lower is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_lower_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_lower_elem is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_lower_equal<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_lower_equal is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_lower_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!(
            "sparse_lower_equal_elem is not supported for SparseCOO as it outputs a dense tensor"
        );
    }

    fn sparse_abs<const D: usize>(mut tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        tensor.values = tensor.values.map(|values| values.abs());
        tensor
    }

    fn sparse_powf<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        panic!("sparse_powf is unsupported for SparseCOO until scatter supports other reduction methods");
    }

    fn sparse_powi<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        panic!("sparse_powi is unsupported for SparseCOO until scatter supports other reduction methods");
    }

    fn sparse_powf_scalar<const D: usize>(
        mut lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        lhs.values = lhs.values.map(|values| values.powf_scalar(rhs));
        lhs
    }

    fn sparse_powi_scalar<const D: usize>(
        mut lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        lhs.values = lhs.values.map(|values| values.powi_scalar(rhs));
        lhs
    }

    fn sparse_clamp<const D: usize>(
        mut tensor: SparseTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        tensor.values = tensor.values.map(|values| values.clamp(min, max));
        tensor
    }

    fn sparse_clamp_min<const D: usize>(
        mut tensor: SparseTensor<Self, D>,
        min: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        tensor.values = tensor.values.map(|values| values.clamp_min(min));
        tensor
    }

    fn sparse_clamp_max<const D: usize>(
        mut tensor: SparseTensor<Self, D>,
        max: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        tensor.values = tensor.values.map(|values| values.clamp_max(max));
        tensor
    }

    fn sparse_select<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<Self, 1>,
    ) -> SparseTensor<Self, D> {
        if tensor.coordinates.is_none() && tensor.values.is_none() {
            return tensor;
        }

        let coordinates = tensor
            .coordinates
            .expect("Mismatch between coordinates and values");
        let values = tensor
            .values
            .expect("Mismatch between coordinates and values");
        let device = tensor.device;
        let mut shape = tensor.shape;
        let indices = Tensor::<B, 1, Int>::new(indices);

        let nnz = coordinates.shape().dims[1];
        let dim_coords = coordinates
            .clone()
            .slice([dim..dim + 1, 0..nnz])
            .squeeze::<1>(0);
        let indices = indices.select(0, dim_coords);
        let indices_len = indices.shape().num_elements();
        let coordinates = coordinates.slice_assign(
            [dim..dim + 1, 0..nnz],
            indices.unsqueeze::<2>().repeat(1, D),
        );

        shape.dims[dim] = indices_len;

        SparseCOOTensor {
            coordinates: Some(coordinates),
            values: Some(values),
            shape,
            device,
        }
    }

    fn sparse_select_assign<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
        _indices: burn_tensor::ops::IntTensor<Self, 1>,
        _values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_gather<const D: usize>(
        _dim: usize,
        _tensor: SparseTensor<Self, D>,
        _indices: burn_tensor::ops::IntTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_scatter<const D: usize>(
        _dim: usize,
        _tensor: SparseTensor<Self, D>,
        _indices: burn_tensor::ops::IntTensor<Self, D>,
        _values: SparseTensor<Self, D>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_sum<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        tensor
            .values
            .map(|values| Self::sparse_to_sparse(values.sum().into_primitive().tensor()))
            .unwrap_or(Self::sparse_empty(Shape::new([1]), &tensor.device))
    }

    fn sparse_sum_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        panic!("sparse_sum_dim unsupported for SparseCOO");
    }

    fn sparse_prod<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        if tensor.coordinates.is_none() && tensor.values.is_none() {
            return Self::sparse_empty(Shape::new([1]), &tensor.device);
        }

        let coordinates = tensor
            .coordinates
            .expect("Mismatch between coordinates and values");
        let values = tensor
            .values
            .expect("Mismatch between coordinates and values");
        let device = tensor.device;
        let shape = tensor.shape;

        if shape.num_elements() != coordinates.dims()[1] {
            Self::sparse_empty(Shape::new([1]), &device)
        } else {
            Self::sparse_to_sparse(values.sum().into_primitive().tensor())
        }
    }

    fn sparse_prod_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        panic!("sparse_prod_dim is not supported for SparseCOO until scatter supports product reduction")
    }

    fn sparse_mean<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, 1> {
        tensor
            .values
            .map(|values| {
                let elems = values.shape().num_elements();
                Self::sparse_to_sparse((values.sum() / elems as f32).into_primitive().tensor())
            })
            .unwrap_or(Self::sparse_empty(Shape::new([1]), &tensor.device))
    }

    fn sparse_mean_dim<const D: usize>(
        _tensor: SparseTensor<Self, D>,
        _dim: usize,
    ) -> SparseTensor<Self, D> {
        panic!("mean_dim is not supported for SparseCOO until scatter supports mean reduction");
    }

    fn sparse_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_equal_elem is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_not_equal_elem<const D: usize>(
        _lhs: SparseTensor<Self, D>,
        _rhs: FloatElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("sparse_not_equal_elem is not supported for SparseCOO as it outputs a dense tensor");
    }

    fn sparse_remainder_scalar<const D: usize>(
        mut lhs: SparseTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> SparseTensor<Self, D> {
        lhs.values = lhs.values.map(|v| v.remainder_scalar(rhs));
        lhs
    }

    fn sparse_neg<const D: usize>(mut tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        tensor.values = tensor.values.map(|v| v.neg());
        tensor
    }

    fn sparse_sign<const D: usize>(mut tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        tensor.values = tensor.values.map(|values| values.sign());
        tensor
    }

    fn sparse_remove_zeros<const D: usize>(
        tensor: Self::SparseTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        if tensor.coordinates.is_none() && tensor.values.is_none() {
            return tensor;
        }

        let _coordinates = tensor
            .coordinates
            .expect("Mismatch between coordinates and values");
        let _values = tensor
            .values
            .expect("Mismatch between coordinates and values");
        let _device = tensor.device;
        let _shape = tensor.shape;

        // let zeros = tensor.values.map(|values| values.equal_elem(0).nonzero());
        todo!()
    }
}
