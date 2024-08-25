use super::coo::{flatten_coordinates, unflatten_coordinates, SparseCOOTensor, COO};
use burn_tensor::cast::ToElement;
use burn_tensor::ops::{FloatElem, SparseBoolOps};
use burn_tensor::{backend::Backend, ops::SparseFloatOps, Tensor};
use burn_tensor::{
    Bool, ElementConversion, Float, Shape, Sparse, SparseStorage, TensorData, TensorKind,
    TensorPrimitive,
};
use burn_tensor::{Device, Int};

impl<B: Backend> SparseFloatOps<COO, B> for COO {
    fn float_to_sparse<const D: usize>(
        dense: <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        let dense: Tensor<B, D> = Tensor::from_primitive(TensorPrimitive::Float(dense));

        let shape = dense.shape();
        let device = dense.device();

        let significant = dense.clone().not_equal_elem(0.0);
        if !significant.clone().any().into_scalar() {
            return Self::float_empty(dense.shape(), &device);
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

        SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        }
    }

    fn float_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        SparseCOOTensor {
            coordinates: None,
            values: None,
            shape,
            device: device.clone(),
        }
    }

    fn float_to_dense<const D: usize>(
        sparse: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> B::FloatTensorPrimitive<D> {
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

    fn float_spmm<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <Float as TensorKind<B>>::Primitive<D>,
    ) -> <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = lhs;

        let rhs: Tensor<B, D, Float> = Tensor::from_primitive(rhs);
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

        let gather_index = gather_index
            .transpose()
            .repeat_dim(1, rhs_shape.dims[D - 1]);
        let scatter_index = scatter_index
            .transpose()
            .repeat_dim(1, rhs_shape.dims[D - 1]);
        let values = values.unsqueeze_dim(1).repeat_dim(1, rhs_shape.dims[D - 1]);

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

    fn float_sddmm<const D: usize>(
        lhs: <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D>,
        rhs: <B as burn_tensor::backend::Backend>::FloatTensorPrimitive<D>,
        sparse: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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
            .repeat_dim(1, lhs_coordinates.shape().dims[1]);
        let rhs_coordinates = lhs_coordinates.clone().gather(0, swizzle);

        let row_indices = flatten_coordinates::<B, D, 1>(lhs_coordinates, shape.clone(), &device);

        shape.dims.swap(D - 1, D - 2);
        let col_indices = flatten_coordinates::<B, D, 1>(rhs_coordinates, shape.clone(), &device);

        let row_indices = row_indices.transpose().repeat_dim(1, lhs_dims[D - 1]);
        let col_indices = col_indices.transpose().repeat_dim(1, rhs_dims[D - 1]);

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

    fn float_coalesce_sum<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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

    fn float_remove_zeros<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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
        let shape = tensor.shape;

        todo!()
    }

    fn float_number_nonzero<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> usize {
        match tensor.coordinates {
            Some(coordinates) => coordinates.shape().dims[1],
            None => 0,
        }
    }

    fn float_density<const D: usize>(
        sparse: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> f32 {
        match sparse.coordinates {
            Some(coordinates) => {
                coordinates.shape().dims[1] as f32 / sparse.shape.num_elements() as f32
            }
            None => 0.0,
        }
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D1> {
        todo!()
    }

    fn float_device<const D: usize>(
        tensor: &<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> burn_tensor::Device<B> {
        tensor.device.clone()
    }

    fn float_to_device<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        device: &burn_tensor::Device<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        SparseCOOTensor {
            coordinates: tensor.coordinates.map(|t| t.to_device(device)),
            values: tensor.values.map(|t| t.to_device(device)),
            shape: tensor.shape,
            device: device.clone(),
        }
    }

    fn float_shape<const D: usize>(
        tensor: &<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> burn_tensor::Shape<D> {
        tensor.shape.clone()
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D1>,
        out_shape: burn_tensor::Shape<D2>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D2> {
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

    fn float_transpose<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        let d = tensor.shape.dims.len();
        let mut axes: Vec<usize> = (0..d).collect();
        axes.swap(d - 1, d - 2);
        Self::float_permute(tensor, &axes)
    }

    fn float_swap_dims<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim1: usize,
        dim2: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        let d = tensor.shape.dims.len();
        let mut axes: Vec<usize> = (0..d).collect();
        axes.swap(dim1, dim2);
        Self::float_permute(tensor, &axes)
    }

    fn float_permute<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        axes: &[usize],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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

    fn float_flip<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        axes: &[usize],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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
            .repeat_dim(1, nnz)
            .bool();

        let flipped: Tensor<B, 2, Int> = Tensor::<_, 1, _>::from_ints(shape.dims, &device)
            .unsqueeze_dim(1)
            .repeat_dim(1, nnz)
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

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D1>,
        ranges: [std::ops::Range<usize>; D2],
        mut value: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D1>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D1> {
        let value_nnz = value
            .coordinates
            .as_ref()
            .map(|coords| coords.shape().dims[1])
            .unwrap_or(0);

        let mut ranges = Vec::from(ranges);
        ranges.extend(tensor.shape.dims[ranges.len()..D1].iter().map(|&l| 0..l));
        let ranges: [core::ops::Range<usize>; D1] = ranges.try_into().expect("D2 must be <= D1");

        let shape = tensor.shape.clone();
        let sliced = Self::float_reshape(
            Self::float_slice(tensor.clone(), ranges.clone()),
            shape.clone(),
        );
        let tensor = Self::float_sub(tensor, sliced);
        let offset = Tensor::<B, 1, Int>::from_ints(ranges.map(|r| r.start), &tensor.device);
        let offset = offset.unsqueeze_dim::<2>(1).repeat_dim(1, value_nnz);

        value.shape = shape;
        value.coordinates = value.coordinates.map(|coords| coords + offset);

        Self::float_add(tensor, value)
    }

    fn float_repeat_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
        times: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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

        let values = values.repeat_dim(0, times);

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

    fn float_cat<const D: usize>(
        tensors: Vec<<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_equal<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_not_equal<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn float_any<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, 1> {
        let SparseCOOTensor {
            coordinates,
            values: _,
            shape: _,
            device: _,
        } = tensor;
        let any = coordinates.is_some();
        let bool = Tensor::<B, 1, Bool>::from([any]).into_primitive();
        <Self as SparseBoolOps<COO, B>>::bool_to_sparse(bool)
    }

    fn float_any_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        panic!("any_dim is unsupported for COO until scatter supports any-based reduction");
    }

    fn float_all<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, 1> {
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
        let bool = Tensor::<B, 1, Bool>::from([all]).into_primitive();
        <Self as SparseBoolOps<COO, B>>::bool_to_sparse(bool)
    }

    fn float_all_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        panic!("all_dim is unsupported for COO until scatter supports all-based reduction");
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D2> {
        todo!()
    }

    fn float_add<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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

        Self::float_coalesce_sum(result)
    }

    fn float_sub<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        Self::float_add(
            lhs,
            Self::float_mul_scalar(rhs, FloatElem::<B>::from_elem(-1.0)),
        )
    }

    fn float_mul<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("float_mul is unsupported until scatter supports multiplication based reduction");
    }

    fn float_mul_scalar<const D: usize>(
        mut lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        lhs.values = lhs.values.map(|values| values.mul_scalar(rhs));
        lhs
    }

    fn float_div<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("float_div is unsupported until scatter supports multiplication based reduction");
    }

    fn float_div_scalar<const D: usize>(
        mut lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        lhs.values = lhs.values.map(|values| values.div_scalar(rhs));
        lhs
    }

    fn float_max<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("max is unsupported for COO until scatter supports max reduction");
    }

    fn float_max_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("max_dim is unsupported for COO until scatter supports max reduction");
    }

    fn float_min<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("min is unsupported for COO until scatter supports min reduction");
    }

    fn float_min_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("min_dim is unsupported for COO until scatter supports min reduction");
    }

    fn float_abs<const D: usize>(
        mut tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        tensor.values = tensor.values.map(|values| values.abs());
        tensor
    }

    fn float_sign<const D: usize>(
        mut tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        tensor.values = tensor.values.map(|values| values.sign());
        tensor
    }

    fn float_powf<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("float_powf is unsupported for COO until scatter supports other reduction methods");
    }

    fn float_powi<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("float_powi is unsupported for COO until scatter supports other reduction methods");
    }

    fn float_powf_scalar<const D: usize>(
        mut lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        lhs.values = lhs.values.map(|values| values.powf_scalar(rhs));
        lhs
    }

    fn float_powi_scalar<const D: usize>(
        mut lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        lhs.values = lhs.values.map(|values| values.powi_scalar(rhs));
        lhs
    }

    fn float_clamp<const D: usize>(
        mut tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        min: burn_tensor::ops::FloatElem<B>,
        max: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        tensor.values = tensor.values.map(|values| values.clamp(min, max));
        if min.to_f64() == 0f64 || max.to_f64() == 0f64 {
            // Clamp can zero elements if a boundary is zero
            Self::float_remove_zeros(tensor)
        } else {
            tensor
        }
    }

    fn float_clamp_min<const D: usize>(
        mut tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        min: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        tensor.values = tensor.values.map(|values| values.clamp_min(min));
        if min.to_f64() == 0f64 {
            // min can zero elements if boundary is 0
            Self::float_remove_zeros(tensor)
        } else {
            tensor
        }
    }

    fn float_clamp_max<const D: usize>(
        mut tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        max: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        tensor.values = tensor.values.map(|values| values.clamp_max(max));
        if max.to_f64() == 0f64 {
            // max can zero elements if boundary is 0
            Self::float_remove_zeros(tensor)
        } else {
            tensor
        }
    }

    fn float_select<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<B, 1>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
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
            indices.unsqueeze::<2>().repeat_dim(1, D),
        );

        shape.dims[dim] = indices_len;

        SparseCOOTensor {
            coordinates: Some(coordinates),
            values: Some(values),
            shape,
            device,
        }
    }

    fn float_select_assign<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<B, 1>,
        values: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        indices: burn_tensor::ops::IntTensor<B, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        indices: burn_tensor::ops::IntTensor<B, D>,
        values: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        todo!()
    }

    fn float_sum<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, 1> {
        tensor
            .values
            .map(|values| Self::float_to_sparse(values.sum().into_primitive().tensor()))
            .unwrap_or(Self::float_empty(Shape::new([1]), &tensor.device))
    }

    fn float_sum_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("float_sum_dim unsupported for COO");
    }

    fn float_prod_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("float_prod_dim is not supported for COO until scatter supports product reduction")
    }

    fn float_mean<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, 1> {
        let num_elems = tensor.shape.num_elements();
        Self::float_div_scalar(
            Self::float_sum(tensor),
            ElementConversion::elem(num_elems as f32),
        )
    }

    fn float_mean_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        panic!("float_mean_dim is not supported for COO until scatter supports mean reduction");
    }

    fn float_remainder_scalar<const D: usize>(
        mut lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        lhs.values = lhs.values.map(|values| values.remainder_scalar(rhs));
        lhs
    }

    fn float_neg<const D: usize>(
        mut tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Float, D> {
        tensor.values = tensor.values.map(|values| values.neg());
        tensor
    }
}
