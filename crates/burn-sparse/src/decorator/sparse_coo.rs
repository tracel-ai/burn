use crate::backend::SparseBackend;
use crate::backend::SparseTensor;
use crate::decorator::SparseCOO;
use crate::decorator::SparseDecorator;
use burn_tensor::ops::IntTensorOps;
use burn_tensor::Device;
use burn_tensor::{
    backend::Backend, Bool, ElementConversion, Float, Int, Shape, Tensor, TensorData,
    TensorPrimitive,
};
use half::vec;

#[derive(Clone, Debug)]
pub struct SparseCOOTensor<B: Backend, const D: usize> {
    pub coordinates: Option<Tensor<B, 2, Int>>,
    pub values: Option<Tensor<B, 1, Float>>,
    pub shape: Shape<D>,
    pub device: Device<B>,
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

        let num_nonzero = coordinates.shape().dims[1];

        let dense: Tensor<B, 1, Float> = Tensor::zeros(Shape::new([shape.num_elements()]), &device);

        let mut strides_data = [[1]; D];
        for i in (0..D - 1).rev() {
            strides_data[i] = [strides_data[i + 1][0] * shape.dims[i + 1] as i64];
        }

        let strides_data: TensorData = TensorData::from(strides_data);

        let strides: Tensor<B, 2, Int> = Tensor::from_data(strides_data, &device);

        let coordinates = strides.mul(coordinates).sum_dim(0).flatten(0, 1);

        let dense = dense.select_assign(0, coordinates, values);

        dense.reshape(shape).into_primitive().tensor()
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

    fn sparse_sddmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
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
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = tensor;

        let mut indices = Vec::from(indices);
        indices.extend(shape.dims[indices.len()..D1].iter().map(|&l| 0..l));
        let indices: [core::ops::Range<usize>; D1] = indices.try_into().expect("D2 must be <= D1");
        let out_shape = Shape::new(indices.clone().map(|r| r.end));

        let (Some(coordinates), Some(values)) = (coordinates, values) else {
            // All zeros, exit early
            return SparseCOOTensor {
                coordinates: None,
                values: None,
                shape: out_shape,
                device,
            };
        };

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

        let nonzero = mask.not_equal_elem(B::IntElem::from_elem(0)).nonzero();

        let indices_dim1 = nonzero
            .get(0)
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
        let dense = B::float_from_data(data, &device);
        Self::sparse_to_sparse(dense)
    }

    fn sparse_into_data<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> impl core::future::Future<Output = TensorData> + Send {
        B::float_into_data(Self::sparse_to_dense(tensor))
    }

    fn sparse_reshape<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        out_shape: Shape<D2>,
    ) -> SparseTensor<Self, D2> {
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
                shape: out_shape,
                device,
            };
        };

        // Flatten the coordinates:
        let mut strides_data = [[1]; D1];
        for i in (0..D1 - 1).rev() {
            strides_data[i] = [strides_data[i + 1][0] * shape.dims[i + 1] as i64];
        }
        let strides_data: TensorData = TensorData::from(strides_data);
        let strides: Tensor<B, 2, Int> = Tensor::from_data(strides_data, &device);
        let flat_coordinates: Tensor<B, 1, Int> = strides.mul(coordinates).sum_dim(0).flatten(0, 1);

        // Convert the flattened coordinates to the new shape
        let mut remaining_flat_coordinates = flat_coordinates.clone();
        let mut new_coordinates = Vec::with_capacity(D2);

        for &dim_size in out_shape.dims.iter().rev() {
            let size = dim_size as i64;
            let new_coord = remaining_flat_coordinates.clone().remainder_scalar(size);
            new_coordinates.push(new_coord.clone());
            remaining_flat_coordinates = remaining_flat_coordinates.div_scalar(size);
        }

        new_coordinates.reverse();
        let coordinates = Tensor::stack(new_coordinates, 0);

        let coordinates = Some(coordinates);
        let values = Some(values);

        SparseCOOTensor {
            coordinates,
            values,
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
        value: SparseTensor<Self, D1>,
    ) -> SparseTensor<Self, D1> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = tensor;
        todo!()
    }

    fn sparse_repeat<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> SparseTensor<Self, D> {
        let SparseCOOTensor {
            coordinates,
            values,
            mut shape,
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
        tensors: Vec<SparseTensor<Self, D>>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_not_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_any<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = tensor;
        let any = !matches!(coordinates, None);
        Tensor::<B, 1, Bool>::from([any]).into_primitive()
    }

    fn sparse_any_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("any_dim is unsupported for the SparseCOO Decorator due to performance issues, convert to dense explicitly to ensure you understand");
    }

    fn sparse_all<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = tensor;
        let all = match coordinates {
            Some(coordinates) => shape.num_elements() == coordinates.shape().dims[1],
            None => false,
        };
        Tensor::<B, 1, Bool>::from([all]).into_primitive()
    }

    fn sparse_all_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        panic!("all_dim is unsupported for the SparseCOO Decorator due to performance issues, convert to dense explicitly to ensure you understand");
    }

    fn sparse_expand<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
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
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        } = Self::sparse_reshape(tensor, Shape::new([original_shape.num_elements()]));
        let (Some(coordinates), Some(values)) = (coordinates, values) else {
            // All zeros, exit early
            return Self::sparse_reshape(
                SparseCOOTensor {
                    coordinates: None,
                    values: None,
                    shape,
                    device,
                },
                original_shape,
            );
        };

        let nnz = coordinates.shape().dims[1];
        if nnz <= 1 {
            // impossible to be uncoalesced
            return SparseCOOTensor {
                coordinates: Some(coordinates),
                values: Some(values),
                shape: original_shape,
                device,
            };
        }

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
        // this is technically O(nnz) but only in super rare and likely constructed cases
        // lots of inspiration could be taken from pytorch_scatter for better implementations
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

        let coordinates = Some(coordinates);
        let values = Some(values);

        // reshape back into the original shape and send it!
        let out = SparseCOOTensor {
            coordinates,
            values,
            shape,
            device,
        };

        Self::sparse_reshape(out, original_shape)
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
}
