use crate::backend::SparseBackend;
use crate::backend::SparseTensor;
use crate::decorator::SparseCOO;
use crate::decorator::SparseDecorator;
use burn_tensor::{
    backend::Backend, ElementConversion, Float, Int, Shape, Tensor, TensorData, TensorPrimitive,
};

#[derive(Clone, Debug)]
pub struct SparseCOOTensor<B: Backend, const D: usize> {
    pub coordinates: Tensor<B, 2, Int>,
    pub values: Tensor<B, 1, Float>,
    pub shape: Shape<D>,
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

        let significant = dense.clone().not_equal_elem(0.0);

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

        Self::SparseTensorPrimitive {
            coordinates,
            values,
            shape,
        }
    }

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
        } = sparse;

        let num_nonzero = coordinates.shape().dims[1];
        let device = coordinates.device();

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
        } = lhs;

        let rhs: Tensor<B, D, Float> = Tensor::from_primitive(TensorPrimitive::Float(rhs));
        let rhs_shape = rhs.shape();
        let device = coordinates.device();
        let nnz = coordinates.shape().dims[1];

        // Ensure they are of the correct shape to multiply
        if shape.dims[D - 1] != rhs_shape.dims[D - 2] {
            panic!("Invalid shape for matrix multiplication");
        }

        // Ensure batches are the same
        if D > 2 && rhs_shape.dims[0..D - 2] != shape.dims[0..D - 2] {
            panic!("Batches must be of the same shape");
        }

        let mut out_shape = shape.clone();
        out_shape.dims[D - 1] = rhs_shape.dims[D - 1];

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
        tensor.values.device()
    }

    fn sparse_to_device<const D: usize>(
        tensor: SparseTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        SparseCOOTensor {
            coordinates: tensor.coordinates.to_device(device),
            values: tensor.values.to_device(device),
            shape: tensor.shape,
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
            coordinates: Tensor::from_primitive(B::int_empty(
                burn_tensor::Shape::new([0, 0]),
                &device,
            )),
            values: Tensor::from_primitive(TensorPrimitive::Float(B::float_empty(
                burn_tensor::Shape::new([0]),
                &device,
            ))),
            shape,
        }
    }

    fn sparse_slice<const D1: usize, const D2: usize>(
        tensor: Self::SparseTensorPrimitive<D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> SparseTensor<Self, D1> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
        } = tensor;

        let device = coordinates.device();
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

        SparseCOOTensor {
            coordinates,
            values,
            shape,
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
    ) -> impl std::future::Future<Output = TensorData> + Send {
        // TODO this could be way better
        B::float_into_data(Self::sparse_to_dense(tensor))
    }
}
