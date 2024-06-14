#![allow(clippy::single_range_in_vec_init)]

use alloc::vec::Vec;

#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
use alloc::format;
#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
use alloc::string::String;
#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
use alloc::vec;

use burn_common::{reader::Reader, stub::Mutex};
use core::iter::repeat;
use core::{fmt::Debug, ops::Range};
use serde::{Deserialize, Deserializer};

#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
use serde::{Serialize, Serializer};

use crate::check::TensorCheck;
use crate::tensor::api::chunk::chunk;
use crate::tensor::api::narrow::narrow;
use crate::Element;
use crate::{backend::Backend, check, Bool, Data, DataSerialize, Float, Int, Shape, TensorKind};

/// A tensor with a given backend, shape and data type.
#[derive(new, Clone, Debug)]
pub struct Tensor<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub(crate) primitive: K::Primitive<D>,
}

impl<B, const D: usize, K, T> From<T> for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    T: Into<Data<K::Elem, D>>,
{
    fn from(value: T) -> Self {
        Tensor::from_data(value.into(), &Default::default())
    }
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> K::Primitive<D> {
        self.primitive
    }

    /// Converts from a primitive tensor into a tensor.
    pub fn from_primitive(tensor: K::Primitive<D>) -> Self {
        Self::new(tensor)
    }

    /// Create an empty tensor of the given shape.
    pub fn empty<S: Into<Shape<D>>>(shape: S, device: &B::Device) -> Self {
        Self::new(K::empty(shape.into(), device))
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// Equivalent to `tensor.shape().dims`.
    pub fn dims(&self) -> [usize; D] {
        Self::shape(self).dims
    }

    /// Returns the shape of the current tensor.
    pub fn shape(&self) -> Shape<D> {
        K::shape(&self.primitive)
    }

    /// Reshape the tensor to have the given shape.
    ///
    /// A `-1` in the shape is used to infer the remaining dimensions, e.g.: `[2, -1]`
    /// will reshape the tensor with [2, 3, 4] dimensions to [2, 12].
    ///
    /// A `0` in the shape instructs to keep the current dimension from the original tensor,
    /// e.g.: `[2, 0, 4]` will reshape the tensor with [2, 3, 4] dimensions to [2, 3, 4].
    /// This is useful when reshaping tensors with unknown dimensions and combining with `-1`
    /// to infer the remaining dimensions, e.g. `[0, -1]` will reshape the tensor
    /// with [1, 3, 4] dimensions to [1, 12].
    ///
    /// # Arguments
    /// - `shape`: The new shape of the tensor.
    ///
    /// # Panics
    /// - If the tensor contains more than one `-1` in the shape.
    /// - If the tensor contains values that are not positive (other than -1).
    /// - If the shape does not match the number of elements of the original shape.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///    // Given a 3D tensor with dimensions (2, 3, 4), reshape it to (2, 12)
    ///    let reshaped_tensor: Tensor::<B, 2> = tensor.reshape([2, -1]);
    ///    // The resulting tensor will have dimensions (2, 12).
    ///    println!("{:?}", reshaped_tensor.shape());
    /// }
    /// ```
    pub fn reshape<const D2: usize, S: ReshapeArgs<D2>>(self, shape: S) -> Tensor<B, D2, K> {
        // Convert reshape args to shape
        let shape = shape.into_shape(&self);
        Tensor::new(K::reshape::<D, D2>(self.primitive, shape))
    }

    /// Transpose the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    pub fn transpose(self) -> Tensor<B, D, K> {
        Tensor::new(K::transpose(self.primitive))
    }

    /// Swaps two dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Tensor<B, D, K> {
        Tensor::new(K::swap_dims(self.primitive, dim1, dim2))
    }

    /// Permute the dimensions of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of the dimensions. The length of the axes
    ///            must be equal to the number of dimensions of the tensor.
    ///            The values must be unique and in the range of the number of dimensions.
    ///            The values can be negative, in which case they are used as an offset from the end.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    pub fn permute(self, axes: [isize; D]) -> Tensor<B, D, K> {
        // Convert the axes to usize and handle negative values without using vector
        let mut transformed_axes: [usize; D] = [0; D];
        for (i, &x) in axes.iter().enumerate() {
            transformed_axes[i] = if x < 0 {
                (D as isize + x) as usize
            } else {
                x as usize
            };
        }

        // Check if the axes are valid after the transformation
        check!(TensorCheck::permute(transformed_axes));

        Tensor::new(K::permute(self.primitive, transformed_axes))
    }

    /// Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.
    ///
    /// Other dimensions of input that are not explicitly moved remain in their original order and appear
    /// at the positions not specified in destination.
    ///
    /// # Arguments
    ///
    /// * `src` - The dimension(s) to move. The values must be unique and in the range of the number of dimensions.
    ///              The values can be negative, in which case they are used as an offset from the end.
    ///
    /// * `dst` - Destination positions for each of the original dims. These must also be unique.
    ///
    /// # Panics
    ///
    /// - If the source and destination dimensions are not of the same length.
    /// - If the source and destination vectors contain duplicate values.
    /// - If the source and destination vectors contain values that are out of bounds.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions moved.
    // This is a semantic sugar for `permute`. It is used widely enough, so we define a separate Op
    // for it
    pub fn movedim<S1: MovedimArgs, S2: MovedimArgs>(self, src: S1, dst: S2) -> Tensor<B, D, K> {
        let source_dims = src.into_dim_vec::<D>();
        let destination_dims = dst.into_dim_vec::<D>();

        check!(TensorCheck::movedim_args_length(
            &source_dims,
            &destination_dims
        ));

        let mut m = [-1; D];
        for (&d, &s) in destination_dims.iter().zip(source_dims.iter()) {
            m[d] = s as isize;
        }
        let mut axes: [isize; D] = [0; D];
        let mut source_i = 0;
        for (dest_i, item) in axes.iter_mut().enumerate().take(D) {
            *item = if m[dest_i] != -1 {
                m[dest_i]
            } else {
                while source_dims.contains(&source_i) {
                    source_i += 1;
                }
                let result = source_i as isize;
                source_i += 1;
                result
            };
        }

        self.permute(axes)
    }

    /// Reverse the order of elements in the tensor along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - The dimensions to reverse. The values must be unique and in the range of the number of dimensions.
    ///            The values can be negative, in which case they are used as an offset from the end.
    ///
    /// # Returns
    ///
    /// The tensor with the axes flipped.
    pub fn flip<const N: usize>(self, axes: [isize; N]) -> Tensor<B, D, K> {
        // Convert the axes to usize and handle negative values without using vector
        let mut transformed_axes: [usize; N] = [0; N];
        for (i, &x) in axes.iter().enumerate() {
            transformed_axes[i] = if x < 0 {
                (D as isize + x) as usize
            } else {
                x as usize
            };
        }

        // Check if the axes are valid
        check!(TensorCheck::flip(D, &transformed_axes));

        Tensor::new(K::flip(self.primitive, &transformed_axes))
    }

    /// Flatten the tensor along a given range of dimensions.
    ///
    /// This function collapses the specified range of dimensions into a single dimension,
    /// effectively flattening the tensor in that range.
    ///
    /// # Arguments
    ///
    /// - `start_dim`: The starting dimension of the range to be flattened.
    /// - `end_dim`: The ending dimension of the range to be flattened (inclusive).
    ///
    /// # Type Parameters
    ///
    /// - `D2`: The resulting number of dimensions in the flattened tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified range of dimensions flattened.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 4]), &device);
    ///
    ///     // Given a 3D tensor with dimensions (2, 3, 4), flatten the dimensions between indices 1 and 2:
    ///     let flattened_tensor: Tensor::<B, 2> = tensor.flatten(1, 2);
    ///
    ///     // The resulting tensor will have dimensions (2, 12).
    ///    println!("{:?}", flattened_tensor.shape());
    /// }
    ///
    /// ```
    pub fn flatten<const D2: usize>(self, start_dim: usize, end_dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::flatten::<D, D2>(start_dim, end_dim));

        let current_dims = self.shape().dims;
        let mut new_dims: [usize; D2] = [0; D2];
        let mut flatten_dims = 1;

        for i in current_dims[start_dim..=end_dim].iter() {
            flatten_dims *= i;
        }

        new_dims[..start_dim].copy_from_slice(&current_dims[..start_dim]);
        new_dims[start_dim] = flatten_dims;
        new_dims[start_dim + 1..].copy_from_slice(&current_dims[end_dim + 1..]);

        Tensor::new(K::reshape::<D, D2>(self.primitive, new_dims.into()))
    }

    /// Squeeze the tensor along the given dimension, removing the specified dimension
    /// of size one, and effectively reducing the rank of the tensor by one.
    ///
    /// # Arguments
    ///
    /// - `dim`: The dimension to be squeezed.
    ///
    /// # Type Parameters
    ///
    ///  - 'D2': The resulting number of dimensions in the squeezed tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified dimension removed.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 1, 4]), &device);
    ///
    ///     // Given a 3D tensor with dimensions (2, 1, 4), squeeze the dimension 1
    ///     let squeezed_tensor: Tensor::<B, 2> = tensor.squeeze(1);
    ///
    ///     // Resulting tensor will have dimensions (2, 4)
    ///     println!("{:?}", squeezed_tensor.shape());
    /// }
    /// ```
    pub fn squeeze<const D2: usize>(self, dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::squeeze::<D2>(dim, &self.shape().dims));

        let current_dims = self.shape().dims;
        let mut new_dims: [usize; D2] = [0; D2];

        new_dims[..dim].copy_from_slice(&current_dims[..dim]);
        new_dims[dim..].copy_from_slice(&current_dims[dim + 1..]);

        Tensor::new(K::reshape::<D, D2>(self.primitive, new_dims.into()))
    }

    /// Removes specified dimensions of size 1 from a tensor's shape. This function takes a tensor and
    /// an array of dimensions (`dims`) to be squeezed. If `dims` is provided, only the dimensions
    /// specified in this array will be removed. Each dimension in `dims` should correspond to a size of 1
    /// in the tensor; otherwise, the dimension will not be squeezed. If `dims` is empty, all single-dimensional entries
    /// in the tensor will be removed. If entries in `dims` are negative, then dimensions will be counted
    /// from the back.
    ///
    /// # Arguments
    ///
    /// - `dims`: The dimension(s) to be squeezed.
    ///
    /// # Type Parameters
    ///
    ///  - 'D2': The resulting number of dimensions in the squeezed tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified dimensions removed.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 4>::ones(Shape::new([2, 1, 4, 1]), &device);
    ///
    ///     // Given a 4D tensor with dimensions (2, 1, 4, 1), squeeze the 1 and 3 dimensions
    ///     let squeezed_tensor: Tensor::<B, 2> = tensor.squeeze_dims(&[1, 3]);
    ///
    ///     // Resulting tensor will have dimensions (2, 4)
    ///     println!("{:?}", squeezed_tensor.shape());
    /// }
    /// ```
    pub fn squeeze_dims<const D2: usize>(self, dims: &[isize]) -> Tensor<B, D2, K> {
        let current_dims = self.shape().dims;
        let mut dim_indices: Vec<usize>;

        // Check if dims is empty, if yes then assign dim_indices all single-dimensional entries
        if dims.is_empty() {
            dim_indices = current_dims
                .iter()
                .enumerate()
                .filter_map(|(index, &dim)| if dim == 1 { Some(index) } else { None })
                .collect();
        } else {
            // If negative dims, count from the back
            dim_indices = dims
                .iter()
                .map(|&d| {
                    if d < 0 {
                        (current_dims.len() as isize + d) as usize
                    } else {
                        d as usize
                    }
                })
                .collect();
        }

        // Sort indices and remove duplicates
        dim_indices.sort_unstable();
        dim_indices.dedup();

        // Make sure squeeze_dims doesn't result in a tensor with < 1 dimensions
        check!(TensorCheck::squeeze_dims_input::<D2>(
            &dim_indices,
            &current_dims
        ));

        // Calculate new dimensions
        let mut new_dims = Vec::new();
        for (index, &dim_size) in current_dims.iter().enumerate() {
            // Exclude the dimension if it's explicitly marked for squeezing
            if dim_indices.contains(&index) {
                check!(TensorCheck::squeeze::<D2>(index, &current_dims));
                continue;
            }
            new_dims.push(dim_size);
        }

        // Check that after squeezing, we still respect the D2 size
        check!(TensorCheck::squeeze_dims_len::<D2>(new_dims.len()));

        Tensor::new(K::reshape::<D, D2>(self.primitive, new_dims.into()))
    }

    /// Unsqueeze the current tensor. Create new dimensions to fit the given size.
    ///
    /// If the output size is higher than the current tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]), &device);
    ///     let tensor = tensor.unsqueeze::<4>();
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [1, 1, 3, 3] }
    /// }
    /// ```
    pub fn unsqueeze<const D2: usize>(self) -> Tensor<B, D2, K> {
        check!(TensorCheck::unsqueeze::<D, D2>());

        let mut dims = [1; D2];
        let num_ones = D2 - D;
        let shape = self.shape();

        dims[num_ones..(D + num_ones)].copy_from_slice(&shape.dims[..D]);

        let shape = Shape::new(dims);
        self.reshape(shape)
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]), &device);
    ///     let tensor: Tensor<B, 3> = tensor.unsqueeze_dim(1);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [3, 1, 3] }
    /// }
    /// ```
    pub fn unsqueeze_dim<const D2: usize>(self, dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::unsqueeze_dim::<{ D }>(dim));

        let mut dims = [1; D2];
        let shape = self.shape();

        dims[0..dim].copy_from_slice(&shape.dims[0..dim]);

        if dim < D {
            dims[dim] = 1;
            dims[(dim + 1)..].copy_from_slice(&shape.dims[dim..]);
        } else {
            dims[dim] = 1;
        }

        let shape = Shape::new(dims);
        self.reshape(shape)
    }

    /// Creates a new tensor with added dimensions of size one inserted at the specified indices.
    /// The indices can be negative, in which case they are counted from the last to the first dimension.
    /// the axes can contain duplicates, in which case the number of dimensions inserted at the index
    /// is the number of duplicates.
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([3, 4, 5]), &device);
    ///     let tensor: Tensor<B, 6> = tensor.unsqueeze_dims(&[0, -1, -1]);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [1, 3, 4, 5, 1, 1] }
    /// }
    /// ```
    pub fn unsqueeze_dims<const D2: usize>(self, axes: &[isize]) -> Tensor<B, D2, K> {
        let mut new_dims = [1; D2];
        let old_dims = self.shape().dims;
        //for checking if the dimension is in the acceptable range

        //part 1: convert the negative indices to positive
        let mut dim_indices = axes
            .iter()
            .map(|d| {
                // check if the dimension is in the acceptable range
                check!(TensorCheck::unsqueeze_dims::<{ D2 }>(*d));
                (if *d < 0 { d + D2 as isize } else { *d }) as usize
            })
            .collect::<Vec<usize>>();

        //sort the indices
        dim_indices.sort_unstable();

        //Now use this to copy the chunks of the dims
        let mut prev_idx: usize = 0;
        let mut current_left_b: usize = 0;
        let mut current_right_b: usize = 0;
        let mut offset: usize = 0;
        dim_indices.iter().for_each(|d| {
            //check if there is space for at least one dimension
            if prev_idx < *d {
                current_right_b = *d - offset;
                //copy the chunks of the dims
                if current_right_b < D {
                    new_dims[prev_idx..*d]
                        .copy_from_slice(&old_dims[current_left_b..current_right_b]);
                } else {
                    new_dims[prev_idx..*d].copy_from_slice(&old_dims[current_left_b..]);
                }
                prev_idx = *d + 1;
                //offset is equal to the number of extracted elements from the original shape
                offset += current_right_b - current_left_b;
                current_left_b = current_right_b;
            } else {
                //it's sorted so the only reason this would happen
                //is if multiple indices are the same
                prev_idx += 1;
            }
        });
        //copy over anything past the index of the last new dimension
        if current_left_b < D {
            new_dims[prev_idx..].copy_from_slice(&old_dims[current_left_b..]);
        }

        //lastly, create the shape and reshape
        let shape = Shape::new(new_dims);
        self.reshape(shape)
    }

    /// Returns a tensor containing the elements selected from the given ranges.
    ///
    /// # Panics
    ///
    /// If a range exceeds the number of elements on a dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     // Create a tensor with a single dimension of ints between 0 and 11
    ///     let tensor = Tensor::<B, 1, burn_tensor::Int>::arange(0..12, &device);
    ///     // Select elements 0, 1, 2, 3 from the first dimension
    ///     let tensor_slices = tensor.clone().slice([0..4]);
    ///     println!("\nexpecting [0,1,2,3] : {:?}", tensor);
    ///     println!("expecting [4] : {:?}", tensor.dims());
    ///
    ///     // Create a Tensor with 3 dimensions
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]), &device);
    ///     // This slice will select the element 0 on the first dimension,
    ///     // elements 0,1,2 of the second dimension and element 1 of third dimension
    ///     let tensor_slices = tensor.slice([0..1, 0..3, 1..2]);
    ///     println!("expecting [1, 3, 1] : {:?}", tensor_slices.dims());
    ///
    ///     // Create a tensor of ints from 0 to 11 and reshape it into three dimensions
    ///     let tensor = Tensor::<B, 1, burn_tensor::Int>::arange(0..12, &device);
    ///     let tensor = tensor.reshape([1, 3, 4]);
    ///     println!("\nexpecting [[[0,1,2,3],[4,5,6,7],[8,9,10,11]]] : {:?}", tensor);
    ///     println!("expecting [1, 3, 4] : {:?}", tensor.dims());
    ///     // Select element 0 of first dimension, elements 1,2 of second dimension
    ///     // and element 1 of third dimension
    ///     //
    ///     // This is the equivalent of this pseudo code
    ///     // let mut v = vec![[[]]];
    ///     // v[0][0][0] = tensor[0][1][1];
    ///     // v[0][1][0] = tensor[0][2][1];
    ///     let tensor_slices = tensor.slice([0..1, 1..3, 1..2]);
    ///     println!("\nexpecting [1, 2, 1] : {:?}", tensor_slices.dims());
    ///     println!("expecting [[[5],[9]]] : {:?}", tensor_slices);
    /// }
    /// ```
    pub fn slice<const D2: usize>(self, ranges: [core::ops::Range<usize>; D2]) -> Self {
        check!(TensorCheck::slice(&self.shape(), &ranges));
        Self::new(K::slice(self.primitive, ranges))
    }

    /// Returns a copy of the current tensor with the selected elements changed to the new ones at
    /// the selected indices.
    ///
    /// # Panics
    ///
    /// - If a range exceeds the number of elements on a dimension.
    /// - If the given values don't match the given ranges.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 3>::ones([2, 3, 3], &device);
    ///     let values = Tensor::<B, 3>::zeros([1, 1, 1], &device);
    ///     let tensor_sliced = tensor.slice_assign([0..1, 0..1, 0..1], values);
    ///     println!("{:?}", tensor_sliced.dims()); // [2, 3, 3]
    /// }
    /// ```
    pub fn slice_assign<const D2: usize>(
        self,
        ranges: [core::ops::Range<usize>; D2],
        values: Self,
    ) -> Self {
        check!(TensorCheck::slice_assign(
            &self.shape(),
            &values.shape(),
            &ranges
        ));
        Self::new(K::slice_assign(self.primitive, ranges, values.primitive))
    }

    /// Returns the device of the current tensor.
    pub fn device(&self) -> B::Device {
        K::device(&self.primitive)
    }

    /// Returns a new tensor on the given device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self::new(K::to_device(self.primitive, device))
    }

    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    /// Returns the data of the current tensor.
    pub async fn into_data(self) -> Data<K::Elem, D> {
        K::into_data(self.primitive).read().await
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    /// Returns the data of the current tensor.
    pub fn into_data(self) -> Data<K::Elem, D> {
        K::into_data(self.primitive).read()
    }

    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    /// Returns the data of the current tensor.
    pub async fn to_data(&self) -> Data<K::Elem, D> {
        K::into_data(self.primitive.clone()).read().await
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    /// Returns the data of the current tensor without taking ownership.
    pub fn to_data(&self) -> Data<K::Elem, D> {
        Self::into_data(self.clone())
    }

    /// Create a tensor from the given data on the given device.
    pub fn from_data<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<Data<K::Elem, D>>,
    {
        Self::new(K::from_data(data.into(), device))
    }

    /// Repeat the tensor along the given dimension.
    pub fn repeat(self, dim: usize, times: usize) -> Self {
        Self::new(K::repeat(self.primitive, dim, times))
    }

    /// Applies element-wise equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Equal", &self, &other));
        K::equal(self.primitive, other.primitive)
    }

    /// Applies element-wise non-equality comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn not_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("NotEqual", &self, &other));
        K::not_equal(self.primitive, other.primitive)
    }

    /// Concatenates all tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// If all tensors don't have the same shape.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        check!(TensorCheck::cat(&tensors, dim));

        Self::new(K::cat(
            tensors.into_iter().map(|vector| vector.primitive).collect(),
            dim,
        ))
    }

    /// Concatenates all tensors into a new one along a new dimension.
    ///
    /// # Panics
    ///
    /// If all tensors don't have the same shape.
    /// Given dimension is not with range of 0..D2
    pub fn stack<const D2: usize>(tensors: Vec<Tensor<B, D, K>>, dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::stack(&tensors, dim));
        let tensors = tensors.into_iter().map(|t| t.unsqueeze_dim(dim)).collect();
        Tensor::<B, D2, K>::cat(tensors, dim)
    }

    /// Iterate over slices of tensors alongside a given dimension.
    ///
    /// # Panics
    ///
    /// Given dimension is less than tensor rank.
    ///
    /// # Returns
    ///
    /// A tensor iterator.
    pub fn iter_dim(self, dim: usize) -> DimIter<B, D, K> {
        check!(TensorCheck::dim_ops::<D>("iter_dim", dim));
        DimIter::new(self, dim)
    }

    /// Returns a new tensor with the given dimension narrowed to the given range.
    ///
    /// # Panics
    ///
    /// - If the dimension is greater than the number of dimensions of the tensor.
    /// - If the given range exceeds the number of elements on the given dimension.
    ///
    /// # Returns
    ///
    /// A new tensor with the given dimension narrowed to the given range.
    pub fn narrow(self, dim: usize, start: usize, length: usize) -> Self {
        check!(TensorCheck::dim_ops::<D>("narrow", dim));
        check!(TensorCheck::narrow(&self, dim, start, length));
        Self::new(narrow::<B, D, K>(self.primitive, dim, start, length))
    }

    /// Attempts to split the tensor along the given dimension into chunks.
    /// May return less chunks than requested if the tensor size is not divisible by the number of chunks.
    ///
    /// When the given dimension is evenly divisible by the number of chunks, the chunks will be of equal size.
    /// Otherwise all chunks will be of equal size except for the last one.
    ///
    /// # Panics
    ///
    ///  If the dimension is greater than the number of dimensions of the tensor.
    ///
    /// # Returns
    /// A vector of tensors.
    pub fn chunk(self, chunks: usize, dim: usize) -> Vec<Self> {
        check!(TensorCheck::dim_ops::<D>("chunk", dim));
        chunk::<B, D, K>(self.primitive, chunks, dim)
            .into_iter()
            .map(|v| Self::new(v))
            .collect()
    }

    /// Tests if any element in the `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` containing a single element, True if any element in the input tensor
    /// evaluates to True, False otherwise.
    pub fn any(self) -> Tensor<B, 1, Bool> {
        K::any(self.primitive)
    }

    /// Tests if any element in the `tensor` evaluates to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if any element along this dim in the input
    /// evaluates to True, False otherwise.
    pub fn any_dim(self, dim: usize) -> Tensor<B, D, Bool> {
        K::any_dim(self.primitive, dim)
    }

    /// Tests if all elements in the `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` with a single element, True if all elements in the input tensor
    /// evaluate to True, False otherwise.
    pub fn all(self) -> Tensor<B, 1, Bool> {
        K::all(self.primitive)
    }

    /// Tests if all elements in the `tensor` evaluate to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if all elements along this dim in the input
    /// evaluates to True, False otherwise.
    pub fn all_dim(self, dim: usize) -> Tensor<B, D, Bool> {
        K::all_dim(self.primitive, dim)
    }

    /// Convert the tensor into a scalar.
    ///
    /// # Panics
    ///
    /// If the tensor doesn't have one element.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn into_scalar(self) -> K::Elem {
        check!(TensorCheck::into_scalar(&self.shape()));
        let data = self.into_data();
        data.value[0]
    }

    /// Convert the tensor into a scalar.
    ///
    /// # Panics
    ///
    /// If the tensor doesn't have one element.
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn into_scalar(self) -> K::Elem {
        check!(TensorCheck::into_scalar(&self.shape()));
        let data = self.into_data().await;
        data.value[0]
    }

    /// Broadcast the tensor to the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape to broadcast the tensor to.
    ///             Can contain -1 for dimensions that should be inferred.
    ///             The number of elements in the shape must be greater or equal as
    ///             the number of dimensions of the tensor.
    ///
    /// # Panics
    ///
    /// If the tensor cannot be broadcasted to the given shape.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape.
    pub fn expand<const D2: usize, S: BroadcastArgs<D, D2>>(self, shape: S) -> Tensor<B, D2, K> {
        let shape = shape.into_shape(&self.shape());
        check!(TensorCheck::expand("expand", &self.shape(), &shape,));

        Tensor::<B, D2, K>::new(K::expand(self.primitive, shape))
    }
}

/// Iterator given by (Tensor::iter_dim).
pub struct DimIter<B, const D: usize, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    start: usize,
    end: usize,
    dim: usize,
    ranges: [Range<usize>; D],
    tensor: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: BasicOps<B>> Iterator for DimIter<B, D, K> {
    type Item = Tensor<B, D, K>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        let mut ranges = self.ranges.clone();
        ranges[self.dim] = self.start..(self.start + 1);

        let slice = self.tensor.clone().slice(ranges);
        self.start += 1;

        Some(slice)
    }
}

impl<B: Backend, const D: usize, K: BasicOps<B>> DoubleEndedIterator for DimIter<B, D, K> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        let mut ranges = self.ranges.clone();
        ranges[self.dim] = (self.end - 1)..self.end;

        let slice = self.tensor.clone().slice(ranges);
        self.end = self.end.saturating_sub(1);

        Some(slice)
    }
}

impl<B: Backend, const D: usize, K: BasicOps<B>> DimIter<B, D, K> {
    fn new(tensor: Tensor<B, D, K>, dim: usize) -> Self {
        let dims = tensor.dims();
        let ranges = dims
            .iter()
            .map(|&dim| 0..dim)
            .collect::<Vec<Range<usize>>>();
        let ranges: [Range<usize>; D] = ranges.try_into().unwrap();
        Self {
            end: dims[dim],
            ranges,
            start: 0,
            dim,
            tensor,
        }
    }
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    <K as BasicOps<B>>::Elem: Debug,
{
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    #[inline]
    fn push_newline_indent(acc: &mut String, indent: usize) {
        acc.push('\n');
        for _ in 0..indent {
            acc.push(' ');
        }
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn fmt_inner_tensor(
        &self,
        acc: &mut String,
        depth: usize,
        multi_index: &mut [usize],
        range: (usize, usize),
    ) {
        let (start, end) = range;
        for i in start..end {
            if i > 0 {
                acc.push_str(", ");
            }
            multi_index[depth] = i;
            let range: [core::ops::Range<usize>; D] =
                core::array::from_fn(|i| multi_index[i]..multi_index[i] + 1);

            let elem = &self.clone().slice(range).into_data().value[0];
            acc.push_str(&format!("{elem:?}"));
        }
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn fmt_outer_tensor(
        &self,
        acc: &mut String,
        depth: usize,
        multi_index: &mut [usize],
        print_options: &PrintOptions,
        summarize: bool,
        range: (usize, usize),
    ) {
        let (start, end) = range;
        for i in start..end {
            if i > start {
                acc.push(',');
                Self::push_newline_indent(acc, depth + 1);
            }
            acc.push('[');
            multi_index[depth] = i;
            self.display_recursive(acc, depth + 1, multi_index, print_options, summarize);
            acc.push(']');
        }
    }

    /// Recursively formats the tensor data for display and appends it to the provided accumulator string.
    ///
    /// This function is designed to work with tensors of any dimensionality.
    /// It traverses the tensor dimensions recursively, converting the elements
    /// to strings and appending them to the accumulator string with the
    /// appropriate formatting.
    ///
    /// # Arguments
    ///
    /// * `acc` - A mutable reference to a `String` used as an accumulator for the formatted output.
    /// * `depth` - The current depth of the tensor dimensions being processed.
    /// * `multi_index` - A mutable slice of `usize` representing the current indices in each dimension.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn display_recursive(
        &self,
        acc: &mut String,
        depth: usize,
        multi_index: &mut [usize],
        print_options: &PrintOptions,
        summarize: bool,
    ) {
        let edge_items = print_options.edge_items;

        if depth == 0 {
            acc.push('[');
        }

        if depth == self.dims().len() - 1 {
            // if we are at the innermost dimension, just push its elements into the accumulator
            if summarize && self.dims()[depth] > 2 * edge_items {
                // print the starting `edge_items` elements
                self.fmt_inner_tensor(acc, depth, multi_index, (0, edge_items));
                acc.push_str(", ...");
                // print the last `edge_items` elements
                self.fmt_inner_tensor(
                    acc,
                    depth,
                    multi_index,
                    (self.dims()[depth] - edge_items, self.dims()[depth]),
                );
            } else {
                // print all the elements
                self.fmt_inner_tensor(acc, depth, multi_index, (0, self.dims()[depth]));
            }
        } else {
            // otherwise, iterate through the current dimension and recursively display the inner tensors
            if summarize && self.dims()[depth] > 2 * edge_items {
                self.fmt_outer_tensor(
                    acc,
                    depth,
                    multi_index,
                    print_options,
                    summarize,
                    (0, edge_items),
                );

                acc.push(',');
                Self::push_newline_indent(acc, depth + 1);
                acc.push_str("...");
                Self::push_newline_indent(acc, depth + 1);

                self.fmt_outer_tensor(
                    acc,
                    depth,
                    multi_index,
                    print_options,
                    summarize,
                    (self.dims()[depth] - edge_items, self.dims()[depth]),
                );
            } else {
                self.fmt_outer_tensor(
                    acc,
                    depth,
                    multi_index,
                    print_options,
                    summarize,
                    (0, self.dims()[depth]),
                );
            }
        }

        if depth == 0 {
            acc.push(']');
        }
    }
}

/// Options for Tensor pretty printing
pub struct PrintOptions {
    /// number of elements to start summarizing tensor
    pub threshold: usize,
    /// number of starting elements and ending elements to display
    pub edge_items: usize,
}

static PRINT_OPTS: Mutex<PrintOptions> = Mutex::new(PrintOptions::const_default());

impl PrintOptions {
    // We cannot use the default trait as it's not const.
    const fn const_default() -> Self {
        Self {
            threshold: 1000,
            edge_items: 3,
        }
    }
}

/// Set print options
pub fn set_print_options(options: PrintOptions) {
    *PRINT_OPTS.lock().unwrap() = options
}

/// Pretty print tensors
impl<B, const D: usize, K> core::fmt::Display for Tensor<B, D, K>
where
    B: Backend,
    B::IntElem: core::fmt::Display,
    K: BasicOps<B>,
    <K as BasicOps<B>>::Elem: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Tensor {{")?;

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        {
            let po = PRINT_OPTS.lock().unwrap();
            let mut acc = String::new();
            let mut multi_index = vec![0; D];
            let summarize = self.shape().num_elements() > po.threshold;

            self.display_recursive(&mut acc, 0, &mut multi_index, &po, summarize);

            writeln!(f, "  data:")?;
            write!(f, "{acc}")?;
            writeln!(f, ",")?;
        }

        writeln!(f, "  shape:  {:?},", self.dims())?;
        writeln!(f, "  device:  {:?},", self.device())?;
        writeln!(f, "  backend:  {:?},", B::name())?;
        writeln!(f, "  kind:  {:?},", K::name())?;
        writeln!(f, "  dtype:  {:?},", K::elem_type_name())?;
        write!(f, "}}")
    }
}

/// Transpose marker (zero-size type). Used to sugar the transpose of a tensor, e.g.
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::{Tensor, T};
///
/// fn example<B: Backend>() {
///     let device = Default::default();
///     let tensor = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
///     let transposed = tensor^T;
/// }
/// ```
pub struct T;

impl<B: Backend, const D: usize> core::ops::BitXor<T> for Tensor<B, D> {
    type Output = Self;
    fn bitxor(self, _: T) -> Self::Output {
        self.transpose()
    }
}

/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicOps<B: Backend>: TensorKind<B> {
    /// The type of the tensor elements.
    type Elem: Element;

    /// Creates an empty tensor with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The empty tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating empty tensors, users should prefer the [Tensor::empty](Tensor::empty) function,
    /// which is more high-level and designed for public use.
    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D>;

    /// Returns the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the shape of a tensor, users should prefer the [Tensor::shape](Tensor::shape) function,
    /// which is more high-level and designed for public use.
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D>;

    /// Reshapes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// The reshaped tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For reshaping a tensor, users should prefer the [Tensor::reshape](Tensor::reshape) function,
    /// which is more high-level and designed for public use.
    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2>;

    /// Transposes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Swaps two dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D>;

    /// Flips the tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to flip.
    /// * `axes` - The axes to flip the tensor along.
    ///
    /// # Returns
    ///
    /// The tensor with the axes flipped.
    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D>;

    ///  Select tensor elements corresponding for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges of the elements to select.
    ///
    /// # Returns
    ///
    /// The selected elements.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements of a tensor, users should prefer the [Tensor::slice](Tensor::slice) function,
    /// which is more high-level and designed for public use.
    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        range: [Range<usize>; D2],
    ) -> Self::Primitive<D1>;

    ///  Assigns the given value to the tensor elements corresponding for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges of the elements to select.
    /// * `value` - The value to assign.
    ///
    /// # Returns
    ///
    /// The tensor with the assigned values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For assigning values to elements of a tensor, users should prefer the [Tensor::slice_assign](Tensor::slice_assign) function,
    /// which is more high-level and designed for public use.
    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1>;

    /// Returns the device on which the tensor is allocated.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device on which the tensor is allocated.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the device of a tensor, users should prefer the [Tensor::device](Tensor::device) function,
    /// which is more high-level and designed for public use.
    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> B::Device;

    /// Moves the tensor to the given device.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `device` - The device on which the tensor will be moved.
    ///
    /// # Returns
    ///
    /// The tensor on the given device.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For moving a tensor to a device, users should prefer the [Tensor::to_device](Tensor::to_device) function,
    /// which is more high-level and designed for public use.
    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &B::Device,
    ) -> Self::Primitive<D>;

    /// Extracts the data from the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For extracting the data of a tensor, users should prefer the [Tensor::into_data](Tensor::into_data) function,
    /// which is more high-level and designed for public use.
    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Reader<Data<Self::Elem, D>>;

    /// Creates a tensor from the given data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor from data, users should prefer the [Tensor::from_data](Tensor::from_data) function,
    /// which is more high-level and designed for public use.
    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D>;

    /// Repeat the tensor along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension along which the tensor will be repeated.
    /// * `times` - The number of times the tensor will be repeated.
    ///
    /// # Returns
    ///
    /// The repeated tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For repeating a tensor, users should prefer the [Tensor::repeat](Tensor::repeat) function,
    /// which is more high-level and designed for public use.
    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D>;

    /// Concatenates the given tensors along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `vectors` - The tensors to concatenate.
    /// * `dim` - The dimension along which the tensors will be concatenated.
    ///
    /// # Returns
    ///
    /// The concatenated tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For concatenating tensors, users should prefer the [Tensor::cat](Tensor::cat) function,
    /// which is more high-level and designed for public use.
    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D>;

    /// Equates the given tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The tensor of booleans indicating whether the corresponding elements are equal.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For equating tensors, users should prefer the [Tensor::equal](Tensor::equal) function,
    /// which is more high-level and designed for public use.
    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;

    /// Applies element-wise non-equality comparison between the given tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The tensor of booleans indicating whether the corresponding elements are equal.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For non-equality comparison of tensors, users should prefer the [Tensor::not_equal](Tensor::not_equal)
    /// function, which is more high-level and designed for public use.
    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;

    /// Returns the name of the element type.
    fn elem_type_name() -> &'static str {
        core::any::type_name::<Self::Elem>()
    }

    /// Tests if any element in the `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the input tensor evaluates to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the [Tensor::any](Tensor::any) function
    /// which is more high-level and designed for public use.
    fn any<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool>;

    /// Tests if any element in the tensor evaluates to True along a given dimension dim.
    ///
    /// # Arguments
    ///
    /// * tensor - The tensor to test.
    /// * dim - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same size as input tensor, except in the dim axis where the size is 1.
    /// Returns True if any element in the input tensor along the given dimension evaluates to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the [Tensor::any_dim](Tensor::any_dim) function,
    /// which is more high-level and designed for public use.
    fn any_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool>;

    /// Tests if all elements in the `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if all elements in the input tensor evaluates to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the [Tensor::all](Tensor::all) function,
    /// which is more high-level and designed for public use.
    fn all<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool>;

    /// Tests if all elements in the `tensor` evaluate to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same size as input `tensor`, except in the `dim` axis where the size is 1.
    /// Returns True if all elements in the input tensor along the given dimension evaluate to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the [Tensor::all_dim](Tensor::all_dim) function,
    /// which is more high-level and designed for public use.
    fn all_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool>;

    /// Broadcasts the given tensor to the specified shape.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to broadcast.
    /// * `shape` - The shape to broadcast to.
    ///
    /// # Returns
    ///
    /// The broadcasted tensor.
    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2>;
}

impl<B: Backend> BasicOps<B> for Float {
    type Elem = B::FloatElem;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::float_empty(shape, device)
    }

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::float_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::float_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::float_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        check!(TensorCheck::swap_dims::<D>(dim1, dim2));
        B::float_swap_dims(tensor, dim1, dim2)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::float_slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::float_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::float_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::float_to_device(tensor, device)
    }

    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Reader<Data<Self::Elem, D>> {
        B::float_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::float_from_data(data, device)
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        B::float_repeat(tensor, dim, times)
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::float_cat(vectors, dim)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_not_equal(lhs, rhs))
    }

    fn any<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool> {
        Tensor::new(B::float_any(tensor))
    }

    fn any_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_any_dim(tensor, dim))
    }

    fn all<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool> {
        Tensor::new(B::float_all(tensor))
    }

    fn all_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_all_dim(tensor, dim))
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        B::float_permute(tensor, axes)
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::float_expand(tensor, shape)
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        B::float_flip(tensor, axes)
    }
}

impl<B: Backend> BasicOps<B> for Int {
    type Elem = B::IntElem;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::int_empty(shape, device)
    }
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::int_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::int_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::int_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        check!(TensorCheck::swap_dims::<D>(dim1, dim2));
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::int_slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::int_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::int_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::int_to_device(tensor, device)
    }

    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Reader<Data<Self::Elem, D>> {
        B::int_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::int_from_data(data, device)
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        B::int_repeat(tensor, dim, times)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_not_equal(lhs, rhs))
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::int_cat(vectors, dim)
    }

    fn any<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool> {
        Tensor::new(B::int_any(tensor))
    }

    fn any_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_any_dim(tensor, dim))
    }

    fn all<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool> {
        Tensor::new(B::int_all(tensor))
    }

    fn all_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_all_dim(tensor, dim))
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        B::int_permute(tensor, axes)
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::int_expand(tensor, shape)
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        B::int_flip(tensor, axes)
    }
}

impl<B: Backend> BasicOps<B> for Bool {
    type Elem = bool;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::bool_empty(shape, device)
    }
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::bool_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::bool_reshape(tensor, shape)
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::bool_transpose(tensor)
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        check!(TensorCheck::swap_dims::<D>(dim1, dim2));
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::bool_slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::bool_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::bool_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::bool_to_device(tensor, device)
    }

    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Reader<Data<Self::Elem, D>> {
        B::bool_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::bool_from_data(data, device)
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        B::bool_repeat(tensor, dim, times)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::bool_equal(lhs, rhs))
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::bool_not_equal(lhs, rhs))
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::bool_cat(vectors, dim)
    }

    fn any<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool> {
        Tensor::new(B::bool_any(tensor))
    }

    fn any_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(B::bool_any_dim(tensor, dim))
    }

    fn all<const D: usize>(tensor: Self::Primitive<D>) -> Tensor<B, 1, Bool> {
        Tensor::new(B::bool_all(tensor))
    }

    fn all_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(B::bool_all_dim(tensor, dim))
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        B::bool_permute(tensor, axes)
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::bool_expand(tensor, shape)
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        B::bool_flip(tensor, axes)
    }
}

/// Trait used for movedim arguments
pub trait MovedimArgs {
    /// Converts into a set of dimensions `Vec<usize>` for the `tensor.movedim()` function
    fn into_dim_vec<const D: usize>(self) -> Vec<usize>;
}

impl MovedimArgs for Vec<i32> {
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        let set = self
            .iter()
            .map(|&dim| {
                if dim < 0 {
                    (D as i32 + dim) as usize
                } else {
                    dim as usize
                }
            })
            .collect::<Vec<usize>>();
        check!(TensorCheck::movedim_args_vec::<D>(&set));

        set
    }
}

impl MovedimArgs for Vec<usize> {
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        check!(TensorCheck::movedim_args_vec::<D>(&self));
        self
    }
}

impl MovedimArgs for usize {
    #[allow(clippy::vec_init_then_push)]
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        check!(TensorCheck::movedim_args_usize::<D>(self));

        let mut set = Vec::with_capacity(1);
        set.push(self);

        set
    }
}

impl MovedimArgs for i32 {
    #[allow(clippy::vec_init_then_push)]
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        check!(TensorCheck::movedim_args_i32::<D>(self));

        let dim = if self < 0 {
            (D as i32 + self) as usize
        } else {
            self as usize
        };

        let mut set = Vec::with_capacity(1);
        set.push(dim);

        set
    }
}

/// Trait used for reshape arguments.
pub trait ReshapeArgs<const D2: usize> {
    /// Converts to a shape.
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2>;
}

impl<const D2: usize> ReshapeArgs<D2> for Shape<D2> {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2> {
        check!(TensorCheck::reshape_args_usize(&tensor.shape(), &self));

        self
    }
}
impl<const D2: usize> ReshapeArgs<D2> for [usize; D2] {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2> {
        let shape = Shape::from(self);

        check!(TensorCheck::reshape_args_usize(&tensor.shape(), &shape));

        shape
    }
}

impl<const D2: usize> ReshapeArgs<D2> for [i32; D2] {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2> {
        // Validate the reshape arguments
        check!(TensorCheck::reshape_args_i32(&self));

        // Temporary shape
        let mut new_shape: [i32; D2] = [1; D2];

        // We need to find the index of the 0 dimension and
        // replace it with the actual dimension value.
        for (i, &s) in self.iter().enumerate() {
            if s != 0 {
                new_shape[i] = s;
            } else {
                new_shape[i] = tensor.dims()[i] as i32;
            }
        }

        // Find the index of the inferred dimension (-1)
        let infer_index = new_shape.iter().position(|x| x == &-1);

        // Handle the case where the dimension is inferred (via -1)
        if let Some(index) = infer_index {
            // Handle the case where the dimension is inferred
            let mut product = 1;
            for (i, &s) in new_shape.iter().enumerate() {
                if i != index {
                    product *= s;
                }
            }
            let product_current = tensor.shape().num_elements() as i32;

            new_shape[index] = product_current / product;

            // Check if the reshape is valid
            if product_current % product != 0 {
                panic!(
                    "Cannot reshape tensor of shape {:?} to shape {:?}",
                    tensor.shape(),
                    new_shape
                );
            }
        };

        // Convert each element to usize
        let new_shape: [usize; D2] = new_shape.map(|x| x as usize);

        Shape::from(new_shape)
    }
}

/// Trait used for broadcast arguments.
pub trait BroadcastArgs<const D1: usize, const D2: usize> {
    /// Converts to a shape.
    fn into_shape(self, shape: &Shape<D1>) -> Shape<D2>;
}

impl<const D1: usize, const D2: usize> BroadcastArgs<D1, D2> for Shape<D2> {
    fn into_shape(self, _shape: &Shape<D1>) -> Shape<D2> {
        self
    }
}
impl<const D1: usize, const D2: usize> BroadcastArgs<D1, D2> for [usize; D2] {
    fn into_shape(self, _shape: &Shape<D1>) -> Shape<D2> {
        Shape::from(self)
    }
}

impl<const D1: usize, const D2: usize> BroadcastArgs<D1, D2> for [i32; D2] {
    // Passing -1 as the size for a dimension means not changing the size of that dimension.
    fn into_shape(self, shape: &Shape<D1>) -> Shape<D2> {
        if self.len() < shape.dims.len() {
            panic!("Broadcast arguments must be greater than the number of dimensions");
        }

        if self.iter().any(|&x| x < -1 || x == 0) {
            panic!("Broadcast arguments must be positive or -1");
        }

        // Zip the two shapes in reverse order and replace -1 with the actual dimension value.
        let new_shape: Vec<_> = self
            .iter()
            .rev()
            .zip(shape.dims.iter().rev().chain(repeat(&0)).take(self.len())) // Pad the original shape with 0s
            .map(|(&x, &y)| if x == -1 { y } else { x as usize })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        if new_shape.iter().any(|&x| x == 0) {
            panic!("Cannot substitute -1 for a non-existing dimension");
        }

        let new_shape: [usize; D2] = new_shape.try_into().unwrap();

        Shape::from(new_shape)
    }
}

#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
impl<B, const D: usize, K> Serialize for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Debug + Copy + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let data = self.to_data();
        // manually construct instead of calling `serialize` to move instead of clone value
        let serialized: DataSerialize<K::Elem> = DataSerialize {
            value: data.value,
            shape: data.shape.dims.to_vec(),
        };
        serialized.serialize(serializer)
    }
}

impl<'de, B, const D: usize, K> Deserialize<'de> for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Debug + Copy + Deserialize<'de>,
{
    fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let data_res: Result<DataSerialize<K::Elem>, De::Error> =
            DataSerialize::deserialize(deserializer);
        let tensor = Tensor::from_data(data_res?, &<B::Device as Default>::default());
        Ok(tensor)
    }
}
