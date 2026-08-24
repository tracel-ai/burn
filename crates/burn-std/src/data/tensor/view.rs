use core::ops::{Index, IndexMut};

use crate::Shape;
use crate::element::Element;
use crate::indexing::AsIndex;
use crate::tensor::{DType, ravel_index};

use super::{DataError, TensorData};

impl TensorData {
    /// Returns an [`Index`] view wrapper of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let view: TensorDataView<f64> = data.try_view().unwrap();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with the requested view.
    pub fn try_view<E: Element>(&self) -> Result<TensorDataView<'_, E>, DataError> {
        TensorDataView::<E>::try_view(self)
    }

    /// Returns a [`TensorDataView<E>`] of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let view: TensorDataView<f64> = data.view();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Returns
    /// The view.
    ///
    /// # Panics
    ///
    /// Panics if the view can't be created because storage access fails or the dtype, byte
    /// representation, or element count is incompatible with `E`.
    #[track_caller]
    pub fn view<E: Element>(&self) -> TensorDataView<'_, E> {
        self.try_view()
            .unwrap_or_else(|err| panic!("Failed to create TensorData view: {err}"))
    }

    /// Returns a [`TensorDataViewMut<E>`] of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let mut view: TensorDataViewMut<f64> = data.try_mut_view().unwrap();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    ///
    /// view[&[0, 0]] = 10.0;
    /// assert_eq!(view[&[0, 0]], 10.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with the requested view.
    pub fn try_mut_view<E: Element>(&mut self) -> Result<TensorDataViewMut<'_, E>, DataError> {
        TensorDataViewMut::<E>::try_mut_view(self)
    }

    /// Returns a [`TensorDataViewMut<E>`] of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let mut view: TensorDataViewMut<f64> = data.mut_view();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    ///
    /// view[&[0, 0]] = 10.0;
    /// assert_eq!(view[&[0, 0]], 10.0);
    /// ```
    ///
    /// # Returns
    /// The mut view.
    ///
    /// # Panics
    ///
    /// Panics if the view can't be created because storage access fails or the dtype, byte
    /// representation, or element count is incompatible with `E`.
    #[track_caller]
    pub fn mut_view<E: Element>(&mut self) -> TensorDataViewMut<'_, E> {
        self.try_mut_view()
            .unwrap_or_else(|err| panic!("Failed to create mutable TensorData view: {err}"))
    }
}

/// Typed [`Index`] view over a [`TensorData`].
///
/// Creating a view materializes lazy storage into host-accessible memory when necessary. It does
/// not perform dtype conversion.
///
/// # Example
/// ```rust,no_run
/// use burn_std::*;
///
/// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let view: TensorDataView<f64> = data.view();
///
/// assert_eq!(view.shape(), &data.shape);
/// assert_eq!(&view.dtype(), &data.dtype);
///
/// assert_eq!(view[&[0, 0]], 1.0);
/// assert_eq!(view[&[0, 1]], 2.0);
/// assert_eq!(view[&[1, 0]], 3.0);
/// assert_eq!(view[&[1, 1]], 4.0);
/// ```
#[derive(Debug)]
pub struct TensorDataView<'a, E: Element> {
    values: &'a [E],
    shape: &'a Shape,
    dtype: DType,
}

impl<'a, E: Element> TensorDataView<'a, E> {
    /// Creates a typed indexed view over `data`.
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let view: TensorDataView<f64> = data.try_view().unwrap();
    ///
    /// assert_eq!(view.shape(), &data.shape);
    /// assert_eq!(&view.dtype(), &data.dtype);
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with `E`.
    pub fn try_view(data: &'a TensorData) -> Result<TensorDataView<'a, E>, DataError> {
        let shape = &data.shape;
        let dtype = data.dtype;
        let expected = shape.num_elements();
        let values = data.as_slice::<E>()?;
        let actual = values.len();

        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }

        Ok(TensorDataView {
            values,
            shape,
            dtype,
        })
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &Shape {
        self.shape
    }

    /// Returns the dtype of the view.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Ravels the index via [`ravel_index`] and the view's shape.
    pub fn ravel_index<I: AsIndex>(&self, index: &[I]) -> usize {
        ravel_index(index, self.shape)
    }
}

impl<'a, I: AsIndex, E: Element> Index<&[I]> for TensorDataView<'a, E> {
    type Output = E;

    fn index(&self, index: &[I]) -> &Self::Output {
        let o = self.ravel_index(index);
        &self.values[o]
    }
}

/// Typed mutable [`IndexMut`] view over a [`TensorData`].
///
/// Creating a mutable view materializes lazy storage and performs copy-on-write when necessary.
/// It does not perform dtype conversion.
///
/// # Example
/// ```rust,no_run
/// use burn_std::*;
///
/// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let shape = data.shape.clone();
/// let dtype = data.dtype;
/// let mut view: TensorDataViewMut<f64> = data.mut_view();
///
/// assert_eq!(view.shape(), &shape);
/// assert_eq!(&view.dtype(), &dtype);
///
/// assert_eq!(view[&[0, 0]], 1.0);
/// assert_eq!(view[&[0, 1]], 2.0);
/// assert_eq!(view[&[1, 0]], 3.0);
/// assert_eq!(view[&[1, 1]], 4.0);
///
/// view[&[0, 0]] = 10.0;
/// assert_eq!(view[&[0, 0]], 10.0);
/// ```
#[derive(Debug)]
pub struct TensorDataViewMut<'a, E: Element> {
    values: &'a mut [E],
    // `as_mut_slice` borrows the entire `TensorData`, so the view can't also retain a reference to
    // its shape. Keep an owned copy until storage and metadata can be borrowed as disjoint fields.
    shape: Shape,
    dtype: DType,
}

impl<'a, E: Element> TensorDataViewMut<'a, E> {
    /// Creates a typed mutable indexed view over `data`.
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let dtype = data.dtype;
    ///
    /// let mut view: TensorDataViewMut<f64> =
    ///     TensorDataViewMut::try_mut_view(&mut data).unwrap();
    ///
    /// assert_eq!(view.shape(), &shape);
    /// assert_eq!(&view.dtype(), &dtype);
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    ///
    /// view[&[0, 0]] = 10.0;
    /// assert_eq!(view[&[0, 0]], 10.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with `E`.
    pub fn try_mut_view(data: &'a mut TensorData) -> Result<TensorDataViewMut<'a, E>, DataError> {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        let expected = shape.num_elements();
        let values = data.as_mut_slice::<E>()?;
        let actual = values.len();

        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }

        Ok(TensorDataViewMut {
            values,
            shape,
            dtype,
        })
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the dtype of the view.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Ravels the dims via [`ravel_index`] and the view's shape.
    pub fn ravel_index<I: AsIndex>(&self, index: &[I]) -> usize {
        ravel_index(index, &self.shape)
    }
}

impl<'a, I, E> Index<&[I]> for TensorDataViewMut<'a, E>
where
    I: AsIndex,
    E: Element,
{
    type Output = E;

    fn index(&self, index: &[I]) -> &Self::Output {
        let o = self.ravel_index::<I>(index);
        &self.values[o]
    }
}

impl<'a, I, E> IndexMut<&[I]> for TensorDataViewMut<'a, E>
where
    I: AsIndex,
    E: Element,
{
    fn index_mut(&mut self, index: &[I]) -> &mut Self::Output {
        let o = self.ravel_index::<I>(index);
        &mut self.values[o]
    }
}
