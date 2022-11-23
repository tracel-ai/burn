use crate::backend::Backend;
use crate::{Distribution, Shape, Tensor};

pub trait Dim {
    fn to_string() -> String;
}
pub trait NamedDims<B: Backend> {
    type Tensor;
    fn to_string() -> String;
}

impl<B: Backend, D1> NamedDims<B> for (D1,)
where
    B: Backend,
    D1: Dim,
{
    type Tensor = Tensor<B, 1>;
    fn to_string() -> String {
        format!("[{}]", D1::to_string())
    }
}

impl<B: Backend, D1, D2> NamedDims<B> for (D1, D2)
where
    B: Backend,
    D1: Dim,
    D2: Dim,
{
    type Tensor = Tensor<B, 2>;
    fn to_string() -> String {
        format!("[{}, {}]", D1::to_string(), D2::to_string())
    }
}

impl<B: Backend, D1, D2, D3> NamedDims<B> for (D1, D2, D3)
where
    B: Backend,
    D1: Dim,
    D2: Dim,
    D3: Dim,
{
    type Tensor = Tensor<B, 3>;
    fn to_string() -> String {
        format!(
            "[{}, {}, {}]",
            D1::to_string(),
            D2::to_string(),
            D3::to_string()
        )
    }
}

impl<B: Backend, D1, D2, D3, D4> NamedDims<B> for (D1, D2, D3, D4)
where
    B: Backend,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
{
    type Tensor = Tensor<B, 4>;
    fn to_string() -> String {
        format!(
            "[{}, {}, {}, {}]",
            D1::to_string(),
            D2::to_string(),
            D3::to_string(),
            D4::to_string()
        )
    }
}

pub struct NamedTensor<B: Backend, D: NamedDims<B>> {
    tensor: D::Tensor,
}

impl<B: Backend, const D: usize, ND: NamedDims<B>> std::fmt::Display for NamedTensor<B, ND>
where
    ND: NamedDims<B, Tensor = Tensor<B, D>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!(
            "NamedTensor[shape={:?}, dims={}]",
            self.shape().dims,
            ND::to_string(),
        ))
    }
}

#[macro_export]
macro_rules! NamedDim {
    ($name:ident) => {
        pub struct $name;
        impl Dim for $name {
            fn to_string() -> String {
                stringify!($name).to_string()
            }
        }
    };
}

pub trait Permut<N, const D1: usize, const D2: usize> {
    fn permut(&self) -> N;
}

impl<B: Backend, const D: usize, ND> NamedTensor<B, ND>
where
    ND: NamedDims<B, Tensor = Tensor<B, D>>,
{
    fn from_tensor(tensor: ND::Tensor) -> Self {
        Self { tensor }
    }

    pub fn random<S: Into<Shape<D>>>(shape: S, distribution: Distribution<B::Elem>) -> Self {
        Self::from_tensor(Tensor::random(shape, distribution))
    }

    pub fn shape(&self) -> &Shape<D> {
        self.tensor.shape()
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        Self::from_tensor(self.tensor.mul(&rhs.tensor))
    }

    pub fn reshape<const D2: usize, S, ND2>(&self, shape: S, _: ND2) -> NamedTensor<B, ND2>
    where
        S: Into<Shape<D2>>,
        ND2: NamedDims<B, Tensor = Tensor<B, D2>>,
    {
        NamedTensor::from_tensor(self.tensor.reshape(shape))
    }

    pub fn permut<ND2, const D1: usize, const D2: usize>(&self) -> NamedTensor<B, ND2>
    where
        ND2: NamedDims<B, Tensor = Tensor<B, D>>,
        Self: Permut<NamedTensor<B, ND2>, D1, D2>,
    {
        Permut::permut(self)
    }
}

macro_rules! generate_permut {
    (2 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim> Permut<NamedTensor<B, $output>, $dim1, $dim2>
            for NamedTensor<B, (D1, D2)>
        {
            fn permut(&self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };

    (3 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim, D3: Dim> Permut<NamedTensor<B, $output>, $dim1, $dim2>
            for NamedTensor<B, (D1, D2, D3)>
        {
            fn permut(&self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };

    (4 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim, D3: Dim, D4: Dim>
            Permut<NamedTensor<B, $output>, $dim1, $dim2> for NamedTensor<B, (D1, D2, D3, D4)>
        {
            fn permut(&self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };
}

generate_permut!(2 => (D2, D1), (0, 1));
generate_permut!(3 => (D2, D1, D3), (0, 1));
generate_permut!(3 => (D3, D2, D1), (0, 2));
generate_permut!(3 => (D1, D3, D2), (1, 2));
generate_permut!(4 => (D2, D1, D3, D4), (0, 1));
generate_permut!(4 => (D3, D2, D1, D4), (0, 2));
generate_permut!(4 => (D4, D2, D3, D1), (0, 3));
generate_permut!(4 => (D1, D3, D2, D4), (1, 2));
generate_permut!(4 => (D1, D4, D3, D2), (1, 3));

impl<B: Backend, Batch: Dim, X: Dim, Y: Dim> NamedTensor<B, (Batch, X, Y)> {
    pub fn matmul<Z: Dim>(
        &self,
        rhs: &NamedTensor<B, (Batch, Y, Z)>,
    ) -> NamedTensor<B, (Batch, X, Z)> {
        let tensor = self.tensor.matmul(&rhs.tensor);
        NamedTensor { tensor }
    }
}
