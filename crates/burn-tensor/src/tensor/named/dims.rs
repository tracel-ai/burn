use alloc::format;
use alloc::string::String;

use crate::Tensor;
use crate::backend::Backend;

/// Dimension trait.
pub trait Dim: core::fmt::Debug {
    /// Converts the dimension to a string.
    fn to_string() -> String;
}

/// Named dimensions trait.
pub trait NamedDims<B: Backend>: core::fmt::Debug {
    /// Tensor type.
    type Tensor;

    /// Converts the named dimensions to a string.
    fn to_string() -> String;
}

/// Named dimension macro.
#[macro_export]
macro_rules! NamedDim {
    ($name:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name;
        impl Dim for $name {
            fn to_string() -> String {
                stringify!($name).to_string()
            }
        }
    };
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
