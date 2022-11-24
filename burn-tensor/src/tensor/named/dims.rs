use crate::backend::Backend;
use crate::Tensor;

pub trait Dim {
    fn to_string() -> String;
}

pub trait NamedDims<B: Backend> {
    type Tensor;
    fn to_string() -> String;
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
