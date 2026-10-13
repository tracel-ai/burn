mod elementwise;
mod linear;
mod matmul;
mod reduce;

pub use elementwise::Linearity;
pub use linear::LinearRule;
pub use matmul::MatmulRule;
pub use reduce::{ReduceRule, Reduction};
