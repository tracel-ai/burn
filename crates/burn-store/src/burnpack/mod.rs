pub mod base;
pub mod reader;
pub mod store;
pub mod writer;

#[cfg(test)]
mod tests {
    use crate::TensorSnapshot;

    mod edge_cases;
    mod header;
    mod helpers;
    mod reader;
    mod round_trip;
    mod writer;
}
