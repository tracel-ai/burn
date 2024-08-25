#[derive(Debug, Default, Clone)]
pub struct SparseCSR;

#[derive(Debug, Default, Clone)]
pub struct SparseCOO;

pub trait SparseRepresentation: Clone + Default + Send + Sync + 'static + core::fmt::Debug {
    fn name() -> String;
}

impl SparseRepresentation for SparseCOO {
    fn name() -> String {
        "SparseCOO".to_owned()
    }
}

impl SparseRepresentation for SparseCSR {
    fn name() -> String {
        "SparseCSR".to_owned()
    }
}
