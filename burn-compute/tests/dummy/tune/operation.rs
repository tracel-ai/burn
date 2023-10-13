use burn_compute::tune::{HashableResources, Operation};
use derive_new::new;

#[derive(new, PartialEq, Eq, Hash)]
pub struct ArraysResource {
    pub sizes: [usize; 3],
}

impl HashableResources for ArraysResource {
    fn key(&self) -> String {
        let mut hash = String::new();
        for size in self.sizes {
            let exp = f32::ceil(f32::log2(size as f32)) as u32;
            hash.push_str(2_u32.pow(exp).to_string().as_str());
            hash.push_str(",");
        }
        hash
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct AdditionOp {}
impl Operation for AdditionOp {
    type Resources = ArraysResource;
}

#[derive(PartialEq, Eq, Hash)]
pub struct MultiplicationOp {}
impl Operation for MultiplicationOp {
    type Resources = ArraysResource;
}

#[derive(PartialEq, Eq, Hash)]
pub struct CacheTestOp {}
impl Operation for CacheTestOp {
    type Resources = ArraysResource;
}
