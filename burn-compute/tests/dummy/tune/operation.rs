use burn_compute::tune::{InputHashable, Operation};
use derive_new::new;

#[derive(new, PartialEq, Eq, Hash)]
pub struct ArrayHashable {
    pub sizes: [usize; 3],
}

impl InputHashable for ArrayHashable {
    fn custom_hash(&self) -> String {
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
    type Input = ArrayHashable;
}

#[derive(PartialEq, Eq, Hash)]
pub struct MultiplicationOp {}
impl Operation for MultiplicationOp {
    type Input = ArrayHashable;
}

#[derive(PartialEq, Eq, Hash)]
pub struct CacheTestOp {}
impl Operation for CacheTestOp {
    type Input = ArrayHashable;
}
