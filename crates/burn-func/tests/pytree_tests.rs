use burn_backend::{TensorData, TensorMetadata, ops::FloatTensorOps};
use burn_flex::{Flex, FlexDevice};
use burn_func::{FuncTensor, PyTree};

fn tensor(values: &[f32]) -> FuncTensor<Flex, 1> {
    FuncTensor::from_data(TensorData::from(values), &FlexDevice)
}

#[derive(Clone, Debug, PyTree)]
struct Derived<B: burn_func::burn_backend::Backend, const N: usize> {
    weights: FuncTensor<B, 1>,
    bias: FuncTensor<B, 1>,
    #[pytree(skip)]
    metadata: usize,
    #[pytree(skip)]
    label: String,
}

#[test]
fn tuple_and_array_traversal() {
    let tuple = (tensor(&[1.0]), tensor(&[2.0, 3.0]));
    assert_eq!(tuple.num_leaves(), 2);

    let array = [tensor(&[1.0]), tensor(&[2.0])];
    assert_eq!(array.num_leaves(), 2);

    let mut visited = Vec::new();
    tuple.visit_float(&mut |leaf| visited.push(leaf.shape().num_elements()));
    assert_eq!(visited, vec![1, 2]);
}

#[test]
fn vec_and_option_traversal() {
    let values = vec![tensor(&[1.0]), tensor(&[2.0]), tensor(&[3.0])];
    assert_eq!(values.num_leaves(), 3);

    let some = Some(tensor(&[1.0]));
    assert_eq!(some.num_leaves(), 1);
    let none: Option<FuncTensor<Flex, 1>> = None;
    assert_eq!(none.num_leaves(), 0);
}

#[test]
fn derived_struct_traversal_and_skipped_fields() {
    let params = Derived::<Flex, 1> {
        weights: tensor(&[1.0, 2.0]),
        bias: tensor(&[3.0]),
        metadata: 42,
        label: "linear".to_string(),
    };
    assert_eq!(params.num_leaves(), 2);

    let rebound = params.map_float::<Flex, _>(&mut |leaf| leaf.clone());
    assert_eq!(rebound.metadata, 42);
    assert_eq!(rebound.label, "linear");
    assert_eq!(rebound.weights.shape().num_elements(), 2);
}

#[test]
fn zip_float_pairs_leaves() {
    let lhs = (tensor(&[1.0, 2.0]), tensor(&[3.0]));
    let rhs = (tensor(&[10.0, 20.0]), tensor(&[30.0]));

    let sum =
        lhs.zip_float::<Flex, Flex, _>(&rhs, &mut |a, b| Flex::float_add(a.clone(), b.clone()));
    let values = sum.0.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(values, vec![11.0, 22.0]);
}

#[test]
fn primitive_scalars_have_no_leaves() {
    assert_eq!(PyTree::<Flex>::num_leaves(&42usize), 0);
    assert_eq!(PyTree::<Flex>::num_leaves(&3.5f64), 0);
    assert_eq!(PyTree::<Flex>::num_leaves(&true), 0);
    assert_eq!(PyTree::<Flex>::num_leaves(&"metadata".to_string()), 0);
}

#[test]
fn skipped_field_requires_b_independence() {
    // Compile-only assertion: the following struct should compile because `usize`
    // is B-independent. `Tensor<B, 1, Int>` in a skipped slot would NOT compile.
    #[derive(Clone, PyTree)]
    struct WithMetadata<B: burn_func::burn_backend::Backend> {
        w: FuncTensor<B, 1>,
        #[pytree(skip)]
        step: usize,
    }
    let _ = WithMetadata::<Flex> {
        w: tensor(&[1.0]),
        step: 5,
    };
}

mod unimported_scope {
    #[derive(Clone, burn_func::PyTree)]
    pub struct Isolated<B: burn_func::burn_backend::Backend> {
        pub w: burn_func::FuncTensor<B, 1>,
        pub b: burn_func::FuncTensor<B, 1>,
    }
}

#[test]
fn macro_hygiene_without_imported_trait() {
    let s = unimported_scope::Isolated::<Flex> {
        w: tensor(&[1.0]),
        b: tensor(&[2.0]),
    };
    assert_eq!(PyTree::<Flex>::num_leaves(&s), 2);
}
