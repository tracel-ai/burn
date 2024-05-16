use burn_cube::{CubeContext, PrimitiveVariable, F32};
use burn_jit::gpu::Item;

type ElemType = F32;

mod elsewhere {
    use burn_cube::{cube, Float};

    #[cube]
    pub fn my_func<F: Float>(x: F) -> F {
        x * F::lit(2)
    }
}

mod here {
    use burn_cube::{cube, Float};

    use crate::elsewhere;

    #[cube]
    pub fn caller<F: Float>(x: F) {
        let _ = x + elsewhere::my_func::<F>(x);
    }

    #[cube]
    pub fn no_call_ref<F: Float>(x: F) {
        let _ = x + x * F::lit(2);
    }
}

#[test]
fn cube_call_equivalent_to_no_call_no_arg_test() {
    let mut caller_context = CubeContext::root();
    let x = caller_context.create_local(Item::Scalar(ElemType::into_elem()));
    here::caller_expand::<ElemType>(&mut caller_context, x);
    let caller_scope = caller_context.into_scope();

    let mut no_call_context = CubeContext::root();
    let x = no_call_context.create_local(Item::Scalar(ElemType::into_elem()));
    here::no_call_ref_expand::<ElemType>(&mut no_call_context, x);
    let no_call_scope = no_call_context.into_scope();

    assert_eq!(
        format!("{:?}", caller_scope.operations),
        format!("{:?}", no_call_scope.operations)
    );
}
