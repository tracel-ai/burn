use burn_cube::{cube, range, range_expand, Array, CubeContext, Float, UInt};
use burn_jit::gpu::{Elem, Item};

// #[cube]
pub fn kernel(mut lhs: Array<Float>, rhs: Float, end: UInt, unroll: bool) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;

    for i in range(0usize, end, unroll) {
        lhs[i] = tmp2 + lhs[i];
    }
}

#[allow(unused_mut)]
pub fn kernel_expand(
    context: &mut burn_cube::CubeContext,
    mut lhs: <Array<Float> as burn_cube::RuntimeType>::ExpandType,
    rhs: <Float as burn_cube::RuntimeType>::ExpandType,
    end: <UInt as burn_cube::RuntimeType>::ExpandType,
    unroll: <bool as burn_cube::RuntimeType>::ExpandType,
) -> () {
    let tmp1 = {
        let _lhs = rhs.clone();
        let _rhs = rhs.clone();
        burn_cube::mul::expand(context, _lhs, _rhs)
    };
    let tmp2 = {
        let _lhs = tmp1;
        let _rhs = rhs;
        burn_cube::add::expand(context, _lhs, _rhs)
    };
    range_expand(context, 0usize.into(), end, unroll, |context, i| {
        {
            let _array = lhs.clone();
            let _index = i.clone();
            let _value = {
                let _lhs = tmp2.clone();
                let _rhs = {
                    let _array = lhs.clone();
                    let _index = i;
                    burn_cube::index::expand(context, _array, _index)
                };
                burn_cube::add::expand(context, _lhs, _rhs)
            };
            burn_cube::index_assign::expand(context, _array, _index, _value)
        };
    });
}

#[test]
fn test_simple_add() {
    let mut context = CubeContext::root();

    let lhs = context.create_local(Item::Scalar(Elem::Float));
    let rhs = context.create_local(Item::Scalar(Elem::Float));
    let end = context.create_local(Item::Scalar(Elem::UInt));

    kernel_expand(&mut context, lhs, rhs, end, false);
    let scope = context.into_scope();

    for op in scope.operations.iter() {
        println!("{op:?}");
    }

    panic!("nop");
}
