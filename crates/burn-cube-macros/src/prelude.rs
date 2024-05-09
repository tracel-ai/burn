use std::collections::HashSet;

use crate::VariableKey;

pub(crate) fn get_prelude(needed_functions: &HashSet<VariableKey>) -> proc_macro2::TokenStream {
    let mut prelude = quote::quote! {
        use super::*;
    };

    for func_name in needed_functions
        .iter()
        .map(|variable| variable.name.as_str())
    {
        let func_code = match func_name {
            "float_new" => Some(codegen_float_new()),
            "int_new" => Some(codegen_int_new()),
            "uint_new" => Some(codegen_uint_new()),
            "bool_new" => Some(codegen_bool_new()),
            "to_int" => Some(codegen_to_int()),
            "to_float" => Some(codegen_to_float()),
            "to_uint" => Some(codegen_to_uint()),
            "to_bool" => Some(codegen_to_bool()),
            _ => None,
        };

        if func_code.is_some() {
            prelude.extend(func_code);
        }
    }

    prelude
}

fn codegen_float_new() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn float_new<F: burn_cube::Float>(val: f32) -> F {
            F::new(val, 1)
        }

        pub fn float_new_expand<F: burn_cube::Float>(
            context: &mut CubeContext,
            val: f32,
        ) -> <F as burn_cube::RuntimeType>::ExpandType {
            val.into()
        }
    }
}

fn codegen_int_new() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn int_new<I: burn_cube::Int>(val: i32) -> I {
            I::new(val, 1)
        }

        pub fn int_new_expand<I: burn_cube::Int>(
            context: &mut CubeContext,
            val: i32,
        ) -> <I as burn_cube::RuntimeType>::ExpandType {
            val.into()
        }
    }
}

fn codegen_uint_new() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn uint_new(val: u32) -> UInt {
            UInt {
                val,
                vectorization: 1,
            }
        }
        pub fn uint_new_expand(
            context: &mut CubeContext,
            val: u32,
        ) -> <UInt as burn_cube::RuntimeType>::ExpandType {
            val.into()
        }
    }
}

fn codegen_bool_new() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn bool_new(val: bool) -> Bool{
            Bool {
                val,
                vectorization: 1,
            }
        }
        pub fn bool_new_expand(
            context: &mut CubeContext,
            val: bool,
        ) -> <Bool as burn_cube::RuntimeType>::ExpandType {
            val.into()
        }
    }
}

fn codegen_to_int() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn to_int<R: burn_cube::RuntimeType, I: Int>(input: R) -> I {
            I::new(0, 1)
        }
        pub fn to_int_expand<R: burn_cube::RuntimeType, I: Int>(
            context: &mut CubeContext,
            val: burn_cube::ExpandElement,
        ) -> <I as burn_cube::RuntimeType>::ExpandType {
            let elem = Elem::Int(I::into_kind());
            let new_var = context.create_local(match val.item() {
                Item::Vec4(_) => Item::Vec4(elem),
                Item::Vec3(_) => Item::Vec3(elem),
                Item::Vec2(_) => Item::Vec2(elem),
                Item::Scalar(_) => Item::Scalar(elem),
            });
            burn_cube::assign::expand(context, val.into(), new_var.clone());
            new_var
        }
    }
}

fn codegen_to_float() -> proc_macro2::TokenStream {
    // R: type we come from
    // F: kind of float we want as output
    quote::quote! {
        pub fn to_float<R: burn_cube::RuntimeType, F: Float>(input: R) -> F {
            // TODO: make val and vectorization accessible through trait
            F::new(0., 1)
        }
        pub fn to_float_expand<R: burn_cube::RuntimeType, F: Float>(
            context: &mut CubeContext,
            val: burn_cube::ExpandElement,
        ) -> burn_cube::ExpandElement {
            let elem = Elem::Float(F::into_kind());
            let new_var = context.create_local(match val.item() {
                Item::Vec4(_) => Item::Vec4(elem),
                Item::Vec3(_) => Item::Vec3(elem),
                Item::Vec2(_) => Item::Vec2(elem),
                Item::Scalar(_) => Item::Scalar(elem),
            });
            burn_cube::assign::expand(context, val.into(), new_var.clone());
            new_var
        }
    }
}

fn codegen_to_uint() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn to_uint<R: burn_cube::RuntimeType>(input: R) -> UInt {
            UInt {
                val: 0,
                vectorization: 1,
            }
        }
        pub fn to_uint_expand(
            context: &mut CubeContext,
            val: burn_cube::ExpandElement,
        ) -> <UInt as burn_cube::RuntimeType>::ExpandType {
            let elem = Elem::UInt;
            let new_var = context.create_local(match val.item() {
                Item::Vec4(_) => Item::Vec4(elem),
                Item::Vec3(_) => Item::Vec3(elem),
                Item::Vec2(_) => Item::Vec2(elem),
                Item::Scalar(_) => Item::Scalar(elem),
            });
            burn_cube::assign::expand(context, val.into(), new_var.clone());
            new_var
        }
    }
}

fn codegen_to_bool() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn to_bool<R: burn_cube::RuntimeType>(input: R) -> Bool {
            Bool {
                val: true,
                vectorization: 1,
            }
        }
        pub fn to_bool_expand(
            context: &mut CubeContext,
            val: burn_cube::ExpandElement,
        ) -> <UInt as burn_cube::RuntimeType>::ExpandType {
            let elem = Elem::Bool;
            let new_var = context.create_local(match val.item() {
                Item::Vec4(_) => Item::Vec4(elem),
                Item::Vec3(_) => Item::Vec3(elem),
                Item::Vec2(_) => Item::Vec2(elem),
                Item::Scalar(_) => Item::Scalar(elem),
            });
            burn_cube::assign::expand(context, val.into(), new_var.clone());
            new_var
        }
    }
}
