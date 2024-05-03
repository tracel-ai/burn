use std::collections::HashSet;

use crate::VariableKey;

pub(crate) fn get_prelude(needed_functions: &HashSet<VariableKey>) -> proc_macro2::TokenStream {
    let mut prelude = proc_macro2::TokenStream::new();

    for func_name in needed_functions
        .iter()
        .map(|variable| variable.name.as_str())
    {
        let func_code = match func_name {
            "float_new" => Some(codegen_float_new()),
            "int_new" => Some(codegen_int_new()),
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
        pub fn float_new(val: f32) -> Float {
            Float {
                val,
                vectorization: 1,
            }
        }
        pub fn float_new_expand(
            context: &mut CubeContext,
            val: f32,
        ) -> <Float as burn_cube::RuntimeType>::ExpandType {
            // TODO: 0. becomes 0..into()
            val.into()
        }
    }
}

fn codegen_int_new() -> proc_macro2::TokenStream {
    quote::quote! {
        pub fn int_new(val: i32) -> Int {
            Int {
                val,
                vectorization: 1,
            }
        }
        pub fn int_new_expand(
            context: &mut CubeContext,
            val: i32,
        ) -> <Int as burn_cube::RuntimeType>::ExpandType {
            val.into()
        }
    }
}
