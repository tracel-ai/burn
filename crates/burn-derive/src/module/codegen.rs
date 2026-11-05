use super::{codegen_struct::ModuleFieldType, display, generics::module_generics};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Attribute;

/// Basic trait to be implemented for Module generation.
pub(crate) trait ModuleCodegen {
    fn gen_num_params(&self) -> TokenStream;
    fn gen_visit(&self) -> TokenStream;
    fn gen_collect_devices(&self) -> TokenStream;
    fn gen_to_device(&self) -> TokenStream;
    fn gen_fork(&self) -> TokenStream;
    fn gen_map(&self) -> TokenStream;
    fn gen_valid(&self) -> TokenStream;
    fn gen_train(&self) -> TokenStream;
    fn gen_materialize(&self) -> TokenStream;
    fn gen_clone(&self) -> TokenStream;

    fn gen_display(&self) -> TokenStream;

    fn fields(&self) -> Vec<(&syn::Field, &ModuleFieldType)>;
}

pub(crate) fn generate_module_standard<Codegen: ModuleCodegen>(
    ast: &syn::DeriveInput,
    codegen: Codegen,
) -> TokenStream {
    let name = &ast.ident;

    let generics = module_generics(ast, codegen.fields());

    let display_fn = display::display_fn(ast);
    let attributes_fn = codegen.gen_display();
    let num_params_fn = codegen.gen_num_params();
    let visit = codegen.gen_visit();
    let map_mut = codegen.gen_map();
    let collect_devices = codegen.gen_collect_devices();
    let to_device = codegen.gen_to_device();
    let fork = codegen.gen_fork();
    let valid_fn = codegen.gen_valid();
    let train_fn = codegen.gen_train();
    let materialize_fn = codegen.gen_materialize();
    let clone_fn = codegen.gen_clone();

    let (generics_module, generics_ty_module, generics_where_module) = generics.split_for_impl();

    let mut codegen = quote! {

        impl #generics_module burn::module::Module for #name #generics_ty_module #generics_where_module {
            #num_params_fn

            #visit
            #map_mut

            #collect_devices
            #to_device
            #fork

            #valid_fn

            #train_fn
            #materialize_fn
        }

        impl #generics_module core::fmt::Display for #name #generics_ty_module #generics_where_module {
            #display_fn
        }

        impl #generics_module burn::module::ModuleDisplayDefault for #name #generics_ty_module #generics_where_module {
            #attributes_fn

            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }

        impl #generics_module Clone for #name #generics_ty_module #generics_where_module {
            #clone_fn
        }
    };

    if !has_custom_display(&ast.attrs) {
        codegen.extend(quote! {
            impl #generics_module burn::module::ModuleDisplay for #name #generics_ty_module #generics_where_module {

            }
        });
    }

    codegen
}

// Without any participating fields, the type is stateless.
pub(crate) fn generate_module_stateless<Codegen: ModuleCodegen>(
    ast: &syn::DeriveInput,
    codegen: Codegen,
) -> TokenStream {
    let name = &ast.ident;
    let bounded_generics = module_generics(ast, codegen.fields());
    let (generics, generics_ty, generics_where) = bounded_generics.split_for_impl();

    let display_fn = display::display_fn(ast);
    let attributes_fn = codegen.gen_display();
    let clone_fn = codegen.gen_clone();

    let mut codegen = quote! {
        impl #generics burn::module::Module for #name #generics_ty #generics_where {
            burn::empty!(module);
        }

        impl #generics core::fmt::Display for #name #generics_ty #generics_where {
            #display_fn
        }

        impl #generics burn::module::ModuleDisplayDefault for #name #generics_ty #generics_where {
            #attributes_fn
        }

        impl #generics Clone for #name #generics_ty #generics_where {
            #clone_fn
        }
    };

    if !has_custom_display(&ast.attrs) {
        codegen.extend(quote! {
            impl  #generics burn::module::ModuleDisplay for #name #generics_ty #generics_where {

            }
        });
    }

    codegen
}

fn has_custom_display(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("module")
            && attr
                .parse_nested_meta(|meta| {
                    if meta.path.is_ident("custom_display") {
                        Ok(())
                    } else {
                        Err(meta.error("unsupported attribute"))
                    }
                })
                .is_ok()
    })
}
