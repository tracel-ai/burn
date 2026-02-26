use super::{display, record::ModuleRecordCodegen};
use crate::{
    module::generics::{GenericKind, ModuleGenerics},
    shared::generics::GenericsHelper,
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Attribute, Generics, parse_quote};

/// Basic trait to be implemented for Module generation.
pub(crate) trait ModuleCodegen {
    type RecordCodegen: ModuleRecordCodegen;

    fn gen_num_params(&self) -> TokenStream;
    fn gen_visit(&self) -> TokenStream;
    fn gen_collect_devices(&self) -> TokenStream;
    fn gen_to_device(&self) -> TokenStream;
    fn gen_fork(&self) -> TokenStream;
    fn gen_map(&self) -> TokenStream;
    fn gen_valid(&self) -> TokenStream;
    fn gen_from_inner(&self) -> TokenStream;
    fn gen_into_record(&self) -> TokenStream;
    fn gen_load_record(&self) -> TokenStream;
    fn gen_clone(&self) -> TokenStream;

    fn record_codegen(self) -> Self::RecordCodegen;

    fn gen_display(&self) -> TokenStream;

    fn module_generics(&self) -> &ModuleGenerics;
}

pub(crate) fn generate_module_standard<Codegen: ModuleCodegen>(
    ast: &syn::DeriveInput,
    codegen: Codegen,
) -> TokenStream {
    let name = &ast.ident;

    let generics = GenericsParser::from_ast(&ast.generics, codegen.module_generics());

    let display_fn = display::display_fn(ast);
    let attributes_fn = codegen.gen_display();
    let num_params_fn = codegen.gen_num_params();
    let visit = codegen.gen_visit();
    let map_mut = codegen.gen_map();
    let collect_devices = codegen.gen_collect_devices();
    let to_device = codegen.gen_to_device();
    let fork = codegen.gen_fork();
    let valid_fn = codegen.gen_valid();
    let from_inner_fn = codegen.gen_from_inner();
    let into_record_fn = codegen.gen_into_record();
    let load_record_fn = codegen.gen_load_record();
    let clone_fn = codegen.gen_clone();

    let record = codegen.record_codegen();
    let record_name = Ident::new(format!("{name}Record").as_str(), name.span());
    let (record_type, record_generics) = record.gen_record_type(&record_name, &generics.module);

    let (generics_module, generics_ty_module, generics_where_module) =
        generics.module.split_for_impl();
    let (generics_module_autodiff, generics_ty_module_autodiff, generics_where_module_autodiff) =
        generics.module_autodiff.split_for_impl();
    let (generics_module_has_autodiff, _generics_ty, generics_where_module_has_autodiff) =
        generics.module_has_autodiff.split_for_impl();
    let (_, generics_ty_record, _) = record_generics.split_for_impl();

    let generics_ty_inner_module = generics.inner_module_ty;
    let generics_ty_train_module = generics.train_module_ty;
    let generics_ty_train_inner_module = generics.train_inner_ty;

    let mut codegen = quote! {
        impl #generics_module burn::module::Module<B> for #name #generics_ty_module #generics_where_module {
            type Record = #record_name #generics_ty_record;

            #load_record_fn
            #into_record_fn

            #num_params_fn

            #visit
            #map_mut

            #collect_devices
            #to_device
            #fork

        }

        impl #generics_module_autodiff burn::module::AutodiffModule<B> for #name #generics_ty_module_autodiff #generics_where_module_autodiff
        {
            type InnerModule=#name<B::InnerBackend, #generics_ty_inner_module>;

            #valid_fn

            #from_inner_fn
        }

        impl #generics_module_has_autodiff burn::module::HasAutodiffModule<B> for #name<B::InnerBackend, #generics_ty_train_module> #generics_where_module_has_autodiff
        {
            type TrainModule=#name<B, #generics_ty_train_inner_module>;
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

        #record_type
    };

    if !has_custom_display(&ast.attrs) {
        codegen.extend(quote! {
            impl #generics_module burn::module::ModuleDisplay for #name #generics_ty_module #generics_where_module {

            }
        });
    }

    codegen
}

// TODO: wait that means nothing is persistent... (empty!)

// When there is no backend in the generic parameter, the type is considered as a constant.
pub(crate) fn generate_module_const(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (generics, generics_ty, generics_where) = ast.generics.split_for_impl();

    let backend: syn::Generics = parse_quote! { <B: burn::tensor::backend::Backend >};
    let backend_ad: syn::Generics = parse_quote! { <B: burn::tensor::backend::AutodiffBackend >};

    let mut generics_module = ast.generics.clone();
    let mut generics_module_autodiff = ast.generics.clone();

    for param in backend.params.into_iter() {
        generics_module.params.push(param);
    }
    for param in backend_ad.params.into_iter() {
        generics_module_autodiff.params.push(param);
    }
    let (generics_module, _, _) = generics_module.split_for_impl();
    let (generics_module_ad, _, _) = generics_module_autodiff.split_for_impl();

    let display_fn = display::display_fn(ast);
    let attributes_fn = display::attributes_fn(ast);

    let mut codegen = quote! {
        impl #generics_module burn::module::Module<B> for #name #generics_ty #generics_where {
            burn::empty!(module);
        }

        impl #generics_module_ad burn::module::AutodiffModule<B>
            for #name #generics_ty #generics_where {
            burn::empty!(ad_module, #name #generics_ty);
        }

        impl #generics core::fmt::Display for #name #generics_ty #generics_where {
            #display_fn
        }


        impl #generics burn::module::ModuleDisplayDefault for #name #generics_ty #generics_where {
            #attributes_fn
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

struct GenericsParser {
    module: Generics,
    module_autodiff: Generics,
    module_has_autodiff: Generics,
    inner_module_ty: TokenStream,
    train_module_ty: TokenStream,
    train_inner_ty: TokenStream,
}

impl GenericsParser {
    fn from_ast(generics: &Generics, module_generics: &ModuleGenerics) -> Self {
        let mut module = GenericsHelper::new(generics.clone());
        let mut module_autodiff = GenericsHelper::new(generics.clone());
        let mut module_has_autodiff = GenericsHelper::new(generics.clone());

        let backend_trait = module.fetch_backend_trait();

        module_autodiff.add_predicate(parse_quote! {
                B: burn::tensor::backend::AutodiffBackend
        });

        module_autodiff.add_predicate(parse_quote! {
                <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: #backend_trait
        });

        module_has_autodiff.add_predicate(parse_quote! {
                B: burn::tensor::backend::AutodiffBackend
        });

        module_has_autodiff.add_predicate(parse_quote! {
                <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: #backend_trait
        });

        let mut generics_names_except_backend = quote! {};
        let mut train_generics_names_except_backend = quote! {};
        let mut train_inner_generics_names_except_backend = quote! {};

        module
        .types()
        .into_iter()
        .filter(|ident| ident != "B")
        .for_each(|ident| {
            // By default, require module bound
            let mut requires_module_bound = true;
            let mut generic_kind = None;
            if !module_generics.is_empty() {
                generic_kind = module_generics.get_generic_kind(&ident);
                let has_module_bound = matches!(generic_kind, Some(GenericKind::Module));
                let is_unbounded = matches!(generic_kind, Some(GenericKind::Plain));

                requires_module_bound = has_module_bound || is_unbounded;
            }

            if requires_module_bound {
                module.add_predicate(
                    parse_quote! {
                        #ident: burn::module::Module<B>
                    }
                );

                module.add_predicate(
                    parse_quote! {
                        #ident: burn::module::ModuleDisplay
                    }
                );

                module_autodiff.add_predicate(
                    parse_quote! {
                        #ident: burn::module::AutodiffModule<B>
                    }
                );

                module_autodiff.add_predicate(
                    parse_quote! {
                        <#ident as burn::module::AutodiffModule<B>>::InnerModule: burn::module::Module<B::InnerBackend>
                    }
                );

                module_autodiff.add_predicate(
                    parse_quote! {
                        <#ident as burn::module::AutodiffModule<B>>::InnerModule: burn::module::ModuleDisplay
                    }
                );

                generics_names_except_backend.extend(quote! { <#ident as burn::module::AutodiffModule<B>>::InnerModule, });

                module_autodiff.add_predicate(
                    parse_quote! {
                        #ident: burn::module::ModuleDisplay
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident: burn::module::Module<B::InnerBackend>
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident: burn::module::ModuleDisplay
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident: burn::module::HasAutodiffModule<B>
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident::TrainModule: burn::module::ModuleDisplay
                    }
                );
                train_generics_names_except_backend.extend(quote! { #ident, });
                train_inner_generics_names_except_backend.extend(quote! { #ident::TrainModule, });
            }
            else {
                match generic_kind {
                    // Add required bounds at a higher level to make error message more explicit
                    Some(GenericKind::Constant) => {
                        module_autodiff.add_predicate(
                            parse_quote! {
                                #ident: Clone + core::fmt::Debug + Send + burn::serde::Serialize + burn::serde::de::DeserializeOwned
                            }
                        );
                    },
                    Some(GenericKind::Skip) => {
                        module_autodiff.add_predicate(
                            parse_quote! {
                                #ident: Clone + core::fmt::Debug + Send
                            }
                        );
                    },
                    _ => {}
                }

                // Pass through
                generics_names_except_backend.extend(quote! { #ident, });
                train_generics_names_except_backend.extend(quote! { #ident, });
                train_inner_generics_names_except_backend.extend(quote! { #ident, });
            }

        });

        module.consts().into_iter().for_each(|ident| {
            generics_names_except_backend.extend(quote! { #ident, });
            train_generics_names_except_backend.extend(quote! { #ident, });
            train_inner_generics_names_except_backend.extend(quote! { #ident, });
        });

        Self {
            module: module.generics,
            module_autodiff: module_autodiff.generics,
            module_has_autodiff: module_has_autodiff.generics,
            inner_module_ty: generics_names_except_backend,
            train_module_ty: train_generics_names_except_backend,
            train_inner_ty: train_inner_generics_names_except_backend,
        }
    }
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
