use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, GenericParam, TypeParamBound, parse_quote};

pub(crate) fn derive_impl(input: &DeriveInput) -> TokenStream {
    match expand(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

struct FieldCode {
    field_types: Vec<syn::Type>,
    skipped_types: Vec<syn::Type>,
    num_exprs: Vec<TokenStream2>,
    visit_stmts: Vec<TokenStream2>,
    map_construct: TokenStream2,
    zip_construct: TokenStream2,
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let backend = find_backend_param(input)?;
    let rebound = format_ident!("__B2");
    let zip_backend = format_ident!("__B3");
    let with_backend = with_backend_type(name, &input.generics, &backend, &rebound);
    let fields = field_code(input, &backend, &rebound, &zip_backend)?;

    let other_params: Vec<syn::Ident> = input
        .generics
        .type_params()
        .filter(|param| param.ident != backend)
        .map(|param| param.ident.clone())
        .collect();

    let mut impl_generics = input.generics.clone();
    {
        let where_clause = impl_generics.make_where_clause();
        for ty in &fields.field_types {
            // A concrete field type such as `FuncTensor<B, D>` already has a known `PyTree`
            // implementation. Adding the bound here makes the compiler prefer the where-bound
            // projection over that implementation, which breaks the rebound field types.
            if mentions_other_type_param(ty, &other_params) {
                where_clause
                    .predicates
                    .push(parse_quote! { #ty: ::burn_func::PyTree<#backend> });
            }
        }
        for ty in &fields.skipped_types {
            where_clause
                .predicates
                .push(parse_quote! { #ty: ::core::clone::Clone });
        }
        let (_, ty_generics, _) = input.generics.split_for_impl();
        let self_ty = quote! { #name #ty_generics };
        where_clause
            .predicates
            .push(parse_quote! { #self_ty: ::core::clone::Clone + ::core::marker::Send });
    }

    let (impl_generics, _, where_clause) = impl_generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let num_exprs = &fields.num_exprs;
    let visit_stmts = &fields.visit_stmts;
    let map_construct = &fields.map_construct;
    let zip_construct = &fields.zip_construct;

    Ok(quote! {
        impl #impl_generics ::burn_func::PyTree<#backend> for #name #ty_generics #where_clause {
            type WithBackend<#rebound: ::burn_func::burn_backend::Backend> = #with_backend;

            fn num_leaves(&self) -> usize {
                0 #(+ #num_exprs)*
            }

            fn map_float<#rebound: ::burn_func::burn_backend::Backend, F>(
                &self,
                f: &mut F,
            ) -> Self::WithBackend<#rebound>
            where
                F: FnMut(
                    &::burn_func::burn_backend::tensor::FloatTensor<#backend>,
                ) -> ::burn_func::burn_backend::tensor::FloatTensor<#rebound>,
            {
                #map_construct
            }

            fn visit_float<F>(&self, f: &mut F)
            where
                F: FnMut(&::burn_func::burn_backend::tensor::FloatTensor<#backend>),
            {
                #(#visit_stmts)*
            }

            fn zip_float<#rebound: ::burn_func::burn_backend::Backend, #zip_backend: ::burn_func::burn_backend::Backend, F>(
                &self,
                other: &Self::WithBackend<#rebound>,
                f: &mut F,
            ) -> Self::WithBackend<#zip_backend>
            where
                F: FnMut(
                    &::burn_func::burn_backend::tensor::FloatTensor<#backend>,
                    &::burn_func::burn_backend::tensor::FloatTensor<#rebound>,
                ) -> ::burn_func::burn_backend::tensor::FloatTensor<#zip_backend>,
            {
                #zip_construct
            }
        }
    })
}

fn find_backend_param(input: &DeriveInput) -> syn::Result<syn::Ident> {
    if let Some(param) = input
        .generics
        .type_params()
        .find(|param| param.ident == "B")
    {
        return Ok(param.ident.clone());
    }

    for param in input.generics.type_params() {
        for bound in &param.bounds {
            if let TypeParamBound::Trait(trait_bound) = bound
                && trait_bound
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "Backend")
            {
                return Ok(param.ident.clone());
            }
        }
    }

    Err(syn::Error::new_spanned(
        &input.generics,
        "PyTree requires a backend type parameter named `B` or bounded by `Backend`",
    ))
}

fn with_backend_type(
    name: &syn::Ident,
    generics: &syn::Generics,
    backend: &syn::Ident,
    rebound: &syn::Ident,
) -> TokenStream2 {
    let args: Vec<TokenStream2> = generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(lifetime) => {
                let lifetime = &lifetime.lifetime;
                quote! { #lifetime }
            }
            GenericParam::Type(ty) => {
                let ident = &ty.ident;
                if ident == backend {
                    quote! { #rebound }
                } else {
                    quote! { #ident }
                }
            }
            GenericParam::Const(constant) => {
                let ident = &constant.ident;
                quote! { #ident }
            }
        })
        .collect();

    if args.is_empty() {
        quote! { #name }
    } else {
        quote! { #name<#(#args),*> }
    }
}

fn field_code(
    input: &DeriveInput,
    backend: &syn::Ident,
    rebound: &syn::Ident,
    zip_backend: &syn::Ident,
) -> syn::Result<FieldCode> {
    let name = &input.ident;

    let data = match &input.data {
        Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "PyTree can only be derived for structs",
            ));
        }
    };

    match &data.fields {
        Fields::Named(fields) => {
            let mut code = FieldCode {
                field_types: Vec::new(),
                skipped_types: Vec::new(),
                num_exprs: Vec::new(),
                visit_stmts: Vec::new(),
                map_construct: TokenStream2::new(),
                zip_construct: TokenStream2::new(),
            };
            let mut names = Vec::new();
            let mut map_values = Vec::new();
            let mut zip_values = Vec::new();

            for field in &fields.named {
                let ident = field.ident.as_ref().unwrap();
                names.push(ident.clone());
                if is_skipped(field) {
                    if mentions_ident(&field.ty, backend) {
                        return Err(syn::Error::new_spanned(
                            &field.ty,
                            "#[pytree(skip)] requires a B-independent type; close over int/bool tensors outside the parameter tree",
                        ));
                    }
                    code.skipped_types.push(field.ty.clone());
                    map_values.push(quote! { self.#ident.clone() });
                    zip_values.push(quote! { self.#ident.clone() });
                } else {
                    code.field_types.push(field.ty.clone());
                    code.num_exprs.push(quote! {
                        ::burn_func::PyTree::<#backend>::num_leaves(&self.#ident)
                    });
                    code.visit_stmts.push(quote! {
                        ::burn_func::PyTree::<#backend>::visit_float(&self.#ident, f);
                    });
                    map_values.push(quote! {
                        ::burn_func::PyTree::<#backend>::map_float::<#rebound, _>(&self.#ident, f)
                    });
                    zip_values.push(quote! {
                        ::burn_func::PyTree::<#backend>::zip_float::<#rebound, #zip_backend, _>(
                            &self.#ident,
                            &other.#ident,
                            f,
                        )
                    });
                }
            }

            code.map_construct = quote! { #name { #(#names: #map_values),* } };
            code.zip_construct = quote! { #name { #(#names: #zip_values),* } };
            Ok(code)
        }
        Fields::Unnamed(fields) => {
            let mut code = FieldCode {
                field_types: Vec::new(),
                skipped_types: Vec::new(),
                num_exprs: Vec::new(),
                visit_stmts: Vec::new(),
                map_construct: TokenStream2::new(),
                zip_construct: TokenStream2::new(),
            };
            let mut map_values = Vec::new();
            let mut zip_values = Vec::new();

            for (index, field) in fields.unnamed.iter().enumerate() {
                let index = syn::Index::from(index);
                if is_skipped(field) {
                    if mentions_ident(&field.ty, backend) {
                        return Err(syn::Error::new_spanned(
                            &field.ty,
                            "#[pytree(skip)] requires a B-independent type; close over int/bool tensors outside the parameter tree",
                        ));
                    }
                    code.skipped_types.push(field.ty.clone());
                    map_values.push(quote! { self.#index.clone() });
                    zip_values.push(quote! { self.#index.clone() });
                } else {
                    code.field_types.push(field.ty.clone());
                    code.num_exprs.push(quote! {
                        ::burn_func::PyTree::<#backend>::num_leaves(&self.#index)
                    });
                    code.visit_stmts.push(quote! {
                        ::burn_func::PyTree::<#backend>::visit_float(&self.#index, f);
                    });
                    map_values.push(quote! {
                        ::burn_func::PyTree::<#backend>::map_float::<#rebound, _>(&self.#index, f)
                    });
                    zip_values.push(quote! {
                        ::burn_func::PyTree::<#backend>::zip_float::<#rebound, #zip_backend, _>(
                            &self.#index,
                            &other.#index,
                            f,
                        )
                    });
                }
            }

            code.map_construct = quote! { #name(#(#map_values),*) };
            code.zip_construct = quote! { #name(#(#zip_values),*) };
            Ok(code)
        }
        Fields::Unit => Ok(FieldCode {
            field_types: Vec::new(),
            skipped_types: Vec::new(),
            num_exprs: Vec::new(),
            visit_stmts: Vec::new(),
            map_construct: quote! { #name },
            zip_construct: quote! { #name },
        }),
    }
}

fn mentions_other_type_param(ty: &syn::Type, other_params: &[syn::Ident]) -> bool {
    struct Visitor<'a> {
        other_params: &'a [syn::Ident],
        found: bool,
    }

    impl<'ast, 'a> syn::visit::Visit<'ast> for Visitor<'a> {
        fn visit_ident(&mut self, ident: &'ast syn::Ident) {
            if self.other_params.iter().any(|param| param == ident) {
                self.found = true;
            }
        }
    }

    let mut visitor = Visitor {
        other_params,
        found: false,
    };
    syn::visit::visit_type(&mut visitor, ty);
    visitor.found
}

fn mentions_ident(ty: &syn::Type, target: &syn::Ident) -> bool {
    struct Visitor<'a> {
        target: &'a syn::Ident,
        found: bool,
    }

    impl<'ast, 'a> syn::visit::Visit<'ast> for Visitor<'a> {
        fn visit_ident(&mut self, ident: &'ast syn::Ident) {
            if ident == self.target {
                self.found = true;
            }
        }
    }

    let mut visitor = Visitor {
        target,
        found: false,
    };
    syn::visit::visit_type(&mut visitor, ty);
    visitor.found
}

fn is_skipped(field: &syn::Field) -> bool {
    let mut skipped = false;

    for attr in &field.attrs {
        if !attr.path().is_ident("pytree") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") || meta.path.is_ident("constant") {
                skipped = true;
            }
            Ok(())
        });
    }

    skipped
}
