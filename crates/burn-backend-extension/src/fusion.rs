//! Lower backend extension methods to opaque Fusion operations.
//!
//! Metadata runs on the calling thread before registration; the inner backend call is deferred.
//! Tuples are traversed here, while tensors and derived extension values delegate to the runtime's
//! `FusionValueAdapter` trait for metadata flattening, validation, handle publication, and reconstruction.
use crate::ir::{OperationOutput, TensorKind, with_backend};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Expr, FnArg, GenericParam, ItemTrait, Pat, ReturnType, TraitItem, Type};

#[allow(clippy::large_enum_variant)]
enum FusionBehavior {
    Default,
    Meta(Expr),
    Tensor { dtype: Expr, shape: Expr },
}

fn unsupported(item: impl quote::ToTokens) -> syn::Error {
    syn::Error::new_spanned(
        item,
        "unsupported lazy Fusion signature; use #[fusion(default)] or omit Fusion from #[backend_extension] and implement the trait manually",
    )
}

pub(crate) fn expand(item: &ItemTrait) -> syn::Result<TokenStream> {
    let trait_name = &item.ident;
    let mut methods = Vec::new();
    let mut diagnostics = Vec::new();
    for item in &item.items {
        let TraitItem::Fn(method) = item else {
            continue;
        };
        match expand_method(trait_name, method) {
            Ok(Some(method)) => methods.push(method),
            Ok(None) => {}
            Err(error) => {
                // A disabled method must not cause a Fusion diagnostic in this configuration.
                let gates: Vec<_> = method
                    .attrs
                    .iter()
                    .filter(|a| a.path().is_ident("cfg") || a.path().is_ident("cfg_attr"))
                    .collect();
                if gates.is_empty() {
                    return Err(error);
                }
                let error = error.into_compile_error();
                diagnostics.push(quote!(#(#gates)* #error));
            }
        }
    }
    Ok(quote! {
        impl<B: burn::backend::fusion::FusionBackend + #trait_name> #trait_name for burn::backend::fusion::Fusion<B> {
            #(#diagnostics)*
            #(#methods)*
        }
    })
}

/// Emit one implementation method, or omit it to inherit the trait's default body.
fn expand_method(
    trait_name: &syn::Ident,
    method: &syn::TraitItemFn,
) -> syn::Result<Option<TokenStream>> {
    let annotations: Vec<_> = method
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("fusion"))
        .collect();
    if annotations.len() != 1 {
        return Err(syn::Error::new_spanned(
            method,
            "each Fusion method requires exactly one #[fusion(dtype = ..., shape = ...)], #[fusion(meta = callable)], or #[fusion(default)]",
        ));
    }
    let mut behavior = None;
    let mut dtype = None;
    let mut shape = None;
    let mut id = None;
    annotations[0].parse_nested_meta(|meta| {
        if meta.path.is_ident("id") {
            if id.is_some() {
                return Err(meta.error("duplicate Fusion id"));
            }
            id = Some(meta.value()?.parse::<syn::LitStr>()?);
            return Ok(());
        }
        if meta.path.is_ident("dtype") || meta.path.is_ident("shape") {
            if behavior.is_some() {
                return Err(meta.error("dtype and shape cannot be combined with meta or default"));
            }
            let slot = if meta.path.is_ident("dtype") {
                &mut dtype
            } else {
                &mut shape
            };
            if slot.is_some() {
                return Err(meta.error("duplicate Fusion metadata field"));
            }
            *slot = Some(meta.value()?.parse::<Expr>()?);
            return Ok(());
        }
        if behavior.is_some() || dtype.is_some() || shape.is_some() {
            return Err(meta.error("conflicting Fusion behaviors"));
        }
        behavior = Some(if meta.path.is_ident("default") {
            FusionBehavior::Default
        } else if meta.path.is_ident("meta") {
            FusionBehavior::Meta(meta.value()?.parse()?)
        } else {
            return Err(meta.error("expected dtype, shape, meta, default, or id"));
        });
        Ok(())
    })?;
    let behavior = match (behavior, dtype, shape) {
        (Some(behavior), None, None) => behavior,
        (None, Some(dtype), Some(shape)) => FusionBehavior::Tensor { dtype, shape },
        (None, _, _) => {
            return Err(syn::Error::new_spanned(
                annotations[0],
                "provide both dtype and shape, or use meta or default",
            ));
        }
        _ => unreachable!("conflicting behaviors are rejected while parsing"),
    };
    let sig = &method.sig;
    let mut signature = sig.clone();
    for arg in &mut signature.inputs {
        if let FnArg::Typed(arg) = arg {
            arg.attrs
                .retain(|a| !a.path().is_ident("extension_type") && !a.path().is_ident("fusion"));
        }
    }
    let name = &sig.ident;
    let attrs = method.attrs.iter().filter(|a| !a.path().is_ident("fusion"));
    let args: Vec<_> = sig
        .inputs
        .iter()
        .map(|a| match a {
            FnArg::Typed(p) => match p.pat.as_ref() {
                Pat::Ident(i) => Ok((&i.ident, p)),
                _ => Err(unsupported(a)),
            },
            _ => Err(unsupported(a)),
        })
        .collect::<syn::Result<_>>()?;
    let mut expr = match behavior {
        FusionBehavior::Default => {
            if id.is_some() {
                return Err(syn::Error::new_spanned(
                    annotations[0],
                    "default does not register an operation; id does not apply",
                ));
            }
            if method.default.is_none() {
                return Err(syn::Error::new_spanned(
                    method,
                    "#[fusion(default)] requires a default body",
                ));
            }
            return Ok(None);
        }
        FusionBehavior::Meta(expr) => expr,
        FusionBehavior::Tensor { dtype, shape } => {
            if !matches!(&sig.output, ReturnType::Type(_, ty) if TensorKind::from_type(ty).is_some())
            {
                return Err(syn::Error::new_spanned(
                    &sig.output,
                    "dtype and shape require a single tensor output; use #[fusion(meta = callable)] for structured outputs",
                ));
            }
            tensor_metadata(dtype, shape, &args)?
        }
    };
    if !matches!(expr, Expr::Path(_) | Expr::Closure(_)) {
        return Err(unsupported(expr));
    }
    if sig.asyncness.is_some() || !sig.generics.params.is_empty() {
        return Err(unsupported(sig));
    }
    let mut metadata_args = Vec::new();
    let mut prepare = Vec::new();
    let mut retrieve = Vec::new();
    let mut invoke = Vec::new();
    let mut inputs = Vec::new();
    let mut metadata_inputs = Vec::new();
    let mut visits = Vec::new();
    let mut scalars = Vec::new();
    for (name, arg) in &args {
        let scalar = scalar_argument(name, arg)?;
        let is_ext = arg
            .attrs
            .iter()
            .any(|a| a.path().is_ident("extension_type"));
        let borrowed = matches!(arg.ty.as_ref(), Type::Reference(_));
        if matches!(arg.ty.as_ref(), Type::Reference(r) if r.mutability.is_some()) {
            return Err(unsupported(arg));
        }
        let ty = match arg.ty.as_ref() {
            Type::Reference(r) => r.elem.as_ref(),
            ty => ty,
        };
        let adapter = field_adapter(ty, is_ext)?;
        if let Some(adapter) = adapter {
            if scalar.is_some() {
                return Err(syn::Error::new_spanned(
                    arg,
                    "#[fusion(scalar)] applies to ordinary arguments, not tensor or extension inputs",
                ));
            }
            if is_ext && borrowed {
                return Err(unsupported(arg));
            }
            let adapter = if is_ext {
                with_backend(ty, quote!(B))
            } else {
                adapter
            };
            let adapter =
                quote!(<#adapter as burn::backend::fusion::custom::FusionValueAdapter<B>>);
            let meta = format_ident!("__metadata_{name}");
            metadata_inputs.push(quote!(let #meta = #adapter::to_metadata(&#name);));
            metadata_args.push(quote!(&#meta));
            visits.push(quote!(#adapter::visit_fused_tensors(&#name, &mut __visit);));
            retrieve.push(
                quote!(let #name = #adapter::resolve_inputs(&#meta, &mut __input_iter, __handles);),
            );
            invoke.push(if borrowed {
                quote!(&#name)
            } else {
                quote!(#name)
            });
            let value = if borrowed {
                quote!((*#name).clone())
            } else {
                quote!(#name)
            };
            inputs.push(quote!(#adapter::append_input_ir(#value, &mut __inputs);));
        } else {
            if borrowed
                || matches!(arg.ty.as_ref(), Type::ImplTrait(_))
                || crate::ir::type_contains_self(&arg.ty)
            {
                return Err(unsupported(arg));
            }
            if let Some(value) = scalar {
                scalars.push(quote!(burn::backend::fusion::custom::ScalarIr::from(
                    burn::backend::Scalar::from((#value).clone())
                )));
            }
            metadata_args.push(quote!(&#name));
            // Operation::execute borrows the closure, so the backend receives cloned options.
            invoke.push(quote!(#name.clone()));
            prepare.push(quote!(__capture(&#name);));
        }
    }
    if inputs.is_empty() {
        return Err(unsupported(sig));
    }
    // An immediately invoked closure needs explicit parameter types for expressions like x.shape.
    if let Expr::Closure(closure) = &mut expr {
        if closure.inputs.len() != args.len() {
            return Err(syn::Error::new_spanned(
                closure,
                "metadata must accept every argument in declaration order",
            ));
        }
        for (pat, (_, arg)) in closure.inputs.iter_mut().zip(&args) {
            if !matches!(pat, Pat::Type(_)) {
                let ty = if TensorKind::from_type(&arg.ty).is_some() {
                    quote!(burn::backend::fusion::custom::TensorSpec)
                } else if arg
                    .attrs
                    .iter()
                    .any(|a| a.path().is_ident("extension_type"))
                {
                    let ty = with_backend(&arg.ty, quote!(B));
                    quote!(<#ty as burn::backend::fusion::custom::ExtensionMetadata>::Metadata)
                } else {
                    let ty = &arg.ty;
                    quote!(#ty)
                };
                *pat = Pat::Type(syn::PatType {
                    attrs: Vec::new(),
                    pat: Box::new(pat.clone()),
                    colon_token: Default::default(),
                    ty: Box::new(syn::parse_quote!(&#ty)),
                });
            }
        }
    }
    let ReturnType::Type(_, ty) = &sig.output else {
        return Err(unsupported(sig));
    };
    reject_borrowed_output(ty)?;
    let output = OperationOutput::extension(ty);
    let mut specs = Vec::new();
    let mut validate = Vec::new();
    let mut publish = Vec::new();
    let reconstruct = walk_output(
        &output,
        quote!(__meta),
        quote!(__output),
        &mut specs,
        &mut validate,
        &mut publish,
    )?;
    let call = quote!(<B as #trait_name>::#name(#(#invoke),*));
    let call = if sig.unsafety.is_some() {
        quote!(unsafe { #call })
    } else {
        call
    };
    let id = id
        .map(|id| quote!(#id))
        .unwrap_or_else(|| quote!(stringify!(#name)));
    Ok(Some(quote! {
        #(#attrs)* #signature {
            use burn::backend::fusion::custom as __fusion;
            fn __capture<T: Clone + Send + Sync + 'static>(_: &T) {}
            let mut __client = None;
            let mut __device_id = None;
            let mut __visit = |tensor: &burn::backend::fusion::FusionTensor<B::FusionRuntime>| {
                let id = burn::backend::Device::to_id(tensor.client.device());
                assert_eq!(*__device_id.get_or_insert(id), id, "Fusion custom inputs must share a device");
                __client.get_or_insert_with(|| tensor.client.clone());
            };
            #(#visits)*
            let __client = __client.expect("Fusion custom operation requires at least one input tensor");
            let __device = __client.device().clone();
            #(#prepare)*
            #(#metadata_inputs)*
            let __meta = (#expr)(#(#metadata_args),*);
            let mut __specs = Vec::new();
            #(#specs)*
            let __execution_meta = __meta.clone();
            let __scalars: Vec<__fusion::ScalarIr> = vec![#(#scalars),*];
            let mut __inputs = Vec::new();
            #(#inputs)*
            let __outputs: Vec<_> = __specs.into_iter().map(|s| __fusion::TensorIr::uninit(__client.create_empty_handle(), s.shape, s.dtype)).collect();
            let __desc = __fusion::CustomOpIr::with_scalars(#id, &__inputs, &__outputs, __scalars);
            let __op = __fusion::OperationFn(move |__handles: &mut __fusion::HandleContainer<<B::FusionRuntime as burn::backend::fusion::FusionRuntime>::FusionHandle>| {
                let __meta = &__execution_meta;
                let mut __input_iter = __inputs.iter();
                #(#retrieve)*
                let __output = #call;
                let mut __expected = __outputs.iter();
                #(#validate)*
                let mut __expected = __outputs.iter();
                #(#publish)*
                Ok(())
            });
            let mut __tensors = __client.register(__fusion::StreamId::current(), __fusion::OperationIr::Custom(__desc), __op).into_iter();
            #reconstruct
        }
    }))
}

/// Primitive parameters are scalar IR inputs; custom types can opt in or provide an encoding.
/// The macro cannot resolve aliases or inspect a type declared in another crate.
fn scalar_argument(name: &syn::Ident, arg: &syn::PatType) -> syn::Result<Option<Expr>> {
    let attrs: Vec<_> = arg
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("fusion"))
        .collect();
    if attrs.len() > 1 {
        return Err(syn::Error::new_spanned(
            arg,
            "expected one #[fusion(scalar)] or #[fusion(scalar = expression)]",
        ));
    }
    if let Some(attr) = attrs.first() {
        let mut value = None;
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("scalar") || value.is_some() {
                return Err(meta.error("expected scalar or scalar = expression"));
            }
            value = Some(if meta.input.peek(syn::Token![=]) {
                meta.value()?.parse()?
            } else {
                syn::parse_quote!(#name)
            });
            Ok(())
        })?;
        return value.map(Some).ok_or_else(|| {
            syn::Error::new_spanned(attr, "expected scalar or scalar = expression")
        });
    }
    let Type::Path(ty) = arg.ty.as_ref() else {
        return Ok(None);
    };
    let segments: Vec<_> = ty.path.segments.iter().collect();
    let primitive_path = segments.len() == 1
        || (segments.len() == 3
            && (segments[0].ident == "core" || segments[0].ident == "std")
            && segments[1].ident == "primitive");
    let primitive = ty.qself.is_none()
        && primitive_path
        && segments
            .iter()
            .all(|s| matches!(s.arguments, syn::PathArguments::None))
        && segments.last().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "bool"
                    | "f32"
                    | "f64"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "isize"
                    | "u8"
                    | "u16"
                    | "u32"
                    | "u64"
                    | "usize"
            )
        });
    Ok(primitive.then(|| syn::parse_quote!(#name)))
}

/// Lower field expressions to the same metadata callback used by `meta`.
/// Separate scopes give tensor names their dtype or borrowed shape without rewriting expressions.
fn tensor_metadata(
    dtype: Expr,
    shape: Expr,
    args: &[(&syn::Ident, &syn::PatType)],
) -> syn::Result<Expr> {
    let names: Vec<_> = args.iter().map(|(name, _)| name).collect();
    let tensors: Vec<_> = args
        .iter()
        .filter(|(_, arg)| TensorKind::from_type(&arg.ty).is_some())
        .map(|(name, _)| *name)
        .collect();
    // A bare operand borrows its shape; computed expressions already return an owned Shape.
    let shape = if matches!(&shape, Expr::Path(path) if tensors.iter().any(|name| path.path.is_ident(*name)))
    {
        quote!((#shape).clone())
    } else {
        quote!(#shape)
    };
    syn::parse2(quote! {
        |#(#names),*| {
            let _ = (#(#names),*);
            let __dtype = { #(#[allow(unused_variables)] let #tensors = #tensors.dtype;)* #dtype };
            let __shape = { #(#[allow(unused_variables)] let #tensors = &#tensors.shape;)* #shape };
            burn::backend::fusion::custom::TensorSpec::new(__shape, __dtype)
        }
    })
}

/// Build each output phase in the same tensor-leaf order, returning the reconstruction expression.
/// Keep validation and publication separate so no handles are published until all outputs pass.
fn walk_output(
    out: &OperationOutput,
    meta: TokenStream,
    value: TokenStream,
    specs: &mut Vec<TokenStream>,
    validate: &mut Vec<TokenStream>,
    publish: &mut Vec<TokenStream>,
) -> syn::Result<TokenStream> {
    if let OperationOutput::Tuple(items) = out {
        let items = items
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let i = syn::Index::from(i);
                walk_output(
                    o,
                    quote!(#meta.#i),
                    quote!(#value.#i),
                    specs,
                    validate,
                    publish,
                )
            })
            .collect::<syn::Result<Vec<_>>>()?;
        return Ok(quote!((#(#items,)*)));
    }
    let adapter = match out {
        OperationOutput::Tensor(kind) => {
            let kind = kind.variant();
            quote!(burn::backend::fusion::custom::#kind)
        }
        OperationOutput::Extension(ty) => with_backend(ty, quote!(B)),
        _ => return Err(unsupported(quote!(output))),
    };
    let adapter = quote!(<#adapter as burn::backend::fusion::custom::FusionValueAdapter<B>>);
    specs.push(quote!(#adapter::append_output_specs(&#meta, &mut __specs);));
    validate
        .push(quote!(#adapter::validate_outputs(&#value, &#meta, &mut __expected, &__device)?;));
    publish.push(quote!(#adapter::register_output_handles(#value, &mut __expected, __handles);));
    Ok(quote!(#adapter::build_fused_output(&#meta, &mut __tensors)))
}

pub(crate) fn derive(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let mut enabled = false;
    let mut cfg = None;
    for attr in input
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("extension_type"))
    {
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("fusion") || enabled {
                return Err(meta.error("expected fusion or fusion: cfg(...)"));
            }
            enabled = true;
            if meta.input.peek(syn::Token![:]) {
                meta.input.parse::<syn::Token![:]>()?;
                cfg = Some(meta.input.parse::<syn::Meta>()?);
            }
            Ok(())
        })?;
    }
    if !enabled {
        return Ok(quote!());
    }
    let gate = cfg.map(|c| quote!(#[#c]));
    let result = derive_adapter(input, &gate).unwrap_or_else(|e| {
        let e = e.into_compile_error();
        quote!(#gate #e)
    });
    Ok(result)
}

/// Tensor leaves and nested extension values use the same adapter interface.
fn field_adapter(ty: &Type, is_ext: bool) -> syn::Result<Option<TokenStream>> {
    if let Some(kind) = TensorKind::from_type(ty) {
        if matches!(ty, Type::Reference(_)) {
            return Err(unsupported(ty));
        }
        let kind = kind.variant();
        Ok(Some(quote!(burn::backend::fusion::custom::#kind)))
    } else if is_ext {
        Ok(Some(quote!(#ty)))
    } else {
        Ok(None)
    }
}

/// Mirror structs and enum variants in metadata; reuse the dispatch derive's field order.
fn derive_adapter(
    input: &syn::DeriveInput,
    gate: &Option<TokenStream>,
) -> syn::Result<TokenStream> {
    use crate::derive::{collect_cases, gen_case_ctor, gen_case_pattern};
    if input.generics.params.len() != 1 {
        return Err(unsupported(input));
    }
    let Some(GenericParam::Type(backend)) = input.generics.params.first() else {
        return Err(unsupported(input));
    };
    let b = &backend.ident;
    let name = &input.ident;
    let metadata = format_ident!("{name}Metadata");
    let mut definition = input.clone();
    definition.ident = metadata.clone();
    definition.generics = Default::default();
    definition.attrs.clear();
    let fields: Vec<_> = match &mut definition.data {
        syn::Data::Struct(data) => data.fields.iter_mut().collect(),
        syn::Data::Enum(data) => data
            .variants
            .iter_mut()
            .flat_map(|v| {
                v.attrs.retain(|a| a.path().is_ident("doc"));
                v.fields.iter_mut()
            })
            .collect(),
        _ => return Err(unsupported(input)),
    };
    for field in fields {
        let is_ext = field
            .attrs
            .iter()
            .any(|a| a.path().is_ident("extension_type"));
        if field_adapter(&field.ty, is_ext)?.is_some() {
            field.ty = if TensorKind::from_type(&field.ty).is_some() {
                syn::parse_quote!(burn::backend::fusion::custom::TensorSpec)
            } else {
                // Resolve through the original type so ordinary imports and aliases work.
                let ty = with_backend(&field.ty, quote!(burn::backend::Dispatch));
                syn::parse_quote!(<#ty as burn::backend::fusion::custom::ExtensionMetadata>::Metadata)
            };
        }
        field.attrs.retain(|a| a.path().is_ident("doc"));
    }
    let cases = collect_cases(input)?;
    let mut meta_cases = collect_cases(&definition)?;
    // Distinct bindings let validation match actual values and their expected variant together.
    for case in &mut meta_cases {
        for field in &mut case.fields {
            field.bind = format_ident!("{}_meta", field.bind);
        }
    }
    let mut describe = Vec::new();
    let mut visit = Vec::new();
    let mut flatten = Vec::new();
    let mut read = Vec::new();
    let mut specs = Vec::new();
    let mut validate = Vec::new();
    let mut publish = Vec::new();
    let mut reconstruct = Vec::new();
    for (case, meta_case) in cases.iter().zip(&meta_cases) {
        let value_pattern = gen_case_pattern(case, |_| true);
        let tensor_pattern = gen_case_pattern(case, |i| {
            case.fields[i].tensor_kind.is_some() || case.fields[i].is_ext
        });
        let meta_pattern = gen_case_pattern(meta_case, |_| true);
        let meta_tensor_pattern = gen_case_pattern(meta_case, |i| {
            case.fields[i].tensor_kind.is_some() || case.fields[i].is_ext
        });
        let mut descriptions = Vec::new();
        let mut visits = Vec::new();
        let mut inputs = Vec::new();
        let mut reads = Vec::new();
        let mut spec_fields = Vec::new();
        let mut checks = Vec::new();
        let mut publications = Vec::new();
        let mut reconstructions = Vec::new();
        for (field, meta_field) in case.fields.iter().zip(&meta_case.fields) {
            let value = &field.bind;
            let meta = &meta_field.bind;
            if let Some(adapter) = field_adapter(&field.ty, field.is_ext)? {
                let adapter =
                    quote!(<#adapter as burn::backend::fusion::custom::FusionValueAdapter<#b>>);
                descriptions.push(quote!(#adapter::to_metadata(#value)));
                visits.push(quote!(#adapter::visit_fused_tensors(#value, visit);));
                inputs.push(quote!(#adapter::append_input_ir(#value, inputs);));
                reads.push(quote!(#adapter::resolve_inputs(#meta, inputs, handles)));
                spec_fields.push(quote!(#adapter::append_output_specs(#meta, out);));
                checks.push(quote!(#adapter::validate_outputs(#value, #meta, specs, device)?;));
                publications
                    .push(quote!(#adapter::register_output_handles(#value, specs, handles);));
                reconstructions.push(quote!(#adapter::build_fused_output(#meta, tensors)));
            } else {
                descriptions.push(quote!(#value.clone()));
                reads.push(quote!(#meta.clone()));
                // Ordinary output fields come from metadata; execution only supplies tensors.
                reconstructions.push(quote!(#meta.clone()));
            }
        }
        let description = gen_case_ctor(meta_case, &descriptions);
        let restored = gen_case_ctor(case, &reads);
        let reconstructed = gen_case_ctor(case, &reconstructions);
        describe.push(quote!(#value_pattern => #description));
        visit.push(quote!(#tensor_pattern => { #(#visits)* }));
        flatten.push(quote!(#tensor_pattern => { #(#inputs)* }));
        read.push(quote!(#meta_pattern => #restored));
        specs.push(quote!(#meta_pattern => { #(#spec_fields)* }));
        validate.push(quote!((#tensor_pattern, #meta_tensor_pattern) => { #(#checks)* Ok(()) }));
        publish.push(quote!(#tensor_pattern => { #(#publications)* }));
        reconstruct.push(quote!(#meta_pattern => #reconstructed));
    }
    let mismatch = matches!(input.data, syn::Data::Enum(_)).then(|| quote! {
        _ => Err(burn::backend::fusion::ExecutionError::generic("Fusion custom output variant differs from its metadata")),
    });
    let bounds = &backend.bounds;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        #gate
        #[doc = "Tensor metadata and ordinary fields for the corresponding backend extension value."]
        #[derive(Clone, Debug)]
        #definition
        #gate
        impl #impl_generics burn::backend::fusion::custom::ExtensionMetadata for #name #ty_generics #where_clause {
            type Metadata = #metadata;
        }
        #gate
        impl<#b: burn::backend::fusion::FusionBackend + #bounds> burn::backend::fusion::custom::FusionValueAdapter<#b> for #name<#b> #where_clause {
            type Metadata = #metadata;
            type Inner = Self;
            type Fused = #name<burn::backend::fusion::Fusion<#b>>;
            fn to_metadata(value: &Self::Fused) -> Self::Metadata { match value { #(#describe,)* } }
            fn visit_fused_tensors(value: &Self::Fused, visit: &mut impl FnMut(&burn::backend::fusion::FusionTensor<#b::FusionRuntime>)) { match value { #(#visit,)* } }
            fn append_input_ir(value: Self::Fused, inputs: &mut Vec<burn::backend::fusion::custom::TensorIr>) { match value { #(#flatten,)* } }
            fn resolve_inputs(meta: &Self::Metadata, inputs: &mut core::slice::Iter<'_, burn::backend::fusion::custom::TensorIr>, handles: &mut burn::backend::fusion::custom::HandleContainer<<#b::FusionRuntime as burn::backend::fusion::FusionRuntime>::FusionHandle>) -> Self::Inner { match meta { #(#read,)* } }
            fn append_output_specs(meta: &Self::Metadata, out: &mut Vec<burn::backend::fusion::custom::TensorSpec>) { match meta { #(#specs,)* } }
            fn validate_outputs(value: &Self, meta: &Self::Metadata, specs: &mut core::slice::Iter<'_, burn::backend::fusion::custom::TensorIr>, device: &#b::Device) -> Result<(), burn::backend::fusion::ExecutionError> {
                match (value, meta) { #(#validate,)* #mismatch }
            }
            fn register_output_handles(value: Self, specs: &mut core::slice::Iter<'_, burn::backend::fusion::custom::TensorIr>, handles: &mut burn::backend::fusion::custom::HandleContainer<<#b::FusionRuntime as burn::backend::fusion::FusionRuntime>::FusionHandle>) { match value { #(#publish,)* } }
            fn build_fused_output(meta: &Self::Metadata, tensors: &mut std::vec::IntoIter<burn::backend::fusion::FusionTensor<#b::FusionRuntime>>) -> Self::Fused { match meta { #(#reconstruct,)* } }
        }
    })
}

fn reject_borrowed_output(ty: &Type) -> syn::Result<()> {
    match ty {
        Type::Reference(_) => Err(unsupported(ty)),
        Type::Tuple(tuple) => {
            for ty in &tuple.elems {
                reject_borrowed_output(ty)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn expand_trait(tokens: TokenStream) -> String {
        expand(&syn::parse2(tokens).unwrap()).unwrap().to_string()
    }
    #[test]
    fn lowers_closure_borrowed_and_tuple_outputs() {
        let out = expand_trait(quote! {
            trait Ext: Backend {
                #[fusion(meta = |x, option| (x.clone(), x.clone()))]
                fn op(x: &FloatTensor<Self>, option: usize) -> (FloatTensor<Self>, FloatTensor<Self>);
            }
        });
        assert!(out.contains("with_scalars (stringify ! (op)"));
        assert!(out.contains("TensorSpec"));
        assert!(out.contains(":: resolve_inputs"));
        assert!(out.contains("validate_outputs"));
        assert!(out.contains("append_input_ir"));
        syn::parse_str::<syn::ItemImpl>(&out).unwrap();
    }
    #[test]
    fn preserves_explicit_optimizer_contract() {
        let out = expand_trait(quote! {
            trait Ext: Backend {
                #[fusion(id = "matmul", dtype = lhs, shape = lhs)]
                fn op(lhs: FloatTensor<Self>, #[fusion(scalar = strategy.to_code())] strategy: Strategy, flag: bool,
                    #[extension_type] rhs: Operand<Self>) -> FloatTensor<Self>;
            }
        });
        assert!(out.contains("with_scalars (\"matmul\""));
        let strategy = out
            .find("Scalar :: from ((strategy . to_code ()) . clone ())")
            .unwrap();
        let flag = out.find("Scalar :: from ((flag) . clone ())").unwrap();
        assert!(strategy < flag);
        assert!(!out.contains("fusion (scalar"));
        assert!(out.contains("ExtensionMetadata"));
        syn::parse_str::<syn::ItemImpl>(&out).unwrap();
    }

    #[test]
    fn scalar_parameter_annotations_are_validated_and_stripped() {
        for argument in [
            quote!(#[fusion()] value: f32),
            quote!(#[fusion(other)] value: f32),
            quote!(#[fusion(scalar, scalar)] value: f32),
            quote!(#[fusion(scalar)] #[fusion(scalar)] value: f32),
            quote!(#[fusion(scalar)] value: FloatTensor<Self>),
            quote!(#[fusion(scalar)] #[extension_type] value: Options<Self>),
        ] {
            let item = syn::parse2(quote! {
                trait Ext: Backend {
                    #[fusion(dtype = x, shape = x)]
                    fn op(x: FloatTensor<Self>, #argument) -> FloatTensor<Self>;
                }
            })
            .unwrap();
            assert!(expand(&item).is_err(), "accepted {argument}");
        }
        for backends in [quote!(Cube), quote!(Cube, Fusion: cfg(feature = "fusion"))] {
            let output = crate::extension::expand(backends, quote! {
                trait Ext: Backend {
                    #[fusion(dtype = x, shape = x)]
                    fn op(x: FloatTensor<Self>, #[fusion(scalar)] amount: Amount) -> FloatTensor<Self>;
                }
            }).unwrap();
            assert!(!output.to_string().contains("fusion (scalar"));
            syn::parse2::<syn::File>(output).unwrap();
        }
    }

    #[test]
    fn detects_primitive_scalars_and_requires_alias_opt_in() {
        for (ty, expected) in [
            (quote!(f32), true),
            (quote!(f64), true),
            (quote!(bool), true),
            (quote!(usize), true),
            (quote!(isize), true),
            (quote!(u64), true),
            (quote!(core::primitive::i16), true),
            (quote!(std::primitive::u8), true),
            (quote!(Amount), false),
            (quote!(options::u32), false),
            (quote!(String), false),
            (quote!(i128), false),
        ] {
            let arg = syn::parse2(quote!(value: #ty)).unwrap();
            assert_eq!(
                scalar_argument(&syn::parse_quote!(value), &arg)
                    .unwrap()
                    .is_some(),
                expected,
                "{ty}"
            );
        }
    }

    #[test]
    fn inherits_defaults() {
        let out = expand_trait(quote! {
            trait Ext: Backend {
                #[fusion(default)]
                fn default_op(x: FloatTensor<Self>) -> FloatTensor<Self> { x }
            }
        });
        assert!(!out.contains("fn default_op"));
        syn::parse_str::<syn::ItemImpl>(&out).unwrap();
    }
    #[test]
    fn rejects_incomplete_duplicate_and_conflicting_fields() {
        for attr in [
            quote!(dtype = x),
            quote!(shape = x),
            quote!(dtype = x, dtype = x, shape = x),
            quote!(shape = x, shape = x, dtype = x),
            quote!(meta = metadata, dtype = x, shape = x),
            quote!(dtype = x, shape = x, meta = metadata),
            quote!(default, dtype = x, shape = x),
            quote!(shape = x, default),
            quote!(default, id = "op"),
            quote!(default, scalars = []),
            quote!(dtype = x, shape = x, id = "a", id = "b"),
            quote!(dtype = x, shape = x, scalars = [], scalars = []),
        ] {
            let item = syn::parse2(quote! {
                trait Ext: Backend {
                    #[fusion(#attr)]
                    fn op(x: FloatTensor<Self>) -> FloatTensor<Self> { x }
                }
            })
            .unwrap();
            assert!(expand(&item).is_err(), "accepted {attr}");
        }
        let item = syn::parse2(quote! {
            trait Ext: Backend {
                #[fusion(dtype = x, shape = x)]
                fn op(x: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>);
            }
        })
        .unwrap();
        assert!(
            expand(&item)
                .unwrap_err()
                .to_string()
                .contains("single tensor output")
        );
    }
    #[test]
    fn rejects_missing_conflicting_and_unsupported_behaviors() {
        for method in [
            quote!(
                #[fusion(custom = helper)]
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(default)]
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta, default)]
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                async fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op<T>(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op(x: FloatTensor<Self>, v: &usize) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(dtype = x, shape = x)]
                fn op(x: &mut FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op(x: FloatTensor<Self>) -> Vec<FloatTensor<Self>>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op(x: usize) -> FloatTensor<Self>;
            ),
        ] {
            let item = syn::parse2(quote!(trait Ext: Backend { #method })).unwrap();
            assert!(expand(&item).is_err());
        }
    }
    #[test]
    fn gates_missing_annotations_and_named_metadata() {
        let out = crate::extension::expand(
            quote!(Cube, Fusion: cfg(feature = "fusion")),
            quote! {
                trait Ext: Backend { fn op(x: FloatTensor<Self>) -> FloatTensor<Self>; }
            },
        )
        .unwrap()
        .to_string();
        assert!(out.contains("cfg (feature = \"fusion\")"));
        assert!(out.contains("compile_error"));
        let input = syn::parse2(quote! {
            #[extension_type(fusion: cfg(feature = "fusion"))]
            pub struct Outer<B: Backend> { pub x: IntTensor<B>, #[extension_type] pub nested: Inner<B> }
        }).unwrap();
        let out = derive(&input).unwrap();
        syn::parse2::<syn::File>(out.clone()).unwrap();
        assert!(out.to_string().contains("ExtensionMetadata"));
    }
}
