//! Emit an extension trait's `Fusion<B>` implementation from its parsed [`LazyTrait`], and the
//! Fusion adapters of derived extension values.
//!
//! Metadata runs on the calling thread before registration; the inner backend call is deferred.
//! Tuples are traversed here, while tensors and derived extension values delegate to the runtime's
//! `FusionValueAdapter` trait for metadata flattening, validation, handle publication, and reconstruction.
use crate::ir::{TensorKind, with_backend};
use crate::parse::{ArgumentKind, DecodeCheck, LazyMethod, LazyTrait, unsupported};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{GenericParam, Type};

/// The `Fusion<B>` implementation of `model`, which reports the errors of its malformed methods.
/// `remote` is the remote backend's gate, when it is generated.
pub(crate) fn expand(model: &LazyTrait<'_>, remote: Option<&TokenStream>) -> TokenStream {
    let trait_name = model.name;
    let methods = model.methods.iter().map(|method| match method {
        Ok(method) => FusionMethod::new(method, remote).implementation(trait_name),
        Err(malformed) => malformed.reported(),
    });
    quote! {
        impl<B: burn::backend::fusion::FusionBackend + #trait_name> #trait_name for burn::backend::fusion::Fusion<B> {
            #(#methods)*
        }
    }
}

/// A lazily executed method as the `Fusion<B>` implementation registers it.
struct FusionMethod<'a> {
    method: &'a LazyMethod<'a>,
    /// The remote backend's gate, when it is generated: the IR then also carries each extension
    /// argument's variant and ordinary fields and each enum output's variant, so a graph fused for
    /// the remote backend reaches the server complete.
    remote: Option<&'a TokenStream>,
}

impl<'a> FusionMethod<'a> {
    fn new(method: &'a LazyMethod<'a>, remote: Option<&'a TokenStream>) -> Self {
        Self { method, remote }
    }

    /// The `Fusion<B>` method: register the operation and defer the inner backend call.
    fn implementation(&self, trait_name: &syn::Ident) -> TokenStream {
        let method = self.method;
        let mut metadata_args = Vec::new();
        let mut prepare = Vec::new();
        let mut retrieve = Vec::new();
        let mut invoke = Vec::new();
        let mut inputs = Vec::new();
        let mut metadata_inputs = Vec::new();
        let mut input_counts = Vec::new();
        let mut visits = Vec::new();
        for argument in &method.arguments {
            let name = argument.name;
            match argument.adapter(quote!(B)) {
                Some(adapter) => {
                    let borrowed =
                        matches!(argument.kind, ArgumentKind::Tensor { borrowed: true, .. });
                    let adapter =
                        quote!(<#adapter as burn::backend::fusion::custom::FusionValueAdapter<B>>);
                    let meta = format_ident!("__metadata_{name}");
                    let input = format_ident!("__input_{name}");
                    input_counts.push(quote!(#adapter::tensor_count(&#meta)));
                    metadata_inputs.push(quote!(let #meta = #adapter::to_metadata(&#name);));
                    metadata_args.push(quote!(&#meta));
                    visits.push(quote!(#adapter::visit_fused_tensors(&#name, &mut __visit);));
                    retrieve
                        .push(quote!(let #name = #adapter::resolve_inputs(&#input, __handles);));
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
                    inputs.push(
                        quote!(let #input = #adapter::append_input_ir(#value, &mut __inputs);),
                    );
                }
                None => {
                    metadata_args.push(quote!(&#name));
                    // Operation::execute borrows the closure, so the backend receives cloned options.
                    invoke.push(quote!(#name.clone()));
                    prepare.push(quote!(__capture(&#name);));
                }
            }
        }
        let scalars = method.scalars(DecodeCheck::Skip);
        let remote_scalars = self.remote.map(|gate| {
            let inputs = method.extension_scalars(quote!(B));
            let outputs = method.output_variants(&quote!(B));
            quote!(#gate { #(#inputs)* #(#outputs)* })
        });
        let mut output_counts = Vec::new();
        let mut validate = Vec::new();
        let mut publish = Vec::new();
        // Validation and publication are separate phases so no handles are published until all
        // outputs pass.
        let output_state = method.output.map_leaves(&quote!(B), &mut |adapter, member| {
            let adapter =
                quote!(<#adapter as burn::backend::fusion::custom::FusionValueAdapter<B>>);
            output_counts.push(quote!(#adapter::tensor_count(&__meta #member)));
            validate.push(quote!(#adapter::validate_outputs(&__output #member, &__output_state #member, {
                #[cfg(debug_assertions)] { Some(&__device) }
                #[cfg(not(debug_assertions))] { None }
            })?;));
            publish.push(quote!(#adapter::register_output_handles(__output #member, &__output_state #member, __handles);));
            quote!(#adapter::append_output_ir(&__meta #member, &mut __outputs, &mut || __client.create_empty_handle()))
        });
        let reconstruct = method.output.map_leaves(&quote!(B), &mut |adapter, member| {
            quote!(<#adapter as burn::backend::fusion::custom::FusionValueAdapter<B>>::build_fused_output(&__meta #member, &mut __tensors))
        });
        let call = method.call(trait_name, quote!(B), &invoke);
        let expr = method.metadata(quote!(B));
        let id = &method.id;
        let signature = &method.signature;
        let attrs = method.attrs();
        quote! {
            #(#attrs)* #signature {
                use burn::backend::fusion::custom as __fusion;
                fn __capture<T: Clone + Send + Sync + 'static>(_: &T) {}
                let mut __client = None;
                #[cfg(debug_assertions)]
                let mut __device_id = None;
                let mut __visit = |tensor: &burn::backend::fusion::FusionTensor<B::FusionRuntime>| {
                    #[cfg(debug_assertions)] {
                        let id = burn::backend::Device::to_id(tensor.client.device());
                        assert_eq!(*__device_id.get_or_insert(id), id, "Fusion custom inputs must share a device");
                    }
                    __client.get_or_insert_with(|| tensor.client.clone());
                };
                #(#visits)*
                let __client = __client.expect("Fusion custom operation requires at least one input tensor");
                #[cfg(debug_assertions)]
                let __device = __client.device().clone();
                #(#prepare)*
                #(#metadata_inputs)*
                let __meta = (#expr)(#(#metadata_args),*);
                #[allow(unused_mut)]
                let mut __scalars = vec![#(#scalars),*];
                #remote_scalars
                let mut __inputs = Vec::with_capacity(0 #(+ #input_counts)*);
                #(#inputs)*
                let mut __outputs = Vec::with_capacity(0 #(+ #output_counts)*);
                let __output_state = #output_state;
                let __desc = __fusion::CustomOpIr {
                    id: (#id).into(),
                    inputs: __inputs,
                    outputs: __outputs,
                    scalars: __scalars,
                };
                let __op = __fusion::OperationFn(move |__handles: &mut __fusion::HandleContainer<<B::FusionRuntime as burn::backend::fusion::FusionRuntime>::FusionHandle>| {
                    #(#retrieve)*
                    let __output = #call;
                    #(#validate)*
                    #(#publish)*
                    Ok(())
                });
                let mut __tensors = __client.register(__fusion::StreamId::current(), __fusion::OperationIr::Custom(__desc), __op).into_iter();
                #reconstruct
            }
        }
    }
}

pub(crate) fn derive(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let mut enabled = false;
    let mut remote = false;
    let mut cfg = None;
    for attr in input
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("extension_type"))
    {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("remote") && !remote {
                remote = true;
                return Ok(());
            }
            if !meta.path.is_ident("fusion") || enabled {
                return Err(meta.error("expected fusion, fusion: cfg(...), or remote"));
            }
            enabled = true;
            if meta.input.peek(syn::Token![:]) {
                meta.input.parse::<syn::Token![:]>()?;
                cfg = Some(meta.input.parse::<syn::Meta>()?);
            }
            Ok(())
        })?;
    }
    if remote && !enabled {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "remote extends the Fusion metadata; enable fusion beside it: \
             #[extension_type(fusion, remote)]",
        ));
    }
    reject_unread_scalar_encodings(input, remote)?;
    if !enabled {
        return Ok(quote!());
    }
    let gate = cfg.map(|c| quote!(#[#c]));
    let result = derive_adapter(input, &gate, remote).unwrap_or_else(|e| {
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

#[derive(Clone, Copy)]
enum FusionType {
    Metadata,
    Input,
    Output,
}

/// Preserve the value's structure while replacing tensor leaves with their phase-specific state.
fn fusion_type_definition(
    input: &syn::DeriveInput,
    name: syn::Ident,
    kind: FusionType,
) -> syn::Result<syn::DeriveInput> {
    let mut definition = input.clone();
    definition.ident = name;
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
        let is_ext = crate::derive::is_nested_extension(&field.attrs);
        if field_adapter(&field.ty, is_ext)?.is_some() {
            field.ty = if TensorKind::from_type(&field.ty).is_some() {
                match kind {
                    FusionType::Metadata => {
                        syn::parse_quote!(burn::backend::fusion::custom::TensorSpec)
                    }
                    FusionType::Input => syn::parse_quote!(burn::backend::fusion::custom::TensorIr),
                    FusionType::Output => {
                        syn::parse_quote!(burn::backend::fusion::custom::OutputTensor)
                    }
                }
            } else {
                let ty = with_backend(&field.ty, quote!(burn::backend::Dispatch));
                let associated = match kind {
                    FusionType::Metadata => quote!(Metadata),
                    FusionType::Input => quote!(Input),
                    FusionType::Output => quote!(Output),
                };
                syn::parse_quote!(<#ty as burn::backend::fusion::custom::ExtensionMetadata>::#associated)
            };
        } else if matches!(kind, FusionType::Output) {
            // Ordinary output fields are reconstructed on the caller, not retained for execution.
            field.ty = syn::parse_quote!(());
        }
        field.attrs.retain(|a| a.path().is_ident("doc"));
    }
    Ok(definition)
}

/// Mirror structs and enum variants in metadata and compact execution state.
fn derive_adapter(
    input: &syn::DeriveInput,
    gate: &Option<TokenStream>,
    remote: bool,
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
    let input_state = format_ident!("{name}FusionInput");
    let output_state = format_ident!("{name}FusionOutput");
    let definition = fusion_type_definition(input, metadata.clone(), FusionType::Metadata)?;
    let input_definition = fusion_type_definition(input, input_state.clone(), FusionType::Input)?;
    let output_definition =
        fusion_type_definition(input, output_state.clone(), FusionType::Output)?;
    let cases = collect_cases(input)?;
    let mut meta_cases = collect_cases(&definition)?;
    let mut input_cases = collect_cases(&input_definition)?;
    let mut output_cases = collect_cases(&output_definition)?;
    for case in meta_cases
        .iter_mut()
        .chain(&mut input_cases)
        .chain(&mut output_cases)
    {
        for field in &mut case.fields {
            field.bind = format_ident!("{}_state", field.bind);
        }
    }
    let mut describe = Vec::new();
    let mut visit = Vec::new();
    let mut counts = Vec::new();
    let mut flatten = Vec::new();
    let mut read = Vec::new();
    let mut outputs = Vec::new();
    let mut validate = Vec::new();
    let mut publish = Vec::new();
    let mut reconstruct = Vec::new();
    let is_enum = matches!(input.data, syn::Data::Enum(_));
    let mut remote = remote.then(|| crate::remote::RemoteAdapterDerive::new(name, b, is_enum));
    for (((case, meta_case), input_case), output_case) in cases
        .iter()
        .zip(&meta_cases)
        .zip(&input_cases)
        .zip(&output_cases)
    {
        let is_tensor = |i: usize| case.fields[i].tensor_kind.is_some() || case.fields[i].is_ext;
        let value_pattern = gen_case_pattern(case, |_| true);
        let tensor_pattern = gen_case_pattern(case, is_tensor);
        let meta_pattern = gen_case_pattern(meta_case, |_| true);
        let meta_tensor_pattern = gen_case_pattern(meta_case, is_tensor);
        let input_pattern = gen_case_pattern(input_case, |_| true);
        let output_pattern = gen_case_pattern(output_case, is_tensor);
        let mut descriptions = Vec::new();
        let mut visits = Vec::new();
        let mut tensor_counts = Vec::new();
        let mut inputs = Vec::new();
        let mut reads = Vec::new();
        let mut output_fields = Vec::new();
        let mut checks = Vec::new();
        let mut publications = Vec::new();
        let mut reconstructions = Vec::new();
        for (field, meta_field) in case.fields.iter().zip(&meta_case.fields) {
            let value = &field.bind;
            let state = &meta_field.bind;
            if let Some(adapter) = field_adapter(&field.ty, field.is_ext)? {
                if let Some(remote) = &mut remote {
                    remote.adapted_field(&adapter, value, state);
                }
                let adapter =
                    quote!(<#adapter as burn::backend::fusion::custom::FusionValueAdapter<#b>>);
                descriptions.push(quote!(#adapter::to_metadata(#value)));
                visits.push(quote!(#adapter::visit_fused_tensors(#value, visit);));
                tensor_counts.push(quote!(#adapter::tensor_count(#state)));
                inputs.push(quote!(#adapter::append_input_ir(#value, inputs)));
                reads.push(quote!(#adapter::resolve_inputs(#state, handles)));
                output_fields.push(quote!(#adapter::append_output_ir(#state, out, create)));
                checks.push(quote!(#adapter::validate_outputs(#value, #state, device)?;));
                publications
                    .push(quote!(#adapter::register_output_handles(#value, #state, handles);));
                reconstructions.push(quote!(#adapter::build_fused_output(#state, tensors)));
            } else {
                if let Some(remote) = &mut remote {
                    remote.ordinary_field(field, value, state)?;
                }
                descriptions.push(quote!(#value.clone()));
                inputs.push(quote!(#value));
                reads.push(quote!(#state.clone()));
                output_fields.push(quote!(()));
                reconstructions.push(quote!(#state.clone()));
            }
        }
        let description = gen_case_ctor(meta_case, &descriptions);
        let input_state = gen_case_ctor(input_case, &inputs);
        let output_state = gen_case_ctor(output_case, &output_fields);
        let restored = gen_case_ctor(case, &reads);
        let reconstructed = gen_case_ctor(case, &reconstructions);
        describe.push(quote!(#value_pattern => #description));
        visit.push(quote!(#tensor_pattern => { #(#visits)* }));
        counts.push(quote!(#meta_tensor_pattern => { 0 #(+ #tensor_counts)* }));
        flatten.push(quote!(#value_pattern => #input_state));
        read.push(quote!(#input_pattern => #restored));
        outputs.push(quote!(#meta_tensor_pattern => #output_state));
        validate.push(quote!((#tensor_pattern, #output_pattern) => { #(#checks)* Ok(()) }));
        publish.push(quote!((#tensor_pattern, #output_pattern) => { #(#publications)* }));
        reconstruct.push(quote!(#meta_pattern => #reconstructed));
        if let Some(remote) = &mut remote {
            remote.end_case(case, meta_case);
        }
    }
    let mismatch = is_enum.then(|| quote! {
        _ => Err(burn::backend::fusion::ExecutionError::generic("Fusion custom output variant differs from its metadata")),
    });
    let unreachable =
        is_enum.then(|| quote! { _ => unreachable!("output variants were validated"), });
    let bounds = &backend.bounds;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let remote = remote.map(|remote| remote.finish(input, backend, &metadata, gate));
    Ok(quote! {
        #remote
        #gate
        #[doc = "Tensor metadata and ordinary fields for the corresponding backend extension value."]
        #[derive(Clone, Debug)]
        #definition
        #gate
        #[doc(hidden)]
        #input_definition
        #gate
        #[doc(hidden)]
        #output_definition
        #gate
        impl #impl_generics burn::backend::fusion::custom::ExtensionMetadata for #name #ty_generics #where_clause {
            type Metadata = #metadata;
            type Input = #input_state;
            type Output = #output_state;
        }
        #gate
        impl<#b: burn::backend::fusion::FusionBackend + #bounds> burn::backend::fusion::custom::FusionValueAdapter<#b> for #name<#b> #where_clause {
            type Metadata = #metadata;
            type Input = #input_state;
            type Output = #output_state;
            type Inner = Self;
            type Fused = #name<burn::backend::fusion::Fusion<#b>>;
            fn to_metadata(value: &Self::Fused) -> Self::Metadata { match value { #(#describe,)* } }
            fn visit_fused_tensors(value: &Self::Fused, visit: &mut impl FnMut(&burn::backend::fusion::FusionTensor<#b::FusionRuntime>)) { match value { #(#visit,)* } }
            fn tensor_count(meta: &Self::Metadata) -> usize { match meta { #(#counts,)* } }
            fn append_input_ir(value: Self::Fused, inputs: &mut Vec<burn::backend::fusion::custom::TensorIr>) -> Self::Input { match value { #(#flatten,)* } }
            fn resolve_inputs(input: &Self::Input, handles: &mut burn::backend::fusion::custom::HandleContainer<<#b::FusionRuntime as burn::backend::fusion::FusionRuntime>::FusionHandle>) -> Self::Inner { match input { #(#read,)* } }
            fn append_output_ir(meta: &Self::Metadata, out: &mut Vec<burn::backend::fusion::custom::TensorIr>, create: &mut impl FnMut() -> burn::backend::fusion::custom::TensorId) -> Self::Output { match meta { #(#outputs,)* } }
            fn validate_outputs(value: &Self, output: &Self::Output, device: Option<&#b::Device>) -> Result<(), burn::backend::fusion::ExecutionError> {
                match (value, output) { #(#validate,)* #mismatch }
            }
            fn register_output_handles(value: Self, output: &Self::Output, handles: &mut burn::backend::fusion::custom::HandleContainer<<#b::FusionRuntime as burn::backend::fusion::FusionRuntime>::FusionHandle>) { match (value, output) { #(#publish,)* #unreachable } }
            fn build_fused_output(meta: &Self::Metadata, tensors: &mut std::vec::IntoIter<burn::backend::fusion::FusionTensor<#b::FusionRuntime>>) -> Self::Fused { match meta { #(#reconstruct,)* } }
        }
    })
}

/// Reject `#[extension_type(scalar ..)]` where nothing reads it: on a tensor or nested field,
/// which travels as IR inputs, or on any field of a type without `remote`.
fn reject_unread_scalar_encodings(input: &syn::DeriveInput, remote: bool) -> syn::Result<()> {
    for case in crate::derive::collect_cases(input)? {
        for field in &case.fields {
            let Some(annotation) = field.scalar_annotation() else {
                continue;
            };
            if !remote {
                return Err(syn::Error::new_spanned(
                    annotation,
                    "a scalar encoding sends the field to the remote backend; opt the type in \
                     with #[extension_type(fusion, remote)]",
                ));
            }
            if field.is_ext || field.tensor_kind.is_some() {
                return Err(syn::Error::new_spanned(
                    annotation,
                    "a scalar encoding applies to ordinary fields; tensor and nested extension \
                     fields travel as the operation's input tensors",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn expand_trait(tokens: TokenStream) -> String {
        let item = syn::parse2(tokens).unwrap();
        expand(&LazyTrait::new(&item), None).to_string()
    }
    #[test]
    fn lowers_closure_borrowed_and_tuple_outputs() {
        let out = expand_trait(quote! {
            trait Ext: Backend {
                #[fusion(meta = |x, option| (x.clone(), x.clone()))]
                fn op(x: &FloatTensor<Self>, option: usize) -> (FloatTensor<Self>, FloatTensor<Self>);
            }
        });
        assert!(out.contains("id : (concat ! (stringify ! (Ext) , \"::\" , stringify ! (op)))"));
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
                fn op(lhs: FloatTensor<Self>, #[fusion(scalar = mode.to_code())] mode: Mode, flag: bool,
                    #[extension_type] rhs: Operand<Self>) -> FloatTensor<Self>;
            }
        });
        assert!(out.contains("id : (\"matmul\")"));
        let mode = out
            .find("Scalar :: from ((mode . to_code ()) . clone ())")
            .unwrap();
        let flag = out.find("Scalar :: from ((flag) . clone ())").unwrap();
        assert!(mode < flag);
        assert!(!out.contains("fusion (scalar"));
        assert!(out.contains("ExtensionMetadata"));
        syn::parse_str::<syn::ItemImpl>(&out).unwrap();
    }

    #[test]
    fn strips_scalar_parameter_annotations() {
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

    #[test]
    fn rejects_scalar_encodings_nothing_reads() {
        for (input, message) in [
            (
                quote! {
                    #[extension_type(fusion)]
                    pub struct Store<B: Backend> {
                        buffer: FloatTensor<B>,
                        #[extension_type(scalar = bits.to_code(), decode = Bits::from_code)]
                        bits: Bits,
                    }
                },
                "opt the type in",
            ),
            (
                quote! {
                    pub struct Store<B: Backend>(
                        FloatTensor<B>,
                        #[extension_type(scalar = value.0, decode = Factor)] Factor,
                    );
                },
                "opt the type in",
            ),
            (
                quote! {
                    #[extension_type(fusion, remote)]
                    pub struct Store<B: Backend> {
                        #[extension_type(scalar = 0u64, decode = |_| unreachable!())]
                        buffer: FloatTensor<B>,
                    }
                },
                "applies to ordinary fields",
            ),
            (
                quote! {
                    #[extension_type(fusion, remote)]
                    pub enum Store<B: Backend> {
                        Nested {
                            #[extension_type]
                            #[extension_type(scalar = 0u64, decode = |_| unreachable!())]
                            inner: Inner<B>,
                        },
                    }
                },
                "applies to ordinary fields",
            ),
        ] {
            let error = derive(&syn::parse2(input.clone()).unwrap()).unwrap_err();
            assert!(error.to_string().contains(message), "{input}: {error}");
        }
    }
}
