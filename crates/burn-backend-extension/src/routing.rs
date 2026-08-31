//! Shared routing code generation for backend dispatch and backend extensions.
//!
//! Both attribute macros normalize a method into [`Operation`] and use this module for the routing
//! contract. Generated code selects a concrete backend and autodiff context from one routing tensor,
//! unwraps every input for the selected backend, invokes the operation, and wraps its output back
//! into dispatch tensors. All tensor-bearing inputs are required to share that context; routing
//! deliberately doesn't scan them to validate the contract at runtime. Backend selection remains an
//! enum match: this layer does not introduce trait-object dispatch or backend lookup tables.
//!
//! A required bare tensor can be matched and extracted directly. Operations containing only
//! optional, vector, quantization-parameter, or extension inputs first search those inputs for a
//! routing tensor because their runtime value may contain no tensor at all.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ir::{
    self, InputKind, Invocation, Operation, OperationInput, OperationOutput, TensorKind,
};

/// How a generated operation's output tensor is represented at the dispatch boundary.
#[derive(Clone, Copy)]
pub(crate) enum OutputRoute {
    /// A concrete primitive with disabled autodiff metadata.
    ConcreteDisabled,
    /// A concrete primitive carried by an enabled autodiff context.
    ConcreteEnabled,
    /// An autodiff primitive carried by an enabled autodiff context.
    Autodiff,
}

/// Whether the active operation inputs contain a float tensor.
///
/// Dispatch can decide this statically for the common required-tensor cases. Operations whose
/// floats are confined to optional or vector inputs defer only this boolean to runtime.
pub(crate) enum FloatInputPresence {
    /// At least one float input is unconditionally present.
    Always,
    /// The signature cannot carry a float input.
    Never,
    /// Float presence depends on the active runtime inputs.
    Runtime(TokenStream),
}

/// Paths used by generated code.
///
/// `backend_dispatch` expands inside `burn-dispatch`, while backend extensions expand in downstream
/// crates through the public `burn::backend` facade. Keeping the paths explicit lets both frontends
/// share the routing mechanics without coupling their generated public representation.
pub(crate) struct RoutingPaths {
    /// Internal backend alias used while invoking the selected backend implementation.
    backend_alias: syn::Ident,
    /// Module containing the concrete backend types and the `Autodiff` wrapper.
    backend_root: TokenStream,
    /// Facade containing all dispatch representation types and checkpointing metadata.
    dispatch_root: TokenStream,
    /// `AutodiffBackend` is not re-exported by the internal dispatch facade.
    autodiff_trait: TokenStream,
    /// Only extension operations use this path; their expansion targets a downstream crate.
    extension_trait: TokenStream,
    /// The internal and downstream facades expose quantization parameters at different paths.
    quantization_parameters: TokenStream,
    /// Module containing the concrete checkpoint strategy types.
    checkpoint_root: TokenStream,
}

impl RoutingPaths {
    pub(crate) fn dispatch() -> Self {
        Self {
            backend_alias: format_ident!("B"),
            backend_root: quote!(crate::backends),
            dispatch_root: quote!(crate),
            autodiff_trait: quote!(burn_backend::AutodiffBackend),
            extension_trait: quote!(burn::backend::ExtensionType),
            quantization_parameters: quote!(
                burn_backend::quantization::QuantizationParametersPrimitive
            ),
            checkpoint_root: quote!(burn_autodiff::checkpoint::strategy),
        }
    }

    pub(crate) fn extension() -> Self {
        Self {
            backend_alias: format_ident!("_B"),
            backend_root: quote!(burn::backend),
            dispatch_root: quote!(burn::backend),
            autodiff_trait: quote!(burn::backend::AutodiffBackend),
            extension_trait: quote!(burn::backend::ExtensionType),
            quantization_parameters: quote!(burn::backend::QuantizationParametersPrimitive),
            checkpoint_root: quote!(burn::backend::autodiff::checkpoint::strategy),
        }
    }
}

/// Configuration for the nested concrete-backend match inside an autodiff dispatch kind.
pub(crate) struct AutodiffKindMatch {
    pub(crate) cfg_attr: Option<TokenStream>,
    pub(crate) inner_kind: TokenStream,
    pub(crate) nested_autodiff: Option<TokenStream>,
    pub(crate) fallback: TokenStream,
}

/// Concrete and autodiff arms for matching a dispatch tensor's backend kind.
pub(crate) struct BackendKindArms {
    pub(crate) concrete: Vec<TokenStream>,
    pub(crate) autodiff: Option<TokenStream>,
}

/// Generate the backend-kind match arms shared by dispatch and backend extensions.
///
/// The caller supplies only the body for each backend. This keeps backend cfg attributes, autodiff
/// nesting, payload binding, and unsupported-backend fallbacks identical across both frontends.
pub(crate) fn backend_kind_arms(
    paths: &RoutingPaths,
    backends: &[ir::Backend],
    payload: Option<&syn::Ident>,
    autodiff: Option<AutodiffKindMatch>,
    mut body: impl FnMut(usize, &ir::Backend, bool) -> TokenStream,
) -> BackendKindArms {
    let dispatch_root = &paths.dispatch_root;
    let dispatch_kind = quote!(#dispatch_root::DispatchTensorKind);
    let payload = payload
        .map(|payload| quote!(#payload))
        .unwrap_or_else(|| quote!(_));
    let concrete = backends
        .iter()
        .enumerate()
        .map(|(index, backend)| {
            let cfg_attr = &backend.cfg_attr;
            let ident = &backend.ident;
            let body = body(index, backend, false);
            quote! {
                #cfg_attr
                #dispatch_kind::#ident(#payload) => #body,
            }
        })
        .collect();
    let autodiff = autodiff.map(|autodiff| {
        let cfg_attr = autodiff.cfg_attr;
        let inner_kind = autodiff.inner_kind;
        let nested_autodiff = autodiff
            .nested_autodiff
            .map(|body| quote!(#dispatch_kind::Autodiff(_) => #body,));
        let fallback = autodiff.fallback;
        let inner = backends.iter().enumerate().map(|(index, backend)| {
            let backend_cfg = &backend.cfg_attr;
            let ident = &backend.ident;
            let body = body(index, backend, true);
            quote! {
                #backend_cfg
                #dispatch_kind::#ident(#payload) => #body,
            }
        });
        quote! {
            #cfg_attr
            #dispatch_kind::Autodiff(__inner) => match #inner_kind {
                #(#inner)*
                #nested_autodiff
                #[allow(unreachable_patterns)]
                _ => #fallback,
            }
        }
    });

    BackendKindArms { concrete, autodiff }
}

/// Select the concrete autodiff backend type from its runtime checkpointing strategy.
pub(crate) fn with_autodiff_backend(
    paths: &RoutingPaths,
    backend: &syn::Ident,
    strategy_expr: TokenStream,
    alias: &syn::Ident,
    body: TokenStream,
) -> TokenStream {
    let RoutingPaths {
        backend_root,
        dispatch_root,
        checkpoint_root,
        ..
    } = paths;

    quote! {
        match #strategy_expr {
            #dispatch_root::GradientCheckpointingStrategy::Balanced => {
                type #alias = #backend_root::Autodiff<#backend_root::#backend, #checkpoint_root::BalancedCheckpointing>;
                #body
            }
            #dispatch_root::GradientCheckpointingStrategy::Disabled => {
                type #alias = #backend_root::Autodiff<#backend_root::#backend, #checkpoint_root::NoCheckpointing>;
                #body
            }
        }
    }
}

/// Wrap one concrete or autodiff tensor primitive in its dispatch enum representation.
pub(crate) fn wrap_tensor_kind(
    paths: &RoutingPaths,
    kind: TensorKind,
    backend: &syn::Ident,
    route: OutputRoute,
    value: TokenStream,
    autodiff_backend: &syn::Ident,
) -> TokenStream {
    let RoutingPaths {
        dispatch_root,
        autodiff_trait,
        ..
    } = paths;
    let variant = kind.variant();

    if kind == TensorKind::Float && matches!(route, OutputRoute::Autodiff) {
        quote! {
            #dispatch_root::DispatchTensorKind::Autodiff(
                #dispatch_root::DispatchTensorKind::#backend(
                    #dispatch_root::BackendTensor::Autodiff(#value)
                ).into()
            )
        }
    } else if kind == TensorKind::Float && matches!(route, OutputRoute::ConcreteEnabled) {
        quote! {
            #dispatch_root::DispatchTensorKind::Autodiff(
                #dispatch_root::DispatchTensorKind::#backend(
                    #dispatch_root::BackendTensor::Autodiff(
                        <#autodiff_backend as #autodiff_trait>::from_inner(#value)
                    )
                ).into()
            )
        }
    } else {
        quote! { #dispatch_root::DispatchTensorKind::#backend(#dispatch_root::BackendTensor::#variant(#value)) }
    }
}

enum RoutingCandidate {
    /// An expression of type `&DispatchTensor`.
    Required(TokenStream),
    /// An expression of type `Option<&DispatchTensor>`.
    Optional(TokenStream),
}

pub(crate) enum BackendSelection<'a> {
    /// Select and extract the backend directly from a tensor input.
    FromInput(&'a OperationInput),
    /// Select the backend through the operation's routing candidates.
    FromCandidates,
}

pub(crate) fn backend_selection(operation: &Operation) -> BackendSelection<'_> {
    operation
        .inputs
        .iter()
        .find(|input| matches!(input.kind, InputKind::Tensor { .. }))
        .map(BackendSelection::FromInput)
        .unwrap_or(BackendSelection::FromCandidates)
}

enum CandidateSelection {
    Missing,
    Optional(TokenStream),
    Required(TokenStream),
}

fn select_candidate(candidates: Vec<RoutingCandidate>) -> CandidateSelection {
    let mut optional = Vec::new();
    for candidate in candidates {
        match candidate {
            RoutingCandidate::Optional(candidate) => optional.push(candidate),
            RoutingCandidate::Required(candidate) => {
                if optional.is_empty() {
                    return CandidateSelection::Required(candidate);
                }
                let optional = option_chain(optional);
                return CandidateSelection::Required(quote!((#optional).unwrap_or(#candidate)));
            }
        }
    }

    if optional.is_empty() {
        CandidateSelection::Missing
    } else {
        CandidateSelection::Optional(option_chain(optional))
    }
}

fn option_chain(candidates: Vec<TokenStream>) -> TokenStream {
    match candidates.split_first() {
        None => quote!(Option::None),
        Some((first, rest)) => quote!(#first #(.or_else(|| #rest))*),
    }
}

/// Generate `__routing_tensor` and `__routing_tensor_is_float` bindings from ordered candidates.
///
/// Float candidates are preferred, then any tensor candidate. Optional tensors and empty vectors
/// fall through instead of making backend selection depend on one statically chosen container.
fn routing_tensor_init(
    float_candidates: Vec<RoutingCandidate>,
    any_candidates: Vec<RoutingCandidate>,
    tensor_ty: TokenStream,
    missing_message: &str,
) -> TokenStream {
    let float = select_candidate(float_candidates);
    let any = select_candidate(any_candidates);
    let any_fallback = match any {
        CandidateSelection::Required(expression) => expression,
        CandidateSelection::Optional(expression) => quote!((#expression).expect(#missing_message)),
        CandidateSelection::Missing => quote!(Option::None.expect(#missing_message)),
    };

    match float {
        CandidateSelection::Required(expression) => quote! {
            let __routing_tensor: &#tensor_ty = #expression;
            let __routing_tensor_is_float = true;
        },
        CandidateSelection::Missing => quote! {
            let __routing_tensor: &#tensor_ty = #any_fallback;
            let __routing_tensor_is_float = false;
        },
        CandidateSelection::Optional(expression) => quote! {
            let __float_routing_tensor: Option<&#tensor_ty> = #expression;
            let (__routing_tensor, __routing_tensor_is_float) = match __float_routing_tensor {
                Some(__routing_tensor) => (__routing_tensor, true),
                None => (#any_fallback, false),
            };
        },
    }
}

pub(crate) fn operation_routing_tensor(
    operation: &Operation,
    paths: &RoutingPaths,
    missing_message: &str,
) -> TokenStream {
    let dispatch_root = &paths.dispatch_root;
    let candidates = |float_only| {
        operation
            .inputs
            .iter()
            .filter_map(|input| routing_candidate(input, paths, float_only))
            .collect()
    };
    routing_tensor_init(
        candidates(true),
        candidates(false),
        quote!(#dispatch_root::DispatchTensor),
        missing_message,
    )
}

fn routing_candidate(
    input: &OperationInput,
    paths: &RoutingPaths,
    float_only: bool,
) -> Option<RoutingCandidate> {
    let name = &input.name;
    match &input.kind {
        InputKind::Tensor { kind, borrowed } if !float_only || *kind == TensorKind::Float => {
            let tensor = if *borrowed {
                quote!(#name)
            } else {
                quote!(&#name)
            };
            Some(RoutingCandidate::Required(tensor))
        }
        InputKind::OptionTensor(kind) if !float_only || *kind == TensorKind::Float => {
            Some(RoutingCandidate::Optional(quote!(#name.as_ref())))
        }
        InputKind::VecTensor(kind) if !float_only || *kind == TensorKind::Float => {
            Some(RoutingCandidate::Optional(quote!(#name.first())))
        }
        InputKind::QuantizationParameters => {
            Some(RoutingCandidate::Required(quote!(&#name.scales)))
        }
        InputKind::Extension(ty) => {
            let dispatch_root = &paths.dispatch_root;
            let dispatch_backend = quote!(#dispatch_root::Dispatch);
            let extension_trait = &paths.extension_trait;
            let target = ir::with_backend(ty, dispatch_backend.clone());
            let method = if float_only {
                format_ident!("routing_float_tensor")
            } else {
                format_ident!("routing_tensor")
            };
            Some(RoutingCandidate::Optional(quote! {
                <#target as #extension_trait<#dispatch_backend>>::#method(&#name)
            }))
        }
        InputKind::Tensor { .. }
        | InputKind::OptionTensor(_)
        | InputKind::VecTensor(_)
        | InputKind::Other => None,
    }
}

pub(crate) fn invoke(operation: &Operation, paths: &RoutingPaths) -> TokenStream {
    match &operation.invocation {
        Invocation::Body(body) => quote!(let __output = #body;),
        Invocation::Trait {
            trait_name,
            await_call,
            unsafe_call,
            generic_args,
        } => {
            let backend_alias = &paths.backend_alias;
            let name = &operation.name;
            let args = operation.inputs.iter().map(|input| &input.name);
            let generic_args = (!generic_args.is_empty()).then(|| quote!(::<#(#generic_args),*>));
            let call = quote!(<#backend_alias as #trait_name>::#name #generic_args (#(#args),*));
            let call = if *unsafe_call {
                quote!(unsafe { #call })
            } else {
                call
            };
            let await_output = await_call.then(|| quote!(.await));
            quote! {
                let __output = #call #await_output;
            }
        }
    }
}

fn extract_inputs(
    operation: &Operation,
    paths: &RoutingPaths,
    backend: &syn::Ident,
    autodiff: bool,
    autodiff_cfg: Option<&TokenStream>,
    has_autodiff_variant: bool,
    selected: Option<SelectedInput<'_>>,
) -> TokenStream {
    let extraction = Extraction {
        paths,
        backend,
        autodiff,
        autodiff_cfg,
        has_autodiff_variant,
    };
    let inputs = operation.inputs.iter().map(|input| {
        if let Some(selected) = selected
            && core::ptr::eq(input, selected.input)
        {
            return extract_selected_input(input, &extraction, selected.autodiff_variant);
        }
        extract_input(input, &extraction)
    });
    quote!(#(#inputs)*)
}

#[derive(Clone, Copy)]
struct SelectedInput<'a> {
    input: &'a OperationInput,
    autodiff_variant: bool,
}

struct Extraction<'a> {
    paths: &'a RoutingPaths,
    backend: &'a syn::Ident,
    autodiff: bool,
    autodiff_cfg: Option<&'a TokenStream>,
    has_autodiff_variant: bool,
}

fn extract_selected_input(
    input: &OperationInput,
    extraction: &Extraction<'_>,
    autodiff_variant: bool,
) -> TokenStream {
    let InputKind::Tensor { kind, borrowed } = input.kind else {
        unreachable!("selected input must be a tensor")
    };
    let name = &input.name;
    let selected = format_ident!("__burn_selected");
    let dispatch_root = &extraction.paths.dispatch_root;
    let backend_alias = &extraction.paths.backend_alias;
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    assert!(
        kind == TensorKind::Float || !autodiff_variant,
        "only a float input can directly select an autodiff primitive"
    );
    let validate_context = if kind == TensorKind::Float && autodiff_variant {
        quote! {
            let #context::Enabled(_) = __burn_selected_context else {
                panic!("an autodiff float primitive must have an enabled autodiff context")
            };
        }
    } else if kind == TensorKind::Float {
        quote! {
            let #context::Disabled = __burn_selected_context else {
                panic!("an enabled float tensor must use an autodiff primitive")
            };
        }
    } else {
        TokenStream::new()
    };

    if kind == TensorKind::Float && extraction.autodiff && !autodiff_variant {
        let autodiff_trait = &extraction.paths.autodiff_trait;
        if borrowed {
            let lifted = format_ident!("__lifted_{name}");
            quote! {
                #validate_context
                let #lifted = <#backend_alias as #autodiff_trait>::from_inner(#selected.as_float().clone());
                let #name = &#lifted;
            }
        } else {
            quote! {
                #validate_context
                let #name = <#backend_alias as #autodiff_trait>::from_inner(#selected.float());
            }
        }
    } else {
        let accessor = if kind == TensorKind::Float && extraction.autodiff {
            format_ident!("autodiff")
        } else {
            kind.accessor()
        };
        let accessor = if borrowed {
            format_ident!("as_{accessor}")
        } else {
            accessor
        };
        quote!(#validate_context let #name = #selected.#accessor();)
    }
}

fn extract_input(input: &OperationInput, extraction: &Extraction<'_>) -> TokenStream {
    let name = &input.name;
    match &input.kind {
        InputKind::Tensor { kind, borrowed } => extract_tensor(name, *kind, *borrowed, extraction),
        InputKind::OptionTensor(kind) => {
            let tensor = format_ident!("__tensor");
            let extract = extract_tensor(&tensor, *kind, false, extraction);
            quote!(let #name = #name.map(|__tensor| { #extract __tensor });)
        }
        InputKind::VecTensor(kind) => {
            let tensor = format_ident!("__tensor");
            let extract = extract_tensor(&tensor, *kind, false, extraction);
            // Only the backend-dispatch frontend constructs `VecTensor`; `burn-dispatch` exposes
            // `alloc`, so this path never leaks into a downstream extension expansion.
            quote! {
                let #name = #name.into_iter().map(|__tensor| {
                    #extract
                    __tensor
                }).collect::<alloc::vec::Vec<_>>();
            }
        }
        InputKind::QuantizationParameters => {
            let parameters = &extraction.paths.quantization_parameters;
            let scales = format_ident!("__scales");
            let tensor = format_ident!("__tensor");
            let extract_scales = extract_tensor(&scales, TensorKind::Float, false, extraction);
            let extract_global = extract_tensor(&tensor, TensorKind::Float, false, extraction);
            quote! {
                let #parameters { scales: __scales, global: __global } = #name;
                #extract_scales
                let __global = __global.map(|__tensor| { #extract_global __tensor });
                let #name = #parameters { scales: __scales, global: __global };
            }
        }
        InputKind::Extension(ty) => extract_extension(
            name,
            ty,
            extraction.paths,
            extraction.backend,
            extraction.autodiff,
        ),
        InputKind::Other => TokenStream::new(),
    }
}

fn cfg_attr(cfg: Option<&TokenStream>) -> TokenStream {
    cfg.cloned().unwrap_or_default()
}

fn extract_tensor(
    name: &syn::Ident,
    kind: TensorKind,
    borrowed: bool,
    extraction: &Extraction<'_>,
) -> TokenStream {
    let backend = extraction.backend;
    let backend_alias = &extraction.paths.backend_alias;
    let dispatch_root = &extraction.paths.dispatch_root;
    let dispatch_kind = quote!(#dispatch_root::DispatchTensorKind);
    let autodiff_trait = &extraction.paths.autodiff_trait;
    let ad_cfg = cfg_attr(extraction.autodiff_cfg);
    let accessor = kind.accessor();
    let accessor_ref = format_ident!("as_{accessor}");
    let invalid_autodiff_float = extraction.has_autodiff_variant.then(|| {
        quote! {
            #ad_cfg
            #dispatch_kind::Autodiff(_) => panic!("autodiff float input reached concrete dispatch"),
        }
    });
    let invalid_autodiff_kind = extraction.has_autodiff_variant.then(|| quote! {
        #ad_cfg
        #dispatch_kind::Autodiff(_) => panic!("only float tensors may use an autodiff primitive"),
    });

    if kind == TensorKind::Float && extraction.autodiff {
        if borrowed {
            let lifted = format_ident!("__lifted_{name}");
            quote! {
                let #lifted;
                let #name = match &#name.kind {
                    #dispatch_kind::Autodiff(__inner) => {
                        match __inner.as_ref() {
                            #dispatch_kind::#backend(__inner) => __inner.as_autodiff(),
                            _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                        }
                    }
                    #dispatch_kind::#backend(__inner) => {
                        #lifted = <#backend_alias as #autodiff_trait>::from_inner(__inner.as_float().clone());
                        &#lifted
                    }
                    _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                };
            }
        } else {
            quote! {
                let #name = match #name.kind {
                    #dispatch_kind::Autodiff(__inner) => {
                        match *__inner {
                            #dispatch_kind::#backend(__inner) => __inner.autodiff(),
                            _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                        }
                    }
                    #dispatch_kind::#backend(__inner) => {
                        <#backend_alias as #autodiff_trait>::from_inner(__inner.float())
                    }
                    _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                };
            }
        }
    } else if kind == TensorKind::Float {
        if borrowed {
            quote! {
                let #name = match &#name.kind {
                    #dispatch_kind::#backend(__inner) => __inner.as_float(),
                    #invalid_autodiff_float
                    _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                };
            }
        } else {
            quote! {
                let #name = match #name.kind {
                    #dispatch_kind::#backend(__inner) => __inner.float(),
                    #invalid_autodiff_float
                    _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                };
            }
        }
    } else if borrowed {
        quote! {
            let #name = match &#name.kind {
                #dispatch_kind::#backend(__inner) => __inner.#accessor_ref(),
                #invalid_autodiff_kind
                _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
            };
        }
    } else {
        quote! {
            let #name = match #name.kind {
                #dispatch_kind::#backend(__inner) => __inner.#accessor(),
                #invalid_autodiff_kind
                _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
            };
        }
    }
}

fn extract_extension(
    name: &syn::Ident,
    ty: &syn::Type,
    paths: &RoutingPaths,
    backend: &syn::Ident,
    autodiff: bool,
) -> TokenStream {
    let backend_root = &paths.backend_root;
    let dispatch_root = &paths.dispatch_root;
    let dispatch_kind = quote!(#dispatch_root::DispatchTensorKind);
    let backend_tensor = quote!(#dispatch_root::BackendTensor);
    let autodiff_trait = &paths.autodiff_trait;
    let extension_trait = &paths.extension_trait;
    let backend_alias = &paths.backend_alias;
    let mismatch = quote!(panic!("backend extension input is on the wrong backend"));
    if autodiff {
        let target = ir::with_backend(ty, quote!(#backend_alias));
        quote! {
            let #name = <#target as #extension_trait<#backend_alias>>::map_from_dispatch(#name, |kind| {
                let tensor = match kind {
                    #dispatch_kind::Autodiff(inner) => match *inner {
                        #dispatch_kind::#backend(tensor) => tensor,
                        #[allow(unreachable_patterns)]
                        _ => #mismatch,
                    },
                    #dispatch_kind::#backend(tensor) => tensor,
                    #[allow(unreachable_patterns)]
                    _ => #mismatch,
                };
                match tensor {
                    #backend_tensor::Autodiff(tensor) => #backend_tensor::Float(tensor),
                    #backend_tensor::Float(tensor) => #backend_tensor::Float(
                        <#backend_alias as #autodiff_trait>::from_inner(tensor)
                    ),
                    #backend_tensor::Int(tensor) => #backend_tensor::Int(tensor),
                    #backend_tensor::Bool(tensor) => #backend_tensor::Bool(tensor),
                    #backend_tensor::Quantized(tensor) => #backend_tensor::Quantized(tensor),
                }
            });
        }
    } else {
        let target = ir::with_backend(ty, quote!(#backend_root::#backend));
        quote! {
            let #name = <#target as #extension_trait<#backend_root::#backend>>::map_from_dispatch(
                #name,
                |kind| match kind {
                    #dispatch_kind::#backend(tensor) => tensor,
                    #[allow(unreachable_patterns)]
                    _ => #mismatch,
                },
            );
        }
    }
}

pub(crate) fn wrap_output(
    output: &OperationOutput,
    paths: &RoutingPaths,
    backend: &syn::Ident,
    route: OutputRoute,
    value: TokenStream,
) -> TokenStream {
    match output {
        OperationOutput::Plain => value,
        OperationOutput::Tensor(kind) => {
            let dispatch_root = &paths.dispatch_root;
            let wrapped =
                wrap_tensor_kind(paths, *kind, backend, route, value, &format_ident!("AD"));
            quote!(#dispatch_root::DispatchTensor { kind: #wrapped, autodiff: __ad_ctx })
        }
        OperationOutput::Option(inner) => {
            let wrapped = wrap_output(inner, paths, backend, route, quote!(__value));
            quote!(#value.map(|__value| #wrapped))
        }
        OperationOutput::Vec(inner) => {
            let wrapped = wrap_output(inner, paths, backend, route, quote!(__value));
            quote!(#value.into_iter().map(|__value| #wrapped).collect())
        }
        OperationOutput::Tuple(items) => {
            let bindings: Vec<_> = (0..items.len())
                .map(|index| format_ident!("__out_{index}"))
                .collect();
            let wrapped = items
                .iter()
                .zip(&bindings)
                .map(|(item, binding)| wrap_output(item, paths, backend, route, quote!(#binding)));
            quote!({ let (#(#bindings,)*) = #value; (#(#wrapped,)*) })
        }
        OperationOutput::Extension(ty) => {
            let extension_trait = &paths.extension_trait;
            let variants = [
                TensorKind::Float,
                TensorKind::Int,
                TensorKind::Bool,
                TensorKind::Quantized,
            ]
            .into_iter()
            .map(|kind| {
                let variant = kind.variant();
                let dispatch_root = &paths.dispatch_root;
                let wrapped = wrap_tensor_kind(
                    paths,
                    kind,
                    backend,
                    route,
                    quote!(__tensor),
                    &format_ident!("AD"),
                );
                quote!(#dispatch_root::BackendTensor::#variant(__tensor) => #wrapped,)
            });
            let _ = ty;
            quote! {
                #extension_trait::map_to_dispatch(
                    #value,
                    |tensor| match tensor {
                        #(#variants)*
                        #[allow(unreachable_patterns)]
                        _ => unreachable!("unexpected output tensor variant"),
                    },
                    __ad_ctx,
                )
            }
        }
    }
}

/// Shared route fragments after a concrete backend is known.
///
/// The two macro frontends arrange these fragments differently, but input extraction, invocation,
/// output wrapping, and checkpoint-strategy selection stay defined in one place.
struct RouteFragments {
    concrete_inputs: TokenStream,
    concrete_invoke: TokenStream,
    concrete_disabled: TokenStream,
    concrete_enabled: TokenStream,
    autodiff_call: TokenStream,
    concrete_enabled_output: TokenStream,
}

fn route_fragments(
    operation: &Operation,
    paths: &RoutingPaths,
    backend: &syn::Ident,
    autodiff_attr: Option<&TokenStream>,
    has_autodiff_variant: bool,
    selected: Option<SelectedInput<'_>>,
) -> RouteFragments {
    let concrete_inputs = extract_inputs(
        operation,
        paths,
        backend,
        false,
        autodiff_attr,
        has_autodiff_variant,
        selected,
    );
    let concrete_invoke = invoke(operation, paths);
    let concrete_disabled = wrap_output(
        &operation.output,
        paths,
        backend,
        OutputRoute::ConcreteDisabled,
        quote!(__output),
    );
    let concrete_enabled = wrap_output(
        &operation.output,
        paths,
        backend,
        OutputRoute::ConcreteEnabled,
        quote!(__output),
    );
    let autodiff_inputs = extract_inputs(
        operation,
        paths,
        backend,
        true,
        autodiff_attr,
        has_autodiff_variant,
        selected,
    );
    let autodiff_output = wrap_output(
        &operation.output,
        paths,
        backend,
        OutputRoute::Autodiff,
        quote!(__output),
    );
    let autodiff_call = with_autodiff_backend(
        paths,
        backend,
        quote!(__strategy),
        &paths.backend_alias,
        quote! {
            #autodiff_inputs
            #concrete_invoke
            #autodiff_output
        },
    );
    let concrete_enabled_output = with_autodiff_backend(
        paths,
        backend,
        quote!(__strategy),
        &format_ident!("AD"),
        concrete_enabled.clone(),
    );

    RouteFragments {
        concrete_inputs,
        concrete_invoke,
        concrete_disabled,
        concrete_enabled,
        autodiff_call,
        concrete_enabled_output,
    }
}

/// Generate backend-dispatch routes after the routing tensor selected a concrete backend.
pub(crate) fn dispatch_backend_routes(
    operation: &Operation,
    backend: &syn::Ident,
    float_inputs: &FloatInputPresence,
    routing_tensor_is_autodiff: bool,
    source: Option<&OperationInput>,
    autodiff_attr: &TokenStream,
    no_autodiff_attr: &TokenStream,
) -> TokenStream {
    let paths = RoutingPaths::dispatch();
    let selected = source.map(|input| SelectedInput {
        input,
        autodiff_variant: routing_tensor_is_autodiff,
    });
    let fragments = route_fragments(
        operation,
        &paths,
        backend,
        Some(autodiff_attr),
        true,
        selected,
    );
    let RouteFragments {
        concrete_inputs,
        concrete_invoke,
        concrete_disabled,
        concrete_enabled,
        autodiff_call,
        concrete_enabled_output,
    } = fragments;
    let dispatch_root = &paths.dispatch_root;
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    let backend_root = &paths.backend_root;
    let backend_alias = &paths.backend_alias;

    let disabled = if routing_tensor_is_autodiff {
        quote! {
            #context::Disabled => {
                panic!("an autodiff float primitive must have an enabled autodiff context")
            }
        }
    } else {
        quote! {
            #context::Disabled => {
                type #backend_alias = #backend_root::#backend;
                #concrete_inputs
                #concrete_invoke
                #concrete_disabled
            }
        }
    };
    let autodiff_enabled = quote! {
        #autodiff_attr
        #context::Enabled(__strategy) => #autodiff_call,
        #no_autodiff_attr
        #context::Enabled(_) => panic!("autodiff context requires the `autodiff` feature"),
    };
    let concrete_enabled_route = if operation.output.contains_float() {
        quote! {
            #autodiff_attr
            #context::Enabled(__strategy) => {
                type #backend_alias = #backend_root::#backend;
                #concrete_inputs
                #concrete_invoke
                #concrete_enabled_output
            }
            #no_autodiff_attr
            #context::Enabled(_) => panic!("autodiff context requires the `autodiff` feature"),
        }
    } else {
        quote! {
            #context::Enabled(_) => {
                type #backend_alias = #backend_root::#backend;
                #concrete_inputs
                #concrete_invoke
                #concrete_enabled
            }
        }
    };
    let enabled = match float_inputs {
        FloatInputPresence::Always => autodiff_enabled,
        FloatInputPresence::Never => concrete_enabled_route,
        FloatInputPresence::Runtime(expression) => {
            let concrete_runtime = if operation.output.contains_float() {
                quote! {
                    #autodiff_attr
                    false => {
                        type #backend_alias = #backend_root::#backend;
                        #concrete_inputs
                        #concrete_invoke
                        #concrete_enabled_output
                    }
                    #no_autodiff_attr
                    false => panic!("autodiff context requires the `autodiff` feature"),
                }
            } else {
                quote! {
                    false => {
                        type #backend_alias = #backend_root::#backend;
                        #concrete_inputs
                        #concrete_invoke
                        #concrete_enabled
                    }
                }
            };
            quote! {
                #context::Enabled(__strategy) => match #expression {
                    #autodiff_attr
                    true => #autodiff_call,
                    #no_autodiff_attr
                    true => panic!("autodiff context requires the `autodiff` feature"),
                    #concrete_runtime
                }
            }
        }
    };
    quote!(match __ad_ctx { #disabled #enabled })
}

/// Generate backend-extension routes after the routing tensor selected a concrete backend.
pub(crate) fn extension_backend_routes(
    operation: &Operation,
    backend: &syn::Ident,
    autodiff: bool,
    autodiff_attr: Option<&TokenStream>,
    operation_name: &syn::Ident,
) -> TokenStream {
    let paths = RoutingPaths::extension();
    let fragments = route_fragments(operation, &paths, backend, autodiff_attr, autodiff, None);
    let RouteFragments {
        concrete_inputs,
        concrete_invoke,
        concrete_disabled,
        concrete_enabled,
        autodiff_call,
        concrete_enabled_output,
    } = fragments;
    let dispatch_root = &paths.dispatch_root;
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    let backend_root = &paths.backend_root;
    let backend_alias = &paths.backend_alias;
    let strategy = quote!(#dispatch_root::GradientCheckpointingStrategy);
    let attr = autodiff_attr.cloned().unwrap_or_default();
    let concrete_enabled_output = if operation.output.contains_float() {
        concrete_enabled_output
    } else {
        concrete_enabled
    };
    let concrete_route = if !autodiff {
        quote! {
            let #context::Disabled = __ad_ctx else {
                unimplemented!("Autodiff not supported for custom op `{}`", stringify!(#operation_name))
            };
            type #backend_alias = #backend_root::#backend;
            #concrete_inputs
            #concrete_invoke
            #concrete_disabled
        }
    } else if autodiff_attr.is_none() {
        quote! {
            type #backend_alias = #backend_root::#backend;
            #concrete_inputs
            #concrete_invoke
            match __ad_ctx {
                #context::Disabled => #concrete_disabled,
                #context::Enabled(__strategy) => #concrete_enabled_output,
            }
        }
    } else {
        quote! {
            let __concrete_strategy: Option<#strategy> = match __ad_ctx {
                #context::Disabled => None,
                #attr
                #context::Enabled(__strategy) => Some(__strategy),
                #[allow(unreachable_patterns)]
                _ => unimplemented!("Autodiff not supported for custom op `{}`", stringify!(#operation_name)),
            };
            type #backend_alias = #backend_root::#backend;
            #concrete_inputs
            #concrete_invoke
            match __concrete_strategy {
                None => #concrete_disabled,
                #attr
                Some(__strategy) => #concrete_enabled_output,
                #[allow(unreachable_patterns)]
                _ => unreachable!("enabled concrete output requires an autodiff strategy"),
            }
        }
    };
    if autodiff {
        let enabled = if autodiff_attr.is_some() {
            quote! {
                #attr
                (true, #context::Enabled(__strategy)) => #autodiff_call,
                (true, #context::Enabled(_)) => {
                    unimplemented!("Autodiff not supported for custom op `{}`", stringify!(#operation_name))
                }
            }
        } else {
            quote!((true, #context::Enabled(__strategy)) => #autodiff_call,)
        };
        quote! {
            match (__has_float_input, __ad_ctx) {
                #enabled
                (_, __ad_ctx) => { #concrete_route }
            }
        }
    } else {
        concrete_route
    }
}
