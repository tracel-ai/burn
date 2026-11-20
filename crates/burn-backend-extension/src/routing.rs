//! Shared routing code generation for backend dispatch and backend extensions.
//!
//! Both attribute macros normalize a method into [`Operation`] and use this module for the routing
//! contract. Generated code selects a concrete backend from one routing tensor, merges the autodiff
//! context of every input, unwraps the inputs for the resolved backend and context, invokes the
//! operation, and wraps its output back into dispatch tensors. Backend selection remains an enum
//! match: this layer does not introduce trait-object dispatch or backend lookup tables.
//!
//! A required bare tensor can be matched and extracted directly. Operations containing only
//! optional, vector, quantization-parameter, or extension inputs first search those inputs for a
//! routing tensor because their runtime value may contain no tensor at all.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ir::{
    self, InputKind, Invocation, Operation, OperationInput, OperationOutput, TensorKind,
};

/// What a generated route targets.
#[derive(Clone, Copy)]
pub enum Target<'a> {
    /// A concrete backend.
    Concrete(&'a syn::Ident),
    /// The `Autodiff<Dispatch>` level, which needs no concrete backend.
    Autodiff,
}

/// How a generated operation's output tensor is represented at the dispatch boundary.
#[derive(Clone, Copy)]
pub enum OutputRoute<'a> {
    /// A concrete primitive with disabled autodiff metadata.
    ConcreteDisabled(&'a syn::Ident),
    /// A concrete primitive carried by an enabled autodiff context.
    ConcreteEnabled(&'a syn::Ident),
    /// An autodiff primitive carried by an enabled autodiff context.
    Autodiff,
}

/// Paths used by generated code.
///
/// `backend_dispatch` expands inside `burn-dispatch`, while backend extensions expand in downstream
/// crates through the public `burn::backend` facade. Keeping the paths explicit lets both frontends
/// share the routing mechanics without coupling their generated public representation.
pub struct RoutingPaths {
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
    pub fn dispatch() -> Self {
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

    pub fn extension() -> Self {
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

/// Generate the concrete backend-kind match arms shared by dispatch and backend extensions.
///
/// The caller supplies the body for each backend and appends the autodiff arm itself, since that
/// arm no longer selects a concrete backend.
pub fn backend_kind_arms(
    paths: &RoutingPaths,
    backends: &[ir::Backend],
    payload: Option<&syn::Ident>,
    mut body: impl FnMut(usize, &ir::Backend) -> TokenStream,
) -> Vec<TokenStream> {
    let dispatch_root = &paths.dispatch_root;
    let dispatch_kind = quote!(#dispatch_root::DispatchTensorKind);
    let payload = payload
        .map(|payload| quote!(#payload))
        .unwrap_or_else(|| quote!(_));
    backends
        .iter()
        .enumerate()
        .map(|(index, backend)| {
            let cfg_attr = &backend.cfg_attr;
            let ident = &backend.ident;
            let body = body(index, backend);
            quote! {
                #cfg_attr
                #dispatch_kind::#ident(#payload) => #body,
            }
        })
        .collect()
}

/// Select `Autodiff<Dispatch>` for the runtime checkpointing strategy.
pub fn with_autodiff_backend(
    paths: &RoutingPaths,
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
                type #alias = #backend_root::Autodiff<#dispatch_root::Dispatch, #checkpoint_root::BalancedCheckpointing>;
                #body
            }
            #dispatch_root::GradientCheckpointingStrategy::Disabled => {
                type #alias = #backend_root::Autodiff<#dispatch_root::Dispatch, #checkpoint_root::NoCheckpointing>;
                #body
            }
        }
    }
}

/// Wrap a tensor primitive in its dispatch enum representation.
pub fn wrap_tensor_kind(
    paths: &RoutingPaths,
    kind: TensorKind,
    route: OutputRoute<'_>,
    value: TokenStream,
    autodiff_backend: &syn::Ident,
) -> TokenStream {
    let RoutingPaths {
        dispatch_root,
        autodiff_trait,
        ..
    } = paths;
    let variant = kind.variant();

    if matches!(route, OutputRoute::Autodiff) {
        return if kind == TensorKind::Float {
            quote!(#dispatch_root::DispatchTensorKind::autodiff(#value))
        } else {
            // `Autodiff<Dispatch>`'s int, bool and quantized primitives are dispatch tensors.
            quote!((#value).kind)
        };
    }

    let backend = match route {
        OutputRoute::ConcreteDisabled(backend) | OutputRoute::ConcreteEnabled(backend) => backend,
        OutputRoute::Autodiff => unreachable!(),
    };
    if kind == TensorKind::Float && matches!(route, OutputRoute::ConcreteEnabled(_)) {
        quote! {
            #dispatch_root::DispatchTensorKind::autodiff(
                <#autodiff_backend as #autodiff_trait>::from_inner(
                    #dispatch_root::DispatchTensor {
                        kind: #dispatch_root::DispatchTensorKind::#backend(
                            #dispatch_root::BackendTensor::Float(#value)
                        ),
                        autodiff: #dispatch_root::DispatchAutodiffContext::Disabled,
                    }
                )
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

pub enum BackendSelection<'a> {
    /// Select and extract the backend directly from a tensor input.
    FromInput(&'a OperationInput),
    /// Select the backend through the operation's routing candidates.
    FromCandidates,
}

pub fn backend_selection(operation: &Operation) -> BackendSelection<'_> {
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

/// Generate the `__routing_tensor` binding from ordered candidates.
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
        },
        CandidateSelection::Missing => quote! {
            let __routing_tensor: &#tensor_ty = #any_fallback;
        },
        CandidateSelection::Optional(expression) => quote! {
            let __float_routing_tensor: Option<&#tensor_ty> = #expression;
            let __routing_tensor: &#tensor_ty = match __float_routing_tensor {
                Some(__routing_tensor) => __routing_tensor,
                None => #any_fallback,
            };
        },
    }
}

pub fn operation_routing_tensor(
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

/// Generate the autodiff routing state for every active tensor input.
///
/// `__ad_ctx` determines the output association and validates checkpointing compatibility;
/// `__has_autodiff_float_input` determines whether execution needs the autodiff backend. Computing
/// them together avoids walking tensor containers twice.
pub fn operation_autodiff_routing(
    operation: &Operation,
    paths: &RoutingPaths,
) -> (TokenStream, bool) {
    let dispatch_root = &paths.dispatch_root;
    let extension_trait = &paths.extension_trait;
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    let enabled = quote!(#context::Enabled(_));
    let mut can_have_float = false;
    let merges = operation.inputs.iter().filter_map(|input| {
        let name = &input.name;
        match &input.kind {
            InputKind::Tensor { kind, .. } => {
                let float = (*kind == TensorKind::Float).then(|| {
                    can_have_float = true;
                    quote! {
                        __has_autodiff_float_input |= matches!(__input_context, #enabled);
                    }
                });
                Some(quote! {
                    let __input_context = #name.autodiff;
                    #float
                    __ad_ctx = __ad_ctx.merge(__input_context);
                })
            }
            InputKind::OptionTensor(kind) => {
                let float = (*kind == TensorKind::Float).then(|| {
                    can_have_float = true;
                    quote! {
                        __has_autodiff_float_input |= matches!(__input_context, #enabled);
                    }
                });
                Some(quote! {
                if let Some(__tensor) = #name.as_ref() {
                    let __input_context = __tensor.autodiff;
                    #float
                    __ad_ctx = __ad_ctx.merge(__input_context);
                }
                })
            }
            InputKind::VecTensor(kind) => {
                let float = (*kind == TensorKind::Float).then(|| {
                    can_have_float = true;
                    quote! {
                        __has_autodiff_float_input |= matches!(__input_context, #enabled);
                    }
                });
                Some(quote! {
                for __tensor in &#name {
                    let __input_context = __tensor.autodiff;
                    #float
                    __ad_ctx = __ad_ctx.merge(__input_context);
                }
                })
            }
            InputKind::QuantizationParameters => {
                can_have_float = true;
                Some(quote! {
                let __input_context = #name.scales.autodiff;
                __has_autodiff_float_input |= matches!(__input_context, #enabled);
                __ad_ctx = __ad_ctx.merge(__input_context);
                if let Some(__tensor) = #name.global.as_ref() {
                    let __input_context = __tensor.autodiff;
                    __has_autodiff_float_input |= matches!(__input_context, #enabled);
                    __ad_ctx = __ad_ctx.merge(__input_context);
                }
                })
            }
            InputKind::Extension(ty) => {
                can_have_float = true;
                let dispatch_backend = quote!(#dispatch_root::Dispatch);
                let target = ir::with_backend(ty, dispatch_backend.clone());
                Some(quote! {
                    let (__input_context, __input_has_autodiff_float) =
                        <#target as #extension_trait<#dispatch_backend>>::autodiff_routing(&#name);
                    __has_autodiff_float_input |= __input_has_autodiff_float;
                    __ad_ctx = __ad_ctx.merge(__input_context);
                })
            }
            InputKind::Other => None,
        }
    });

    (
        quote! {
            let mut __ad_ctx = #context::Disabled;
            let mut __has_autodiff_float_input = false;
            #(#merges)*
        },
        can_have_float,
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

pub fn invoke(operation: &Operation, paths: &RoutingPaths) -> TokenStream {
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
    target: Target<'_>,
    autodiff_cfg: Option<&TokenStream>,
    selected: Option<SelectedInput<'_>>,
) -> TokenStream {
    let extraction = Extraction {
        paths,
        target,
        autodiff_cfg,
    };
    let inputs = operation.inputs.iter().map(|input| {
        if let Some(selected) = selected
            && core::ptr::eq(input, selected.input)
        {
            return extract_selected_input(input, &extraction);
        }
        extract_input(input, &extraction)
    });
    quote!(#(#inputs)*)
}

#[derive(Clone, Copy)]
struct SelectedInput<'a> {
    input: &'a OperationInput,
}

struct Extraction<'a> {
    paths: &'a RoutingPaths,
    target: Target<'a>,
    autodiff_cfg: Option<&'a TokenStream>,
}

impl Extraction<'_> {
    fn has_autodiff_variant(&self) -> bool {
        self.autodiff_cfg.is_some()
    }
}

fn extract_selected_input(input: &OperationInput, extraction: &Extraction<'_>) -> TokenStream {
    let InputKind::Tensor { kind, borrowed } = input.kind else {
        unreachable!("selected input must be a tensor")
    };
    let name = &input.name;
    let selected = format_ident!("__burn_selected");
    let dispatch_root = &extraction.paths.dispatch_root;
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    let validate_context = if kind == TensorKind::Float {
        quote! {
            let #context::Disabled = __burn_selected_context else {
                panic!("an enabled float tensor must use an autodiff primitive")
            };
        }
    } else {
        TokenStream::new()
    };
    let accessor = kind.accessor();
    let accessor = if borrowed {
        format_ident!("as_{accessor}")
    } else {
        accessor
    };
    quote!(#validate_context let #name = #selected.#accessor();)
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
        InputKind::Extension(ty) => {
            extract_extension(name, ty, extraction.paths, extraction.target)
        }
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
    match extraction.target {
        Target::Autodiff => extract_tensor_autodiff(name, kind, borrowed, extraction.paths),
        Target::Concrete(backend) => {
            extract_tensor_concrete(name, kind, borrowed, backend, extraction)
        }
    }
}

fn extract_tensor_autodiff(
    name: &syn::Ident,
    kind: TensorKind,
    borrowed: bool,
    paths: &RoutingPaths,
) -> TokenStream {
    let backend_alias = &paths.backend_alias;
    let dispatch_root = &paths.dispatch_root;
    let dispatch_kind = quote!(#dispatch_root::DispatchTensorKind);
    let autodiff_trait = &paths.autodiff_trait;
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    let validate_enabled_float = quote! {
        let #context::Enabled(_) = #name.autodiff else {
            panic!("an autodiff float primitive must have an enabled autodiff context")
        };
    };
    let validate_disabled_float = quote! {
        let #context::Disabled = #name.autodiff else {
            panic!("an enabled float tensor must use an autodiff primitive")
        };
    };

    if kind == TensorKind::Float {
        if borrowed {
            let lifted = format_ident!("__lifted_{name}");
            quote! {
                let #lifted;
                let #name = match &#name.kind {
                    #dispatch_kind::Autodiff(__inner) => {
                        #validate_enabled_float
                        &**__inner
                    }
                    _ => {
                        #validate_disabled_float
                        #lifted = <#backend_alias as #autodiff_trait>::from_inner(#name.clone());
                        &#lifted
                    }
                };
            }
        } else {
            quote! {
                let #name = match #name.kind {
                    #dispatch_kind::Autodiff(__inner) => {
                        #validate_enabled_float
                        *__inner
                    }
                    __kind => {
                        #validate_disabled_float
                        <#backend_alias as #autodiff_trait>::from_inner(
                            #dispatch_root::DispatchTensor {
                                kind: __kind,
                                autodiff: #dispatch_root::DispatchAutodiffContext::Disabled,
                            }
                        )
                    }
                };
            }
        }
    } else if borrowed {
        let cleared = format_ident!("__cleared_{name}");
        quote! {
            let #cleared;
            let #name = match &#name.kind {
                #dispatch_kind::Autodiff(_) => panic!("only float tensors may use an autodiff primitive"),
                __kind => {
                    #cleared = #dispatch_root::DispatchTensor {
                        kind: __kind.clone(),
                        autodiff: #dispatch_root::DispatchAutodiffContext::Disabled,
                    };
                    &#cleared
                }
            };
        }
    } else {
        // Inside `Autodiff<Dispatch>`, int, bool and quantized tensors are primitive values: their
        // device association must not make the primitive computation look autodiff-enabled.
        quote! {
            let #name = match #name.kind {
                #dispatch_kind::Autodiff(_) => panic!("only float tensors may use an autodiff primitive"),
                __kind => #dispatch_root::DispatchTensor {
                    kind: __kind,
                    autodiff: #dispatch_root::DispatchAutodiffContext::Disabled,
                },
            };
        }
    }
}

fn extract_tensor_concrete(
    name: &syn::Ident,
    kind: TensorKind,
    borrowed: bool,
    backend: &syn::Ident,
    extraction: &Extraction<'_>,
) -> TokenStream {
    let dispatch_root = &extraction.paths.dispatch_root;
    let dispatch_kind = quote!(#dispatch_root::DispatchTensorKind);
    let ad_cfg = cfg_attr(extraction.autodiff_cfg);
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    let validate_disabled_float = quote! {
        let #context::Disabled = #name.autodiff else {
            panic!("an enabled float tensor must use an autodiff primitive")
        };
    };
    let invalid_autodiff_float = extraction.has_autodiff_variant().then(|| {
        quote! {
            #ad_cfg
            #dispatch_kind::Autodiff(_) => panic!("autodiff float input reached concrete dispatch"),
        }
    });
    let invalid_autodiff_kind = extraction.has_autodiff_variant().then(|| quote! {
        #ad_cfg
        #dispatch_kind::Autodiff(_) => panic!("only float tensors may use an autodiff primitive"),
    });

    if kind == TensorKind::Float {
        if borrowed {
            quote! {
                let #name = match &#name.kind {
                    #dispatch_kind::#backend(__inner) => {
                        #validate_disabled_float
                        __inner.as_float()
                    }
                    #invalid_autodiff_float
                    _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                };
            }
        } else {
            quote! {
                let #name = match #name.kind {
                    #dispatch_kind::#backend(__inner) => {
                        #validate_disabled_float
                        __inner.float()
                    }
                    #invalid_autodiff_float
                    _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
                };
            }
        }
    } else {
        let accessor = kind.accessor();
        let accessor_ref = format_ident!("as_{accessor}");
        if borrowed {
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
}

fn extract_extension(
    name: &syn::Ident,
    ty: &syn::Type,
    paths: &RoutingPaths,
    target: Target<'_>,
) -> TokenStream {
    let backend_root = &paths.backend_root;
    let dispatch_root = &paths.dispatch_root;
    let dispatch_kind = quote!(#dispatch_root::DispatchTensorKind);
    let extension_trait = &paths.extension_trait;
    match target {
        Target::Autodiff => {
            let backend_alias = &paths.backend_alias;
            let target_ty = ir::with_backend(ty, quote!(#backend_alias));
            quote! {
                let #name = <#target_ty as #extension_trait<#backend_alias>>::map_from_dispatch(
                    #name,
                    |__tensor| __tensor.into_autodiff_backend::<#backend_alias>(),
                );
            }
        }
        Target::Concrete(backend) => {
            let target_ty = ir::with_backend(ty, quote!(#backend_root::#backend));
            let mismatch = quote!(panic!("backend extension input is on the wrong backend"));
            quote! {
                let #name = <#target_ty as #extension_trait<#backend_root::#backend>>::map_from_dispatch(
                    #name,
                    |__tensor| {
                        match __tensor.kind {
                            #dispatch_kind::#backend(tensor) => tensor,
                            #[allow(unreachable_patterns)]
                            _ => #mismatch,
                        }
                    },
                );
            }
        }
    }
}

pub fn wrap_output(
    output: &OperationOutput,
    paths: &RoutingPaths,
    route: OutputRoute<'_>,
    value: TokenStream,
) -> TokenStream {
    match output {
        OperationOutput::Plain => value,
        OperationOutput::Tensor(kind) => {
            let dispatch_root = &paths.dispatch_root;
            let wrapped = wrap_tensor_kind(paths, *kind, route, value, &format_ident!("AD"));
            quote!(#dispatch_root::DispatchTensor { kind: #wrapped, autodiff: __ad_ctx })
        }
        OperationOutput::Option(inner) => {
            let wrapped = wrap_output(inner, paths, route, quote!(__value));
            quote!(#value.map(|__value| #wrapped))
        }
        OperationOutput::Vec(inner) => {
            let wrapped = wrap_output(inner, paths, route, quote!(__value));
            quote!(#value.into_iter().map(|__value| #wrapped).collect())
        }
        OperationOutput::Tuple(items) => {
            let bindings: Vec<_> = (0..items.len())
                .map(|index| format_ident!("__out_{index}"))
                .collect();
            let wrapped = items
                .iter()
                .zip(&bindings)
                .map(|(item, binding)| wrap_output(item, paths, route, quote!(#binding)));
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
                let wrapped =
                    wrap_tensor_kind(paths, kind, route, quote!(__tensor), &format_ident!("AD"));
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

/// Shared concrete route fragments after a backend is known.
struct RouteFragments {
    concrete_inputs: TokenStream,
    concrete_invoke: TokenStream,
    concrete_disabled: TokenStream,
    concrete_enabled: TokenStream,
    concrete_enabled_output: TokenStream,
}

fn route_fragments(
    operation: &Operation,
    paths: &RoutingPaths,
    backend: &syn::Ident,
    source: Option<&OperationInput>,
    autodiff_cfg: Option<&TokenStream>,
) -> RouteFragments {
    let selected = source.map(|input| SelectedInput { input });
    let concrete_inputs = extract_inputs(
        operation,
        paths,
        Target::Concrete(backend),
        autodiff_cfg,
        selected,
    );
    let concrete_invoke = invoke(operation, paths);
    let concrete_disabled = wrap_output(
        &operation.output,
        paths,
        OutputRoute::ConcreteDisabled(backend),
        quote!(__output),
    );
    let concrete_enabled = wrap_output(
        &operation.output,
        paths,
        OutputRoute::ConcreteEnabled(backend),
        quote!(__output),
    );
    let concrete_enabled_output = with_autodiff_backend(
        paths,
        quote!(__strategy),
        &format_ident!("AD"),
        concrete_enabled.clone(),
    );
    RouteFragments {
        concrete_inputs,
        concrete_invoke,
        concrete_disabled,
        concrete_enabled,
        concrete_enabled_output,
    }
}

/// The route body that runs an operation on `Autodiff<Dispatch>`.
///
/// It is independent of the selected concrete backend: the autodiff level is part of the tensor
/// data, so every level uses the same monomorphization.
pub fn autodiff_route(operation: &Operation, paths: &RoutingPaths) -> TokenStream {
    let dispatch_root = &paths.dispatch_root;
    let backend_alias = &paths.backend_alias;
    let inputs = extract_inputs(operation, paths, Target::Autodiff, None, None);
    let invoke = invoke(operation, paths);
    let output = wrap_output(
        &operation.output,
        paths,
        OutputRoute::Autodiff,
        quote!(__output),
    );
    with_autodiff_backend(
        paths,
        quote!(__strategy),
        backend_alias,
        quote! {
            let __ad_ctx = #dispatch_root::DispatchAutodiffContext::Enabled(__strategy);
            #inputs
            #invoke
            #output
        },
    )
}

/// Generate concrete-backend routes for a dispatched tensor operation.
pub fn dispatch_backend_routes(
    operation: &Operation,
    backend: &syn::Ident,
    source: Option<&OperationInput>,
) -> TokenStream {
    let paths = RoutingPaths::dispatch();
    let autodiff_cfg = quote!(#[cfg(feature = "autodiff")]);
    let fragments = route_fragments(operation, &paths, backend, source, Some(&autodiff_cfg));
    let RouteFragments {
        concrete_inputs,
        concrete_invoke,
        concrete_disabled,
        concrete_enabled,
        concrete_enabled_output,
    } = fragments;
    let dispatch_root = &paths.dispatch_root;
    let context = quote!(#dispatch_root::DispatchAutodiffContext);
    let backend_root = &paths.backend_root;
    let backend_alias = &paths.backend_alias;

    let disabled = quote! {
        #context::Disabled => {
            type #backend_alias = #backend_root::#backend;
            #concrete_inputs
            #concrete_invoke
            #concrete_disabled
        }
    };
    let enabled = if operation.output.contains_float() {
        quote! {
            #[cfg(feature = "autodiff")]
            #context::Enabled(__strategy) => {
                type #backend_alias = #backend_root::#backend;
                #concrete_inputs
                #concrete_invoke
                #concrete_enabled_output
            }
            #[cfg(not(feature = "autodiff"))]
            #context::Enabled(_) => panic!("autodiff context requires the `autodiff` feature"),
        }
    } else {
        quote! {
            #[cfg(feature = "autodiff")]
            #context::Enabled(_) => {
                type #backend_alias = #backend_root::#backend;
                #concrete_inputs
                #concrete_invoke
                #concrete_enabled
            }
            #[cfg(not(feature = "autodiff"))]
            #context::Enabled(_) => panic!("autodiff context requires the `autodiff` feature"),
        }
    };
    quote!(match __ad_ctx { #disabled #enabled })
}

/// Generate concrete-backend routes for a backend extension after it selected a backend.
pub fn extension_backend_routes(
    operation: &Operation,
    backend: &syn::Ident,
    autodiff: bool,
    autodiff_attr: Option<&TokenStream>,
    operation_name: &syn::Ident,
) -> TokenStream {
    let paths = RoutingPaths::extension();
    let fragments = route_fragments(operation, &paths, backend, None, autodiff_attr);
    let RouteFragments {
        concrete_inputs,
        concrete_invoke,
        concrete_disabled,
        concrete_enabled,
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
    if !autodiff {
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
    }
}
