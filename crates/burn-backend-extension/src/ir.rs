//! Normalized operation model shared by backend dispatch and backend extensions.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Block, GenericArgument, Ident, PathArguments, Type};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorKind {
    Float,
    Int,
    Bool,
    Quantized,
}

impl TensorKind {
    pub(crate) fn from_type(ty: &Type) -> Option<Self> {
        match ty {
            Type::Path(path) => match path.path.segments.last()?.ident.to_string().as_str() {
                "Float" | "FloatTensor" | "FloatTensorPrimitive" => Some(Self::Float),
                "Int" | "IntTensor" | "IntTensorPrimitive" => Some(Self::Int),
                "Bool" | "BoolTensor" | "BoolTensorPrimitive" => Some(Self::Bool),
                "Quantized" | "QuantizedTensor" | "QuantizedTensorPrimitive" => {
                    Some(Self::Quantized)
                }
                _ => None,
            },
            Type::Reference(reference) => Self::from_type(&reference.elem),
            Type::Paren(paren) => Self::from_type(&paren.elem),
            _ => None,
        }
    }

    pub(crate) fn variant(self) -> Ident {
        format_ident!("{self:?}")
    }

    pub(crate) fn accessor(self) -> Ident {
        format_ident!("{}", format!("{self:?}").to_lowercase())
    }
}

#[derive(Clone)]
pub(crate) enum InputKind {
    Tensor { kind: TensorKind, borrowed: bool },
    OptionTensor(TensorKind),
    VecTensor(TensorKind),
    QuantizationParameters,
    Extension(Box<Type>),
    Other,
}

#[derive(Clone)]
pub(crate) struct OperationInput {
    pub(crate) name: Ident,
    pub(crate) kind: InputKind,
}

impl OperationInput {
    pub(crate) fn dispatch(name: Ident, ty: &Type) -> Option<Self> {
        let kind = InputKind::dispatch(ty)?;
        Some(Self { name, kind })
    }
}

impl InputKind {
    pub(crate) fn owned_tensor(ty: &Type) -> Option<Self> {
        TensorKind::from_type(ty).map(|kind| Self::Tensor {
            kind,
            borrowed: false,
        })
    }

    fn dispatch(ty: &Type) -> Option<Self> {
        if let Type::Reference(reference) = ty
            && let Some(kind) = TensorKind::from_type(&reference.elem)
        {
            return Some(Self::Tensor {
                kind,
                borrowed: true,
            });
        }
        if let Some(tensor) = Self::owned_tensor(ty) {
            return Some(tensor);
        }
        if let Some(inner) = container_inner(ty, "Option")
            && let Some(kind) = TensorKind::from_type(inner)
        {
            return Some(Self::OptionTensor(kind));
        }
        if let Some(inner) = container_inner(ty, "Vec")
            && let Some(kind) = TensorKind::from_type(inner)
        {
            return Some(Self::VecTensor(kind));
        }
        type_is(ty, "QuantizationParametersPrimitive").then_some(Self::QuantizationParameters)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum OperationOutput {
    Tensor(TensorKind),
    Option(Box<OperationOutput>),
    Vec(Box<OperationOutput>),
    Tuple(Vec<OperationOutput>),
    Extension(Box<Type>),
    Plain,
}

impl OperationOutput {
    pub(crate) fn dispatch(ty: &Type) -> syn::Result<Self> {
        if let Some(kind) = TensorKind::from_type(ty) {
            return Ok(Self::Tensor(kind));
        }
        if let Some(inner) = container_inner(ty, "Option") {
            return Ok(Self::Option(Box::new(Self::dispatch(inner)?)));
        }
        if let Some(inner) = container_inner(ty, "Vec") {
            return Ok(Self::Vec(Box::new(Self::dispatch(inner)?)));
        }
        if let Type::Tuple(tuple) = ty {
            return Ok(Self::Tuple(
                tuple
                    .elems
                    .iter()
                    .map(Self::dispatch)
                    .collect::<syn::Result<_>>()?,
            ));
        }
        if type_contains_self(ty) {
            return Err(syn::Error::new_spanned(
                ty,
                "named outputs containing `Self` aren't supported; dispatch through a private inherent helper returning tensors, options, or tuples and reconstruct the named value explicitly",
            ));
        }
        Ok(Self::Plain)
    }

    pub(crate) fn extension(ty: &Type) -> Self {
        if let Some(kind) = TensorKind::from_type(ty) {
            return Self::Tensor(kind);
        }
        if let Some(inner) = container_inner(ty, "Option") {
            return Self::Option(Box::new(Self::extension(inner)));
        }
        if let Some(inner) = container_inner(ty, "Vec") {
            return Self::Vec(Box::new(Self::extension(inner)));
        }
        if let Type::Tuple(tuple) = ty {
            return Self::Tuple(tuple.elems.iter().map(Self::extension).collect());
        }
        if type_contains_self(ty) {
            Self::Extension(Box::new(ty.clone()))
        } else {
            Self::Plain
        }
    }

    pub(crate) fn contains_float(&self) -> bool {
        match self {
            Self::Tensor(TensorKind::Float) | Self::Extension(_) => true,
            Self::Option(inner) | Self::Vec(inner) => inner.contains_float(),
            Self::Tuple(items) => items.iter().any(Self::contains_float),
            Self::Tensor(_) | Self::Plain => false,
        }
    }
}

#[derive(Clone)]
pub(crate) enum Invocation {
    Body(Block),
    Trait {
        trait_name: Ident,
        await_call: bool,
        unsafe_call: bool,
        generic_args: Vec<Ident>,
    },
}

#[derive(Clone)]
pub(crate) struct Operation {
    pub(crate) name: Ident,
    pub(crate) inputs: Vec<OperationInput>,
    pub(crate) output: OperationOutput,
    pub(crate) invocation: Invocation,
}

#[derive(Clone)]
pub(crate) struct Backend {
    pub(crate) ident: Ident,
    pub(crate) cfg_attr: Option<TokenStream>,
}

/// Rewrite a type's final generic argument to `param`, e.g. `Ty<Self>` to `Ty<Wgpu>`.
pub(crate) fn with_backend(ty: &Type, param: TokenStream) -> TokenStream {
    if let Type::Path(type_path) = ty {
        let mut path = type_path.path.clone();
        if let Some(last) = path.segments.last_mut()
            && let PathArguments::AngleBracketed(args) = &mut last.arguments
            && args.args.len() == 1
        {
            args.args[0] = GenericArgument::Type(
                syn::parse2(param).expect("backend parameter must be a valid type"),
            );
            return quote!(#path);
        }
    }
    quote!(#ty)
}

fn container_inner<'a>(ty: &'a Type, name: &str) -> Option<&'a Type> {
    let Type::Path(path) = ty else { return None };
    let segment = path.path.segments.last()?;
    if segment.ident != name {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.first()? {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    }
}

fn type_is(ty: &Type, name: &str) -> bool {
    let Type::Path(path) = ty else { return false };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == name)
}

fn type_contains_self(ty: &Type) -> bool {
    match ty {
        Type::Path(path) if path.path.is_ident("Self") => true,
        Type::Path(path) => path.path.segments.iter().any(|segment| {
            let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
                return false;
            };
            arguments.args.iter().any(|argument| match argument {
                GenericArgument::Type(ty) => type_contains_self(ty),
                GenericArgument::AssocType(assoc) => type_contains_self(&assoc.ty),
                _ => false,
            })
        }),
        Type::Reference(reference) => type_contains_self(&reference.elem),
        Type::Paren(paren) => type_contains_self(&paren.elem),
        Type::Tuple(tuple) => tuple.elems.iter().any(type_contains_self),
        Type::Array(array) => type_contains_self(&array.elem),
        Type::Slice(slice) => type_contains_self(&slice.elem),
        _ => false,
    }
}
