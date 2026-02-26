use std::collections::{HashMap, HashSet};

use proc_macro2::Ident;
use syn::{GenericParam, Generics, Type, TypeParamBound, WherePredicate, visit::Visit};

#[derive(Debug)]
pub enum GenericKind {
    /// A generic with `Module<B>` bound.
    Module,
    /// A generic used in a field marked by `#[module(constant)]`.
    Constant,
    /// A generic used in a field marked by `#[module(skip)]`.
    Skip,
    /// A plain generic that does not fit any of the above conditions.
    Plain,
}

#[derive(Debug)]
pub struct ModuleGenerics {
    kinds: HashMap<Ident, GenericKind>,
}

impl ModuleGenerics {
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn get_generic_kind(&self, ident: &Ident) -> Option<&GenericKind> {
        self.kinds.get(ident)
    }

    pub fn is_bounded_module(&self, ident: &Ident) -> bool {
        self.kinds
            .get(ident)
            .map(|kind| matches!(kind, GenericKind::Module))
            .unwrap_or(false)
    }

    pub fn update(&mut self, ident: &Ident, kind: GenericKind) {
        self.kinds.insert(ident.clone(), kind);
    }

    pub fn contains(&self, ident: &Ident) -> bool {
        self.kinds.contains_key(ident)
    }
}

pub fn parse_module_generics(generics: &Generics) -> ModuleGenerics {
    let mut kinds = HashMap::new();

    // Check inline bounds e.g. `M: Module<B>`
    for param in &generics.params {
        if let GenericParam::Type(type_param) = param {
            let ident = &type_param.ident;
            if ident != "B" {
                if has_module_bound(&type_param.bounds) {
                    kinds.insert(ident.clone(), GenericKind::Module);
                } else {
                    kinds.insert(ident.clone(), GenericKind::Plain);
                }
            }
        }
    }

    // Check `where` clauses
    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            if let WherePredicate::Type(pt) = predicate {
                // We only care if the bounded type is a simple identifier (like 'M')
                if let Type::Path(p) = &pt.bounded_ty
                    && let Some(ident) = p.path.get_ident()
                    && ident != "B"
                {
                    if has_module_bound(&pt.bounds) {
                        kinds.insert(ident.clone(), GenericKind::Module);
                    } else {
                        kinds.insert(ident.clone(), GenericKind::Plain);
                    }
                }
            }
        }
    }

    ModuleGenerics { kinds }
}

// TODO: remove special cases for `ident == "B"`, this could be used to check for `Backend` bound.

/// Helper to check if a list of bounds contains "Module".
fn has_module_bound(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, syn::token::Plus>,
) -> bool {
    has_bound(bounds, "Module")
}

/// Helper to check if a list of bounds contains the specified bound.
fn has_bound(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, syn::token::Plus>,
    ident: &str,
) -> bool {
    bounds.iter().any(|bound| {
        if let TypeParamBound::Trait(trait_bound) = bound
            && let Some(segment) = trait_bound.path.segments.last()
        {
            return segment.ident == ident;
        }
        false
    })
}

pub fn parse_ty_generics(ty: &Type, declared: &ModuleGenerics) -> HashSet<Ident> {
    struct Collector<'a> {
        generics: HashSet<Ident>,
        declared: &'a ModuleGenerics,
    }

    impl<'ast, 'a> Visit<'ast> for Collector<'a> {
        fn visit_type_path(&mut self, type_path: &'ast syn::TypePath) {
            if type_path.qself.is_none()
                && let Some(ident) = type_path.path.get_ident()
                && (self.declared.contains(ident) || ident == "B")
            {
                self.generics.insert(ident.clone());
            }

            syn::visit::visit_type_path(self, type_path);
        }
    }

    let mut collector = Collector {
        generics: HashSet::new(),
        declared,
    };
    collector.visit_type(ty);
    collector.generics
}
