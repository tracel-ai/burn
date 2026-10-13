use super::codegen_struct::ModuleFieldType;
use syn::{DeriveInput, Field, Generics, Type, parse_quote, visit::Visit};

/// Infer requirements from field types, not from every generic appearing inside them.
/// A `Wrapper<T>` can implement Module without requiring `T: Module`.
pub(crate) fn module_generics(
    ast: &DeriveInput,
    fields: Vec<(&Field, &ModuleFieldType)>,
) -> Generics {
    let mut generics = ast.generics.clone();
    let predicates = &mut generics.make_where_clause().predicates;
    // Debug may be derived separately and impose its own generic bounds. Send is a
    // property of the whole type, including skipped fields.
    predicates.push(parse_quote!(Self: core::fmt::Debug + Send));
    for (field, kind) in fields {
        for ty in field_bound_types(&field.ty, &ast.ident) {
            if kind.is_module {
                predicates
                    .push(parse_quote!(#ty: burn::module::Module + burn::module::ModuleDisplay));
            } else {
                predicates.push(parse_quote!(#ty: Clone + core::fmt::Debug + Send));
            }
        }
    }
    generics
}

fn field_bound_types<'a>(ty: &'a Type, name: &syn::Ident) -> Vec<&'a Type> {
    if !contains_self(ty, name) {
        return vec![ty];
    }

    // Vec<Self>: Module would make proving Self: Module depend on itself.
    // For recursive containers, only bound the non-recursive components (e.g.
    // M in Vec<(M, Self)>). Check the recursive field in the generated bodies,
    // where the current impl is available. Custom recursive wrappers may need
    // explicit bounds because their requirements cannot be inferred syntactically.
    match ty {
        Type::Path(path) if path.qself.is_none() => {
            let segment = path.path.segments.last().unwrap();
            if matches!(segment.ident.to_string().as_str(), "Box" | "Option" | "Vec")
                && let syn::PathArguments::AngleBracketed(args) = &segment.arguments
            {
                args.args
                    .iter()
                    .filter_map(|arg| match arg {
                        syn::GenericArgument::Type(ty) => Some(ty),
                        _ => None,
                    })
                    .flat_map(|ty| field_bound_types(ty, name))
                    .collect()
            } else {
                Vec::new()
            }
        }
        Type::Tuple(tuple) => tuple
            .elems
            .iter()
            .flat_map(|ty| field_bound_types(ty, name))
            .collect(),
        Type::Array(array) => field_bound_types(&array.elem, name),
        Type::Paren(paren) => field_bound_types(&paren.elem, name),
        Type::Group(group) => field_bound_types(&group.elem, name),
        _ => Vec::new(),
    }
}

fn contains_self(ty: &Type, name: &syn::Ident) -> bool {
    struct Finder<'a> {
        name: &'a syn::Ident,
        found: bool,
    }
    impl<'ast> Visit<'ast> for Finder<'_> {
        fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
            if ty
                .path
                .segments
                .last()
                .is_some_and(|s| s.ident == "Self" || s.ident == *self.name)
            {
                self.found = true;
            }
            syn::visit::visit_type_path(self, ty);
        }
    }
    let mut finder = Finder { name, found: false };
    finder.visit_type(ty);
    finder.found
}
