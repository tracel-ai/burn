use super::{FieldKind, classify, optim_path};
use quote::quote;
use syn::{Data, DeriveInput, Fields};

pub(crate) fn derive_mapping(ast: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let Data::Struct(data) = &ast.data else {
        return Err(syn::Error::new_spanned(
            name,
            "StateMappable requires named struct fields",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            name,
            "StateMappable requires named struct fields",
        ));
    };
    let mut mapped = Vec::new();
    for field in &fields.named {
        let ident = field.ident.as_ref().unwrap();
        let path = quote!(path.field(stringify!(#ident)));
        let value = match classify(&field.ty)? {
            FieldKind::Tensor => quote!(mapper.map(self.#ident, &#path)),
            FieldKind::OptionTensor(_) => quote!(self.#ident.map(|t| mapper.map(t, &#path))),
            FieldKind::VecTensor(_) => quote!(self.#ident.into_iter().enumerate()
                .map(|(index, t)| mapper.map(t, &#path.index(index))).collect()),
            FieldKind::Nested => {
                quote!(__optim::StateMappable::map_tensors_at(self.#ident, mapper, &#path))
            }
            FieldKind::OptionNested(_) => {
                quote!(self.#ident.map(|s| __optim::StateMappable::map_tensors_at(s, mapper, &#path)))
            }
            FieldKind::Scalar | FieldKind::OptionScalar(_) => quote!(self.#ident),
        };
        mapped.push(quote!(#ident: #value));
    }
    let root = optim_path()?;
    Ok(quote! {
        const _: () = {
            use #root as __optim;
            impl #impl_generics __optim::StateMappable for #name #ty_generics #where_clause {
                fn map_tensors_at<__M: __optim::OptimizerStateMapper>(self, mapper: &__M, path: &__optim::StatePath<'_>) -> Self {
                    Self { #(#mapped),* }
                }
            }
        };
    })
}
