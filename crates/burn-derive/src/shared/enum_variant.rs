use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{FieldsNamed, Variant};

/// Process a variant of an enum where the output is the result of the given mapper.
pub(crate) fn map_enum_variant<Mapper>(
    variant: &Variant,
    mapper: Mapper,
) -> (TokenStream, TokenStream)
where
    Mapper: Fn(&Ident) -> TokenStream,
{
    let gen_fields_unnamed = |num: usize| {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for i in 0..num {
            let arg_name = Ident::new(&format!("arg_{i}"), Span::call_site());
            let input = quote! { #arg_name };
            let output = mapper(&arg_name);

            inputs.push(input);
            outputs.push(output);
        }

        (quote! (( #(#inputs),* )), quote! (( #(#outputs),* )))
    };
    let gen_fields_named = |fields: &FieldsNamed| {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        fields.named.iter().for_each(|field| {
            let ident = field.ident.as_ref().expect("Named field to have a name.");
            let input = quote! { #ident };
            let output = mapper(ident);

            inputs.push(input);
            outputs.push(quote! {
                #ident: #output
            });
        });

        (quote! {{ #(#inputs),* }}, quote! {{ #(#outputs),* }})
    };

    match &variant.fields {
        syn::Fields::Named(fields) => gen_fields_named(fields),
        syn::Fields::Unnamed(_) => gen_fields_unnamed(variant.fields.len()),
        syn::Fields::Unit => (quote! {}, quote! {}),
    }
}

/// An enum variant (simplified).
pub(crate) struct EnumVariant {
    pub ident: syn::Ident,
    pub ty: syn::Type,
}
pub(crate) fn parse_variants(ast: &syn::DeriveInput) -> syn::Result<Vec<EnumVariant>> {
    let enum_data = match &ast.data {
        syn::Data::Enum(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                ast,
                "Module can only be derived for enums.",
            ));
        }
    };

    let mut variants = Vec::new();

    for variant in enum_data.variants.iter() {
        match &variant.fields {
            syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                let field = &fields.unnamed[0];

                variants.push(EnumVariant {
                    ident: variant.ident.clone(),
                    ty: field.ty.clone(),
                });
            }
            syn::Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive only supports tuple enum variants with exactly one field.",
                ));
            }
            syn::Fields::Named(_) => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive does not support struct enum variants.",
                ));
            }
            syn::Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive does not support unit enum variants.",
                ));
            }
        }
    }

    Ok(variants)
}
