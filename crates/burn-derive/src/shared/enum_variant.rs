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
