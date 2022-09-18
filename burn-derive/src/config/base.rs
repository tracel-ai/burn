use crate::shared::{
    attrubute::AttributeAnalyzer,
    field::{parse_fields, FieldTypeAnalyzer},
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

pub(crate) fn config_attr_impl(item: &syn::DeriveInput) -> TokenStream {
    let name = item.ident.clone();
    let fields = parse_fields(item);
    let fields = fields.into_iter().map(FieldTypeAnalyzer::new).collect();
    let config = Config { name, fields }.analyze();

    let constructor = config.gen_constructor_impl();

    quote! {
        #constructor
    }
    .into()
}

struct Config {
    name: Ident,
    fields: Vec<FieldTypeAnalyzer>,
}

struct ConfigAnalyzer {
    name: Ident,
    fields_required: Vec<FieldTypeAnalyzer>,
    fields_option: Vec<FieldTypeAnalyzer>,
    fields_default: Vec<(FieldTypeAnalyzer, Vec<AttributeAnalyzer>)>,
}

impl ConfigAnalyzer {
    fn gen_constructor_impl(&self) -> TokenStream {
        let mut body = quote! {};
        let mut names = Vec::new();

        for field in self.fields_required.iter() {
            let name = field.ident();
            let ty = &field.field.ty;

            body.extend(quote! {
                #name: #name,
            });
            names.push(quote! {
                #name: #ty
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();

            body.extend(quote! {
                #name: None,
            });
        }

        for (field, attributes) in self.fields_default.iter() {
            let name = field.ident();
            let value = attributes.first().unwrap();
            let items = value.items();

            match items.first() {
                Some(item) => {
                    let value = &item.value;

                    body.extend(quote! {
                        #name: #value,
                    });
                }
                _ => panic!("No value provided for default field {}", name.to_string()),
            };
        }

        let body = quote! {
            pub fn new(
                #(#names),*
            ) -> Self {
                Self { #body }
            }
        };
        self.wrap_impl_block(body)
    }

    fn wrap_impl_block(&self, tokens: TokenStream) -> TokenStream {
        let name = &self.name;

        quote! {
            impl #name {
                #tokens
            }
        }
        .into()
    }
}

impl Config {
    fn analyze(&self) -> ConfigAnalyzer {
        let mut fields_required = Vec::new();
        let mut fields_option = Vec::new();
        let mut fields_default = Vec::new();

        for field in self.fields.iter() {
            let attributes: Vec<AttributeAnalyzer> = field
                .attributes()
                .filter(|attr| attr.has_name("config"))
                .collect();

            if !attributes.is_empty() {
                fields_default.push((field.clone(), attributes));
                continue;
            }

            if field.is_of_type(&["Option"]) {
                fields_option.push(field.clone());
                continue;
            }

            fields_required.push(field.clone());
        }

        ConfigAnalyzer {
            name: self.name.clone(),
            fields_required,
            fields_option,
            fields_default,
        }
    }
}
