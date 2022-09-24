use crate::shared::{
    attribute::AttributeItem,
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
    let builders = config.gen_builder_fn_impl();
    let serde = config.gen_serde();
    let clone = config.gen_clone();
    let display = config.gen_display();
    let config_impl = config.gen_config_impl();

    quote! {
        #config_impl
        #constructor
        #builders
        #serde
        #clone
        #display
    }
}

struct Config {
    name: Ident,
    fields: Vec<FieldTypeAnalyzer>,
}

#[derive(Debug)]
struct ConfigAnalyzer {
    name: Ident,
    fields_required: Vec<FieldTypeAnalyzer>,
    fields_option: Vec<FieldTypeAnalyzer>,
    fields_default: Vec<(FieldTypeAnalyzer, AttributeItem)>,
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

        for (field, attribute) in self.fields_default.iter() {
            let name = field.ident();
            let value = &attribute.value;

            body.extend(quote! {
                #name: #value,
            });
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

    fn gen_builder_fn_impl(&self) -> TokenStream {
        let mut body = quote! {};

        for (field, _) in self.fields_default.iter() {
            let name = field.ident();
            let ty = &field.field.ty;
            let fn_name = Ident::new(&format!("with_{}", name), name.span());

            body.extend(quote! {
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();
            let ty = &field.field.ty;
            let fn_name = Ident::new(&format!("with_{}", name), name.span());

            body.extend(quote! {
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        self.wrap_impl_block(body)
    }

    fn names(&self) -> Vec<FieldTypeAnalyzer> {
        let mut names = Vec::new();

        for field in self.fields_required.iter() {
            names.push(field.clone());
        }

        for field in self.fields_option.iter() {
            names.push(field.clone());
        }

        for (field, _) in self.fields_default.iter() {
            names.push(field.clone());
        }

        names
    }

    fn name_types(&self, names: &[FieldTypeAnalyzer]) -> Vec<TokenStream> {
        let mut name_types = Vec::new();

        for field in names.iter() {
            let name = field.ident();
            let ty = &field.field.ty;

            name_types.push(quote! {
                #name: #ty
            });
        }

        name_types
    }

    fn gen_serde(&self) -> TokenStream {
        let names = self.names();
        let name_types = self.name_types(&names);

        let struct_gen = self.gen_serde_struct(&name_types);
        let serialize_gen = self.gen_serialize_fn(&names);
        let deserialize_gen = self.gen_deserialize_fn(&names);

        quote! {
            #struct_gen
            #serialize_gen
            #deserialize_gen
        }
    }

    fn serde_struct_ident(&self) -> Ident {
        Ident::new(&format!("{}Serde", self.name), self.name.span())
    }

    fn gen_clone(&self) -> TokenStream {
        let name = &self.name;
        let names = self.names().into_iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });

        quote! {
            impl Clone for #name {
                fn clone(&self) -> Self {
                    Self {
                        #(#names),*
                    }
                }
            }

        }
    }

    pub fn gen_serialize_fn(&self, names: &[FieldTypeAnalyzer]) -> TokenStream {
        let struct_name = self.serde_struct_ident();
        let names = names.iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });
        let name = &self.name;

        quote! {
            impl serde::Serialize for #name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer {
                    let serde_state = #struct_name {
                        #(#names),*
                    };
                    serde_state.serialize(serializer)
                }
            }

        }
    }

    pub fn gen_display(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl std::fmt::Display for #name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str(&burn::config::config_to_json(self))
                }
            }
        }
    }

    pub fn gen_config_impl(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl burn::config::Config for #name {
            }
        }
    }

    pub fn gen_deserialize_fn(&self, names: &[FieldTypeAnalyzer]) -> TokenStream {
        let struct_name = self.serde_struct_ident();
        let names = names.iter().map(|name| {
            let name = name.ident();
            quote! { #name: serde_state.#name }
        });
        let name = &self.name;

        quote! {
            impl<'de> serde::Deserialize<'de> for #name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de> {
                    let serde_state = #struct_name::deserialize(deserializer)?;
                    Ok(#name {
                        #(#names),*
                    })
                }
            }

        }
    }

    pub fn gen_serde_struct(&self, names: &[TokenStream]) -> TokenStream {
        let struct_name = self.serde_struct_ident();
        quote! {
            #[derive(serde::Serialize, serde::Deserialize)]
            struct #struct_name {
                #(#names),*
            }

        }
    }

    fn wrap_impl_block(&self, tokens: TokenStream) -> TokenStream {
        let name = &self.name;

        quote! {
            impl #name {
                #tokens
            }
        }
    }
}

impl Config {
    fn analyze(&self) -> ConfigAnalyzer {
        let mut fields_required = Vec::new();
        let mut fields_option = Vec::new();
        let mut fields_default = Vec::new();

        for field in self.fields.iter() {
            let attributes: Vec<AttributeItem> = field
                .attributes()
                .filter(|attr| attr.has_name("config"))
                .map(|attr| attr.items())
                .filter_map(|attr| attr.first().map(Clone::clone))
                .collect();

            if !attributes.is_empty() {
                let item = attributes.first().unwrap().clone();
                fields_default.push((field.clone(), item));
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
