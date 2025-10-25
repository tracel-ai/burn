use super::ConfigAnalyzer;
use crate::shared::{attribute::AttributeItem, field::FieldTypeAnalyzer};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

pub struct ConfigStructAnalyzer {
    name: Ident,
    fields_required: Vec<FieldTypeAnalyzer>,
    fields_option: Vec<FieldTypeAnalyzer>,
    fields_default: Vec<(FieldTypeAnalyzer, AttributeItem)>,
}

impl ConfigStructAnalyzer {
    pub fn new(
        name: Ident,
        fields_required: Vec<FieldTypeAnalyzer>,
        fields_option: Vec<FieldTypeAnalyzer>,
        fields_default: Vec<(FieldTypeAnalyzer, AttributeItem)>,
    ) -> Self {
        Self {
            name,
            fields_required,
            fields_option,
            fields_default,
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

    fn serde_struct_ident(&self) -> Ident {
        Ident::new(&format!("{}Serde", self.name), self.name.span())
    }

    fn gen_serialize_fn(
        &self,
        struct_name: &Ident,
        struct_gen: &TokenStream,
        names: &[FieldTypeAnalyzer],
    ) -> TokenStream {
        let name = &self.name;
        let names = names.iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });

        quote! {
            impl burn::serde::Serialize for #name {

                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: burn::serde::Serializer {
                    #[derive(burn::serde::Serialize)]
                    #[serde(crate = "burn::serde")]
                    #struct_gen

                    let serde_state = #struct_name {
                        #(#names),*
                    };
                    serde_state.serialize(serializer)
                }
            }

        }
    }

    fn gen_deserialize_fn(
        &self,
        struct_name: &Ident,
        struct_gen: &TokenStream,
        names: &[FieldTypeAnalyzer],
    ) -> TokenStream {
        let name = &self.name;
        let names = names.iter().map(|name| {
            let name = name.ident();
            quote! { #name: serde_state.#name }
        });

        quote! {
            impl<'de> burn::serde::Deserialize<'de> for #name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: burn::serde::Deserializer<'de> {
                    #[derive(burn::serde::Deserialize)]
                    #[serde(crate = "burn::serde")]
                    #struct_gen

                    let serde_state = #struct_name::deserialize(deserializer)?;
                    Ok(#name {
                        #(#names),*
                    })
                }
            }

        }
    }

    fn gen_serde_struct(&self, names: &[TokenStream]) -> TokenStream {
        let struct_name = self.serde_struct_ident();

        quote! {
            struct #struct_name {
                #(#names),*
            }

        }
    }
}

impl ConfigAnalyzer for ConfigStructAnalyzer {
    fn gen_new_fn(&self) -> TokenStream {
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
            match value {
                syn::Lit::Str(value) => {
                    let stream: proc_macro2::TokenStream = value.value().parse().unwrap();

                    body.extend(quote! {
                        #name: #stream,
                    });
                }
                _ => {
                    body.extend(quote! {
                        #name: #value,
                    });
                }
            };
        }

        let body = quote! {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                #(#names),*
            ) -> Self {
                Self { #body }
            }
        };
        self.wrap_impl_block(body)
    }

    fn gen_builder_fns(&self) -> TokenStream {
        let mut body = quote! {};

        for (field, _) in self.fields_default.iter() {
            let name = field.ident();
            let doc = field.doc().unwrap_or_else(|| {
                quote! {
                        /// Set the default value for the field.
                }
            });
            let ty = &field.field.ty;
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                #doc
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();
            let ty = &field.field.ty;
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                /// Set the default value for the field.
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        self.wrap_impl_block(body)
    }

    fn gen_serde_impl(&self) -> TokenStream {
        let names = self.names();

        let struct_name = self.serde_struct_ident();
        let name_types = self.name_types(&names);
        let struct_gen = self.gen_serde_struct(&name_types);

        let serialize_gen = self.gen_serialize_fn(&struct_name, &struct_gen, &names);
        let deserialize_gen = self.gen_deserialize_fn(&struct_name, &struct_gen, &names);

        quote! {
            #serialize_gen
            #deserialize_gen
        }
    }

    fn gen_clone_impl(&self) -> TokenStream {
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

    fn gen_display_impl(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl core::fmt::Display for #name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.write_str(&burn::config::config_to_json(self))
                }
            }
        }
    }

    fn gen_config_impl(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl burn::config::Config for #name {
            }
        }
    }
}
