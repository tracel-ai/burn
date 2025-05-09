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
        let body_tokens = self
            .fields_required
            .iter()
            .map(|field| {
                let name = field.ident();
                quote! {
                    #name: #name,
                }
            })
            .chain(self.fields_option.iter().map(|field| {
                let name = field.ident();
                quote! {
                    #name: None,
                }
            }))
            .chain(self.fields_default.iter().map(|(field, attribute)| {
                let name = field.ident();
                let value = &attribute.value;
                match value {
                    syn::Lit::Str(value) => {
                        #[allow(unused_variables)]
                        let value_str = value.value();
                        quote! {
                            #name: serde_json::from_str(#value_str)
                                .unwrap_or_else(|e| panic!("Failed to parse default value for field '{}': {}", stringify!(#name), e)),
                        }
                    }
                    _ => quote! {
                        #name: #value,
                    },
                }
            }));

        let name_tokens = self.fields_required.iter().map(|field| {
            let name = field.ident();
            let ty = &field.field.ty;
            quote! {
                #name: #ty
            }
        });

        let field_doc_tokens = self
            .fields_required
            .iter()
            .map(|field| {
                #[allow(unused_variables)]
                let name = field.ident();
                #[allow(unused_variables)]
                let doc = field.doc().unwrap_or_else(|| {
                    quote! {
                        /// Required field.
                    }
                });
                quote! {
                    /// - `#name`: #doc
                }
            })
            .chain(self.fields_option.iter().map(|field| {
                #[allow(unused_variables)]
                let name = field.ident();
                #[allow(unused_variables)]
                let doc = field.doc().unwrap_or_else(|| {
                    quote! {
                        /// Optional field.
                    }
                });
                quote! {
                    /// - `#name`: #doc
                }
            }))
            .chain(self.fields_default.iter().map(|(field, attribute)| {
                #[allow(unused_variables)]
                let name = field.ident();
                #[allow(unused_variables)]
                let doc = field.doc().unwrap_or_else(|| {
                    quote! {
                        /// Field with default value.
                    }
                });
                let value = &attribute.value;
                let default_doc = match value {
                    syn::Lit::Str(value) => {
                        #[allow(unused_variables)]
                        let value_str = value.value();
                        quote! {
                            /// Default: #value_str
                        }
                    }
                    _ => quote! {
                        /// Default: #value
                    },
                };
                quote! {
                    /// - `#name`: #doc
                    #default_doc
                }
            }));

        let impl_block = quote! {
            /// Create a new instance of the config.
            ///
            /// Fields:
            #(#field_doc_tokens)*
            pub fn new(
                #(#name_tokens),*
            ) -> Self {
                Self { #(#body_tokens)* }
            }
        };
        self.wrap_impl_block(impl_block)
    }

    fn gen_builder_fns(&self) -> TokenStream {
        let builder_tokens = self
            .fields_default
            .iter()
            .map(|(field, attribute)| {
                let name = field.ident();
                let doc = field.doc().unwrap_or_else(|| {
                    quote! {
                        /// Set the default value for the field.
                    }
                });
                let ty = &field.field.ty;
                let fn_name = Ident::new(&format!("with_{name}"), name.span());
                let value = &attribute.value;
                let default_doc = match value {
                    syn::Lit::Str(value) => {
                        let _value_str = value.value();
                        quote! {
                            ///
                            /// Defaults to `#_value_str`.
                        }
                    }
                    _ => quote! {
                        ///
                        /// Defaults to `#value`.
                    },
                };

                quote! {
                    #doc
                    #default_doc
                    pub fn #fn_name(mut self, #name: #ty) -> Self {
                        self.#name = #name;
                        self
                    }
                }
            })
            .chain(self.fields_option.iter().map(|field| {
                let name = field.ident();
                let doc = field.doc().unwrap_or_else(|| {
                    quote! {
                        /// Set the optional field value.
                    }
                });
                let ty = &field.field.ty;
                let fn_name = Ident::new(&format!("with_{name}"), name.span());

                quote! {
                    #doc
                    ///
                    /// Defaults to `None`.
                    pub fn #fn_name(mut self, #name: #ty) -> Self {
                        self.#name = #name;
                        self
                    }
                }
            }));

        let impl_block = quote! {
            #(#builder_tokens)*
        };
        self.wrap_impl_block(impl_block)
    }

    fn gen_serde_impl(&self) -> TokenStream {
        let names = self.names();

        let struct_name = self.serde_struct_ident();
        let name_types = self.name_types(&names);
        let struct_gen = self.gen_serde_struct(&name_types[..]);

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
