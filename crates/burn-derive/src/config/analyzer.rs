use super::ConfigEnumAnalyzer;
use crate::config::ConfigStructAnalyzer;
use crate::shared::field::FieldTypeAnalyzer;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Field, Ident};

pub struct ConfigAnalyzerFactory {}

pub trait ConfigAnalyzer {
    fn gen_new_fn(&self) -> TokenStream {
        quote! {}
    }
    fn gen_builder_fns(&self) -> TokenStream {
        quote! {}
    }
    fn gen_serde_impl(&self) -> TokenStream;
    fn gen_clone_impl(&self) -> TokenStream;
    fn gen_display_impl(&self) -> TokenStream;
    fn gen_config_impl(&self) -> TokenStream;
}

impl ConfigAnalyzerFactory {
    pub fn new() -> Self {
        Self {}
    }

    pub fn create_analyzer(&self, item: &syn::DeriveInput) -> syn::Result<Box<dyn ConfigAnalyzer>> {
        let name = item.ident.clone();
        let config_type = parse_asm(item);

        Ok(match config_type {
            ConfigType::Struct(data) => Box::new(self.create_struct_analyzer(name, data)?),
            ConfigType::Enum(data) => Box::new(self.create_enum_analyzer(name, data)),
        })
    }

    fn create_struct_analyzer(
        &self,
        name: Ident,
        fields: Vec<Field>,
    ) -> syn::Result<ConfigStructAnalyzer> {
        let fields = fields.into_iter().map(FieldTypeAnalyzer::new);

        let mut fields_required = Vec::new();
        let mut fields_option = Vec::new();
        let mut fields_default = Vec::new();
        let mut fields_into = Vec::new();

        for field in fields {
            let mut default = None;
            let mut into = false;
            for attr in &field.field.attrs {
                if !attr.path().is_ident("config") {
                    continue;
                }
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("default") {
                        if default.is_some() {
                            return Err(meta.error("duplicate `default` attribute"));
                        }
                        default = Some(meta.value()?.parse()?);
                    } else if meta.path.is_ident("into") {
                        if into {
                            return Err(meta.error("duplicate `into` attribute"));
                        }
                        into = true;
                    } else {
                        return Err(meta.error("expected `default` or `into`"));
                    }
                    Ok(())
                })?;
            }

            if into {
                fields_into.push(field.ident());
            }

            if let Some(default) = default {
                fields_default.push((field, default));
                continue;
            }

            if field.is_of_type(&["Option"]) {
                fields_option.push(field);
                continue;
            }

            fields_required.push(field);
        }

        Ok(ConfigStructAnalyzer::new(
            name,
            fields_required,
            fields_option,
            fields_default,
            fields_into,
        ))
    }

    fn create_enum_analyzer(&self, name: Ident, data: syn::DataEnum) -> ConfigEnumAnalyzer {
        ConfigEnumAnalyzer::new(name, data)
    }
}

enum ConfigType {
    Struct(Vec<Field>),
    Enum(syn::DataEnum),
}

fn parse_asm(ast: &syn::DeriveInput) -> ConfigType {
    match &ast.data {
        syn::Data::Struct(struct_data) => {
            ConfigType::Struct(struct_data.fields.clone().into_iter().collect())
        }
        syn::Data::Enum(enum_data) => ConfigType::Enum(enum_data.clone()),
        syn::Data::Union(_) => panic!("Only struct and enum can be derived"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Malformed field attributes must report diagnostics instead of being ignored or panicking.
    #[test]
    fn rejects_invalid_field_attributes() {
        for (attributes, message) in [
            (quote!(#[config(into = true)]), "expected `,`"),
            (quote!(#[config(into)] #[config(into)]), "duplicate `into`"),
            (
                quote!(#[config(default = 1, default = 2)]),
                "duplicate `default`",
            ),
            (quote!(#[config(convert)]), "expected `default` or `into`"),
        ] {
            let item = syn::parse_quote! {
                struct InvalidConfig {
                    #attributes
                    value: usize,
                }
            };
            let Err(error) = ConfigAnalyzerFactory::new().create_analyzer(&item) else {
                panic!("expected invalid attributes to be rejected");
            };
            assert!(error.to_string().contains(message), "{error}");
        }
    }
}
