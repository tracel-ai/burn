use super::ConfigEnumAnalyzer;
use crate::config::ConfigStructAnalyzer;
use crate::shared::{attribute::AttributeItem, field::FieldTypeAnalyzer};
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

    pub fn create_analyzer(&self, item: &syn::DeriveInput) -> Box<dyn ConfigAnalyzer> {
        let name = item.ident.clone();
        let config_type = parse_asm(item);

        match config_type {
            ConfigType::Struct(data) => Box::new(self.create_struct_analyzer(name, data)),
            ConfigType::Enum(data) => Box::new(self.create_enum_analyzer(name, data)),
        }
    }

    fn create_struct_analyzer(&self, name: Ident, fields: Vec<Field>) -> ConfigStructAnalyzer {
        let fields = fields.into_iter().map(FieldTypeAnalyzer::new);

        let mut fields_required = Vec::new();
        let mut fields_option = Vec::new();
        let mut fields_default = Vec::new();

        for field in fields {
            let attributes: Vec<AttributeItem> = field
                .attributes()
                .filter(|attr| attr.has_name("config"))
                .map(|attr| attr.item())
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

        ConfigStructAnalyzer::new(name, fields_required, fields_option, fields_default)
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
