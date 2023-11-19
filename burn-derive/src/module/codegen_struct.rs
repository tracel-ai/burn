use crate::shared::field::{parse_fields, FieldTypeAnalyzer};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

use super::codegen::ModuleCodegen;

pub(crate) struct StructModuleCodegen {
    pub fields: Vec<FieldTypeAnalyzer>,
}

impl ModuleCodegen for StructModuleCodegen {
    fn gen_num_params(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                num_params += burn::module::Module::<B>::num_params(&self.#name);
            }
        });

        quote! {
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                #body
                num_params
            }
        }
    }

    fn gen_visit(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                burn::module::Module::visit(&self.#name, visitor);
            }
        });

        quote! {
            fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
                #body
            }
        }
    }

    fn gen_map(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::Module::map(self.#name, mapper);
            }
        });

        quote! {
            fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    fn gen_valid(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::AutodiffModule::<B>::valid(&self.#name);
            }
        });

        quote! {
            fn valid(&self) -> Self::InnerModule {
                #body

                Self::InnerModule {
                    #(#names),*
                }
            }
        }
    }

    fn gen_into_record(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                #name: burn::module::Module::<B>::into_record(self.#name),
            }
        });

        quote! {
            fn into_record(self) -> Self::Record {
                Self::Record {
                    #body
                }
            }
        }
    }

    fn gen_load_record(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                #name: burn::module::Module::<B>::load_record(self.#name, record.#name),
            }
        });

        quote! {
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    #body
                }
            }
        }
    }

    fn gen_clone(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = self.#name.clone();
            }
        });

        quote! {
            fn clone(&self) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }
}

impl StructModuleCodegen {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            fields: parse_fields(ast)
                .into_iter()
                .map(FieldTypeAnalyzer::new)
                .collect(),
        }
    }

    fn gen_fields_fn_names<F>(&self, func: F) -> (Vec<Ident>, TokenStream)
    where
        F: Fn(Ident) -> TokenStream,
    {
        let mut body = quote! {};
        let mut names = Vec::new();

        for field in self.fields.iter() {
            let name = field.ident();

            names.push(name.clone());
            body.extend(func(field.ident()));
        }

        (names, body)
    }

    fn gen_fields_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(Ident) -> TokenStream,
    {
        let mut body = quote! {};

        for field in self.fields.iter() {
            body.extend(func(field.ident()));
        }

        body
    }
}
