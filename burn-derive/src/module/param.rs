use crate::shared::field::{parse_fields, FieldTypeAnalyzer};
use proc_macro2::TokenStream;
use quote::quote;

pub struct Param {
    fields_param: Vec<FieldTypeAnalyzer>,
    fields_other: Vec<FieldTypeAnalyzer>,
}

impl Param {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        let fields_param = parse_fields(ast)
            .into_iter()
            .map(FieldTypeAnalyzer::new)
            .filter(FieldTypeAnalyzer::is_param)
            .collect();
        let fields_other = parse_fields(ast)
            .into_iter()
            .map(FieldTypeAnalyzer::new)
            .filter(|val| !FieldTypeAnalyzer::is_param(val))
            .collect();

        Self {
            fields_param,
            fields_other,
        }
    }

    pub fn gen_num_params_fn(&self) -> TokenStream {
        let mut body = quote! {
            let mut num_params = 0;
        };
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                num_params += self.#name.num_params();
            });
        }
        body.extend(quote! {
            num_params
        });

        quote! {
            fn num_params(&self) -> usize {
                #body
            }
        }
    }

    pub fn gen_visit_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                self.#name.visit(visitor);
            });
        }

        quote! {
            fn visit<V: burn::module::ModuleVisitor<Self::Backend>>(&self, visitor: &mut V) {
                #body
            }
        }
    }

    pub fn gen_visit_mut_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                self.#name.visit_mut(visitor);
            });
        }

        quote! {
            fn visit_mut<V: burn::module::ModuleVisitorMut<Self::Backend>>(&mut self, visitor: &mut V) {
                #body
            }
        }
    }

    pub fn gen_devices_fn(&self) -> TokenStream {
        let mut body = quote! {
            let mut devices = Vec::new();
        };
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                devices.append(&mut self.#name.devices());
            });
        }

        body.extend(quote! {
            devices
        });

        quote! {
            fn devices(&self) -> Vec<B::Device> {
                #body
            }
        }
    }

    pub fn gen_to_device_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                self.#name.to_device(device);
            });
        }

        quote! {
            fn to_device(&mut self, device: B::Device) {
                #body
            }
        }
    }

    pub fn gen_detach_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                self.#name.detach();
            });
        }

        quote! {
            fn detach(&mut self) {
                #body
            }
        }
    }

    pub fn gen_inner_fn(&self) -> TokenStream {
        let mut body = quote! {};
        let mut names = Vec::new();
        for field in self.fields_param.iter() {
            let name = field.ident();
            names.push(name.clone());

            body.extend(quote! {
                let #name = self.#name.inner();
            });
        }
        for field in self.fields_other.iter() {
            let name = field.ident();
            names.push(name.clone());

            body.extend(quote! {
                let #name = self.#name.clone();
            });
        }

        quote! {
            fn inner(&self) -> Self::InnerModule {
                #body

                Self::InnerModule {
                    #(#names),*
                }
            }
        }
    }

    pub fn gen_clone_fn(&self) -> TokenStream {
        let mut body = quote! {};
        let mut names = Vec::new();
        let mut fields = Vec::new();

        fields.append(&mut self.fields_param.clone());
        fields.append(&mut self.fields_other.clone());
        for field in fields {
            let name = field.ident();
            names.push(name.clone());

            body.extend(quote! {
                let #name = self.#name.clone();
            });
        }

        quote! {
            fn clone(&self) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    pub fn gen_state_fn(&self) -> TokenStream {
        let mut body = quote! {
            let mut state = burn::module::StateNamed::new();
        };
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                state.register_state(stringify!(#name), self.#name.state());
            });
        }

        quote! {
            fn state(&self) -> burn::module::State<<Self::Backend as burn::tensor::backend::Backend>::Elem>
            {
                #body
                burn::module::State::StateNamed(state)
            }
        }
    }

    pub fn gen_load_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields_param.iter() {
            let name = field.ident();
            body.extend(quote! {
                let state_mod = state.get(stringify!(#name)).ok_or(
                    burn::module::LoadingError::new(format!(
                        "Missing module '{}' from state",
                        stringify!(#name),
                    )))?;
                self.#name.load(state_mod).map_err(|err| {
                    burn::module::LoadingError::new(format!("Can't load module {}: {}", stringify!(#name), err))
                })?;
            });
        }
        quote! {
            fn load(&mut self, state: &burn::module::State<<Self::Backend as burn::tensor::backend::Backend>::Elem>) -> Result<(), burn::module::LoadingError>
            {
                #body
                Ok(())
            }
        }
    }
}
