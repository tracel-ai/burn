use crate::field::FieldTypeAnalyzer;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Field;

pub struct Param {
    fields: Vec<FieldTypeAnalyzer>,
}

impl Param {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        let fields = parse_fields(ast)
            .into_iter()
            .map(FieldTypeAnalyzer::new)
            .filter(FieldTypeAnalyzer::is_param)
            .collect();

        Self { fields }
    }

    pub fn gen_num_params_fn(&self) -> TokenStream {
        let mut body = quote! {
            let mut num_params = 0;
        };
        for field in self.fields.iter() {
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

    pub fn gen_update_params_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields.iter() {
            let name = field.ident();
            body.extend(quote! {
                self.#name.update_params(grads, optim);
            });
        }

        quote! {
            fn update_params<O: burn::optim::Optimizer<B>>(&mut self, grads: &burn::tensor::Gradients, optim: &mut O)
                where
                B: burn::tensor::back::ad::Backend {
                #body
            }
        }
        .into()
    }

    pub fn gen_devices_fn(&self) -> TokenStream {
        let mut body = quote! {
            let mut devices = Vec::new();
        };
        for field in self.fields.iter() {
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
        .into()
    }

    pub fn gen_to_device_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields.iter() {
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
        .into()
    }

    pub fn gen_state_fn(&self) -> TokenStream {
        let mut body = quote! {
            let mut state = burn::module::State::new(self.name());
        };
        for field in self.fields.iter() {
            let name = field.ident();
            body.extend(quote! {
                state.register_child(self.#name.state(stringify!(#name)));
            });
        }

        quote! {
            fn state(&self) -> burn::module::State<Self::Backend>
            where
                <Self::Backend as burn::tensor::back::Backend>::Elem: serde::Serialize,
                <Self::Backend as burn::tensor::back::Backend>::Elem: serde::de::DeserializeOwned,
            {
                #body
                state
            }
        }
        .into()
    }

    pub fn gen_load_from_parent_fn(&self) -> TokenStream {
        let mut body = quote! {};
        for field in self.fields.iter() {
            let name = field.ident();
            body.extend(quote! {
                let name_new = if name == "" {
                    format!("{}.{}", self.name(), stringify!(#name))
                } else {
                    format!("{}.{}.{}", name, self.name(), stringify!(#name))
                };
                self.#name.load_from_parent(name_new.as_str(), state);
            });
        }

        quote! {
            fn load_from_parent(&mut self, name: &str, state: &burn::module::State<Self::Backend>)
            where
                <Self::Backend as burn::tensor::back::Backend>::Elem: serde::Serialize,
                <Self::Backend as burn::tensor::back::Backend>::Elem: serde::de::DeserializeOwned,
            {
                #body
            }
        }
        .into()
    }

    pub fn gen_load_fn(&self) -> TokenStream {
        quote! {
            fn load(&mut self, state: &burn::module::State<Self::Backend>)
            where
                <Self::Backend as burn::tensor::back::Backend>::Elem: serde::Serialize,
                <Self::Backend as burn::tensor::back::Backend>::Elem: serde::de::DeserializeOwned,
            {
                self.load_from_parent("", state)
            }
        }
        .into()
    }

}

fn parse_fields(ast: &syn::DeriveInput) -> Vec<Field> {
    let mut fields = Vec::new();

    match &ast.data {
        syn::Data::Struct(struct_data) => {
            for field in struct_data.fields.iter() {
                fields.push(field.clone());
            }
        }
        syn::Data::Enum(_) => panic!("Only struct can be derived"),
        syn::Data::Union(_) => panic!("Only struct cna be derived"),
    };
    fields
}
