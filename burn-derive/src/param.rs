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

        quote!(
            fn num_params(&self) -> usize {
                #body
            }
        )
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
