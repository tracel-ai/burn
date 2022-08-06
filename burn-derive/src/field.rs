use proc_macro2::Ident;
use syn::{Field, Type, TypePath};

pub struct FieldTypeAnalyzer {
    pub field: Field,
}

impl FieldTypeAnalyzer {
    pub fn new(field: Field) -> Self {
        FieldTypeAnalyzer { field }
    }

    pub fn ident(&self) -> Ident {
        self.field.ident.clone().unwrap()
    }

    pub fn is_of_type(&self, paths: &Vec<&str>) -> bool {
        match &self.field.ty {
            syn::Type::Path(path) => {
                let name = Self::path_name(&path);
                paths.contains(&name.as_str())
            }
            _ => false,
        }
    }

    #[allow(dead_code)]
    pub fn first_generic_field(&self) -> TypePath {
        let err = || {
            panic!(
                "Field {} as no generic",
                self.field.ident.clone().unwrap().to_string()
            )
        };
        match &self.field.ty {
            syn::Type::Path(path) => Self::path_generic_argument(path),
            _ => err(),
        }
    }
    pub fn path_generic_argument(path: &TypePath) -> TypePath {
        let segment = path.path.segments.last().unwrap();
        let err = || {
            panic!(
                "Path segment {} has no generic",
                segment.ident.clone().to_string(),
            )
        };
        match &segment.arguments {
            syn::PathArguments::None => err(),
            syn::PathArguments::AngleBracketed(param) => {
                let first_param = param.args.first().unwrap();
                match first_param {
                    syn::GenericArgument::Type(ty) => match ty {
                        Type::Path(path) => {
                            return path.clone();
                        }
                        _ => err(),
                    },
                    _ => err(),
                }
            }
            syn::PathArguments::Parenthesized(_) => err(),
        }
    }

    pub fn path_name(path: &TypePath) -> String {
        let length = path.path.segments.len();
        let mut name = String::new();
        for (i, segment) in path.path.segments.iter().enumerate() {
            if i == length - 1 {
                name += segment.ident.to_string().as_str();
            } else {
                let tmp = segment.ident.to_string() + "::";
                name += tmp.as_str();
            }
        }
        name
    }

    pub fn is_param(&self) -> bool {
        let params_types = vec!["Param", "burn::Param"];
        self.is_of_type(&params_types)
    }
}
