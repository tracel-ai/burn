use super::codegen::ModuleCodegen;
use proc_macro2::{Ident, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{Field, Member};

pub(crate) struct StructModuleCodegen {
    pub name: Ident,
    pub fields: Vec<ModuleField>,
}

impl ModuleCodegen for StructModuleCodegen {
    fn gen_num_params(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_module {
                quote! {
                    num_params += burn::module::Module::num_params(&self.#name);
                }
            } else {
                quote! {} // other fields have 0 params
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
        let struct_name = self.name.to_string();
        let container_type = format!("Struct:{}", struct_name);
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_module {
                let name_str = name.to_token_stream().to_string();
                quote! {
                    visitor.enter_module(#name_str, #container_type);
                    burn::module::Module::visit(&self.#name, visitor);
                    visitor.exit_module(#name_str, #container_type);
                }
            } else {
                quote! {}
            }
        });

        quote! {
            fn visit<Visitor: burn::module::ModuleVisitor>(&self, visitor: &mut Visitor) {
                #body
            }
        }
    }

    fn gen_collect_devices(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_module {
                quote! {
                    let devices = burn::module::Module::collect_devices(&self.#name, devices);
                }
            } else {
                quote! {}
            }
        });

        quote! {
            fn collect_devices(
                &self,
                devices: burn::module::Devices
            ) -> burn::module::Devices {
                #body
                devices
            }
        }
    }

    fn gen_to_device(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|local, name, field_type| {
            if field_type.is_module {
                quote! {
                    let #local = burn::module::Module::to_device(self.#name, device);
                }
            } else {
                quote! { let #local = self.#name; }
            }
        });

        quote! {
            fn to_device(self, device: &burn::tensor::Device) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_fork(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|local, name, field_type| {
            if field_type.is_module {
                quote! {
                    let #local = burn::module::Module::fork(self.#name, device);
                }
            } else {
                quote! { let #local = self.#name; }
            }
        });

        quote! {
            fn fork(self, device: &burn::tensor::Device) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_map(&self) -> TokenStream {
        let struct_name = self.name.to_string();
        let container_type = format!("Struct:{}", struct_name);
        let (names, body) = self.gen_fields_fn_names(|local, name, field_type| {
            if field_type.is_module {
                let name_str = name.to_token_stream().to_string();
                quote! {
                    mapper.enter_module(#name_str, #container_type);
                    let #local = burn::module::Module::map(self.#name, mapper);
                    mapper.exit_module(#name_str, #container_type);
                }
            } else {
                quote! { let #local = self.#name; }
            }
        });

        quote! {
            fn map<Mapper: burn::module::ModuleMapper>(self, mapper: &mut Mapper) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_valid(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|local, name, field_type| {
            if field_type.is_module {
                quote! {
                    let #local = burn::module::Module::valid(&self.#name);
                }
            } else {
                quote! { let #local = self.#name.clone(); }
            }
        });

        quote! {
            fn valid(&self) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_train(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|local, name, field_type| {
            if field_type.is_module {
                quote! {
                    let #local = burn::module::Module::train(self.#name);
                }
            } else {
                quote! { let #local = self.#name; }
            }
        });

        quote! {
            fn train(self) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_materialize(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|local, name, field_type| {
            if field_type.is_module {
                quote! {
                    let #local = burn::module::Module::materialize(self.#name);
                }
            } else {
                quote! { let #local = self.#name; }
            }
        });

        quote! {
            fn materialize(self) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_clone(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|local, name, _field_type| {
            quote! {
                let #local = self.#name.clone();
            }
        });

        quote! {
            fn clone(&self) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn fields(&self) -> Vec<(&Field, &ModuleFieldType)> {
        self.fields
            .iter()
            .map(|field| (&field.field, &field.field_type))
            .collect()
    }

    fn gen_display(&self) -> TokenStream {
        let struct_ident = &self.name;
        let field_prints = self.fields.iter().map(|field| {
            let field_name = &field.member;
            if field.field_type.is_module {
                // Standard module type, use underlying `ModuleDisplay` impl
                quote! { .add(stringify!(#field_name), &self.#field_name) }
            } else {
                // Not a module, use the debug implementation
                quote! {
                    .add_debug_attribute(stringify!(#field_name), &self.#field_name)
                }
            }
        });
        quote! {
            fn content(&self, mut content: burn::module::Content) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&stringify!(#struct_ident))
                    #(#field_prints)*
                    .optional()
            }
        }
    }
}

impl StructModuleCodegen {
    pub fn from_ast(ast: &syn::DeriveInput) -> syn::Result<Self> {
        Ok(Self {
            name: ast.ident.clone(),
            fields: parse_module_fields(ast)?,
        })
    }

    fn gen_fields_fn_names<F>(&self, func: F) -> (Vec<TokenStream>, TokenStream)
    where
        F: Fn(Ident, Member, &ModuleFieldType) -> TokenStream,
    {
        let mut body = quote! {};
        let mut names = Vec::new();

        for (index, field) in self.fields.iter().enumerate() {
            let member = &field.member;
            let local = format_ident!("__burn_field_{index}");
            names.push(quote! { #member: #local });
            body.extend(func(local, member.clone(), &field.field_type));
        }

        (names, body)
    }

    fn gen_fields_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(Member, &ModuleFieldType) -> TokenStream,
    {
        let mut body = quote! {};
        for field in &self.fields {
            body.extend(func(field.member.clone(), &field.field_type));
        }
        body
    }

    pub fn has_module_fields(&self) -> bool {
        self.fields.iter().any(|f| f.field_type.is_module)
    }
}

#[derive(new, Debug)]
pub struct ModuleField {
    pub field: Field,
    pub member: Member,
    pub field_type: ModuleFieldType,
}

#[derive(Debug)]
pub enum ModuleFieldAttribute {
    Skip,
}

#[derive(Default, Debug)]
pub struct ModuleFieldType {
    pub is_module: bool,
    pub attr: Option<ModuleFieldAttribute>,
}

pub(crate) fn parse_module_fields(ast: &syn::DeriveInput) -> syn::Result<Vec<ModuleField>> {
    let syn::Data::Struct(data) = &ast.data else {
        return Err(syn::Error::new_spanned(ast, "Only structs are supported"));
    };
    data.fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let member = match &field.ident {
                Some(ident) => Member::Named(ident.clone()),
                None => Member::Unnamed(index.into()),
            };
            Ok(ModuleField::new(
                field.clone(),
                member,
                parse_module_field_type(field)?,
            ))
        })
        .collect()
}

pub(crate) fn parse_module_field_type(field: &Field) -> syn::Result<ModuleFieldType> {
    let mut field_type = ModuleFieldType::default();
    for attr in &field.attrs {
        if attr.path().is_ident("module") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("skip") {
                    if is_param_type(&field.ty) {
                        return Err(meta.error("Fields of type 'Param' should not be marked as 'skip'. Use a 'Tensor' instead."));
                    }
                    field_type.attr = Some(ModuleFieldAttribute::Skip);
                    Ok(())
                } else {
                    let path = meta.path.to_token_stream().to_string();
                    Err(meta.error(format!("Unsupported module attribute: {path}")))
                }
            })?;
        }
    }
    field_type.is_module = !is_primitive_type(&field.ty) && field_type.attr.is_none();
    Ok(field_type)
}

fn type_matches_ident(ty: &syn::Type, idents: &[&str]) -> bool {
    if let syn::Type::Path(type_path) = ty {
        // Look at the last segment of the path (e.g., 'Param' in 'burn::module::Param')
        if let Some(segment) = type_path.path.segments.last() {
            return idents.contains(&segment.ident.to_string().as_str());
        }
    }
    false
}

fn is_primitive_ident(ident: &str) -> bool {
    matches!(
        ident,
        "bool"
            | "char"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f32"
            | "f64"
            | "String"
    )
}

fn is_primitive_type(ty: &syn::Type) -> bool {
    match ty {
        // e.g. usize, String, or Option<T>
        syn::Type::Path(type_path) => {
            let segment = match type_path.path.segments.last() {
                Some(seg) => seg,
                None => return false,
            };

            let ident = segment.ident.to_string();

            // Direct primitive
            if is_primitive_ident(&ident) {
                return true;
            }

            // Only known containers can inherit constant classification from their contents.
            // A custom type such as Block<f32> may still contain trainable parameters.
            match &segment.arguments {
                syn::PathArguments::AngleBracketed(args) if is_constant_container(&ident) => {
                    args.args.iter().all(|arg| {
                        if let syn::GenericArgument::Type(inner_ty) = arg {
                            is_primitive_type(inner_ty)
                        } else {
                            false
                        }
                    })
                }
                _ => false,
            }
        }

        // e.g. (T, U, ...)
        syn::Type::Tuple(tuple) => tuple.elems.iter().all(is_primitive_type),

        // e.g. [T; N]
        syn::Type::Array(array) => is_primitive_type(&array.elem),

        // e.g. &T or &mut T
        syn::Type::Reference(reference) => is_primitive_type(&reference.elem),

        // e.g. *const T / *mut T
        syn::Type::Ptr(ptr) => is_primitive_type(&ptr.elem),

        // e.g. (T)
        syn::Type::Paren(paren) => is_primitive_type(&paren.elem),

        // slices `[T]`
        syn::Type::Slice(slice) => is_primitive_type(&slice.elem),

        _ => false,
    }
}

fn is_constant_container(ident: &str) -> bool {
    matches!(
        ident,
        "Option"
            | "Result"
            | "Vec"
            | "Box"
            | "Arc"
            | "Weak"
            | "HashMap"
            | "BTreeMap"
            | "HashSet"
            | "BTreeSet"
            | "VecDeque"
            | "LinkedList"
            | "BinaryHeap"
            | "PhantomData"
            | "Cell"
            | "RefCell"
            | "Mutex"
            | "RwLock"
    )
}

fn is_param_type(ty: &syn::Type) -> bool {
    type_matches_ident(ty, &["Param"])
}

#[cfg(test)]
mod tests {
    use super::{is_primitive_type, parse_module_fields};

    #[test]
    fn skipped_parameters_remain_an_error() {
        let ast = syn::parse_quote! {
            struct Model {
                #[module(skip)]
                weight: Param<Tensor<2>>,
            }
        };
        let error = parse_module_fields(&ast).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Fields of type 'Param' should not be marked as 'skip'")
        );
    }

    #[test]
    fn unsupported_field_attributes_remain_an_error() {
        let ast = syn::parse_quote! {
            struct Model {
                #[module(unknown)]
                state: Config,
            }
        };
        let error = parse_module_fields(&ast).unwrap_err();
        assert_eq!(error.to_string(), "Unsupported module attribute: unknown");
    }

    #[test]
    fn containers_must_not_hide_custom_modules() {
        for ty in [
            "Block<f32>",
            "Option<Vec<Block<f32>>>",
            "Box<Block<f32>>",
            "Arc<Block<f32>>",
            "HashMap<String, Block<f32>>",
            "Result<usize, Block<f32>>",
        ] {
            assert!(!is_primitive_type(&syn::parse_str(ty).unwrap()), "{ty}");
        }
    }
}
