use syn::{Attribute, Ident, Meta, NestedMeta};

pub struct AttributeAnalyzer {
    attr: Attribute,
}

#[derive(Clone)]
pub struct AttributeItem {
    pub ident: Ident,
    pub value: syn::Lit,
}

impl AttributeAnalyzer {
    pub fn new(attr: Attribute) -> Self {
        Self { attr }
    }

    pub fn items(&self) -> Vec<AttributeItem> {
        let config = match self.attr.parse_meta() {
            Ok(val) => val,
            Err(err) => panic!("Fail to parse items: {err:?}"),
        };
        let nested = match config {
            Meta::List(val) => val.nested,
            _ => return Vec::new(),
        };

        let mut output = Vec::new();
        for pair in nested.into_iter() {
            if let NestedMeta::Meta(Meta::NameValue(value)) = pair {
                output.push(AttributeItem {
                    ident: value.path.get_ident().unwrap().clone(),
                    value: value.lit,
                });
            };
        }
        output
    }

    pub fn has_name(&self, name: &str) -> bool {
        Self::path_syn_name(&self.attr.path) == name
    }

    fn path_syn_name(path: &syn::Path) -> String {
        let length = path.segments.len();
        let mut name = String::new();
        for (i, segment) in path.segments.iter().enumerate() {
            if i == length - 1 {
                name += segment.ident.to_string().as_str();
            } else {
                let tmp = segment.ident.to_string() + "::";
                name += tmp.as_str();
            }
        }
        name
    }
}
