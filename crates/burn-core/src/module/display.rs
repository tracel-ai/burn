use alloc::{
    borrow::ToOwned,
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::any;
use core::fmt::{Display, Write};

/// Default display settings for a module.
pub trait ModuleDisplayDefault {
    /// Attributes of the module used for display purposes.
    fn content(&self, _content: Content) -> Option<Content>;
}

/// Trait to implement custom display settings for a module.
pub trait ModuleDisplay: ModuleDisplayDefault {
    //TODO mention that attribute needs to be passed

    /// Formats the module with provided display settings.
    fn format(&self, passed_settings: DisplaySettings) -> String {
        let settings = if let Some(custom_settings) = self.custom_settings() {
            custom_settings.inherit(passed_settings)
        } else {
            passed_settings
        };

        let indent = " ".repeat(settings.level * settings.indentation_size());
        let indent_close_braces = " ".repeat((settings.level - 1) * settings.indentation_size());

        let settings = settings.level_up();

        let self_type = extract_type_name::<Self>();

        // Use custom content if it is implemented and show_all_attributes is false,
        // otherwise use default content
        let content = if !settings.show_all_attributes() {
            self.custom_content(Content::new(settings.clone()))
                .unwrap_or_else(|| {
                    self.content(Content::new(settings.clone()))
                        .unwrap_or_else(|| {
                            panic!("Default content should be implemented for {self_type}.")
                        })
                })
        } else {
            self.content(Content::new(settings.clone()))
                .unwrap_or_else(|| panic!("Default content should be implemented for {self_type}."))
        };

        let top_level_type = if let Some(top_level_type) = content.top_level_type {
            top_level_type.to_owned()
        } else {
            self_type.to_owned()
        };

        // If there is only one item in the content, return it or no attributes
        if let Some(item) = content.single_item {
            return item;
        } else if content.attributes.is_empty() {
            return top_level_type.to_string();
        }

        let mut result = String::new();

        // Print the struct name
        if settings.new_line_after_attribute() {
            writeln!(result, "{} {{", top_level_type).unwrap();
        } else {
            write!(result, "{} {{", top_level_type).unwrap();
        }

        for (i, attribute) in content.attributes.iter().enumerate() {
            if settings.new_line_after_attribute() {
                writeln!(result, "{indent}{}: {}", attribute.name, attribute.value).unwrap();
            } else if i == 0 {
                write!(result, "{}: {}", attribute.name, attribute.value).unwrap();
            } else {
                write!(result, ", {}: {}", attribute.name, attribute.value).unwrap();
            }
        }

        write!(result, "{indent_close_braces}}}").unwrap();

        result
    }

    /// Custom display settings for the module.
    fn custom_settings(&self) -> Option<DisplaySettings> {
        None
    }

    /// Custom attributes for the module.
    fn custom_content(&self, _content: Content) -> Option<Content> {
        None
    }
}

/// Custom module display settings.
#[derive(Debug, Clone)]
pub struct DisplaySettings {
    /// Whether to print the module parameter ids.
    show_param_id: Option<bool>,

    /// Whether to print the module attributes.
    show_all_attributes: Option<bool>,

    /// Whether to print the module number of parameters.
    show_num_parameters: Option<bool>,

    /// Print new line after an attribute.
    new_line_after_attribute: Option<bool>,

    /// Indentation size.
    indentation_size: Option<usize>,

    /// Level of indentation.
    level: usize,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        DisplaySettings {
            show_param_id: None,
            show_all_attributes: None, // by default, show only the attributes that are not default
            show_num_parameters: None,
            new_line_after_attribute: None,
            indentation_size: None,
            level: 1,
        }
    }
}

impl DisplaySettings {
    /// Create a new format settings.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set a flag to show module parameters.
    pub fn with_show_param_id(mut self, flag: bool) -> Self {
        self.show_param_id = Some(flag);
        self
    }

    /// Set a flag to show module attributes.
    pub fn with_show_all_attributes(mut self, flag: bool) -> Self {
        self.show_all_attributes = Some(flag);
        self
    }

    /// Set a flag to show the number of module parameters.
    pub fn with_show_num_parameters(mut self, flag: bool) -> Self {
        self.show_num_parameters = Some(flag);
        self
    }

    /// Set a flag to print a new line after an attribute.
    pub fn with_new_line_after_attribute(mut self, flag: bool) -> Self {
        self.new_line_after_attribute = Some(flag);
        self
    }

    /// Inherit settings from the provided settings and return a new settings object.
    pub fn inherit(self, top: Self) -> Self {
        let mut updated = self.clone();

        if let Some(show_param_id) = top.show_param_id {
            updated.show_param_id = Some(show_param_id);
        };

        if let Some(show_all_attributes) = top.show_all_attributes {
            updated.show_all_attributes = Some(show_all_attributes);
        }

        if let Some(show_num_parameters) = top.show_num_parameters {
            updated.show_num_parameters = Some(show_num_parameters);
        }

        if let Some(new_line_after_attribute) = top.new_line_after_attribute {
            updated.new_line_after_attribute = Some(new_line_after_attribute);
        }

        if let Some(indentation_size) = top.indentation_size {
            updated.indentation_size = Some(indentation_size);
        }

        updated
    }

    /// A convenience method to wrap the DisplaySettings struct in an option
    pub fn optional(self) -> Option<Self> {
        Some(self)
    }

    /// Increase the level of indentation
    pub fn level_up(mut self) -> Self {
        self.level += 1;
        self
    }

    /// Gets show_param_id flag, substitutes false if not set.
    ///
    /// This flag is used to print the module parameter ids.
    pub fn show_param_id(&self) -> bool {
        self.show_param_id.unwrap_or(false)
    }

    /// Gets show_all_attributes, substitutes false if not set.
    ///
    /// This flag is used to force to print all module attributes, overriding custom attributes.
    pub fn show_all_attributes(&self) -> bool {
        self.show_all_attributes.unwrap_or(false)
    }

    /// Gets show_num_parameters, substitutes false if not set.
    ///
    /// This flag is used to print the number of module parameters.
    pub fn show_num_parameters(&self) -> bool {
        self.show_num_parameters.unwrap_or(false)
    }

    /// Gets new_line_after_attribute, substitutes true if not set.
    ///
    /// This flag is used to print a new line after an attribute.
    pub fn new_line_after_attribute(&self) -> bool {
        self.new_line_after_attribute.unwrap_or(true)
    }

    /// Gets indentation_size, substitutes 2 if not set.
    pub fn indentation_size(&self) -> usize {
        self.indentation_size.unwrap_or(2)
    }
}
/// Struct to store the attributes of a module.
#[derive(Clone, Debug)]
pub struct Content {
    /// List of attributes.
    pub attributes: Vec<Attribute>,

    /// Single item content.
    pub single_item: Option<String>,

    /// Display settings.
    pub display_settings: DisplaySettings,

    /// Top level type name.
    pub top_level_type: Option<String>,
}

impl Content {
    /// Create a new attributes struct.
    pub fn new(display_settings: DisplaySettings) -> Self {
        Content {
            attributes: Vec::new(),
            single_item: None,
            display_settings,
            top_level_type: None,
        }
    }

    /// Add an attribute to the format settings. The value will be formatted and stored as a string.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the attribute.
    /// * `value` - Value of the attribute.
    pub fn add<T: ModuleDisplay + ?Sized>(mut self, name: &str, value: &T) -> Self {
        if self.single_item.is_some() {
            panic!("Cannot add multiple attributes when single item is set.");
        }

        let attribute = Attribute {
            name: name.to_owned(),
            value: value.format(self.display_settings.clone()), // TODO level + 1
            ty: any::type_name::<T>().to_string(),
        };
        self.attributes.push(attribute);
        self
    }

    /// Add a single item.
    ///
    /// # Arguments
    ///
    /// * `value` - Rendered string of the single item.
    pub fn add_single<T: ModuleDisplay>(mut self, value: &T) -> Self {
        if !self.attributes.is_empty() {
            panic!("Cannot add single item when attributes are set.");
        }

        self.single_item = Some(value.format(self.display_settings.clone()));

        self
    }

    /// Add a single item.
    pub fn add_formatted<T: Display>(mut self, value: &T) -> Self {
        if !self.attributes.is_empty() {
            panic!("Cannot add single item when attributes are set.");
        }

        self.single_item = Some(format!("{}", value));
        self
    }

    /// A convenience method to wrap the Attributes struct in an option
    /// because it is often used as an optional field.
    pub fn optional(self) -> Option<Self> {
        if self.attributes.is_empty() && self.single_item.is_none() && self.top_level_type.is_none()
        {
            None
        } else {
            Some(self)
        }
    }

    /// Set the top level type name.
    pub fn set_top_level_type(mut self, ty: &str) -> Self {
        self.top_level_type = Some(ty.to_owned());
        self
    }
}

/// Attribute to print in the display method.
#[derive(Clone, Debug)]
pub struct Attribute {
    /// Name of the attribute.
    pub name: String,

    /// Value of the attribute.
    pub value: String,

    /// Type of the attribute.
    pub ty: String,
}

/// Extracts the short name of a type T
pub fn extract_type_name<T: ?Sized>() -> &'static str {
    // Get the full type name of T, including module path and generic parameters
    let ty = any::type_name::<T>();

    // Find the first occurrence of '<' in the full type name
    // If not found, use the length of the type name
    let end = ty.find('<').unwrap_or(ty.len());

    // Slice the type name up to the first '<' or the end
    let ty = &ty[0..end];

    // Find the last occurrence of "::" in the sliced type name
    // If found, add 2 to skip the "::" itself
    // If not found, start from the beginning of the type name
    let start = ty.rfind("::").map(|i| i + 2).unwrap_or(0);

    // Find the last occurrence of '<' in the sliced type name
    // If not found, use the length of the type name
    let end = ty.rfind('<').unwrap_or(ty.len());

    // If the start index is less than the end index,
    // return the slice of the type name from start to end
    // Otherwise, return the entire sliced type name
    if start < end {
        &ty[start..end]
    } else {
        ty
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_extract_type_name_usize() {
//         let x: usize = 42;
//         assert_eq!(extract_type_name(x), "usize");
//     }

//     #[test]
//     fn test_extract_type_name_vec() {
//         let x = vec![1, 2, 3];
//         assert_eq!(extract_type_name(x), "Vec");
//     }

//     #[test]
//     fn test_extract_type_name_option() {
//         let x = Some(42);
//         assert_eq!(extract_type_name(x), "Option");
//     }

//     #[test]
//     fn test_extract_type_name_result() {
//         let x: Result<i32, String> = Ok(42);
//         assert_eq!(extract_type_name(x), "Result");
//     }

//     #[test]
//     fn test_extract_type_name_hashmap() {
//         use std::collections::HashMap;
//         let mut x = HashMap::new();
//         x.insert("key", "value");
//         assert_eq!(extract_type_name(x), "HashMap");
//     }
// }
