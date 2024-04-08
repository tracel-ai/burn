use std::collections::HashMap;

/// Kernel source code abstraction allowing for templating.
///
/// The templates can have text placeholders in the form {{ label }}.
/// They will be updated with their proper value when `generate` is called.
#[derive(Debug)]
pub struct SourceTemplate {
    items: HashMap<String, String>,
    templates: Vec<String>,
}

impl SourceTemplate {
    /// Create a new source template.
    pub fn new<S>(template: S) -> Self
    where
        S: Into<String>,
    {
        Self {
            items: HashMap::new(),
            templates: vec![template.into()],
        }
    }

    /// Register the value for a placeholder item.
    ///
    /// # Notes
    ///
    /// The value can't have placeholders, since it would require recursive templating with
    /// possibly circular dependencies. If you want to add a value that has some
    /// placeholders, consider adding a new template to the source using
    /// [add_template](SourceTemplate::add_template). The added template can be a function, and you can
    /// register the function call instead.
    pub fn register<Name, Value>(mut self, name: Name, value: Value) -> Self
    where
        Name: Into<String>,
        Value: Into<String>,
    {
        self.items.insert(name.into(), value.into());
        self
    }

    /// Add a new template.
    pub fn add_template<S>(mut self, template: S) -> Self
    where
        S: Into<String>,
    {
        self.templates.push(template.into());
        self
    }

    /// Complete the template and returns the source code.
    pub fn complete(mut self) -> String {
        let mut source = self.templates.remove(0);

        for s in self.templates.into_iter() {
            source.push_str(&s);
        }

        let template = text_placeholder::Template::new(&source);
        let mut context = HashMap::new();

        for (key, value) in self.items.iter() {
            context.insert(key.as_str(), value.as_str());
        }

        template.fill_with_hashmap(&context)
    }
}
