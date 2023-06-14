use std::collections::HashMap;

#[derive(Debug)]
pub struct Source {
    map: HashMap<String, String>,
    sources: Vec<String>,
}

impl Source {
    pub fn new<S>(source: S) -> Self
    where
        S: Into<String>,
    {
        Self {
            map: HashMap::new(),
            sources: vec![source.into()],
        }
    }

    pub fn register<Name, Value>(mut self, name: Name, value: Value) -> Self
    where
        Name: Into<String>,
        Value: Into<String>,
    {
        self.map.insert(name.into(), value.into());
        self
    }

    pub fn add_source<S>(mut self, source: S) -> Self
    where
        S: Into<String>,
    {
        self.sources.push(source.into());
        self
    }

    pub fn generate(mut self) -> String {
        let mut source = self.sources.remove(0);

        for s in self.sources.into_iter() {
            source.push_str(&s);
        }

        let template = text_placeholder::Template::new(&source);
        let mut context = HashMap::new();

        for (key, value) in self.map.iter() {
            context.insert(key.as_str(), value.as_str());
        }

        template.fill_with_hashmap(&context)
    }
}
