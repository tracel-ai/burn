use crate::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};

use alloc::{format, vec::Vec};

use burn_tensor::backend::{AutodiffBackend, Backend};
use core::fmt::Debug;

impl<T, B> Module<B> for Option<T>
where
    T: Module<B> + Debug + Send + Clone,
    B: Backend,
{
    type Record = Option<T::Record>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        if let Some(module) = self {
            module.visit(visitor)
        }
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        self.map(|module| module.map(mapper))
    }

    fn load_record(self, record: Self::Record) -> Self {
        let is_constant = self.num_params() == 0;

        if is_constant {
            return self;
        }

        self.zip(record)
            .map(|(module, record)| module.load_record(record))
    }

    fn into_record(self) -> Self::Record {
        self.map(Module::into_record)
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        self.map(|module| module.to_device(device))
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        self.map(|module| module.fork(device))
    }

    fn collect_devices(&self, mut devices: Vec<B::Device>) -> Vec<B::Device> {
        if let Some(module) = self.as_ref() {
            devices = module.collect_devices(devices);
        }

        devices
    }
}

impl<T: ModuleDisplay> ModuleDisplayDefault for Option<T> {
    fn content(&self, content: Content) -> Option<Content> {
        match self {
            Some(module) => content.add_single(module).optional(),
            None => content.add_single("None").optional(),
        }
    }
}

impl<T: ModuleDisplay> ModuleDisplay for Option<T> {}

impl<T, B> AutodiffModule<B> for Option<T>
where
    T: AutodiffModule<B> + Debug + Send + Clone,
    B: AutodiffBackend,
{
    type InnerModule = Option<T::InnerModule>;

    fn valid(&self) -> Self::InnerModule {
        self.as_ref().map(|module| module.valid())
    }
}

impl<T, B> Module<B> for Vec<T>
where
    T: Module<B> + Debug + Send + Clone,
    B: Backend,
{
    type Record = Vec<T::Record>;

    fn num_params(&self) -> usize {
        let mut num_params = 0;
        for module in self.iter() {
            num_params += module.num_params();
        }

        num_params
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.iter().for_each(|module| {
            module.visit(visitor);
        });
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        self.into_iter().map(|module| module.map(mapper)).collect()
    }

    fn into_record(self) -> Self::Record {
        self.into_iter().map(Module::into_record).collect()
    }

    fn load_record(self, record: Self::Record) -> Self {
        assert_eq!(
            self.len(),
            record.len(),
            r#"[Load Record Error] The vec record does not the same length as the module.
            Make sure you module initialization is compatible with the record being loaded.
            "#,
        );

        self.into_iter()
            .zip(record)
            .map(|(module, record)| module.load_record(record))
            .collect()
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        self.into_iter()
            .map(|module| module.to_device(device))
            .collect()
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        self.into_iter().map(|module| module.fork(device)).collect()
    }

    fn collect_devices(&self, mut devices: Vec<B::Device>) -> Vec<B::Device> {
        for module in self.iter() {
            devices = module.collect_devices(devices);
        }

        devices
    }
}

impl<T: ModuleDisplay> ModuleDisplayDefault for Vec<T> {
    fn content(&self, content: Content) -> Option<Content> {
        self.iter()
            .enumerate()
            .fold(content, |acc, (i, module)| {
                let index = format!("{}", i);
                acc.add(&index, module)
            })
            .set_top_level_type(format!("Vec<0..{}>", self.len()).as_str())
            .optional()
    }
}

impl<T: ModuleDisplay> ModuleDisplay for Vec<T> {}

impl<T, B> AutodiffModule<B> for Vec<T>
where
    T: AutodiffModule<B> + Debug + Send + Clone,
    B: AutodiffBackend,
{
    type InnerModule = Vec<T::InnerModule>;

    fn valid(&self) -> Self::InnerModule {
        self.iter().map(|module| module.valid()).collect()
    }
}

impl<const N: usize, T, B> Module<B> for [T; N]
where
    T: Module<B> + Debug + Send + Clone + Copy,
    T::Record: Debug,
    B: Backend,
{
    type Record = [T::Record; N];

    fn collect_devices(&self, mut devices: Vec<B::Device>) -> Vec<B::Device> {
        for module in self.iter() {
            devices = module.collect_devices(devices);
        }

        devices
    }

    fn num_params(&self) -> usize {
        let mut num_params = 0;
        for module in self.iter() {
            num_params += module.num_params();
        }

        num_params
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.iter().for_each(|module| {
            module.visit(visitor);
        });
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        self.map(|module| module.map(mapper))
    }

    fn load_record(self, record: Self::Record) -> Self {
        self.into_iter()
            .zip(record)
            .map(|(module, record)| module.load_record(record))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn into_record(self) -> Self::Record {
        self.map(Module::into_record)
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        self.map(|module| module.to_device(device))
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        self.map(|module| module.fork(device))
    }
}

impl<const N: usize, T: ModuleDisplay> ModuleDisplayDefault for [T; N] {
    fn content(&self, content: Content) -> Option<Content> {
        self.iter()
            .enumerate()
            .fold(content, |acc, (i, module)| {
                let index = format!("{}", i);
                acc.add(&index, module)
            })
            .set_top_level_type(format!("[0..{}]", self.len()).as_str())
            .optional()
    }
}

impl<const N: usize, T: ModuleDisplay> ModuleDisplay for [T; N] {}

impl<const N: usize, T, B> AutodiffModule<B> for [T; N]
where
    T: AutodiffModule<B> + Debug + Send + Clone + Copy,
    T::InnerModule: Copy + Debug,
    <T::InnerModule as Module<B::InnerBackend>>::Record: Debug,
    <T as Module<B>>::Record: Debug,
    B: AutodiffBackend,
{
    type InnerModule = [T::InnerModule; N];

    fn valid(&self) -> Self::InnerModule {
        self.map(|module| module.valid())
    }
}

/// A macro for generating implementations for tuple modules of different sizes.
/// For example: `impl_module_tuple!([L0, L1][0, 1])`.
/// Would generate an implementation for a tuple of size 2.
/// For this macro to work properly, please adhear to the convention:
/// `impl_module_tuple!([L0, L1, ..., Ln][0, 1, ..., n])`.
macro_rules! impl_module_tuple {
    // `$l` represents the generic modules.
    // `$i` represents the indices of the modules in the tuple.
    ([$($l:ident),*][$($i:tt),*]) => {
        impl<B, $($l,)*> Module<B> for ($($l,)*)
        where
            B: Backend,
            $($l: Module<B> + Debug + Send + Clone,)*
        {
            type Record = ($($l::Record),*);

            fn collect_devices(&self, mut devices: Vec<B::Device>) -> Vec<B::Device> {
                $(devices = self.$i.collect_devices(devices);)*
                devices
            }

            fn fork(self, device: &<B as Backend>::Device) -> Self {
                ($(self.$i.fork(device),)*)
            }

            fn to_device(self, device: &<B as Backend>::Device) -> Self {
                ($(self.$i.to_device(device),)*)
            }

            fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
                $(self.$i.visit(visitor);)*
            }

            fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
                ($(self.$i.map(mapper),)*)
            }

            fn load_record(self, record: Self::Record) -> Self {
                ($(self.$i.load_record(record.$i),)*)
            }

            fn into_record(self) -> Self::Record {
                ($(self.$i.into_record(),)*)
            }
        }

        impl<B, $($l,)*> AutodiffModule<B> for ($($l,)*)
        where
            B: AutodiffBackend,
            $($l: AutodiffModule<B> + Debug + Send + Clone,)*
        {
            type InnerModule = ($($l::InnerModule,)*);

            fn valid(&self) -> Self::InnerModule {
                ($(self.$i.valid(),)*)
            }
        }

        impl<$($l,)*> ModuleDisplayDefault for ($($l,)*)
        where
            $($l: ModuleDisplay,)*
        {
            fn content(&self, content: Content) -> Option<Content> {
                let content = content
                    $(.add(&format!("{}", $i), &self.$i))*
                    .set_top_level_type(format!("({})", stringify!($($l),*)).as_str());
                content.optional()
            }
        }

        impl<$($l,)*> ModuleDisplay for ($($l,)*) where $($l: ModuleDisplay,)* {}

    };
}

impl_module_tuple!([L0, L1][0, 1]);
impl_module_tuple!([L0, L1, L2][0, 1, 2]);
impl_module_tuple!([L0, L1, L2, L3][0, 1, 2, 3]);
impl_module_tuple!([L0, L1, L2, L3, L4][0, 1, 2, 3, 4]);
impl_module_tuple!([L0, L1, L2, L3, L4, L5][0, 1, 2, 3, 4, 5]);
impl_module_tuple!([L0, L1, L2, L3, L4, L5, L6][0, 1, 2, 3, 4, 5, 6]);
impl_module_tuple!([L0, L1, L2, L3, L4, L5, L6, L7][0, 1, 2, 3, 4, 5, 6, 7]);
impl_module_tuple!([L0, L1, L2, L3, L4, L5, L6, L7, L8][0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_module_tuple!([L0, L1, L2, L3, L4, L5, L6, L7, L8, L9][0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn dont_override_constant_module_when_loading_record() {
        let module = Some(42);

        let record = Module::<TestBackend>::into_record(module);
        let loaded = Module::<TestBackend>::load_record(module, record);

        assert_eq!(loaded, module);
    }
    #[test]
    fn dont_override_constant_module_when_loading_none_record() {
        let module = Some(42);

        let record = None;
        let loaded = Module::<TestBackend>::load_record(module, record);

        assert_eq!(loaded, module);
    }
}
