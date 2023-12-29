#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
mod batch_norm {
    use burn::{
        module::Module, nn::{BatchNorm, BatchNormConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        norm1: BatchNorm<B, 2>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                norm1: burn::module::Module::<B>::load_record(self.norm1, record.norm1),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                norm1: burn::module::Module::<B>::into_record(self.norm1),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.norm1);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.norm1, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let norm1 = burn::module::Module::<B>::map(self.norm1, mapper);
            Self { norm1 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.norm1, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let norm1 = burn::module::Module::<B>::to_device(self.norm1, device);
            Self { norm1 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let norm1 = burn::module::Module::<B>::fork(self.norm1, device);
            Self { norm1 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let norm1 = burn::module::AutodiffModule::<B>::valid(&self.norm1);
            Self::InnerModule { norm1 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let norm1 = self.norm1.clone();
            Self { norm1 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub norm1: <BatchNorm<B, 2> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < BatchNorm < B, 2 > as burn :: module :: Module < B > > :: Record as burn\n:: record :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub norm1: <<BatchNorm<
            B,
            2,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecordItem",
                "norm1",
                &&self.norm1,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                norm1: ::core::clone::Clone::clone(&self.norm1),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<BatchNorm<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "norm1",
                    &self.norm1,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<BatchNorm<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "norm1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"norm1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<BatchNorm<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<BatchNorm<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<BatchNorm<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { norm1: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<BatchNorm<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("norm1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<BatchNorm<
                                                B,
                                                2,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("norm1")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { norm1: __field0 })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["norm1"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                norm1: burn::record::Record::into_item::<S>(self.norm1),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                norm1: burn::record::Record::from_item::<S>(item.norm1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecord",
                "norm1",
                &&self.norm1,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                norm1: ::core::clone::Clone::clone(&self.norm1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Net",
                "norm1",
                &&self.norm1,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let norm1 = BatchNormConfig::new(4).init_with(record.norm1);
            Self { norm1 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            self.norm1.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "batch_norm::tests::batch_norm2d"]
        pub const batch_norm2d: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("batch_norm::tests::batch_norm2d"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/batch_norm/mod.rs",
                start_line: 36usize,
                start_col: 8usize,
                end_line: 36usize,
                end_col: 20usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(batch_norm2d())),
        };
        fn batch_norm2d() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/batch_norm2d");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<Backend, 4>::ones([1, 5, 2, 2]) - 0.3;
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                    [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                    [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                    [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                    [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 5);
        }
    }
}
mod boolean {
    use burn::{
        module::{Module, Param},
        tensor::{backend::Backend, Bool, Tensor},
    };
    struct Net<B: Backend> {
        buffer: Param<Tensor<B, 1, Bool>>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                buffer: burn::module::Module::<
                    B,
                >::load_record(self.buffer, record.buffer),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                buffer: burn::module::Module::<B>::into_record(self.buffer),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.buffer);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.buffer, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let buffer = burn::module::Module::<B>::map(self.buffer, mapper);
            Self { buffer }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.buffer, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let buffer = burn::module::Module::<B>::to_device(self.buffer, device);
            Self { buffer }
        }
        fn fork(self, device: &B::Device) -> Self {
            let buffer = burn::module::Module::<B>::fork(self.buffer, device);
            Self { buffer }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let buffer = burn::module::AutodiffModule::<B>::valid(&self.buffer);
            Self::InnerModule { buffer }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let buffer = self.buffer.clone();
            Self { buffer }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub buffer: <Param<Tensor<B, 1, Bool>> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Param < Tensor < B, 1, Bool > > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub buffer: <<Param<
            Tensor<B, 1, Bool>,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecordItem",
                "buffer",
                &&self.buffer,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                buffer: ::core::clone::Clone::clone(&self.buffer),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Param<
                Tensor<B, 1, Bool>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "buffer",
                    &self.buffer,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Param<
                Tensor<B, 1, Bool>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "buffer" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"buffer" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Param<
                        Tensor<B, 1, Bool>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Param<
                        Tensor<B, 1, Bool>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Param<
                                Tensor<B, 1, Bool>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { buffer: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Param<
                                Tensor<B, 1, Bool>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("buffer"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Param<
                                                Tensor<B, 1, Bool>,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("buffer")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { buffer: __field0 })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["buffer"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                buffer: burn::record::Record::into_item::<S>(self.buffer),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                buffer: burn::record::Record::from_item::<S>(item.buffer),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecord",
                "buffer",
                &&self.buffer,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                buffer: ::core::clone::Clone::clone(&self.buffer),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Net",
                "buffer",
                &&self.buffer,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            Self { buffer: record.buffer }
        }
        /// Forward pass of the model.
        pub fn forward(&self, _x: Tensor<B, 2>) -> Tensor<B, 1, Bool> {
            self.buffer.val()
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::{
            record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
            tensor::Data,
        };
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "boolean::tests::boolean"]
        pub const boolean: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("boolean::tests::boolean"),
                ignore: true,
                ignore_message: ::core::option::Option::Some(
                    "It appears loading boolean tensors are not supported yet",
                ),
                source_file: "burn-import/pytorch-tests/tests/boolean/mod.rs",
                start_line: 41usize,
                start_col: 8usize,
                end_line: 41usize,
                end_col: 15usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(boolean())),
        };
        #[ignore = "It appears loading boolean tensors are not supported yet"]
        fn boolean() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/integer");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<Backend, 2>::ones([3, 3]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                1,
                Bool,
            >::from_bool(Data::from([true, false, true]));
            match (&output.to_data(), &expected.to_data()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        }
    }
}
mod buffer {
    use burn::{
        module::{Module, Param},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        buffer: Param<Tensor<B, 2>>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                buffer: burn::module::Module::<
                    B,
                >::load_record(self.buffer, record.buffer),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                buffer: burn::module::Module::<B>::into_record(self.buffer),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.buffer);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.buffer, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let buffer = burn::module::Module::<B>::map(self.buffer, mapper);
            Self { buffer }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.buffer, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let buffer = burn::module::Module::<B>::to_device(self.buffer, device);
            Self { buffer }
        }
        fn fork(self, device: &B::Device) -> Self {
            let buffer = burn::module::Module::<B>::fork(self.buffer, device);
            Self { buffer }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let buffer = burn::module::AutodiffModule::<B>::valid(&self.buffer);
            Self::InnerModule { buffer }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let buffer = self.buffer.clone();
            Self { buffer }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub buffer: <Param<Tensor<B, 2>> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Param < Tensor < B, 2 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub buffer: <<Param<
            Tensor<B, 2>,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecordItem",
                "buffer",
                &&self.buffer,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                buffer: ::core::clone::Clone::clone(&self.buffer),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Param<
                Tensor<B, 2>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "buffer",
                    &self.buffer,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Param<
                Tensor<B, 2>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "buffer" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"buffer" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Param<
                        Tensor<B, 2>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Param<
                        Tensor<B, 2>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Param<
                                Tensor<B, 2>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { buffer: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Param<
                                Tensor<B, 2>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("buffer"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Param<
                                                Tensor<B, 2>,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("buffer")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { buffer: __field0 })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["buffer"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                buffer: burn::record::Record::into_item::<S>(self.buffer),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                buffer: burn::record::Record::from_item::<S>(item.buffer),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecord",
                "buffer",
                &&self.buffer,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                buffer: ::core::clone::Clone::clone(&self.buffer),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Net",
                "buffer",
                &&self.buffer,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            Self { buffer: record.buffer }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            self.buffer.val() + x
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "buffer::tests::buffer"]
        pub const buffer: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("buffer::tests::buffer"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/buffer/mod.rs",
                start_line: 36usize,
                start_col: 8usize,
                end_line: 36usize,
                end_col: 14usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(buffer())),
        };
        fn buffer() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/buffer");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<Backend, 2>::ones([3, 3]);
            let output = model.forward(input);
            let expected = Tensor::<Backend, 2>::ones([3, 3]) * 2.0;
            output.to_data().assert_approx_eq(&expected.to_data(), 3);
        }
    }
}
mod complex_nested {
    use std::{env, path::Path};
    use burn::record::{
        FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder,
        NamedMpkGzFileRecorder, PrettyJsonFileRecorder, Recorder,
    };
    use burn::{
        module::Module,
        nn::{
            conv::{Conv2d, Conv2dConfig},
            BatchNorm, BatchNormConfig, Linear, LinearConfig,
        },
        tensor::{
            activation::{log_softmax, relu},
            backend::Backend, Tensor,
        },
    };
    struct ConvBlock<B: Backend> {
        conv: Conv2d<B>,
        norm: BatchNorm<B, 2>,
    }
    impl<B: Backend> burn::module::Module<B> for ConvBlock<B> {
        type Record = ConvBlockRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                conv: burn::module::Module::<B>::load_record(self.conv, record.conv),
                norm: burn::module::Module::<B>::load_record(self.norm, record.norm),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                conv: burn::module::Module::<B>::into_record(self.conv),
                norm: burn::module::Module::<B>::into_record(self.norm),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.conv);
            num_params += burn::module::Module::<B>::num_params(&self.norm);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.conv, visitor);
            burn::module::Module::visit(&self.norm, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let conv = burn::module::Module::<B>::map(self.conv, mapper);
            let norm = burn::module::Module::<B>::map(self.norm, mapper);
            Self { conv, norm }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.norm, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let conv = burn::module::Module::<B>::to_device(self.conv, device);
            let norm = burn::module::Module::<B>::to_device(self.norm, device);
            Self { conv, norm }
        }
        fn fork(self, device: &B::Device) -> Self {
            let conv = burn::module::Module::<B>::fork(self.conv, device);
            let norm = burn::module::Module::<B>::fork(self.norm, device);
            Self { conv, norm }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for ConvBlock<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = ConvBlock<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let conv = burn::module::AutodiffModule::<B>::valid(&self.conv);
            let norm = burn::module::AutodiffModule::<B>::valid(&self.norm);
            Self::InnerModule { conv, norm }
        }
    }
    impl<B: Backend> core::fmt::Display for ConvBlock<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(
                format_args!("{0}[num_params={1}]", "ConvBlock", self.num_params()),
            )
        }
    }
    impl<B: Backend> Clone for ConvBlock<B> {
        fn clone(&self) -> Self {
            let conv = self.conv.clone();
            let norm = self.norm.clone();
            Self { conv, norm }
        }
    }
    /// The record type for the module.
    pub struct ConvBlockRecord<B: Backend> {
        /// The module record associative type.
        pub conv: <Conv2d<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub norm: <BatchNorm<B, 2> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Conv2d < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < BatchNorm < B, 2 > as burn :: module :: Module < B > >\n:: Record as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct ConvBlockRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub conv: <<Conv2d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub norm: <<BatchNorm<
            B,
            2,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for ConvBlockRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "ConvBlockRecordItem",
                "conv",
                &self.conv,
                "norm",
                &&self.norm,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for ConvBlockRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> ConvBlockRecordItem<B, S> {
            ConvBlockRecordItem {
                conv: ::core::clone::Clone::clone(&self.conv),
                norm: ::core::clone::Clone::clone(&self.norm),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for ConvBlockRecordItem<B, S>
        where
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<BatchNorm<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "ConvBlockRecordItem",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv",
                    &self.conv,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "norm",
                    &self.norm,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for ConvBlockRecordItem<B, S>
        where
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<BatchNorm<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "conv" => _serde::__private::Ok(__Field::__field0),
                            "norm" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"conv" => _serde::__private::Ok(__Field::__field0),
                            b"norm" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<BatchNorm<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<ConvBlockRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<BatchNorm<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = ConvBlockRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct ConvBlockRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct ConvBlockRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<BatchNorm<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct ConvBlockRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(ConvBlockRecordItem {
                            conv: __field0,
                            norm: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<BatchNorm<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Conv2d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("norm"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<BatchNorm<
                                                B,
                                                2,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("norm")?
                            }
                        };
                        _serde::__private::Ok(ConvBlockRecordItem {
                            conv: __field0,
                            norm: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["conv", "norm"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "ConvBlockRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<
                            ConvBlockRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for ConvBlockRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = ConvBlockRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            ConvBlockRecordItem {
                conv: burn::record::Record::into_item::<S>(self.conv),
                norm: burn::record::Record::into_item::<S>(self.norm),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                conv: burn::record::Record::from_item::<S>(item.conv),
                norm: burn::record::Record::from_item::<S>(item.norm),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for ConvBlockRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "ConvBlockRecord",
                "conv",
                &self.conv,
                "norm",
                &&self.norm,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for ConvBlockRecord<B> {
        #[inline]
        fn clone(&self) -> ConvBlockRecord<B> {
            ConvBlockRecord {
                conv: ::core::clone::Clone::clone(&self.conv),
                norm: ::core::clone::Clone::clone(&self.norm),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for ConvBlock<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "ConvBlock",
                "conv",
                &self.conv,
                "norm",
                &&self.norm,
            )
        }
    }
    struct Net<B: Backend> {
        conv_blocks: Vec<ConvBlock<B>>,
        norm1: BatchNorm<B, 2>,
        fc1: Linear<B>,
        fc2: Linear<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                conv_blocks: burn::module::Module::<
                    B,
                >::load_record(self.conv_blocks, record.conv_blocks),
                norm1: burn::module::Module::<B>::load_record(self.norm1, record.norm1),
                fc1: burn::module::Module::<B>::load_record(self.fc1, record.fc1),
                fc2: burn::module::Module::<B>::load_record(self.fc2, record.fc2),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                conv_blocks: burn::module::Module::<B>::into_record(self.conv_blocks),
                norm1: burn::module::Module::<B>::into_record(self.norm1),
                fc1: burn::module::Module::<B>::into_record(self.fc1),
                fc2: burn::module::Module::<B>::into_record(self.fc2),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.conv_blocks);
            num_params += burn::module::Module::<B>::num_params(&self.norm1);
            num_params += burn::module::Module::<B>::num_params(&self.fc1);
            num_params += burn::module::Module::<B>::num_params(&self.fc2);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.conv_blocks, visitor);
            burn::module::Module::visit(&self.norm1, visitor);
            burn::module::Module::visit(&self.fc1, visitor);
            burn::module::Module::visit(&self.fc2, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let conv_blocks = burn::module::Module::<B>::map(self.conv_blocks, mapper);
            let norm1 = burn::module::Module::<B>::map(self.norm1, mapper);
            let fc1 = burn::module::Module::<B>::map(self.fc1, mapper);
            let fc2 = burn::module::Module::<B>::map(self.fc2, mapper);
            Self {
                conv_blocks,
                norm1,
                fc1,
                fc2,
            }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv_blocks, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.norm1, devices);
            let devices = burn::module::Module::<B>::collect_devices(&self.fc1, devices);
            let devices = burn::module::Module::<B>::collect_devices(&self.fc2, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let conv_blocks = burn::module::Module::<
                B,
            >::to_device(self.conv_blocks, device);
            let norm1 = burn::module::Module::<B>::to_device(self.norm1, device);
            let fc1 = burn::module::Module::<B>::to_device(self.fc1, device);
            let fc2 = burn::module::Module::<B>::to_device(self.fc2, device);
            Self {
                conv_blocks,
                norm1,
                fc1,
                fc2,
            }
        }
        fn fork(self, device: &B::Device) -> Self {
            let conv_blocks = burn::module::Module::<B>::fork(self.conv_blocks, device);
            let norm1 = burn::module::Module::<B>::fork(self.norm1, device);
            let fc1 = burn::module::Module::<B>::fork(self.fc1, device);
            let fc2 = burn::module::Module::<B>::fork(self.fc2, device);
            Self {
                conv_blocks,
                norm1,
                fc1,
                fc2,
            }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let conv_blocks = burn::module::AutodiffModule::<
                B,
            >::valid(&self.conv_blocks);
            let norm1 = burn::module::AutodiffModule::<B>::valid(&self.norm1);
            let fc1 = burn::module::AutodiffModule::<B>::valid(&self.fc1);
            let fc2 = burn::module::AutodiffModule::<B>::valid(&self.fc2);
            Self::InnerModule {
                conv_blocks,
                norm1,
                fc1,
                fc2,
            }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let conv_blocks = self.conv_blocks.clone();
            let norm1 = self.norm1.clone();
            let fc1 = self.fc1.clone();
            let fc2 = self.fc2.clone();
            Self {
                conv_blocks,
                norm1,
                fc1,
                fc2,
            }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub conv_blocks: <Vec<ConvBlock<B>> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub norm1: <BatchNorm<B, 2> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub fc1: <Linear<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub fc2: <Linear<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Vec < ConvBlock < B > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < BatchNorm < B, 2 > as burn :: module :: Module < B > >\n:: Record as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned, < < Linear < B > as burn :: module :: Module\n< B > > :: Record as burn :: record :: Record > :: Item < S > : serde ::\nSerialize + serde :: de :: DeserializeOwned, < < Linear < B > as burn ::\nmodule :: Module < B > > :: Record as burn :: record :: Record > :: Item < S >\n: serde :: Serialize + serde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub conv_blocks: <<Vec<
            ConvBlock<B>,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub norm1: <<BatchNorm<
            B,
            2,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub fc1: <<Linear<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub fc2: <<Linear<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field4_finish(
                f,
                "NetRecordItem",
                "conv_blocks",
                &self.conv_blocks,
                "norm1",
                &self.norm1,
                "fc1",
                &self.fc1,
                "fc2",
                &&self.fc2,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                conv_blocks: ::core::clone::Clone::clone(&self.conv_blocks),
                norm1: ::core::clone::Clone::clone(&self.norm1),
                fc1: ::core::clone::Clone::clone(&self.fc1),
                fc2: ::core::clone::Clone::clone(&self.fc2),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Vec<
                ConvBlock<B>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<BatchNorm<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1 + 1 + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv_blocks",
                    &self.conv_blocks,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "norm1",
                    &self.norm1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "fc1",
                    &self.fc1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "fc2",
                    &self.fc2,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Vec<
                ConvBlock<B>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<BatchNorm<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __field2,
                    __field3,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            2u64 => _serde::__private::Ok(__Field::__field2),
                            3u64 => _serde::__private::Ok(__Field::__field3),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "conv_blocks" => _serde::__private::Ok(__Field::__field0),
                            "norm1" => _serde::__private::Ok(__Field::__field1),
                            "fc1" => _serde::__private::Ok(__Field::__field2),
                            "fc2" => _serde::__private::Ok(__Field::__field3),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"conv_blocks" => _serde::__private::Ok(__Field::__field0),
                            b"norm1" => _serde::__private::Ok(__Field::__field1),
                            b"fc1" => _serde::__private::Ok(__Field::__field2),
                            b"fc2" => _serde::__private::Ok(__Field::__field3),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Vec<
                        ConvBlock<B>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<BatchNorm<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Vec<
                        ConvBlock<B>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<BatchNorm<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Vec<
                                ConvBlock<B>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 4 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<BatchNorm<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct NetRecordItem with 4 elements",
                                    ),
                                );
                            }
                        };
                        let __field2 = match _serde::de::SeqAccess::next_element::<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        2usize,
                                        &"struct NetRecordItem with 4 elements",
                                    ),
                                );
                            }
                        };
                        let __field3 = match _serde::de::SeqAccess::next_element::<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        3usize,
                                        &"struct NetRecordItem with 4 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv_blocks: __field0,
                            norm1: __field1,
                            fc1: __field2,
                            fc2: __field3,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Vec<
                                ConvBlock<B>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<BatchNorm<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field2: _serde::__private::Option<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field3: _serde::__private::Option<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "conv_blocks",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Vec<
                                                ConvBlock<B>,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("norm1"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<BatchNorm<
                                                B,
                                                2,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field2 => {
                                    if _serde::__private::Option::is_some(&__field2) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("fc1"),
                                        );
                                    }
                                    __field2 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Linear<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field3 => {
                                    if _serde::__private::Option::is_some(&__field3) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("fc2"),
                                        );
                                    }
                                    __field3 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Linear<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv_blocks")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("norm1")?
                            }
                        };
                        let __field2 = match __field2 {
                            _serde::__private::Some(__field2) => __field2,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("fc1")?
                            }
                        };
                        let __field3 = match __field3 {
                            _serde::__private::Some(__field3) => __field3,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("fc2")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv_blocks: __field0,
                            norm1: __field1,
                            fc1: __field2,
                            fc2: __field3,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &[
                    "conv_blocks",
                    "norm1",
                    "fc1",
                    "fc2",
                ];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                conv_blocks: burn::record::Record::into_item::<S>(self.conv_blocks),
                norm1: burn::record::Record::into_item::<S>(self.norm1),
                fc1: burn::record::Record::into_item::<S>(self.fc1),
                fc2: burn::record::Record::into_item::<S>(self.fc2),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                conv_blocks: burn::record::Record::from_item::<S>(item.conv_blocks),
                norm1: burn::record::Record::from_item::<S>(item.norm1),
                fc1: burn::record::Record::from_item::<S>(item.fc1),
                fc2: burn::record::Record::from_item::<S>(item.fc2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field4_finish(
                f,
                "NetRecord",
                "conv_blocks",
                &self.conv_blocks,
                "norm1",
                &self.norm1,
                "fc1",
                &self.fc1,
                "fc2",
                &&self.fc2,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                conv_blocks: ::core::clone::Clone::clone(&self.conv_blocks),
                norm1: ::core::clone::Clone::clone(&self.norm1),
                fc1: ::core::clone::Clone::clone(&self.fc1),
                fc2: ::core::clone::Clone::clone(&self.fc2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field4_finish(
                f,
                "Net",
                "conv_blocks",
                &self.conv_blocks,
                "norm1",
                &self.norm1,
                "fc1",
                &self.fc1,
                "fc2",
                &&self.fc2,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let conv_blocks = <[_]>::into_vec(
                #[rustc_box]
                ::alloc::boxed::Box::new([
                    ConvBlock {
                        conv: Conv2dConfig::new([2, 4], [3, 2])
                            .init_with(record.conv_blocks[0].conv.clone()),
                        norm: BatchNormConfig::new(2)
                            .init_with(record.conv_blocks[0].norm.clone()),
                    },
                    ConvBlock {
                        conv: Conv2dConfig::new([4, 6], [3, 2])
                            .init_with(record.conv_blocks[1].conv.clone()),
                        norm: BatchNormConfig::new(4)
                            .init_with(record.conv_blocks[1].norm.clone()),
                    },
                ]),
            );
            let norm1 = BatchNormConfig::new(6).init_with(record.norm1);
            let fc1 = LinearConfig::new(120, 12).init_with(record.fc1);
            let fc2 = LinearConfig::new(12, 10).init_with(record.fc2);
            Self {
                conv_blocks,
                norm1,
                fc1,
                fc2,
            }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
            let x = self.conv_blocks[0].forward(x);
            let x = self.conv_blocks[1].forward(x);
            let x = self.norm1.forward(x);
            let x = x.reshape([0, -1]);
            let x = self.fc1.forward(x);
            let x = relu(x);
            let x = self.fc2.forward(x);
            log_softmax(x, 1)
        }
    }
    impl<B: Backend> ConvBlock<B> {
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            let x = self.conv.forward(x);
            self.norm.forward(x)
        }
    }
    type TestBackend = burn_ndarray::NdArray<f32>;
    fn model_test(record: NetRecord<TestBackend>, precision: usize) {
        let model = Net::<TestBackend>::new_with(record);
        let input = Tensor::<TestBackend, 4>::ones([1, 2, 9, 6]) - 0.5;
        let output = model.forward(input);
        let expected = Tensor::<
            TestBackend,
            2,
        >::from_data([
            [
                -2.306_613,
                -2.058_945_4,
                -2.298_372_7,
                -2.358_294,
                -2.296_395_5,
                -2.416_090_5,
                -2.107_669,
                -2.428_420_8,
                -2.526_469,
                -2.319_918_6,
            ],
        ]);
        output.to_data().assert_approx_eq(&expected.to_data(), precision);
    }
    extern crate test;
    #[cfg(test)]
    #[rustc_test_marker = "complex_nested::json_full_record"]
    pub const json_full_record: test::TestDescAndFn = test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("complex_nested::json_full_record"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "burn-import/pytorch-tests/tests/complex_nested/mod.rs",
            start_line: 111usize,
            start_col: 4usize,
            end_line: 111usize,
            end_col: 20usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::IntegrationTest,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(json_full_record())),
    };
    fn json_full_record() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/full/complex_nested");
        let record = PrettyJsonFileRecorder::<FullPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");
        model_test(record, 8);
    }
    extern crate test;
    #[cfg(test)]
    #[rustc_test_marker = "complex_nested::json_half_record"]
    pub const json_half_record: test::TestDescAndFn = test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("complex_nested::json_half_record"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "burn-import/pytorch-tests/tests/complex_nested/mod.rs",
            start_line: 123usize,
            start_col: 4usize,
            end_line: 123usize,
            end_col: 20usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::IntegrationTest,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(json_half_record())),
    };
    fn json_half_record() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/half/complex_nested");
        let record = PrettyJsonFileRecorder::<HalfPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");
        model_test(record, 4);
    }
    extern crate test;
    #[cfg(test)]
    #[rustc_test_marker = "complex_nested::mpk_full_record"]
    pub const mpk_full_record: test::TestDescAndFn = test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("complex_nested::mpk_full_record"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "burn-import/pytorch-tests/tests/complex_nested/mod.rs",
            start_line: 135usize,
            start_col: 4usize,
            end_line: 135usize,
            end_col: 19usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::IntegrationTest,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(mpk_full_record())),
    };
    fn mpk_full_record() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/full/complex_nested");
        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");
        model_test(record, 8);
    }
    extern crate test;
    #[cfg(test)]
    #[rustc_test_marker = "complex_nested::mpk_half_record"]
    pub const mpk_half_record: test::TestDescAndFn = test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("complex_nested::mpk_half_record"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "burn-import/pytorch-tests/tests/complex_nested/mod.rs",
            start_line: 147usize,
            start_col: 4usize,
            end_line: 147usize,
            end_col: 19usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::IntegrationTest,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(mpk_half_record())),
    };
    fn mpk_half_record() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/half/complex_nested");
        let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");
        model_test(record, 4);
    }
    extern crate test;
    #[cfg(test)]
    #[rustc_test_marker = "complex_nested::mpk_gz_full_record"]
    pub const mpk_gz_full_record: test::TestDescAndFn = test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("complex_nested::mpk_gz_full_record"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "burn-import/pytorch-tests/tests/complex_nested/mod.rs",
            start_line: 159usize,
            start_col: 4usize,
            end_line: 159usize,
            end_col: 22usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::IntegrationTest,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(mpk_gz_full_record())),
    };
    fn mpk_gz_full_record() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/full/complex_nested");
        let record = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");
        model_test(record, 8);
    }
    extern crate test;
    #[cfg(test)]
    #[rustc_test_marker = "complex_nested::mpk_gz_half_record"]
    pub const mpk_gz_half_record: test::TestDescAndFn = test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("complex_nested::mpk_gz_half_record"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "burn-import/pytorch-tests/tests/complex_nested/mod.rs",
            start_line: 171usize,
            start_col: 4usize,
            end_line: 171usize,
            end_col: 22usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::IntegrationTest,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(mpk_gz_half_record())),
    };
    fn mpk_gz_half_record() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/half/complex_nested");
        let record = NamedMpkGzFileRecorder::<HalfPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");
        model_test(record, 4);
    }
}
mod conv1d {
    use burn::{
        module::Module, nn::conv::{Conv1d, Conv1dConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        conv1: Conv1d<B>,
        conv2: Conv1d<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                conv1: burn::module::Module::<B>::load_record(self.conv1, record.conv1),
                conv2: burn::module::Module::<B>::load_record(self.conv2, record.conv2),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                conv1: burn::module::Module::<B>::into_record(self.conv1),
                conv2: burn::module::Module::<B>::into_record(self.conv2),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.conv1);
            num_params += burn::module::Module::<B>::num_params(&self.conv2);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.conv1, visitor);
            burn::module::Module::visit(&self.conv2, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let conv1 = burn::module::Module::<B>::map(self.conv1, mapper);
            let conv2 = burn::module::Module::<B>::map(self.conv2, mapper);
            Self { conv1, conv2 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv1, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv2, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::to_device(self.conv1, device);
            let conv2 = burn::module::Module::<B>::to_device(self.conv2, device);
            Self { conv1, conv2 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::fork(self.conv1, device);
            let conv2 = burn::module::Module::<B>::fork(self.conv2, device);
            Self { conv1, conv2 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let conv1 = burn::module::AutodiffModule::<B>::valid(&self.conv1);
            let conv2 = burn::module::AutodiffModule::<B>::valid(&self.conv2);
            Self::InnerModule { conv1, conv2 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let conv1 = self.conv1.clone();
            let conv2 = self.conv2.clone();
            Self { conv1, conv2 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub conv1: <Conv1d<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub conv2: <Conv1d<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Conv1d < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < Conv1d < B > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub conv1: <<Conv1d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub conv2: <<Conv1d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecordItem",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Conv1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Conv1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv1",
                    &self.conv1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv2",
                    &self.conv2,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Conv1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Conv1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "conv1" => _serde::__private::Ok(__Field::__field0),
                            "conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"conv1" => _serde::__private::Ok(__Field::__field0),
                            b"conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Conv1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Conv1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Conv1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Conv1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Conv1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<Conv1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Conv1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<Conv1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Conv1d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv2"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Conv1d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv1")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv2")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["conv1", "conv2"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                conv1: burn::record::Record::into_item::<S>(self.conv1),
                conv2: burn::record::Record::into_item::<S>(self.conv2),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                conv1: burn::record::Record::from_item::<S>(item.conv1),
                conv2: burn::record::Record::from_item::<S>(item.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecord",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "Net",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let conv1 = Conv1dConfig::new(2, 2, 2).init_with(record.conv1);
            let conv2 = Conv1dConfig::new(2, 2, 2)
                .with_bias(false)
                .init_with(record.conv2);
            Self { conv1, conv2 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
            let x = self.conv1.forward(x);
            self.conv2.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "conv1d::tests::conv1d"]
        pub const conv1d: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("conv1d::tests::conv1d"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/conv1d/mod.rs",
                start_line: 42usize,
                start_col: 8usize,
                end_line: 42usize,
                end_col: 14usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(conv1d())),
        };
        fn conv1d() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/conv1d");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                3,
            >::from_data([
                [
                    [
                        0.93708336,
                        0.65559506,
                        0.31379688,
                        0.19801933,
                        0.41619217,
                        0.28432965,
                    ],
                    [
                        0.33977574,
                        0.523_940_8,
                        0.798_063_9,
                        0.77176833,
                        0.01122457,
                        0.80996025,
                    ],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                3,
            >::from_data([
                [
                    [0.02987457, 0.03134188, 0.04234261, -0.02437721],
                    [-0.03788019, -0.02972012, -0.00806090, -0.01981254],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 7);
        }
    }
}
mod conv2d {
    use burn::{
        module::Module, nn::conv::{Conv2d, Conv2dConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        conv1: Conv2d<B>,
        conv2: Conv2d<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                conv1: burn::module::Module::<B>::load_record(self.conv1, record.conv1),
                conv2: burn::module::Module::<B>::load_record(self.conv2, record.conv2),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                conv1: burn::module::Module::<B>::into_record(self.conv1),
                conv2: burn::module::Module::<B>::into_record(self.conv2),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.conv1);
            num_params += burn::module::Module::<B>::num_params(&self.conv2);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.conv1, visitor);
            burn::module::Module::visit(&self.conv2, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let conv1 = burn::module::Module::<B>::map(self.conv1, mapper);
            let conv2 = burn::module::Module::<B>::map(self.conv2, mapper);
            Self { conv1, conv2 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv1, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv2, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::to_device(self.conv1, device);
            let conv2 = burn::module::Module::<B>::to_device(self.conv2, device);
            Self { conv1, conv2 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::fork(self.conv1, device);
            let conv2 = burn::module::Module::<B>::fork(self.conv2, device);
            Self { conv1, conv2 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let conv1 = burn::module::AutodiffModule::<B>::valid(&self.conv1);
            let conv2 = burn::module::AutodiffModule::<B>::valid(&self.conv2);
            Self::InnerModule { conv1, conv2 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let conv1 = self.conv1.clone();
            let conv2 = self.conv2.clone();
            Self { conv1, conv2 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub conv1: <Conv2d<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub conv2: <Conv2d<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Conv2d < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < Conv2d < B > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub conv1: <<Conv2d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub conv2: <<Conv2d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecordItem",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv1",
                    &self.conv1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv2",
                    &self.conv2,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "conv1" => _serde::__private::Ok(__Field::__field0),
                            "conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"conv1" => _serde::__private::Ok(__Field::__field0),
                            b"conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Conv2d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv2"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Conv2d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv1")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv2")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["conv1", "conv2"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                conv1: burn::record::Record::into_item::<S>(self.conv1),
                conv2: burn::record::Record::into_item::<S>(self.conv2),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                conv1: burn::record::Record::from_item::<S>(item.conv1),
                conv2: burn::record::Record::from_item::<S>(item.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecord",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "Net",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let conv1 = Conv2dConfig::new([2, 2], [2, 2]).init_with(record.conv1);
            let conv2 = Conv2dConfig::new([2, 2], [2, 2])
                .with_bias(false)
                .init_with(record.conv2);
            Self { conv1, conv2 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            let x = self.conv1.forward(x);
            self.conv2.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "conv2d::tests::conv2d"]
        pub const conv2d: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("conv2d::tests::conv2d"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/conv2d/mod.rs",
                start_line: 42usize,
                start_col: 8usize,
                end_line: 42usize,
                end_col: 14usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(conv2d())),
        };
        fn conv2d() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/conv2d");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [
                        [0.024_595_8, 0.25883394, 0.93905586, 0.416_715_5, 0.713_979_7],
                        [0.267_644_3, 0.990_609, 0.28845078, 0.874_962_4, 0.505_920_8],
                        [0.23659128, 0.757_007_4, 0.23458993, 0.64705235, 0.355_621_4],
                        [0.445_182_8, 0.01930594, 0.26160914, 0.771_317, 0.37846136],
                        [0.99802476, 0.900_794_2, 0.476_588_2, 0.16625845, 0.804_481_1],
                    ],
                    [
                        [0.65517855, 0.17679012, 0.824_772_3, 0.803_550_9, 0.943_447_5],
                        [0.21972018, 0.417_697, 0.49031407, 0.57302874, 0.12054086],
                        [0.14518881, 0.772_002_3, 0.38275403, 0.744_236_7, 0.52850497],
                        [0.664_172_4, 0.60994434, 0.681_799_7, 0.74785537, 0.03694397],
                        [0.751_675_7, 0.148_438_4, 0.12274551, 0.530_407_2, 0.414_796_4],
                    ],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [
                        [-0.02502128, 0.00250649, 0.04841233],
                        [0.04589614, -0.00296854, 0.01991477],
                        [0.02920526, 0.059_497_3, 0.04326791],
                    ],
                    [
                        [-0.04825336, 0.080_190_9, -0.02375088],
                        [0.02885434, 0.09638263, -0.07460806],
                        [0.02004079, 0.06244051, 0.035_887_1],
                    ],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 7);
        }
    }
}
mod conv_transpose1d {
    use burn::{
        module::Module, nn::conv::{ConvTranspose1d, ConvTranspose1dConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        conv1: ConvTranspose1d<B>,
        conv2: ConvTranspose1d<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                conv1: burn::module::Module::<B>::load_record(self.conv1, record.conv1),
                conv2: burn::module::Module::<B>::load_record(self.conv2, record.conv2),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                conv1: burn::module::Module::<B>::into_record(self.conv1),
                conv2: burn::module::Module::<B>::into_record(self.conv2),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.conv1);
            num_params += burn::module::Module::<B>::num_params(&self.conv2);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.conv1, visitor);
            burn::module::Module::visit(&self.conv2, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let conv1 = burn::module::Module::<B>::map(self.conv1, mapper);
            let conv2 = burn::module::Module::<B>::map(self.conv2, mapper);
            Self { conv1, conv2 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv1, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv2, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::to_device(self.conv1, device);
            let conv2 = burn::module::Module::<B>::to_device(self.conv2, device);
            Self { conv1, conv2 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::fork(self.conv1, device);
            let conv2 = burn::module::Module::<B>::fork(self.conv2, device);
            Self { conv1, conv2 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let conv1 = burn::module::AutodiffModule::<B>::valid(&self.conv1);
            let conv2 = burn::module::AutodiffModule::<B>::valid(&self.conv2);
            Self::InnerModule { conv1, conv2 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let conv1 = self.conv1.clone();
            let conv2 = self.conv2.clone();
            Self { conv1, conv2 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub conv1: <ConvTranspose1d<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub conv2: <ConvTranspose1d<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < ConvTranspose1d < B > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < ConvTranspose1d < B > as burn :: module :: Module < B >\n> :: Record as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub conv1: <<ConvTranspose1d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub conv2: <<ConvTranspose1d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecordItem",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<ConvTranspose1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<ConvTranspose1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv1",
                    &self.conv1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv2",
                    &self.conv2,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<ConvTranspose1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<ConvTranspose1d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "conv1" => _serde::__private::Ok(__Field::__field0),
                            "conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"conv1" => _serde::__private::Ok(__Field::__field0),
                            b"conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<ConvTranspose1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<ConvTranspose1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<ConvTranspose1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<ConvTranspose1d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<ConvTranspose1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<ConvTranspose1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<ConvTranspose1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<ConvTranspose1d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<ConvTranspose1d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv2"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<ConvTranspose1d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv1")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv2")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["conv1", "conv2"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                conv1: burn::record::Record::into_item::<S>(self.conv1),
                conv2: burn::record::Record::into_item::<S>(self.conv2),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                conv1: burn::record::Record::from_item::<S>(item.conv1),
                conv2: burn::record::Record::from_item::<S>(item.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecord",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "Net",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let conv1 = ConvTranspose1dConfig::new([2, 2], 2).init_with(record.conv1);
            let conv2 = ConvTranspose1dConfig::new([2, 2], 2).init_with(record.conv2);
            Self { conv1, conv2 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
            let x = self.conv1.forward(x);
            self.conv2.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "conv_transpose1d::tests::conv_transpose1d"]
        pub const conv_transpose1d: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("conv_transpose1d::tests::conv_transpose1d"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/conv_transpose1d/mod.rs",
                start_line: 40usize,
                start_col: 8usize,
                end_line: 40usize,
                end_col: 24usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(conv_transpose1d())),
        };
        fn conv_transpose1d() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/conv_transpose1d");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                3,
            >::from_data([[[0.93708336, 0.65559506], [0.31379688, 0.19801933]]]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                3,
            >::from_data([
                [
                    [0.02935525, 0.01119324, -0.01356167, -0.00682688],
                    [0.01644749, -0.01429807, 0.00083987, 0.00279229],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 8);
        }
    }
}
mod conv_transpose2d {
    use burn::{
        module::Module, nn::conv::{ConvTranspose2d, ConvTranspose2dConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        conv1: ConvTranspose2d<B>,
        conv2: ConvTranspose2d<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                conv1: burn::module::Module::<B>::load_record(self.conv1, record.conv1),
                conv2: burn::module::Module::<B>::load_record(self.conv2, record.conv2),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                conv1: burn::module::Module::<B>::into_record(self.conv1),
                conv2: burn::module::Module::<B>::into_record(self.conv2),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.conv1);
            num_params += burn::module::Module::<B>::num_params(&self.conv2);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.conv1, visitor);
            burn::module::Module::visit(&self.conv2, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let conv1 = burn::module::Module::<B>::map(self.conv1, mapper);
            let conv2 = burn::module::Module::<B>::map(self.conv2, mapper);
            Self { conv1, conv2 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv1, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv2, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::to_device(self.conv1, device);
            let conv2 = burn::module::Module::<B>::to_device(self.conv2, device);
            Self { conv1, conv2 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::fork(self.conv1, device);
            let conv2 = burn::module::Module::<B>::fork(self.conv2, device);
            Self { conv1, conv2 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let conv1 = burn::module::AutodiffModule::<B>::valid(&self.conv1);
            let conv2 = burn::module::AutodiffModule::<B>::valid(&self.conv2);
            Self::InnerModule { conv1, conv2 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let conv1 = self.conv1.clone();
            let conv2 = self.conv2.clone();
            Self { conv1, conv2 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub conv1: <ConvTranspose2d<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub conv2: <ConvTranspose2d<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < ConvTranspose2d < B > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < ConvTranspose2d < B > as burn :: module :: Module < B >\n> :: Record as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub conv1: <<ConvTranspose2d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub conv2: <<ConvTranspose2d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecordItem",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<ConvTranspose2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<ConvTranspose2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv1",
                    &self.conv1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv2",
                    &self.conv2,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<ConvTranspose2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<ConvTranspose2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "conv1" => _serde::__private::Ok(__Field::__field0),
                            "conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"conv1" => _serde::__private::Ok(__Field::__field0),
                            b"conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<ConvTranspose2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<ConvTranspose2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<ConvTranspose2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<ConvTranspose2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<ConvTranspose2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<ConvTranspose2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<ConvTranspose2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<ConvTranspose2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<ConvTranspose2d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv2"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<ConvTranspose2d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv1")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv2")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["conv1", "conv2"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                conv1: burn::record::Record::into_item::<S>(self.conv1),
                conv2: burn::record::Record::into_item::<S>(self.conv2),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                conv1: burn::record::Record::from_item::<S>(item.conv1),
                conv2: burn::record::Record::from_item::<S>(item.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecord",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "Net",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let conv1 = ConvTranspose2dConfig::new([2, 2], [2, 2])
                .init_with(record.conv1);
            let conv2 = ConvTranspose2dConfig::new([2, 2], [2, 2])
                .init_with(record.conv2);
            Self { conv1, conv2 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            let x = self.conv1.forward(x);
            self.conv2.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "conv_transpose2d::tests::conv_transpose2d"]
        pub const conv_transpose2d: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("conv_transpose2d::tests::conv_transpose2d"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/conv_transpose2d/mod.rs",
                start_line: 40usize,
                start_col: 8usize,
                end_line: 40usize,
                end_col: 24usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(conv_transpose2d())),
        };
        fn conv_transpose2d() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/conv_transpose2d");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
                    [[0.713_979_7, 0.267_644_3], [0.990_609, 0.28845078]],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [
                        [0.04547675, 0.01879685, -0.01636661, 0.00310803],
                        [0.02090115, 0.01192738, -0.048_240_2, 0.02252235],
                        [0.03249975, -0.00460748, 0.05003899, 0.04029131],
                        [0.02185687, -0.10226749, -0.06508022, -0.01267705],
                    ],
                    [
                        [0.00277598, -0.00513832, -0.059_048_3, 0.00567626],
                        [-0.03149522, -0.195_757_4, 0.03474613, 0.01997269],
                        [-0.10096474, 0.00679589, 0.041_919_7, -0.02464108],
                        [-0.03174751, 0.02963913, -0.02703723, -0.01860938],
                    ],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 7);
        }
    }
}
mod embedding {
    use burn::{
        module::Module, nn::{Embedding, EmbeddingConfig},
        tensor::{backend::Backend, Int, Tensor},
    };
    struct Net<B: Backend> {
        embed: Embedding<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                embed: burn::module::Module::<B>::load_record(self.embed, record.embed),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                embed: burn::module::Module::<B>::into_record(self.embed),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.embed);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.embed, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let embed = burn::module::Module::<B>::map(self.embed, mapper);
            Self { embed }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.embed, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let embed = burn::module::Module::<B>::to_device(self.embed, device);
            Self { embed }
        }
        fn fork(self, device: &B::Device) -> Self {
            let embed = burn::module::Module::<B>::fork(self.embed, device);
            Self { embed }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let embed = burn::module::AutodiffModule::<B>::valid(&self.embed);
            Self::InnerModule { embed }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let embed = self.embed.clone();
            Self { embed }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub embed: <Embedding<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Embedding < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub embed: <<Embedding<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecordItem",
                "embed",
                &&self.embed,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                embed: ::core::clone::Clone::clone(&self.embed),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Embedding<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "embed",
                    &self.embed,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Embedding<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "embed" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"embed" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Embedding<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Embedding<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Embedding<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { embed: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Embedding<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("embed"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Embedding<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("embed")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { embed: __field0 })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["embed"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                embed: burn::record::Record::into_item::<S>(self.embed),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                embed: burn::record::Record::from_item::<S>(item.embed),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecord",
                "embed",
                &&self.embed,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                embed: ::core::clone::Clone::clone(&self.embed),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Net",
                "embed",
                &&self.embed,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let embed = EmbeddingConfig::new(10, 3).init_with(record.embed);
            Self { embed }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
            self.embed.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "embedding::tests::embedding"]
        pub const embedding: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("embedding::tests::embedding"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/embedding/mod.rs",
                start_line: 36usize,
                start_col: 8usize,
                end_line: 36usize,
                end_col: 17usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(embedding())),
        };
        fn embedding() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/embedding");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                2,
                Int,
            >::from_data([[1, 2, 4, 5], [4, 3, 2, 9]]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                3,
            >::from_data([
                [
                    [-1.609_484_9, -0.10016718, -0.609_188_9],
                    [-0.97977227, -1.609_096_3, -0.712_144_6],
                    [-0.22227049, 1.687_113_4, -0.32062083],
                    [-0.29934573, 1.879_345_7, -0.07213178],
                ],
                [
                    [-0.22227049, 1.687_113_4, -0.32062083],
                    [0.303_722, -0.777_314_3, -0.25145486],
                    [-0.97977227, -1.609_096_3, -0.712_144_6],
                    [-0.02878714, 2.357_111, -1.037_338_7],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 3);
        }
    }
}
mod group_norm {
    use burn::{
        module::Module, nn::{GroupNorm, GroupNormConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        norm1: GroupNorm<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                norm1: burn::module::Module::<B>::load_record(self.norm1, record.norm1),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                norm1: burn::module::Module::<B>::into_record(self.norm1),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.norm1);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.norm1, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let norm1 = burn::module::Module::<B>::map(self.norm1, mapper);
            Self { norm1 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.norm1, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let norm1 = burn::module::Module::<B>::to_device(self.norm1, device);
            Self { norm1 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let norm1 = burn::module::Module::<B>::fork(self.norm1, device);
            Self { norm1 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let norm1 = burn::module::AutodiffModule::<B>::valid(&self.norm1);
            Self::InnerModule { norm1 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let norm1 = self.norm1.clone();
            Self { norm1 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub norm1: <GroupNorm<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < GroupNorm < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub norm1: <<GroupNorm<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecordItem",
                "norm1",
                &&self.norm1,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                norm1: ::core::clone::Clone::clone(&self.norm1),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<GroupNorm<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "norm1",
                    &self.norm1,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<GroupNorm<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "norm1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"norm1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<GroupNorm<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<GroupNorm<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<GroupNorm<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { norm1: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<GroupNorm<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("norm1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<GroupNorm<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("norm1")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { norm1: __field0 })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["norm1"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                norm1: burn::record::Record::into_item::<S>(self.norm1),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                norm1: burn::record::Record::from_item::<S>(item.norm1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecord",
                "norm1",
                &&self.norm1,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                norm1: ::core::clone::Clone::clone(&self.norm1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Net",
                "norm1",
                &&self.norm1,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let norm1 = GroupNormConfig::new(2, 6).init_with(record.norm1);
            Self { norm1 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            self.norm1.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "group_norm::tests::group_norm"]
        pub const group_norm: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("group_norm::tests::group_norm"),
                ignore: true,
                ignore_message: ::core::option::Option::Some(
                    "Failing possibly due to bug in Burn\'s group norm.",
                ),
                source_file: "burn-import/pytorch-tests/tests/group_norm/mod.rs",
                start_line: 37usize,
                start_col: 8usize,
                end_line: 37usize,
                end_col: 18usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(group_norm())),
        };
        #[ignore = "Failing possibly due to bug in Burn's group norm."]
        fn group_norm() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/group_norm");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[0.757_631_6, 0.27931088], [0.40306926, 0.73468447]],
                    [[0.02928156, 0.799_858_6], [0.39713734, 0.75437194]],
                    [[0.569_508_5, 0.43877792], [0.63868046, 0.524_665_9]],
                    [[0.682_614_1, 0.305_149_5], [0.46354562, 0.45498633]],
                    [[0.572_472, 0.498_002_6], [0.93708336, 0.65559506]],
                    [[0.31379688, 0.19801933], [0.41619217, 0.28432965]],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[1.042_578_5, -1.122_016_7], [-0.56195974, 0.938_733_6]],
                    [[-2.253_500_7, 1.233_672_9], [-0.588_804_1, 1.027_827_3]],
                    [[0.19124532, -0.40036356], [0.504_276_5, -0.01168585]],
                    [[1.013_829_2, -0.891_984_6], [-0.09224463, -0.13546038]],
                    [[0.45772314, 0.08172822], [2.298_641_4, 0.877_410_4]],
                    [[-0.84832406, -1.432_883_4], [-0.331_331_5, -0.997_103_7]],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 3);
        }
    }
}
mod integer {
    use burn::{
        module::{Module, Param},
        tensor::{backend::Backend, Int, Tensor},
    };
    struct Net<B: Backend> {
        buffer: Param<Tensor<B, 1, Int>>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                buffer: burn::module::Module::<
                    B,
                >::load_record(self.buffer, record.buffer),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                buffer: burn::module::Module::<B>::into_record(self.buffer),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.buffer);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.buffer, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let buffer = burn::module::Module::<B>::map(self.buffer, mapper);
            Self { buffer }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.buffer, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let buffer = burn::module::Module::<B>::to_device(self.buffer, device);
            Self { buffer }
        }
        fn fork(self, device: &B::Device) -> Self {
            let buffer = burn::module::Module::<B>::fork(self.buffer, device);
            Self { buffer }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let buffer = burn::module::AutodiffModule::<B>::valid(&self.buffer);
            Self::InnerModule { buffer }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let buffer = self.buffer.clone();
            Self { buffer }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub buffer: <Param<Tensor<B, 1, Int>> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Param < Tensor < B, 1, Int > > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub buffer: <<Param<
            Tensor<B, 1, Int>,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecordItem",
                "buffer",
                &&self.buffer,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                buffer: ::core::clone::Clone::clone(&self.buffer),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Param<
                Tensor<B, 1, Int>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "buffer",
                    &self.buffer,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Param<
                Tensor<B, 1, Int>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "buffer" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"buffer" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Param<
                        Tensor<B, 1, Int>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Param<
                        Tensor<B, 1, Int>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Param<
                                Tensor<B, 1, Int>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { buffer: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Param<
                                Tensor<B, 1, Int>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("buffer"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Param<
                                                Tensor<B, 1, Int>,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("buffer")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { buffer: __field0 })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["buffer"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                buffer: burn::record::Record::into_item::<S>(self.buffer),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                buffer: burn::record::Record::from_item::<S>(item.buffer),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecord",
                "buffer",
                &&self.buffer,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                buffer: ::core::clone::Clone::clone(&self.buffer),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Net",
                "buffer",
                &&self.buffer,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            Self { buffer: record.buffer }
        }
        /// Forward pass of the model.
        pub fn forward(&self, _x: Tensor<B, 2>) -> Tensor<B, 1, Int> {
            self.buffer.val()
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::{
            record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
            tensor::Data,
        };
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "integer::tests::integer"]
        pub const integer: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("integer::tests::integer"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/integer/mod.rs",
                start_line: 39usize,
                start_col: 8usize,
                end_line: 39usize,
                end_col: 15usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(integer())),
        };
        fn integer() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/integer");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<Backend, 2>::ones([3, 3]);
            let output = model.forward(input);
            let expected = Tensor::<Backend, 1, Int>::from_data(Data::from([1, 2, 3]));
            match (&output.to_data(), &expected.to_data()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        }
    }
}
mod key_remap {
    use burn::{
        module::Module, nn::conv::{Conv2d, Conv2dConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        conv1: Conv2d<B>,
        conv2: Conv2d<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                conv1: burn::module::Module::<B>::load_record(self.conv1, record.conv1),
                conv2: burn::module::Module::<B>::load_record(self.conv2, record.conv2),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                conv1: burn::module::Module::<B>::into_record(self.conv1),
                conv2: burn::module::Module::<B>::into_record(self.conv2),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.conv1);
            num_params += burn::module::Module::<B>::num_params(&self.conv2);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.conv1, visitor);
            burn::module::Module::visit(&self.conv2, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let conv1 = burn::module::Module::<B>::map(self.conv1, mapper);
            let conv2 = burn::module::Module::<B>::map(self.conv2, mapper);
            Self { conv1, conv2 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv1, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.conv2, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::to_device(self.conv1, device);
            let conv2 = burn::module::Module::<B>::to_device(self.conv2, device);
            Self { conv1, conv2 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let conv1 = burn::module::Module::<B>::fork(self.conv1, device);
            let conv2 = burn::module::Module::<B>::fork(self.conv2, device);
            Self { conv1, conv2 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let conv1 = burn::module::AutodiffModule::<B>::valid(&self.conv1);
            let conv2 = burn::module::AutodiffModule::<B>::valid(&self.conv2);
            Self::InnerModule { conv1, conv2 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let conv1 = self.conv1.clone();
            let conv2 = self.conv2.clone();
            Self { conv1, conv2 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub conv1: <Conv2d<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub conv2: <Conv2d<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Conv2d < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < Conv2d < B > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub conv1: <<Conv2d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub conv2: <<Conv2d<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecordItem",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv1",
                    &self.conv1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "conv2",
                    &self.conv2,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Conv2d<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "conv1" => _serde::__private::Ok(__Field::__field0),
                            "conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"conv1" => _serde::__private::Ok(__Field::__field0),
                            b"conv2" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Conv2d<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct NetRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<Conv2d<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Conv2d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("conv2"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Conv2d<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv1")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("conv2")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            conv1: __field0,
                            conv2: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["conv1", "conv2"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                conv1: burn::record::Record::into_item::<S>(self.conv1),
                conv2: burn::record::Record::into_item::<S>(self.conv2),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                conv1: burn::record::Record::from_item::<S>(item.conv1),
                conv2: burn::record::Record::from_item::<S>(item.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "NetRecord",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                conv1: ::core::clone::Clone::clone(&self.conv1),
                conv2: ::core::clone::Clone::clone(&self.conv2),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "Net",
                "conv1",
                &self.conv1,
                "conv2",
                &&self.conv2,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let conv1 = Conv2dConfig::new([2, 2], [2, 2]).init_with(record.conv1);
            let conv2 = Conv2dConfig::new([2, 2], [2, 2])
                .with_bias(false)
                .init_with(record.conv2);
            Self { conv1, conv2 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            let x = self.conv1.forward(x);
            self.conv2.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "key_remap::tests::key_remap"]
        pub const key_remap: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("key_remap::tests::key_remap"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/key_remap/mod.rs",
                start_line: 42usize,
                start_col: 8usize,
                end_line: 42usize,
                end_col: 17usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(key_remap())),
        };
        fn key_remap() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/key_remap");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [
                        [0.024_595_8, 0.25883394, 0.93905586, 0.416_715_5, 0.713_979_7],
                        [0.267_644_3, 0.990_609, 0.28845078, 0.874_962_4, 0.505_920_8],
                        [0.23659128, 0.757_007_4, 0.23458993, 0.64705235, 0.355_621_4],
                        [0.445_182_8, 0.01930594, 0.26160914, 0.771_317, 0.37846136],
                        [0.99802476, 0.900_794_2, 0.476_588_2, 0.16625845, 0.804_481_1],
                    ],
                    [
                        [0.65517855, 0.17679012, 0.824_772_3, 0.803_550_9, 0.943_447_5],
                        [0.21972018, 0.417_697, 0.49031407, 0.57302874, 0.12054086],
                        [0.14518881, 0.772_002_3, 0.38275403, 0.744_236_7, 0.52850497],
                        [0.664_172_4, 0.60994434, 0.681_799_7, 0.74785537, 0.03694397],
                        [0.751_675_7, 0.148_438_4, 0.12274551, 0.530_407_2, 0.414_796_4],
                    ],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [
                        [-0.02502128, 0.00250649, 0.04841233],
                        [0.04589614, -0.00296854, 0.01991477],
                        [0.02920526, 0.059_497_3, 0.04326791],
                    ],
                    [
                        [-0.04825336, 0.080_190_9, -0.02375088],
                        [0.02885434, 0.09638263, -0.07460806],
                        [0.02004079, 0.06244051, 0.035_887_1],
                    ],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 7);
        }
    }
}
mod layer_norm {
    use burn::{
        module::Module, nn::{LayerNorm, LayerNormConfig},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        norm1: LayerNorm<B>,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                norm1: burn::module::Module::<B>::load_record(self.norm1, record.norm1),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                norm1: burn::module::Module::<B>::into_record(self.norm1),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.norm1);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.norm1, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let norm1 = burn::module::Module::<B>::map(self.norm1, mapper);
            Self { norm1 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.norm1, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let norm1 = burn::module::Module::<B>::to_device(self.norm1, device);
            Self { norm1 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let norm1 = burn::module::Module::<B>::fork(self.norm1, device);
            Self { norm1 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let norm1 = burn::module::AutodiffModule::<B>::valid(&self.norm1);
            Self::InnerModule { norm1 }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let norm1 = self.norm1.clone();
            Self { norm1 }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub norm1: <LayerNorm<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < LayerNorm < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub norm1: <<LayerNorm<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecordItem",
                "norm1",
                &&self.norm1,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                norm1: ::core::clone::Clone::clone(&self.norm1),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<LayerNorm<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "norm1",
                    &self.norm1,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<LayerNorm<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "norm1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"norm1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<LayerNorm<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<LayerNorm<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<LayerNorm<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { norm1: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<LayerNorm<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("norm1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<LayerNorm<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("norm1")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem { norm1: __field0 })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["norm1"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                norm1: burn::record::Record::into_item::<S>(self.norm1),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                norm1: burn::record::Record::from_item::<S>(item.norm1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetRecord",
                "norm1",
                &&self.norm1,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                norm1: ::core::clone::Clone::clone(&self.norm1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Net",
                "norm1",
                &&self.norm1,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let norm1 = LayerNormConfig::new(4).init_with(record.norm1);
            Self { norm1 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            self.norm1.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "layer_norm::tests::layer_norm"]
        pub const layer_norm: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("layer_norm::tests::layer_norm"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/layer_norm/mod.rs",
                start_line: 36usize,
                start_col: 8usize,
                end_line: 36usize,
                end_col: 18usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(layer_norm())),
        };
        fn layer_norm() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/layer_norm");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[0.757_631_6, 0.27931088], [0.40306926, 0.73468447]],
                    [[0.02928156, 0.799_858_6], [0.39713734, 0.75437194]],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[0.99991274, -0.999_912_5], [-0.999_818_3, 0.999_818_3]],
                    [[-0.999_966_2, 0.99996626], [-0.99984336, 0.99984336]],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 3);
        }
    }
}
mod linear {
    use burn::{
        module::Module, nn::{Linear, LinearConfig, ReLU},
        tensor::{backend::Backend, Tensor},
    };
    struct Net<B: Backend> {
        fc1: Linear<B>,
        fc2: Linear<B>,
        relu: ReLU,
    }
    impl<B: Backend> burn::module::Module<B> for Net<B> {
        type Record = NetRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                fc1: burn::module::Module::<B>::load_record(self.fc1, record.fc1),
                fc2: burn::module::Module::<B>::load_record(self.fc2, record.fc2),
                relu: burn::module::Module::<B>::load_record(self.relu, record.relu),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                fc1: burn::module::Module::<B>::into_record(self.fc1),
                fc2: burn::module::Module::<B>::into_record(self.fc2),
                relu: burn::module::Module::<B>::into_record(self.relu),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.fc1);
            num_params += burn::module::Module::<B>::num_params(&self.fc2);
            num_params += burn::module::Module::<B>::num_params(&self.relu);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.fc1, visitor);
            burn::module::Module::visit(&self.fc2, visitor);
            burn::module::Module::visit(&self.relu, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let fc1 = burn::module::Module::<B>::map(self.fc1, mapper);
            let fc2 = burn::module::Module::<B>::map(self.fc2, mapper);
            let relu = burn::module::Module::<B>::map(self.relu, mapper);
            Self { fc1, fc2, relu }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<B>::collect_devices(&self.fc1, devices);
            let devices = burn::module::Module::<B>::collect_devices(&self.fc2, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.relu, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let fc1 = burn::module::Module::<B>::to_device(self.fc1, device);
            let fc2 = burn::module::Module::<B>::to_device(self.fc2, device);
            let relu = burn::module::Module::<B>::to_device(self.relu, device);
            Self { fc1, fc2, relu }
        }
        fn fork(self, device: &B::Device) -> Self {
            let fc1 = burn::module::Module::<B>::fork(self.fc1, device);
            let fc2 = burn::module::Module::<B>::fork(self.fc2, device);
            let relu = burn::module::Module::<B>::fork(self.relu, device);
            Self { fc1, fc2, relu }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for Net<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = Net<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let fc1 = burn::module::AutodiffModule::<B>::valid(&self.fc1);
            let fc2 = burn::module::AutodiffModule::<B>::valid(&self.fc2);
            let relu = burn::module::AutodiffModule::<B>::valid(&self.relu);
            Self::InnerModule {
                fc1,
                fc2,
                relu,
            }
        }
    }
    impl<B: Backend> core::fmt::Display for Net<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0}[num_params={1}]", "Net", self.num_params()))
        }
    }
    impl<B: Backend> Clone for Net<B> {
        fn clone(&self) -> Self {
            let fc1 = self.fc1.clone();
            let fc2 = self.fc2.clone();
            let relu = self.relu.clone();
            Self { fc1, fc2, relu }
        }
    }
    /// The record type for the module.
    pub struct NetRecord<B: Backend> {
        /// The module record associative type.
        pub fc1: <Linear<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub fc2: <Linear<B> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub relu: <ReLU as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Linear < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned, < < Linear < B > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned, < < ReLU as burn :: module :: Module < B > >\n:: Record as burn :: record :: Record > :: Item < S > : serde :: Serialize +\nserde :: de :: DeserializeOwned,"
    )]
    pub struct NetRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub fc1: <<Linear<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub fc2: <<Linear<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
        /// Field to be serialized.
        pub relu: <<ReLU as burn::module::Module<
            B,
        >>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "NetRecordItem",
                "fc1",
                &self.fc1,
                "fc2",
                &self.fc2,
                "relu",
                &&self.relu,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetRecordItem<B, S> {
            NetRecordItem {
                fc1: ::core::clone::Clone::clone(&self.fc1),
                fc2: ::core::clone::Clone::clone(&self.fc2),
                relu: ::core::clone::Clone::clone(&self.relu),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetRecordItem<B, S>
        where
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<ReLU as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetRecordItem",
                    false as usize + 1 + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "fc1",
                    &self.fc1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "fc2",
                    &self.fc2,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "relu",
                    &self.relu,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetRecordItem<B, S>
        where
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
            <<ReLU as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __field2,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            2u64 => _serde::__private::Ok(__Field::__field2),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "fc1" => _serde::__private::Ok(__Field::__field0),
                            "fc2" => _serde::__private::Ok(__Field::__field1),
                            "relu" => _serde::__private::Ok(__Field::__field2),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"fc1" => _serde::__private::Ok(__Field::__field0),
                            b"fc2" => _serde::__private::Ok(__Field::__field1),
                            b"relu" => _serde::__private::Ok(__Field::__field2),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<ReLU as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                    <<ReLU as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetRecordItem with 3 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct NetRecordItem with 3 elements",
                                    ),
                                );
                            }
                        };
                        let __field2 = match _serde::de::SeqAccess::next_element::<
                            <<ReLU as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        2usize,
                                        &"struct NetRecordItem with 3 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            fc1: __field0,
                            fc2: __field1,
                            relu: __field2,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field2: _serde::__private::Option<
                            <<ReLU as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("fc1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Linear<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("fc2"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Linear<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field2 => {
                                    if _serde::__private::Option::is_some(&__field2) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("relu"),
                                        );
                                    }
                                    __field2 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<ReLU as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("fc1")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("fc2")?
                            }
                        };
                        let __field2 = match __field2 {
                            _serde::__private::Some(__field2) => __field2,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("relu")?
                            }
                        };
                        _serde::__private::Ok(NetRecordItem {
                            fc1: __field0,
                            fc2: __field1,
                            relu: __field2,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["fc1", "fc2", "relu"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<NetRecordItem<B, S>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetRecordItem {
                fc1: burn::record::Record::into_item::<S>(self.fc1),
                fc2: burn::record::Record::into_item::<S>(self.fc2),
                relu: burn::record::Record::into_item::<S>(self.relu),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                fc1: burn::record::Record::from_item::<S>(item.fc1),
                fc2: burn::record::Record::from_item::<S>(item.fc2),
                relu: burn::record::Record::from_item::<S>(item.relu),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "NetRecord",
                "fc1",
                &self.fc1,
                "fc2",
                &self.fc2,
                "relu",
                &&self.relu,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone for NetRecord<B> {
        #[inline]
        fn clone(&self) -> NetRecord<B> {
            NetRecord {
                fc1: ::core::clone::Clone::clone(&self.fc1),
                fc2: ::core::clone::Clone::clone(&self.fc2),
                relu: ::core::clone::Clone::clone(&self.relu),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Net<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "Net",
                "fc1",
                &self.fc1,
                "fc2",
                &self.fc2,
                "relu",
                &&self.relu,
            )
        }
    }
    impl<B: Backend> Net<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetRecord<B>) -> Self {
            let fc1 = LinearConfig::new(2, 3).init_with(record.fc1);
            let fc2 = LinearConfig::new(3, 4).init_with(record.fc2);
            let relu = ReLU::default();
            Self { fc1, fc2, relu }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            let x = self.fc1.forward(x);
            let x = self.relu.forward(x);
            self.fc2.forward(x)
        }
    }
    struct NetWithBias<B: Backend> {
        fc1: Linear<B>,
    }
    impl<B: Backend> burn::module::Module<B> for NetWithBias<B> {
        type Record = NetWithBiasRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                fc1: burn::module::Module::<B>::load_record(self.fc1, record.fc1),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                fc1: burn::module::Module::<B>::into_record(self.fc1),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.fc1);
            num_params
        }
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
            burn::module::Module::visit(&self.fc1, visitor);
        }
        fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
            let fc1 = burn::module::Module::<B>::map(self.fc1, mapper);
            Self { fc1 }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<B>::collect_devices(&self.fc1, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let fc1 = burn::module::Module::<B>::to_device(self.fc1, device);
            Self { fc1 }
        }
        fn fork(self, device: &B::Device) -> Self {
            let fc1 = burn::module::Module::<B>::fork(self.fc1, device);
            Self { fc1 }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for NetWithBias<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = NetWithBias<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let fc1 = burn::module::AutodiffModule::<B>::valid(&self.fc1);
            Self::InnerModule { fc1 }
        }
    }
    impl<B: Backend> core::fmt::Display for NetWithBias<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(
                format_args!("{0}[num_params={1}]", "NetWithBias", self.num_params()),
            )
        }
    }
    impl<B: Backend> Clone for NetWithBias<B> {
        fn clone(&self) -> Self {
            let fc1 = self.fc1.clone();
            Self { fc1 }
        }
    }
    /// The record type for the module.
    pub struct NetWithBiasRecord<B: Backend> {
        /// The module record associative type.
        pub fc1: <Linear<B> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(
        bound = "< < Linear < B > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record > :: Item < S > : serde :: Serialize + serde :: de ::\nDeserializeOwned,"
    )]
    pub struct NetWithBiasRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
        /// Field to be serialized.
        pub fc1: <<Linear<
            B,
        > as burn::module::Module<B>>::Record as burn::record::Record>::Item<S>,
    }
    #[automatically_derived]
    impl<
        B: ::core::fmt::Debug + Backend,
        S: ::core::fmt::Debug + burn::record::PrecisionSettings,
    > ::core::fmt::Debug for NetWithBiasRecordItem<B, S> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetWithBiasRecordItem",
                "fc1",
                &&self.fc1,
            )
        }
    }
    #[automatically_derived]
    impl<
        B: ::core::clone::Clone + Backend,
        S: ::core::clone::Clone + burn::record::PrecisionSettings,
    > ::core::clone::Clone for NetWithBiasRecordItem<B, S> {
        #[inline]
        fn clone(&self) -> NetWithBiasRecordItem<B, S> {
            NetWithBiasRecordItem {
                fc1: ::core::clone::Clone::clone(&self.fc1),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
        for NetWithBiasRecordItem<B, S>
        where
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "NetWithBiasRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "fc1",
                    &self.fc1,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > _serde::Deserialize<'de> for NetWithBiasRecordItem<B, S>
        where
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record>::Item<
                S,
            >: serde::Serialize + serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "fc1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"fc1" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<NetWithBiasRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record>::Item<
                        S,
                    >: serde::Serialize + serde::de::DeserializeOwned,
                {
                    type Value = NetWithBiasRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct NetWithBiasRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct NetWithBiasRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(NetWithBiasRecordItem {
                            fc1: __field0,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Linear<
                                B,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key)
                            = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("fc1"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Linear<
                                                B,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("fc1")?
                            }
                        };
                        _serde::__private::Ok(NetWithBiasRecordItem {
                            fc1: __field0,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["fc1"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "NetWithBiasRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<
                            NetWithBiasRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record for NetWithBiasRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = NetWithBiasRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            NetWithBiasRecordItem {
                fc1: burn::record::Record::into_item::<S>(self.fc1),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
            Self {
                fc1: burn::record::Record::from_item::<S>(item.fc1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetWithBiasRecord<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetWithBiasRecord",
                "fc1",
                &&self.fc1,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Backend> ::core::clone::Clone
    for NetWithBiasRecord<B> {
        #[inline]
        fn clone(&self) -> NetWithBiasRecord<B> {
            NetWithBiasRecord {
                fc1: ::core::clone::Clone::clone(&self.fc1),
            }
        }
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for NetWithBias<B> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "NetWithBias",
                "fc1",
                &&self.fc1,
            )
        }
    }
    impl<B: Backend> NetWithBias<B> {
        /// Create a new model from the given record.
        pub fn new_with(record: NetWithBiasRecord<B>) -> Self {
            let fc1 = LinearConfig::new(2, 3).init_with(record.fc1);
            Self { fc1 }
        }
        /// Forward pass of the model.
        pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
            self.fc1.forward(x)
        }
    }
    #[cfg(test)]
    mod tests {
        type Backend = burn_ndarray::NdArray<f32>;
        use std::{env, path::Path};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        use super::*;
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "linear::tests::linear"]
        pub const linear: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("linear::tests::linear"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/linear/mod.rs",
                start_line: 63usize,
                start_col: 8usize,
                end_line: 63usize,
                end_col: 14usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(linear())),
        };
        fn linear() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/labeled/linear");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = Net::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[0.63968194, 0.97427773], [0.830_029_9, 0.04443115]],
                    [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [
                        [0.09778349, -0.13756673, 0.04962806, 0.08856435],
                        [0.03163241, -0.02848549, 0.01437942, 0.11905234],
                    ],
                    [
                        [0.07628226, -0.10757702, 0.03656857, 0.03824598],
                        [0.05443089, -0.06904714, 0.02744314, 0.09997337],
                    ],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 6);
        }
        extern crate test;
        #[cfg(test)]
        #[rustc_test_marker = "linear::tests::linear_with_bias"]
        pub const linear_with_bias: test::TestDescAndFn = test::TestDescAndFn {
            desc: test::TestDesc {
                name: test::StaticTestName("linear::tests::linear_with_bias"),
                ignore: false,
                ignore_message: ::core::option::Option::None,
                source_file: "burn-import/pytorch-tests/tests/linear/mod.rs",
                start_line: 95usize,
                start_col: 8usize,
                end_line: 95usize,
                end_col: 24usize,
                compile_fail: false,
                no_run: false,
                should_panic: test::ShouldPanic::No,
                test_type: test::TestType::IntegrationTest,
            },
            testfn: test::StaticTestFn(|| test::assert_test_result(linear_with_bias())),
        };
        fn linear_with_bias() {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let file_path = Path::new(&out_dir).join("model/guessed/linear_with_bias");
            let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
                .load(file_path)
                .expect("Failed to decode state");
            let model = NetWithBias::<Backend>::new_with(record);
            let input = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [[0.63968194, 0.97427773], [0.830_029_9, 0.04443115]],
                    [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
                ],
            ]);
            let output = model.forward(input);
            let expected = Tensor::<
                Backend,
                4,
            >::from_data([
                [
                    [
                        [-0.00432095, -1.107_101_2, 0.870_691_4],
                        [0.024_595_5, -0.954_462_9, 0.48518157],
                    ],
                    [
                        [0.34315687, -0.757_384_2, 0.548_288],
                        [-0.06608963, -1.072_072_7, 0.645_800_5],
                    ],
                ],
            ]);
            output.to_data().assert_approx_eq(&expected.to_data(), 6);
        }
    }
}
#[rustc_main]
#[coverage(off)]
pub fn main() -> () {
    extern crate test;
    test::test_main_static(
        &[
            &batch_norm2d,
            &boolean,
            &buffer,
            &json_full_record,
            &json_half_record,
            &mpk_full_record,
            &mpk_gz_full_record,
            &mpk_gz_half_record,
            &mpk_half_record,
            &conv1d,
            &conv2d,
            &conv_transpose1d,
            &conv_transpose2d,
            &embedding,
            &group_norm,
            &integer,
            &key_remap,
            &layer_norm,
            &linear,
            &linear_with_bias,
        ],
    )
}
