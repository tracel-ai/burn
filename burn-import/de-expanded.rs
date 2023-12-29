#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
use serde::de::{self, DeserializeSeed, IntoDeserializer, Visitor};
use serde::{forward_to_deserialize_any, Deserialize, Serialize};
use std::collections::HashMap;
struct MyStruct1 {
    a: MyStruct3,
    b: MyStruct2,
}
#[doc(hidden)]
#[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl _serde::Serialize for MyStruct1 {
        fn serialize<__S>(
            &self,
            __serializer: __S,
        ) -> _serde::__private::Result<__S::Ok, __S::Error>
        where
            __S: _serde::Serializer,
        {
            let mut __serde_state = _serde::Serializer::serialize_struct(
                __serializer,
                "MyStruct1",
                false as usize + 1 + 1,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "a",
                &self.a,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "b",
                &self.b,
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
    impl<'de> _serde::Deserialize<'de> for MyStruct1 {
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
                        "a" => _serde::__private::Ok(__Field::__field0),
                        "b" => _serde::__private::Ok(__Field::__field1),
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
                        b"a" => _serde::__private::Ok(__Field::__field0),
                        b"b" => _serde::__private::Ok(__Field::__field1),
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
            struct __Visitor<'de> {
                marker: _serde::__private::PhantomData<MyStruct1>,
                lifetime: _serde::__private::PhantomData<&'de ()>,
            }
            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                type Value = MyStruct1;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private::Formatter,
                ) -> _serde::__private::fmt::Result {
                    _serde::__private::Formatter::write_str(
                        __formatter,
                        "struct MyStruct1",
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
                        MyStruct3,
                    >(&mut __seq)? {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct MyStruct1 with 2 elements",
                                ),
                            );
                        }
                    };
                    let __field1 = match _serde::de::SeqAccess::next_element::<
                        MyStruct2,
                    >(&mut __seq)? {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct MyStruct1 with 2 elements",
                                ),
                            );
                        }
                    };
                    _serde::__private::Ok(MyStruct1 {
                        a: __field0,
                        b: __field1,
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
                    let mut __field0: _serde::__private::Option<MyStruct3> = _serde::__private::None;
                    let mut __field1: _serde::__private::Option<MyStruct2> = _serde::__private::None;
                    while let _serde::__private::Some(__key)
                        = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                        match __key {
                            __Field::__field0 => {
                                if _serde::__private::Option::is_some(&__field0) {
                                    return _serde::__private::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("a"),
                                    );
                                }
                                __field0 = _serde::__private::Some(
                                    _serde::de::MapAccess::next_value::<MyStruct3>(&mut __map)?,
                                );
                            }
                            __Field::__field1 => {
                                if _serde::__private::Option::is_some(&__field1) {
                                    return _serde::__private::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("b"),
                                    );
                                }
                                __field1 = _serde::__private::Some(
                                    _serde::de::MapAccess::next_value::<MyStruct2>(&mut __map)?,
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
                            _serde::__private::de::missing_field("a")?
                        }
                    };
                    let __field1 = match __field1 {
                        _serde::__private::Some(__field1) => __field1,
                        _serde::__private::None => {
                            _serde::__private::de::missing_field("b")?
                        }
                    };
                    _serde::__private::Ok(MyStruct1 {
                        a: __field0,
                        b: __field1,
                    })
                }
            }
            #[doc(hidden)]
            const FIELDS: &'static [&'static str] = &["a", "b"];
            _serde::Deserializer::deserialize_struct(
                __deserializer,
                "MyStruct1",
                FIELDS,
                __Visitor {
                    marker: _serde::__private::PhantomData::<MyStruct1>,
                    lifetime: _serde::__private::PhantomData,
                },
            )
        }
    }
};
#[automatically_derived]
impl ::core::fmt::Debug for MyStruct1 {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_struct_field2_finish(
            f,
            "MyStruct1",
            "a",
            &self.a,
            "b",
            &&self.b,
        )
    }
}
#[automatically_derived]
impl ::core::clone::Clone for MyStruct1 {
    #[inline]
    fn clone(&self) -> MyStruct1 {
        MyStruct1 {
            a: ::core::clone::Clone::clone(&self.a),
            b: ::core::clone::Clone::clone(&self.b),
        }
    }
}
struct MyStruct2 {
    a: i32,
    b: i32,
    c: i32,
}
#[doc(hidden)]
#[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl _serde::Serialize for MyStruct2 {
        fn serialize<__S>(
            &self,
            __serializer: __S,
        ) -> _serde::__private::Result<__S::Ok, __S::Error>
        where
            __S: _serde::Serializer,
        {
            let mut __serde_state = _serde::Serializer::serialize_struct(
                __serializer,
                "MyStruct2",
                false as usize + 1 + 1 + 1,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "a",
                &self.a,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "b",
                &self.b,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "c",
                &self.c,
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
    impl<'de> _serde::Deserialize<'de> for MyStruct2 {
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
                        "a" => _serde::__private::Ok(__Field::__field0),
                        "b" => _serde::__private::Ok(__Field::__field1),
                        "c" => _serde::__private::Ok(__Field::__field2),
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
                        b"a" => _serde::__private::Ok(__Field::__field0),
                        b"b" => _serde::__private::Ok(__Field::__field1),
                        b"c" => _serde::__private::Ok(__Field::__field2),
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
            struct __Visitor<'de> {
                marker: _serde::__private::PhantomData<MyStruct2>,
                lifetime: _serde::__private::PhantomData<&'de ()>,
            }
            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                type Value = MyStruct2;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private::Formatter,
                ) -> _serde::__private::fmt::Result {
                    _serde::__private::Formatter::write_str(
                        __formatter,
                        "struct MyStruct2",
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
                        i32,
                    >(&mut __seq)? {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct MyStruct2 with 3 elements",
                                ),
                            );
                        }
                    };
                    let __field1 = match _serde::de::SeqAccess::next_element::<
                        i32,
                    >(&mut __seq)? {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct MyStruct2 with 3 elements",
                                ),
                            );
                        }
                    };
                    let __field2 = match _serde::de::SeqAccess::next_element::<
                        i32,
                    >(&mut __seq)? {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    2usize,
                                    &"struct MyStruct2 with 3 elements",
                                ),
                            );
                        }
                    };
                    _serde::__private::Ok(MyStruct2 {
                        a: __field0,
                        b: __field1,
                        c: __field2,
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
                    let mut __field0: _serde::__private::Option<i32> = _serde::__private::None;
                    let mut __field1: _serde::__private::Option<i32> = _serde::__private::None;
                    let mut __field2: _serde::__private::Option<i32> = _serde::__private::None;
                    while let _serde::__private::Some(__key)
                        = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                        match __key {
                            __Field::__field0 => {
                                if _serde::__private::Option::is_some(&__field0) {
                                    return _serde::__private::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("a"),
                                    );
                                }
                                __field0 = _serde::__private::Some(
                                    _serde::de::MapAccess::next_value::<i32>(&mut __map)?,
                                );
                            }
                            __Field::__field1 => {
                                if _serde::__private::Option::is_some(&__field1) {
                                    return _serde::__private::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("b"),
                                    );
                                }
                                __field1 = _serde::__private::Some(
                                    _serde::de::MapAccess::next_value::<i32>(&mut __map)?,
                                );
                            }
                            __Field::__field2 => {
                                if _serde::__private::Option::is_some(&__field2) {
                                    return _serde::__private::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("c"),
                                    );
                                }
                                __field2 = _serde::__private::Some(
                                    _serde::de::MapAccess::next_value::<i32>(&mut __map)?,
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
                            _serde::__private::de::missing_field("a")?
                        }
                    };
                    let __field1 = match __field1 {
                        _serde::__private::Some(__field1) => __field1,
                        _serde::__private::None => {
                            _serde::__private::de::missing_field("b")?
                        }
                    };
                    let __field2 = match __field2 {
                        _serde::__private::Some(__field2) => __field2,
                        _serde::__private::None => {
                            _serde::__private::de::missing_field("c")?
                        }
                    };
                    _serde::__private::Ok(MyStruct2 {
                        a: __field0,
                        b: __field1,
                        c: __field2,
                    })
                }
            }
            #[doc(hidden)]
            const FIELDS: &'static [&'static str] = &["a", "b", "c"];
            _serde::Deserializer::deserialize_struct(
                __deserializer,
                "MyStruct2",
                FIELDS,
                __Visitor {
                    marker: _serde::__private::PhantomData::<MyStruct2>,
                    lifetime: _serde::__private::PhantomData,
                },
            )
        }
    }
};
#[automatically_derived]
impl ::core::fmt::Debug for MyStruct2 {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_struct_field3_finish(
            f,
            "MyStruct2",
            "a",
            &self.a,
            "b",
            &self.b,
            "c",
            &&self.c,
        )
    }
}
#[automatically_derived]
impl ::core::clone::Clone for MyStruct2 {
    #[inline]
    fn clone(&self) -> MyStruct2 {
        MyStruct2 {
            a: ::core::clone::Clone::clone(&self.a),
            b: ::core::clone::Clone::clone(&self.b),
            c: ::core::clone::Clone::clone(&self.c),
        }
    }
}
struct MyStruct3 {
    x: String,
    y: String,
}
#[doc(hidden)]
#[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl _serde::Serialize for MyStruct3 {
        fn serialize<__S>(
            &self,
            __serializer: __S,
        ) -> _serde::__private::Result<__S::Ok, __S::Error>
        where
            __S: _serde::Serializer,
        {
            let mut __serde_state = _serde::Serializer::serialize_struct(
                __serializer,
                "MyStruct3",
                false as usize + 1 + 1,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "x",
                &self.x,
            )?;
            _serde::ser::SerializeStruct::serialize_field(
                &mut __serde_state,
                "y",
                &self.y,
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
    impl<'de> _serde::Deserialize<'de> for MyStruct3 {
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
                        "x" => _serde::__private::Ok(__Field::__field0),
                        "y" => _serde::__private::Ok(__Field::__field1),
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
                        b"x" => _serde::__private::Ok(__Field::__field0),
                        b"y" => _serde::__private::Ok(__Field::__field1),
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
            struct __Visitor<'de> {
                marker: _serde::__private::PhantomData<MyStruct3>,
                lifetime: _serde::__private::PhantomData<&'de ()>,
            }
            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                type Value = MyStruct3;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private::Formatter,
                ) -> _serde::__private::fmt::Result {
                    _serde::__private::Formatter::write_str(
                        __formatter,
                        "struct MyStruct3",
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
                        String,
                    >(&mut __seq)? {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct MyStruct3 with 2 elements",
                                ),
                            );
                        }
                    };
                    let __field1 = match _serde::de::SeqAccess::next_element::<
                        String,
                    >(&mut __seq)? {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct MyStruct3 with 2 elements",
                                ),
                            );
                        }
                    };
                    _serde::__private::Ok(MyStruct3 {
                        x: __field0,
                        y: __field1,
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
                    let mut __field0: _serde::__private::Option<String> = _serde::__private::None;
                    let mut __field1: _serde::__private::Option<String> = _serde::__private::None;
                    while let _serde::__private::Some(__key)
                        = _serde::de::MapAccess::next_key::<__Field>(&mut __map)? {
                        match __key {
                            __Field::__field0 => {
                                if _serde::__private::Option::is_some(&__field0) {
                                    return _serde::__private::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("x"),
                                    );
                                }
                                __field0 = _serde::__private::Some(
                                    _serde::de::MapAccess::next_value::<String>(&mut __map)?,
                                );
                            }
                            __Field::__field1 => {
                                if _serde::__private::Option::is_some(&__field1) {
                                    return _serde::__private::Err(
                                        <__A::Error as _serde::de::Error>::duplicate_field("y"),
                                    );
                                }
                                __field1 = _serde::__private::Some(
                                    _serde::de::MapAccess::next_value::<String>(&mut __map)?,
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
                            _serde::__private::de::missing_field("x")?
                        }
                    };
                    let __field1 = match __field1 {
                        _serde::__private::Some(__field1) => __field1,
                        _serde::__private::None => {
                            _serde::__private::de::missing_field("y")?
                        }
                    };
                    _serde::__private::Ok(MyStruct3 {
                        x: __field0,
                        y: __field1,
                    })
                }
            }
            #[doc(hidden)]
            const FIELDS: &'static [&'static str] = &["x", "y"];
            _serde::Deserializer::deserialize_struct(
                __deserializer,
                "MyStruct3",
                FIELDS,
                __Visitor {
                    marker: _serde::__private::PhantomData::<MyStruct3>,
                    lifetime: _serde::__private::PhantomData,
                },
            )
        }
    }
};
#[automatically_derived]
impl ::core::fmt::Debug for MyStruct3 {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::debug_struct_field2_finish(
            f,
            "MyStruct3",
            "x",
            &self.x,
            "y",
            &&self.y,
        )
    }
}
#[automatically_derived]
impl ::core::clone::Clone for MyStruct3 {
    #[inline]
    fn clone(&self) -> MyStruct3 {
        MyStruct3 {
            x: ::core::clone::Clone::clone(&self.x),
            y: ::core::clone::Clone::clone(&self.y),
        }
    }
}
enum Values {
    StructValue(HashMap<String, Values>),
    StringValue(String),
    IntValue(i32),
}
#[automatically_derived]
impl ::core::fmt::Debug for Values {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        match self {
            Values::StructValue(__self_0) => {
                ::core::fmt::Formatter::debug_tuple_field1_finish(
                    f,
                    "StructValue",
                    &__self_0,
                )
            }
            Values::StringValue(__self_0) => {
                ::core::fmt::Formatter::debug_tuple_field1_finish(
                    f,
                    "StringValue",
                    &__self_0,
                )
            }
            Values::IntValue(__self_0) => {
                ::core::fmt::Formatter::debug_tuple_field1_finish(
                    f,
                    "IntValue",
                    &__self_0,
                )
            }
        }
    }
}
#[automatically_derived]
impl ::core::clone::Clone for Values {
    #[inline]
    fn clone(&self) -> Values {
        match self {
            Values::StructValue(__self_0) => {
                Values::StructValue(::core::clone::Clone::clone(__self_0))
            }
            Values::StringValue(__self_0) => {
                Values::StringValue(::core::clone::Clone::clone(__self_0))
            }
            Values::IntValue(__self_0) => {
                Values::IntValue(::core::clone::Clone::clone(__self_0))
            }
        }
    }
}
struct ValuesDeserializer {
    value: Option<Values>,
}
impl ValuesDeserializer {
    fn new(value: Values) -> Self {
        ValuesDeserializer {
            value: Some(value),
        }
    }
}
impl<'de> de::Deserializer<'de> for ValuesDeserializer {
    type Error = de::value::Error;
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(Values::StructValue(map)) => visitor.visit_map(MyMapAccess::new(map)),
            Some(Values::StringValue(s)) => visitor.visit_string(s),
            Some(Values::IntValue(i)) => visitor.visit_i32(i),
            None => Err(de::Error::custom("No value to deserialize")),
        }
    }
    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        {
            ::std::io::_print(format_args!("deserialize_struct\n"));
        };
        {
            ::std::io::_print(format_args!("self.value: {0:?}\n", self.value));
        };
        {
            ::std::io::_print(format_args!("name: {0:?}\n", name));
        };
        {
            ::std::io::_print(format_args!("fields: {0:?}\n", fields));
        };
        match self.value {
            Some(Values::StructValue(map)) => visitor.visit_map(MyMapAccess::new(map)),
            _ => Err(de::Error::custom("Expected struct")),
        }
    }
    #[inline]
    fn deserialize_bool<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_i8<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_i16<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_i32<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_i64<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_i128<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_u8<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_u16<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_u32<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_u64<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_u128<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_f32<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_f64<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_char<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_str<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_string<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_bytes<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_byte_buf<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_option<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_unit<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_unit_struct<V>(
        self,
        name: &'static str,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        let _ = name;
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        name: &'static str,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        let _ = name;
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_seq<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_tuple<V>(
        self,
        len: usize,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        let _ = len;
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_tuple_struct<V>(
        self,
        name: &'static str,
        len: usize,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        let _ = name;
        let _ = len;
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_map<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        let _ = name;
        let _ = variants;
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_identifier<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
    #[inline]
    fn deserialize_ignored_any<V>(
        self,
        visitor: V,
    ) -> ::serde::__private::Result<V::Value, Self::Error>
    where
        V: ::serde::de::Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
}
use serde::de::MapAccess;
struct MyMapAccess {
    iter: std::collections::hash_map::IntoIter<String, Values>,
    next_value: Option<Values>,
}
impl MyMapAccess {
    fn new(map: HashMap<String, Values>) -> Self {
        MyMapAccess {
            iter: map.into_iter(),
            next_value: None,
        }
    }
}
impl<'de> MapAccess<'de> for MyMapAccess
where
    String: IntoDeserializer<'de>,
    Values: IntoDeserializer<'de> + Clone,
{
    type Error = serde::de::value::Error;
    fn next_key_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.iter.next() {
            Some((k, v)) => {
                self.next_value = Some(v);
                seed.deserialize(k.into_deserializer()).map(Some)
            }
            None => Ok(None),
        }
    }
    fn next_value_seed<T>(&mut self, seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.next_value.take() {
            Some(v) => seed.deserialize(v.into_deserializer()),
            None => Err(serde::de::Error::custom("value missing")),
        }
    }
}
impl<'de> IntoDeserializer<'de, de::value::Error> for Values {
    type Deserializer = ValuesDeserializer;
    fn into_deserializer(self) -> Self::Deserializer {
        ValuesDeserializer::new(self)
    }
}
fn main() {
    let mut struct3_map = HashMap::new();
    struct3_map.insert("x".to_string(), Values::StringValue("Hello".to_string()));
    struct3_map.insert("y".to_string(), Values::StringValue("World".to_string()));
    let mut struct2_map = HashMap::new();
    struct2_map.insert("a".to_string(), Values::IntValue(1));
    struct2_map.insert("b".to_string(), Values::IntValue(2));
    let mut struct1_map = HashMap::new();
    struct1_map.insert("a".to_string(), Values::StructValue(struct3_map));
    struct1_map.insert("b".to_string(), Values::StructValue(struct2_map));
    let values = Values::StructValue(struct1_map);
    let deserializer = ValuesDeserializer::new(values);
    let result = MyStruct1::deserialize(deserializer);
    match result {
        Ok(my_struct1) => {
            ::std::io::_print(
                format_args!("Deserialized MyStruct1: {0:?}\n", my_struct1),
            );
        }
        Err(e) => {
            ::std::io::_print(format_args!("Failed to deserialize: {0:?}\n", e));
        }
    }
}
