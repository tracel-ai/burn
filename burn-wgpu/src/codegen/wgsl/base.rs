use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum WgslVariable {
    Input(u16, WgslItem),
    Scalar(u16, WgslItem),
    Local(u16, WgslItem),
    Output(u16, WgslItem),
    Constant(f64, WgslItem),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum WgslElem {
    F32,
    I32,
    U32,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum WgslItem {
    Vec4(WgslElem),
    Vec3(WgslElem),
    Vec2(WgslElem),
    Scalar(WgslElem),
}

#[derive(Debug, Clone)]
pub struct IndexedWgslVariable {
    var: WgslVariable,
    index: usize,
}

impl WgslVariable {
    pub fn index(&self, index: usize) -> IndexedWgslVariable {
        IndexedWgslVariable {
            var: self.clone(),
            index,
        }
    }

    pub fn item(&self) -> &WgslItem {
        match self {
            Self::Input(_, e) => e,
            Self::Scalar(_, e) => e,
            Self::Local(_, e) => e,
            Self::Output(_, e) => e,
            Self::Constant(_, e) => e,
        }
    }
}

impl WgslItem {
    pub fn elem(&self) -> &WgslElem {
        match self {
            WgslItem::Vec4(e) => e,
            WgslItem::Vec3(e) => e,
            WgslItem::Vec2(e) => e,
            WgslItem::Scalar(e) => e,
        }
    }
}

impl WgslElem {
    pub fn size(&self) -> usize {
        match self {
            Self::F32 => core::mem::size_of::<f32>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }
}

impl Display for WgslElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::I32 => f.write_str("i32"),
            Self::U32 => f.write_str("u32"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

impl Display for WgslItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslItem::Vec4(elem) => f.write_fmt(format_args!("vec4<{elem}>")),
            WgslItem::Vec3(elem) => f.write_fmt(format_args!("vec3<{elem}>")),
            WgslItem::Vec2(elem) => f.write_fmt(format_args!("vec2<{elem}>")),
            WgslItem::Scalar(elem) => f.write_fmt(format_args!("{elem}")),
        }
    }
}

impl Display for WgslVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslVariable::Input(number, _) => f.write_fmt(format_args!("input_{number}")),
            WgslVariable::Local(number, _) => f.write_fmt(format_args!("local_{number}")),
            WgslVariable::Output(number, _) => f.write_fmt(format_args!("output_{number}")),
            WgslVariable::Scalar(number, item) => {
                f.write_fmt(format_args!("scalars_{item}[{number}]"))
            }
            WgslVariable::Constant(number, item) => match item {
                WgslItem::Vec4(elem) => f.write_fmt(format_args!(
                    "
vec4(
    {elem}({number}),
    {elem}({number}),
    {elem}({number}),
    {elem}({number}),
)"
                )),
                WgslItem::Vec3(elem) => f.write_fmt(format_args!(
                    "
vec3(
    {elem}({number}),
    {elem}({number}),
    {elem}({number}),
)"
                )),
                WgslItem::Vec2(elem) => f.write_fmt(format_args!(
                    "
vec2(
    {elem}({number}),
    {elem}({number}),
)"
                )),
                WgslItem::Scalar(elem) => f.write_fmt(format_args!("{elem}({number})")),
            },
        }
    }
}

impl Display for IndexedWgslVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let should_index = |item: &WgslItem| match item {
            WgslItem::Vec4(_) => true,
            WgslItem::Vec3(_) => true,
            WgslItem::Vec2(_) => true,
            WgslItem::Scalar(_) => false,
        };

        let var = &self.var;
        let item = var.item();
        let index = self.index;

        match self.var {
            WgslVariable::Scalar(_, _) => f.write_fmt(format_args!("{var}")),
            _ => match should_index(item) {
                true => f.write_fmt(format_args!("{var}[{index}]")),
                false => f.write_fmt(format_args!("{var}")),
            },
        }
    }
}
