use super::{Component, Elem, InstructionSettings, Item, Variable};
use std::fmt::Display;

pub trait Binary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item = out.item();
        let settings = Self::settings(*item.elem());

        match item {
            Item::Vec4(elem) => {
                if settings.native_vec4 && lhs.item() == rhs.item() {
                    Self::format_native_vec4(f, lhs, rhs, out, elem)
                } else {
                    Self::unroll_vec4(f, lhs, rhs, out, elem)
                }
            }
            Item::Vec3(elem) => {
                if settings.native_vec3 && lhs.item() == rhs.item() {
                    Self::format_native_vec3(f, lhs, rhs, out, elem)
                } else {
                    Self::unroll_vec3(f, lhs, rhs, out, elem)
                }
            }
            Item::Vec2(elem) => {
                if settings.native_vec2 && lhs.item() == rhs.item() {
                    Self::format_native_vec2(f, lhs, rhs, out, elem)
                } else {
                    Self::unroll_vec2(f, lhs, rhs, out, elem)
                }
            }
            Item::Scalar(elem) => Self::format_scalar(f, *lhs, *rhs, *out, elem),
        }
    }

    fn settings(_elem: Elem) -> InstructionSettings {
        InstructionSettings::default()
    }

    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component;

    fn format_native_vec4(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *lhs, *rhs, *out, elem)
    }

    fn format_native_vec3(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *lhs, *rhs, *out, elem)
    }

    fn format_native_vec2(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *lhs, *rhs, *out, elem)
    }

    fn unroll_vec2(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);

        let out0 = out.index(0);
        let out1 = out.index(1);

        Self::format_scalar(f, lhs0, rhs0, out0, elem)?;
        Self::format_scalar(f, lhs1, rhs1, out1, elem)?;

        Ok(())
    }

    fn unroll_vec3(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);
        let lhs2 = lhs.index(2);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);
        let rhs2 = rhs.index(2);

        let out0 = out.index(0);
        let out1 = out.index(1);
        let out2 = out.index(2);

        Self::format_scalar(f, lhs0, rhs0, out0, elem)?;
        Self::format_scalar(f, lhs1, rhs1, out1, elem)?;
        Self::format_scalar(f, lhs2, rhs2, out2, elem)?;

        Ok(())
    }

    fn unroll_vec4(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);
        let lhs2 = lhs.index(2);
        let lhs3 = lhs.index(3);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);
        let rhs2 = rhs.index(2);
        let rhs3 = rhs.index(3);

        let out0 = out.index(0);
        let out1 = out.index(1);
        let out2 = out.index(2);
        let out3 = out.index(3);

        Self::format_scalar(f, lhs0, rhs0, out0, elem)?;
        Self::format_scalar(f, lhs1, rhs1, out1, elem)?;
        Self::format_scalar(f, lhs2, rhs2, out2, elem)?;
        Self::format_scalar(f, lhs3, rhs3, out3, elem)?;

        Ok(())
    }
}

macro_rules! operator {
    ($name:ident, $op:expr) => {
        operator!(
            $name,
            $op,
            InstructionSettings {
                native_vec4: false,
                native_vec3: false,
                native_vec2: false,
            }
        );
    };
    ($name:ident, $op:expr, $vectorization:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {lhs} {} {rhs};\n", $op))
            }

            #[allow(unused_variables)]
            fn settings(elem: Elem) -> InstructionSettings {
                $vectorization
            }
        }
    };
}

macro_rules! function {
    ($name:ident, $op:expr) => {
        function!(
            $name,
            $op,
            InstructionSettings {
                native_vec4: false,
                native_vec3: false,
                native_vec2: true,
            }
        );
    };
    ($name:ident, $op:expr, $vectorization:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_scalar<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {}({lhs}, {rhs});\n", $op))
            }

            #[allow(unused_variables)]
            fn settings(elem: Elem) -> InstructionSettings {
                $vectorization
            }
        }
    };
}

operator!(Add, "+");
operator!(Sub, "-");
operator!(Div, "/");
operator!(Mul, "*");
operator!(Modulo, "%");
operator!(Equal, "==");
operator!(NotEqual, "!=");
operator!(Lower, "<");
operator!(LowerEqual, "<=");
operator!(Greater, ">");
operator!(GreaterEqual, ">=");
operator!(ShiftLeft, "<<");
operator!(ShiftRight, ">>");
operator!(BitwiseAnd, "&");
operator!(BitwiseXor, "^");
operator!(Or, "||");
operator!(And, "&&");

function!(Powf, "powf");
function!(Max, "max");
function!(Min, "min");

pub struct IndexAssign;
pub struct Index;

impl Binary for IndexAssign {
    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component,
    {
        let elem_rhs = rhs.elem();
        // Cast only when necessary.
        if elem != elem_rhs {
            if let Elem::Bool = elem_rhs {
                match rhs.item() {
                    Item::Vec4(_) => {
                        f.write_fmt(format_args!("{out}[{lhs}] = make_uint4({elem}({rhs}.x), {elem}({rhs}.y), {elem}({rhs}.z), {elem}({rhs}.w));\n"))
                    },
                    Item::Vec3(_) => todo!(),
                    Item::Vec2(_) => todo!(),
                    Item::Scalar(_) => todo!(),
                }
            } else {
                f.write_fmt(format_args!("{out}[{lhs}] = {elem}({rhs});\n"))
            }
        } else {
            f.write_fmt(format_args!("{out}[{lhs}] = {rhs};\n"))
        }
    }

    fn unroll_vec2(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);

        Self::format_scalar(f, lhs0, rhs0, *out, elem)?;
        Self::format_scalar(f, lhs1, rhs1, *out, elem)?;

        Ok(())
    }

    fn unroll_vec3(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);
        let lhs2 = lhs.index(2);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);
        let rhs2 = rhs.index(2);

        Self::format_scalar(f, lhs0, rhs0, *out, elem)?;
        Self::format_scalar(f, lhs1, rhs1, *out, elem)?;
        Self::format_scalar(f, lhs2, rhs2, *out, elem)?;

        Ok(())
    }

    fn unroll_vec4(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);
        let lhs2 = lhs.index(2);
        let lhs3 = lhs.index(3);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);
        let rhs2 = rhs.index(2);
        let rhs3 = rhs.index(3);

        Self::format_scalar(f, lhs0, rhs0, *out, elem)?;
        Self::format_scalar(f, lhs1, rhs1, *out, elem)?;
        Self::format_scalar(f, lhs2, rhs2, *out, elem)?;
        Self::format_scalar(f, lhs3, rhs3, *out, elem)?;

        Ok(())
    }

    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        if let Variable::Local {
            id: _,
            item: _,
            depth: _,
        } = out
        {
            return IndexAssignVector::format(f, lhs, rhs, out);
        };

        let elem = out.elem();

        match lhs.item() {
            Item::Vec4(_) => Self::unroll_vec4(f, lhs, rhs, out, elem),
            Item::Vec3(_) => Self::unroll_vec3(f, lhs, rhs, out, elem),
            Item::Vec2(_) => Self::unroll_vec2(f, lhs, rhs, out, elem),
            Item::Scalar(_) => Self::format_scalar(f, *lhs, *rhs, *out, elem),
        }
    }
}

impl Binary for Index {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        if let Variable::Local {
            id: _,
            item: _,
            depth: _,
        } = lhs
        {
            return IndexVector::format(f, lhs, rhs, out);
        }

        Self::format_scalar(f, *lhs, *rhs, *out, out.elem())
    }

    fn format_scalar<Lhs, Rhs, Out>(
        f: &mut std::fmt::Formatter<'_>,
        lhs: Lhs,
        rhs: Rhs,
        out: Out,
        _elem: Elem,
    ) -> std::fmt::Result
    where
        Lhs: Component,
        Rhs: Component,
        Out: Component,
    {
        f.write_fmt(format_args!("{out} = {lhs}[{rhs}];\n"))
    }
}

/// The goal is to support indexing of vectorized types.
///
/// # Examples
///
/// ```c
/// float4 rhs;
/// float item = var[0]; // We want that.
/// float item = var.x; // So we compile to that.
/// ```
struct IndexVector;

/// The goal is to support indexing of vectorized types.
///
/// # Examples
///
/// ```c
/// float4 var;
///
/// var[0] = 1.0; // We want that.
/// var.x = 1.0;  // So we compile to that.
/// ```
struct IndexAssignVector;

impl IndexVector {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let index = match rhs {
            Variable::ConstantScalar(value, _elem) => *value as usize,
            _ => {
                let elem = out.elem();
                return f.write_fmt(format_args!("{out} = *(({elem}*)&{lhs} + {rhs});\n"));
            }
        };

        let out = out.index(index);
        let lhs = lhs.index(index);

        f.write_fmt(format_args!("{out} = {lhs};\n"))
    }
}

impl IndexAssignVector {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let index = match lhs {
            Variable::ConstantScalar(value, _) => *value as usize,
            _ => {
                let elem = out.elem();
                return f.write_fmt(format_args!("*(({elem}*)&{out} + {lhs}) = {rhs};\n"));
            }
        };

        let out = out.index(index);
        let rhs = rhs.index(index);

        f.write_fmt(format_args!("{out} = {rhs};\n"))
    }
}
