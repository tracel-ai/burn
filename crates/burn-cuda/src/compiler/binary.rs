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
        let out2 = out.index(3);
        let out3 = out.index(2);

        Self::format_scalar(f, lhs0, rhs0, out0, elem)?;
        Self::format_scalar(f, lhs1, rhs1, out1, elem)?;
        Self::format_scalar(f, lhs2, rhs2, out2, elem)?;
        Self::format_scalar(f, lhs3, rhs3, out3, elem)?;

        Ok(())
    }
}

macro_rules! operator {
    ($name:ident, $op:expr) => {
        operator!($name, $op, |_elem| InstructionSettings {
            native_vec4: true,
            native_vec3: true,
            native_vec2: true,
        });
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

            fn settings(elem: Elem) -> InstructionSettings {
                $vectorization(elem)
            }
        }
    };
}

macro_rules! function {
    ($name:ident, $op:expr) => {
        function!($name, $op, |_elem| InstructionSettings {
            native_vec4: true,
            native_vec3: true,
            native_vec2: true,
        });
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

            fn settings(elem: Elem) -> InstructionSettings {
                $vectorization(elem)
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
        // Cast only when necessary.
        if elem != rhs.elem() {
            f.write_fmt(format_args!("{out}[{lhs}] = {elem}({rhs});\n"))
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

        f.write_fmt(format_args!("{out}[{lhs0}] = {elem}({rhs0});\n"))?;
        f.write_fmt(format_args!("{out}[{lhs1}] = {elem}({rhs1});\n"))
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

        f.write_fmt(format_args!("{out}[{lhs0}] = {elem}({rhs0});\n"))?;
        f.write_fmt(format_args!("{out}[{lhs1}] = {elem}({rhs1});\n"))?;
        f.write_fmt(format_args!("{out}[{lhs2}] = {elem}({rhs2});\n"))
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

        f.write_fmt(format_args!("{out}[{lhs0}] = {elem}({rhs0});\n"))?;
        f.write_fmt(format_args!("{out}[{lhs1}] = {elem}({rhs1});\n"))?;
        f.write_fmt(format_args!("{out}[{lhs2}] = {elem}({rhs2});\n"))?;
        f.write_fmt(format_args!("{out}[{lhs3}] = {elem}({rhs3});\n"))
    }

    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        match lhs.item() {
            Item::Vec4(elem) => Self::unroll_vec4(f, lhs, rhs, out, elem),
            Item::Vec3(elem) => Self::unroll_vec3(f, lhs, rhs, out, elem),
            Item::Vec2(elem) => Self::unroll_vec2(f, lhs, rhs, out, elem),
            Item::Scalar(elem) => Self::format_scalar(f, *lhs, *rhs, *out, elem),
        }
    }
}
