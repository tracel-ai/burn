use super::{Component, Elem, Item, Variable};
use std::fmt::Display;

pub trait Binary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        match out.item() {
            Item::Vec4(elem) => Self::format_vec4(f, lhs, rhs, out, elem),
            Item::Vec3(elem) => Self::format_vec3(f, lhs, rhs, out, elem),
            Item::Vec2(elem) => Self::format_vec2(f, lhs, rhs, out, elem),
            Item::Scalar(elem) => Self::format_base(f, *lhs, *rhs, *out, elem),
        }
    }

    fn native_support_vec4(_elem: Elem) -> bool {
        false
    }

    fn native_support_vec3(_elem: Elem) -> bool {
        false
    }

    fn native_support_vec2(_elem: Elem) -> bool {
        false
    }

    fn format_base<Lhs, Rhs, Out>(
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

    fn format_vec2(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        if Self::native_support_vec2(elem) {
            if lhs.item() == rhs.item() {
                Self::format_base(f, *lhs, *rhs, *out, elem)?;
                return Ok(());
            }
        }

        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);

        let out0 = out.index(0);
        let out1 = out.index(1);

        Self::format_base(f, lhs0, rhs0, out0, elem)?;
        Self::format_base(f, lhs1, rhs1, out1, elem)?;

        Ok(())
    }

    fn format_vec3(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        if Self::native_support_vec3(elem) {
            if lhs.item() == rhs.item() {
                Self::format_base(f, *lhs, *rhs, *out, elem)?;
                return Ok(());
            }
        }

        let lhs0 = lhs.index(0);
        let lhs1 = lhs.index(1);
        let lhs2 = lhs.index(2);

        let rhs0 = rhs.index(0);
        let rhs1 = rhs.index(1);
        let rhs2 = rhs.index(2);

        let out0 = out.index(0);
        let out1 = out.index(1);
        let out2 = out.index(2);

        Self::format_base(f, lhs0, rhs0, out0, elem)?;
        Self::format_base(f, lhs1, rhs1, out1, elem)?;
        Self::format_base(f, lhs2, rhs2, out2, elem)?;

        Ok(())
    }

    fn format_vec4(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable,
        rhs: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        if Self::native_support_vec4(elem) {
            if lhs.item() == rhs.item() {
                Self::format_base(f, *lhs, *rhs, *out, elem)?;
                return Ok(());
            }
        }

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

        Self::format_base(f, lhs0, rhs0, out0, elem)?;
        Self::format_base(f, lhs1, rhs1, out1, elem)?;
        Self::format_base(f, lhs2, rhs2, out2, elem)?;
        Self::format_base(f, lhs3, rhs3, out3, elem)?;

        Ok(())
    }
}

macro_rules! operator {
    ($name:ident, $op:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn format_base<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {lhs} {} {rhs};\n", $op))
            }
        }
    };
    ($name:ident, $op:expr, $vec:expr) => {
        pub struct $name;

        impl Binary for $name {
            fn native_support_vec4(elem: Elem) -> bool {
                $vec(elem)
            }

            fn native_support_vec2(elem: Elem) -> bool {
                $vec(elem)
            }

            fn format_base<Lhs: Display, Rhs: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                lhs: Lhs,
                rhs: Rhs,
                out: Out,
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {lhs} {} {rhs};\n", $op))
            }
        }
    };
}

operator!(Add, "+", |elem| elem == Elem::F32);
operator!(Sub, "-");
operator!(Div, "/");
operator!(Mul, "*");
operator!(Modulo, "%");
operator!(Equal, "==");
operator!(Lower, "<");
operator!(LowerEqual, "<=");
operator!(Greater, ">");
operator!(GreaterEqual, ">=");

pub struct IndexAssign;

impl Binary for IndexAssign {
    fn format_base<Lhs, Rhs, Out>(
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

    fn format_vec2(
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

    fn format_vec3(
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

    fn format_vec4(
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
            Item::Vec4(elem) => Self::format_vec4(f, lhs, rhs, out, elem),
            Item::Vec3(elem) => Self::format_vec3(f, lhs, rhs, out, elem),
            Item::Vec2(elem) => Self::format_vec2(f, lhs, rhs, out, elem),
            Item::Scalar(elem) => Self::format_base(f, *lhs, *rhs, *out, elem),
        }
    }
}
