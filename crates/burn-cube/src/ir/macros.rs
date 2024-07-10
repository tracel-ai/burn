use super::Variable;

#[macro_export(local_inner_macros)]
/// Cube Pseudo Assembly.
macro_rules! cpa {
    // out = lhs + rhs
    ($scope:expr, $out:ident = $lhs:ident + $rhs:expr) => {
        cpa!($scope, $out = add($lhs, $rhs))
    };
    // out += input
    ($scope:expr, $out:ident += $input:ident) => {
        cpa!($scope, $out = add($out, $input))
    };
    // out = add(lhs, rhs)
    ($scope:expr, $out:ident = add($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Add(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs - rhs
    ($scope:expr, $out:ident = $lhs:ident - $rhs:expr) => {
        cpa!($scope, $out = sub($lhs, $rhs))
    };
    // out = sub(lhs, rhs)
    ($scope:expr, $out:ident = sub($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Sub(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs * rhs
    ($scope:expr, $out:ident = $lhs:ident * $rhs:expr) => {
        cpa!($scope, $out = mul($lhs, $rhs))
    };
    // out *= input
    ($scope:expr, $out:ident *= $input:ident) => {
        cpa!($scope, $out = mul($out, $input))
    };
    // out = mul(lhs, rhs)
    ($scope:expr, $out:ident = mul($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Mul(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs / rhs
    ($scope:expr, $out:ident = $lhs:ident / $rhs:expr) => {
        cpa!($scope, $out = div($lhs, $rhs))
    };
    // out = div(lhs, rhs)
    ($scope:expr, $out:ident = div($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Div(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs % rhs
    ($scope:expr, $out:ident = $lhs:ident % $rhs:expr) => {
        cpa!($scope, $out = modulo($lhs, $rhs))
    };
    // out = modulo(lhs, rhs)
    ($scope:expr, $out:ident = modulo($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Modulo(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = powf(lhs, rhs)
    ($scope:expr, $out:ident = powf($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Powf(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs && rhs
    ($scope:expr, $out:ident = $lhs:ident && $rhs:expr) => {
        cpa!($scope, $out = and($lhs, $rhs))
    };
    // out = and(lhs, rhs)
    ($scope:expr, $out:ident = and($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::And(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs || rhs
    ($scope:expr, $out:ident = $lhs:ident || $rhs:expr) => {
        cpa!($scope, $out = or($lhs, $rhs))
    };
    // out = or(lhs, rhs)
    ($scope:expr, $out:ident = or($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Or(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = !input
    ($scope:expr, $out:ident = !$input:expr) => {
        cpa!($scope, $out = not($input))
    };
    // out = not(input)
    ($scope:expr, $out:ident = not($input:expr)) => {
        $scope.register($crate::ir::Operator::Not(
            cpa!(unary $input, $out)
        ));
    };
    // out = lhs & rhs
    ($scope:expr, $out: ident = $lhs:ident & $rhs:ident) => {
        cpa!($scope, $out = bitwise_and($lhs, $rhs))
    };
    // out = bitwise_and(lhs, rhs)
    ($scope:expr, $out:ident = bitwise_and($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::BitwiseAnd(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs ^ rhs
    ($scope:expr, $out: ident = $lhs:ident ^ $rhs:ident) => {
        cpa!($scope, $out = bitwise_xor($lhs, $rhs))
    };
    // out = bitwise_xor(lhs, rhs)
    ($scope:expr, $out:ident = bitwise_xor($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::BitwiseXor(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs << rhs
    ($scope:expr, $out: ident = $lhs:ident << $rhs:ident) => {
        cpa!($scope, $out = shift_left($lhs, $rhs))
    };
    // out = shift_left(lhs, rhs)
    ($scope:expr, $out:ident = shift_left($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::ShiftLeft(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs >> rhs
    ($scope:expr, $out: ident = $lhs:ident >> $rhs:ident) => {
        cpa!($scope, $out = shift_right($lhs, $rhs))
    };
    // out = shift_right(lhs, rhs)
    ($scope:expr, $out:ident = shift_right($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::ShiftRight(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs == rhs
    ($scope:expr, $out:ident = $lhs:ident == $rhs:expr) => {
        cpa!($scope, $out = equal($lhs, $rhs))
    };
    // out = equal(lhs, rhs)
    ($scope:expr, $out:ident = equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Equal(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs != rhs
    ($scope:expr, $out:ident = $lhs:ident != $rhs:expr) => {
        cpa!($scope, $out = not_equal($lhs, $rhs))
    };
    // out = not_equal(lhs, rhs)
    ($scope:expr, $out:ident = not_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::NotEqual(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs > rhs
    ($scope:expr, $out:ident = $lhs:ident > $rhs:expr) => {
        cpa!($scope, $out = greater($lhs, $rhs))
    };
    // out = greater(lhs, rhs)
    ($scope:expr, $out:ident = greater($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Greater(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs >= rhs
    ($scope:expr, $out:ident = $lhs:ident >= $rhs:expr) => {
        cpa!($scope, $out = greater_equal($lhs, $rhs))
    };
    // out = greater_equal(lhs, rhs)
    ($scope:expr, $out:ident = greater_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::GreaterEqual(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs < rhs
    ($scope:expr, $out:ident = $lhs:ident < $rhs:expr) => {
        cpa!($scope, $out = lower($lhs, $rhs))
    };
    // out = lower(lhs, rhs)
    ($scope:expr, $out:ident = lower($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Lower(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs <= rhs
    ($scope:expr, $out:ident = $lhs:ident <= $rhs:expr) => {
        cpa!($scope, $out = lower_equal($lhs, $rhs))
    };
    // out = lower_equal(lhs, rhs)
    ($scope:expr, $out:ident = lower_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::LowerEqual(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = max(lhs, rhs)
    ($scope:expr, $out:ident = max($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Max(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = min(lhs, rhs)
    ($scope:expr, $out:ident = min($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Min(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs[rhs]
    ($scope:expr, $out:ident = $lhs:ident[$rhs:expr]) => {
        cpa!($scope, $out = index($lhs, $rhs))
    };
    // out = index(lhs, rhs)
    ($scope:expr, $out:ident = index($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::ir::Operator::Index(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = unchecked(lhs[rhs])
    ($scope:expr, $out:ident = unchecked($lhs:ident[$rhs:expr])) => {
        $scope.register($crate::ir::Operator::UncheckedIndex(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out[lhs] = rhs
    ($scope:expr, $out:ident[$lhs:ident] = $rhs:expr) => {
        $scope.register($crate::ir::Operator::IndexAssign(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // unchecked(out[lhs]) = rhs
    ($scope:expr, unchecked($out:ident[$lhs:ident]) = $rhs:expr) => {
        $scope.register($crate::ir::Operator::UncheckedIndexAssign(
            cpa!(binary $lhs, $rhs, $out)
        ));
    };
    // out = |input|
    ($scope:expr, $out:ident = |$input:ident|) => {
        cpa!($scope, $out = abs($input))
    };
    // out = abs(input)
    ($scope:expr, $out:ident = abs($input:expr)) => {
        $scope.register($crate::ir::Operator::Abs(
            cpa!(unary $input, $out)
        ));
    };
    // out = exp(input)
    ($scope:expr, $out:ident = exp($input:expr)) => {
        $scope.register($crate::ir::Operator::Exp(
            cpa!(unary $input, $out)
        ));
    };
    // out = log(input)
    ($scope:expr, $out:ident = log($input:expr)) => {
        $scope.register($crate::ir::Operator::Log(
            cpa!(unary $input, $out)
        ));
    };
    // out = log1p(input)
    ($scope:expr, $out:ident = log1p($input:expr)) => {
        $scope.register($crate::ir::Operator::Log1p(
            cpa!(unary $input, $out)
        ));
    };
    // out = cos(input)
    ($scope:expr, $out:ident = cos($input:expr)) => {
        $scope.register($crate::ir::Operator::Cos(
            cpa!(unary $input, $out)
        ));
    };
    // out = sin(input)
    ($scope:expr, $out:ident = sin($input:expr)) => {
        $scope.register($crate::ir::Operator::Sin(
            cpa!(unary $input, $out)
        ));
    };
    // out = tanh(input)
    ($scope:expr, $out:ident = tanh($input:expr)) => {
        $scope.register($crate::ir::Operator::Tanh(
            cpa!(unary $input, $out)
        ));
    };
    // out = sqrt(input)
    ($scope:expr, $out:ident = sqrt($input:expr)) => {
        $scope.register($crate::ir::Operator::Sqrt(
            cpa!(unary $input, $out)
        ));
    };
    // out = floor(input)
    ($scope:expr, $out:ident = floor($input:expr)) => {
        $scope.register($crate::ir::Operator::Floor(
            cpa!(unary $input, $out)
        ));
    };
    // out = ceil(input)
    ($scope:expr, $out:ident = ceil($input:expr)) => {
        $scope.register($crate::ir::Operator::Ceil(
            cpa!(unary $input, $out)
        ));
    };
    // out = erf(input)
    ($scope:expr, $out:ident = erf($input:expr)) => {
        $scope.register($crate::ir::Operator::Erf(
            cpa!(unary $input, $out)
        ));
    };
    // out = input
    ($scope:expr, $out:ident = $input:ident) => {
        $scope.register($crate::ir::Operator::Assign(
            cpa!(unary $input, $out)
        ));
    };
    // out = vec4(a, b, c, d)
    ($scope:expr, $out:ident = vec4($a:ident,$b:ident,$c:ident,$d:ident)) => {
        let i = $scope.zero(Elem::UInt);
        cpa!($scope, $out[i] = $a);
        cpa!($scope, i = i + 1u32);
        cpa!($scope, $out[i] = $b);
        cpa!($scope, i = i + 1u32);
        cpa!($scope, $out[i] = $c);
        cpa!($scope, i = i + 1u32);
        cpa!($scope, $out[i] = $d);
    };
    // out = input
    ($scope:expr, $out:ident = $input:ident) => {
        cpa!($scope, $out = cast($input))
    };
    // out = cast(input)
    ($scope:expr, $out:ident = cast($input:expr)) => {
        $scope.register($crate::ir::Operator::Assign(
            cpa!(unary $input, $out)
        ));
    };
    // out = shape(tensor, dim)
    ($scope:expr, $out:ident = shape($input:expr, $dim:expr)) => {
        $scope.register($crate::ir::Metadata::Shape {
            dim: $dim.into(),
            var: $input.into(),
            out: $out.into(),
        });
    };
    // out = stride(tensor, dim)
    ($scope:expr, $out:ident = stride($input:expr, $dim:expr)) => {
        $scope.register($crate::ir::Metadata::Stride {
            dim: $dim.into(),
            var: $input.into(),
            out: $out.into(),
        });
    };
    // out = len(array)
    ($scope:expr, $out:ident = len($input:expr)) => {
        $scope.register($crate::ir::Metadata::Length {
            var: $input.into(),
            out: $out.into(),
        });
    };
    // range(start, end).for_each(|i, scope| { ... })
    ($scope:expr, range($start:expr, $end:expr).for_each($arg:expr)) => {
        $crate::ir::RangeLoop::register($scope, $start.into(), $end.into(), $arg);
    };
    // range(start, end, unroll).for_each(|i, scope| { ... })
    ($scope:expr, range($start:expr, $end:expr, $unroll:expr).for_each($arg:expr)) => {
        if $unroll {
            $crate::ir::UnrolledRangeLoop::register($scope, $start.into(), $end.into(), $arg);
        } else {
            $crate::ir::RangeLoop::register($scope, $start.into(), $end.into(), $arg);
        }
    };
    // loop(|scope| { ... })
    ($scope:expr, loop($arg:expr)) => {
        $crate::ir::Loop::register($scope, $arg);
    };
    // if (cond).then(|scope| { ... })
    ($scope:expr, if ($cond:expr).then($arg:expr)) => {
        $crate::ir::If::register($scope, $cond.into(), $arg);
    };
    // if (cond).then(|scope| { ... }).else(|scope| { ... })
    ($scope:expr, if ($cond:expr).then($arg_if:expr).else($arg_else:expr)) => {
        $crate::ir::IfElse::register($scope, $cond.into(), $arg_if, $arg_else);
    };
    (binary $lhs:expr, $rhs:expr, $out:expr) => {
        $crate::ir::BinaryOperator {
            lhs: $lhs.into(),
            rhs: $rhs.into(),
            out: $out.into(),
        }
    };
    (unary $input:expr, $out:expr) => {
        $crate::ir::UnaryOperator {
            input: $input.into(),
            out: $out.into(),
        }
    };
}

impl From<bool> for Variable {
    fn from(value: bool) -> Self {
        Self::ConstantScalar {
            value: if value { 1.0 } else { 0.0 },
            elem: super::Elem::Bool,
        }
    }
}

impl From<i32> for Variable {
    fn from(value: i32) -> Self {
        Self::ConstantScalar {
            value: value as f64,
            elem: super::Elem::Int(super::IntKind::I32),
        }
    }
}

impl From<i64> for Variable {
    fn from(value: i64) -> Self {
        Self::ConstantScalar {
            value: value as f64,
            elem: super::Elem::Int(super::IntKind::I64),
        }
    }
}

impl From<f32> for Variable {
    fn from(value: f32) -> Self {
        Self::ConstantScalar {
            value: value as f64,
            elem: super::Elem::Float(super::FloatKind::F32),
        }
    }
}

impl From<f64> for Variable {
    fn from(value: f64) -> Self {
        Self::ConstantScalar {
            value,
            elem: super::Elem::Float(super::FloatKind::F64),
        }
    }
}

impl From<u32> for Variable {
    fn from(value: u32) -> Self {
        Self::ConstantScalar {
            value: value as f64,
            elem: super::Elem::UInt,
        }
    }
}

impl From<usize> for Variable {
    fn from(value: usize) -> Self {
        Self::ConstantScalar {
            value: value as f64,
            elem: super::Elem::UInt,
        }
    }
}

pub(crate) use cpa;
