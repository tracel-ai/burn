use super::Variable;

macro_rules! gpu {
    // out = lhs + rhs
    ($scope:expr, $out:ident = $lhs:ident + $rhs:expr) => {
        gpu!($scope, $out = add($lhs, $rhs))
    };
    // out += input
    ($scope:expr, $out:ident += $input:ident) => {
        gpu!($scope, $out = add($out, $input))
    };
    // out = add(lhs, rhs)
    ($scope:expr, $out:ident = add($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Add(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs - rhs
    ($scope:expr, $out:ident = $lhs:ident - $rhs:expr) => {
        gpu!($scope, $out = sub($lhs, $rhs))
    };
    // out = sub(lhs, rhs)
    ($scope:expr, $out:ident = sub($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Sub(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs * rhs
    ($scope:expr, $out:ident = $lhs:ident * $rhs:expr) => {
        gpu!($scope, $out = mul($lhs, $rhs))
    };
    // out *= input
    ($scope:expr, $out:ident *= $input:ident) => {
        gpu!($scope, $out = mul($out, $input))
    };
    // out = mul(lhs, rhs)
    ($scope:expr, $out:ident = mul($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Mul(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs / rhs
    ($scope:expr, $out:ident = $lhs:ident / $rhs:expr) => {
        gpu!($scope, $out = div($lhs, $rhs))
    };
    // out = div(lhs, rhs)
    ($scope:expr, $out:ident = div($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Div(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs % rhs
    ($scope:expr, $out:ident = $lhs:ident % $rhs:expr) => {
        gpu!($scope, $out = modulo($lhs, $rhs))
    };
    // out = modulo(lhs, rhs)
    ($scope:expr, $out:ident = modulo($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Modulo(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = powf(lhs, rhs)
    ($scope:expr, $out:ident = powf($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Powf(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs && rhs
    ($scope:expr, $out:ident = $lhs:ident && $rhs:expr) => {
        gpu!($scope, $out = and($lhs, $rhs))
    };
    // out = and(lhs, rhs)
    ($scope:expr, $out:ident = and($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::And(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs || rhs
    ($scope:expr, $out:ident = $lhs:ident || $rhs:expr) => {
        gpu!($scope, $out = or($lhs, $rhs))
    };
    // out = or(lhs, rhs)
    ($scope:expr, $out:ident = or($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Or(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = !input
    ($scope:expr, $out:ident = !$input:expr) => {
        gpu!($scope, $out = not($input))
    };
    // out = not(input)
    ($scope:expr, $out:ident = not($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Not(
            gpu!(unary $input, $out)
        ));
    };
    // out = lhs & rhs
    ($scope:expr, $out: ident = $lhs:ident & $rhs:ident) => {
        gpu!($scope, $out = bitwise_and($lhs, $rhs))
    };
    // out = bitwise_and(lhs, rhs)
    ($scope:expr, $out:ident = bitwise_and($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::BitwiseAnd(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs ^ rhs
    ($scope:expr, $out: ident = $lhs:ident ^ $rhs:ident) => {
        gpu!($scope, $out = bitwise_xor($lhs, $rhs))
    };
    // out = bitwise_xor(lhs, rhs)
    ($scope:expr, $out:ident = bitwise_xor($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::BitwiseXor(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs << rhs
    ($scope:expr, $out: ident = $lhs:ident << $rhs:ident) => {
        gpu!($scope, $out = shift_left($lhs, $rhs))
    };
    // out = shift_left(lhs, rhs)
    ($scope:expr, $out:ident = shift_left($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::ShiftLeft(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs >> rhs
    ($scope:expr, $out: ident = $lhs:ident >> $rhs:ident) => {
        gpu!($scope, $out = shift_right($lhs, $rhs))
    };
    // out = shift_right(lhs, rhs)
    ($scope:expr, $out:ident = shift_right($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::ShiftRight(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs == rhs
    ($scope:expr, $out:ident = $lhs:ident == $rhs:expr) => {
        gpu!($scope, $out = equal($lhs, $rhs))
    };
    // out = equal(lhs, rhs)
    ($scope:expr, $out:ident = equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Equal(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs != rhs
    ($scope:expr, $out:ident = $lhs:ident != $rhs:expr) => {
        gpu!($scope, $out = not_equal($lhs, $rhs))
    };
    // out = not_equal(lhs, rhs)
    ($scope:expr, $out:ident = not_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::NotEqual(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs > rhs
    ($scope:expr, $out:ident = $lhs:ident > $rhs:expr) => {
        gpu!($scope, $out = greater($lhs, $rhs))
    };
    // out = greater(lhs, rhs)
    ($scope:expr, $out:ident = greater($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Greater(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs >= rhs
    ($scope:expr, $out:ident = $lhs:ident >= $rhs:expr) => {
        gpu!($scope, $out = greater_equal($lhs, $rhs))
    };
    // out = greater_equal(lhs, rhs)
    ($scope:expr, $out:ident = greater_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::GreaterEqual(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs < rhs
    ($scope:expr, $out:ident = $lhs:ident < $rhs:expr) => {
        gpu!($scope, $out = lower($lhs, $rhs))
    };
    // out = lower(lhs, rhs)
    ($scope:expr, $out:ident = lower($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Lower(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs <= rhs
    ($scope:expr, $out:ident = $lhs:ident <= $rhs:expr) => {
        gpu!($scope, $out = lower_equal($lhs, $rhs))
    };
    // out = lower_equal(lhs, rhs)
    ($scope:expr, $out:ident = lower_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::LowerEqual(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = max(lhs, rhs)
    ($scope:expr, $out:ident = max($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Max(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = min(lhs, rhs)
    ($scope:expr, $out:ident = min($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Min(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs[rhs]
    ($scope:expr, $out:ident = $lhs:ident[$rhs:expr]) => {
        gpu!($scope, $out = index($lhs, $rhs))
    };
    // out = index(lhs, rhs)
    ($scope:expr, $out:ident = index($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Index(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out[lhs] = rhs
    ($scope:expr, $out:ident[$lhs:ident] = $rhs:expr) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::IndexAssign(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = |input|
    ($scope:expr, $out:ident = |$input:ident|) => {
        gpu!($scope, $out = abs($input))
    };
    // out = abs(input)
    ($scope:expr, $out:ident = abs($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Abs(
            gpu!(unary $input, $out)
        ));
    };
    // out = exp(input)
    ($scope:expr, $out:ident = exp($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Exp(
            gpu!(unary $input, $out)
        ));
    };
    // out = log(input)
    ($scope:expr, $out:ident = log($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Log(
            gpu!(unary $input, $out)
        ));
    };
    // out = log1p(input)
    ($scope:expr, $out:ident = log1p($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Log1p(
            gpu!(unary $input, $out)
        ));
    };
    // out = cos(input)
    ($scope:expr, $out:ident = cos($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Cos(
            gpu!(unary $input, $out)
        ));
    };
    // out = sin(input)
    ($scope:expr, $out:ident = sin($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Sin(
            gpu!(unary $input, $out)
        ));
    };
    // out = tanh(input)
    ($scope:expr, $out:ident = tanh($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Tanh(
            gpu!(unary $input, $out)
        ));
    };
    // out = sqrt(input)
    ($scope:expr, $out:ident = sqrt($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Sqrt(
            gpu!(unary $input, $out)
        ));
    };
    // out = ceil(input)
    ($scope:expr, $out:ident = ceil($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Ceil(
            gpu!(unary $input, $out)
        ));
    };
    // out = erf(input)
    ($scope:expr, $out:ident = erf($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Erf(
            gpu!(unary $input, $out)
        ));
    };
    // out = input
    ($scope:expr, $out:ident = $input:ident) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Assign(
            gpu!(unary $input, $out)
        ));
    };
    // out = vec4(a, b, c, d)
    ($scope:expr, $out:ident = vec4($a:ident,$b:ident,$c:ident,$d:ident)) => {
        let i = $scope.zero(Elem::UInt);
        gpu!($scope, $out[i] = $a);
        gpu!($scope, i = i + 1u32);
        gpu!($scope, $out[i] = $b);
        gpu!($scope, i = i + 1u32);
        gpu!($scope, $out[i] = $c);
        gpu!($scope, i = i + 1u32);
        gpu!($scope, $out[i] = $d);
    };
    // out = input
    ($scope:expr, $out:ident = $input:ident) => {
        gpu!($scope, $out = cast($input))
    };
    // out = cast(input)
    ($scope:expr, $out:ident = cast($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Assign(
            gpu!(unary $input, $out)
        ));
    };
    // out = shape(tensor, dim)
    ($scope:expr, $out:ident = shape($input:expr, $dim:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Metadata::Shape {
            dim: $dim.into(),
            var: $input.into(),
            out: $out.into(),
        });
    };
    // out = stride(tensor, dim)
    ($scope:expr, $out:ident = stride($input:expr, $dim:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Metadata::Stride {
            dim: $dim.into(),
            var: $input.into(),
            out: $out.into(),
        });
    };
    // out = len(array)
    ($scope:expr, $out:ident = len($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Metadata::ArrayLength {
            var: $input.into(),
            out: $out.into(),
        });
    };
    // range(start, end).for_each(|i, scope| { ... })
    ($scope:expr, range($start:expr, $end:expr).for_each($arg:expr)) => {
        $crate::codegen::dialect::gpu::RangeLoop::register($scope, $start.into(), $end.into(), $arg);
    };
    // range(start, end, unroll).for_each(|i, scope| { ... })
    ($scope:expr, range($start:expr, $end:expr, $unroll:expr).for_each($arg:expr)) => {
        if $unroll {
            $crate::codegen::dialect::gpu::UnrolledRangeLoop::register($scope, $start.into(), $end.into(), $arg);
        } else {
            $crate::codegen::dialect::gpu::RangeLoop::register($scope, $start.into(), $end.into(), $arg);
        }
    };
    // loop(|scope| { ... })
    ($scope:expr, loop($arg:expr)) => {
        $crate::codegen::dialect::gpu::Loop::register($scope, $arg);
    };
    // if (cond).then(|scope| { ... })
    ($scope:expr, if ($cond:expr).then($arg:expr)) => {
        $crate::codegen::dialect::gpu::If::register($scope, $cond.into(), $arg);
    };
    // if (cond).then(|scope| { ... }).else(|scope| { ... })
    ($scope:expr, if ($cond:expr).then($arg_if:expr).else($arg_else:expr)) => {
        $crate::codegen::dialect::gpu::IfElse::register($scope, $cond.into(), $arg_if, $arg_else);
    };
    (binary $lhs:expr, $rhs:expr, $out:expr) => {
        $crate::codegen::dialect::gpu::BinaryOperator {
            lhs: $lhs.into(),
            rhs: $rhs.into(),
            out: $out.into(),
        }
    };
    (unary $input:expr, $out:expr) => {
        $crate::codegen::dialect::gpu::UnaryOperator {
            input: $input.into(),
            out: $out.into(),
        }
    };
}

impl From<bool> for Variable {
    fn from(value: bool) -> Self {
        Self::ConstantScalar(if value { 1.0 } else { 0.0 }, super::Elem::Bool)
    }
}

impl From<i32> for Variable {
    fn from(value: i32) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::Int)
    }
}

impl From<f32> for Variable {
    fn from(value: f32) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::Float)
    }
}

impl From<u32> for Variable {
    fn from(value: u32) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::UInt)
    }
}

impl From<usize> for Variable {
    fn from(value: usize) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::UInt)
    }
}

pub(crate) use gpu;
