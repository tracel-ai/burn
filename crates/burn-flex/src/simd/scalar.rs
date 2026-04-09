/// Comparison operation type.
#[derive(Clone, Copy)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// Scalar boolean NOT: out[i] = !a[i] (0 becomes 1, non-zero becomes 0)
#[inline]
pub fn bool_not_u8(a: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = (a[i] == 0) as u8;
    }
}

/// Scalar boolean NOT in-place: a[i] = !a[i]
#[inline]
pub fn bool_not_inplace_u8(a: &mut [u8]) {
    for x in a.iter_mut() {
        *x = (*x == 0) as u8;
    }
}

/// Scalar boolean AND: out[i] = a[i] & b[i]
#[inline]
pub fn bool_and_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = a[i] & b[i];
    }
}

/// Scalar boolean OR: out[i] = a[i] | b[i]
#[inline]
pub fn bool_or_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = a[i] | b[i];
    }
}

/// Scalar boolean XOR: out[i] = a[i] ^ b[i]
#[inline]
pub fn bool_xor_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = a[i] ^ b[i];
    }
}

/// Scalar boolean AND in-place: a[i] &= b[i]
#[inline]
pub fn bool_and_inplace_u8(a: &mut [u8], b: &[u8]) {
    for i in 0..a.len() {
        a[i] &= b[i];
    }
}

/// Scalar boolean OR in-place: a[i] |= b[i]
#[inline]
pub fn bool_or_inplace_u8(a: &mut [u8], b: &[u8]) {
    for i in 0..a.len() {
        a[i] |= b[i];
    }
}

/// Scalar boolean XOR in-place: a[i] ^= b[i]
#[inline]
pub fn bool_xor_inplace_u8(a: &mut [u8], b: &[u8]) {
    for i in 0..a.len() {
        a[i] ^= b[i];
    }
}
