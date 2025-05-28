pub type T = f32;
pub const PI: T = 3.141592653589793;
pub(crate) const SMIDGEN: T = 1e-6;
// These are used in fuzzy_cmp for tests
pub(crate) const CMP_SMIDGEN: T = 1e-2;
pub(crate) const MAX_FUZZY_CMP_DIFF: T = 2e-2;
pub use const_soft_float::soft_f32::SoftF32;
use const_soft_float::soft_f64::SoftF64;
pub use uom::si::f32::*;
#[derive(Copy, Clone)]
pub struct _SoftFloat(SoftF32);
pub const fn SoftFloat(v: T) -> _SoftFloat {
    _SoftFloat(SoftF32(v))
}
impl _SoftFloat {
    pub const fn to_float(&self) -> T {
        self.0.to_f32()
    }
    pub const fn sqrt(&self) -> Self {
        _SoftFloat(self.0.sqrt())
    }
    pub const fn exp(&self) -> Self {
        _SoftFloat(SoftF32(
            SoftF64(self.0.to_f32() as f64).exp().to_f64() as f32
        ))
    }
    pub const fn powf(&self, other: Self) -> Self {
        let other = SoftF64(other.0.to_f32() as f64);
        _SoftFloat(SoftF32(
            SoftF64(self.0.to_f32() as f64).powf(other).to_f64() as f32,
        ))
    }
    pub const fn powi(&self, other: i32) -> Self {
        _SoftFloat(self.0.powi(other))
    }
    pub const fn mul(&self, other: Self) -> Self {
        _SoftFloat(self.0.mul(other.0))
    }
    pub const fn div(&self, other: Self) -> Self {
        _SoftFloat(self.0.div(other.0))
    }
    pub const fn sub(&self, other: Self) -> Self {
        _SoftFloat(self.0.sub(other.0))
    }
    pub const fn add(&self, other: Self) -> Self {
        _SoftFloat(self.0.add(other.0))
    }
}
