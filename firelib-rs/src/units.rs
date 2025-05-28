// fuel load
pub mod areal_mass_density {

    unit! {
        system: uom::si;
        quantity: uom::si::areal_mass_density;

        @pound_per_square_foot:  4.535_924_E-1 / (3.048_E-1 * 3.048_E-1); "lb/ft²", "pound mass per square foot", "pounds mass per square foot";

    }
}

// HeatOfCombustion es uom::si::available_energy

// ByramsIntensity es uom::si::linear_power_density
pub mod linear_power_density {

    unit! {
        system: uom::si;
        quantity: uom::si::linear_power_density;

        @btu_foot_sec: 1.054_350E3 / 3.048_E-1; "btu/ft/sec", "BTU per foot per sec", "BTU per foot per sec";
    }
}

use crate::firelib::float;
use crate::firelib::float::*;
pub const FOOT_TO_M: float::T = 3.048_E-1;
pub const POUND_TO_KG: float::T = 4.535_924_E-1;
pub const BTU: float::T = 1.054_350_E3;
use uom::lib::marker::PhantomData;

pub const fn load_to_imperial(x: &ArealMassDensity) -> float::T {
    let ArealMassDensity { value, .. } = x;
    SoftFloat(*value)
        .div(SoftFloat(POUND_TO_KG).div(SoftFloat(FOOT_TO_M).powi(2)))
        .to_float()
}
pub const fn load_from_imperial(x: float::T) -> ArealMassDensity {
    let value = SoftFloat(x)
        .mul(SoftFloat(POUND_TO_KG).div(SoftFloat(FOOT_TO_M).powi(2)))
        .to_float();
    ArealMassDensity {
        value,
        units: PhantomData,
        dimension: PhantomData,
    }
}

pub const fn density_to_imperial(x: &MassDensity) -> float::T {
    let MassDensity { value, .. } = x;
    SoftFloat(*value)
        .div(SoftFloat(POUND_TO_KG).div(SoftFloat(FOOT_TO_M).powi(3)))
        .to_float()
}

pub const fn density_from_imperial(x: float::T) -> MassDensity {
    let value = SoftFloat(x)
        .mul(SoftFloat(POUND_TO_KG).div(SoftFloat(FOOT_TO_M).powi(3)))
        .to_float();
    MassDensity {
        value,
        units: PhantomData,
        dimension: PhantomData,
    }
}

pub const fn savr_to_imperial(x: &ReciprocalLength) -> float::T {
    let ReciprocalLength { value, .. } = x;
    SoftFloat(*value)
        .div(SoftFloat(1.0).div(SoftFloat(FOOT_TO_M)))
        .to_float()
}

pub const fn savr_from_imperial(x: float::T) -> ReciprocalLength {
    let value = SoftFloat(x)
        .mul(SoftFloat(1.0).div(SoftFloat(FOOT_TO_M)))
        .to_float();
    ReciprocalLength {
        value,
        units: PhantomData,
        dimension: PhantomData,
    }
}

pub const fn heat_to_imperial(x: &AvailableEnergy) -> float::T {
    let AvailableEnergy { value, .. } = x;
    SoftFloat(*value)
        .div(SoftFloat(BTU).div(SoftFloat(POUND_TO_KG)))
        .to_float()
}

pub const fn heat_from_imperial(x: float::T) -> AvailableEnergy {
    let value = SoftFloat(x)
        .mul(SoftFloat(BTU).div(SoftFloat(POUND_TO_KG)))
        .to_float();
    AvailableEnergy {
        value,
        units: PhantomData,
        dimension: PhantomData,
    }
}

pub const fn length_from_imperial(x: float::T) -> Length {
    Length {
        units: PhantomData,
        dimension: PhantomData,
        value: SoftFloat(x).mul(SoftFloat(FOOT_TO_M)).to_float(),
    }
}

pub const fn length_to_imperial(x: &Length) -> float::T {
    let Length { value, .. } = x;
    SoftFloat(*value).div(SoftFloat(FOOT_TO_M)).to_float()
}

// ReactionIntensity es uom::si::heat_flux_density
// ReactionVelocity es uom::si::frequency
// HeatPerUnitArea es uom::si::radiant_exposure
pub mod radiant_exposure {

    unit! {
        system: uom::si;
        quantity: uom::si::radiant_exposure;

        @btu_sq_foot: 1.054_350E3 / (3.048_E-1 * 3.048_E-1); "btu/ft²", "BTU per square foot", "BTU per square foot";
    }
}

pub mod heat_flux_density {

    unit! {
        system: uom::si;
        quantity: uom::si::heat_flux_density;

        @btu_sq_foot_min: 1.054_350E3 / (3.048_E-1 * 3.048_E-1) / 60.0; "btu/ft²/min", "BTU per square foot per minute", "BTU per square foot per minute";
    }
}
pub mod reciprocal_length {

    unit! {
        system: uom::si;
        quantity: uom::si::reciprocal_length;

        @reciprocal_foot: 1.0 / 3.048_E-1; "ft⁻¹", "reciprocal foot", "reciprocal feet";
    }
}
