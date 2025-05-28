// fuel load
pub mod areal_mass_density {
    
    unit! {
        system: uom::si;
        quantity: uom::si::areal_mass_density;

        @pound_per_square_foot: (3.048_E-1 * 3.048_E-1) / 4.535_924_E-1; "lb/ft²", "pound mass per square foot", "pounds mass per square foot";

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

pub const FOOT_TO_M : f64 = 3.048_E-1;
pub const POUND_TO_KG : f64 = 4.535_924_E-1;
pub const BTU : f64 = 1.054_350_E3;
use const_soft_float::soft_f64::SoftF64;
use std::marker::PhantomData;
use uom::si::f64::*;

pub const fn load_to_imperial(x: &ArealMassDensity) -> f64 {
    let ArealMassDensity { value, ..} = x;
    SoftF64(*value)
        .mul(SoftF64(POUND_TO_KG).div(SoftF64(FOOT_TO_M).powi(2)))
        .to_f64()
}
pub const fn load_from_imperial(x: f64) -> ArealMassDensity {
    let value = SoftF64(x)
        .div(SoftF64(POUND_TO_KG).div(SoftF64(FOOT_TO_M).powi(2)))
        .to_f64();
    ArealMassDensity {
        value, units: PhantomData, dimension: PhantomData
    }
}

pub const fn density_to_imperial(x: &MassDensity) -> f64 {
    let MassDensity { value, ..} = x;
    SoftF64(*value)
        .mul(SoftF64(POUND_TO_KG).div(SoftF64(FOOT_TO_M).powi(3)))
        .to_f64()
}

pub const fn density_from_imperial(x: f64) -> MassDensity {
    let value = SoftF64(x)
        .div(SoftF64(POUND_TO_KG).div(SoftF64(FOOT_TO_M).powi(3)))
        .to_f64();
    MassDensity {
        value, units: PhantomData, dimension: PhantomData
    }
}

pub const fn savr_to_imperial(x: &ReciprocalLength) -> f64 {
    let ReciprocalLength { value, ..} = x;
    SoftF64(*value)
        .mul(SoftF64(1.0).div(SoftF64(FOOT_TO_M)))
        .to_f64()
}

pub const fn savr_from_imperial(x: f64) -> ReciprocalLength {
    let value = 
        SoftF64(x)
            .div(SoftF64(1.0).div(SoftF64(FOOT_TO_M)))
            .to_f64();
    ReciprocalLength {
        value, units: PhantomData, dimension: PhantomData
    }
}

pub const fn heat_to_imperial(x: &AvailableEnergy) -> f64 {
    let AvailableEnergy { value, ..} = x;
    SoftF64(*value)
        .div(SoftF64(BTU).div(SoftF64(POUND_TO_KG)))
        .to_f64()
}

pub const fn heat_from_imperial(x: f64) -> AvailableEnergy {
    let value = SoftF64(x)
        .mul(SoftF64(BTU).div(SoftF64(POUND_TO_KG)))
        .to_f64();
    AvailableEnergy {
        value, units: PhantomData, dimension: PhantomData
    }
}

pub const fn extract_ratio(x: &Ratio) -> f64 {
    let Ratio { value, ..} = x;
    *value
}

pub const fn mk_ratio(x: f64) -> Ratio {
    Ratio {
        units: PhantomData,
        dimension: PhantomData,
        value: x
    }
}

pub const fn length_from_imperial(x: f64) -> Length {
    Length {
        units: PhantomData,
        dimension: PhantomData,
        value: SoftF64(x).mul(SoftF64(FOOT_TO_M)).to_f64()
    }
}

pub const fn length_to_imperial(x: &Length) -> f64 {
    let Length { value, ..} = x;
    SoftF64(*value).div(SoftF64(FOOT_TO_M)).to_f64()
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
