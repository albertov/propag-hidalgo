// fuel load
pub mod areal_mass_density {
    unit! {
        system: uom::si;
        quantity: uom::si::areal_mass_density;

        @pound_per_square_foot: (3.048_E-1 * 3.048_E-1) / 4.535_924_E-1 ; "lb/ft²", "pound mass per square foot", "pounds mass per square foot";

    }
}

// HeatOfCombustion es uom::si::available_energy

// ByramsIntensity es uom::si::linear_power_density
pub mod linear_power_density {
    unit! {
        system: uom::si;
        quantity: uom::si::linear_power_density;

        @btu_foot_sec: 1.054_350_E3 / 3.048_E-1; "btu/ft/sec", "BTU per foot per sec", "BTU per foot per sec";
    }
}

// ReactionIntensity es uom::si::heat_flux_density
// ReactionVelocity es uom::si::frequency
// HeatPerUnitArea es uom::si::radiant_exposure
pub mod radiant_exposure {
    unit! {
        system: uom::si;
        quantity: uom::si::radiant_exposure;

        @btu_sq_foot: 1.054_350_E3 / (3.048_E-1 * 3.048_E-1); "btu/ft²", "BTU per square foot", "BTU per square foot";
    }
}

pub mod heat_flux_density {
    unit! {
        system: uom::si;
        quantity: uom::si::heat_flux_density;

        @btu_sq_foot_min: 1.054_350_E3 / (3.048_E-1 * 3.048_E-1) / 60.0; "btu/ft²/min", "BTU per square foot per minute", "BTU per square foot per minute";
    }
}
pub mod reciprocal_length {
    unit! {
        system: uom::si;
        quantity: uom::si::reciprocal_length;

        @reciprocal_foot: 1.0 / 3.048_E-1; "ft⁻¹", "reciprocal foot", "reciprocal feet";
    }
}
