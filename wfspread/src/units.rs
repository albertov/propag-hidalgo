// fuel load
mod areal_mass_density {
    unit! {
        system: uom::si;
        quantity: uom::si::areal_mass_density;

        @pound_per_square_foot: 4.8824276270581; "lb/ft²", "pound mass per square foot", "pounds mass per square foot";
    }
}

mod mass_density {
    unit! {
        system: uom::si;
        quantity: uom::si::mass_density;

        @pound_per_cubic_foot: 16.018463374; "lb/ft³", "pound mass per cubic foot", "pounds mass per cubic foot";
    }
}

// HeatOfCombustion es uom::si::available_energy
mod available_energy {
    unit! {
        system: uom::si;
        quantity: uom::si::available_energy;

        @btu_per_pound: 1.054_350_E3 / 0.453592; "btu/lb", "BTU per pound", "BTUs per pound";
    }
}
// ByramsIntensity es uom::si::linear_power_density
mod linear_power_density {
    unit! {
        system: uom::si;
        quantity: uom::si::linear_power_density;

        @btu_foot_sec: 1.054_350_E3 / 3.048_E-1; "btu/ft/sec", "BTU per foot per sec", "BTU per foot per sec";
    }
}

// ReactionVelocity es uom::si::frequency
// HeatPerUnitArea es uom::si::radiant_exposure
mod radiant_exposure {
    unit! {
        system: uom::si;
        quantity: uom::si::radiant_exposure;

        @btu_sq_foot: 1.054_350_E3 / (3.048_E-1 * 3.048_E-1); "btu/ft²", "BTU per square foot", "BTU per square foot";
    }
}

mod heat_flux_density {
    unit! {
        system: uom::si;
        quantity: uom::si::heat_flux_density;

        @btu_sq_foot_min: 1.054_350_E3 / (3.048_E-1 * 3.048_E-1) / 60.0; "btu/ft²/min", "BTU per square foot per minute", "BTU per square foot per minute";
    }

    // SaToVolRatio es uom::si::reciprocal_length
    mod reciprocal_length {
        unit! {
            system: uom::si;
            quantity: uom::si::reciprocal_length;

            @perFoot: 3.048_E1; "ft⁻¹", "per foot", "per feet";
        }
    }
}
