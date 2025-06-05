use crate::float;
use crate::TerrainCuda;
use float::PI;
#[allow(unused_imports)]
use num_traits::Float;

/// Output data for hourly fuel moisture calculation
#[derive(Debug, Clone)]
pub struct HourlyMoistureResults {
    /// 1-hour fuel moisture values for each hour [0-23]
    pub d1hr: [float::T; 24],
    /// 10-hour fuel moisture values for each hour [0-23]
    pub d10hr: [float::T; 24],
    /// 100-hour fuel moisture values for each hour [0-23]
    pub d100hr: [float::T; 24],
}

// Calculate hourly fuel moisture values using dmoist algorithms
//
// This function takes meteorological inputs and returns calculated fuel moisture
// values for each hour of the day (0-23) that can be used with TerrainCuda structures.
//
// # Arguments
// * `temperature` - Hourly temperature in degrees Celsius [0-23]
// * `humidity` - Hourly relative humidity in percentage [0-23]
// * `cloud_cover` - Hourly cloud cover in percentage [0-23]
// * `slope` - Fixed slope value in degrees
// * `aspect` - Fixed aspect value in degrees
// * `precipitation_6_days` - Daily total precipitation for the last 6 days in mm [day-6 to day-1]
// * `month` - Month (1-12) for seasonal adjustments
// * `fuel_model` - Fuel model code (0-13) for model-specific adjustments
//
// # Returns
// * `HourlyMoistureResults` containing d1hr, d10hr, and d100hr values for each hour
//
// # Example
// ```no_run
// use firelib::fuel_moisture::calculate_hourly_fuel_moisture;
//
// let temperature = [15.0; 24]; // 15°C for all hours
// let humidity = [60.0; 24];    // 60% RH for all hours
// let cloud_cover = [50.0; 24]; // 50% cloud cover for all hours
// let slope = 30.0;             // 30 degree slope
// let aspect = 180.0;           // South-facing
// let precipitation_6_days = [0.0, 2.0, 0.0, 5.0, 0.0, 1.0]; // mm for last 6 days
// let month = 6;                // June
//
// let results = calculate_hourly_fuel_moisture(
//     &temperature, &humidity, &cloud_cover, slope, aspect,
//     &precipitation_6_days, month, 2
// );
// ```
#[allow(clippy::too_many_arguments)]
pub fn calculate_hourly_fuel_moisture(
    temperature: &[float::T; 24],
    humidity: &[float::T; 24],
    cloud_cover: &[float::T; 24],
    slope: float::T,
    aspect: float::T,
    precipitation_6_days: &[float::T; 6],
    month: i32,
    fuel_model: i32,
) -> HourlyMoistureResults {
    let mut d1hr_results = [0.0; 24];
    let mut d10hr_results = [0.0; 24];
    let mut d100hr_results = [0.0; 24];

    // Calculate maximum precipitation effect for the 6-day period
    let max_precipitation_effect = dmoist::efecto_precipitacion_maximo(month, precipitation_6_days);

    for (hour, d1hr) in d1hr_results.iter_mut().enumerate() {
        // Calculate base humidity index (HCB)
        let hcb_base = dmoist::hcb(temperature[hour], humidity[hour], hour as i32) as float::T;

        // Correct HCB for terrain and cloud shading effects
        let hcb_corrected = dmoist::corrige_hcb_por_sombreado(
            hcb_base,
            cloud_cover[hour],
            month,
            aspect,
            slope,
            fuel_model,
            hour as i32,
        );

        // Calculate ignition probability without precipitation correction
        let prob_ignition_uncorrected = dmoist::probabilidad_ignicion(
            temperature[hour],
            cloud_cover[hour],
            hcb_corrected,
            fuel_model,
        );

        // Correct ignition probability for precipitation effects
        let prob_ignition_corrected =
            dmoist::corrige_prob_por_pp(prob_ignition_uncorrected, max_precipitation_effect);

        // Calculate 1-hour fuel moisture
        *d1hr = dmoist::d1hr(
            hcb_corrected,
            prob_ignition_uncorrected,
            prob_ignition_corrected,
        );
    }

    // For 10hr and 100hr, we need previous values for the calculation
    // For now, we'll use a simplified approach and calculate them based on d1hr
    // In a complete implementation, these would need historical values
    for (hour, d10hr) in d10hr_results.iter_mut().enumerate() {
        // Calculate 10-hour moisture using the dmoist algorithm
        // We need values from 6 and 15 hours ago, but for simplicity we'll use current and previous day estimates
        let d1hr_6h = d1hr_results[6];
        let d1hr_15h = d1hr_results[15];
        let d10hr_moisture = dmoist::hco_x10hr(hour as i32, d1hr_results[hour], d1hr_6h, d1hr_15h);

        *d10hr = d10hr_moisture as float::T;
    }

    for (hour, d100hr) in d100hr_results.iter_mut().enumerate() {
        // For 100hr, apply the algorithm again to get the next size class
        let d10hr_6h = d10hr_results[6];
        let d10hr_15h = d10hr_results[15];
        let d100hr_moisture =
            dmoist::hco_x10hr(hour as i32, d10hr_results[hour], d10hr_6h, d10hr_15h);
        *d100hr = d100hr_moisture as float::T;
    }

    HourlyMoistureResults {
        d1hr: d1hr_results,
        d10hr: d10hr_results,
        d100hr: d100hr_results,
    }
}

/// Create a TerrainCuda instance with calculated fuel moisture for a specific hour
///
/// This is a convenience function that combines fuel moisture calculation with
/// TerrainCuda creation for a single hour.
///
/// # Arguments
/// * `temperature` - Hourly temperature in degrees Celsius [0-23]
/// * `humidity` - Hourly relative humidity in percentage [0-23]
/// * `cloud_cover` - Hourly cloud cover in percentage [0-23]
/// * `slope` - Fixed slope value in degrees
/// * `aspect` - Fixed aspect value in degrees
/// * `precipitation_6_days` - Daily total precipitation for the last 6 days in mm [day-6 to day-1]
/// * `month` - Month (1-12) for seasonal adjustments
/// * `fuel_model` - Fuel model code (0-13)
/// * `hour` - Hour of day (0-23) to extract values for
/// * `wind_speed` - Wind speed in m/s
/// * `wind_azimuth` - Wind direction in degrees
///
/// # Returns
/// * `TerrainCuda` instance ready for fire behavior calculations
#[allow(clippy::too_many_arguments)]
pub fn create_terrain_with_fuel_moisture(
    temperature: &[float::T; 24],
    humidity: &[float::T; 24],
    cloud_cover: &[float::T; 24],
    slope: float::T,
    aspect: float::T,
    precipitation_6_days: &[float::T; 6],
    month: i32,
    fuel_model: u8,
    hour: usize,
    wind_speed: float::T,
    wind_azimuth: float::T,
) -> TerrainCuda {
    assert!(hour < 24, "Hour must be 0-23");

    let moisture_results = calculate_hourly_fuel_moisture(
        temperature,
        humidity,
        cloud_cover,
        slope,
        aspect,
        precipitation_6_days,
        month,
        fuel_model as i32,
    );

    // Calculate live fuel moisture based on month
    let live_moisture = dmoist::humedad_vivo(month) as float::T / 100.0; // Convert from percentage to ratio

    TerrainCuda {
        fuel_code: fuel_model,
        d1hr: moisture_results.d1hr[hour] / 100.0, // Convert from percentage to ratio
        d10hr: moisture_results.d10hr[hour] / 100.0,
        d100hr: moisture_results.d100hr[hour] / 100.0,
        herb: live_moisture,
        wood: live_moisture,
        wind_speed,
        wind_azimuth,
        slope: (slope * PI / 180.0).tan() as float::T, // Convert degrees to slope ratio
        aspect: aspect as float::T,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hourly_moisture_calculation() {
        let temperature = [15.0; 24];
        let humidity = [60.0; 24];
        let cloud_cover = [50.0; 24];
        let slope = 30.0;
        let aspect = 180.0;
        let precipitation_6_days = [0.0, 2.0, 0.0, 5.0, 0.0, 1.0];
        let month = 6;

        let results = calculate_hourly_fuel_moisture(
            &temperature,
            &humidity,
            &cloud_cover,
            slope,
            aspect,
            &precipitation_6_days,
            month,
            2,
        );

        // Basic sanity checks
        assert_eq!(results.d1hr.len(), 24);
        assert_eq!(results.d10hr.len(), 24);
        assert_eq!(results.d100hr.len(), 24);

        // All values should be non-negative
        for i in 0..24 {
            assert!(
                results.d1hr[i] >= 0.0,
                "d1hr[{}] = {} should be >= 0",
                i,
                results.d1hr[i]
            );
            assert!(
                results.d10hr[i] >= 0.0,
                "d10hr[{}] = {} should be >= 0",
                i,
                results.d10hr[i]
            );
            assert!(
                results.d100hr[i] >= 0.0,
                "d100hr[{}] = {} should be >= 0",
                i,
                results.d100hr[i]
            );
        }
    }

    #[test]
    fn test_terrain_creation() {
        let temperature = [20.0; 24];
        let humidity = [50.0; 24];
        let cloud_cover = [25.0; 24];
        let slope = 15.0;
        let aspect = 270.0;
        let precipitation_6_days = [0.0; 6];
        let month = 8;

        let terrain = create_terrain_with_fuel_moisture(
            &temperature,
            &humidity,
            &cloud_cover,
            slope,
            aspect,
            &precipitation_6_days,
            month,
            3,
            12,
            5.0,
            180.0,
        );

        assert_eq!(terrain.fuel_code, 3);
        assert_eq!(terrain.wind_speed, 5.0);
        assert_eq!(terrain.wind_azimuth, 180.0);
        assert!(terrain.d1hr >= 0.0 && terrain.d1hr <= 1.0); // Should be ratio between 0-1
        assert!(terrain.d10hr >= 0.0 && terrain.d10hr <= 1.0);
        assert!(terrain.d100hr >= 0.0 && terrain.d100hr <= 1.0);
    }

    #[test]
    #[should_panic(expected = "Hour must be 0-23")]
    fn test_invalid_hour() {
        let temperature = [15.0; 24];
        let humidity = [60.0; 24];
        let cloud_cover = [50.0; 24];
        let slope = 0.0;
        let aspect = 0.0;
        let precipitation_6_days = [0.0; 6];
        let month = 1;

        create_terrain_with_fuel_moisture(
            &temperature,
            &humidity,
            &cloud_cover,
            slope,
            aspect,
            &precipitation_6_days,
            month,
            1,
            25, // Invalid hour
            0.0,
            0.0,
        ); // Should panic
    }
}
