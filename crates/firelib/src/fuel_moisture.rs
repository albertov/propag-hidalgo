use crate::TerrainCuda;

// Re-export types and functions from dmoist
pub use dmoist::{calculate_hourly_fuel_moisture, HourlyMoistureResults};

/// Create a TerrainCuda instance with calculated fuel moisture for a specific hour
///
/// This is a convenience function that combines fuel moisture calculation with
/// TerrainCuda creation for a single hour.
///
/// # Arguments
/// * `temperature` - Hourly temperature in degrees Celsius [0-23]
/// * `humidity` - Hourly relative humidity in percentage [0-23]
/// * `cloud_cover` - Hourly cloud cover in percentage [0-23]
/// * `slope` - Fixed slope ratio
/// * `aspect` - Fixed aspect value in radians
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
    temperature: &[f32; 24],
    humidity: &[f32; 24],
    cloud_cover: &[f32; 24],
    slope: f32,
    aspect: f32,
    precipitation_6_days: &[f32; 6],
    month: i32,
    fuel_model: u8,
    hour: usize,
    wind_speed: &[f32; 24],
    wind_azimuth: &[f32; 24],
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
    let live_moisture = dmoist::humedad_vivo(month) / 100.0; // Convert from percentage to ratio

    TerrainCuda {
        fuel_code: fuel_model,
        d1hr: moisture_results.d1hr[hour] / 100.0, // Convert from percentage to ratio
        d10hr: moisture_results.d10hr[hour] / 100.0,
        d100hr: moisture_results.d100hr[hour] / 100.0,
        herb: live_moisture,
        wood: live_moisture,
        wind_speed: wind_speed[hour],
        wind_azimuth: wind_azimuth[hour],
        slope,
        aspect,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_creation() {
        let temperature = [20.0; 24];
        let humidity = [50.0; 24];
        let cloud_cover = [25.0; 24];
        let slope = 0.268; // 15 degree slope as ratio (tan(15°))
        let aspect = 4.712; // 270 degrees as radians (3π/2)
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
        let wind_speed = [50.0; 24];
        let wind_azimuth = [0.0; 24];
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
            &wind_speed,
            &wind_azimuth,
        ); // Should panic
    }
}
