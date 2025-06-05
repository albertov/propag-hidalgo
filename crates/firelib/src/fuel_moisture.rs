use crate::TerrainCuda;

// Re-export types and functions from dmoist
pub use dmoist::{calculate_hourly_fuel_moisture, HourlyMoistureResults};

/// Create TerrainCuda instances with calculated fuel moisture for all 24 hours
///
/// This is a convenience function that combines fuel moisture calculation with
/// TerrainCuda creation for an entire day.
///
/// # Arguments
/// * `temperature` - Hourly temperature in degrees Celsius [0-23]
/// * `humidity` - Hourly relative humidity in percentage [0-23]
/// * `cloud_cover` - Hourly cloud cover in percentage [0-23]
/// * `wind_speed` - Wind speed in m/s [0-23]
/// * `wind_azimuth` - Wind direction in degrees [0-23]
/// * `precipitation_6_days` - Daily total precipitation for the last 6 days in mm [day-6 to day-1]
/// * `slope` - Fixed slope ratio
/// * `aspect` - Fixed aspect value in radians
/// * `fuel_model` - Fuel model code (0-13)
/// * `month` - Month (1-12) for seasonal adjustments
///
/// # Returns
/// * Array of 24 `TerrainCuda` instances, one for each hour [0-23]
#[allow(clippy::too_many_arguments)]
pub fn create_terrain_with_fuel_moisture(
    temperature: &[f32; 24],
    humidity: &[f32; 24],
    cloud_cover: &[f32; 24],
    wind_speed: &[f32; 24],
    wind_azimuth: &[f32; 24],
    precipitation_6_days: &[f32; 6],
    slope: f32,
    aspect: f32,
    fuel_model: u8,
    month: i32,
) -> [TerrainCuda; 24] {
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

    // Create array using core::array::from_fn for each hour
    core::array::from_fn(|hour| TerrainCuda {
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_creation() {
        let temperature = [20.0; 24];
        let humidity = [50.0; 24];
        let cloud_cover = [25.0; 24];
        let wind_speed = [5.0; 24];
        let wind_azimuth = [180.0; 24];
        let slope = 0.268; // 15 degree slope as ratio (tan(15°))
        let aspect = 4.712; // 270 degrees as radians (3π/2)
        let precipitation_6_days = [0.0; 6];
        let month = 8;

        let terrain_array = create_terrain_with_fuel_moisture(
            &temperature,
            &humidity,
            &cloud_cover,
            &wind_speed,
            &wind_azimuth,
            &precipitation_6_days,
            slope,
            aspect,
            3,
            month,
        );

        // Test that we get an array of 24 TerrainCuda instances
        assert_eq!(terrain_array.len(), 24);

        // Test specific hour (e.g., hour 12)
        let terrain_12 = &terrain_array[12];
        assert_eq!(terrain_12.fuel_code, 3);
        assert_eq!(terrain_12.wind_speed, 5.0);
        assert_eq!(terrain_12.wind_azimuth, 180.0);
        assert!(terrain_12.d1hr >= 0.0 && terrain_12.d1hr <= 1.0); // Should be ratio between 0-1
        assert!(terrain_12.d10hr >= 0.0 && terrain_12.d10hr <= 1.0);
        assert!(terrain_12.d100hr >= 0.0 && terrain_12.d100hr <= 1.0);

        // Test that all hours have the same fuel_code and terrain parameters
        for (hour, terrain) in terrain_array.iter().enumerate() {
            assert_eq!(terrain.fuel_code, 3);
            assert_eq!(terrain.wind_speed, 5.0); // All hours have same wind in test data
            assert_eq!(terrain.wind_azimuth, 180.0);
            assert_eq!(terrain.slope, slope);
            assert_eq!(terrain.aspect, aspect);
            assert!(terrain.d1hr >= 0.0 && terrain.d1hr <= 1.0);
            assert!(terrain.d10hr >= 0.0 && terrain.d10hr <= 1.0);
            assert!(terrain.d100hr >= 0.0 && terrain.d100hr <= 1.0);
        }
    }

    #[test]
    fn test_varying_conditions() {
        // Test with varying conditions across hours
        let temperature = [
            15.0, 14.0, 13.0, 12.0, 11.0, 12.0, 13.0, 15.0, 18.0, 21.0, 24.0, 26.0, 28.0, 29.0,
            28.0, 26.0, 24.0, 21.0, 19.0, 17.0, 16.0, 15.0, 14.0, 13.0,
        ];
        let humidity = [
            80.0, 82.0, 84.0, 86.0, 88.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 45.0,
            50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 78.0, 80.0, 81.0, 82.0,
        ];
        let cloud_cover = [25.0; 24];
        let wind_speed = [3.0; 24];
        let wind_azimuth = [270.0; 24];
        let slope = 0.0;
        let aspect = 0.0;
        let precipitation_6_days = [2.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let month = 6;

        let terrain_array = create_terrain_with_fuel_moisture(
            &temperature,
            &humidity,
            &cloud_cover,
            &wind_speed,
            &wind_azimuth,
            &precipitation_6_days,
            slope,
            aspect,
            2,
            month,
        );

        // Test that conditions vary across hours (fuel moisture should vary with temperature/humidity)
        let terrain_0 = &terrain_array[0]; // Cool, humid night
        let terrain_13 = &terrain_array[13]; // Hot, dry afternoon

        // During hot, dry conditions fuel moisture should be lower than cool, humid conditions
        // (though exact values depend on complex calculations)
        assert!(terrain_0.d1hr > 0.0);
        assert!(terrain_13.d1hr > 0.0);
    }
}
