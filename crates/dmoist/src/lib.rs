#![feature(import_trait_associated_functions)]
#![no_std]
use num::traits::FloatConst;
#[allow(unused_imports)]
use num::Float;
use num::Float::nan;

/// Output data for hourly fuel moisture calculation
#[derive(Debug, Clone)]
pub struct HourlyMoistureResults {
    /// 1-hour fuel moisture values for each hour [0-23]
    pub d1hr: [f32; 24],
    /// 10-hour fuel moisture values for each hour [0-23]
    pub d10hr: [f32; 24],
    /// 100-hour fuel moisture values for each hour [0-23]
    pub d100hr: [f32; 24],
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
// * `slope` - Fixed slope ratio
// * `aspect` - Fixed aspect value in radians
// * `precipitation_6_days` - Daily total precipitation for the last 6 days in mm [day-6 to day-1]
// * `month` - Month (1-12) for seasonal adjustments
// * `fuel_model` - Fuel model code (0-13) for model-specific adjustments
//
// # Returns
// * `HourlyMoistureResults` containing d1hr, d10hr, and d100hr values for each hour
//
// # Example
// ```no_run
// use dmoist::calculate_hourly_fuel_moisture;
//
// let temperature = [15.0; 24]; // 15°C for all hours
// let humidity = [60.0; 24];    // 60% RH for all hours
// let cloud_cover = [50.0; 24]; // 50% cloud cover for all hours
// let slope = 0.577;            // 30 degree slope as ratio (tan(30°))
// let aspect = 3.14159;         // South-facing (π radians)
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
    temperature: &[f32; 24],
    humidity: &[f32; 24],
    cloud_cover: &[f32; 24],
    slope: f32,
    aspect: f32,
    precipitation_6_days: &[f32; 6],
    month: i32,
    fuel_model: i32,
) -> HourlyMoistureResults {
    let mut d1hr_results = [0.0; 24];
    let mut d10hr_results = [0.0; 24];
    let mut d100hr_results = [0.0; 24];

    // Calculate maximum precipitation effect for the 6-day period
    let max_precipitation_effect = maximum_precipitation_effect(month, precipitation_6_days);

    for (hour, d1hr_value) in d1hr_results.iter_mut().enumerate() {
        // Calculate base humidity index (HCB)
        let hcb_base = humidity_content_base(temperature[hour], humidity[hour], hour as i32) as f32;

        // Correct HCB for terrain and cloud shading effects
        let hcb_corrected = correct_hcb_for_shading(
            hcb_base,
            cloud_cover[hour],
            month,
            aspect * 180.0 / f32::PI(),       // Convert radians to degrees
            slope.atan() * 180.0 / f32::PI(), // Convert ratio to degrees
            fuel_model,
            hour as i32,
        );

        // Calculate ignition probability without precipitation correction
        let prob_ignition_uncorrected = ignition_probability(
            temperature[hour],
            cloud_cover[hour],
            hcb_corrected,
            fuel_model,
        );

        // Correct ignition probability for precipitation effects
        let prob_ignition_corrected = correct_probability_for_precipitation(
            prob_ignition_uncorrected,
            max_precipitation_effect,
        );

        // Calculate 1-hour fuel moisture
        *d1hr_value = d1hr(
            hcb_corrected,
            prob_ignition_uncorrected,
            prob_ignition_corrected,
        );
    }

    // For 10hr and 100hr, we need previous values for the calculation
    // For now, we'll use a simplified approach and calculate them based on d1hr
    // In a complete implementation, these would need historical values
    for (hour, d10hr_value) in d10hr_results.iter_mut().enumerate() {
        // Calculate 10-hour moisture using the dmoist algorithm
        // We need values from 6 and 15 hours ago, but for simplicity we'll use current and previous day estimates
        let d1hr_6h = d1hr_results[6];
        let d1hr_15h = d1hr_results[15];
        let d10hr_moisture =
            moisture_content_by_10hr(hour as i32, d1hr_results[hour], d1hr_6h, d1hr_15h);

        *d10hr_value = d10hr_moisture;
    }

    for (hour, d100hr_value) in d100hr_results.iter_mut().enumerate() {
        // For 100hr, apply the algorithm again to get the next size class
        let d10hr_6h = d10hr_results[6];
        let d10hr_15h = d10hr_results[15];
        let d100hr_moisture =
            moisture_content_by_10hr(hour as i32, d10hr_results[hour], d10hr_6h, d10hr_15h);
        *d100hr_value = d100hr_moisture;
    }

    HourlyMoistureResults {
        d1hr: d1hr_results,
        d10hr: d10hr_results,
        d100hr: d100hr_results,
    }
}

/// Returns the effect of a certain precipitation as a function
/// of the month and how many days have passed since the day being
/// calculated.
///
/// Params:
///   * month [1..12]
///   * precipitation (ml)
///   * offset of the day [0..5] (0 day before prediction)
fn _precipitation_effect(month: i32, precipitation: f32, offset: usize) -> f32 {
    let tabla = [
        // april-october
        [
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],  // [1-5)
            [-1.0, 5.0, 1.0, 0.0, 0.0, 0.0],   // [5-15)
            [-1.0, 15.0, 5.0, 1.0, 0.0, 0.0],  // [15-35)
            [-1.0, 35.0, 15.0, 5.0, 1.0, 0.0], // [35-INF)
        ],
        // rest
        [
            [-1.0, -1.0, -1.0, -1.0, 1.0, 0.0], // [1-5)
            [-1.0, -1.0, -1.0, -1.0, 5.0, 1.0], // [5-15)
            [-1.0, -1.0, -1.0, 15.0, 5.0, 1.0], // [15-35)
            [-1.0, -1.0, 35.0, 15.0, 5.0, 1.0], // [35-INF)
        ],
    ];

    if precipitation > 0.0 {
        let month_index = if (4..=10).contains(&month) { 0 } else { 1 };
        let precipitation_index = if (1.0..5.0).contains(&precipitation) {
            0
        } else if (5.0..15.0).contains(&precipitation) {
            1
        } else if (15.0..35.0).contains(&precipitation) {
            2
        } else {
            3
        };

        let effect = tabla[month_index][precipitation_index][offset];
        if effect > -1.0 {
            return effect;
        }
    }
    precipitation
}

fn max_f32_iter<I>(iter: I) -> f32
where
    I: Iterator<Item = f32>,
{
    iter.fold(f32::NEG_INFINITY, |a, b| a.max(b))
}

pub fn maximum_precipitation_effect(month: i32, precipitation: &[f32; 6]) -> f32 {
    // Returns the maximum precipitation effect of the last 6 days
    max_f32_iter((0..6).map(|i| _precipitation_effect(month, precipitation[i], i)))
}

pub fn humidity_content_base(temperature: f32, relative_humidity: f32, _hour: i32) -> i32 {
    // Indices: (day/night, temperature, humidity)
    let tabla = [
        [
            //    0  5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
            // From 8-20
            [
                1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 8, 9, 9, 10, 11, 12, 12, 13, 13, 14,
            ], // < 0
            [
                1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 13,
            ], // 0-9
            [
                1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 9, 10, 11, 12, 12, 12, 13,
            ], // 10-20
            [
                1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13,
            ], // 21-31
            [
                1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13,
            ], // 32-42
            [
                1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 12,
            ], // >42
        ],
        [
            // From 20-8
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 17, 19, 21, 24, 25, 25,
            ],
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 14, 16, 18, 21, 24, 25, 25,
            ],
            [
                1, 2, 3, 4, 5, 6, 6, 8, 8, 9, 10, 11, 11, 12, 14, 16, 17, 20, 23, 25, 25,
            ],
            [
                1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 15, 17, 20, 23, 25, 25,
            ],
            [
                1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 13, 14, 16, 19, 22, 25, 25,
            ],
            [
                1, 2, 3, 3, 4, 5, 6, 6, 8, 9, 9, 9, 10, 11, 12, 14, 16, 19, 21, 24, 25,
            ],
        ],
    ];

    let hora_idx = 0; // Only use day table if (8 <= hora < 20) else 1
    let humidity_idx = (relative_humidity / 5.0).floor().clamp(0.0, 20.0) as usize;
    let temperature_idx = if temperature < 0.0 {
        0
    } else if (0.0..=9.0).contains(&temperature) {
        1
    } else if temperature > 9.0 && temperature <= 20.0 {
        2
    } else if temperature > 20.0 && temperature <= 31.0 {
        3
    } else if temperature > 31.0 && temperature <= 42.0 {
        4
    } else {
        5
    };

    tabla[hora_idx][temperature_idx][humidity_idx]
}

pub fn shading(cloudiness: f32, fuel_model: i32) -> f32 {
    let mut shading_value = cloudiness * 0.75;
    if (7..=12).contains(&fuel_model) {
        shading_value = if shading_value < 50.0 { 75.0 } else { 100.0 };
    }
    shading_value
}

pub fn ignition_probability(temperature: f32, cloudiness: f32, hcs: f32, fuel_model: i32) -> f32 {
    let tabla = [
        [
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=1 (TSECA>40)
            [
                100, 100, 100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 30, 20, 20, 20, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=2 (TSECA 35-40)
            [
                100, 100, 90, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=3 (TSECA 30-35)
            [
                100, 100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=4 (TSECA 25-30)
            [
                100, 100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=5 (TSECA 20-25)
            [
                100, 100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=6 (TSECA 15-20)
            [
                100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=7 (TSECA 10-15)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=8 (TSECA 5-10)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=1 (SOMB=0-10) and Class_TSECA=9 (TSECA 0-5)
            [
                100, 90, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10,
            ],
        ],
        [
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=1 (TSECA>40)
            [
                100, 100, 100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=2 (TSECA 35-40)
            [
                100, 100, 90, 80, 70, 60, 50, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=3 (TSECA 30-35)
            [
                100, 100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=4 (TSECA 25-30)
            [
                100, 100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=5 (TSECA 20-25)
            [
                100, 100, 80, 70, 60, 50, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=6 (TSECA 15-20)
            [
                100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=7 (TSECA 10-15)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=8 (TSECA 5-10)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=2 (SOMB=10-50) and Class_TSECA=9 (TSECA 0-5)
            [
                100, 80, 70, 60, 50, 50, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10, 10,
            ],
        ],
        [
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=1 (TSECA>40)
            [
                100, 100, 90, 80, 70, 60, 50, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=2 (TSECA 35-40)
            [
                100, 100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=3 (TSECA 30-35)
            [
                100, 100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=4 (TSECA 25-30)
            [
                100, 100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=5 (TSECA 20-25)
            [
                100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=6 (TSECA 15-20)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=7 (TSECA 10-15)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=8 (TSECA 5-10)
            [
                100, 90, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=3 (SOMB=50-90) and Class_TSECA=9 (TSECA 0-5)
            [
                100, 80, 70, 60, 50, 50, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10, 10,
            ],
        ],
        [
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=1 (TSECA>40)
            [
                100, 100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=2 (TSECA 35-40)
            [
                100, 100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=3 (TSECA 30-35)
            [
                100, 100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=4 (TSECA 25-30)
            [
                100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=5 (TSECA 20-25)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=6 (TSECA 15-20)
            [
                100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=7 (TSECA 10-15)
            [
                100, 90, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=8 (TSECA 5-10)
            [
                100, 80, 70, 60, 50, 50, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10, 10,
            ],
            // Class_SOMB=4 (SOMB=90-100) and Class_TSECA=9 (TSECA 0-5)
            [
                100, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10, 10,
            ],
        ],
    ];

    if fuel_model == 0 {
        0.0
    } else if hcs > 0.0 && hcs < 18.0 {
        let shading_val = shading(cloudiness, fuel_model);
        let shading_idx = if (0.0..10.0).contains(&shading_val) {
            0
        } else if (10.0..50.0).contains(&shading_val) {
            1
        } else if (50.0..90.0).contains(&shading_val) {
            2
        } else {
            3
        };

        let temperature_idx = (9.0 - (temperature / 5.0).ceil()).clamp(0.0, 8.0) as usize;
        let hcs_idx = (hcs.floor() as usize) - 1;

        return tabla[shading_idx][temperature_idx][hcs_idx] as _;
    } else {
        return 0.0;
    }
}

fn _is_north(orientation: f32) -> bool {
    (315.0..=360.0).contains(&orientation) || (0.0..45.0).contains(&orientation)
}

fn _is_east(orientation: f32) -> bool {
    (45.0..135.0).contains(&orientation)
}

fn _is_south(orientation: f32) -> bool {
    (135.0..225.0).contains(&orientation)
}

fn _is_west(orientation: f32) -> bool {
    (225.0..315.0).contains(&orientation)
}

fn _corrector_for_shading(
    orientation: f32,
    slope: f32,
    shading_val: f32,
    month: i32,
    hour_param: i32,
) -> f32 {
    // Between 20 and 6 (pseudo-dawn) apply values of 20,
    // between 6 and 8. It's an attempt to make the variation come out
    // continuous since the tables don't contemplate correction at night but
    // they are wrong
    let (hour, shading) = if !(6..=19).contains(&hour_param) {
        (19, 100.0)
    } else if hour_param < 8 {
        (8, 100.0)
    } else {
        (hour_param, shading_val)
    };

    // index shading
    let shading_idx = if shading < 50.0 { 0 } else { 1 };

    // index hour
    let hour_idx = ((hour - 8) / 2).clamp(0, 5) as usize;

    // Index month
    let month_idx = match month {
        5..=8 => 0,
        2 | 3 | 4 | 9 | 10 => 1,
        11 | 12 | 1 => 2,
        _ => return nan(), // invalid month
    };

    // Flat terrain is south orientation
    let orientation_adjusted = if slope == 0.0 { 180.0 } else { orientation };

    // Index slope/orientation
    let orientation_slope_idx = if _is_north(orientation_adjusted) {
        if slope <= 30.0 {
            0
        } else {
            1
        }
    } else if _is_east(orientation_adjusted) {
        if slope <= 30.0 {
            2
        } else {
            3
        }
    } else if _is_south(orientation_adjusted) {
        if slope <= 30.0 {
            4
        } else {
            5
        }
    } else if _is_west(orientation_adjusted) {
        if slope <= 30.0 {
            6
        } else {
            7
        }
    } else {
        return nan();
    };

    let tabla = [
        [
            // may,jun,jul,ago
            [
                // Exposed (< 50% fuel in shade)
                [3, 1, 0, 0, 1, 3], // N, 0-30%
                [4, 2, 1, 1, 2, 4], // N, >30%
                [2, 1, 0, 0, 1, 4], // E, 0-30%
                [2, 0, 0, 1, 3, 5], // E, >30%
                [3, 1, 0, 0, 1, 3], // S, 0-30%
                [3, 1, 1, 1, 1, 3], // S, >30%
                [3, 1, 0, 0, 1, 3], // O, 0-30%
                [5, 3, 1, 0, 0, 2], // O, >30%
            ],
            [
                // Shaded (>= 50% fuel in shade)
                // Slope doesn't influence so row is duplicated for both cases
                [5, 4, 3, 3, 4, 5],
                [5, 4, 3, 3, 4, 5],
                [4, 4, 3, 4, 4, 5],
                [4, 4, 3, 4, 4, 5],
                [4, 4, 3, 3, 4, 5],
                [4, 4, 3, 3, 4, 5],
                [5, 4, 3, 3, 4, 4],
                [5, 4, 3, 3, 4, 4],
            ],
        ],
        [
            // feb,mar,abr,sep,oct
            [
                [4, 2, 1, 1, 2, 4],
                [4, 3, 3, 3, 3, 4],
                [4, 2, 1, 1, 2, 4],
                [3, 1, 1, 2, 4, 5],
                [4, 2, 1, 1, 2, 4],
                [4, 2, 1, 1, 2, 4],
                [4, 2, 1, 1, 2, 4],
                [5, 4, 2, 1, 1, 3],
            ],
            [
                [5, 5, 4, 4, 5, 5],
                [5, 5, 4, 4, 5, 5],
                [5, 4, 4, 4, 5, 5],
                [5, 4, 4, 4, 5, 5],
                [5, 4, 4, 4, 4, 5],
                [5, 4, 4, 4, 4, 5],
                [5, 5, 4, 4, 4, 5],
                [5, 5, 4, 4, 4, 5],
            ],
        ],
        [
            // nov,dic,ene
            [
                [5, 4, 3, 3, 4, 5],
                [5, 5, 5, 5, 5, 5],
                [5, 4, 3, 3, 4, 5],
                [5, 4, 3, 2, 5, 5],
                [5, 4, 3, 2, 4, 5],
                [5, 3, 1, 1, 3, 5],
                [5, 4, 3, 3, 4, 5],
                [5, 5, 4, 2, 3, 5],
            ],
            [
                // All exposures and slopes
                [5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5],
            ],
        ],
    ];

    tabla[month_idx][shading_idx][orientation_slope_idx][hour_idx] as _
}

pub fn correct_hcb_for_shading(
    hcb: f32,
    cloudiness: f32,
    month: i32,
    orientation: f32,
    slope: f32,
    fuel_model: i32,
    hour: i32,
) -> f32 {
    if fuel_model == 0 {
        0.0
    } else {
        let shading_val = shading(cloudiness, fuel_model);
        hcb + _corrector_for_shading(orientation, slope, shading_val, month, hour)
    }
}

pub fn correct_probability_for_precipitation(probability: f32, precipitation_effect: f32) -> f32 {
    // Corrects probability by precipitation effect
    // The following if-elif with formula is:
    // PROBIG - Int(2.61863 + (0.988897 * PP6) - (0.00632351 * ((PP6) ^ 2)))
    let adjustment = if precipitation_effect > 0.0 && precipitation_effect < 5.0 {
        -5.0
    } else if (5.0..15.0).contains(&precipitation_effect) {
        -10.0
    } else if (15.0..35.0).contains(&precipitation_effect) {
        -20.0
    } else if precipitation_effect >= 35.0 {
        -30.0
    } else {
        0.0
    };

    (probability + adjustment).clamp(0.0, 100.0)
}

pub fn d1hr(hcs: f32, prob_ign_sc: f32, prob_ign: f32) -> f32 {
    if prob_ign <= 0.0 {
        if hcs > 0.0 {
            30.0 // prob. is not 0 for being non-combustible -> maximum humidity
        } else {
            0.0 // no fuel moisture, assume non-combustible
        }
    } else {
        prob_ign_sc / prob_ign * hcs
    }
}

// Calculates the humidity of the thickest fuel, i.e., if 'moisture' is
// 1hr returns 10hr, if 10hr -> 100hr.

// @param hour: The hour of the day
// @param moisture: The moisture of the finest fuel
// @param moisture_6: The moisture of the finest fuel at 6 hours
// @param moisture_15: The moisture of the finest fuel at 15 hours
pub fn moisture_content_by_10hr(
    hour: i32,
    moisture: f32,
    moisture_6: f32,
    moisture_15: f32,
) -> f32 {
    if moisture == 0.0 {
        0.0
    } else {
        let a_las_6 = moisture_15 + (0.4142 * (moisture_6 - moisture_15));
        let a_las_15 = moisture_6 - (0.5571 * (moisture_6 - moisture_15));

        if (a_las_6 - a_las_15).abs() < 0.001 {
            if a_las_6 < 0.001 {
                moisture
            } else {
                a_las_6
            }
        } else {
            // interpolate the value based on the hour of the day
            if (0..6).contains(&hour) {
                a_las_15 + (((a_las_6 - a_las_15) / 15.0) * ((hour + 15) as f32))
            } else if (6..15).contains(&hour) {
                a_las_6 + (((a_las_15 - a_las_6) / 9.0) * ((hour - 6) as f32))
            } else if (15..24).contains(&hour) {
                a_las_15 + (((a_las_6 - a_las_15) / 15.0) * ((hour - 15) as f32))
            } else {
                moisture // fallback for invalid hour
            }
        }
    }
}

// Live fuel moisture estimation based on month
pub fn live_moisture(month: i32) -> f32 {
    match month {
        1 | 2 => 100.0,
        3..=5 => 200.0,
        6 => 100.0,
        7 | 8 => 80.0,
        9 | 10 => 90.0,
        11 | 12 => 100.0,
        _ => nan(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec::Vec;

    // Parser for simple tuple -> value mappings
    fn parse_simple_fixtures(content: &str) -> Vec<(Vec<f32>, f32)> {
        let mut fixtures = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if let Some((left, right)) = line.split_once(" -> ") {
                let left = left.trim();
                let right = right.trim();

                // Parse tuple like "(1, 2, 3)"
                if left.starts_with('(') && left.ends_with(')') {
                    let inner = &left[1..left.len() - 1];
                    let params: Vec<f32> = inner
                        .split(',')
                        .map(|s| s.trim().parse().unwrap_or(0.0))
                        .collect();
                    let result: f32 = right.parse().unwrap_or(0.0);
                    fixtures.push((params, result));
                }
            }
        }

        fixtures
    }

    // Parser for month + range fixtures like "[1] + range(6) -> 3"
    fn parse_precipitation_fixtures(content: &str) -> Vec<(i32, f32)> {
        let mut fixtures = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if let Some((left, right)) = line.split_once(" -> ") {
                let left = left.trim();
                let right = right.trim();

                // Parse "[1] + range(6)" format
                if left.starts_with('[') && left.contains("] + range(6)") {
                    if let Some(bracket_end) = left.find(']') {
                        let month_str = &left[1..bracket_end];
                        if let Ok(month) = month_str.parse::<i32>() {
                            if let Ok(expected) = right.parse::<f32>() {
                                fixtures.push((month, expected));
                            }
                        }
                    }
                }
            }
        }

        fixtures
    }

    // Parser for month-only fixtures like "(1,) -> 100"
    fn parse_month_fixtures(content: &str) -> Vec<(i32, f32)> {
        let mut fixtures = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if let Some((left, right)) = line.split_once(" -> ") {
                let left = left.trim();
                let right = right.trim();

                // Parse "(month,)" format
                if left.starts_with('(') && left.ends_with(",)") {
                    let month_str = &left[1..left.len() - 2];
                    if let (Ok(month), Ok(result)) =
                        (month_str.parse::<i32>(), right.parse::<f32>())
                    {
                        fixtures.push((month, result));
                    }
                }
            }
        }

        fixtures
    }

    #[test]
    fn test_correct_hcb_for_shading() {
        let content = include_str!("../fixtures/corrige_hcb_por_sombreado.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 7 {
                let result = correct_hcb_for_shading(
                    params[0],        // hcb
                    params[1],        // cloudiness
                    params[2] as i32, // month
                    params[3],        // orientation
                    params[4],        // slope
                    params[5] as i32, // fuel_model
                    params[6] as i32, // hour
                );
                assert_eq!(
                    { result },
                    expected,
                    "correct_hcb_for_shading({}, {}, {}, {}, {}, {}, {}) = {} != {}",
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    params[4],
                    params[5],
                    params[6],
                    result,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_correct_probability_for_precipitation() {
        let content = include_str!("../fixtures/corrige_prob_por_pp.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 2 {
                let result = correct_probability_for_precipitation(
                    params[0], // probability
                    params[1], // precipitation_effect
                );
                assert_eq!(
                    { result },
                    expected,
                    "correct_probability_for_precipitation({}, {}) = {} != {}",
                    params[0],
                    params[1],
                    result,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_d1hr() {
        let content = include_str!("../fixtures/d1hr.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 3 {
                let result = d1hr(
                    params[0], // hcs
                    params[1], // prob_ign_sc
                    params[2], // prob_ign
                );
                assert_eq!(
                    result.round(),
                    expected,
                    "d1hr({}, {}, {}) = {} != {}",
                    params[0],
                    params[1],
                    params[2],
                    result,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_maximum_precipitation_effect() {
        let content = include_str!("../fixtures/efecto_precipitacion_maximo.txt");
        let fixtures = parse_precipitation_fixtures(content);

        for (month, expected) in fixtures {
            // Create test array [0,1,2,3,4,5] as indicated by "range(6)"
            let precipitation = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
            let result = maximum_precipitation_effect(month, &precipitation);
            assert_eq!(
                result, expected,
                "maximum_precipitation_effect({}, {:?}) = {} != {}",
                month, precipitation, result, expected
            );
        }
    }

    #[test]
    fn test_humidity_content_base() {
        let content = include_str!("../fixtures/hcb.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 3 {
                let result = humidity_content_base(
                    params[0],        // temperature
                    params[1],        // relative_humidity
                    params[2] as i32, // hour
                );
                assert_eq!(
                    result as f32, expected,
                    "humidity_content_base({}, {}, {}) = {} != {}",
                    params[0], params[1], params[2], result, expected
                );
            }
        }
    }

    #[test]
    fn test_moisture_content_by_10hr() {
        let content = include_str!("../fixtures/hco_x10hr.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 4 {
                let result = moisture_content_by_10hr(
                    params[0] as i32, // hour
                    params[1],        // moisture
                    params[2],        // moisture_6
                    params[3],        // moisture_15
                );
                assert_eq!(
                    result.round(),
                    expected,
                    "moisture_content_by_10hr({}, {}, {}, {}) = {} != {}",
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    result,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_live_moisture() {
        let content = include_str!("../fixtures/humedad_vivo.txt");
        let fixtures = parse_month_fixtures(content);

        for (month, expected) in fixtures {
            let result = live_moisture(month);
            assert_eq!(
                result, expected,
                "live_moisture({}) = {} != {}",
                month, result, expected
            );
        }
    }

    #[test]
    fn test_ignition_probability() {
        let content = include_str!("../fixtures/probabilidad_ignicion.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 4 {
                let result = ignition_probability(
                    params[0],        // temperature
                    params[1],        // cloudiness
                    params[2],        // hcs
                    params[3] as i32, // fuel_model
                );
                assert_eq!(
                    { result },
                    expected,
                    "ignition_probability({}, {}, {}, {}) = {} != {}",
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    result,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_shading() {
        let content = include_str!("../fixtures/sombreado.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 2 {
                let result = shading(
                    params[0],        // cloudiness
                    params[1] as i32, // fuel_model
                );
                assert_eq!(
                    { result },
                    expected,
                    "shading({}, {}) = {} != {}",
                    params[0],
                    params[1],
                    result,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_humidity_content_base_edge_cases() {
        // Test edge cases not covered in fixtures
        assert_eq!(humidity_content_base(50.0, 0.0, 12), 1);
        assert_eq!(humidity_content_base(-5.0, 100.0, 15), 14);

        // Test zero precipitation
        assert_eq!(_precipitation_effect(5, 0.0, 1), 0.0);

        // Test zero moisture in moisture_content_by_10hr
        assert_eq!(moisture_content_by_10hr(12, 0.0, 10.0, 20.0), 0.0);

        // Test invalid months
        assert!(live_moisture(13).is_nan());
    }
}
