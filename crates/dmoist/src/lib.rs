use std::f64::consts::PI;

fn max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

fn min(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

/// Returns the effect of a certain precipitation as a function
/// of the month and how many days have passed since the day being
/// calculated.
///
/// Params:
///   * mes [1..12]
///   * pp (ml)
///   * offset of the day [0..5] (0 day before prediction)
fn _efecto_precipitacion(mes: i32, pp: f64, offset: usize) -> f64 {
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

    if pp > 0.0 {
        let indice_mes = if (4..=10).contains(&mes) { 0 } else { 1 };
        let indice_pp = if (1.0..5.0).contains(&pp) {
            0
        } else if (5.0..15.0).contains(&pp) {
            1
        } else if (15.0..35.0).contains(&pp) {
            2
        } else {
            3
        };

        let efecto = tabla[indice_mes][indice_pp][offset];
        if efecto > -1.0 {
            return efecto;
        }
    }
    pp
}

fn max_f64_iter<I>(iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    iter.fold(f64::NEG_INFINITY, |a, b| a.max(b))
}

pub fn efecto_precipitacion_maximo(mes: i32, pp: &[f64; 6]) -> f64 {
    // Returns the maximum precipitation effect of the last 6 days
    max_f64_iter((0..6).map(|i| _efecto_precipitacion(mes, pp[i], i)))
}

pub fn hcb(temperatura: f64, humedad_relativa: f64, _hora: i32) -> i32 {
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
    let humedad_idx = max(0.0, min(20.0, (humedad_relativa / 5.0).floor())) as usize;
    let temperatura_idx = if temperatura < 0.0 {
        0
    } else if (0.0..=9.0).contains(&temperatura) {
        1
    } else if temperatura > 9.0 && temperatura <= 20.0 {
        2
    } else if temperatura > 20.0 && temperatura <= 31.0 {
        3
    } else if temperatura > 31.0 && temperatura <= 42.0 {
        4
    } else if temperatura > 42.0 {
        5
    } else {
        0 // fallback
    };

    tabla[hora_idx][temperatura_idx][humedad_idx]
}

pub fn sombreado(nubosidad: f64, modelo_combustible: i32) -> f64 {
    let mut sombreado = nubosidad * 0.75;
    if (7..=12).contains(&modelo_combustible) {
        sombreado = if sombreado < 50.0 { 75.0 } else { 100.0 };
    }
    sombreado
}

pub fn probabilidad_ignicion(
    temperatura: f64,
    nubosidad: f64,
    hcs: f64,
    modelo_combustible: i32,
) -> i32 {
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

    if modelo_combustible == 0 {
        0
    } else if hcs > 0.0 && hcs < 18.0 {
        let sombreado_val = sombreado(nubosidad, modelo_combustible);
        let sombreado_idx = if (0.0..10.0).contains(&sombreado_val) {
            0
        } else if (10.0..50.0).contains(&sombreado_val) {
            1
        } else if (50.0..90.0).contains(&sombreado_val) {
            2
        } else {
            3
        };

        let temperatura_idx = max(0.0, min(8.0, 9.0 - (temperatura / 5.0).ceil())) as usize;
        let hcs_idx = (hcs.floor() as usize) - 1;

        return tabla[sombreado_idx][temperatura_idx][hcs_idx];
    } else {
        return 0;
    }
}

fn _es_norte(orientacion: f64) -> bool {
    (315.0..=360.0).contains(&orientacion) || (0.0..45.0).contains(&orientacion)
}

fn _es_este(orientacion: f64) -> bool {
    (45.0..135.0).contains(&orientacion)
}

fn _es_sur(orientacion: f64) -> bool {
    (135.0..225.0).contains(&orientacion)
}

fn _es_oeste(orientacion: f64) -> bool {
    (225.0..315.0).contains(&orientacion)
}

fn _corrector_por_sombreado(
    orientacion: f64,
    pendiente: f64,
    sombreado_val: f64,
    mes: i32,
    hora_param: i32,
) -> i32 {
    // Between 20 and 6 (pseudo-dawn) apply values of 20,
    // between 6 and 8. It's an attempt to make the variation come out
    // continuous since the tables don't contemplate correction at night but
    // they are wrong
    let (hora, sombreado) = if !(6..=19).contains(&hora_param) {
        (19, 100.0)
    } else if hora_param < 8 {
        (8, 100.0)
    } else {
        (hora_param, sombreado_val)
    };

    // index shading
    let sombreado_idx = if sombreado < 50.0 { 0 } else { 1 };

    // index hour
    let hora_idx = min(5.0, ((hora - 8) / 2).into()) as usize;

    // Index month
    let mes_idx = match mes {
        5..=8 => 0,
        2 | 3 | 4 | 9 | 10 => 1,
        11 | 12 | 1 => 2,
        _ => return 0, // invalid month
    };

    // Flat terrain is south orientation
    let orientacion_adjusted = if pendiente == 0.0 { 180.0 } else { orientacion };

    // Index slope/orientation
    let orientacion_pendiente_idx = if _es_norte(orientacion_adjusted) {
        if pendiente <= 30.0 {
            0
        } else {
            1
        }
    } else if _es_este(orientacion_adjusted) {
        if pendiente <= 30.0 {
            2
        } else {
            3
        }
    } else if _es_sur(orientacion_adjusted) {
        if pendiente <= 30.0 {
            4
        } else {
            5
        }
    } else if _es_oeste(orientacion_adjusted) {
        if pendiente <= 30.0 {
            6
        } else {
            7
        }
    } else {
        return 0; // invalid orientation
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

    tabla[mes_idx][sombreado_idx][orientacion_pendiente_idx][hora_idx]
}

pub fn corrige_hcb_por_sombreado(
    hcb: i32,
    nubosidad: f64,
    mes: i32,
    orientacion: f64,
    pendiente: f64,
    modelo_combustible: i32,
    hora: i32,
) -> i32 {
    if modelo_combustible == 0 {
        0
    } else {
        let sombreado_val = sombreado(nubosidad, modelo_combustible);
        hcb + _corrector_por_sombreado(orientacion, pendiente, sombreado_val, mes, hora)
    }
}

pub fn corrige_prob_por_pp(prob: i32, efecto_precipitacion: f64) -> i32 {
    // Corrects probability by precipitation effect
    // The following if-elif with formula is:
    // PROBIG - Int(2.61863 + (0.988897 * PP6) - (0.00632351 * ((PP6) ^ 2)))
    if prob == 0 {
        0
    } else {
        let prob_ajustada = prob as f64
            + if efecto_precipitacion > 0.0 && efecto_precipitacion < 5.0 {
                -5.0
            } else if (5.0..15.0).contains(&efecto_precipitacion) {
                -10.0
            } else if (15.0..35.0).contains(&efecto_precipitacion) {
                -20.0
            } else if efecto_precipitacion >= 35.0 {
                -30.0
            } else {
                0.0
            };

        max(0.0, min(100.0, prob_ajustada)) as i32
    }
}

pub fn d1hr(hcs: f64, prob_ign_sc: f64, prob_ign: f64) -> f64 {
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

// Calculates the humidity of the thickest fuel, i.e., if 'humedad' is
// 1hr returns 10hr, if 10hr -> 100hr.

// @param hora: The hour of the day
// @param humedad: The humidity of the finest fuel
// @param humedad_6: The humidity of the finest fuel at 6 hours
// @param humedad_15: The humidity of the finest fuel at 15 hours
pub fn hco_x10hr(hora: i32, humedad: f64, humedad_6: f64, humedad_15: f64) -> f64 {
    if humedad == 0.0 {
        0.0
    } else {
        let a_las_6 = humedad_15 + (0.4142 * (humedad_6 - humedad_15));
        let a_las_15 = humedad_6 - (0.5571 * (humedad_6 - humedad_15));

        if (a_las_6 - a_las_15).abs() < 0.001 {
            if a_las_6 < 0.001 {
                humedad
            } else {
                a_las_6
            }
        } else {
            // interpolate the value based on the hour of the day
            if (0..6).contains(&hora) {
                a_las_15 + (((a_las_6 - a_las_15) / 15.0) * ((hora + 15) as f64))
            } else if (6..15).contains(&hora) {
                a_las_6 + (((a_las_15 - a_las_6) / 9.0) * ((hora - 6) as f64))
            } else if (15..24).contains(&hora) {
                a_las_15 + (((a_las_6 - a_las_15) / 15.0) * ((hora - 15) as f64))
            } else {
                humedad // fallback for invalid hour
            }
        }
    }
}

// Live fuel moisture estimation based on month
pub fn humedad_vivo(mes: i32) -> f64 {
    match mes {
        1 | 2 => 100.0,
        3..=5 => 200.0,
        6 => 100.0,
        7 | 8 => 80.0,
        9 | 10 => 90.0,
        11 | 12 => 100.0,
        _ => f64::NAN,
    }
}

pub fn dir_viento(u: f64, v: f64) -> f64 {
    let dir = u.atan2(v) * 180.0 / PI;
    if dir >= 0.0 {
        dir
    } else {
        dir + 360.0
    }
}

pub fn mod_viento(u: f64, v: f64) -> f64 {
    (u * u + v * v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    // Parser for simple tuple -> value mappings
    fn parse_simple_fixtures(content: &str) -> Vec<(Vec<f64>, f64)> {
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
                    let params: Vec<f64> = inner
                        .split(',')
                        .map(|s| s.trim().parse().unwrap_or(0.0))
                        .collect();
                    let result: f64 = right.parse().unwrap_or(0.0);
                    fixtures.push((params, result));
                }
            }
        }

        fixtures
    }

    // Parser for month + range fixtures like "[1] + range(6) -> 3"
    fn parse_precipitation_fixtures(content: &str) -> Vec<(i32, f64)> {
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
                            if let Ok(expected) = right.parse::<f64>() {
                                fixtures.push((month, expected));
                            }
                        }
                    }
                }
            }
        }

        fixtures
    }

    // Parser for wind direction fixtures like "0, 0 -> 0"
    fn parse_wind_fixtures(content: &str) -> Vec<((f64, f64), f64)> {
        let mut fixtures = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if let Some((left, right)) = line.split_once(" -> ") {
                let left = left.trim();
                let right = right.trim();

                // Parse "u, v" format
                if let Some(comma_pos) = left.find(',') {
                    let u_str = left[..comma_pos].trim();
                    let v_str = left[comma_pos + 1..].trim();

                    if let (Ok(u), Ok(v), Ok(result)) = (
                        u_str.parse::<f64>(),
                        v_str.parse::<f64>(),
                        right.parse::<f64>(),
                    ) {
                        fixtures.push(((u, v), result));
                    }
                }
            }
        }

        fixtures
    }

    // Parser for month-only fixtures like "(1,) -> 100"
    fn parse_month_fixtures(content: &str) -> Vec<(i32, f64)> {
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
                        (month_str.parse::<i32>(), right.parse::<f64>())
                    {
                        fixtures.push((month, result));
                    }
                }
            }
        }

        fixtures
    }

    #[test]
    fn test_corrige_hcb_por_sombreado() {
        let content = include_str!("../fixtures/corrige_hcb_por_sombreado.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 7 {
                let result = corrige_hcb_por_sombreado(
                    params[0] as i32, // hcb
                    params[1],        // nubosidad
                    params[2] as i32, // mes
                    params[3],        // orientacion
                    params[4],        // pendiente
                    params[5] as i32, // modelo_combustible
                    params[6] as i32, // hora
                );
                assert_eq!(
                    result as f64,
                    expected,
                    "corrige_hcb_por_sombreado({}, {}, {}, {}, {}, {}, {}) = {} != {}",
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
    fn test_corrige_prob_por_pp() {
        let content = include_str!("../fixtures/corrige_prob_por_pp.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 2 {
                let result = corrige_prob_por_pp(
                    params[0] as i32, // prob
                    params[1],        // efecto_pp
                );
                assert_eq!(
                    result as f64, expected,
                    "corrige_prob_por_pp({}, {}) = {} != {}",
                    params[0], params[1], result, expected
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
    fn test_dir_viento() {
        let content = include_str!("../fixtures/dir_viento.txt");
        let fixtures = parse_wind_fixtures(content);

        for ((u, v), expected) in fixtures {
            let result = dir_viento(u, v);
            assert!(
                approx_eq(result, expected),
                "dir_viento({}, {}) = {} != {}",
                u,
                v,
                result,
                expected
            );
        }
    }

    #[test]
    fn test_efecto_precipitacion_maximo() {
        let content = include_str!("../fixtures/efecto_precipitacion_maximo.txt");
        let fixtures = parse_precipitation_fixtures(content);

        for (mes, expected) in fixtures {
            // Create test array [0,1,2,3,4,5] as indicated by "range(6)"
            let pp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
            let result = efecto_precipitacion_maximo(mes, &pp);
            assert_eq!(
                result, expected,
                "efecto_precipitacion_maximo({}, {:?}) = {} != {}",
                mes, pp, result, expected
            );
        }
    }

    #[test]
    fn test_hcb() {
        let content = include_str!("../fixtures/hcb.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 3 {
                let result = hcb(
                    params[0],        // temperatura
                    params[1],        // humedad_relativa
                    params[2] as i32, // hora
                );
                assert_eq!(
                    result as f64, expected,
                    "hcb({}, {}, {}) = {} != {}",
                    params[0], params[1], params[2], result, expected
                );
            }
        }
    }

    #[test]
    fn test_hco_x10hr() {
        let content = include_str!("../fixtures/hco_x10hr.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 4 {
                let result = hco_x10hr(
                    params[0] as i32, // hora
                    params[1],        // humedad
                    params[2],        // humedad_6
                    params[3],        // humedad_15
                );
                assert_eq!(
                    result.round(),
                    expected,
                    "hco_x10hr({}, {}, {}, {}) = {} != {}",
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
    fn test_humedad_vivo() {
        let content = include_str!("../fixtures/humedad_vivo.txt");
        let fixtures = parse_month_fixtures(content);

        for (mes, expected) in fixtures {
            let result = humedad_vivo(mes);
            assert_eq!(
                result, expected,
                "humedad_vivo({}) = {} != {}",
                mes, result, expected
            );
        }
    }

    #[test]
    fn test_mod_viento() {
        let content = include_str!("../fixtures/mod_viento.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 2 {
                let result = mod_viento(params[0], params[1]);
                assert!(
                    approx_eq(result, expected),
                    "mod_viento({}, {}) = {} != {}",
                    params[0],
                    params[1],
                    result,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_probabilidad_ignicion() {
        let content = include_str!("../fixtures/probabilidad_ignicion.txt");
        let fixtures = parse_simple_fixtures(content);

        for (params, expected) in fixtures {
            if params.len() == 4 {
                let result = probabilidad_ignicion(
                    params[0],        // temperatura
                    params[1],        // nubosidad
                    params[2],        // hcs
                    params[3] as i32, // modelo_combustible
                );
                assert_eq!(
                    result as f64, expected,
                    "probabilidad_ignicion({}, {}, {}, {}) = {} != {}",
                    params[0], params[1], params[2], params[3], result, expected
                );
            }
        }
    }

    #[test]
    fn test_sombreado() {
        let content = include_str!("../fixtures/sombreado.txt");

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if let Some((left, right)) = line.split_once(" -> ") {
                let left = left.trim();
                let right = right.trim();

                // Handle special case "8, 7 -> 75" (without parentheses)
                if left.contains(", ") && !left.starts_with('(') {
                    if let Some(comma_pos) = left.find(", ") {
                        let nub_str = left[..comma_pos].trim();
                        let mod_str = left[comma_pos + 2..].trim();

                        if let (Ok(nubosidad), Ok(modelo), Ok(expected)) = (
                            nub_str.parse::<f64>(),
                            mod_str.parse::<i32>(),
                            right.parse::<f64>(),
                        ) {
                            let result = sombreado(nubosidad, modelo);
                            assert_eq!(
                                result, expected,
                                "sombreado({}, {}) = {} != {}",
                                nubosidad, modelo, result, expected
                            );
                        }
                    }
                }
                // Handle normal tuple format "(nub, mod) -> result"
                else if left.starts_with('(') && left.ends_with(')') {
                    let inner = &left[1..left.len() - 1];
                    if let Some(comma_pos) = inner.find(',') {
                        let nub_str = inner[..comma_pos].trim();
                        let mod_str = inner[comma_pos + 1..].trim();

                        if let (Ok(nubosidad), Ok(modelo), Ok(expected)) = (
                            nub_str.parse::<f64>(),
                            mod_str.parse::<i32>(),
                            right.parse::<f64>(),
                        ) {
                            let result = sombreado(nubosidad, modelo);
                            assert_eq!(
                                result, expected,
                                "sombreado({}, {}) = {} != {}",
                                nubosidad, modelo, result, expected
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_basic_functions() {
        // Test max and min functions
        assert_eq!(max(5.0, 3.0), 5.0);
        assert_eq!(max(3.0, 5.0), 5.0);
        assert_eq!(min(5.0, 3.0), 3.0);
        assert_eq!(min(3.0, 5.0), 3.0);
    }

    #[test]
    fn test_edge_cases() {
        // Test modelo_combustible == 0 cases
        assert_eq!(probabilidad_ignicion(25.0, 50.0, 10.0, 0), 0);
        assert_eq!(
            corrige_hcb_por_sombreado(15, 50.0, 6, 180.0, 30.0, 0, 12),
            0
        );

        // Test zero precipitation
        assert_eq!(_efecto_precipitacion(5, 0.0, 1), 0.0);

        // Test zero humidity in hco_x10hr
        assert_eq!(hco_x10hr(12, 0.0, 10.0, 20.0), 0.0);

        // Test invalid months
        assert!(humedad_vivo(0).is_nan());
        assert!(humedad_vivo(13).is_nan());
    }
}
