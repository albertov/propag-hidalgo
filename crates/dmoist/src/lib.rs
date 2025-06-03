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
        let indice_pp = if pp >= 1.0 && pp < 5.0 {
            0
        } else if pp >= 5.0 && pp < 15.0 {
            1
        } else if pp >= 15.0 && pp < 35.0 {
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
    } else if temperatura >= 0.0 && temperatura <= 9.0 {
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
        return 0;
    } else if hcs > 0.0 && hcs < 18.0 {
        let sombreado_val = sombreado(nubosidad, modelo_combustible);
        let sombreado_idx = if sombreado_val >= 0.0 && sombreado_val < 10.0 {
            0
        } else if sombreado_val >= 10.0 && sombreado_val < 50.0 {
            1
        } else if sombreado_val >= 50.0 && sombreado_val < 90.0 {
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
    let (hora, sombreado) = if hora_param > 19 || hora_param < 6 {
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
        5 | 6 | 7 | 8 => 0,
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
            } else if efecto_precipitacion >= 5.0 && efecto_precipitacion < 15.0 {
                -10.0
            } else if efecto_precipitacion >= 15.0 && efecto_precipitacion < 35.0 {
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
            if hora >= 0 && hora < 6 {
                a_las_15 + (((a_las_6 - a_las_15) / 15.0) * ((hora + 15) as f64))
            } else if hora >= 6 && hora < 15 {
                a_las_6 + (((a_las_15 - a_las_6) / 9.0) * ((hora - 6) as f64))
            } else if hora >= 15 && hora < 24 {
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
        3 | 4 | 5 => 200.0,
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
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_corrige_hcb_por_sombreado() {
        // Test cases from corrige_hcb_por_sombreado.txt
        assert_eq!(corrige_hcb_por_sombreado(15, 100.0, 4, 0.0, 60.0, 0, 0), 0);
        assert_eq!(
            corrige_hcb_por_sombreado(15, 100.0, 12, 315.0, 45.0, 9, 0),
            20
        );
        assert_eq!(
            corrige_hcb_por_sombreado(0, 100.0, 10, 315.0, 15.0, 1, 0),
            5
        );
        assert_eq!(
            corrige_hcb_por_sombreado(20, 40.0, 5, 225.0, 60.0, 3, 6),
            25
        );
        assert_eq!(
            corrige_hcb_por_sombreado(15, 0.0, 10, 135.0, 45.0, 1, 6),
            20
        );
        assert_eq!(corrige_hcb_por_sombreado(5, 20.0, 9, 90.0, 45.0, 5, 12), 6);
        assert_eq!(corrige_hcb_por_sombreado(0, 60.0, 11, 0.0, 0.0, 3, 6), 5);
        assert_eq!(
            corrige_hcb_por_sombreado(20, 60.0, 3, 0.0, 15.0, 11, 12),
            24
        );
        assert_eq!(corrige_hcb_por_sombreado(15, 20.0, 3, 90.0, 60.0, 6, 6), 20);
        assert_eq!(
            corrige_hcb_por_sombreado(5, 100.0, 6, 315.0, 30.0, 5, 12),
            8
        );
        assert_eq!(
            corrige_hcb_por_sombreado(10, 0.0, 4, 135.0, 15.0, 10, 0),
            15
        );
        assert_eq!(
            corrige_hcb_por_sombreado(10, 60.0, 11, 45.0, 30.0, 2, 12),
            13
        );
        assert_eq!(
            corrige_hcb_por_sombreado(20, 80.0, 5, 135.0, 45.0, 7, 12),
            23
        );
        assert_eq!(
            corrige_hcb_por_sombreado(15, 100.0, 6, 90.0, 30.0, 3, 0),
            20
        );
        assert_eq!(
            corrige_hcb_por_sombreado(10, 20.0, 2, 90.0, 30.0, 4, 18),
            14
        );

        // Test hour out of range - should clamp
        assert_eq!(
            corrige_hcb_por_sombreado(10, 60.0, 11, 135.0, 45.0, 0, 25),
            0
        );
    }

    #[test]
    fn test_corrige_prob_por_pp() {
        // Test cases from corrige_prob_por_pp.txt
        assert_eq!(corrige_prob_por_pp(100, 0.0), 100);
        assert_eq!(corrige_prob_por_pp(100, 1.0), 95);
        assert_eq!(corrige_prob_por_pp(100, 2.0), 95);
        assert_eq!(corrige_prob_por_pp(100, 3.0), 95);
        assert_eq!(corrige_prob_por_pp(100, 4.0), 95);
        assert_eq!(corrige_prob_por_pp(100, 5.0), 90);
        assert_eq!(corrige_prob_por_pp(100, 6.0), 90);
        assert_eq!(corrige_prob_por_pp(100, 15.0), 80);
        assert_eq!(corrige_prob_por_pp(100, 35.0), 70);
        assert_eq!(corrige_prob_por_pp(100, 36.0), 70);
        assert_eq!(corrige_prob_por_pp(0, 15.0), 0);
        assert_eq!(corrige_prob_por_pp(0, 40.0), 0);
        assert_eq!(corrige_prob_por_pp(101, 0.0), 100); // Clamped to 100
    }

    #[test]
    fn test_d1hr() {
        // Test cases from d1hr.txt
        assert_eq!(d1hr(0.0, 0.0, 0.0).round(), 0.0);
        assert_eq!(d1hr(0.0, 0.0, -5.0).round(), 0.0);
        assert_eq!(d1hr(0.0, 10.0, 10.0).round(), 0.0);
        assert_eq!(d1hr(5.0, 0.0, 0.0).round(), 30.0);
        assert_eq!(d1hr(5.0, 10.0, 10.0).round(), 5.0);
        assert_eq!(d1hr(5.0, 10.0, 5.0).round(), 10.0);
        assert_eq!(d1hr(5.0, 20.0, 20.0).round(), 5.0);
        assert_eq!(d1hr(5.0, 20.0, 15.0).round(), 7.0);
        assert_eq!(d1hr(10.0, 10.0, 10.0).round(), 10.0);
        assert_eq!(d1hr(10.0, 20.0, 15.0).round(), 13.0);
        assert_eq!(d1hr(15.0, 20.0, 20.0).round(), 15.0);
        assert_eq!(d1hr(20.0, 20.0, 20.0).round(), 20.0);
        assert_eq!(d1hr(25.0, 20.0, 20.0).round(), 25.0);
        assert_eq!(d1hr(30.0, 30.0, 30.0).round(), 30.0);
        assert_eq!(d1hr(35.0, 40.0, 40.0).round(), 35.0);
    }

    #[test]
    fn test_dir_viento() {
        // Test cases from dir_viento.txt
        assert!(approx_eq(dir_viento(0.0, 0.0), 0.0));
        assert!(approx_eq(dir_viento(1.0, 0.0), 90.0));
        assert!(approx_eq(dir_viento(1.0, 1.0), 45.0));
        assert!(approx_eq(dir_viento(0.0, 1.0), 0.0));
        assert!(approx_eq(dir_viento(0.0, -1.0), 180.0));
        assert!(approx_eq(dir_viento(-1.0, 1.0), 315.0));
        assert!(approx_eq(dir_viento(-1.0, 0.0), 270.0));
    }

    #[test]
    fn test_efecto_precipitacion_maximo() {
        // Test cases from efecto_precipitacion_maximo.txt
        // Testing with [1] + range(6) which means month 1 with values 0,1,2,3,4,5
        assert_eq!(
            efecto_precipitacion_maximo(1, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            3.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(2, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            3.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(3, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            3.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(4, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(5, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(6, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(7, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(8, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(9, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(10, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(11, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            3.0
        );
        assert_eq!(
            efecto_precipitacion_maximo(12, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            3.0
        );
    }

    #[test]
    fn test_hcb() {
        // Test cases from hcb.txt
        assert_eq!(hcb(-20.0, 0.0, 0), 1);
        assert_eq!(hcb(-20.0, 5.0, 0), 2);
        assert_eq!(hcb(-20.0, 10.0, 0), 2);
        assert_eq!(hcb(-20.0, 15.0, 0), 3);
        assert_eq!(hcb(-20.0, 20.0, 0), 4);
        assert_eq!(hcb(-20.0, 25.0, 0), 5);
        assert_eq!(hcb(-20.0, 100.0, 0), 14);
        assert_eq!(hcb(0.0, 0.0, 0), 1);
        assert_eq!(hcb(5.0, 0.0, 0), 1);
        assert_eq!(hcb(10.0, 0.0, 0), 1);
        assert_eq!(hcb(15.0, 0.0, 0), 1);
        assert_eq!(hcb(20.0, 0.0, 0), 1);
        assert_eq!(hcb(25.0, 10.0, 0), 2);
        assert_eq!(hcb(30.0, 100.0, 0), 13);
        assert_eq!(hcb(35.0, 100.0, 0), 13);
        assert_eq!(hcb(40.0, 100.0, 0), 13);
        assert_eq!(hcb(45.0, 100.0, 0), 12);
    }

    #[test]
    fn test_hco_x10hr() {
        // Test cases from hco_x10hr.txt
        assert_eq!(hco_x10hr(1, 10.0, 25.0, 25.0).round(), 25.0);
        assert_eq!(hco_x10hr(6, 10.0, 5.0, 0.0).round(), 2.0);
        assert_eq!(hco_x10hr(23, 10.0, 0.0, 25.0).round(), 14.0);
        assert_eq!(hco_x10hr(19, 5.0, 20.0, 25.0).round(), 23.0);
        assert_eq!(hco_x10hr(8, 20.0, 20.0, 0.0).round(), 8.0);
        assert_eq!(hco_x10hr(15, 25.0, 0.0, 25.0).round(), 14.0);
        assert_eq!(hco_x10hr(1, 20.0, 15.0, 20.0).round(), 18.0);
        assert_eq!(hco_x10hr(3, 15.0, 5.0, 25.0).round(), 17.0);
        assert_eq!(hco_x10hr(6, 10.0, 0.0, 20.0).round(), 12.0);
        assert_eq!(hco_x10hr(22, 5.0, 25.0, 20.0).round(), 22.0);
        assert_eq!(hco_x10hr(9, 0.0, 20.0, 25.0).round(), 0.0);
        assert_eq!(hco_x10hr(0, 5.0, 0.0, 5.0).round(), 3.0);
    }

    #[test]
    fn test_humedad_vivo() {
        // Test cases from humedad_vivo.txt
        assert_eq!(humedad_vivo(1), 100.0);
        assert_eq!(humedad_vivo(2), 100.0);
        assert_eq!(humedad_vivo(3), 200.0);
        assert_eq!(humedad_vivo(4), 200.0);
        assert_eq!(humedad_vivo(5), 200.0);
        assert_eq!(humedad_vivo(6), 100.0);
        assert_eq!(humedad_vivo(7), 80.0);
        assert_eq!(humedad_vivo(8), 80.0);
        assert_eq!(humedad_vivo(9), 90.0);
        assert_eq!(humedad_vivo(10), 90.0);
        assert_eq!(humedad_vivo(11), 100.0);
        assert_eq!(humedad_vivo(12), 100.0);
        assert!(humedad_vivo(13).is_nan()); // Invalid month
    }

    #[test]
    fn test_mod_viento() {
        // Test cases from mod_viento.txt
        let test_cases = [
            ((-3.0, -3.0), 4.2426406871192848),
            ((-3.0, -2.0), 3.6055512754639891),
            ((-3.0, -1.0), 3.1622776601683795),
            ((-3.0, 0.0), 3.0),
            ((-3.0, 1.0), 3.1622776601683795),
            ((-2.0, -2.0), 2.8284271247461903),
            ((-1.0, -1.0), 1.4142135623730951),
            ((0.0, 0.0), 0.0),
            ((1.0, 1.0), 1.4142135623730951),
            ((2.0, 2.0), 2.8284271247461903),
        ];

        for ((u, v), expected) in test_cases.iter() {
            assert!(approx_eq(mod_viento(*u, *v), *expected));
        }
    }

    #[test]
    fn test_probabilidad_ignicion() {
        // Test cases from probabilidad_ignicion.txt
        assert_eq!(probabilidad_ignicion(0.0, 40.0, 7.0, 1), 40);
        assert_eq!(probabilidad_ignicion(0.0, 0.0, 8.0, 9), 30);
        assert_eq!(probabilidad_ignicion(20.0, 20.0, 23.0, 8), 0); // hcs >= 18
        assert_eq!(probabilidad_ignicion(20.0, 60.0, 11.0, 4), 20);
        assert_eq!(probabilidad_ignicion(35.0, 70.0, 18.0, 7), 0); // hcs >= 18
        assert_eq!(probabilidad_ignicion(-10.0, 100.0, 24.0, 12), 0); // hcs >= 18
        assert_eq!(probabilidad_ignicion(15.0, 30.0, 8.0, 11), 40);
        assert_eq!(probabilidad_ignicion(45.0, 20.0, 20.0, 8), 0); // hcs >= 18
        assert_eq!(probabilidad_ignicion(0.0, 20.0, 11.0, 3), 20);
        assert_eq!(probabilidad_ignicion(25.0, 30.0, 18.0, 9), 0); // hcs >= 18
        assert_eq!(probabilidad_ignicion(20.0, 100.0, 18.0, 0), 0); // modelo_combustible == 0
        assert_eq!(probabilidad_ignicion(25.0, 90.0, 1.0, 0), 0); // modelo_combustible == 0
    }

    #[test]
    fn test_sombreado() {
        // Test cases from sombreado.txt
        assert_eq!(sombreado(8.0, 7), 75.0);
        assert_eq!(sombreado(0.0, 0), 0.0);
        assert_eq!(sombreado(0.0, 7), 75.0);
        assert_eq!(sombreado(5.0, 1), 3.75);
        assert_eq!(sombreado(10.0, 7), 75.0);
        assert_eq!(sombreado(15.0, 3), 11.25);
        assert_eq!(sombreado(20.0, 7), 75.0);
        assert_eq!(sombreado(25.0, 5), 18.75);
        assert_eq!(sombreado(30.0, 8), 75.0);
        assert_eq!(sombreado(35.0, 6), 26.25);
        assert_eq!(sombreado(40.0, 9), 75.0);
        assert_eq!(sombreado(45.0, 4), 33.75);
        assert_eq!(sombreado(50.0, 10), 75.0);
        assert_eq!(sombreado(55.0, 2), 41.25);
        assert_eq!(sombreado(60.0, 11), 75.0);
        assert_eq!(sombreado(65.0, 1), 48.75);
        assert_eq!(sombreado(70.0, 7), 100.0); // >= 50 && modelo 7-12
        assert_eq!(sombreado(75.0, 8), 100.0); // >= 50 && modelo 7-12
        assert_eq!(sombreado(100.0, 1), 75.0);
        assert_eq!(sombreado(105.0, 1), 78.75);
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
