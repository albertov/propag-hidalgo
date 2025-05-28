#![no_std]

#[cfg(test)]
#[cfg(not(target_os = "cuda"))]
extern crate std;

use approx::abs_diff_ne;

pub use approx::AbsDiffEq;

use core::ffi::c_char;
use core::ffi::CStr;
pub use glam::*;
pub use num_traits::NumCast;
#[cfg(target_os = "cuda")]
extern crate cuda_std;

extern crate cust_core;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
#[repr(C)]
pub struct GT {
    pub x0: f32,
    dx: f32,
    rx: f32,
    y0: f32,
    ry: f32,
    dy: f32,
}

impl GT {
    pub fn new(xs: [f32; 6]) -> Self {
        Self {
            x0: xs[0],
            dx: xs[1],
            rx: xs[2],
            y0: xs[3],
            ry: xs[4],
            dy: xs[5],
        }
    }
    pub fn as_array_64(&self) -> [f64; 6] {
        let GT {
            x0,
            dx,
            rx,
            y0,
            ry,
            dy,
        } = *self;
        [
            x0.into(),
            dx.into(),
            rx.into(),
            y0.into(),
            ry.into(),
            dy.into(),
        ]
    }
    pub fn as_array(&self) -> [f32; 6] {
        let GT {
            x0,
            dx,
            rx,
            y0,
            ry,
            dy,
        } = *self;
        [x0, dx, rx, y0, ry, dy]
    }
}

#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
#[repr(C)]
pub struct GeoTransform {
    pub gt: GT,
    pub inv: GT,
}

impl GeoTransform {
    pub fn new(xs: [f32; 6]) -> Option<Self> {
        Some(Self {
            gt: GT::new(xs),
            inv: Self::invert(xs)?,
        })
    }
    pub fn is_north_up(&self) -> bool {
        self.dy() < 0.0
    }
    pub fn origin(&self) -> Vec2 {
        Vec2 {
            x: self.gt.x0,
            y: self.gt.y0,
        }
    }
    pub fn dx(&self) -> f32 {
        self.gt.dx
    }
    pub fn dy(&self) -> f32 {
        self.gt.dy
    }
    pub fn as_array_64(&self) -> [f64; 6] {
        self.gt.as_array_64()
    }
    pub fn as_array(&self) -> [f32; 6] {
        self.gt.as_array()
    }
    pub fn forward(&self, p: Vec2) -> USizeVec2 {
        let Vec2 { x, y } = p - self.origin();
        let GT { dx, rx, ry, dy, .. } = self.inv;
        let p = Vec2 {
            x: x * dx + y * rx,
            y: x * ry + y * dy,
        };
        USizeVec2 {
            x: NumCast::from(p.x.trunc()).expect("T to usize failed"),
            y: NumCast::from(p.y.trunc()).expect("T to usize failed"),
        }
    }
    pub fn backward(&self, p: USizeVec2) -> Vec2 {
        let p = Vec2 {
            x: <f32 as NumCast>::from(p.x).expect("T from usize failed"),
            y: <f32 as NumCast>::from(p.y).expect("T to from failed"),
        };
        let Vec2 { x, y } = p + self.origin();
        let GT { dx, rx, ry, dy, .. } = self.gt;
        Vec2 {
            x: x * dx + y * rx,
            y: x * ry + y * dy,
        }
    }
    fn invert(gt: [f32; 6]) -> Option<GT> {
        let [x0, a, c, y0, b, d] = gt;
        let det = a * d - b * c;
        if abs_diff_ne!(det.abs(), 0.0) {
            let f = det.recip();
            let a2 = d * f;
            let b2 = -b * f;
            let c2 = -c * f;
            let d2 = a * f;
            Some(GT::new([x0, a2, c2, y0, b2, d2]))
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, cust_core::DeviceCopy)]
#[repr(C)]
pub struct GeoReference {
    pub width: u32,
    pub height: u32,
    pub proj: [u8; 1024],
    pub transform: GeoTransform,
}

impl GeoReference {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        self.width * self.height
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn GeoReference_len(&self) -> u32 {
        self.len()
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn GeoReference_x0(&self) -> f32 {
        if self.transform.dx() > 0.0 {
            self.transform.gt.x0
        } else {
            self.transform.gt.x0 + self.transform.dx() * (self.width as f32)
        }
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn GeoReference_x1(&self) -> f32 {
        if self.transform.dx() > 0.0 {
            self.transform.gt.x0 + self.transform.dx() * (self.width as f32)
        } else {
            self.transform.gt.x0
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn GeoReference_y0(&self) -> f32 {
        if self.transform.dy() > 0.0 {
            self.transform.gt.y0
        } else {
            self.transform.gt.y0 + self.transform.dy() * (self.height as f32)
        }
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn GeoReference_y1(&self) -> f32 {
        if self.transform.dy() > 0.0 {
            self.transform.gt.y0 + self.transform.dy() * (self.height as f32)
        } else {
            self.transform.gt.y0
        }
    }
    pub fn forward(&self, p: Vec2) -> USizeVec2 {
        self.transform.forward(p)
    }
    pub fn backward(&self, p: USizeVec2) -> Vec2 {
        self.transform.backward(p)
    }
    pub fn south_up(bbox: (Vec2, Vec2), pixel_size: Vec2, proj: &str) -> Option<GeoReference> {
        let Vec2 { x: dx, y: dy } = pixel_size;
        let (Vec2 { x: x0, y: y0 }, Vec2 { x: x1, y: y1 }) = bbox;
        let transform = GeoTransform::new([x0, dx, 0.0, y0, 0.0, dy])?;
        let width = NumCast::from(((x1 - x0) / dx).ceil())?;
        let height = NumCast::from(((y1 - y0) / dy).ceil())?;
        let mut proj_buf: [u8; 1024] = [0; 1024];
        (0..proj.len())
            .zip(proj.as_bytes().iter())
            .for_each(|(i, c)| {
                proj_buf[i] = *c;
            });
        Some(GeoReference {
            transform,
            width,
            height,
            proj: proj_buf,
        })
    }
    pub fn north_up(bbox: (Vec2, Vec2), pixel_size: Vec2, proj: &str) -> Option<GeoReference> {
        let Vec2 { x: dx, y: dy } = pixel_size;
        let (Vec2 { x: x0, y: y0 }, Vec2 { x: x1, y: y1 }) = bbox;
        let transform = GeoTransform::new([x0, dx, 0.0, y1, 0.0, -dy])?;
        let width = NumCast::from(((x1 - x0) / dx).ceil())?;
        let height = NumCast::from(((y1 - y0) / dy).ceil())?;
        let mut proj_buf: [u8; 1024] = [0; 1024];
        (0..proj.len())
            .zip(proj.as_bytes().iter())
            .for_each(|(i, c)| {
                proj_buf[i] = *c;
            });
        Some(GeoReference {
            transform,
            width,
            height,
            proj: proj_buf,
        })
    }
    #[unsafe(no_mangle)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe extern "C" fn GeoReference_south_up(
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        dx: f32,
        dy: f32,
        proj: *const c_char,
        result: &mut GeoReference,
    ) -> bool {
        if let Some(proj) = unsafe { CStr::from_ptr(proj).to_str().ok() } {
            match Self::south_up(
                (Vec2 { x: x0, y: y0 }, Vec2 { x: x1, y: y1 }),
                Vec2 { x: dx, y: dy },
                proj,
            ) {
                Some(x) => {
                    *result = x;
                    true
                }
                None => false,
            }
        } else {
            false
        }
    }
    #[unsafe(no_mangle)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe extern "C" fn GeoReference_north_up(
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        dx: f32,
        dy: f32,
        proj: *const c_char,
        result: &mut GeoReference,
    ) -> bool {
        if let Some(proj) = unsafe { CStr::from_ptr(proj).to_str().ok() } {
            match Self::north_up(
                (Vec2 { x: x0, y: y0 }, Vec2 { x: x1, y: y1 }),
                Vec2 { x: dx, y: dy },
                proj,
            ) {
                Some(x) => {
                    *result = x;
                    true
                }
                None => false,
            }
        } else {
            false
        }
    }
    pub fn bbox(&self) -> (Vec2, Vec2) {
        let Self {
            transform,
            width,
            height,
            ..
        } = self;
        let width = <f32 as NumCast>::from(*width).expect("T from u32 failed");
        let height = <f32 as NumCast>::from(*height).expect("T from u32 failed");
        //assert!(dx > 0.0 && dy > 0.0 && rx == 0.0 && ry == 0.0);
        let origin = transform.origin();
        (
            origin,
            Vec2 {
                x: transform.dx() * width,
                y: transform.dy() * height,
            },
        )
    }
    pub fn is_north_up(&self) -> bool {
        self.transform.is_north_up()
    }

    // Azimuth (radian)
    pub fn bearing(&self, a: Vec2, b: Vec2) -> f32 {
        if self.is_north_up() {
            (b.x - a.x).atan2(a.y - b.y)
        } else {
            (b.x - a.x).atan2(b.y - a.y)
        }
    }
    pub fn distance(&self, a: Vec2, b: Vec2) -> f32 {
        (((a.x - b.x) * self.transform.dx()).powi(2) + ((a.y - b.y) * self.transform.dy()).powi(2))
            .sqrt()
    }
}

/*
pub struct Raster<'a, T> {
    geo_ref: GeoReference<f64>,
    data: &'a [T],
}
*/

#[repr(C)]
#[allow(clippy::upper_case_acronyms)]
struct DDA {
    dest: IVec2,
    step: IVec2,
    cur: Option<IVec2>,
    t_max: DVec2,
    delta: DVec2,
}

impl DDA {
    fn new(origin: IVec2, dest: IVec2) -> Self {
        let step = (dest - origin).signum();
        let t_max = (dest - origin).abs();
        let t_max = if t_max.x != 0 && t_max.y != 0 {
            DVec2 {
                x: 1.0 / (t_max.x as f64),
                y: 1.0 / (t_max.y as f64),
            }
        } else {
            DVec2 { x: 0.0, y: 0.0 }
        };
        let cur = Some(origin);
        DDA {
            dest,
            cur,
            t_max,
            delta: t_max,
            step,
        }
    }
}

impl Iterator for DDA {
    type Item = IVec2;

    fn next(&mut self) -> Option<Self::Item> {
        let valid_x = |x: i32| {
            if self.step.x > 0 {
                x <= self.dest.x
            } else {
                x >= self.dest.x
            }
        };
        let valid_y = |y: i32| {
            if self.step.y > 0 {
                y <= self.dest.y
            } else {
                y >= self.dest.y
            }
        };
        match self.cur {
            None => None,
            Some(vec) if vec == self.dest => {
                self.cur = None;
                Some(vec)
            }
            Some(IVec2 { x, y }) if !(valid_x(x) && valid_y(y)) => {
                self.cur = None;
                None
            }
            Some(IVec2 { x, y }) if self.t_max.x == self.t_max.y => {
                let ret = self.cur;
                self.cur = Some(IVec2 {
                    x: x + self.step.x,
                    y: y + self.step.y,
                });
                self.t_max += self.delta;
                ret
            }
            Some(IVec2 { x, y }) if self.t_max.x < self.t_max.y => {
                let ret = self.cur;
                self.cur = Some(IVec2 {
                    x: self.step.x + x,
                    y,
                });
                self.t_max = DVec2 {
                    x: self.t_max.x + self.delta.x,
                    y: self.t_max.y,
                };
                ret
            }
            Some(IVec2 { x, y }) => {
                let ret = self.cur;
                self.cur = Some(IVec2 {
                    x,
                    y: self.step.y + y,
                });
                self.t_max = DVec2 {
                    x: self.t_max.x,
                    y: self.t_max.y + self.delta.y,
                };
                ret
            }
        }
    }
}

pub fn line_to(from: IVec2, to: IVec2) -> impl Iterator<Item = IVec2> {
    DDA::new(from, to)
}
pub fn neighbor_in_direction(from: IVec2, to: IVec2) -> IVec2 {
    if from == to {
        from
    } else {
        let IVec2 { x, y } = from;
        let step = (to - from).signum();
        let t_max = (to - from).abs();
        // allow comparison_chain beacuse we're no_std
        #[allow(clippy::comparison_chain)]
        if t_max.x == t_max.y {
            IVec2 {
                x: x + step.x,
                y: y + step.y,
            }
        } else if t_max.x > t_max.y {
            IVec2 { x: step.x + x, y }
        } else {
            IVec2 { x, y: step.y + y }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::f32::consts::PI;
    use std::vec::*;
    use std::*;

    #[test]
    fn neighbor_in_direction_same_as_line_to() {
        let p = IVec2 { x: 0, y: 0 };
        let cross = (-1000..1000).flat_map(|y| (-1000..1000).map(move |x| (x, y)));
        cross.for_each(|(x, y)| {
            if x != 0 && y != 0 {
                let p2 = IVec2 { x, y };
                assert_eq!(line_to(p, p2).nth(1), Some(neighbor_in_direction(p, p2)));
            }
        });
    }
    #[test]
    fn can_line_to_self() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 0, y: 0 }).collect();
        assert_eq!(points, vec![(IVec2 { x: 0, y: 0 })]);
    }
    #[test]
    fn can_line_to_north() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 0, y: 2 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: 0, y: 1 }),
                (IVec2 { x: 0, y: 2 })
            ]
        );
    }
    #[test]
    fn can_line_to_north2() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 0, y: 1 }).collect();
        assert_eq!(
            points,
            vec![(IVec2 { x: 0, y: 0 }), (IVec2 { x: 0, y: 1 }),]
        );
    }
    #[test]
    fn can_line_to_south() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 0, y: -2 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: 0, y: -1 }),
                (IVec2 { x: 0, y: -2 })
            ]
        );
    }
    #[test]
    fn can_line_to_south2() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 0, y: -1 }).collect();
        assert_eq!(
            points,
            vec![(IVec2 { x: 0, y: 0 }), (IVec2 { x: 0, y: -1 }),]
        );
    }
    #[test]
    fn can_line_to_east() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 2, y: 0 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: 1, y: 0 }),
                (IVec2 { x: 2, y: 0 })
            ]
        );
    }
    #[test]
    fn can_line_to_west() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: -2, y: 0 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: -1, y: 0 }),
                (IVec2 { x: -2, y: 0 })
            ]
        );
    }
    #[test]
    fn can_line_to_north_west() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: -2, y: -2 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: -1, y: -1 }),
                (IVec2 { x: -2, y: -2 })
            ]
        );
    }
    #[test]
    fn can_line_to_north_east() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: -2, y: 2 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: -1, y: 1 }),
                (IVec2 { x: -2, y: 2 })
            ]
        );
    }
    #[test]
    fn can_line_to_south_west() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 2, y: -2 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: 1, y: -1 }),
                (IVec2 { x: 2, y: -2 })
            ]
        );
    }
    #[test]
    fn can_line_to_south_west_2() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 1, y: -1 }).collect();
        assert_eq!(
            points,
            vec![(IVec2 { x: 0, y: 0 }), (IVec2 { x: 1, y: -1 }),]
        );
    }
    #[test]
    fn can_line_to_south_east() {
        let points: Vec<IVec2> = line_to(IVec2 { x: 0, y: 0 }, IVec2 { x: 2, y: 2 }).collect();
        assert_eq!(
            points,
            vec![
                (IVec2 { x: 0, y: 0 }),
                (IVec2 { x: 1, y: 1 }),
                (IVec2 { x: 2, y: 2 })
            ]
        );
    }
    #[test]
    fn special_case_propag() {
        let point: Option<IVec2> =
            line_to(IVec2 { x: 499, y: 99 }, IVec2 { x: 500, y: 100 }).nth(1);
        assert_eq!(point, Some(IVec2 { x: 500, y: 100 }));
    }
    type Result<A> = std::result::Result<A, std::boxed::Box<dyn std::error::Error>>;

    #[test]
    fn south_up_bearing() -> Result<()> {
        let geo_ref = GeoReference::south_up(
            (
                Vec2 {
                    x: -180.0,
                    y: -90.0,
                },
                Vec2 { x: 180.0, y: 90.0 },
            ),
            Vec2 { x: 0.1, y: 0.1 },
            4326,
        )
        .ok_or("should not happen")?;
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 0.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 1.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: -1.0 }),
            PI
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -1.0, y: 0.0 }),
            -PI / 2.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: 0.0 }),
            PI / 2.0
        );

        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: 1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: -1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: 1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: -1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -1.0, y: 1.0 }),
            -PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -1.0, y: -1.0 }),
            -(PI - PI / 4.0)
        );

        Ok(())
    }

    #[test]
    fn north_up_bearing() -> Result<()> {
        let geo_ref = GeoReference::north_up(
            (
                Vec2 {
                    x: -180.0,
                    y: -90.0,
                },
                Vec2 { x: 180.0, y: 90.0 },
            ),
            Vec2 { x: 0.1, y: 0.1 },
            4326,
        )
        .ok_or("should not happen")?;
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 0.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: -1.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 1.0 }),
            PI
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -1.0, y: 0.0 }),
            -PI / 2.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: 0.0 }),
            PI / 2.0
        );

        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: -1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: 1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: -1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: 1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -1.0, y: -1.0 }),
            -PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -1.0, y: 1.0 }),
            -(PI - PI / 4.0)
        );
        Ok(())
    }
}
