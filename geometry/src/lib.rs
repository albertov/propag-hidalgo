#![no_std]

#[cfg(test)]
extern crate std;

use approx::abs_diff_ne;

use approx::AbsDiffEq;

pub use geo_types::*;
use num_traits::{Float, NumCast};

#[cfg(target_os = "cuda")]
extern crate cuda_std;

#[cfg(not(target_os = "cuda"))]
extern crate cust;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
pub enum Crs {
    Epsg(u32),
    Wkt([u8; 1024]),
    Proj4([u8; 1024]),
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
pub struct GeoTransform<T>([T; 6], [T; 6]);

impl<T> GeoTransform<T>
where
    T: CoordFloat + AbsDiffEq,
{
    pub fn new(xs: [T; 6]) -> Option<Self> {
        Some(Self(xs, Self::invert(xs)?))
    }
    pub fn is_north_up(&self) -> bool {
        self.dy() < T::zero()
    }
    pub fn origin(&self) -> Coord<T> {
        Coord {
            x: self.0[0],
            y: self.0[3],
        }
    }
    pub fn dy(&self) -> T {
        self.0[5]
    }
    pub fn dx(&self) -> T {
        self.0[0]
    }
    pub fn as_mut_ref<'a>(&'a mut self) -> &'a mut [T; 6] {
        &mut self.0
    }
    pub fn as_ref<'a>(&'a self) -> &'a [T; 6] {
        &self.0
    }
    pub fn forward(&self, p: Coord<T>) -> Coord<usize> {
        let Coord { x, y } = p - self.origin();
        let [_, dx, rx, _, ry, dy] = self.1;
        let p = Coord {
            x: x * dx + y * rx,
            y: x * ry + y * dy,
        };
        Coord {
            x: NumCast::from(p.x.trunc()).expect("T to usize failed"),
            y: NumCast::from(p.y.trunc()).expect("T to usize failed"),
        }
    }
    pub fn backward(&self, p: Coord<usize>) -> Coord<T> {
        let p = Coord {
            x: NumCast::from(p.x).expect("T from usize failed"),
            y: NumCast::from(p.y).expect("T to from failed"),
        };
        let Coord { x, y } = p + self.origin();
        let [_, dx, rx, _, ry, dy] = self.0;
        Coord {
            x: x * dx + y * rx,
            y: x * ry + y * dy,
        }
    }
    fn invert(gt: [T; 6]) -> Option<[T; 6]> {
        let [x0, a, c, y0, b, d] = gt;
        let det = a * d - b * c;
        if abs_diff_ne!(det.abs(), T::zero()) {
            let f = T::one() / det;
            let a2 = d * f;
            let b2 = -b * f;
            let c2 = -c * f;
            let d2 = a * f;
            Some([x0, a2, c2, y0, b2, d2])
        } else {
            None
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
pub struct GeoReference<T>
where
    T: CoordFloat,
{
    pub transform: GeoTransform<T>,
    pub size: [u32; 2],
    pub crs: Crs,
}

impl<T> GeoReference<T>
where
    T: CoordFloat + Clone + From<u32> + AbsDiffEq,
{
    pub fn len(&self) -> u32 {
        self.size[0] * self.size[1]
    }

    pub fn forward(&self, p: Coord<T>) -> Coord<usize> {
        self.transform.forward(p)
    }
    pub fn backward(&self, p: Coord<usize>) -> Coord<T> {
        self.transform.backward(p)
    }
    pub fn south_up(extent: Rect<T>, pixel_size: Coord<T>, crs: Crs) -> Option<GeoReference<T>> {
        let Coord { x: dx, y: dy } = pixel_size;
        let Coord { x: x0, y: y0 } = extent.min();
        let transform = GeoTransform::new([x0, dx, T::zero(), y0, T::zero(), dy])?;
        let size = [
            NumCast::from(extent.width() / dx)?,
            NumCast::from(extent.height() / dy)?,
        ];
        Some(GeoReference {
            transform,
            size,
            crs,
        })
    }
    pub fn north_up(extent: Rect<T>, pixel_size: Coord<T>, crs: Crs) -> Option<GeoReference<T>> {
        let Coord { x: dx, y: dy } = pixel_size;
        let Coord { x: x0, .. } = extent.min();
        let Coord { y: y1, .. } = extent.max();
        let size = [
            NumCast::from(extent.width() / dx)?,
            NumCast::from(extent.height() / dy)?,
        ];
        let transform = GeoTransform::new([x0, dx, T::zero(), y1, T::zero(), -dy])?;
        Some(GeoReference {
            transform,
            size,
            crs,
        })
    }
    pub fn extent(&self) -> Rect<T> {
        let Self {
            transform, size, ..
        } = self;
        let size = Coord {
            x: size[0].into(),
            y: size[1].into(),
        };
        //assert!(dx > T::zero() && dy > T::zero() && rx == T::zero() && ry == T::zero());
        let origin = transform.origin();
        Rect::new(
            origin,
            origin
                + Coord {
                    x: transform.dx() * size.x,
                    y: transform.dy() * size.y,
                },
        )
    }
    pub fn is_north_up(&self) -> bool {
        self.transform.is_north_up()
    }

    // Azimuth (radian)
    pub fn bearing(&self, a: Coord<T>, b: Coord<T>) -> T {
        if self.is_north_up() {
            Float::atan2(b.x - a.x, a.y - b.y)
        } else {
            Float::atan2(b.x - a.x, b.y - a.y)
        }
    }
    pub fn distance(&self, a: Coord<T>, b: Coord<T>) -> T {
        Float::sqrt(
            ((a.x - b.x) * self.transform.dx()).powi(2)
                + ((a.y - b.y) * self.transform.dy()).powi(2),
        )
    }
}

/*
pub struct Raster<'a, T> {
    geo_ref: GeoReference<f64>,
    data: &'a [T],
}
*/

struct DDA<T>
where
    T: CoordFloat,
{
    dest: Coord<T>,
    step: Coord<T>,
    cur: Option<Coord<T>>,
    t_max: Coord<T>,
    delta: Coord<T>,
}

fn abs<T>(coord: Coord<T>) -> Coord<T>
where
    T: CoordFloat,
{
    let Coord { x, y } = coord;
    Coord {
        x: x.abs(),
        y: y.abs(),
    }
}

fn recip<T>(coord: Coord<T>) -> Coord<T>
where
    T: CoordFloat,
{
    let Coord { x, y } = coord;
    Coord {
        x: T::one() / x,
        y: T::one() / y,
    }
}

fn signum<T>(coord: Coord<T>) -> Coord<T>
where
    T: CoordFloat,
{
    let Coord { x, y } = coord;
    Coord {
        x: x.signum(),
        y: y.signum(),
    }
}

impl<T> DDA<T>
where
    T: CoordFloat,
{
    fn new(origin: Coord<T>, dest: Coord<T>) -> Self {
        let t_max = recip(abs(dest - origin));
        let step = signum(dest - origin);
        let cur = if step.x == T::zero() && step.y == T::zero() {
            None
        } else {
            Some(origin)
        };
        DDA {
            dest,
            cur,
            t_max,
            delta: t_max,
            step,
        }
    }
}

impl<T> Iterator for DDA<T>
where
    T: CoordFloat + AbsDiffEq<Epsilon = T>,
{
    type Item = Coord<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let valid_x = |x: T| {
            if self.step.x > T::zero() {
                x <= self.dest.x
            } else {
                x >= self.dest.x
            }
        };
        let valid_y = |y: T| {
            if self.step.y > T::zero() {
                y <= self.dest.y
            } else {
                y >= self.dest.y
            }
        };
        match self.cur {
            None => None,
            Some(Coord { x, y }) if !(valid_x(x) && valid_y(y)) => {
                self.cur = None;
                None
            }
            Some(Coord { x, y }) if (self.t_max.x - self.t_max.y).abs() < T::default_epsilon() => {
                let ret = self.cur;
                self.cur = Some(Coord {
                    x: x + self.step.x,
                    y: y + self.step.y,
                });
                self.t_max = self.t_max + self.delta;
                ret
            }
            Some(Coord { x, y }) if self.t_max.x < self.t_max.y => {
                let ret = self.cur;
                self.cur = Some(Coord {
                    x: self.step.x + x,
                    y: y,
                });
                self.t_max = Coord {
                    x: self.t_max.x + self.delta.x,
                    y: self.t_max.y,
                };
                ret
            }
            Some(Coord { x, y }) => {
                let ret = self.cur;
                self.cur = Some(Coord {
                    x,
                    y: self.step.y + y,
                });
                self.t_max = Coord {
                    x: self.t_max.x,
                    y: self.t_max.y + self.delta.y,
                };
                ret
            }
        }
    }
}

pub fn line_to<T>(from: Coord<T>, to: Coord<T>) -> impl Iterator<Item = Coord<T>>
where
    T: CoordFloat + AbsDiffEq<Epsilon = T>,
{
    DDA::new(from, to)
}

#[cfg(test)]
mod test {
    use super::*;

    use std::f64::consts::PI;
    use std::vec::*;
    use std::*;

    #[test]
    fn can_line_to_self() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 }).collect();
        assert_eq!(points, vec![(Coord { x: 0.0, y: 0.0 })]);
    }
    #[test]
    fn can_line_to_north() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: 0.0, y: 1.0 }),
                (Coord { x: 0.0, y: 2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_south() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: -2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: 0.0, y: -1.0 }),
                (Coord { x: 0.0, y: -2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_east() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: 2.0, y: 0.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: 1.0, y: 0.0 }),
                (Coord { x: 2.0, y: 0.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_west() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: -2.0, y: 0.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: -1.0, y: 0.0 }),
                (Coord { x: -2.0, y: 0.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_north_west() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: -2.0, y: -2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: -1.0, y: -1.0 }),
                (Coord { x: -2.0, y: -2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_north_east() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: -2.0, y: 2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: -1.0, y: 1.0 }),
                (Coord { x: -2.0, y: 2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_south_west() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: 2.0, y: -2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: 1.0, y: -1.0 }),
                (Coord { x: 2.0, y: -2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_south_east() {
        let points: Vec<Coord<f64>> =
            line_to(Coord { x: 0.0, y: 0.0 }, Coord { x: 2.0, y: 2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Coord { x: 0.0, y: 0.0 }),
                (Coord { x: 1.0, y: 1.0 }),
                (Coord { x: 2.0, y: 2.0 })
            ]
        );
    }
    type Result<A> = std::result::Result<A, std::boxed::Box<dyn std::error::Error>>;

    #[test]
    fn south_up_bearing() -> Result<()> {
        let geo_ref = GeoReference::south_up(
            Rect::new(
                Coord {
                    x: -180.0,
                    y: -90.0,
                },
                Coord { x: 180.0, y: 90.0 },
            ),
            Coord { x: 0.1, y: 0.1 },
            Crs::Epsg(4326),
        )
        .ok_or("should not happen")?;
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 1.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: -1.0 }),
            PI
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: -1.0, y: 0.0 }),
            -PI / 2.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 0.0 }),
            PI / 2.0
        );

        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: -1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: -1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: -1.0, y: 1.0 }),
            -PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: -1.0, y: -1.0 }),
            -(PI - PI / 4.0)
        );

        Ok(())
    }

    #[test]
    fn north_up_bearing() -> Result<()> {
        let geo_ref = GeoReference::north_up(
            Rect::new(
                Coord {
                    x: -180.0,
                    y: -90.0,
                },
                Coord { x: 180.0, y: 90.0 },
            ),
            Coord { x: 0.1, y: 0.1 },
            Crs::Epsg(4326),
        )
        .ok_or("should not happen")?;
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: -1.0 }),
            0.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 1.0 }),
            PI
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: -1.0, y: 0.0 }),
            -PI / 2.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 0.0 }),
            PI / 2.0
        );

        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: -1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: -1.0 }),
            PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }),
            PI - PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: -1.0, y: -1.0 }),
            -PI / 4.0
        );
        assert_eq!(
            geo_ref.bearing(Coord { x: 0.0, y: 0.0 }, Coord { x: -1.0, y: 1.0 }),
            -(PI - PI / 4.0)
        );
        Ok(())
    }
}
