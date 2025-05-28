#![no_std]

#[cfg(test)]
extern crate std;

use approx::abs_diff_ne;

use approx::AbsDiffEq;

pub use vek::*;
use num_traits::{Float, NumCast};

#[cfg(target_os = "cuda")]
extern crate cuda_std;

#[cfg(not(target_os = "cuda"))]
extern crate cust;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum Crs {
    Epsg(u32),
    Wkt([u8; 8]), //FIXME
    Proj4([u8; 8]), //FIXME
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct GeoTransform<T>([T; 6], [T; 6]);

impl<T> GeoTransform<T>
where
    T: AbsDiffEq + Copy + Float + NumCast
{
    pub fn new(xs: [T; 6]) -> Option<Self> {
        Some(Self(xs, Self::invert(xs)?))
    }
    pub fn is_north_up(&self) -> bool {
        self.dy() < T::zero()
    }
    pub fn origin(&self) -> Vec2<T> {
        Vec2 {
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
    pub fn forward(&self, p: Vec2<T>) -> Vec2<usize> {
        let Vec2 { x, y } = p - self.origin();
        let [_, dx, rx, _, ry, dy] = self.1;
        let p = Vec2 {
            x: x * dx + y * rx,
            y: x * ry + y * dy,
        };
        Vec2 {
            x: NumCast::from(p.x.trunc()).expect("T to usize failed"),
            y: NumCast::from(p.y.trunc()).expect("T to usize failed"),
        }
    }
    pub fn backward(&self, p: Vec2<usize>) -> Vec2<T> {
        let p = Vec2 {
            x: <T as NumCast>::from(p.x).expect("T from usize failed"),
            y: <T as NumCast>::from(p.y).expect("T to from failed"),
        };
        let Vec2 { x, y } = p + self.origin();
        let [_, dx, rx, _, ry, dy] = self.0;
        Vec2 {
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
#[repr(C)]
pub struct GeoReference<T>
{
    pub transform: GeoTransform<T>,
    pub size: Extent2<u32>,
    pub crs: Crs,
}

impl<T> GeoReference<T>
where
    T: Clone + From<u32> + AbsDiffEq + Float
{
    pub fn len(&self) -> u32 {
        self.size.w * self.size.h
    }

    pub fn forward(&self, p: Vec2<T>) -> Vec2<usize> {
        self.transform.forward(p)
    }
    pub fn backward(&self, p: Vec2<usize>) -> Vec2<T> {
        self.transform.backward(p)
    }
    pub fn south_up(bbox: Rect<T,T>, pixel_size: Extent2<T>, crs: Crs) -> Option<GeoReference<T>> {
        let Extent2 { w: dx, h: dy } = pixel_size;
        let Rect { x: x0, y: y0, w, h } = bbox;
        let transform = GeoTransform::new([x0, dx, T::zero(), y0, T::zero(), dy])?;
        let size = Extent2 {
            w: NumCast::from(w / dx)?,
            h: NumCast::from(h / dy)?,
        };
        Some(GeoReference {
            transform,
            size,
            crs,
        })
    }
    pub fn north_up(bbox: Rect<T,T>, pixel_size: Extent2<T>, crs: Crs) -> Option<GeoReference<T>> {
        let Extent2 { w: dx, h: dy } = pixel_size;
        let Rect { x: x0, y: y0, w, h } = bbox;
        let y1 = y0 + h;
        let transform = GeoTransform::new([x0, dx, T::zero(), y1, T::zero(), -dy])?;
        let size = Extent2 {
            w: NumCast::from(w / dx)?,
            h: NumCast::from(h / dy)?,
        };
        Some(GeoReference {
            transform,
            size,
            crs,
        })
    }
    pub fn bbox(&self) -> Rect<T,T> {
        let Self {
            transform, size, ..
        } = self;
        let size = Vec2 {
            x: size[0].into(),
            y: size[1].into(),
        };
        //assert!(dx > T::zero() && dy > T::zero() && rx == T::zero() && ry == T::zero());
        let origin = transform.origin();
        Rect::new(
            origin.x,
            origin.y,
            transform.dx() * size.x,
            transform.dy() * size.y,
        )
    }
    pub fn is_north_up(&self) -> bool {
        self.transform.is_north_up()
    }

    // Azimuth (radian)
    pub fn bearing(&self, a: Vec2<T>, b: Vec2<T>) -> T {
        if self.is_north_up() {
            Float::atan2(b.x - a.x, a.y - b.y)
        } else {
            Float::atan2(b.x - a.x, b.y - a.y)
        }
    }
    pub fn distance(&self, a: Vec2<T>, b: Vec2<T>) -> T {
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
{
    dest: Vec2<T>,
    step: Vec2<T>,
    cur: Option<Vec2<T>>,
    t_max: Vec2<T>,
    delta: Vec2<T>,
}

fn abs<T>(coord: Vec2<T>) -> Vec2<T>
where
    T: Float
{
    let Vec2 { x, y } = coord;
    Vec2 {
        x: x.abs(),
        y: y.abs(),
    }
}

fn recip<T>(coord: Vec2<T>) -> Vec2<T>
where
    T: Float
{
    let Vec2 { x, y } = coord;
    Vec2 {
        x: T::one() / x,
        y: T::one() / y,
    }
}

fn signum<T>(coord: Vec2<T>) -> Vec2<T>
where
    T: Float,
{
    let Vec2 { x, y } = coord;
    Vec2 {
        x: x.signum(),
        y: y.signum(),
    }
}

impl<T> DDA<T>
where
    T: Float,
{
    fn new(origin: Vec2<T>, dest: Vec2<T>) -> Self {
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
    T: Float + AbsDiffEq<Epsilon = T>,
{
    type Item = Vec2<T>;

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
            Some(Vec2 { x, y }) if !(valid_x(x) && valid_y(y)) => {
                self.cur = None;
                None
            }
            Some(Vec2 { x, y }) if (self.t_max.x - self.t_max.y).abs() < T::default_epsilon() => {
                let ret = self.cur;
                self.cur = Some(Vec2 {
                    x: x + self.step.x,
                    y: y + self.step.y,
                });
                self.t_max = self.t_max + self.delta;
                ret
            }
            Some(Vec2 { x, y }) if self.t_max.x < self.t_max.y => {
                let ret = self.cur;
                self.cur = Some(Vec2 {
                    x: self.step.x + x,
                    y: y,
                });
                self.t_max = Vec2 {
                    x: self.t_max.x + self.delta.x,
                    y: self.t_max.y,
                };
                ret
            }
            Some(Vec2 { x, y }) => {
                let ret = self.cur;
                self.cur = Some(Vec2 {
                    x,
                    y: self.step.y + y,
                });
                self.t_max = Vec2 {
                    x: self.t_max.x,
                    y: self.t_max.y + self.delta.y,
                };
                ret
            }
        }
    }
}

pub fn line_to<T>(from: Vec2<T>, to: Vec2<T>) -> impl Iterator<Item = Vec2<T>>
where
    T: Float + AbsDiffEq<Epsilon = T>,
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
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 0.0 }).collect();
        assert_eq!(points, vec![(Vec2 { x: 0.0, y: 0.0 })]);
    }
    #[test]
    fn can_line_to_north() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: 0.0, y: 1.0 }),
                (Vec2 { x: 0.0, y: 2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_south() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: -2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: 0.0, y: -1.0 }),
                (Vec2 { x: 0.0, y: -2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_east() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 2.0, y: 0.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: 1.0, y: 0.0 }),
                (Vec2 { x: 2.0, y: 0.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_west() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -2.0, y: 0.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: -1.0, y: 0.0 }),
                (Vec2 { x: -2.0, y: 0.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_north_west() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -2.0, y: -2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: -1.0, y: -1.0 }),
                (Vec2 { x: -2.0, y: -2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_north_east() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: -2.0, y: 2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: -1.0, y: 1.0 }),
                (Vec2 { x: -2.0, y: 2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_south_west() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 2.0, y: -2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: 1.0, y: -1.0 }),
                (Vec2 { x: 2.0, y: -2.0 })
            ]
        );
    }
    #[test]
    fn can_line_to_south_east() {
        let points: Vec<Vec2<f64>> =
            line_to(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 2.0, y: 2.0 }).collect();
        assert_eq!(
            points,
            vec![
                (Vec2 { x: 0.0, y: 0.0 }),
                (Vec2 { x: 1.0, y: 1.0 }),
                (Vec2 { x: 2.0, y: 2.0 })
            ]
        );
    }
    type Result<A> = std::result::Result<A, std::boxed::Box<dyn std::error::Error>>;

    #[test]
    fn south_up_bearing() -> Result<()> {
        let geo_ref = GeoReference::south_up(
            Rect::new(-180.0,-90.0,360.0, 180.0),
            Extent2 { w: 0.1, h: 0.1 },
            Crs::Epsg(4326),
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
            Rect::new(-180.0,-90.0,360.0, 180.0),
            Extent2 { w: 0.1, h: 0.1 },
            Crs::Epsg(4326),
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
