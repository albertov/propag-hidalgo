#![no_std]

#[cfg(test)]
extern crate std;

use approx::AbsDiffEq;

pub use geo::*;
use heapless::String;
use num_traits::NumCast;

#[derive(Clone)]
pub enum Crs {
    Epsg(u32),
    Wkt(String<1024>),
    Proj4(String<1024>),
}

pub struct GeoReference<T>
where
    T: CoordFloat,
{
    pub transform: AffineTransform<T>,
    inv_transform: AffineTransform<T>,
    pub size: Coord<i32>,
    pub crs: Crs,
}

impl<T> GeoReference<T>
where
    T: CoordFloat + Clone + From<i32>,
{
    pub fn forward(&self, p: Coord<T>) -> Coord<usize> {
        let p = self.inv_transform.apply(p);
        Coord {
            x: NumCast::from(p.x.trunc()).expect("T to usize failed"),
            y: NumCast::from(p.y.trunc()).expect("T to usize failed"),
        }
    }
    pub fn backward(&self, p: Coord<usize>) -> Coord<T> {
        self.transform.apply(Coord {
            x: NumCast::from(p.x).expect("usize to T failed"),
            y: NumCast::from(p.y).expect("usize to T failed"),
        })
    }
    pub fn new_south_up(
        extent: Rect<T>,
        pixel_size: Coord<T>,
        crs: Crs,
    ) -> Option<GeoReference<T>> {
        let Coord { x: dx, y: dy } = pixel_size;
        let Coord { x: x0, y: y0 } = extent.min();
        let transform = AffineTransform::new(dx, T::zero(), x0, T::zero(), dy, y0);
        let inv_transform = transform.inverse()?;
        let size = Coord {
            x: NumCast::from(extent.width() / dx)?,
            y: NumCast::from(extent.height() / dy)?,
        };
        Some(GeoReference {
            transform,
            inv_transform,
            size,
            crs,
        })
    }
    pub fn extent(&self) -> Rect<T> {
        let Self {
            transform, size, ..
        } = self;
        let size = Coord {
            x: size.x.into(),
            y: size.y.into(),
        };
        //assert!(dx > T::zero() && dy > T::zero() && rx == T::zero() && ry == T::zero());
        let origin = Coord {
            x: transform.xoff(),
            y: transform.yoff(),
        };
        Rect::new(
            origin,
            origin
                + Coord {
                    x: transform.a() * size.x,
                    y: transform.e() * size.y,
                },
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
}
