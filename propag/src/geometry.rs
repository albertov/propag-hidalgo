pub use gdal::spatial_ref::SpatialRef;

#[derive(PartialEq)]
pub struct GeoTransform<T>([T; 6]);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct V2<T>(T, T);

impl V2<f32> {
    pub fn line_to(self, other: Self) -> impl Iterator<Item = Self> {
        DDA::new(self, other)
    }
}

impl<T: uom::num_traits::Float> V2<T> {
    pub fn recip(self) -> Self {
        V2(self.0.recip(), self.1.recip())
    }
    pub fn abs(self) -> Self {
        V2(self.0.abs(), self.1.abs())
    }
}

impl<T> std::ops::Sub for V2<T>
where
    T: std::ops::Sub<Output = T> + Copy + Clone,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        V2(self.0 - other.0, self.1 - other.1)
    }
}

impl<T> std::ops::Add for V2<T>
where
    T: std::ops::Add<Output = T> + Copy + Clone,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        V2(self.0 + other.0, self.1 + other.1)
    }
}

impl<T> std::ops::Mul for V2<T>
where
    T: std::ops::Mul<Output = T> + Copy + Clone,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        V2(self.0 * other.0, self.1 * other.1)
    }
}

pub struct GeoReference<T> {
    transform: GeoTransform<T>,
    size: V2<i32>,
    crs: SpatialRef,
}

impl<T> GeoReference<T>
where
    T: std::ops::Add<Output = T>
        + Clone
        + Copy
        + std::ops::Mul<Output = T>
        + std::convert::From<i32>,
{
    pub fn extent(&self) -> Extent<T> {
        let Self {
            transform, size, ..
        } = self;
        let size = V2(size.0.into(), size.1.into());
        let GeoTransform([x0, dx, _rx, y0, dy, _ry]) = *transform;
        //assert!(dx > 0.0 && dy > 0.0 && rx == 0.0 && ry == 0.0);
        let origin = V2(x0, y0);
        Extent(origin, origin + size * V2(dx, dy))
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Extent<T>(V2<T>, V2<T>);

impl<T> Extent<T>
where
    T: std::ops::Sub<Output = T> + Copy + Clone + std::ops::Add<Output = T>,
{
    pub fn size(&self) -> V2<T> {
        //V2(self.1.0 - self.0.0, self.1.1 - self.0.1)
        self.1 - self.0
    }
    pub fn buffer(&self, size: V2<T>) -> Self {
        Self(self.0 - size, self.1 + size)
    }
}

impl<T> std::ops::Add for Extent<T>
where
    T: std::ops::Add<Output = T> + Copy + Clone + std::cmp::Ord,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Extent(
            V2(self.0 .0.min(other.0 .0), self.0 .1.min(other.0 .1)),
            V2(self.1 .0.max(other.1 .0), self.1 .1.max(other.1 .1)),
        )
    }
}

pub struct Raster<'a, T> {
    geo_ref: GeoReference<f64>,
    data: &'a [T],
}

struct DDA<T> {
    origin: V2<T>,
    dest: V2<T>,
    step: V2<T>,
    cur: Option<V2<T>>,
    t_max: V2<T>,
    delta: V2<T>,
}

impl DDA<f32> {
    fn new(origin: V2<f32>, dest: V2<f32>) -> Self {
        let t_max = dest - origin;
        let t_max = t_max.recip().abs();
        let step = V2((dest.0 - origin.0).signum(), (dest.1 - origin.1).signum());
        let cur = if step.0 == 0.0 && step.1 == 0.0 {
            None
        } else {
            Some(origin)
        };
        DDA {
            origin,
            dest,
            cur,
            t_max,
            delta: t_max,
            step,
        }
    }
}

impl Iterator for DDA<f32> {
    type Item = V2<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let valid_x = |x: f32| {
            if self.step.0 > 0.0 {
                x <= self.dest.0
            } else {
                x >= self.dest.0
            }
        };
        let valid_y = |y: f32| {
            if self.step.1 > 0.0 {
                y <= self.dest.1
            } else {
                y >= self.dest.1
            }
        };
        match self.cur {
            None => None,
            Some(V2(x, y)) if !(valid_x(x) && valid_y(y)) => {
                self.cur = None;
                None
            }
            Some(V2(x, y)) if (self.t_max.0 - self.t_max.1).abs() < 1e-6 => {
                let ret = self.cur;
                self.cur = Some(V2(x + self.step.0, y + self.step.1));
                self.t_max = self.t_max + self.delta;
                ret
            }
            Some(V2(x, y)) if self.t_max.0 < self.t_max.1 => {
                let ret = self.cur;
                self.cur = Some(V2(self.step.0 + x, y));
                self.t_max = V2(self.t_max.0 + self.delta.0, self.t_max.1);
                ret
            }
            Some(V2(x, y)) => {
                let ret = self.cur;
                self.cur = Some(V2(x, self.step.1 + y));
                self.t_max = V2(self.t_max.0, self.t_max.1 + self.delta.1);
                ret
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn can_add_extents() {
        assert_eq!(
            Extent(V2(0, 0), V2(1, 1)) + Extent(V2(-1, -1), V2(0, 1)),
            Extent(V2(-1, -1), V2(1, 1))
        )
    }
    #[test]
    fn can_line_to_north() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(0.0, 2.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(0.0, 1.0), V2(0.0, 2.0)]);
    }
    #[test]
    fn can_line_to_south() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(0.0, -2.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(0.0, -1.0), V2(0.0, -2.0)]);
    }
    #[test]
    fn can_line_to_east() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(2.0, 0.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(1.0, 0.0), V2(2.0, 0.0)]);
    }
    #[test]
    fn can_line_to_west() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(-2.0, 0.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(-1.0, 0.0), V2(-2.0, 0.0)]);
    }
    #[test]
    fn can_line_to_north_west() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(-2.0, -2.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(-1.0, -1.0), V2(-2.0, -2.0)]);
    }
    #[test]
    fn can_line_to_north_east() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(-2.0, 2.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(-1.0, 1.0), V2(-2.0, 2.0)]);
    }
    #[test]
    fn can_line_to_south_west() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(2.0, -2.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(1.0, -1.0), V2(2.0, -2.0)]);
    }
    #[test]
    fn can_line_to_south_east() {
        let points: Vec<V2<f32>> = V2(0.0, 0.0).line_to(V2(2.0, 2.0)).collect();
        assert_eq!(points, vec![V2(0.0, 0.0), V2(1.0, 1.0), V2(2.0, 2.0)]);
    }
}
