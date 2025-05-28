use gdal::errors::Result;
use gdal::Dataset;
use gdal_sys::*;
use std::marker::PhantomData;

use crate::geometry::GeoReference;
use gdal_sys::GDALDataType::GDT_Float64;

pub struct WarpedDataset<'a>(pub Dataset, PhantomData<&'a ()>);

impl<'a> WarpedDataset<'a> {
    pub fn new(src: &'a Dataset, geo_ref: GeoReference<f64>) -> Result<Self> {
        unsafe {
            let opts = GDALCreateWarpOptions();
            let trans = GDALCreateGenImgProjTransformer3(
                src.projection().as_ptr() as *const i8,
                src.geo_transform()?.as_ptr(),
                geo_ref.crs.to_wkt()?.as_str().as_ptr() as *const i8,
                geo_ref.transform.as_ref().as_ptr(),
            );
            let n_bands = src.raster_count();
            (*opts).pfnTransformer = Some(GDALGenImgProjTransform);
            (*opts).pTransformerArg = trans;
            (*opts).hSrcDS = src.c_dataset();
            (*opts).nBandCount = n_bands as _;
            let src_bands = CPLMalloc(n_bands) as *mut i32;
            let dst_bands = CPLMalloc(n_bands) as *mut i32;
            for i in 0..n_bands {
                *src_bands.add(i) = i as i32 + 1;
                *dst_bands.add(i) = i as i32 + 1;
            }
            (*opts).panSrcBands = src_bands;
            (*opts).panDstBands = dst_bands;
            (*opts).eWorkingDataType = GDT_Float64;
            let mut gt = geo_ref.transform.clone();
            let dst_ds = GDALCreateWarpedVRT(
                src.c_dataset(),
                geo_ref.size.0 as _,
                geo_ref.size.1 as _,
                gt.as_mut_ref().as_mut_ptr(),
                opts,
            );
            GDALDestroyWarpOptions(opts);
            Ok(Self(Dataset::from_c_dataset(dst_ds), PhantomData))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GeoReference, WarpedDataset};
    use crate::geometry::{Extent, V2};
    use gdal::raster::GdalDataType;
    use gdal::spatial_ref::SpatialRef;
    use gdal::DriverManager;

    type Result<A> = std::result::Result<A, Box<dyn std::error::Error>>;

    #[test]
    fn can_create_warped_ds() -> Result<()> {
        let geo_ref = GeoReference::new_south_up(
            Extent(V2(-180.0, -90.0), V2(180.0, 90.0)),
            V2(0.1, 0.1),
            SpatialRef::from_epsg(4326)?,
        )
        .unwrap();
        let d = DriverManager::get_driver_by_name("MEM")?;
        let mut ds = d.create(
            "in-memory",
            geo_ref.size.0.try_into()?,
            geo_ref.size.1.try_into()?,
            3,
        )?;
        ds.set_spatial_ref(&geo_ref.crs)?;
        ds.set_geo_transform(&geo_ref.transform.as_ref())?;
        assert_eq!(ds.raster_count(), 3);
        assert_eq!(ds.raster_size(), (3600, 1800));
        assert_eq!(ds.rasterband(1)?.band_type(), GdalDataType::UInt8);
        let ext = Extent(V2(0.0, 0.0), V2(1000.0, 1000.0));

        let px_size = V2(5.0, 5.0);
        let crs = SpatialRef::from_epsg(25830)?;
        let geo_ref = GeoReference::new_south_up(ext, px_size, crs).unwrap();
        let ds2 = WarpedDataset::new(&ds, geo_ref)?.0;
        assert_eq!(ds2.raster_count(), 3);
        assert_eq!(ds2.raster_size(), (200, 200));
        assert_eq!(ds2.rasterband(1)?.band_type(), GdalDataType::Float64);
        Ok(())
    }
}
