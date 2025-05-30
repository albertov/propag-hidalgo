use ::geometry::*;
use gdal::errors::Result;
use gdal::raster::GdalDataType;
use gdal::spatial_ref::SpatialRef;
use gdal::Dataset;
use gdal_sys::*;

// Takes ownership of the original dataset so no one uses it
// after it is wrapped
#[allow(dead_code)]
pub struct WarpedDataset(Dataset, /* orig ds */ Dataset);

impl WarpedDataset {
    pub fn get(&self) -> &Dataset {
        &self.0
    }

    pub fn new(src: Dataset, dt: GdalDataType, geo_ref: GeoReference) -> Result<Self> {
        let spatial_ref = to_spatial_ref(&geo_ref.proj)?;
        let dst_gt = geo_ref.transform.as_array_64();
        unsafe {
            let trans = GDALCreateGenImgProjTransformer3(
                std::ffi::CString::new(src.projection())?.as_ptr(),
                src.geo_transform()?.as_ptr(),
                std::ffi::CString::new(spatial_ref.to_wkt()?.as_str())?.as_ptr(),
                dst_gt.as_ptr(),
            );
            let n_bands = src.raster_count();
            let opts = GDALCreateWarpOptions();
            (*opts).pfnTransformer = Some(GDALGenImgProjTransform);
            (*opts).pTransformerArg = trans;
            (*opts).hSrcDS = src.c_dataset();
            (*opts).nBandCount = n_bands as _;
            let src_bands = CPLMalloc(n_bands * std::mem::size_of::<i32>()) as *mut i32;
            let dst_bands = CPLMalloc(n_bands * std::mem::size_of::<i32>()) as *mut i32;
            for i in 0..n_bands {
                *src_bands.add(i) = i as i32 + 1;
                *dst_bands.add(i) = i as i32 + 1;
            }
            (*opts).panSrcBands = src_bands;
            (*opts).panDstBands = dst_bands;
            (*opts).eWorkingDataType = dt as _;
            let mut gt = geo_ref.transform.as_array_64();
            let dst_ds = GDALCreateWarpedVRT(
                src.c_dataset(),
                geo_ref.width as i32,
                geo_ref.height as i32,
                gt.as_mut_ptr(),
                opts,
            );
            GDALDestroyWarpOptions(opts);
            Ok(Self(Dataset::from_c_dataset(dst_ds), src))
        }
    }
}

pub fn to_spatial_ref<const N: usize>(proj: &[u8; N]) -> Result<SpatialRef> {
    let ix = proj.iter().position(|x| *x == 0).unwrap_or(0);
    if let Ok(proj) = str::from_utf8(&proj[0..ix]) {
        SpatialRef::from_proj4(proj)
    } else {
        // hack
        SpatialRef::from_proj4("")
    }
}

#[cfg(test)]
mod tests {
    use super::{to_spatial_ref, GeoReference, WarpedDataset};
    use gdal::raster::{Buffer, GdalDataType, GdalType};
    use gdal::DriverManager;
    use geometry::Vec2;

    type Result<A> = std::result::Result<A, Box<dyn std::error::Error>>;

    #[test]
    fn can_create_warped_ds() -> Result<()> {
        {
            let geo_ref = GeoReference::south_up(
                (
                    Vec2 {
                        x: -180.0,
                        y: -90.0,
                    },
                    Vec2 { x: 180.0, y: 90.0 },
                ),
                Vec2 { x: 0.1, y: 0.1 },
                "+proj=longlat +datum=WGS84 +no_defs +type=crs",
            )
            .unwrap();
            let d = DriverManager::get_driver_by_name("MEM")?;
            let mut ds = d.create("in-memory", geo_ref.width as _, geo_ref.height as _, 3)?;
            ds.set_spatial_ref(&to_spatial_ref(&geo_ref.proj)?)?;
            ds.set_geo_transform(&geo_ref.transform.as_array_64())?;
            assert_eq!(ds.raster_count(), 3);
            assert_eq!(ds.raster_size(), (3600, 1800));
            assert_eq!(ds.rasterband(1)?.band_type(), GdalDataType::UInt8);
            let mut buf: Buffer<f64> = ds.rasterband(1)?.read_band_as()?;
            let px = geo_ref.forward(Vec2 { x: -3.0, y: 42.0 });
            // Buffer uses (row, col) indexing
            buf[(px.y, px.x)] = 128.0;
            ds.rasterband(1)?
                .write((0, 0), ds.raster_size(), &mut buf)?;
            ds.flush_cache()?;
            let ext = (Vec2 { x: 4.8e5, y: 4.6e6 }, Vec2 { x: 5.2e5, y: 4.7e6 });

            let px_size = Vec2 { x: 50.0, y: 500.0 };
            let epsg = "+proj=utm +zone=30 +ellps=GRS80 +units=m +no_defs";
            let geo_ref = GeoReference::south_up(ext, px_size, epsg).unwrap();
            let wrapped = WarpedDataset::new(ds, <f64>::datatype(), geo_ref)?;
            let ds2 = wrapped.get();
            assert_eq!(ds2.raster_count(), 3);
            assert_eq!(ds2.raster_size(), (800, 200));
            assert_eq!(ds2.rasterband(1)?.band_type(), GdalDataType::Float64);
            let s = ds2.raster_size();
            let arr: Buffer<f64> = ds2.rasterband(1)?.read_as((0, 0), s, s, None)?;
            assert!(arr.into_iter().filter(|x| *x != 0.0).count() > 0);
        };
        // Un-comment to make valgrind's output less noisy
        //DriverManager::destroy();
        Ok(())
    }
}
