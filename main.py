import time

from matplotlib import pyplot as plt
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
import subprocess
import argparse
import datetime
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dem', required=True)
parser.add_argument('--vector', required=True)
parser.add_argument('--out', required=True)


class GisAnalyst:
    CACHED_PATH = 'cache'

    SLOPE = 'slope'
    ASPECT = 'aspect'
    BIN_SLOPE = 'bin-slope'
    BIN_ASPECT = 'bin-aspect'
    HILLSHADE = 'hillshade'
    INTERSECT = 'intersect'

    _cache = {}
    _cmap = {
        SLOPE: 'viridis',
        ASPECT: 'magma',
        BIN_SLOPE: 'viridis',
        BIN_ASPECT: 'viridis',
        HILLSHADE: 'gray',
        INTERSECT: 'viridis',
    }
    shape = (100, 100)

    def __init__(self, dem_path, vector_path, out_path):
        gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

        self.dem_path = dem_path
        self.vector_path = vector_path
        self.out_path = out_path

        self.dem_filename = os.path.basename(dem_path)
        self.vector_filename = os.path.basename(vector_path)

        if not os.path.exists(self.CACHED_PATH):
            os.makedirs(self.CACHED_PATH)
        self.define_paths()

    def define_paths(self):
        self._cache[self.SLOPE] = os.path.join(self.CACHED_PATH, f"{self.SLOPE.upper()}-{self.dem_path}")
        self._cache[self.ASPECT] = os.path.join(self.CACHED_PATH, f"{self.ASPECT.upper()}-{self.dem_path}")
        self._cache[self.BIN_SLOPE] = os.path.join(self.CACHED_PATH, f"{self.BIN_SLOPE.upper()}-{self.dem_path}")
        self._cache[self.BIN_ASPECT] = os.path.join(self.CACHED_PATH, f"{self.BIN_ASPECT.upper()}-{self.dem_path}")
        self._cache[self.HILLSHADE] = os.path.join(self.CACHED_PATH, f"{self.HILLSHADE.upper()}-{self.dem_path}")
        self._cache[self.INTERSECT] = os.path.join(self.CACHED_PATH, f"{self.ASPECT.upper()}-X-{self.SLOPE.upper()}-{self.dem_path}")

    def set_shape(self, dem):
        arr = dem.GetRasterBand(1).ReadAsArray()
        self.shape = arr.shape

    def process_dem(self):
        dem = gdal.Open(self.dem_path, gdal.GA_ReadOnly)
        self.set_shape(dem)

        self.process_slope(dem)
        self.process_aspect(dem)
        self.process_hillshade(dem)

        dataset = self.intersect_bitmask(
            self._cache[self.BIN_ASPECT],
            self._cache[self.BIN_SLOPE],
            self._cache[self.INTERSECT]
        )
        self.create_plot(self.INTERSECT, dataset)

        dem.FlushCache()
        dem = None

    def process_vector(self):

        vector_shape_path = os.path.join(self.CACHED_PATH, "vector_shape.shp")
        self.create_polygon_from_raster(self._cache[self.HILLSHADE], vector_shape_path)
        clipped_vector = os.path.join(self.CACHED_PATH, "clipped_vector.shp")
        self.clip_vectors(self.vector_path, vector_shape_path, clipped_vector)
        self.buffer_vector(clipped_vector, clipped_vector, 0.001)
        clipped_raster = os.path.join(self.CACHED_PATH, "clipped_raster.tif")
        self.vector_to_raster(clipped_vector, clipped_raster)
        self.overlay_trails_and_dangerous_areas(
            self._cache[self.HILLSHADE],
            clipped_raster,
            self._cache[self.INTERSECT]
        )

    def create_polygon_from_raster(self, raster_path, output_shp_path):
        raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)

        width = raster_ds.RasterXSize
        height = raster_ds.RasterYSize
        geotransform = raster_ds.GetGeoTransform()

        driver = ogr.GetDriverByName('ESRI Shapefile')
        output_ds = driver.CreateDataSource(output_shp_path)
        layer_srs = osr.SpatialReference()
        layer_srs.ImportFromWkt(raster_ds.GetProjection())
        layer = output_ds.CreateLayer('polygon', geom_type=ogr.wkbPolygon, srs=layer_srs)

        rect = ogr.Geometry(ogr.wkbLinearRing)
        rect.AddPoint(geotransform[0], geotransform[3])
        rect.AddPoint(geotransform[0] + width * geotransform[1], geotransform[3])
        rect.AddPoint(geotransform[0] + width * geotransform[1], geotransform[3] + height * geotransform[5])
        rect.AddPoint(geotransform[0], geotransform[3] + height * geotransform[5])
        rect.AddPoint(geotransform[0], geotransform[3])  # Close the rect

        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(rect)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(polygon)

        layer.CreateFeature(feature)

        output_ds.FlushCache()
        raster_ds.FlushCache()
        output_ds = None
        raster_ds = None

    def buffer_vector(self, input_vector, output_vector, buffer_distance):
        input_ds = ogr.Open(input_vector, 0)
        input_layer = input_ds.GetLayer()

        mem_driver = ogr.GetDriverByName('Memory')
        mem_ds = mem_driver.CreateDataSource('')
        mem_layer = mem_ds.CreateLayer('buffered', geom_type=ogr.wkbPolygon)

        for field in input_layer.schema:
            mem_layer.CreateField(field)

        for feature in input_layer:
            geometry = feature.GetGeometryRef()
            buffered_geometry = geometry.Buffer(buffer_distance)

            mem_feature = ogr.Feature(mem_layer.GetLayerDefn())
            mem_feature.SetGeometry(buffered_geometry)

            for i in range(feature.GetFieldCount()):
                value = feature.GetField(i)
                mem_feature.SetField(i, value)

            mem_layer.CreateFeature(mem_feature)

        driver = ogr.GetDriverByName('ESRI Shapefile')
        out_ds = driver.CreateDataSource(output_vector)
        out_layer = out_ds.CreateLayer('buffered', geom_type=ogr.wkbPolygon)

        for field in mem_layer.schema:
            out_layer.CreateField(field)

        for feature in mem_layer:
            out_layer.CreateFeature(feature)

        input_ds = None
        mem_ds = None
        out_ds = None

    def overlay_trails_and_dangerous_areas(self, hillshade_path, t_bitmap_path, da_bitmap_path):
        hillshade_ds = gdal.Open(hillshade_path, gdal.GA_ReadOnly)
        hillshade_array = hillshade_ds.GetRasterBand(1).ReadAsArray()

        t_bitmap_ds = gdal.Open(t_bitmap_path, gdal.GA_ReadOnly)
        t_bitmap_array = t_bitmap_ds.GetRasterBand(1).ReadAsArray()

        da_bitmap_ds = gdal.Open(da_bitmap_path, gdal.GA_ReadOnly)
        da_bitmap_array = da_bitmap_ds.GetRasterBand(1).ReadAsArray()

        trails_overlay = np.zeros_like(hillshade_array, dtype=np.uint8)
        trails_overlay[t_bitmap_array > 0] = 200  # Set trails to yellow
        trails_alpha = 0.8

        dangerous_areas_overlay = np.zeros_like(hillshade_array, dtype=np.uint8)
        dangerous_areas_overlay[da_bitmap_array > 0] = 200  # Set dangerous areas to red
        dangerous_areas_alpha = 0.6

        composite_image = np.stack([hillshade_array, hillshade_array, hillshade_array], axis=-1)
        composite_image = composite_image.astype(np.uint8)
        composite_image[trails_overlay > 0, 0] = (1 - trails_alpha) * composite_image[
            trails_overlay > 0, 0] + trails_alpha * 255
        composite_image[trails_overlay > 0, 1] = (1 - trails_alpha) * composite_image[
            trails_overlay > 0, 1] + trails_alpha * 255
        composite_image[trails_overlay > 0, 2] = 0  # Set blue channel to 0 for yellow color

        composite_image[dangerous_areas_overlay > 0, 0] = (1 - dangerous_areas_alpha) * composite_image[
            dangerous_areas_overlay > 0, 0] + dangerous_areas_alpha * 255
        composite_image[dangerous_areas_overlay > 0, 1] = 0  # Set green channel to 0 for red color
        composite_image[dangerous_areas_overlay > 0, 2] = 0  # Set blue channel to 0 for red color

        plt.imsave(self.out_path, composite_image)

        hillshade_ds = None
        t_bitmap_ds = None
        da_bitmap_ds = None

    def clip_vectors(self, input_vector1, input_vector2, output_vector):
        subprocess.run([
            'ogr2ogr',
            '-f', 'ESRI Shapefile',
            output_vector,
            input_vector1,
            '-clipsrc', input_vector2  # Specify the clip extent using the second input vector
        ])

    def vector_to_raster(self, vector_path, out_path):
        vector_dataset = ogr.Open(vector_path, 0)

        vector_layer = vector_dataset.GetLayer()

        spatial_reference = vector_layer.GetSpatialRef()
        x_min, x_max, y_min, y_max = vector_layer.GetExtent()

        if spatial_reference is None:
            dataset = gdal.Open(self.dem_path, gdal.GA_ReadOnly)
            spatial_reference = dataset.GetSpatialRef()
            dataset.FlushCache()
            dataset = None

        line_width_pixels = 1
        pixel_size = line_width_pixels / self.shape[1]  # Adjust according to your desired pixel size

        driver = gdal.GetDriverByName('GTiff')
        output_raster = driver.Create(out_path, self.shape[1], self.shape[0], 1, gdal.GDT_Byte)
        output_raster.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
        output_raster.SetProjection(spatial_reference.ExportToWkt())

        gdal.RasterizeLayer(output_raster, [1], vector_layer, burn_values=[1])

        vector_dataset = None
        output_raster = None

    def process_hillshade(self, dem):
        dataset = self.dem_processing(
            dem, self.HILLSHADE,
            computeEdges=True,
        )
        dataset = self.create_plot(self.HILLSHADE, dataset)
        self.close_dataset(dataset)

    def process_slope(self, dem):
        dataset = self.dem_processing(
            dem, self.SLOPE,
            computeEdges=True,
            alg='Horn',
            slopeFormat='percent',
        )
        dataset = self.fix_dataset(dataset, self.SLOPE)
        dataset = self.create_plot(self.SLOPE, dataset)
        dataset = self.binmask_processing(dataset, self.SLOPE)
        dataset = self.create_plot(self.BIN_SLOPE, dataset)
        self.close_dataset(dataset)

    def process_aspect(self, dem):
        dataset = self.dem_processing(
            dem, self.ASPECT,
            computeEdges=True,
        )
        dataset = self.fix_dataset(dataset, self.ASPECT)
        dataset = self.create_plot(self.ASPECT, dataset)
        dataset = self.binmask_processing(dataset, self.ASPECT)
        dataset = self.create_plot(self.BIN_ASPECT, dataset)
        self.close_dataset(dataset)

    def intersect_bitmask(self, in_file1, in_file2, out_file):
        ds1 = gdal.Open(in_file1, gdal.GA_ReadOnly)
        ds2 = gdal.Open(in_file2, gdal.GA_ReadOnly)

        band1 = ds1.GetRasterBand(1).ReadAsArray()
        band2 = ds2.GetRasterBand(1).ReadAsArray()

        result_mask = np.logical_and(band1, band2).astype(np.int16)

        driver = gdal.GetDriverByName('GTiff')
        outds = driver.Create(
            out_file,
            xsize=result_mask.shape[1],
            ysize=result_mask.shape[0],
            bands=1,
            eType=gdal.GDT_Int16
        )

        outds.SetGeoTransform(ds1.GetGeoTransform())
        outds.SetProjection(ds1.GetProjection())

        outband = outds.GetRasterBand(1)
        outband.WriteArray(result_mask)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()

        ds1 = None
        ds2 = None
        return outds

    def close_dataset(self, dataset):
        dataset.FlushCache()
        dataset = None

    def create_plot(self, ptype, dataset):
        plt.figure()
        plt.title(f"{ptype[0].upper()}{ptype[1:]} of: {self.dem_path}")
        arr = dataset.GetRasterBand(1).ReadAsArray()
        plt.imshow(arr, cmap=self._cmap.get(ptype, None))
        plt.colorbar()
        plt.savefig(os.path.join(self.CACHED_PATH, f"{ptype}.png"))

        return dataset

    def dem_processing(self, dem, processing, **kwargs):
        options = gdal.DEMProcessingOptions(**kwargs)
        return gdal.DEMProcessing(
            self._cache[processing],
            dem,
            processing,
            options=options
        )

    def binmask_processing(self, dataset, dtype):
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        binmask = None

        if dtype == self.SLOPE:
            binmask = np.where((array > 5), 1, 0)
        if dtype == self.ASPECT:
            binmask = np.where(np.logical_and(150 <= array, array <= 210), 1, 0)

        if binmask is None:
            raise Exception("Binary mask not found")

        driver = gdal.GetDriverByName('GTiff')
        driver.Register()

        outds = driver.Create(
            self._cache[f"bin-{dtype}"],
            xsize=binmask.shape[1],
            ysize=binmask.shape[0],
            bands=1,
            eType=gdal.GDT_Int16
        )
        outds.SetGeoTransform(dataset.GetGeoTransform())
        outds.SetProjection(dataset.GetProjection())

        outband = outds.GetRasterBand(1)
        outband.WriteArray(binmask)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()

        self.close_dataset(dataset)
        return outds

    def fix_dataset(self, in_dataset, dtype):
        band = in_dataset.GetRasterBand(1)

        arr = band.ReadAsArray()
        if dtype == self.ASPECT:
            arr[arr == -9999] = -1
        if dtype == self.SLOPE:
            arr[arr >= 0] /= (10**6)

        out_path = os.path.join(self.CACHED_PATH, f"{dtype.upper()}-FIX-{self.dem_path}")
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(out_path, in_dataset.RasterXSize, in_dataset.RasterYSize, 1, band.DataType)
        out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
        out_dataset.SetProjection(in_dataset.GetProjection())

        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(arr)

        self.close_dataset(in_dataset)

        os.replace(out_path, self._cache[dtype])

        return out_dataset


def main():
    args = parser.parse_args()
    dem = args.dem
    vector = args.vector
    out = args.out
    if not os.path.exists(dem):
        raise Exception('DEM File does not exist.')
    if not os.path.exists(vector):
        raise Exception('Vector File does not exist.')
    if os.path.exists(out):
        Warning('Output file is not empty.')
        res = input('Do you wont to overwrite it? (y/n) ')
        if res.lower() != 'y':
            return
    t = datetime.datetime.now()
    ga = GisAnalyst(dem, vector, out)
    ga.process_dem()
    ga.process_vector()

    print(datetime.datetime.now() - t)


if __name__ == '__main__':
    main()
