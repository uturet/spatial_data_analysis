import os
import warnings
import subprocess
from osgeo import gdal
from matplotlib import pyplot as plt
import argparse
from osgeo_utils.gdal_calc import GDALCalc

parser = argparse.ArgumentParser()
parser.add_argument('--dem', required=True)
parser.add_argument('--vector', required=True)
parser.add_argument('--out', required=True)


class GisAnalyst:
    CACHED_PATH = 'cache'
    SLOPE = 'slope'
    ASPECT = 'aspect'
    HILLSHADE = 'hillshade'
    FTYPE_DEM = 'dem'
    FTYPE_VECTOR = 'vector'
    _cache = {}
    _cmap = {
        'hill': 'gray',
        'aspect': 'magma'
    }

    def __init__(self, dem_path, vector_path, out_path):
        self.dem_path = dem_path
        self.vector_path = vector_path
        self.out_path = out_path

        self.dem_filename = os.path.basename(dem_path)
        self.vector_filename = os.path.basename(vector_path)

        if not os.path.exists(self.CACHED_PATH):
            os.makedirs(self.CACHED_PATH)

    def process(self):
        dem = gdal.Open(self.dem_path, gdal.GA_ReadOnly)

        dataset = self.dem_processing(
            dem, self.SLOPE, self.FTYPE_DEM,
            computeEdges=True,
            alg='Horn',
            slopeFormat='percent',
        )
        dataset = self.create_plot(self.SLOPE, dataset)
        self.close_dataset(dataset)

        dataset = self.dem_processing(
            dem, self.ASPECT, self.FTYPE_DEM,
            computeEdges=True,
        )
        dataset = self.fix_dataset(dataset, self.ASPECT)
        dataset = self.create_plot(self.ASPECT, dataset)
        self.close_dataset(dataset)

        dataset = self.dem_processing(
            dem, self.HILLSHADE, self.FTYPE_DEM,
            computeEdges=True,
        )
        dataset = self.create_plot(self.HILLSHADE, dataset)
        self.close_dataset(dataset)

        dem.FlushCache()
        dem = None

    def close_dataset(self, dataset):
        dataset.FlushCache()
        dataset = None

    def create_plot(self, ptype, dataset):
        plt.figure()
        plt.title(f"{ptype[0].upper()}{ptype[1:]} of: {self.dem_path}")
        arr = dataset.GetRasterBand(1).ReadAsArray()
        print(arr.shape)
        plt.imshow(arr, cmap=self._cmap.get(ptype, None))
        plt.colorbar()
        plt.savefig(os.path.join(self.CACHED_PATH, f"{ptype}.png"))

        return dataset

    def get_path(self, prefix, ftype):
        if ftype not in ('dem', 'vector'):
            raise ValueError('Type is not supported.')
        return os.path.join(self.CACHED_PATH, f"{prefix.upper()}_{getattr(self, ftype + '_filename')}")

    def dem_processing(self, dem, processing, ftype, **kwargs):
        self._cache[processing] = self.get_path(prefix=processing, ftype=ftype)
        options = gdal.DEMProcessingOptions(**kwargs)
        return gdal.DEMProcessing(
            self._cache[processing],
            dem,
            processing,
            options=options
        )

    def fix_dataset(self, in_dataset, dtype):
        band = in_dataset.GetRasterBand(1)

        arr = band.ReadAsArray()
        if dtype == self.ASPECT:
            arr[arr == -9999] = -1
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

    ga = GisAnalyst(dem, vector, out)
    ga.process()


#   binmask for each
#   vector data
#   combine hillshade, binmask and vector togather


if __name__ == '__main__':
    main()
