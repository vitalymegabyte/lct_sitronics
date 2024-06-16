from osgeo import gdal
import sys
import cv2 as cv
import numpy as np
from typing import Any, Tuple
import re

EPSG_REGEX = r"AUTHORITY\[\"EPSG\",\"([\d]+)\"\]\]$"


def process_band(band: np.uint16) -> np.uint8:
    band = band.astype(np.float64)
    values = np.percentile(band, [2, 98])
    min_value = values[0]
    max_value = values[1]

    band[band < min_value] = min_value
    band[band > max_value] = max_value

    band -= min_value
    band /= max_value

    band *= 255

    return band.astype(np.uint8)


def read_img(im_path: str) -> Tuple[np.uint8, Any]:
    dataset = gdal.Open(im_path)

    # Читаем данные из нужных каналов (предполагаем, что RGB находятся в каналах 1, 2 и 3)
    r_band = dataset.GetRasterBand(1).ReadAsArray()
    g_band = dataset.GetRasterBand(2).ReadAsArray()
    b_band = dataset.GetRasterBand(3).ReadAsArray()

    r_band = process_band(r_band)
    g_band = process_band(g_band)
    b_band = process_band(b_band)

    # Создаем RGB-изображение, объединяя каналы
    rgb_image = np.dstack((r_band, g_band, b_band))

    rgb_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

    geo_transform = dataset.GetGeoTransform()

    epsg = re.search(EPSG_REGEX, dataset.GetProjection()).groups(1)[0]

    return rgb_image, geo_transform, epsg


if __name__ == "__main__":
    import sys

    im_path = sys.argv[-1]
    rgb_image = read_img(im_path)

    cv.imwrite("image2.png", rgb_image)
