from datetime import datetime
from convert import read_img
from typing import Tuple

import numpy as np
import cv2


from modules.xfeat import XFeat

# опытным путем установлено, что лучше всего приближение размеров достигается вот так :)
MINIFY_CONSTANT = 8

xfeat = XFeat()


def get_corners_coordinates(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(
        ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    return warped_corners


def pixel_to_geo(geo_transform, pixel_x, pixel_y):
    """
    Преобразует координаты пикселя в географические координаты.

    :param geo_transform: Геотрансформация (tuple),
                          содержащая 6 элементов: (origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height)
    :param pixel_x: Координата пикселя X (int)
    :param pixel_y: Координата пикселя Y (int)
    :return: Географические координаты (tuple) (geo_x, geo_y)
    """
    origin_x = geo_transform[0]
    pixel_width = geo_transform[1]
    row_rotation = geo_transform[2]
    origin_y = geo_transform[3]
    col_rotation = geo_transform[4]
    pixel_height = geo_transform[5]

    geo_x = origin_x + pixel_x * pixel_width + pixel_y * row_rotation
    geo_y = origin_y + pixel_x * col_rotation + pixel_y * pixel_height

    return geo_x, geo_y


def geo_to_pixel(geo_transform, geo_x, geo_y):
    """
    Преобразует географические координаты в координаты пикселя.

    :param geo_transform: Геотрансформация (tuple),
                          содержащая 6 элементов: (origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height)
    :param geo_x: Географические координаты X (float)
    :param geo_y: Географические координаты Y (float)
    :return: Координаты пикселя (tuple) (pixel_x, pixel_y)
    """
    origin_x = geo_transform[0]
    pixel_width = geo_transform[1]
    row_rotation = geo_transform[2]
    origin_y = geo_transform[3]
    col_rotation = geo_transform[4]
    pixel_height = geo_transform[5]

    # Создаем матрицу из геотрансформации
    transform_matrix = np.array(
        [[pixel_width, row_rotation], [col_rotation, pixel_height]]
    )

    # Попробуем инвертировать матрицу
    try:
        inv_transform_matrix = np.linalg.inv(transform_matrix)
    except np.linalg.LinAlgError:
        raise Exception("Геотрансформация не может быть инвертирована")

    # Применяем инвертированную матрицу для нахождения координат пикселя
    diff_x = geo_x - origin_x
    diff_y = geo_y - origin_y

    pixel_coords = np.dot(inv_transform_matrix, [diff_x, diff_y])

    pixel_x, pixel_y = pixel_coords[0], pixel_coords[1]

    return int(pixel_x), int(pixel_y)


def corner_coords(im1_name, im2_name) -> Tuple[tuple, str, datetime, datetime]:

    im1, geo_transform1, epsg1 = read_img(im1_name)
    im2, geo_transform2, epsg2 = read_img(im2_name)

    start_time = datetime.now()

    im2_to_process = im2[::MINIFY_CONSTANT, ::MINIFY_CONSTANT, :]
    # Use out-of-the-box function for extraction + MNN matching
    mkpts_0, mkpts_1 = xfeat.match_xfeat(im1, im2_to_process, top_k=4096)

    corners = get_corners_coordinates(mkpts_0, mkpts_1, im1, im2)

    corners *= MINIFY_CONSTANT

    corners_coordinates = [
        pixel_to_geo(geo_transform2, c[0][0], c[0][1]) for c in corners
    ]
    end_time = datetime.now()
    return corners_coordinates, epsg2, start_time, end_time


if __name__ == "__main__":
    import sys

    # ОБЯЗАТЕЛЬНО первым кроп, вторым подложку
    im1_name = sys.argv[-2]
    im2_name = sys.argv[-1]

    print(corner_coords(im1_name, im2_name))
