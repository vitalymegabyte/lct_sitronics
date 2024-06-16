import argparse
import pandas as pd
from process import corner_coords
import os


def main(crop_name, layout_name):
    coordinates, epsg, start, end = corner_coords(crop_name, layout_name)
    data_dict = {
        "layout_name": layout_name,
        "crop_name": crop_name,
        "ul": list(coordinates[0]),
        "ur": list(coordinates[1]),
        "br": list(coordinates[2]),
        "bl": list(coordinates[3]),
        "crs": f"EPSG:{epsg}",
        "start": start,
        "end": end,
    }

    output_data = pd.DataFrame([data_dict])
    if os.path.exists("coords.csv"):
        output_data = pd.concat((pd.read_csv("coords.csv"), output_data))
    output_data.to_csv("coords.csv", index=False)


if __name__ == "__main__":
    # Инициализация парсера аргументов
    parser = argparse.ArgumentParser(description="Process crop and layout TIFF files")

    # Добавляем аргументы
    parser.add_argument(
        '--crop_name', type=str, required=True, help='Path to the crop TIFF file'
    )
    parser.add_argument(
        '--layout_name', type=str, required=True, help='Path to the layout TIFF file'
    )

    # Парсим аргументы
    args = parser.parse_args()

    # Вызов основной функции с аргументами
    main(args.crop_name, args.layout_name)
