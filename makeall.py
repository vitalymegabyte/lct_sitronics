import os
from main import main

crops = os.listdir('crops')
layouts = os.listdir('layouts')

for crop in crops:
    crop_path = os.path.join('crops', crop)
    for layout in layouts:
        layout_path = os.path.join('layouts', layout)

        main(crop_path, layout_path)
