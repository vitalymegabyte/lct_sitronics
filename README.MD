# ЛЦТ 2024, команда N Aero, кейс Sitronics

Для взаимодействия сделан стандартный CLI, описанный в ТГ:

```
2) В случае выдачи модуля с запуском посредством скрипта, запуск производится с двумя аргументами: 
main.py --crop_name {filename}.tif --layout_name {filename}.tif  включая полные пути. 

Результат обработки записывается в coords.csv файл с колонками: 
«layout_name» имя подложки,
«crop_name» имя снимка,  
«ul», «ur», «br», «bl», где лево-верх, право-верх, право-низ, лево-низ координаты, 
«crs» координатная система в формате «EPSG:{12345}», 
«start» и «end» время в формате «%Y-%m-%dT%h:%m:%s» начала и окончания обработки единичного загруженного снимка, после загрузки подложки и снимка, для точного контроля времени обработки."
```

## Запуск через Docker

**Ахтунг! Хайли рекомменд использовать именно этот способ!**

Для запуска из под докера собираем образ:

```
docker build -t naero_test .
```

Так как полноценнее расписывать Readme и делать правильные тома у авторов сиего шедевра (нас, то есть) нет, мы просто перенесли все подложки и кропы в подпапку layouts, и сделали общий том на текущую папку. Выглядит это так:

```
docker run -v .:/work naero_test python3 main.py --layout_name layouts/layout_2021-10-10.tif --crop_name layouts/crop_0_1_0000.tif
```

Данные выводятся в CSV-шку coords.csv, так же в описанном формате:

```csv
layout_name,crop_name,ul,ur,br,bl,crs,start,end
layouts/layout_2021-10-10.tif,layouts/crop_0_1_0000.tif,"[379040.9521484375, 5799911.265096664]","[-100610.78125, 5782552.625732422]","[504523.41796875, 5746100.986328125]","[401945.96450805664, 5771181.904296875]",EPSG:32637,2024-06-16 19:26:14.376656,2024-06-16 19:26:14.728911
```

По мере осуществления новых прогонов, и добавления новых данных, CSVшка дополняется.