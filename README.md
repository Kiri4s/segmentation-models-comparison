# Сравнение моделей семантической сегментации на DeepGlobe Land Cover Classification датасете

## Обзор

Этот репозиторий посвящен сравнению различных моделей семантической сегментации (PSPNet, U-Net, FPN) на датасете DeepGlobe Land Cover Classification. Цель проекта — исследовать эффективность этих архитектур для задачи классификации типов земельного покрова по спутниковым снимкам. Репозиторий содержит код для подготовки данных, обучения моделей, их оценки и визуализации результатов.

## Структура проекта

```
.
├── .git/
├── .gitignore
├── .python-version
├── dataset/
│   ├── test/
│   ├── train/
│   └── valid/
├── notebooks/
│   ├── evaluate.ipynb
│   ├── training.ipynb
│   └── visualization.ipynb
├── pyproject.toml
├── README.md
├── results/
│   ├── collected/
│   └── ...
├── src/
│   ├── checkpoints/
│   ├── dataset.py
│   ├── diceloss.py
│   ├── evaluate.py
│   ├── main.py
│   ├── utils.py
│   └── visualization.py
└── uv.lock
```

## Содержание

* [Обзор](#обзор)
* [Структура проекта](#структура-проекта)
* [Датасет](#датасет)
* [Модели](#модели)
* [Использование](#использование)
    * [Установка](#установка)
    * [Обучение](#обучение)
    * [Оценка результатов](#оценка-результатов)
* [Результаты](#результаты)
* [Обсуждение Результатов](#обсуждение-результатов)

## Датасет

### Описание

В данном проекте используется датасет [DeepGlobe Land Cover Classification](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset). Датасет содержит спутниковые снимки и соответствующие им маски для семантической сегментации.

### Классы:

| класс                       | цвет      | метка |
|:----------------------------|:---------:|:-----:|
| Город/Поселение             | бирюзовый | 0     |
| Пашня                       | жёлтый    | 1     |
| Пастбище                    | пурпурный | 2     |
| Лес                         | зелёный   | 3     |
| Вода                        | синий     | 4     |
| Бесплодная земля            | белый     | 5     |
| Неизвестно                  | чёрный    | 6     |

### Примеры:

<div style="display:flex; align-items:center;">
    <img src=".//results/dataset_samples/6399_sat.jpg" alt="satelite" style="width:33%; max-width:400px;">
    <img src="./results/dataset_samples/6399_mask.png" alt="mask" style="width:33%; max-width:400px;">
</div>

<div style="display:flex; align-items:center;">
    <img src="./results/dataset_samples/10901_sat.jpg" alt="satelite" style="width:33%; max-width:400px;">
    <img src="./results/dataset_samples/10901_mask.png" alt="mask" style="width:33%; max-width:400px;">
</div>

<div style="display:flex; align-items:center;">
    <img src="./results/dataset_samples/855_sat.jpg" alt="satelite" style="width:33%; max-width:400px;">
    <img src="./results/dataset_samples/855_mask.png" alt="mask" style="width:33%; max-width:400px;">
</div>

Данные для обучения, валидации и тестирования находятся в директории `dataset/train`.

## Модели

В проекте реализованы и сравниваются следующие модели семантической сегментации:

* PSPNet (Pyramid Scene Parsing Network)
* U-Net
* FPN (Feature Pyramid Network)

Для загрузки моделей использовалась библиотека segmentation_models_pytorch.
Исходный код для обучения и оценки находятся в директории `src/`.

### Веса моделей доступны по ссылкам:

* [PSPNet](https://drive.google.com/file/d/1zJhmho7_O2emZrxoupOwFxq2s5REtOqV/view?usp=sharing)
* [U-Net](https://drive.google.com/file/d/1sOP2s_wqlDXmDhAUKfP2y9ODNnyDDrwp/view?usp=sharing)
* [FPN](https://drive.google.com/file/d/1WA_6BwBIqgultfo7x_5gSQFWrxcMTN3A/view?usp=sharing)

### Все модели обучались с конфигом:

```python
class StandartConfig(TypedDict):
    data_dir: str = "../dataset/train"
    val_size: float = 0.2
    test_size: float = 0.1
    transform = None
    target_transform = None
    batch_size: int = 1
    learning_rate: float = 2e-4
    epochs: int = 5
    encoder_name: str = "resnet18"
    encoder_weights: str = "imagenet"
    activation: str = "logsoftmax"
    in_channels: int = 3
    classes: int = 7
    device: str = "mps"
    checkpoints_dir: str = "./checkpoints"
    freeze_encoder_layers: int = -2  # freeze encoder excluding 2 last layers
```

- **Loss**: DiceLoss
- **Optimizer**: Adam
- **Core Metric**: mean IoU

## Использование

### Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/Kiri4s/segmentation-models-comparison.git
cd segmentation-models-comparison
```

2. Скачайте датасет [DeepGlobe Land Cover Classification](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) и расположите его в корневой директории проекта следующим образом:

```
.
└── dataset/
    └── train/
        ├── 119_mask.png
        ├── 119_sat.jpg
        ├── ...
        ├── 998002_mask.png
        └── 998002_sat.jpg
```

3. Установите [uv](https://docs.astral.sh/uv/) (для macOS и Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

4. Установите зависимости проекта из `pyproject.toml`:

```bash
uv sync
```

### Обучение

Для обучения моделей используется скрипт `src/main.py`. Параметры обучения можно изменить в классе `StandartConfig` в `src/main.py`.

Пример запуска обучения для модели U-Net:

```bash
cd src && uv run main.py --model_name='unet'
```

Для возобновления обучения с контрольной точки:

```bash
cd src && uv run main.py --model_name='unet' --load_from_checkpoint="path2checkpoint"
```

Обученные модели и история обучения сохраняются в директории `src/checkpoints/`.

### Оценка результатов

Для оценки результатов используется скрипт `src/evaluate.py`.

```bash
cd src && uv run evaluate.py
```

## Результаты

Результаты экспериментов, включая графики обучения и матрицы ошибок, сохраняются в директории `results/`.
История обучения (train & val loss, val IoU) хранится в директории `src/checkpoints/`.

### Визуальное сравнение

**Для каждой картинки последовательно сверху вниз: pspnet, unet, fpn**

<div style="display:flex; gap:8px; align-items:center;">
    <img src="./results/collected/comparison_0.png" alt="" style="width:50%; max-width:600px;">
</div>
</br>
<div style="display:flex; gap:8px; align-items:center;">
    <img src="./results/collected/comparison_1.png" alt="" style="width:50%; max-width:600px;">
</div>
</br>
<div style="display:flex; gap:8px; align-items:center;">
    <img src="./results/collected/comparison_2.png" alt="" style="width:50%; max-width:600px;">
</div>

### Loss

<div style="display:flex; gap:8px; align-items:center;">
    <img src="./results/collected/train_loss_epochs.png" alt="train loss epochs" style="width:33%; max-width:400px;">
    <img src="./results/collected/train_loss_steps.png" alt="train loss steps" style="width:33%; max-width:400px;">
</div>

### IoU

<div style="display:flex; gap:8px; align-items:center;">
    <img src="./results/collected/val_iou_epochs.png" alt="val iou epochs" style="width:33%; max-width:400px;">
    <img src="./results/collected/val_iou_steps.png" alt="val iou steps" style="width:33%; max-width:400px;">
</div>

### Confusion matrix

<div style="display:flex; gap:8px; align-items:center;">
    <img src="./results/pspnet_Epochs:5_lf:DiceLoss_lr:0.0002_confmatrix.png" alt="pspnet" style="width:33%; max-width:400px;">
    <img src="./results/unet_Epochs_4_lf_DiceLoss_lr_0.0002_confmatrix.png" alt="unet" style="width:33%; max-width:400px;">
    <img src="./results/fpn_Epochs:5_lf:DiceLoss_lr:0.0002_confmatrix.png" alt="fpn" style="width:33%; max-width:400px;">
</div>

### Сравнение моделей по среднему IoU на тестовой выборке

| # | model  | loss      | mean_iou  |
|:-:|:------:|:---------:|:---------:|
| 0 | pspnet |  0.410559 |  0.317334 |
| 1 |   unet |  0.280340 |  0.474416 |
| 2 |    fpn |  0.441761 |  0.478314 |

## Обсуждение Результатов

Оценивая работу моделей визуально, нетрудно заметить что модели unet и fpn демонстрируют результаты похожие по качеству, а pspnet сильно отстаёт от них. Это может быть вызвано недостатком параметров модели (для всех моделей использовалась одна и та же resnet18 backbone сеть, с замороженными слоями для энкодера за исключением двух последних). В поддержку pspnet можно сказать что эта модель имеет меньшее время схождения и более высокую скорость работы (в ~2 раза быстрее unet & fpn). 

По графикам лосса видно, что: pspnet сходится быстрее; unet и fpn ещё имеют потенциал для обучения. Средний IoU у unet и fpn выше чем у pspnet, что подтверждается на предсказаниях.

Матрицы ошибок показали, что для pspnet свойственно предсказывать пастбище как пастбище и пашню равновероятно. Такая же склонность есть и у других моделей, но менее выраженная. Вместо бесплодной земли pspnet предсказывает пашню. Матрицы ошибок у unet и fpn близки к диагональным. Все модели вообще не предсказывают неизвестный класс, предпочитая отнести его (за малочисленностью) к какому-нибудь другому.

Результаты сравнения моделей по среднему IoU на тестовой выборке показали что unet и fpn близки по ппроизводительности, но fpn на ~0.004 пункта лучше. Поскольку лосс fpn выше чем у unet я бы предпочёл первую модель как более способную к дальнейшему обучению.