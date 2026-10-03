# TruFor: воспроизведение Table 1 на CocoGlide

Единственный эксперимент: проверить опубликованное качество локализации подделок
TruFor на **всех 512 fake-изображениях CocoGlide** с официальными весами и метриками.
Целевые значения: **F1-best 0.720; F1 при пороге 0.5 — 0.523**.

Guillaro et al., *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery
Detection and Localization*, CVPR 2023, pp. 20606–20615.
[Оригинальная статья](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf),
[код авторов](https://github.com/grip-unina/TruFor/tree/ae54475df6f41a491d7615100feb19263dec13f7).

## Результаты

| Метрика | Статья, Table 1 | Наш результат | Абсолютная разница | Валидных изображений |
|---|---:|---:|---:|---:|
| Pixel F1, best threshold | 0.720 | 0.719951 | 0.000049 | 512/512 |
| Pixel F1, fixed 0.5 | 0.523 | 0.522640 | 0.000360 | 511/512 |

Совпадение до точности округления статьи. Один результат fixed F1 не определён
в авторской реализации; его ID и traceback сохранены, усреднение ведётся по
конечным значениям отдельно для каждой метрики, как в официальном примере.
Pixel AUC (0.882323, 506/512) — вспомогательная диагностика этого же прогона;
для CocoGlide в Table 1 эталонного pixel AUC нет. Это не image-level detection AUC.

[Отчёт и таблицы](reports/baseline/report.md) ·
[Качественные примеры](reports/baseline/figures/qualitative_analysis.md) ·
[Проверка требований курса](docs/course_checklist.md) ·
[Методика и ограничения](docs/discrepancy_analysis.md).

## Запуск на этом компьютере

Окружение, официальный код, данные и веса подготовлены. Из корня проекта:

```powershell
.\run_experiment.ps1 baseline
```

Команда проверяет данные, обрабатывает все 512 изображений, рассчитывает метрики и
строит отчёт с панелями. Выход: `results/table1_baseline/report.md`.
`sanity` вместо `baseline` — техническая проверка на первых 20 изображениях,
а не второй эксперимент и не полный benchmark.
Повтор той же команды продолжает прерванный запуск. Изменение кода, окружения,
данных или seed требует новой папки `--output`; предсказания разных настроек не смешиваются.

## Установка с нуля (Windows PowerShell)

Python 3.12, Git, NVIDIA GPU с совместимым драйвером. Выполнять последовательно:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/grip-unina/TruFor.git third_party/TruFor
git -C third_party/TruFor checkout ae54475df6f41a491d7615100feb19263dec13f7
.venv\Scripts\python.exe scripts/fetch_assets.py weights data
.venv\Scripts\python.exe -m unittest discover -s tests -v
.\run_experiment.ps1 baseline
```

Точные версии проверенного окружения: `reports/baseline/requirements.freeze.txt`.
Различия с авторской Linux-средой: [environment.md](docs/environment.md).
Модель и метрики импортируются из неизменённого upstream; фактический commit и
чистота checkout проверяются перед инференсом. Данные, веса, `.venv` и сырые NPZ
не предназначены для Git. Компактный проверенный отчёт хранится в `reports/baseline`.

## Отдельные команды

Если PowerShell не разрешает запуск `.ps1`, используйте Python:

```powershell
.venv\Scripts\python.exe scripts/experiment.py prepare
.venv\Scripts\python.exe scripts/experiment.py run --output results/my_baseline
.venv\Scripts\python.exe scripts/experiment.py evaluate --output results/my_baseline
.venv\Scripts\python.exe scripts/make_figures.py --output results/my_baseline
.venv\Scripts\python.exe scripts/verify_upstream.py --output results/my_baseline
```

`--device cpu` поддерживается для инференса, но основной прогон проверен на RTX 3070.
`make_figures.py` и оценка не требуют GPU. `verify_upstream.py` проверяет один пример
на NVIDIA GPU независимым запуском авторского `test.py`.

## Данные, протокол, настройки

- Официальный CocoGlide ZIP; пары строго по `data/table.csv`, все 512 fake-строк.
  Изображения 256x256, маски 0/255, белое = подделка. Размеры, бинарность,
  уникальность ID и SHA256 каждого файла проверяются. Real-изображения не оцениваются.
- Официальный финальный checkpoint, epoch 81; MD5 архива проверяется при загрузке,
  точный SHA256 модели — перед десериализацией. Вся модель, включая Noiseprint++, сохранена.
- Native resolution, без изменения RGB; Pillow RGB → float32 CHW → деление на **256**.
  Batch 1, FP32, cuDNN выключен по официальному YAML, AMP/TF32 выключены, seed 42,
  deterministic algorithms включены. Обучение и подбор гиперпараметров модели не проводятся.
- F1: непосредственно `metrics.py` авторов. Эрозия GT 15x15, дилатация 11x11;
  неопределённая граница исключается. Сохраняется авторский выбор ориентации.
  Best threshold определяется отдельно для каждого изображения с использованием GT:
  это оптимистичная oracle-оценка, не рабочий порог без разметки.
- Основные результаты — **среднее по изображениям**, отдельно по конечным значениям
  каждой метрики; не одна общая F1 на всех пикселях. Полнота прогона и знаменатели явные.
- Некорректные карты и отсутствующие NPZ останавливают оценку. Undefined от авторской
  функции сохраняется вместе с причиной. Нет подмены результатов для совпадения со статьёй.
- Прямой F1, pixel AUC, confidence и ошибки границ служат разбору тех же предсказаний.
  Панели используют прямой порог `>0.5`, поэтому могут выглядеть хуже официального F1,
  выбирающего ориентацию по GT. Noiseprint++ растягивается только для отображения.

## Что сохраняется

`raw/clean/*.npz`: неизменённый RGB, GT, anomaly, confidence, Noiseprint++, score,
размеры, hashes, время. `manifest.json`, `metadata.json`, `model_config.yaml`,
`runner_source.py`, `requirements.freeze.txt`: воспроизводимость конкретного запуска.
`metrics/`: per-image CSV, агрегаты, paper comparison, undefined IDs, версии оценщика.
`figures/`: сравнение со статьёй и пять автоматически выбранных примеров;
`sanity/`: первые 20 панелей в стабильном порядке. `failures.json`: ошибки инференса.

## Границы вывода и сдача

Подтверждается воспроизводимость одной строки Table 1. Нет выводов о превосходстве
над другими моделями, отдельном вкладе Noiseprint++ или обо всех видах подделок.
Одна модель/один набор/один seed: вариативность обучения не оценивается.
Для фиксированных весов проверяется повторяемость; статистическая значимость не заявляется.

По слайдам 13–15 правил курса одного содержательного результата достаточно.
Технические артефакты готовы; **короткая итоговая презентация ещё не создана**.
Студенту нужно уметь объяснить метод, метрики, undefined-случай и ограничения;
согласование темы/цели с преподавателем не заменяется автоматической проверкой.
