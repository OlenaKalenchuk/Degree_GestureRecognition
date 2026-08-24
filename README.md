# ASL Gesture Recognition

У папці api/ міститься backend частина, налаштована з Python FastApi, у gr-frontend/ простий браузерний frontend на базі React + Typescript. У нотбуці ModelSetup.ipynb знаходиться видобування даних з датасету і налаштування моделі.

### Prerequisites
Для роботи проєкту потрібно:
- python (3.14+)
- node.js
- npm

### Installation
Клонуйте репозиторій через команду:
```
git clone https://github.com/OlenaKalenchuk/Degree_GestureRecognition.git
```

Створіть віртуальне середовище python:
```
python -m venv .venv
```

Встановіть `pip` та необхідні бібліотеки:
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset
Для проєкту використано датасет WLASL (Word-Level American Sign Language), описаний та доступний за посиланнями [GitHub](https://dxli94.github.io/WLASL/) (офіційний репозиторій), [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed).

Посилання на статтю: [Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison](https://arxiv.org/abs/1910.11006)
