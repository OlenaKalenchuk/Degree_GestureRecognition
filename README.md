# ASL Gesture Recognition

У папці api/ міститься backend частина, налаштована з Python FastApi, у gr-frontend/ простий браузерний frontend на базі React + Typescript. У нотбуці ModelSetup.ipynb знаходиться видобування даних з датасету і налаштування моделей для класифікації жестів слів ASL (WLASL датасет), у ModelSetup_CTC.ipynb - для класифікації послідовностей жестикуляції символів (ChicagoFSWild датасет).

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
Для проєкту використано датасет WLASL (Word-Level American Sign Language) [1], описаний та доступний за посиланнями [GitHub](https://dxli94.github.io/WLASL/) (офіційний репозиторій), [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed), а також датасет ChicagoFSWild [2], що складається з відео жестикуляції дактильної абетки ASL, доступний за посиланням [Chicago Fingerspelling in the Wild](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm).

Посилання на статті: 
1) Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, Hongdong Li, [Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison](https://arxiv.org/abs/1910.11006)
2) B. Shi, A. Martinez Del Rio, J. Keane, J. Michaux, D. Brentari, G. Shakhnarovich, and K. Livescu, [American Sign Language fingerspelling recognition in the wild](https://arxiv.org/pdf/1810.11438)
