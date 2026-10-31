# Проєкт 2.0 — оптимізація зображень

- `run.py` — єдина точка запуску.
- `labs/lab1` ... `labs/lab5` — основні програми й досліди відповідних лабораторних.
- `src/` — алгоритми та спільні функції, **без змін**.
- `outputs/` — наявні графіки та зображення, **без змін**.

Запускати з кореня проєкту у VS Code:

```powershell
python run.py 1
python run.py 2
python run.py 3
python run.py 4
python run.py 5
python run.py all
```

Окремі досліди та графіки:

```powershell
python run.py 5 --script lab5_experiment4.py
python run.py 5 --script lab5_comparison.py
python run.py 4 --script lab4_tv_main.py
```

`all` запускає **лише основні файли** п'яти лабораторних. Для всіх додаткових дослідів використовуйте `--script`.

Віртуальне середовище `.venv` не включене до архіву; можна користуватися вже налаштованим Python у VS Code.
