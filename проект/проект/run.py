import argparse
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MAIN = {
    1: "lab1_main.py",
    2: "lab2_main.py",
    3: "lab3_main.py",
    4: "lab4_main.py",
    5: "lab5_main.py",
}


def run(lab, script):
    path = ROOT / "labs" / f"lab{lab}" / script
    if not path.is_file():
        raise SystemExit(f"Файл не знайдено: {path}")
    print(f"\n=== Лабораторна {lab}: {script} ===\n")
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lab", choices=["1", "2", "3", "4", "5", "all"])
    parser.add_argument("--script", help="Назва конкретного файла, наприклад lab5_experiment2.py")
    args = parser.parse_args()
    if args.lab == "all":
        if args.script:
            parser.error("--script не використовується з all")
        for lab in MAIN:
            run(lab, MAIN[lab])
    else:
        lab = int(args.lab)
        run(lab, args.script or MAIN[lab])
