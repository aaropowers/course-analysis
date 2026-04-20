from __future__ import annotations

from src.coursework_bootstrap import generate_coursework_assets


def main() -> None:
    summary, validation = generate_coursework_assets()
    print("Coursework dataset generation complete.")
    print(
        f"Cleaned rows: {summary.cleaned_rows}, cleaned students: {summary.cleaned_students}, "
        f"bootstrapped rows: {summary.bootstrapped_rows}, bootstrapped students: {summary.bootstrapped_students}"
    )
    print(f"Synthetic students added: {summary.synthetic_students_added}")
    print("Validation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
