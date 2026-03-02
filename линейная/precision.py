"""
Bonus: программа для расчёта точности обученной модели (R²).
Загружает модель из theta.csv и данные из data.csv,
выводит precision в процентах.
"""
from linear_regression import load_data, load_model, precision


def main() -> None:
    try:
        mileages, prices = load_data("data.csv")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return

    theta0, theta1 = load_model()

    prec = precision(mileages, prices, theta0, theta1)
    print(f"Precision (R²): {prec * 100:.2f}%")


if __name__ == "__main__":
    main()
