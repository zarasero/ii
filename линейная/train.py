import sys

from linear_regression import (
    load_data,
    train_from_csv,
    save_model,
    precision,
    plot_result,
)


def main() -> None:
    data_path = "data.csv"
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    mileages, prices = load_data(data_path)
    theta0, theta1 = train_from_csv(
        data_path,
        learning_rate=0.1,
        iterations=1000,
        verbose=verbose,
    )
    save_model(theta0, theta1)

    print("Training complete.")
    print(f"theta0 = {theta0:.4f}")
    print(f"theta1 = {theta1:.8f}")

    prec = precision(mileages, prices, theta0, theta1)
    print(f"Precision (R²): {prec * 100:.2f}%")

    plot_result(mileages, prices, theta0, theta1, save_path="plot.png")


if __name__ == "__main__":
    main()

