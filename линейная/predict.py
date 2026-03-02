from linear_regression import load_model, estimate_price


def main() -> None:
    theta0, theta1 = load_model()

    try:
        user_input = input("Enter car mileage (km): ").strip()
    except EOFError:
        print("No input provided.")
        return

    if not user_input:
        print("No mileage provided.")
        return

    try:
        mileage = float(user_input)
    except ValueError:
        print("Invalid mileage, please enter a number.")
        return

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price:.2f}")


if __name__ == "__main__":
    main()

