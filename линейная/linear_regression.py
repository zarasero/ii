import csv
from typing import List, Tuple


MODEL_PATH = "theta.csv"


def load_data(path: str) -> Tuple[List[float], List[float]]:
    mileages: List[float] = []
    prices: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        # Read all non-empty lines
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        raise ValueError("Dataset file is empty.")

    # First non-empty line is the header
    raw_header = lines[0]
    header_parts = [h.strip().lstrip("\ufeff") for h in raw_header.split(",")]

    try:
        km_index = header_parts.index("km")
        price_index = header_parts.index("price")
    except ValueError as exc:
        raise ValueError("Dataset must contain 'km' and 'price' columns.") from exc

    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) <= max(km_index, price_index):
            continue
        try:
            mileage = float(parts[km_index])
            price = float(parts[price_index])
        except ValueError:
            continue
        mileages.append(mileage)
        prices.append(price)

    if not mileages or not prices:
        raise ValueError("Dataset is empty or invalid.")

    return mileages, prices


def _normalize_feature(xs: List[float]) -> Tuple[List[float], float, float]:
    minimum = min(xs)
    maximum = max(xs)
    if maximum == minimum:
        # All values are identical
        return [0.0 for _ in xs], minimum, maximum

    span = maximum - minimum
    normalized = [(x - minimum) / span for x in xs]
    return normalized, minimum, maximum


def _cost(xs: List[float], ys: List[float], theta0: float, theta1: float) -> float:
    """Среднеквадратичная ошибка (MSE) для отображения."""
    m = len(xs)
    return sum((theta0 + theta1 * x - y) ** 2 for x, y in zip(xs, ys)) / (2 * m)


def _cost_with_breakdown(
    xs: List[float], ys: List[float], theta0: float, theta1: float, show_first_n: int = 5
) -> Tuple[float, List[Tuple[float, float, float, float, float]]]:
    """
    Считает cost и возвращает breakdown для первых show_first_n точек:
    (mileage, price, estimate, error, squared_error)
    """
    m = len(xs)
    squared_errors: List[float] = []
    breakdown: List[Tuple[float, float, float, float, float]] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        estimate = theta0 + theta1 * x
        error = estimate - y
        sq = error ** 2
        squared_errors.append(sq)
        if i < show_first_n:
            breakdown.append((x, y, estimate, error, sq))
    total = sum(squared_errors)
    cost = total / (2 * m)
    return cost, breakdown, total


def _gradient_descent(
    xs: List[float],
    ys: List[float],
    learning_rate: float,
    iterations: int,
    verbose: bool = False,
) -> Tuple[float, float]:
    if len(xs) != len(ys):
        raise ValueError("Input and output arrays must have the same length.")

    m = len(xs)
    if m == 0:
        raise ValueError("Cannot train on an empty dataset.")

    theta0 = 0.0
    theta1 = 0.0

    print_interval = max(1, iterations // 10) if verbose else 0

    for i in range(iterations):
        sum_errors = 0.0
        sum_errors_x = 0.0

        for x, y in zip(xs, ys):
            estimate = theta0 + theta1 * x
            error = estimate - y
            sum_errors += error
            sum_errors_x += error * x

        tmp_theta0 = learning_rate * (sum_errors / m)
        tmp_theta1 = learning_rate * (sum_errors_x / m)

        if verbose and i == 0:
            new_t0 = theta0 - tmp_theta0
            new_t1 = theta1 - tmp_theta1
            print("\n  --- Итерация 0 (первый шаг) ---")
            print(f"    sum(estimate - price) = {sum_errors:.4f}")
            print(f"    sum((estimate - price) * mileage) = {sum_errors_x:.4f}")
            print(f"    tmp_θ0 = lr * (1/m) * sum = {learning_rate} * {1/m} * {sum_errors:.4f} = {tmp_theta0:.6f}")
            print(f"    tmp_θ1 = lr * (1/m) * sum_x = {learning_rate} * {1/m} * {sum_errors_x:.4f} = {tmp_theta1:.6f}")
            print(f"    θ0 := θ0 - tmp_θ0 = {theta0} - {tmp_theta0:.6f} = {new_t0:.6f}")
            print(f"    θ1 := θ1 - tmp_θ1 = {theta1} - {tmp_theta1:.6f} = {new_t1:.6f}")

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        if verbose and (i == 0 or i == iterations - 1 or (print_interval and i % print_interval == 0)):
            cost, breakdown, total_sq = _cost_with_breakdown(xs, ys, theta0, theta1)
            if i == 0:
                print("\n  --- Расчёт cost (MSE) ---")
                print(f"    Формула: cost = (1/(2*m)) * Σ(estimate - price)²")
                print(f"    Для каждой точки: estimate = θ0 + θ1*x, error = estimate - price, (error)²")
                for j, (x, y, est, err, sq) in enumerate(breakdown):
                    print(f"    Точка {j+1}: x={x:.4f}, price={y:.2f} → estimate={est:.4f}, error={err:.4f}, (error)²={sq:.4f}")
                if len(xs) > len(breakdown):
                    print(f"    ... (ещё {len(xs) - len(breakdown)} точек)")
                print(f"    Σ(estimate - price)² = {total_sq:.4f}")
                print(f"    cost = {total_sq:.4f} / (2*{m}) = {cost:.4f}")
            print(f"  Итерация {i}: θ0={theta0:.6f}, θ1={theta1:.6f}, cost={cost:.4f}")

    return theta0, theta1


def train_from_csv(
    data_path: str,
    learning_rate: float = 0.1,
    iterations: int = 1000,
    verbose: bool = False,
) -> Tuple[float, float]:
    if verbose:
        print("=" * 50)
        print("ПОРЯДОК ОБУЧЕНИЯ")
        print("=" * 50)

    # Шаг 1: Загрузка данных
    mileages, prices = load_data(data_path)
    m = len(mileages)
    if verbose:
        print("\n1. Загрузка данных")
        print(f"   Файл: {data_path}")
        print(f"   Количество примеров m = {m}")
        print(f"   Пробег: min={min(mileages):.0f}, max={max(mileages):.0f} км")
        print(f"   Цена: min={min(prices):.0f}, max={max(prices):.0f}")

    # Шаг 2: Нормализация признака
    norm_mileages, min_mileage, max_mileage = _normalize_feature(mileages)
    span = max_mileage - min_mileage

    if max_mileage == min_mileage:
        avg_price = sum(prices) / len(prices)
        theta0 = avg_price
        theta1 = 0.0
        if verbose:
            print("\n2. Все пробеги одинаковы → константная модель")
        return theta0, theta1

    if verbose:
        print("\n2. Нормализация пробега (для устойчивости обучения)")
        print(f"   x_norm = (x - {min_mileage:.0f}) / {span:.0f}")
        print(f"   Пробег в диапазоне [0, 1]")

    # Шаг 3: Градиентный спуск
    if verbose:
        print("\n3. Градиентный спуск")
        print(f"   Начальные θ0=0, θ1=0")
        print(f"   learning_rate={learning_rate}, iterations={iterations}")
        print(f"   Формулы: tmp_θ0 = lr * (1/m) * Σ(estimate - price)")
        print(f"           tmp_θ1 = lr * (1/m) * Σ((estimate - price) * mileage)")

    a, b = _gradient_descent(
        norm_mileages, prices, learning_rate, iterations, verbose=verbose
    )

    # Шаг 4: Денормализация параметров
    theta1 = b / span
    theta0 = a - b * min_mileage / span

    if verbose:
        print("\n4. Денормализация (перевод в исходный масштаб км)")
        print(f"   Параметры после спуска (норм. пространство): a={a:.6f}, b={b:.6f}")
        print(f"   θ1 = b / span = {b:.6f} / {span:.0f} = {theta1:.8f}")
        print(f"   θ0 = a - b*min/span = {a:.6f} - {b*min_mileage/span:.4f} = {theta0:.4f}")
        print("=" * 50)

    return theta0, theta1


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def precision(
    mileages: List[float],
    prices: List[float],
    theta0: float,
    theta1: float,
) -> float:
    """
    R² (coefficient of determination): 1 - SS_res / SS_tot
    SS_res = sum((price - predicted)^2)
    SS_tot = sum((price - mean(price))^2)
    """
    if len(mileages) != len(prices) or len(prices) < 2:
        return 0.0
    mean_price = sum(prices) / len(prices)
    ss_res = sum(
        (p - estimate_price(m, theta0, theta1)) ** 2 for m, p in zip(mileages, prices)
    )
    ss_tot = sum((p - mean_price) ** 2 for p in prices)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def plot_result(
    mileages: List[float],
    prices: List[float],
    theta0: float,
    theta1: float,
    save_path: str | None = "plot.png",
) -> None:
    """Plot data points and regression line."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    fig, ax = plt.subplots()
    ax.scatter(mileages, prices, label="Data", alpha=0.7)

    x_min, x_max = min(mileages), max(mileages)
    line_x = [x_min, x_max]
    line_y = [estimate_price(x, theta0, theta1) for x in line_x]
    ax.plot(line_x, line_y, "r-", label="Regression line", linewidth=2)

    ax.set_xlabel("Mileage (km)")
    ax.set_ylabel("Price")
    ax.set_title("Linear Regression: Car Price vs Mileage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")
    plt.show()


def save_model(theta0: float, theta1: float, path: str = MODEL_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta0", "theta1"])
        writer.writerow([theta0, theta1])


def load_model(path: str = MODEL_PATH) -> Tuple[float, float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError("Theta file must contain at least a header and one data row.")

        header_parts = [h.strip().lstrip("\ufeff") for h in lines[0].split(",")]
        try:
            t0_index = header_parts.index("theta0")
            t1_index = header_parts.index("theta1")
        except ValueError as exc:
            raise ValueError("Theta file must contain 'theta0' and 'theta1' columns.") from exc

        values = [v.strip() for v in lines[1].split(",")]
        theta0 = float(values[t0_index])
        theta1 = float(values[t1_index])
        return theta0, theta1
    except (FileNotFoundError, ValueError):
        # Before training, both thetas are 0 as specified.
        return 0.0, 0.0

