## ft_linear_regression (Python)

This project implements a simple linear regression with one feature (car mileage) and trains it using gradient descent, as described in the subject.

There are two main programs:

- `train.py` — trains the model on `data.csv` and saves the parameters.
- `predict.py` — loads the saved parameters and predicts the price of a car for a given mileage.

### Data format

The input dataset is expected in `data.csv` with the following format:

```text
km,price
240000,3650
139800,3800
...
```

Both values are treated as floating-point numbers.

### How training works

The model uses the hypothesis:

\[
\text{estimatePrice}(mileage) = \theta_0 + \theta_1 \cdot mileage
\]

Training is performed with gradient descent. For numerical stability, the mileage feature is normalized during training, and the learned parameters are converted back so that the final \(\theta_0\) and \(\theta_1\) work directly with the original mileage values.

The training script:

```bash
python3 train.py
```

This will:

- Read `data.csv`.
- Train the model.
- Save the resulting \(\theta_0\) and \(\theta_1\) into `theta.csv` (as two columns: `theta0,theta1`).

### Predicting a price

To estimate the price for a given mileage:

```bash
python3 predict.py
```

The program will:

- Load \(\theta_0\) and \(\theta_1\) from `theta.csv` if it exists (otherwise both are assumed to be `0.0`).
- Ask you to input a mileage in kilometers.
- Print the estimated price.

### Bonus

- **Plotting**: After training, `train.py` plots the data points and the regression line, and saves the figure to `plot.png`.
- **Precision**: Training prints R² (coefficient of determination). A separate program `precision.py` calculates and prints the precision of the current model:

  ```bash
  python3 precision.py
  ```

### Dependencies

For the bonus features (plotting), install matplotlib:

```bash
python3 -m pip install matplotlib
```

(See also **КАК_ЗАПУСКАТЬ.md** for step-by-step run instructions in Russian.)

