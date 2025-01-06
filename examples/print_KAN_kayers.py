import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

df_2_i = pd.read_csv("./df_kan_linear_2_i_concat.csv")
df_2_o = pd.read_csv("./df_kan_linear_2_o_concat.csv")
df_3_i = pd.read_csv("./df_kan_linear_3_i_concat.csv")
df_3_o = pd.read_csv("./df_kan_linear_3_o_concat.csv")

print(df_2_i.shape)
print(df_2_o.shape)
print(df_3_i.shape)
print(df_3_o.shape)

df_input = df_3_i
df_output = df_3_o

degrees = [1, 2, 3, 4]

d = dict()

for col_input, col_output in zip(df_input.columns, df_output.columns):

    d[f'col_{col_input}'] = dict()

    plt.figure(figsize=(10, 6))
    plt.scatter(df_input[col_input], df_output[col_output], label=f"Data - {col_input}")

    for degree in degrees:
        coeffs = np.polyfit(df_input[col_input], df_output[col_output], degree)

        x_vals = np.linspace(min(df_input[col_input]), max(df_input[col_input]), 100)
        y_vals = np.polyval(coeffs, x_vals)

        y_pred = np.polyval(coeffs, df_input[col_input])

        plt.plot(x_vals, y_vals, label=f"Poly Fit (Degree {degree})")

        d[f'col_{col_input}'][f'degree_{degree}'] = r2_score(df_output[col_output], y_pred)

    d_ = d[f'col_{col_input}']
    print(f"R-squared for column {col_input} | degree 1 = {d_[f'degree_1']:.4f}, "
          f"degree 2 = {d_[f'degree_2']:.4f}, degree 3 = {d_[f'degree_3']:.4f}, degree 4 = {d_[f'degree_4']:.4f}")

    plt.xlabel('Input (x)')
    plt.ylabel('Output (f(x))')
    plt.title(f'Scatter Plot with Polynomial Fits for column {col_input}')
    plt.legend()
    plt.show()

print(d)