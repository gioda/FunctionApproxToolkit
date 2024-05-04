'''
 # @ Author: Giovanni Dalmasso
 # @ Create Time: 04-05-2024 19:59:26
 # @ Modified by: Giovanni Dalmasso
 # @ Last Modified: 04-05-2024 19:59:27
 # @ Description: This Python script demonstrates the use of Taylor and Pade series approximations for the sine function using symbolic computation and numerical plotting.
 '''

import numpy as np
import sympy as sp
from vedo import printc

from approximations import PadeApproximation, TaylorApproximation
from plotter import plot_function_and_pade, plot_function_and_series


def get_function_input():
    """
    Creates and returns a function that computes the sine of a given input using sympy.

    Returns:
        func: A function that takes a single argument x and returns sp.sin(x).
    """
    def func(x):
        return sp.sin(x)  # Sympy sin function
    return func

def main():
    """
    Main function to demonstrate Taylor and Pade series approximations.

    This function first computes and plots the Taylor series approximation for the sine function.
    It then computes and plots the Pade approximation for the same function.
    """
    printc("\nTaylor Series Approximation\n", c="blue")
    # Get the function input
    func = get_function_input()
    degree = 4
    point = 0

    # Create and compute Taylor series
    taylor_series = TaylorApproximation(func, degree, point)
    taylor_series.compute_coefficients()
    printc(taylor_series, c="green")

    # Plot Taylor approximation
    original_func = sp.lambdify(taylor_series.x, taylor_series.func_symbolic, modules=["numpy"])
    plot_function_and_series(original_func, taylor_series.evaluate_series,
                             x_range=(-np.pi, np.pi), y_range=(-1.1, 1.1), trigonometric=True)

    printc("\nPade Approximation\n", c="blue")
    # Create and compute Pade approximation
    pade_approx = PadeApproximation(func, 3, 2)
    pade_approx.compute_coefficients()
    printc(pade_approx, c="green")

    # Lambdify the original function for Pade plotting
    pade_original_func = sp.lambdify(taylor_series.x, func(taylor_series.x), modules=["numpy"])
    plot_function_and_pade(pade_original_func, pade_approx.evaluate, x_range=(-np.pi, np.pi), y_range=(-1.1, 1.1), trigonometric=True)

if __name__ == "__main__":
    main()
