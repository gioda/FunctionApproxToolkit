'''
 # @ Author: Giovanni Dalmasso
 # @ Create Time: 04-05-2024 19:53:32
 # @ Modified by: Giovanni Dalmasso
 # @ Last Modified: 04-05-2024 19:54:35
 # @ Description: This module provides classes for computing and evaluating Taylor and Pade series approximations of mathematical functions.
 '''

from scipy.interpolate import pade
from sympy import diff, factorial, lambdify, series, symbols


class TaylorApproximation:
    """
    A class to compute and evaluate the Taylor series approximation of a given function.

    Attributes:
        x (Symbol): The symbolic representation of the variable.
        degree (int): The degree of the Taylor series.
        point (float): The point around which the Taylor series is expanded.
        func_symbolic (Expr): The symbolic representation of the function.
        coefficients (list): The list of computed coefficients of the Taylor series.

    Methods:
        compute_coefficients(): Computes the coefficients of the Taylor series.
        series(): Constructs the Taylor series expansion as a sympy expression.
        evaluate_series(x_values): Evaluates the Taylor series at given numerical x_values.
        __str__(): Returns a string representation of the Taylor series equation.
    """

    def __init__(self, function, degree, point=0):
        self.x = symbols('x')
        self.degree = degree
        self.point = point
        self.func_symbolic = function(self.x)
        self.coefficients = []

    def compute_coefficients(self):
        """Compute coefficients of the Taylor series at a specific point."""
        for n in range(self.degree + 1):
            nth_derivative = diff(self.func_symbolic, self.x, n)
            coeff = nth_derivative.subs(self.x, self.point) / factorial(n)
            self.coefficients.append(coeff)

    def series(self):
        """Construct the Taylor series expansion as a sympy expression."""
        return sum(coeff * (self.x - self.point) ** n for n, coeff in enumerate(self.coefficients))

    def evaluate_series(self, x_values):
        """Evaluate the Taylor series at given numerical x_values."""
        series_func = lambdify(self.x, self.series(), modules=["numpy"])
        return series_func(x_values)

    def __str__(self):
        """String representation for displaying the Taylor series equation."""
        return f"Taylor series expansion at x={self.point}: {self.series()}"

class PadeApproximation:
    """
    A class to compute and evaluate the Pade approximation of a given function.

    Attributes:
        x (Symbol): The symbolic representation of the variable.
        function (Expr): The symbolic representation of the function.
        order_m (int): The order of the numerator polynomial in the Pade approximation.
        order_n (int): The order of the denominator polynomial in the Pade approximation.
        point (float): The point around which the Pade approximation is computed.
        numerator (poly1d): The numerator polynomial of the Pade approximation.
        denominator (poly1d): The denominator polynomial of the Pade approximation.

    Methods:
        compute_coefficients(): Computes the coefficients for the Pade approximation.
        evaluate(x_values): Evaluates the Pade approximation at given numerical x_values.
        __str__(): Returns a string representation of the Pade approximation equation.
    """

    def __init__(self, function, order_m, order_n, point=0):
        self.x = symbols('x')
        self.function = function(self.x)
        self.order_m = order_m
        self.order_n = order_n
        self.point = point
        self.numerator = None
        self.denominator = None

    def compute_coefficients(self):
        """
        Compute the coefficients for the Pade approximation by first expanding the function into a Taylor series
        around the specified point and then using these coefficients to compute the numerator and denominator
        polynomials of the Pade approximation.
        """
        taylor_series = series(self.function, self.x, n=self.order_m + self.order_n + 1).removeO()
        taylor_coeffs = [float(taylor_series.coeff(self.x, i).evalf()) for i in range(self.order_m + self.order_n + 1)]
        self.numerator, self.denominator = pade(taylor_coeffs, self.order_n)

    def evaluate(self, x_values):
        """
        Use the poly1d objects directly to evaluate at given x values.
        """
        numerator_values = self.numerator(x_values)
        denominator_values = self.denominator(x_values)
        return numerator_values / denominator_values

    def __str__(self):
        return f"Pade Approximation (M={self.order_m}, N={self.order_n}) at x={self.point}: {self.numerator} / {self.denominator}"
