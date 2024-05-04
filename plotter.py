'''
 # @ Author: Giovanni Dalmasso
 # @ Create Time: 04-05-2024 19:57:38
 # @ Modified by: Giovanni Dalmasso
 # @ Last Modified: 04-05-2024 19:57:39
 # @ Description: This module provides functions for plotting mathematical functions and their approximations using Matplotlib and Seaborn.
 '''



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Configure the style and increase the default font sizes using Seaborn
sns.set(style="whitegrid", context="talk")  # 'talk' context is good for larger fonts suitable for presentations
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 1.5,  # Make the frame bolder
    'text.usetex': True  # Use LaTeX for text rendering
})

colors=sns.color_palette("pastel")



def plot_function_and_series(original_func, taylor_func, x_range, y_range=None, title="Function vs Taylor Series Approximation", trigonometric=False):
    """
    Plot the original function and its Taylor series approximation with enhanced visual settings.

    Parameters:
    - original_func: A callable that computes the original function values.
    - taylor_func: A callable that computes the Taylor series values.
    - x_range: Tuple of (start, end) defining the range of x values.
    - y_range: Optional tuple of (start, end) defining the range of y values.
    - title: Optional title for the plot.
    - trigonometric: Set to True if the function is trigonometric and x labels should be in terms of pi.
    """
    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], 400)

    # Evaluate the functions
    y_vals = original_func(x_vals)
    taylor_vals = taylor_func(x_vals)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y_vals, label='Original Function', color=colors[0], linewidth=3)  # Pastel green
    plt.plot(x_vals, taylor_vals, label='Taylor Approximation', color=colors[1], linewidth=2)  # Pastel red
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.grid(True)

    # Set y-axis limits if specified
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    # Set x-axis labels to pi notation if trigonometric
    if trigonometric:
        ax = plt.gca()  # Get current axes
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
        ax.xaxis.set_minor_locator(MultipleLocator(base=np.pi / 4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: rf'${val / np.pi:.0g}\pi$' if val != 0 else '0'))

    plt.show()


def plot_multiple_approximations(original_func, approx_funcs, x_range, y_range=None, title="Function and Taylor Approximations", trigonometric=False):
    """
    Plot the original function and multiple Taylor series approximations.

    Parameters:
    - original_func: A callable that computes the original function values.
    - approx_funcs: List of tuples (callable, label) for each approximation.
    - x_range: Tuple of (start, end) defining the range of x values.
    - y_range: Optional tuple of (start, end) defining the range of y values.
    - title: Optional title for the plot.
    - trigonometric: Set to True if the x-axis should display in terms of pi.
    """
    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], 400)

    # Evaluate the original function
    y_vals = original_func(x_vals)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y_vals, label='Original Function', color='black', linewidth=2)  # Original function in black
    for func, label in approx_funcs:
        taylor_vals = func(x_vals)
        plt.plot(x_vals, taylor_vals, label=label, linewidth=1.5)  # Taylor approximations

    plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.grid(True)

    # Set y-axis limits if specified
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    # Format x-axis for trigonometric functions
    if trigonometric:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
        ax.xaxis.set_minor_locator(MultipleLocator(base=np.pi / 4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: rf'${val / np.pi:.0g}\pi$' if val != 0 else '0'))

    plt.show()


def plot_function_and_pade(original_func, pade_func, x_range, y_range=None, title="Function and Pade Approximation", trigonometric=False):
    """
    Plot the original function and its Pade approximation.

    Parameters:
    - original_func: A callable that computes the original function values.
    - pade_func: A callable that computes the Pade approximation values.
    - x_range: Tuple of (start, end) defining the range of x values.
    - y_range: Optional tuple of (start, end) defining the range of y values.
    - title: Optional title for the plot.
    - trigonometric: Set to True if the function is trigonometric and x labels should be in terms of pi.
    """
    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], 400)

    # Evaluate the functions
    y_vals = original_func(x_vals)
    pade_vals = pade_func(x_vals)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y_vals, label='Original Function', color=colors[0], linewidth=3)
    plt.plot(x_vals, pade_vals, label='Pade Approximation', color=colors[1], linewidth=2)
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.grid(True)

    # Set y-axis limits if specified
    if y_range is not None:
        plt.ylim(y_range)

    # Format x-axis for trigonometric functions if required
    if trigonometric:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
        ax.xaxis.set_minor_locator(MultipleLocator(base=np.pi / 4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: rf'${val / np.pi:.0g}\pi$' if val != 0 else '0'))

    plt.show()