from typing import List, Dict, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import integrate


# Counting methods
METHOD_MANUAL = 0
METHOD_COUNTER_CLASS = 1

def round_to_decimals(value: float, decimal_places: int = 0) -> float:
    """
    Custom round function to round a number to the specified number of decimal places.

    The built-in round function rounds to the closest even number, which may not be desirable
    in all cases. This function ensures that numbers are rounded to the nearest value,
    providing consistent results for all inputs.

    Parameters:
    n (float): The number to be rounded.
    decimals (int): The number of decimal places to round to (default is 0).

    Returns:
    float: The rounded number.
    """
    multiplier = 10 ** decimal_places
    return (value * multiplier + 0.5) // 1 / multiplier

def display_results(value_counts: Dict[float, int]) -> None:
    print(" " * 9 + f"{'Value':<13} {'Frequency':<12} {'Percentage':<10}")
    print("-" * 50)

    total_count = 10**6
    for value, count in sorted(value_counts.items()):
        percentage = (count / total_count) * 100
        if value == 0:
            lower_bound = value
            upper_bound = round_to_decimals(value + 0.0005, 4)
            print(f"{lower_bound:0.4f} <= x < {upper_bound:<10} {count:<12} {percentage:<10.2f}")
        elif value == 1:
            lower_bound = round_to_decimals(value - 0.0005, 4)
            upper_bound = value
            print(f"{lower_bound} <= x <= {upper_bound:<9} {count:<12} {percentage:<10.2f}")
        else:
            lower_bound = round_to_decimals(value - 0.0005, 4)
            upper_bound = round_to_decimals(value + 0.0005, 4)
            print(f"{lower_bound} <= x < {upper_bound:<10} {count:<12} {percentage:<10.2f}")


def generate_random_floats() -> List[float]:
    return [value for value in np.random.random( 10 ** 6 )]

def create_value_bins() -> List[float]:
    return [i / 1000.0 for i in range(1001)]

def initialize_value_count_dict() -> Dict[float, int]:
    return {i / 1000.0: 0 for i in range(1001)}

def count_values_by_method(value_bins: List[float], random_floats: List[float], method: int) -> Dict[float, int]:
    value_count_dict = initialize_value_count_dict()
    rounded_random_floats = [round_to_decimals(value, 3) for value in random_floats]
    if method == METHOD_MANUAL:  # O(n^2) Algorithm
        for bin_value in value_bins:
            for random_float in rounded_random_floats:
                if random_float == bin_value:
                    value_count_dict[bin_value] += 1
    elif method == METHOD_COUNTER_CLASS:  # O(n log n) Algorithm
        random_counts = Counter(rounded_random_floats)
        for number, count in random_counts.items():
            if number in value_count_dict:
                value_count_dict[number] += count
    return value_count_dict

def calculate_cdf_from_pdf(pdf_values: List[float], x_values: List[float]) -> np.ndarray:
    """
    Calculates the cumulative distribution function (CDF) from the probability density function (PDF).

    Parameters:
    pdf_values (List[float]): The values representing the probability density function (PDF).
    x_values (List[float]): The corresponding x values.

    Returns:
    List[float]: The calculated CDF values.
    """
    # Use cumulative trapezoidal integration to calculate the CDF
    return integrate.cumulative_trapezoid(pdf_values, x_values, dx=0.001, initial=0)


def save_pdf_and_cdf_plot_from_pdf(value_counts: Dict[float, int], display: bool, filename: str = "pdf_and_cdf_plot.png"):
    """
    Saves a distribution plot (histogram) and CDF of the random numbers.

    Parameters:
    value_counts (Dict[float, int]): A dictionary of x and y values representing the PDF.
    filename (str): The name of the file to save the plot as (default is 'pdf_and_cdf_plot.png').
    """
    x_values = list(value_counts.keys())
    y_values = [count / 1000 for count in value_counts.values()]

    plt.figure(figsize=(12, 10))

    # Create figure and subplots
    fig, (ax_pdf_hist, ax_pdf, ax_cdf) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[1, 1, 1])
    fig.subplots_adjust(hspace=0.4)  # Increase space between subplots

    # Plot PDF with Histogram
    sns.histplot(x_values[1:-1], bins=1000, kde=False, color='blue', alpha=0.5, ax=ax_pdf_hist)
    ax_pdf_hist.set_xlabel('X Values\n\n')
    ax_pdf_hist.set_ylabel('PDF')
    ax_pdf_hist.set_title('PDF: Probability Density Function (With 1000 Histograms)')

    # Plot PDF without Histogram
    ax_pdf.plot(x_values[1:-1], y_values[1:-1], color='blue')
    ax_pdf.fill_between(x_values[1:-1], y_values[1:-1], alpha=0.3)
    ax_pdf.set_xlabel('X Values\n\n')
    ax_pdf.set_ylabel('PDF')
    ax_pdf.set_title('PDF: Probability Density Function')

    # Calculate CDF
    cdf_values = calculate_cdf_from_pdf(y_values, x_values)

    ax_cdf.plot(x_values, cdf_values, color='red')
    ax_cdf.set_xlabel('X Values')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.set_title('CDF: Cumulative Distribution Function')

    # Add overall title
    fig.suptitle('Probability Density Function and Cumulative Distribution Function', fontsize=16)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    if display:
        plt.show()
    else:
        plt.close()

def create_linear_space(start: float, end: float, num_points: int) -> List[float]:
    return np.linspace(start, end, num_points).tolist()

def create_fy_distribution_function_values() -> Dict[float, float]:
    x_values = create_linear_space(0, 7, 1000)
    y_values = []
    for value in x_values:
        if value <= 0.3:
            y_values.append((0.5 / 0.3) * value)
        elif 0.3 < value <= 0.6:
            y_values.append(0.5)
        elif 0.6 < value <= 2:
            y_values.append(((0.5 / 1.4) * (value - 0.6)) + 0.5)
        else:
            y_values.append(1)
    return dict(zip(x_values, y_values))

def fy_distribution_function(y_value: float) -> float:
    if y_value <= 0.3:
        return (0.5 / 0.3) * y_value
    elif 0.3 < y_value <= 0.6:
        return 0.5
    elif 0.6 < y_value <= 2:
        return ((0.5 / 1.4) * (y_value - 0.6)) + 0.5
    else:
        return 1

def create_fz_distribution_function_values() -> Dict[float, float]:
    x_values = create_linear_space(0, 7, 1000)
    y_values = []
    for value in x_values:
        if value <= -0.5:
            y_values.append(value)
        elif -0.5 <= value <= 0.2:
            y_values.append((0.2 / 0.7) * (value + 0.5))
        elif 0.2 < value <= 0.5:
            y_values.append(0.2)
        elif 0.5 < value <= 1:
            y_values.append(((0.2 / 0.5) * (value - 0.5)) + 0.2)
        elif 1 < value <= 5:
            y_values.append(0.4)
        elif 5 < value <= 7:
            y_values.append(((0.6 / 2) * (value - 5)) + 0.4)
        else:
            y_values.append(1)
    return dict(zip(x_values, y_values))

def fz_distribution_function(z_value: float) -> float:
    if z_value < -0.5:
        return 0
    elif -0.5 < z_value <= 0.2:
        return (0.2 / 0.7) * (z_value + 0.5)
    elif 0.2 < z_value <= 0.5:
        return 0.2
    elif 0.5 < z_value <= 1:
        return ((0.2 / 0.5) * (z_value - 0.5)) + 0.2
    elif 1 < z_value <= 5:
        return 0.4
    elif 5 < z_value <= 7:
        return ((0.6 / 2) * (z_value - 5)) + 0.4
    else:
        return 1

def calculate_pdf_from_cdf(cdf_function: Dict[float, float]) -> Dict[float, float]:
    """
    Calculate the probability density function (PDF) from the cumulative distribution function (CDF).

    Parameters:
    cdf_function (Dict[float, float]): A dictionary mapping x values to F(x).

    Returns:
    Dict[float, float]: A dictionary mapping x values to their corresponding PDF values.
    """
    x_values = np.array(list(cdf_function.keys()))
    cdf_values = np.array(list(cdf_function.values()))

    # Calculate PDF as the derivative of the CDF
    pdf_values = np.gradient(cdf_values, x_values)

    return dict(zip(x_values, pdf_values))

def save_pdf_and_cdf_plot_from_cdf(
        cdf_function: Dict[float, float],
        display: bool,
        filename: str,
        calculate_pdf: bool = True,
        x_range: Tuple[float | int, float | int] = (0, 7),
        y_range: Tuple[float | int, float | int] = (0, 2.5),
        cdf_title: str = 'Cumulative Distribution Function (CDF)',
        cdf_y_label: str = 'Cumulative Probability') -> None:
    """
    Saves a plot of both the CDF and PDF.

    Parameters:
    cdf_function (Dict[float, float]): A dictionary mapping x values to F(x).
    filename (str): The name of the file to save the plot as.
    """
    ax_pdf = 0
    x_values = list(cdf_function.keys())
    cdf_values = list(cdf_function.values())
    pdf_values = list(calculate_pdf_from_cdf(cdf_function).values())

    plt.figure(figsize=(12, 10))

    # Create figure and subplots
    if calculate_pdf:
        fig, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])
        fig.subplots_adjust(hspace=0.4)  # Increase space between subplots
    else:
        fig, (ax_cdf) = plt.subplots(1, 1, figsize=(12, 12), height_ratios=[1])
        fig.subplots_adjust(hspace=0.4)  # Increase space between subplots

    # Plot CDF
    ax_cdf.plot(x_values, cdf_values, color='blue', label='CDF')
    ax_cdf.set_xlabel('Y Values')
    ax_cdf.set_ylabel(cdf_y_label)
    ax_cdf.set_title(cdf_title)
    ax_cdf.legend()
    ax_cdf.grid()

    if calculate_pdf:
        # Plot PDF
        ax_pdf.plot(x_values, pdf_values, color='red', label='PDF')
        ax_pdf.set_xlabel('Y Values')
        ax_pdf.set_ylabel('Probability Density')
        ax_pdf.set_title('Probability Density Function (PDF)')
        ax_pdf.legend()
        ax_pdf.grid()

    plt.xlim([x_range[0], x_range[1]])
    plt.ylim([y_range[0], y_range[1]])
    plt.tight_layout()
    plt.savefig(filename)
    if display:
        plt.show()
    else:
        plt.close()


def calculate_inverse_from_dict(data: Dict[float, float]) -> dict[float, float]:
    """
    Calculate the inverse of the provided (x, y) values in dictionary form
    and return as a new dictionary.

    Parameters:
    data (Dict[float, float]): A dictionary mapping x values to y values.

    Returns:
    Dict[float, float]: A dictionary mapping new x values (original y values) to a list of new y values (original x values).
    """
    x_values = list(data.keys())
    y_values = list(data.values())

    return dict(zip(y_values, x_values))

def fy_inverse_distribution_function(fy_value: float) -> float:
    if fy_value < 0.5:
        return fy_value/(0.5 / 0.3)
    elif fy_value == 0.5:
        return np.random.uniform(0.3, 0.6)
    elif 0.5 < fy_value < 1:
        return (fy_value - 0.5) * (1.4 / 0.5) + 0.6
    elif fy_value >= 1:
        return np.random.uniform(2, 1000)

def fz_inverse_distribution_function(fz_value: float) -> float:
    if fz_value < -0.5:
        return 0
    elif -0.5 <= fz_value < 0.2:
        return (0.7 / 0.2) * fz_value - 0.5
    elif fz_value == 0.2:
        return np.random.uniform(-0.5, 0.2)  # Random value in this range
    elif 0.2 < fz_value < 0.4:
        return (fz_value - 0.2) * (0.5 / 0.2) + 0.5
    elif fz_value == 0.4:
        return np.random.uniform(1, 5)  # Random value in this range
    elif 0.4 < fz_value < 1:
        return (fz_value - 0.4) * (2 / 0.6) + 5
    elif fz_value == 1:
        return np.random.uniform(7, 1000)  # Random value greater than 7

def plot_function(func, x_range: tuple, y_range: tuple, num_points: int = 1000, title: str = "Function Plot", x_label: str = "X",
                  y_label: str = "Y", display: bool = False, file_name:str = "plot.png") -> None:
    """
    Plots a given function over a specified range.

    Parameters:
    - func: The function to plot. It should take a single float argument and return a float.
    - x_range: A tuple specifying the range of x values (min, max).
    - num_points: The number of points to evaluate the function (default is 1000).
    - title: The title of the plot.
    - x_label: The label for the x-axis.
    - y_label: The label for the y-axis.
    """
    # Generate x values
    x_values = np.linspace(x_range[0], x_range[1], num_points)

    # Calculate y values
    y_values = [func(x) for x in x_values]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=f'Plot of {func.__name__}')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim([x_range[0], x_range[1]])
    plt.ylim([y_range[0], y_range[1]])
    plt.grid()
    plt.legend()
    plt.savefig(file_name)
    if display:
        plt.show()
    else:
        plt.close()

def calculate_empirical_cdf_probabilities(
        random_floats: List[float],
        func, # Function that returns F^-1(y) values with the given y
        display: bool = False,
        filename: str = "empirical_cdf_probabilities.png"
    ):

    # we extract the probability FROM THE CDF CREATED FROM PDF OF UNIFORM.
    # in cumulative_probabilities we have [1/10 ** 6, 2/ 10**6, ... ], in which indices that we have a unique data, we
    # extract that probability and then plot it.
    # When we sort our data, the first point is greater than or equal to 1/n of the data,
    # the second point is greater than or equal to 2/n of the data, and so on, until the last point which is greater
    # than or equal to n/n (all) of the data. => Fx(x) =  P( X <= x )

    # Suppose we have the following sorted data: [1, 2, 2, 3, 5]
    #
    # The cumulative probabilities would be [1/5, 2/5, 3/5, 4/5, 5/5]
    # This means:
    #
    # 1/5 (20%) of the data is ≤ 1
    # 2/5 (40%) of the data is ≤ 2
    # 3/5 (60%) of the data is ≤ 3
    # 4/5 (80%) of the data is ≤ 4
    # 5/5 (100%) of the data is ≤ 5
    # this method assumes each data point is equally likely (has equal weight). ( PDF = UNIFORM  )

    sample_data = [func(value) for value in random_floats]

    # Sort the data points
    sorted_data = np.sort(sample_data)
    cumulative_probabilities = np.arange(0, 10 ** 6) / 10 ** 6 # CDF OF UNIFORM

    # Plotting the CDF
    plt.figure(figsize=(10, 6))
    plt.step(sorted_data, cumulative_probabilities, where='post')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(True)
    plt.savefig(filename)
    if display:
        plt.show()
    else:
        plt.close()

def main():
    random_floats = generate_random_floats()
    value_bins = create_value_bins()
    counted_values = count_values_by_method(value_bins, random_floats, method=METHOD_COUNTER_CLASS)
    display_results(counted_values)

    save_pdf_and_cdf_plot_from_pdf(
        counted_values,
        display=False,
        filename="rand_pdf_and_cdf_plot.png"
    )

    # METHOD 1 : ( INCLUDES PDF FUNCTION CALCULATIONS )
    # Using values generated from create_fy_distribution_function_values to plot F(y) function
    fy_distribution_function_values = create_fy_distribution_function_values()
    save_pdf_and_cdf_plot_from_cdf(
        fy_distribution_function_values,
        display=False,
        filename="F(y)_PLOT.png"
    )
    # METHOD 2 : ( DOES NOT INCLUDE PDF FUNCTION CALCULATIONS )
    # Using f(y) directly
    plot_function(
        func=fy_distribution_function,
        x_range=(-0.1,7),
        y_range=(0,1.1),
        num_points=1000,
        title="F(y)",
        x_label="Y",
        y_label="F(Y) without values",
        display=False,
        file_name="F(y)_PLOT_DIRECTLY.png"
        )

    # METHOD 1 :
    # Using values generated from create_fy_distribution_function_values to plot the inverse of F(y) function
    inverse_fy_distribution_function = calculate_inverse_from_dict(fy_distribution_function_values)
    save_pdf_and_cdf_plot_from_cdf(
        inverse_fy_distribution_function,
        display=False,
        calculate_pdf=False,
        filename="F^-1(y)_PLOT.png",
        x_range=(-0.1,1.1),
        y_range=(0,2.5),
        cdf_title = "F^-1(y) plot with values",
        cdf_y_label = "F^-1(y)"
    )
    # METHOD 2 :
    # Using inverse of f(y) directly
    plot_function(
        func=fy_inverse_distribution_function,
        x_range=(-0.1,1.1),
        y_range=(0,2.5),
        num_points=1000,
        title="F^-1(y)",
        x_label="Y",
        y_label="F^-1(Y) without values",
        display=False,
        file_name="F^-1(y)_PLOT_DIRECTLY.png"
        )


    # Calculate and plot the CDF of Fy(y)
    calculate_empirical_cdf_probabilities(
        random_floats,
        fy_inverse_distribution_function,
        display=True,
        filename="empirical_cdf_probability_of_Fy(y).png"
    )

# -----------------------------------------------------------------------------------------------------
    # METHOD 1 : ( INCLUDES PDF FUNCTION CALCULATIONS )
    # Using values generated from create_fy_distribution_function_values to plot F(y) function
    fz_distribution_function_values = create_fz_distribution_function_values()
    save_pdf_and_cdf_plot_from_cdf(
        fz_distribution_function_values,
        display=False,
        filename="F(z)_PLOT.png"
    )
    # METHOD 2 : ( DOES NOT INCLUDE PDF FUNCTION CALCULATIONS )
    # Using f(z) directly
    plot_function(
        func=fz_distribution_function,
        x_range=(-0.8,7),
        y_range=(0,1.1),
        num_points=1000,
        title="F(z)",
        x_label="Z",
        y_label="F(Z) without values",
        display=False,
        file_name="F(Z)_PLOT_DIRECTLY.png"
        )

    # METHOD 1 :
    # Using values generated from create_fz_distribution_function_values to plot the inverse of F(z) function
    inverse_fz_distribution_function = calculate_inverse_from_dict(fz_distribution_function_values)
    save_pdf_and_cdf_plot_from_cdf(
        inverse_fz_distribution_function,
        display=False,
        calculate_pdf=False,
        filename="F^-1(Z)_PLOT.png",
        x_range=(-0.8,1.1),
        y_range=(0,2.5),
        cdf_title = "F^-1(Z) plot with values",
        cdf_y_label = "F^-1(Z)"
    )
    # METHOD 2 :
    # Using inverse of f(z) directly
    plot_function(
        func=fz_inverse_distribution_function,
        x_range=(-0.8,1.1),
        y_range=(0,2.5),
        num_points=1000,
        title="F^-1(z)",
        x_label="Z",
        y_label="F^-1(z) without values",
        display=False,
        file_name="F^-1(z)_PLOT_DIRECTLY.png"
        )

    # Calculate and plot the CDF of Fz(z)
    calculate_empirical_cdf_probabilities(
        random_floats,
        fz_inverse_distribution_function,
        display=True,
        filename="empirical_cdf_probability_of_Fz(z).png"
    )

if __name__ == "__main__":
    main()
