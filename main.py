# Standard Library Imports
from collections import Counter

# Type Hinting Imports
from typing import Callable, Dict, List, Tuple, Union

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import integrate

# Counting methods
METHOD_MANUAL = 0
METHOD_COUNTER_CLASS = 1

# Empirical CDF Probability methods
METHOD_AUTO_CDF = 1
METHOD_RAND = 2

def round_to_decimals(value: float, decimal_places: int = 0) -> float:
    """
    Custom round function to round a number to the specified number of decimal places.

    This function handles both positive and negative numbers correctly, ensuring
    that numbers are rounded to the nearest value, providing consistent results for all inputs.

    Parameters:
    value (float): The number to be rounded.
    decimal_places (int): The number of decimal places to round to (default is 0).

    Returns:
    float: The rounded number.
    """
    multiplier = 10 ** decimal_places
    if value >= 0:
        return (value * multiplier + 0.5) // 1 / multiplier
    else:
        return -((-value * multiplier + 0.5) // 1) / multiplier

def display_results(value_counts: Dict[float, int]) -> None:
    """
    Display the frequency and percentage of values in a user-friendly format.

    Parameters:
    value_counts (Dict[float, int]): A dictionary where keys are value bins and values are frequencies.
    total_count (int): The total number of samples (default is 1,000,000).

    The function prints:
    - Value ranges (with special handling for 0 and 1).
    - The frequency of occurrences for each bin.
    - The percentage of each bin's occurrences relative to the total count.
    """
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

def create_value_bins(x_range:Tuple[Union[int],Union[int]] = (0,1001)) -> List[float]:
    return np.linspace(x_range[0],x_range[1],1000).tolist()
    # return [i / 1000.0 for i in range(x_range[0],x_range[1])]

def initialize_value_count_dict(x_range:Tuple[Union[int],Union[int]] = (0,1)) -> Dict[float, int]:
    """
    Initialize a dictionary for counting occurrences of each bin value.

    Returns:
    Dict[float, int]: A dictionary with bin values as keys and 0 as initial counts.
    """
    values = np.linspace(x_range[0],x_range[1],1000).tolist()
    return {round_to_decimals(value, 3):0 for value in values}

def count_values_by_method(
        value_bins: List[float],
        random_floats: List[float],
        method: int,
        x_range:Tuple[Union[int],Union[int]] = (0,1)
) -> Dict[float, int]:
    """
    Count occurrences of rounded random floats in the specified value bins using a specified method.

    Parameters:
    value_bins (List[float]): A list of bin values to count occurrences against.
    random_floats (List[float]): A list of random float values to be counted.
    method (int): The counting method to use.
                  METHOD_MANUAL (1) uses a manual O(n^2) algorithm.
                  METHOD_COUNTER_CLASS (2) uses Python's Counter class for an O(n log n) approach.

    Returns:
    Dict[float, int]: A dictionary where keys are bin values and values are the counts of occurrences.
    """
    value_count_dict = initialize_value_count_dict(x_range)
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


def save_pdf_and_cdf_plot_from_pdf(
        value_counts: Dict[float, int],
        display: bool,
        filename: str = "pdf_and_cdf_plot.png",
        show_histogram: bool = True,
        x_range: Tuple[int, int] = None,
):
    """
    Saves a plot with both the Probability Density Function (PDF) in two formats:
    a histogram-based PDF and a line plot, as well as the Cumulative Distribution Function (CDF).

    Parameters:
    value_counts (Dict[float, int]): A dictionary where keys are x-values and values are frequencies (PDF).
    display (bool): If True, display the plot interactively; if False, save the plot as an image.
    filename (str): The name of the file to save the plot as (default is 'pdf_and_cdf_plot.png').

    The function performs the following:
    - Plots a histogram to represent the PDF (using the frequency from value_counts).
    - Plots the PDF as a smooth line without a histogram.
    - Computes and plots the CDF based on the PDF data.
    - Displays the plot if `display` is True, or saves it to a file if `display` is False.
    """
    if x_range is None:
        x_values = list(value_counts.keys())
    else:
        x_values = create_value_bins(x_range=x_range)
    y_values = [count / 1000 for count in value_counts.values()]

    plt.figure(figsize=(12, 10))

    if show_histogram:
        # Create figure and subplots
        fig, (ax_pdf_hist, ax_pdf, ax_cdf) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[1, 1, 1])
        fig.subplots_adjust(hspace=0.4)  # Increase space between subplots
        # Plot PDF with Histogram
        sns.histplot(x_values[1:-1], bins=1000, kde=False, color='blue', alpha=0.5, ax=ax_pdf_hist)
        ax_pdf_hist.set_xlabel('X Values\n\n')
        ax_pdf_hist.set_ylabel('PDF')
        ax_pdf_hist.set_title('PDF: Probability Density Function (With 1000 Histograms)')

    else:
        fig, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])
        fig.subplots_adjust(hspace=0.4)  # Increase space between subplots

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
    """
    Creates a list of evenly spaced numbers over a specified interval.

    Parameters:
    start (float): The starting value of the sequence.
    end (float): The end value of the sequence.
    num_points (int): Number of samples to generate.

    Returns:
    List[float]: A list of evenly spaced numbers.
    """
    return np.linspace(start, end, num_points).tolist()

def create_fy_distribution_function_values() -> Dict[float, float]:
    """
    Creates a discrete representation of the FY distribution function.

    This function generates 1000 evenly spaced x-values between 0 and 7,
    and calculates the corresponding y-values (cumulative probabilities)
    based on the piecewise definition of the FY distribution function.

    Returns:
    Dict[float, float]: A dictionary mapping x-values to their corresponding
                        cumulative probabilities.
    """
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
    """
    Calculates the cumulative distribution function (CDF) for the FY distribution.

    This function defines a piecewise CDF with specific behaviors in different ranges of the input value.

    Parameters:
    y_value (float): The input value for which to calculate the CDF.

    Returns:
    float: The cumulative probability (between 0 and 1) for the given y_value.

    The function is defined piecewise as follows:
    1. For y_value <= 0.3:
       Linear increase from 0 to 0.5
    2. For 0.3 < y_value <= 0.6:
       Constant at 0.5
    3. For 0.6 < y_value <= 2:
       Linear increase from 0.5 to 1
    4. For y_value > 2:
       Returns 1
    """
    if y_value <= 0.3:
        return (0.5 / 0.3) * y_value
    elif 0.3 < y_value <= 0.6:
        return 0.5
    elif 0.6 < y_value <= 2:
        return ((0.5 / 1.4) * (y_value - 0.6)) + 0.5
    else:
        return 1

def create_fz_distribution_function_values() -> Dict[float, float]:
    """
    Creates a discrete representation of the FZ distribution function.

    This function generates 1000 evenly spaced x-values between 0 and 7,
    and calculates the corresponding y-values (cumulative probabilities)
    based on the piecewise definition of the FZ distribution function.

    Returns:
    Dict[float, float]: A dictionary mapping x-values to their corresponding
                        cumulative probabilities.
    """
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
    """
    Calculates the cumulative distribution function (CDF) for a custom distribution.

    This function defines a piecewise CDF with specific behaviors in different ranges of the input value.

    Parameters:
    z_value (float): The input value for which to calculate the CDF.

    Returns:
    float: The cumulative probability (between 0 and 1) for the given z_value.

    The function is defined piecewise as follows:
    1. For z_value < -0.5:
       Returns 0
    2. For -0.5 < z_value <= 0.2:
       Linear increase from 0 to 0.2
    3. For 0.2 < z_value <= 0.5:
       Constant at 0.2
    4. For 0.5 < z_value <= 1:
       Linear increase from 0.2 to 0.4
    5. For 1 < z_value <= 5:
       Constant at 0.4
    6. For 5 < z_value <= 7:
       Linear increase from 0.4 to 1
    7. For z_value > 7:
       Returns 1
    """
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
        x_range: Tuple[Union[float, int], Union[float, int]] = (0, 7),
        y_range: Tuple[Union[float, int], Union[float, int]] = (0, 2.5),
        cdf_title: str = 'Cumulative Distribution Function (CDF)',
        cdf_y_label: str = 'Cumulative Probability') -> None:
    """
    Saves a plot of both the Cumulative Distribution Function (CDF) and
    optionally the Probability Density Function (PDF).

    Parameters:
    cdf_function (Dict[float, float]): A dictionary mapping x values to F(x) (CDF values).
    display (bool): If True, display the plot. If False, just save it.
    filename (str): The name of the file to save the plot as.
    calculate_pdf (bool): If True, calculate and plot the PDF. Default is True.
    x_range (Tuple[Union[float, int], Union[float, int]]): The range of x-axis to display. Default is (0, 7).
    y_range (Tuple[Union[float, int], Union[float, int]]): The range of y-axis to display. Default is (0, 2.5).
    cdf_title (str): The title for the CDF plot. Default is 'Cumulative Distribution Function (CDF)'.
    cdf_y_label (str): The y-axis label for the CDF plot. Default is 'Cumulative Probability'.

    Returns:
    None. The function saves the plot to a file and optionally displays it.
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
    """
    Calculates the inverse of a custom distribution function.

    This function defines a piecewise distribution with specific behaviors
    in different ranges of the input value.

    Parameters:
    fy_value (float): The input value, typically between 0 and 1.

    Returns:
    float: The output of the inverse distribution function.

    The function is defined piecewise as follows:
    1. For fy_value < 0.5:
       Linear interpolation from 0 to 0.3
    2. For fy_value == 0.5:
       Returns a random value between 0.3 and 0.6
    3. For 0.5 < fy_value < 1:
       Linear interpolation from 0.6 to 2
    4. For fy_value >= 1:
       Returns a random value between 2 and 1000
    """
    if fy_value < 0.5:
        return fy_value/(0.5 / 0.3)
    elif fy_value == 0.5:
        return np.random.uniform(0.3, 0.6)
    elif 0.5 < fy_value < 1:
        return (fy_value - 0.5) * (1.4 / 0.5) + 0.6
    elif fy_value >= 1:
        return np.random.uniform(2, 1000)

def fz_inverse_distribution_function(fz_value: float) -> float:
    """
    Calculates the inverse of a custom distribution function.

    This function defines a piecewise distribution with specific behaviors
    in different ranges of the input value.

    Parameters:
    fz_value (float): The input value, typically between 0 and 1.

    Returns:
    float: The output of the inverse distribution function.

    The function is defined piecewise as follows:
    1. For fz_value < -0.5:
       Returns 0
    2. For -0.5 <= fz_value < 0.2:
       Linear interpolation from -0.5 to 0.2
    3. For fz_value == 0.2:
       Returns a random value between -0.5 and 0.2
    4. For 0.2 < fz_value < 0.4:
       Linear interpolation from 0.5 to 1
    5. For fz_value == 0.4:
       Returns a random value between 1 and 5
    6. For 0.4 < fz_value < 1:
       Linear interpolation from 5 to 7
    7. For fz_value == 1:
       Returns a random value greater than 7 (up to 1000)
    """
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

def plot_function(func: Callable[[float], float],
                  x_range: Tuple[float, float],
                  y_range: Tuple[float, float],
                  num_points: int = 1000,
                  title: str = "Function Plot",
                  x_label: str = "X",
                  y_label: str = "Y",
                  display: bool = False,
                  file_name: str = "plot.png") -> None:
    """
    Plots a given function over a specified range.

    This function creates a plot of the given function over the specified x-range,
    with customizable labels, title, and display options.

    Parameters:
    - func (Callable[[float], float]): The function to plot. It should take a single float argument and return a float.
    - x_range (Tuple[float, float]): A tuple specifying the range of x values (min, max).
    - y_range (Tuple[float, float]): A tuple specifying the range of y values (min, max) for the plot.
    - num_points (int): The number of points to evaluate the function (default is 1000).
    - title (str): The title of the plot (default is "Function Plot").
    - x_label (str): The label for the x-axis (default is "X").
    - y_label (str): The label for the y-axis (default is "Y").
    - display (bool): If True, display the plot. If False, just save it (default is False).
    - file_name (str): The name of the file to save the plot (default is "plot.png").

    Returns:
    None. The function saves the plot to a file and optionally displays it.
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
        func: Callable[[float], float],
        display: bool = False,
        filename: str = "empirical_cdf_probabilities.png",
        method: int = METHOD_AUTO_CDF
    ) -> None:
    """
    Calculate and plot the empirical cumulative distribution function (CDF) probabilities.

    This function takes a list of random floats, applies a given function to them,
    and then calculates and plots the empirical CDF.

    Parameters:
    - random_floats (List[float]): A list of random float values.
    - func (Callable[[float], float]): A function that returns F^-1(y) values for given y.
    - display (bool): If True, display the plot. If False, just save it. Default is False.
    - filename (str): The filename to save the plot. Default is "empirical_cdf_probabilities.png".

    The function works as follows:
    1. It applies the given function 'func' to each value in 'random_floats'.
    2. It sorts these transformed values.
    3. It creates a uniform CDF (cumulative_probabilities) from 0 to 1 with 10^6 steps.
    4. It plots the sorted data against the uniform CDF.

    The resulting plot represents the empirical CDF of the transformed data.

    Note:
    - This method assumes each data point is equally likely (has equal weight),
      effectively using a uniform probability density function (PDF).
    - The CDF represents the probability that a value is less than or equal to a given value.
    """

    # Apply the function to the random floats
    sample_data = [func(value) for value in random_floats]

    # Sort the data points
    sorted_data = np.sort(sample_data)

    print(sorted_data)
    # # Create uniform CDF
    if method == 1:
        cumulative_probabilities = np.arange(0, 10 ** 6) / 10 ** 6  # CDF of uniform distribution
        # Plotting the CDF
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_data, cumulative_probabilities)
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.grid(True)
        plt.savefig(filename)
        if display:
            plt.show()
        else:
            plt.close()
    elif method == 2:
        value_bins = create_value_bins(x_range=(-1,9))
        counted_values = count_values_by_method(value_bins, sorted_data.tolist(), method=METHOD_COUNTER_CLASS, x_range=(-1,9))
        save_pdf_and_cdf_plot_from_pdf(
            counted_values,
            display=display,
            filename=filename,
            show_histogram = False,
            x_range=(-1, 9)
        )

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

    # Calculate and plot the CDF of Fy(z)
    # USING AUTO CDF ( EXPLAINED IN THE FUNCTION )
    calculate_empirical_cdf_probabilities(
        random_floats,
        fz_inverse_distribution_function,
        display=False,
        filename="empirical_cdf_probability_of_Fy(y)_AUTO.png",
        method=METHOD_AUTO_CDF
    )

    # USING RAND METHOD: WE COUNT THE OCCURRENCE OF EACH INTERVAL MANUALLY
    calculate_empirical_cdf_probabilities(
        random_floats,
        fz_inverse_distribution_function,
        display=True,
        filename="empirical_cdf_probability_of_Fy(y)_RAND.png",
        method=METHOD_RAND
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
    # USING AUTO CDF ( EXPLAINED IN THE FUNCTION )
    calculate_empirical_cdf_probabilities(
        random_floats,
        fz_inverse_distribution_function,
        display=False,
        filename="empirical_cdf_probability_of_Fz(z)AUTO.png",
        method=METHOD_AUTO_CDF
    )

    # USING RAND METHOD: WE COUNT THE OCCURRENCE OF EACH INTERVAL MANUALLY
    calculate_empirical_cdf_probabilities(
        random_floats,
        fz_inverse_distribution_function,
        display=True,
        filename="empirical_cdf_probability_of_Fz(z)_RAND.png",
        method=METHOD_RAND
    )

if __name__ == "__main__":
    main()
