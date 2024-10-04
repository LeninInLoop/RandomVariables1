import random
from typing import List
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Methods to count the numbers
MANUAL = 0
COUNTER_CLASS = 1

def custom_round(n, decimals=0) -> float:
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
    multiplier = 10 ** decimals
    return (n * multiplier + 0.5) // 1 / multiplier

def print_results(separated_numbers_counted:dict[float, int]) -> None:
    print(" " * 9 + f"{'Value':<13} {'Frequency':<12} {'Percentage':<10}")
    print("-" * 50)

    total_count = 10**6
    for value, count in sorted(separated_numbers_counted.items()):
        percentage = (count / total_count) * 100
        if value == 0:
            lower_bound = value
            upper_bound = custom_round(value + 0.0005, 4)
            print(f"{lower_bound:0.4f} <= x < {upper_bound:<10} {count:<12} {percentage:<10.2f}")
        elif value == 1:
            lower_bound = custom_round(value - 0.0005, 4)
            upper_bound = value
            print(f"{lower_bound} <= x <= {upper_bound:<9} {count:<12} {percentage:<10.2f}")
        else:
            lower_bound = custom_round(value - 0.0005, 4)
            upper_bound = custom_round(value + 0.0005, 4)
            print(f"{lower_bound} <= x < {upper_bound:<10} {count:<12} {percentage:<10.2f}")

def generate_random_numbers() -> List[float]:
    return [custom_round(random.random(), 3) for _ in range(10 ** 6)]

def create_separators() -> List[float]:
    return [(_/1000.000) for _ in range(0,1001)]

def create_separated_numbers_dict() -> dict[float, int]:
    return {(_/1000.000):0 for _ in range(0,1001)}

def count_numbers_with_method(separators: List, random_numbers: List[float], method: int) -> dict[float, int]:
    separated_numbers_dict = create_separated_numbers_dict()
    if method == 0: # O(n^2) Algorithm
        for number in separators:
            for random_number in random_numbers:
                if random_number == number:
                    separated_numbers_dict[number] += 1
    elif method == 1: # O(nLog(n)) Algorithm
        random_count = Counter(random_numbers)
        for number, count in random_count.items():
            if number in separated_numbers_dict:
                separated_numbers_dict[number] += count
    return separated_numbers_dict

def save_distribution_plot(random_numbers: List[float], filename: str = "distribution_plot.png"):
    """
    Saves a distribution plot (histogram) of the random numbers.

    Parameters:
    random_numbers (List[float]): A list of random numbers to plot.
    filename (str): The name of the file to save the plot as (default is 'distribution_plot.png').
    """
    plt.figure(figsize=(12, 6))

    # Create a histogram
    sns.histplot(random_numbers, bins=100, kde=True, color='blue', alpha=0.7)

    # Add labels and title
    plt.xlabel('Random Number Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Random Numbers')

    # Save the plot to a file
    plt.savefig(filename)
    plt.show()

def main():
    random_numbers_list = generate_random_numbers()
    separators = create_separators()
    separated_numbers_counted = count_numbers_with_method(separators, random_numbers_list, method = COUNTER_CLASS)
    print_results(separated_numbers_counted)
    save_distribution_plot([_ for _ in separated_numbers_counted.keys()])

if __name__ == "__main__":
    main()
