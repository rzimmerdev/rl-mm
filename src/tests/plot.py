import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Read numbers from a text file
with open("rewards.txt", "r") as file:
    numbers = [float(line.strip()) for line in file if line.strip()]

# ewm smoothing
smoothing = 0.99
values = np.array(numbers)
# use ewm
# rewards = metrics['Reward'].ewm(alpha=1 - smoothing).mean()[100:]
values = pd.Series(values).ewm(alpha=1 - smoothing).mean()


# Plot the numbers
plt.plot(values, marker='o', linestyle='-')
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Plot of Numbers from File")
plt.grid()
plt.show()