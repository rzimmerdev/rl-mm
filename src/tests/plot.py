import matplotlib.pyplot as plt

# Read numbers from a text file
with open("rewards.txt", "r") as file:
    numbers = [float(line.strip()) for line in file if line.strip()]

# moving average
window = 10
numbers = [sum(numbers[i:i+window])/window for i in range(len(numbers)-window)]

# Plot the numbers
plt.plot(numbers, marker='o', linestyle='-')
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Plot of Numbers from File")
plt.grid()
plt.show()