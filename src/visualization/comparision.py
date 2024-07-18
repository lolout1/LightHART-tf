import matplotlib.pyplot as plt

# Data from the table
methods = ['STConvT', 'IT', 'IT(KD)']
accuracies = [99.75, 77, 78.5]

# Plotting the bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(methods, accuracies, color=['blue', 'green', 'red'])

# Adding labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, yval, ha='center', va='bottom', fontweight='bold', fontname='Times New Roman', fontsize=12)

plt.xlabel('Methods', fontname='Times New Roman', fontweight='bold', fontsize=16)
plt.ylabel('Accuracy', fontname='Times New Roman', fontweight='bold', fontsize=16)
plt.title('Accuracy Comparison', fontname='Times New Roman', fontweight='bold', fontsize=18)
plt.ylim(0, 110)  # Adding some space above the highest bar for better visualization
plt.xticks(ticks=range(len(methods)), labels=methods, fontname='Times New Roman', fontweight='bold', fontsize=14)

# Save the figure
plt.savefig('sm_accuracy_comparison.png', dpi = 300)
# Display the plot
