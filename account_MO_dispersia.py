import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the log file
A = np.loadtxt('C098_MO_diss.log')  # Load the data from the log file

# Step 2: Extract Column 2 and Column 3
column2 = A[:, 1]  # Extract the second column (Python uses 0-based indexing)
column3 = A[:, 2]  # Extract the third column
column4 = A[:, 3]  # Extract the third column

print('Mean (мат ожидание) for Vi // Fi   //  kFi')
print([np.mean(column2), np.mean(column3),np.mean(column4) ])  # Display the mean of Column 2 and Column 3
print('Variance (дисперсия) for 2:Vi // Fi   //  kFi ')
print([np.var(column2), np.var(column3), np.var(column4)])  # Display the variance of Column 2 and Column 3

# print('Mean (мат ожидание) for kFi and Fi:')
# print([np.mean(column1), np.mean(column2)])  # Display the mean of Column 2 and Column 3
# print('Variance (дисперсия) for kFi and Fi:')
# print([np.var(column1), np.var(column2)])  # Display the variance of Column 2 and Column 3

# Step 5: Plot Column 2 and Column 3
plt.figure()  # Create a new figure
plt.plot(column3, label='Fi')  # Plot Column 2
plt.plot(column2, label='Vi')  # Plot Column 2
plt.plot(column4, label='kFi')  # Plot Column 2

plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Data from ui_lTr.log (Column 2 and Column 3)')

# Step 6: Add legends for Column 2 and Column 3
plt.legend()  # Add legends to the plot
plt.show()  # Display the plot
