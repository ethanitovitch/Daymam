import matplotlib
import matplotlib.pyplot as plt

total_profit = [50, -50, 100]
total_shoots = [1,2,3]

fig, ax = plt.subplots()
ax.plot(total_shoots, total_profit )

ax.set(xlabel='Shoot #', ylabel='Profit',
       title='Current Earnings')
ax.grid()

fig.savefig("test.png")
plt.show()