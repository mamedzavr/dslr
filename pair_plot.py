import sys
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Dataset

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Как юзать: script.py dataset.csv")
        exit()
    dataset = Dataset(sys.argv[1])

    figure = sns.pairplot(dataset.df, hue="hogwarts_house", height=1.0, plot_kws={'s': 3})
    for ax in figure.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation=45)
        ax.set_ylabel(ax.get_ylabel(), rotation=45)

    plt.show()
