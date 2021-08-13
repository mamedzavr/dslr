import sys
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Dataset


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Как юзать: script.py dataset.csv")
        exit()
    dataset = Dataset(sys.argv[1])
    sns.scatterplot(data=dataset.df, x='astronomy', y='defense_against_the_dark_arts', hue='hogwarts_house', alpha=0.7)
    plt.xlabel('astronomy')
    plt.ylabel('defense_dark_arts')
    plt.show()
