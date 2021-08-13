import sys
import matplotlib.pyplot as plt
from dataset import Dataset

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Как юзать: script.py dataset.csv")
        exit()
    dataset = Dataset(sys.argv[1])

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)

    current_data = dataset.df[dataset.df["hogwarts_house"] == "Gryffindor"]
    scores = current_data.loc[:, "arithmancy":"flying"]
    x = (scores - scores.min()) / (scores.max() - scores.min())
    axs[0][0].hist(x.values.flatten(), bins=40, rwidth=0.8)
    axs[0][0].set_title("Gryffindor")

    current_data = dataset.df[dataset.df["hogwarts_house"] == "Slytherin"]
    scores = current_data.loc[:, "arithmancy":"flying"]
    x = (scores - scores.min()) / (scores.max() - scores.min())
    axs[0][1].hist(x.values.flatten(), bins=40, rwidth=0.8)
    axs[0][1].set_title("Slytherin")

    current_data = dataset.df[dataset.df["hogwarts_house"] == "Ravenclaw"]
    scores = current_data.loc[:, "arithmancy":"flying"]
    x = (scores - scores.min()) / (scores.max() - scores.min())
    axs[1][0].hist(x.values.flatten(), bins=40, rwidth=0.8)
    axs[1][0].set_title("Ravenclaw")

    current_data = dataset.df[dataset.df["hogwarts_house"] == "Hufflepuff"]
    scores = current_data.loc[:, "arithmancy":"flying"]
    x = (scores - scores.min()) / (scores.max() - scores.min())
    axs[1][1].hist(x.values.flatten(), bins=40, rwidth=0.8)
    axs[1][1].set_title("Hufflepuff")

    plt.show()
