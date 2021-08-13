import sys
import pandas as pd


class Dataset:
    def __init__(self, path):
        self.path = path
        try:
            self.df = pd.read_csv(path)
        except FileNotFoundError:
            print("Такого файла нет")
            exit()
        # self.df.drop(columns=['Index', 'Birthday'], inplace=True)
        # self.df.dropna(axis=1, how="all", inplace=True)
        # self.df.dropna(inplace=True)
        self.df.columns = self.df.columns.str.lower()
        self.df.columns = self.df.columns.str.replace(' ', '_')


    @property
    def df_numeric(self):
        return self.df.loc[:, 'arithmancy':'flying']


if __name__ == "__main__":
    dataset = Dataset(sys.argv[1])
    print(dataset.df)
