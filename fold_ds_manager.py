import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection


class FoldDSManager:
    def __init__(self, csv, folds=10, x=None, y=None):
        torch.manual_seed(0)
        df = pd.read_csv(csv)
        self.x = x
        self.y = y
        if y is None:
            self.y = "som"
        if x is None:
            self.x = list(df.columns)
            self.x.remove(self.y)
        self.folds = folds
        x_columns = []
        for a_col in self.x:
            if a_col.startswith("B"):
                x_columns = x_columns + [col for col in df.columns if a_col in col]
            else:
                x_columns = x_columns + [a_col]
        self.x = x_columns
        columns = self.x + [self.y]
        print("Input")
        print(self.x)
        df = df[columns]
        df = df.sample(frac=1)
        self.full_data = df.to_numpy()

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
            test_data = self.full_data[test_index]
            train_x = train_data[:, :-1]
            train_y = train_data[:, -1]
            test_x = test_data[:, :-1]
            test_y = test_data[:, -1]
            validation_x = validation_data[:, :-1]
            validation_y = validation_data[:, -1]

            yield train_x, train_y, test_x, test_y, validation_x, validation_y

    def get_folds(self):
        return self.folds

