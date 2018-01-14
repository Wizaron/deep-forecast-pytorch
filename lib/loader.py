from torch.utils.data import Dataset

class Loader(Dataset):
    """Dataset Reader"""

    def __init__(self, (X, y)):

        self.features = X
        self.targets = y

        self.n_samples = len(self.features)

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        features = self.features[index]
        target = self.targets[index]

        return features, target

    def __len__(self):
        return self.n_samples
