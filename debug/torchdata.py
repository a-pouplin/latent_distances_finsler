import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class Omniglot(Dataset):
    def __init__(self, num_data=1000, data_dir="data/"):
        super(Omniglot, self).__init__()
        # train_dataset = datasets.Omniglot(root=data_dir, background=True, download=True, transform=ToTensor())
        dataset = datasets.Omniglot(root=data_dir, background=False, download=True, transform=ToTensor())
        # dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        self.x = torch.stack([(dataset[i][0]).flatten() for i in range(len(dataset))])[:num_data]
        self.y = torch.stack([torch.tensor(dataset[i][1]) for i in range(len(dataset))])[:num_data]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self):
        return self.x, self.y

    # dataset = datasets.Omniglot(root="data", download=False, background=False, transform=ToTensor())
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=dataset.__len__())
    # images, labels = next(iter(dataloader))


class FashionMNIST(Dataset):
    """Fashion MNIST Dataset
    From Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning
    Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, data_dir="data/"):
        super(FashionMNIST, self).__init__()
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=ToTensor())
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=ToTensor())
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        self.x = torch.stack([(dataset[i][0]).flatten() for i in range(len(dataset))])
        self.y = torch.stack([torch.tensor(dataset[i][1]) for i in range(len(dataset))])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    data = Omniglot()
    x, y = data.x, data.y

    raise
