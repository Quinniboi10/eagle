import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from collections import deque

from eagle import Learner
from eagle.metric import Metric, State, LRMetric, TrainLossMetric, ValidLossMetric

class AccuracyMetric(Metric):
    state = State.VALID
    do_reporting = True
    name = "accuracy"

    def __init__(self, length):
        self.history: deque[float] = deque(maxlen=length)

    def step(self):
        # OK so TECHNICALLY this weights the last batch a little extra but it's really just to make sure it's improving
        self.history.append((self.learner.preds.argmax(dim=1) == self.learner.y).float().mean().item())

    @property
    def value(self):
        return sum(self.history) / len(self.history) * 100

def main():
    train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    valid = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train, batch_size=128, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=128, num_workers=2)

    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.SiLU(), nn.Linear(256, 10))
    optim = torch.optim.AdamW(model.parameters())

    learner = Learner(model, optim, train_loader, valid_loader, nn.CrossEntropyLoss(),
                metrics=[LRMetric(), TrainLossMetric(), ValidLossMetric(), AccuracyMetric(len(valid_loader))]
    )

    learner.to("cpu")
    learner.param_info()
    learner.fit_one_cycle(3)

    learner.recorder.plot("train loss", "valid loss", y_label="Loss", units="epoch")

    learner.save("./artifacts/model_checkpoint")

if __name__ == "__main__":
    main()
