import random


class ProbabilisticDataLoader:
    def __init__(self, dataloader1, dataloader2, prob1=0.1):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.prob1 = prob1
        self.prob2 = 1.0 - prob1

        self.iter1 = iter(dataloader1)
        self.iter2 = iter(dataloader2)

    def __iter__(self):
        self.iter1 = iter(self.dataloader1)
        self.iter2 = iter(self.dataloader2)
        return self

    def __next__(self):
        p = random.random()
        if p < self.prob1:
            return next(self.iter1)
        else:
            return next(self.iter2)
