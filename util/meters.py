class AverageMeter(object):
    """Compute and store the average, standard deviation, and current value"""

    def __init__(self, name="Meter", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sqsum = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sqsum += (val ** 2) * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = ((self.sqsum / self.count) - (self.avg * self.avg)) ** 0.5

    def __str__(self):
        fmtstr = (
            "{name} {val"
            + self.fmt
            + "} (AVG {avg"
            + self.fmt
            + "}, STD {std"
            + self.fmt
            + "})"
        )
        return fmtstr.format(**self.__dict__)

