import math


class LrScheduler(object):
    def __init__(self, mode, base_lr, total_epoches, steps_per_epoch, warm_up_epoch=0):
        self.mode = mode
        self.base_lr = base_lr
        self.total_epoches = total_epoches
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epoches*steps_per_epoch
        self.warm_up_steps = warm_up_epoch*steps_per_epoch

    def __call__(self, optimizer, step_index, epoch):
        step_current = (epoch - 1)*self.steps_per_epoch + step_index
        if self.mode == "cos":
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * step_current / self.total_steps * math.pi))
        elif self.mode == "poly":
            lr = self.base_lr * pow((1 - 1.0 * step_current / self.total_steps), 0.9)
        else:
            raise NotImplemented
        
        # Warm Up Learning Rate
        if self.warm_up_steps > 0 and step_current < self.warm_up_steps:
            lr = lr * 1.0 * step_current / self.warm_up_steps

        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]["lr"] = lr
        else:
            assert False