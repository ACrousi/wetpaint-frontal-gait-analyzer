import numpy as np


class Step_Scheduler():
    def __init__(self, num_sample, warm_up, step_lr, gamma=0.1, **kwargs):
        warm_up_num = warm_up * num_sample
        if len(step_lr) > 0:
            self.eval_interval = lambda epoch: 1
        else:
            self.eval_interval = lambda epoch: 1
        self.lr_lambda = lambda num: num / warm_up_num \
                                     if num < warm_up_num else \
                                     gamma ** np.sum(np.array(step_lr) <= num // num_sample)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda


class Cosine_Scheduler():
    def __init__(self, num_sample, max_epoch, warm_up, eta_min=0, base_lr=1, **kwargs):
        eta_min = float(eta_min) if isinstance(eta_min, str) else eta_min
        eta_min_ratio = eta_min / base_lr if base_lr != 0 else 0
        warm_up_num = warm_up * num_sample
        max_num = max_epoch * num_sample
        # self.eval_interval = lambda epoch: 1 if (epoch+1) > max_epoch - 10 else 5
        self.eval_interval = lambda epoch: 1
        self.lr_lambda = lambda num: num / warm_up_num \
                                     if num < warm_up_num else \
                                     eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (np.cos((num - warm_up_num) / (max_num - warm_up_num) * np.pi) + 1)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda
