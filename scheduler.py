import math


class NoamScheduler:
    def __init__(
        self,
        optimizer,
        d_model: int = 256,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def rate(self, step: int = None) -> float:
        if step is None:
            step = self.step_num
        step = max(step, 1)

        scale = self.d_model ** -0.5
        warmup = step * (self.warmup_steps ** -1.5)
        decay = step ** -0.5
        return self.factor * scale * min(decay, warmup)

    def step(self) -> float:
        self.step_num += 1
        lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict["step_num"]
        self.d_model = state_dict.get("d_model", self.d_model)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.factor = state_dict.get("factor", self.factor)


class FixedScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        return self.optimizer.param_groups[0]["lr"]

    def rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, state_dict):
        self.step_num = state_dict.get("step_num", 0)


def make_scheduler(optimizer, config):
    if config.get("use_noam", True):
        return NoamScheduler(
            optimizer,
            d_model=config.get("d_model", 256),
            warmup_steps=config.get("warmup_steps", 4000),
            factor=config.get("noam_factor", 1.0),
        )

    return FixedScheduler(optimizer)
