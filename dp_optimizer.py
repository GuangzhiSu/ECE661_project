import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np


class DPQuery:
    """Defines the differential privacy mechanism using Gaussian noise."""
    def __init__(self, l2_norm_clip, noise_multiplier):
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier

    def initial_global_state(self):
        return {}

    def initial_sample_state(self, params):
        return [torch.zeros_like(p) for p in params]

    def derive_sample_params(self, global_state):
        return self.l2_norm_clip

    def accumulate_record(self, sample_params, sample_state, grads):
        clipped_grads = [torch.clamp(g, max=sample_params) for g in grads]
        return [s + g for s, g in zip(sample_state, clipped_grads)]

    def get_noised_result(self, sample_state, global_state):
        noised_grads = [
            s + torch.normal(mean=0, std=self.l2_norm_clip * self.noise_multiplier, size=s.size(), device=s.device)
            for s in sample_state
        ]
        return noised_grads, global_state


def make_optimizer_class(base_optimizer):
    """Wraps an existing optimizer class to create a DP-aware optimizer."""
    class DPOptimizer(base_optimizer):
        def __init__(self, params, dp_query, num_microbatches=None, *args, **kwargs):
            super(DPOptimizer, self).__init__(params, *args, **kwargs)
            self.dp_query = dp_query
            self.num_microbatches = num_microbatches
            self.global_state = self.dp_query.initial_global_state()

        def compute_grads(self, loss_fn, model, inputs, targets):
            """Computes gradients with differential privacy."""
            if self.num_microbatches is None:
                self.num_microbatches = len(inputs)

            microbatch_size = len(inputs) // self.num_microbatches
            sample_state = self.dp_query.initial_sample_state(model.parameters())

            for i in range(self.num_microbatches):
                start = i * microbatch_size
                end = start + microbatch_size
                microbatch_inputs = inputs[start:end]
                microbatch_targets = targets[start:end]

                # Compute per-microbatch loss
                microbatch_loss = loss_fn(model(microbatch_inputs), microbatch_targets)
                microbatch_loss.backward()

                # Collect and clip gradients
                grads = [p.grad.clone() for p in model.parameters()]
                sample_state = self.dp_query.accumulate_record(self.dp_query.derive_sample_params(self.global_state), sample_state, grads)

                # Zero out gradients for next microbatch
                model.zero_grad()

            # Add noise and average gradients
            noised_grads, self.global_state = self.dp_query.get_noised_result(sample_state, self.global_state)
            final_grads = [g / self.num_microbatches for g in noised_grads]

            return final_grads

        def apply_grads(self, model, final_grads):
            """Applies the sanitized gradients to the model."""
            with torch.no_grad():
                for param, grad in zip(model.parameters(), final_grads):
                    param.grad = grad
            self.step()

    return DPOptimizer


def make_gaussian_optimizer_class(base_optimizer):
    """Wraps an optimizer to include Gaussian-based differential privacy."""
    class DPGaussianOptimizer(make_optimizer_class(base_optimizer)):
        def __init__(self, params, l2_norm_clip, noise_multiplier, num_microbatches=None, *args, **kwargs):
            dp_query = DPQuery(l2_norm_clip, noise_multiplier)
            super(DPGaussianOptimizer, self).__init__(params, dp_query, num_microbatches, *args, **kwargs)

    return DPGaussianOptimizer


# Examples of DP-aware optimizers
DPAdamOptimizer = make_optimizer_class(torch.optim.Adam)
DPAdagradOptimizer = make_optimizer_class(torch.optim.Adagrad)
DPGradientDescentOptimizer = make_optimizer_class(torch.optim.SGD)

DPAdamGaussianOptimizer = make_gaussian_optimizer_class(torch.optim.Adam)
DPAdagradGaussianOptimizer = make_gaussian_optimizer_class(torch.optim.Adagrad)
DPGradientDescentGaussianOptimizer = make_gaussian_optimizer_class(torch.optim.SGD)
