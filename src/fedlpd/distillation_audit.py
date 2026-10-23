import torch

class DistillationAudit:

    def __init__(self, actor, confidence):
        self.parameters = list(actor.parameters())
        self.before = torch.nn.utils.parameters_to_vector(self.parameters).detach().clone()
        self.confidence = confidence
        self.gradients = []
        self.kl_before = None
        self.loss_before = None

    @torch.no_grad()
    def observe_backward(self, kl, loss):
        if self.kl_before is None:
            self.kl_before = float(kl.mean())
            self.loss_before = float(loss)
        terms = [p.grad.detach().square().sum() for p in self.parameters if p.grad is not None]
        self.gradients.append(torch.stack(terms).sum().sqrt() if terms else self.before.new_tensor(0.0))

    @torch.no_grad()
    def finish(self, kl_after=None, loss_after=None):
        after = torch.nn.utils.parameters_to_vector(self.parameters).detach()
        delta = torch.linalg.vector_norm(after - self.before)
        norm = torch.linalg.vector_norm(self.before)
        grads = torch.stack(self.gradients) if self.gradients else self.before.new_zeros(1)
        c = self.confidence
        return dict(selected_probes=len(c), active_probes=int((c > 0).sum()), mean_confidence=float(c.mean()) if len(c) else 0.0, confidence_sum=float(c.sum()), kl_before=self.kl_before, kl_after=float(kl_after.mean()) if kl_after is not None else None, weighted_loss_before=self.loss_before, weighted_loss_after=float(loss_after) if loss_after is not None else None, gradient_l2_mean=float(grads.mean()), gradient_l2_max=float(grads.max()), actor_parameter_delta_l2=float(delta), actor_parameter_l2_before=float(norm), actor_parameter_relative_delta=float(delta / norm) if norm > 0 else None, measured_steps=len(self.gradients), interpretation='Same selected probes before/after actual actor updates. KL is not environment reward; no extra backward or optimizer step.')
