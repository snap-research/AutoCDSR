import os
from typing import List

import torch
import wandb
from lightning import LightningModule

from src.components.min_norm_solvers import MinNormSolver, gradient_normalizers


def default_optimization_loop(module: LightningModule, loss):
    opt = module.optimizers()
    opt.zero_grad()
    module.manual_backward(loss)
    opt.step()


def linear_scalar_optimization_loop(
    self: LightningModule, loss: torch.Tensor, attn_weight: float
):
    loss += (
        self.attention_score * attn_weight - (self.attention_score * attn_weight).item()
    )
    return loss


def pareto_optimization_loop_(self: LightningModule, loss: torch.Tensor, ignore_table: bool = True):
    pref_vec = torch.tensor([[0.0000, 1.0000], [1.0000, 0.0000], [0.7071, 0.7071]])

    domain_encoder_grad = []
    domain_encoder_grad_members = []
    self.attention_score.backward(retain_graph=True)
    for i, param in self.named_parameters():
        if i == "embedding_table.weight" and ignore_table:
            continue
        if param.grad is not None:
            domain_encoder_grad.append(param.grad.data.clone().flatten())
            domain_encoder_grad_members.append(i)
    self.optimizers().zero_grad()
    domain_encoder_grad = torch.cat(domain_encoder_grad)
    regular_grad = []
    loss.backward(retain_graph=True)
    for i, param in self.named_parameters():
        if i == "embedding_table.weight" and ignore_table:
            continue
        if param.grad is not None and i in domain_encoder_grad_members:
            regular_grad.append(param.grad.data.clone().flatten())
    self.optimizers().zero_grad()
    regular_grad = torch.cat(regular_grad)
    grads = torch.stack([domain_encoder_grad, regular_grad])
    losses_vec = torch.tensor([self.attention_score.item(), loss.item()]).to(
        grads.device
    )

    gn = gradient_normalizers(grads, losses_vec.reshape(-1, 1), "l2")
    for i in range(len(losses_vec)):
        grads[i] = grads[i] / gn[i]

    try:
        weight_vec = get_d_paretomtl(grads, losses_vec, pref_vec.to(grads.device), 1)
        normalize_coeff = 1 / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff
    except:
        weight_vec = [1.0, 1.0]
    loss = loss + self.attention_score * weight_vec[0] / weight_vec[1]
    return loss


def get_d_paretomtl(grads, loss_value, pref_vec, i):
    """calculate the gradient direction for ParetoMTL"""

    # check active constraints
    current_pref_vec = pref_vec[i]
    rest_pref_vecs = pref_vec
    w = rest_pref_vecs - current_pref_vec
    gx = torch.matmul(w, loss_value / torch.norm(loss_value))
    constrain_mask = gx > 0

    # calculate the descent direction
    if torch.sum(constrain_mask) == 0:
        sol, _ = MinNormSolver.find_min_norm_element_FW(
            [[grads[t]] for t in range(len(grads))]
        )
        return torch.tensor(sol).float().to(grads.device)

    vec = torch.cat((grads, torch.matmul(w[constrain_mask], grads)))
    sol, _ = MinNormSolver.find_min_norm_element_FW([[vec[t]] for t in range(len(vec))])
    weight = torch.tensor(sol[: len(loss_value)]).to(grads.device)
    counter = 0
    for idx, mask in enumerate(constrain_mask):
        if mask:
            weight += torch.mul(sol[len(loss_value) + counter], w[idx])
            counter += 1

    return weight.float()
