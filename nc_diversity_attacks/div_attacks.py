import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import faulthandler
faulthandler.enable()

# provides a nice UI element when running in a notebook, otherwise use "import tqdm" only
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from utils import *

# ============================================================================================== #
# ============================================================================================== #
# ============================================================================================== #
# ========================================= CW Attacks ========================================= #
# ============================================================================================== #
# ============================================================================================== #
# ============================================================================================== #

def cw_attack(model, inputs, targets, device, targeted=False, norm_type='inf',  
              epsilon=100., confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, 
              max_steps=1000, abort_early=True, box=(-1., 1.), optimizer_lr=1e-2,
              init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    ae_tol = torch.tensor(1e-4, device=device)

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.tensor(torch.zeros(batch_size), dtype=torch.float, device=device)
    upper_bounds = torch.tensor(torch.ones(batch_size) * c_range[1], dtype=torch.float, device=device)
    scale_consts = torch.tensor(torch.ones(batch_size) * c_range[0], dtype=torch.float, device=device)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(search_steps):

        print('Step', const_step)

        # the minimum norms of perturbations found during optimization
        best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)

        # the perturbed predictions made by the adversarial perturbations with the smallest norms
        best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(1e10, device=device)

        # optimization steps
        for optim_step in range(max_steps):

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)
            
            if norm_type == 'inf':
                inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
                norms = inf_norms
            elif norm_type == 'l2':
                l2_norms = torch.pow(adversaries - inputs, exponent=2)
                l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
                norms = l2_norms
            else:
                raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 
                
            target_activ = torch.sum(targets_oh * pert_outputs, 1)
            maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

            if targeted:           
                # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
                f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            else:
                # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
                f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

            # the total loss of current batch, should be of dimension [1]
            cw_loss = torch.sum(scale_consts * f)
            norm_loss = torch.sum(norms)
            batch_loss = cw_loss + norm_loss

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {}'.format(optim_step, batch_loss, cw_loss, norm_loss))
                
                # print(o_best_norm)
            if abort_early and not optim_step % (max_steps // 10):   
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                if batch_loss == 0:
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization
            pert_predictions = torch.argmax(pert_outputs, dim=1)
            comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
            for i in range(batch_size):
                norm = norms[i]
                cppred = comp_pert_predictions[i]
                ppred = pert_predictions[i]
                tlabel = targets[i]
                ax = adversaries[i] 
                if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                    assert cppred == ppred
                    if norm < best_norm[i]:
                        best_norm[i] = norm
                        best_norm_ppred[i] = ppred
                    if norm < o_best_norm[i]:
                        o_best_norm[i] = norm
                        o_best_norm_ppred[i] = ppred
                        o_best_adversaries[i] = ax

        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets[i]
            if best_norm_ppred[i] != -1:
                # successful: attempt to lower `scale_const` by halving it
                if scale_consts[i] < upper_bounds[i]:
                    upper_bounds[i] = scale_consts[i]
                # `upper_bounds[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds[i] has never been updated by
                # scale_consts[i] until `scale_consts[i] > 0.1 * c_range[1]`
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
            else:
                # failure: multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_consts[i] > lower_bounds[i]:
                    lower_bounds[i] = scale_consts[i]
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                else:
                    scale_consts[i] *= 10
                    
    return o_best_adversaries

def cw_div1_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100.,
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.tensor(torch.zeros(batch_size), dtype=torch.float, device=device)
    upper_bounds = torch.tensor(torch.ones(batch_size) * c_range[1], dtype=torch.float, device=device)
    scale_consts = torch.tensor(torch.ones(batch_size) * c_range[0], dtype=torch.float, device=device)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(search_steps):

        print('Step', const_step)

        # the minimum norms of perturbations found during optimization
        best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)

        # the perturbed predictions made by the adversarial perturbations with the smallest norms
        best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(1e10, device=device)
        ae_tol = torch.tensor(1e-4, device=device)

        # optimization steps
        for optim_step in range(max_steps):

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)
            
            if norm_type == 'inf':
                inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
                norms = inf_norms
            elif norm_type == 'l2':
                l2_norms = torch.pow(adversaries - inputs, exponent=2)
                l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
                norms = l2_norms
            else:
                raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 

            # calculate kl divergence for each input to use for adversary selection
            divs = []
            for i in range(batch_size):
                divs.append(norm_divergence_by_module(data=adversaries[i].unsqueeze(0), model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)) 
            div_norms = torch.tensor(torch.stack(divs), device=device)
            
            # calculate kl divergence for batch to use in loss function
            div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

            target_activ = torch.sum(targets_oh * pert_outputs, 1)
            maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

            if targeted:           
                # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
                f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            else:
                # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
                f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

            # the total loss of current batch, should be of dimension [1]
            cw_loss = torch.sum(scale_consts * f)
            norm_loss = torch.sum(norms)
            
            batch_loss = cw_loss + norm_loss + div_reg

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, cw_loss, norm_loss, div_reg))
                # print(o_best_norm)

            if abort_early and not optim_step % (max_steps // 10):   
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                if batch_loss == 0:
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization
            pert_predictions = torch.argmax(pert_outputs, dim=1)
            comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
            for i in range(batch_size):
                norm = norms[i]
                cppred = comp_pert_predictions[i]
                ppred = pert_predictions[i]
                tlabel = targets[i]
                ax = adversaries[i]
                if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                    assert cppred == ppred
                    if norm < best_norm[i]:
                        best_norm[i] = norm
                        best_norm_ppred[i] = ppred
                    if norm < o_best_norm[i]:
                        o_best_norm[i] = norm
                        o_best_norm_ppred[i] = ppred
                        o_best_adversaries[i] = ax

        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets[i]
            if best_norm_ppred[i] != -1:
                # successful: attempt to lower `scale_const` by halving it
                if scale_consts[i] < upper_bounds[i]:
                    upper_bounds[i] = scale_consts[i]
                # `upper_bounds[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds[i] has never been updated by
                # scale_consts[i] until `scale_consts[i] > 0.1 * c_range[1]`
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
            else:
                # failure: multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_consts[i] > lower_bounds[i]:
                    lower_bounds[i] = scale_consts[i]
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                else:
                    scale_consts[i] *= 10
                    
    return o_best_adversaries #, norms

def cw_div2_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100.,
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    # previous (summed) batch loss, to be used in early stopping policy
    prev_batch_loss = torch.tensor(1e10, device=device)
    ae_tol = torch.tensor(1e-4, device=device)

    # optimization steps
    for optim_step in range(max_steps):

        adversaries = from_tanh_space(inputs_tanh + pert_tanh)
        pert_outputs = model(adversaries)
        
        if norm_type == 'inf':
            inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
            norms = inf_norms
        elif norm_type == 'l2':
            l2_norms = torch.pow(adversaries - inputs, exponent=2)
            l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
            norms = l2_norms
        else:
            raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 

        # calculate kl divergence for each input to use for adversary selection
        divs = []
        for i in range(batch_size):
            divs.append(norm_divergence_by_module(data=adversaries[i].unsqueeze(0), model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)) 
        div_norms = torch.tensor(torch.stack(divs), device=device)

        # calculate kl divergence for batch to use in loss function
        div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

        target_activ = torch.sum(targets_oh * pert_outputs, 1)
        maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

        if targeted:           
            # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
            f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
        else:
            # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
            f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

        # the total loss of current batch, should be of dimension [1]
        cw_loss = torch.sum(f)
        norm_loss = torch.sum(norms)
        
        batch_loss = cw_loss + norm_loss + div_reg

        # Do optimization for one step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # "returns" batch_loss, pert_norms, pert_outputs, adversaries

        if optim_step % log_frequency == 0: 
            print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, cw_loss, norm_loss, div_reg))
            # print(o_best_norm)

        if abort_early and not optim_step % (max_steps // 10):   
            if batch_loss > prev_batch_loss * (1 - ae_tol):
                break
            if batch_loss == 0:
                break
            prev_batch_loss = batch_loss

        # update best attack found during optimization
        pert_predictions = torch.argmax(pert_outputs, dim=1)
        comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
        for i in range(batch_size):
            norm = norms[i]
            cppred = comp_pert_predictions[i]
            ppred = pert_predictions[i]
            tlabel = targets[i]
            ax = adversaries[i]
            if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                assert cppred == ppred
                if norm < o_best_norm[i]:
                    o_best_norm[i] = norm
                    o_best_norm_ppred[i] = ppred
                    o_best_adversaries[i] = ax
                    
    return o_best_adversaries #, norms

def cw_div3_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100., 
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    batch_size = inputs.size(0)
    num_classes = model(torch.tensor(inputs[0][None,:], requires_grad=False)).size(1)

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.tensor(torch.ones(batch_size) * 1e10, dtype=torch.float, device=device)
    o_best_norm_ppred = torch.tensor(-torch.ones(batch_size), dtype=torch.float, device=device)
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    # previous (summed) batch loss, to be used in early stopping policy
    prev_batch_loss = torch.tensor(1e10, device=device)
    ae_tol = torch.tensor(1e-4, device=device)

    # optimization steps
    for optim_step in range(max_steps):

        adversaries = from_tanh_space(inputs_tanh + pert_tanh)
        pert_outputs = model(adversaries)
        
        if norm_type == 'inf':
            inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
            norms = inf_norms
        elif norm_type == 'l2':
            l2_norms = torch.pow(adversaries - inputs, exponent=2)
            l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
            norms = l2_norms
        else:
            raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 

        # calculate kl divergence for each input to use for adversary selection
        divs = []
        for i in range(batch_size):
            divs.append(norm_divergence_by_module(data=adversaries[i].unsqueeze(0), model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)) 
        div_norms = torch.tensor(torch.stack(divs), device=device)

        # calculate kl divergence for batch to use in loss function
        div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

        loss = -1. * nn.CrossEntropyLoss()(pert_outputs, targets)

        # the total loss of current batch, should be of dimension [1]
        ce_loss = torch.sum(loss)
        norm_loss = torch.sum(norms)
        
        batch_loss = ce_loss + norm_loss + div_reg

        # Do optimization for one step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # "returns" batch_loss, pert_norms, pert_outputs, adversaries

        if optim_step % log_frequency == 0: 
            print('batch [{}] batch_loss: {} ce_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, ce_loss, norm_loss, div_reg))
            # print(o_best_norm)

        if abort_early and not optim_step % (max_steps // 10):   
            if batch_loss > prev_batch_loss * (1 - ae_tol):
                break
            if batch_loss == 0:
                break
            prev_batch_loss = batch_loss

        # update best attack found during optimization
        pert_predictions = torch.argmax(pert_outputs, dim=1)
        comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)
        for i in range(batch_size):
            norm = norms[i]
            cppred = comp_pert_predictions[i]
            ppred = pert_predictions[i]
            tlabel = targets[i]
            ax = adversaries[i]
            if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                assert cppred == ppred
                if norm < o_best_norm[i]:
                    o_best_norm[i] = norm
                    o_best_norm_ppred[i] = ppred
                    o_best_adversaries[i] = ax
                    
    return o_best_adversaries #, norms

def cw_div4_attack(model, modules, regularizer_weight, inputs, targets, device, targeted=False, norm_type='inf', epsilon=100., 
                   confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                   abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                   init_rand=False, log_frequency=10):

    inputs = inputs.to(device)
    targets = targets.to(device)
    model.to(device)

    batch_size = inputs.size(0)
    with torch.no_grad():
        num_classes = model(inputs[0].unsqueeze(0)).size(1)

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.zeros(batch_size).to(device) 
    upper_bounds = torch.ones(batch_size).to(device) * c_range[1]
    scale_consts = torch.ones(batch_size).to(device) * c_range[0]

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.ones(batch_size).to(device) * 1e10
    o_best_norm_ppred = torch.ones(batch_size).to(device) * -1.
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)
    targets_oh = F.one_hot(targets).float()

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(search_steps):

        print('Step', const_step)

        # # the minimum norms of perturbations found during optimization
        # best_norm = torch.ones(batch_size).to(device) * 1e10

        # # the perturbed predictions made by the adversarial perturbations with the smallest norms
        # best_norm_ppred = torch.ones(batch_size).to(device)  * -1.

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(1e10).to(device)
        ae_tol = torch.tensor(1e-4).to(device) # abort early tolerance

        # optimization steps
        for optim_step in range(max_steps):

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)

            if norm_type == 'inf':
                inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
                norms = inf_norms
            elif norm_type == 'l2':
                l2_norms = torch.pow(adversaries - inputs, exponent=2)
                l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
                norms = l2_norms
            else:
                raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 
            
            # calculate kl divergence for batch to use in loss function
            div_reg = 0
            if regularizer_weight > 0:
                div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)

            target_activ = torch.sum(targets_oh * pert_outputs, 1)
            maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]

            if targeted:           
                # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
                f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            else:
                # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
                f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)

            cw_loss = torch.sum(scale_consts * f)
            norm_loss = torch.sum(norms)
            
            batch_loss = cw_loss + norm_loss + div_reg

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [{}] batch_loss: {} cw_loss: {} norm_loss: {} div_reg: {}'.format(optim_step, batch_loss, cw_loss, norm_loss, div_reg))
                # print(o_best_norm)

            if abort_early and not optim_step % (max_steps // 10):
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                if batch_loss == 0:
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization
            pert_predictions = torch.argmax(pert_outputs, dim=1)
            comp_pert_predictions = torch.argmax(compensate_confidence(pert_outputs, targets, targeted, confidence), dim=1)

            for i in range(batch_size):
                norm = norms[i]
                cppred = comp_pert_predictions[i]
                ppred = pert_predictions[i]
                tlabel = targets[i]
                ax = adversaries[i]
                if attack_successful(cppred, tlabel, targeted) and norm < epsilon:
                    assert cppred == ppred
                    # if norm < best_norm[i]:
                    #     best_norm[i] = norm
                    #     best_norm_ppred[i] = ppred
                    if norm < o_best_norm[i]:
                        o_best_norm[i] = norm
                        o_best_norm_ppred[i] = ppred
                        o_best_adversaries[i] = ax

        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets[i]
            if o_best_norm_ppred[i] != -1:
            # if best_norm_ppred[i] != -1:
                # print('attack successful')
                # successful: attempt to lower `scale_const` by halving it
                if scale_consts[i] < upper_bounds[i]:
                    upper_bounds[i] = scale_consts[i]
                # `upper_bounds[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds[i] has never been updated by
                # scale_consts[i] until `scale_consts[i] > 0.1 * c_range[1]`
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
            else:
                # print('attack failed')
                # failure: multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_consts[i] > lower_bounds[i]:
                    lower_bounds[i] = scale_consts[i]
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                else:
                    scale_consts[i] *= 10  

    return o_best_adversaries 


def cw_div_reg_attack(model, modules, regularizer_weight, inputs, targets, dataset, device, targeted=False, norm_type='inf', epsilon=100., 
                      confidence=0.0, c_range=(1e-3, 1e10), search_steps=5, max_steps=1000, 
                      abort_early=True, box=(-1., 1.), optimizer_lr=1e-2, 
                      init_rand=False, log_frequency=10):

    inputs = inputs.to(device)
    targets = targets.to(device)
    model.to(device)
    
    classes = torch.unsqueeze(discretize(targets, dataset.boundaries), dim=1)
    
    batch_size = inputs.size(0)
    num_classes = dataset.num_classes
       
    orig_outputs = model(inputs)
    orig_classes = discretize(orig_outputs, dataset.boundaries)  

    # `lower_bounds`, `upper_bounds` and `scale_consts` are used
    # for binary search of each `scale_const` in the batch. The element-wise
    # inquality holds: lower_bounds < scale_consts <= upper_bounds
    lower_bounds = torch.zeros(batch_size).to(device) 
    upper_bounds = torch.ones(batch_size).to(device) * c_range[1]
    scale_consts = torch.ones(batch_size).to(device) * c_range[0]

    # Optimal attack to be found.
    # The three "placeholders" are defined as:
    # - `o_best_norm`        : the smallest norms encountered so far
    # - `o_best_norm_ppred`  : the perturbed predictions made by the adversarial perturbations with the smallest norms
    # - `o_best_adversaries` : the underlying adversarial example of `o_best_norm_ppred`
    o_best_norm = torch.ones(batch_size).to(device) * 1e10
    o_best_norm_ppred = torch.ones(batch_size).to(device) * -1.
    o_best_adversaries = inputs.clone()

    # convert `inputs` to tanh-space
    inputs_tanh = to_tanh_space(inputs)

    # the perturbation tensor (only one we need to track gradients on)
    pert_tanh = torch.zeros(inputs.size(), device=device, requires_grad=True)

    optimizer = optim.Adam([pert_tanh], lr=optimizer_lr)

    for const_step in range(1, search_steps+1):

        print('Step:', const_step)
        print('Scale Consts: \n', scale_consts)

        # previous (summed) batch loss, to be used in early stopping policy
        prev_batch_loss = torch.tensor(1e10).to(device)
        ae_tol = torch.tensor(1e-4).to(device) # abort early tolerance

        # optimization steps
        for optim_step in range(max_steps):

            adversaries = from_tanh_space(inputs_tanh + pert_tanh)
            pert_outputs = model(adversaries)
            pert_classes = discretize(pert_outputs, dataset.boundaries)  
            
            # nll_loss
            
            f = torch.abs(targets - pert_outputs) + confidence
            # f = F.mse_loss(targets, pert_outputs)
            cw_loss = torch.sum(scale_consts * f)
            
            # # cw loss
            # target_activ = torch.sum(targets_oh * pert_outputs, 1)
            # maxother_activ = torch.max(((1 - targets_oh) * pert_outputs - targets_oh * 1e4), 1)[0]
            
            # if targeted:           
            #     # if targeted, optimize to make `target_activ` larger than `maxother_activ` by `confidence`
            #     f = torch.clamp(maxother_activ - target_activ + confidence, min=0.0)
            # else:
            #     # if not targeted, optimize to make `maxother_activ` larger than `target_activ` (the ground truth image labels) by `confidence`
            #     f = torch.clamp(target_activ - maxother_activ + confidence, min=0.0)
            
            # cw_loss = torch.sum(scale_consts * f)
            
            # norm loss
            if norm_type == 'inf':
                inf_norms = torch.norm(adversaries - inputs, p=float("inf"), dim=(1,2,3))
                norms = inf_norms
            elif norm_type == 'l2':
                l2_norms = torch.pow(adversaries - inputs, exponent=2)
                l2_norms = torch.sum(l2_norms.view(l2_norms.size(0), -1), 1)
                norms = l2_norms
            else:
                raise Exception('must provide a valid norm_type for epsilon distance constraint: inf, l2') 
                         
            norm_loss = torch.sum(norms)
            
            # diversity loss
            div_reg = 0
            if regularizer_weight > 0:
                div_reg = norm_divergence_by_module(data=adversaries, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)
  
            batch_loss = cw_loss + norm_loss + div_reg

            # Do optimization for one step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # "returns" batch_loss, pert_norms, pert_outputs, adversaries

            if optim_step % log_frequency == 0: 
                print('batch [%i] \t batch_loss: %.4f cw_loss: %.4f norm_loss: %.4f div_reg: %.4f' % (optim_step, batch_loss, cw_loss, norm_loss, div_reg))
                # print(o_best_norm)

            if abort_early and not optim_step % (max_steps // 10):
                if batch_loss > prev_batch_loss * (1 - ae_tol):
                    break
                if batch_loss == 0:
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization  
            for i in range(batch_size):
                norm = norms[i]
                actual_class = classes[i]
                orig_class = orig_classes[i]
                pert_class = pert_classes[i]
                advx = adversaries[i]
                if ((orig_class != pert_class or orig_class == actual_class)
                    and attack_successful(pert_class, actual_class, targeted) 
                    and norm < epsilon 
                    and norm < o_best_norm[i]):
                    o_best_norm[i] = norm
                    o_best_norm_ppred[i] = pert_class
                    o_best_adversaries[i] = advx

        # binary search of `scale_const`
        if const_step == max_steps:
            print("last step, binary search updates unnecessary...")
            break
            
        for i in range(batch_size):
            if o_best_norm_ppred[i] != -1:
            # if best_norm_ppred[i] != -1:
                # successful: attempt to lower `scale_const` by halving it
                if scale_consts[i] < upper_bounds[i]:
                    upper_bounds[i] = scale_consts[i]
                # `upper_bounds[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds[i] has never been updated by
                # scale_consts[i] until `scale_consts[i] > 0.1 * c_range[1]`
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
            else:
                # failure: multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_consts[i] > lower_bounds[i]:
                    lower_bounds[i] = scale_consts[i]
                if upper_bounds[i] < c_range[1] * 0.1:
                    scale_consts[i] = (lower_bounds[i] + upper_bounds[i]) / 2
                else:
                    scale_consts[i] *= 10  
                    
        new_adversaries = torch.where(o_best_norm_ppred != -1)[0].shape[0]
        print('total number of generated adversaries: %i' % (new_adversaries))

    return o_best_adversaries 

# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #
# ========================================= PGD Attacks ========================================= #
# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #

def pgd_attack(model, 
               modules, 
               regularizer_weight, 
               inputs, 
               targets, 
               device, 
               epsilon=None, 
               num_steps=None, 
               step_size=None,
               log_frequency=10):
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.to(device)
    X, y = Variable(inputs, requires_grad=True), Variable(targets)

    out = model(X)
    orig_err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
               
        div_reg = 0
        if regularizer_weight > 0:
            div_reg = norm_divergence_by_module(data=X_pgd, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)
        with torch.enable_grad():
            ce_loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            loss = ce_loss - div_reg
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    pgd_err = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', pgd_err)
    return orig_err, pgd_err, X_pgd

def pgd_attack_reg(model, 
                   modules, 
                   regularizer_weight, 
                   inputs, 
                   targets, 
                   device, 
                   epsilon=None, 
                   num_steps=None, 
                   step_size=None,
                   log_frequency=10):
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.to(device)
    X, y = Variable(inputs, requires_grad=True), Variable(targets)

    out = model(X).view(-1)
    orig_err = torch.nn.functional.mse_loss(out, y) # (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
               
        div_reg = 0
        if regularizer_weight > 0:
            div_reg = norm_divergence_by_module(data=X_pgd, model=model, modules=modules, device=device, regularizer_weight=regularizer_weight)
        with torch.enable_grad():
            mse_loss = torch.nn.functional.mse_loss(model(X_pgd).view(-1), y) # nn.CrossEntropyLoss()(model(X_pgd), y)
            loss = mse_loss - div_reg
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    pgd_err = torch.nn.functional.mse_loss(model(X_pgd).view(-1), y) # (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', pgd_err)
    return orig_err, pgd_err, X_pgd

# ==================================================================================================== #
# ==================================================================================================== #
# ==================================================================================================== #
# ========================================= Helper Functions ========================================= #
# ==================================================================================================== #
# ==================================================================================================== #
# ==================================================================================================== #

def atanh(x, eps=1e-2):
    """
    The inverse hyperbolic tangent function, missing in pytorch.

    :param x: a tensor or a Variable
    :param eps: used to enhance numeric stability
    :return: :math:`\\tanh^{-1}{x}`, of the same type as ``x``
    """
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def to_tanh_space(x, box=(-1., 1.)):
    """
    Convert a batch of tensors to tanh-space. This method complements the
    implementation of the change-of-variable trick in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in tanh-space, of the same dimension;
             the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return atanh((x - _box_plus) / _box_mul)

def from_tanh_space(x, box=(-1., 1.)):
    """
    Convert a batch of tensors from tanh-space to oridinary image space.
    This method complements the implementation of the change-of-variable trick
    in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in ordinary image space, of the same
             dimension; the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus
  
def compensate_confidence(outputs, targets, targeted, confidence):
    """
    Compensate for ``self.confidence`` and returns a new weighted sum
    vector.

    :param outputs: the weighted sum right before the last layer softmax
           normalization, of dimension [B x M]
    :type outputs: np.ndarray
    :param targets: either the attack targets or the real image labels,
           depending on whether or not ``self.targeted``, of dimension [B]
    :type targets: np.ndarray
    :return: the compensated weighted sum of dimension [B x M]
    :rtype: np.ndarray
    """
    outputs_comp = outputs.clone()
    rng = torch.arange(start=0, end=targets.shape[0], dtype=torch.long)
    # targets = targets.int()
    if targeted:
        # for each image $i$:
        # if targeted, `outputs[i, target]` should be larger than
        # `max(outputs[i, ~target])` by `self.confidence`
        outputs_comp[rng, targets] -= confidence
    else:
        # for each image $i$:
        # if not targeted, `max(outputs[i, ~target])` should be larger
        # than `outputs[i, target]` (the ground truth image labels)
        # by `self.confidence`
        outputs_comp[rng, targets] += confidence
    return outputs_comp
  
def attack_successful(prediction, target, targeted):
    """
    See whether the underlying attack is successful.
    """
    if targeted:
        return prediction == target
    else:
        return prediction != target