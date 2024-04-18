#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   schp.py
@Time    :   4/8/19 2:11 PM
@Desc    :   
@License :   This source code is licensed under the license found in the 
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import modules

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, modules.bn.InPlaceABNSync):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, modules.bn.InPlaceABNSync):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, modules.bn.InPlaceABNSync):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, modules.bn.InPlaceABNSync):
        module.momentum = momenta[module]


def bn_re_estimate(loader, model):
    if not check_bn(model):
        print('No batch norm layer detected')
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for i_iter, batch in enumerate(loader):
        images, labels, _ = batch
        b = images.data.size(0)
        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum
        model(images)
        n += b
    model.apply(lambda module: _set_momenta(module, momenta))


def save_schp_checkpoint(states, is_best_parsing, output_dir, filename='schp_checkpoint.pth.tar'):
    save_path = os.path.join(output_dir, filename)
    if os.path.exists(save_path):
        os.remove(save_path)
    torch.save(states, save_path)
    if is_best_parsing and 'state_dict' in states:
        best_save_path = os.path.join(output_dir, 'model_parsing_best.pth.tar')
        if os.path.exists(best_save_path):
            os.remove(best_save_path)
        torch.save(states, best_save_path)
