import torch
import torch.nn as nn

import numpy as np

# provides a nice UI element when running in a notebook, otherwise use "import tqdm" only
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

def get_model_modules(model, layer_name=None):
    layer_dict = {}
    idx=0
    for name, module in model.named_children():
        if (not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.BatchNorm2d)
            and not isinstance(module, nn.Dropout)
            and not isinstance(module, nn.ReLU)
            and (layer_name is None or layer_name in name)):
            layer_dict[name + '-' + str(idx)] = module
            idx += 1
        else:
            for name_2, module_2 in module.named_children():
                for name_3, module_3 in module_2.named_children():
                    if (not isinstance(module_3, nn.Sequential)
                        and not isinstance(module_3, nn.BatchNorm2d)
                        and not isinstance(module, nn.Dropout)
                        and not isinstance(module, nn.ReLU)
                        and 'shortcut' not in name_3
                        and (layer_name is None or layer_name in name_3)):
                        layer_dict[name_3 + '-' + str(idx)] = module_3
                        idx += 1    
                        
    return layer_dict

def step_through_model(model, prefix=''):
    for name, module in model.named_children():
        path = '{}/{}'.format(prefix, name)
        if (isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)): # test for dataset
            yield (path, name, module)
        else:
            yield from step_through_model(module, path)

def get_model_layers(model, cross_section_size=0):
    layer_dict = {}
    i = 0
    for (path, name, module) in step_through_model(model):
        layer_dict[str(i) + path] = module
        i += 1
    if cross_section_size > 0:
        target_layers = list(layer_dict)[0::cross_section_size] 
        layer_dict = { target_layer: layer_dict[target_layer] for target_layer in target_layers }
    return layer_dict 

def get_layer_output_sizes(model, data, layer_name=None):   
    output_sizes = {}
    hooks = []  
    
    layer_dict = get_model_layers(model)
 
    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        output_sizes[m_key] = list(output.size()[1:])      
    
    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    
    try:
        model(data[:1])  
    finally:
        for h in hooks:
            h.remove() 
            
    return output_sizes

def get_init_dict(model, data, init_value=False, layer_name=None): 
    output_sizes = get_layer_output_sizes(model, data, layer_name)       
    model_layer_dict = {}  
    for layer, output_size in output_sizes.items():
        for index in range(np.prod(output_size)):
            # since we only care about post-activation outputs
            model_layer_dict[(layer, index)] = init_value
    return model_layer_dict

def neurons_covered(model_layer_dict, layer_name=None):
    covered_neurons = len([v for k, v in model_layer_dict.items() if v and (layer_name is None or layer_name in k[0])])
    total_neurons = len([v for k, v in model_layer_dict.items() if layer_name is None or layer_name in k[0]])
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def scale(out, rmax=1, rmin=0):
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

def extract_outputs(model, data, module, force_relu=True):
    outputs = []      
    def hook(module, input, output):
        if force_relu:
            outputs.append(torch.relu(output))   
        else:
            outputs.append(output)
    handle = module.register_forward_hook(hook)     
    model(data)
    handle.remove()
    return torch.stack(outputs)

def update_coverage(model, data, model_layer_dict, threshold=0., layer_name=None):   
    layer_dict = get_model_layers(model) 
    for layer, module in tqdm(layer_dict.items()): 
        outputs = torch.squeeze(torch.sum(extract_outputs(model, data, module), dim=1))
        scaled_outputs = scale(outputs)     
        for i, out in enumerate(scaled_outputs.view(-1)):
            if out > threshold:
                model_layer_dict[(layer, i)] = True
                
def eval_nc(model, data, threshold=0., layer_name=None):
    model_layer_dict = get_init_dict(model, data, False)
    update_coverage(model, data, model_layer_dict, threshold=threshold)
    covered_neurons, total_neurons, neuron_coverage = neurons_covered(model_layer_dict, layer_name)
    return covered_neurons, total_neurons, neuron_coverage


# def layer_iterator(module, layer_dict={}, idx=0, layer_name=None):
#     for name, module in module.named_children():
#         print(name)
#         if isinstance(module, nn.Sequential):
#             return layer_iterator(module, layer_dict, idx, layer_name)
#         else:
#             if (not isinstance(module, nn.BatchNorm2d)
#                 and 'shortcut' not in name
#                 and (layer_name is None or layer_name in name)):
#                 layer_dict[name + '-' + str(idx)] = module
#                 idx += 1  
#         if isinstance(module, BasicBlock):
#             return layer_iterator(module, layer_dict, idx, layer_name)
#     return layer_dict

# layer_iterator(model, layer_name='conv')