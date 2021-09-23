# CW + Diversity Regularization on CIFAR10

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import pickle
import datetime
import glob
import os

import traceback
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# custom code imports
from div_attacks import *
from neuron_coverage import *
from inception_score import *
from fid_score import *
from utils import *

# check if CUDA is available
device = torch.device("cpu")
use_cuda = False
if torch.cuda.is_available():
    print('CUDA is available!')
    device = torch.device("cuda")
    use_cuda = True
else:
    print('CUDA is not available.')

random_seed = 1
torch.manual_seed(random_seed)

date = datetime.date.today()

data_dir = "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Generate a custom batch to ensure that each class is equally represented
num_per_class = 10
dataset = torchvision.datasets.CIFAR10(root=data_dir, 
                                       train=False, 
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor()
                                       ]))

class_distribution = torch.ones(len(np.unique(dataset.targets))) * num_per_class
inputs, targets = generate_batch(dataset, class_distribution, device)

# # Loading Pretrained Models 

# ResNet56 - https://github.com/akamaster/pytorch_resnet_cifar10
from resnet import *
models_dir = 'pretrained_models/cifar10/' 
resnet56 = resnet56().to(device)
state_dict = torch.load(models_dir + 'resnet56.th', map_location=device)['state_dict'] # best_prec1, state_dict
new_state_dict = {}
for k, v in state_dict.items():
    if 'module' in k:
        k = k.replace('module.', '')
    new_state_dict[k]=v   
resnet56.load_state_dict(new_state_dict)

# DenseNet121 - https://github.com/huyvnphan/PyTorch-CIFAR10
from densenet import *
densenet121 = densenet121().to(device)
                    
# # Attack Time
def main():

    models = [resnet56, densenet121]
    num_layers = 5 # the number of layers we want to sample from the network, evenly spaced

    # attack params
    search_steps=10
    targeted=False
    norm_type='inf'
    epsilon=100.
    c_range=(1e-3, 1e10)
    max_steps=1000
    abort_early=True
    optimizer_lr=5e-4
    init_rand=False
    log_frequency = 100

    mean = (0.485, 0.456, 0.406) # the mean used in inputs normalization
    std = (0.229, 0.224, 0.225)  # the standard deviation used in inputs normalization
    box = (min((0 - m) / s for m, s in zip(mean, std)),
           max((1 - m) / s for m, s in zip(mean, std)))

    attack_versions = [cw_div4_attack] # [cw_div1_attack, cw_div2_attack, cw_div3_attack, cw_div4_attack]
    reg_weights = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]
    confidences = [0, 20, 40]

    # neuron coverage params
    nc_thresholds = 0. # all activations are scaled to (0,1) after relu

    # inception score (is) params
    is_cuda = use_cuda
    is_batch_size = 10
    is_resize = True
    is_splits = 10

    # frechet inception distance score (fid) params
    fid_batch_size = 64
    fid_cuda = use_cuda
    real_path = "temp_imgs/cifar/real_cw_cifar10/"
    fake_path = "temp_imgs/cifar/fake_cw_cifar10/"

    with open('logs/cw_cifar10_error_log_' + str(date) + '.txt', 'w') as error_log: 

        for model in models:

            for attack in attack_versions:

                results = []
                model_name = model.__class__.__name__
                save_file_path = 'assets/cw_results_cifar10_' + model_name + '_' + str(date) + '.pkl'   

                # neuron coverage
                covered_neurons, total_neurons, neuron_coverage_000 = eval_nc(model, inputs, 0.00)
                print('neuron_coverage_000:', neuron_coverage_000)
                covered_neurons, total_neurons, neuron_coverage_020 = eval_nc(model, inputs, 0.20)
                print('neuron_coverage_020:', neuron_coverage_020)
                covered_neurons, total_neurons, neuron_coverage_050 = eval_nc(model, inputs, 0.50)
                print('neuron_coverage_050:', neuron_coverage_050)
                covered_neurons, total_neurons, neuron_coverage_075 = eval_nc(model, inputs, 0.75)
                print('neuron_coverage_075:', neuron_coverage_075)

                init = {'desc': 'Initial inputs, targets, classes', 
                        'inputs': inputs,
                        'targets': targets,
                        'classes': classes,
                        'neuron_coverage_000': neuron_coverage_000,
                        'neuron_coverage_020': neuron_coverage_020,
                        'neuron_coverage_050': neuron_coverage_050,
                        'neuron_coverage_075': neuron_coverage_075}
                
                results.append(init) 

                layer_dict = get_model_layers(model)
                n = int(len(layer_dict) / num_layers)
                target_layers = list(layer_dict)[0::n]

                for layer_idx in target_layers:
                    module = layer_dict[layer_idx]
                    for rw in reg_weights:
                        for c in confidences:

                            try:
                            
                                timestamp = str(datetime.datetime.now()).replace(':','.')
                                
                                attack_detail = ['timestamp', timestamp, 
                                                 'attack', attack.__name__, 
                                                 'layer: ', layer_idx, 
                                                 'regularization_weight: ', rw, 
                                                 'confidence: ', c]

                                print(*attack_detail, sep=' ')
                                
                                # adversarial attack 
                                adversaries = attack(model, module, rw, inputs, targets, device, targeted, norm_type, epsilon,
                                                     c, c_range, search_steps, max_steps, abort_early, box,
                                                     optimizer_lr, init_rand, log_frequency)
                               
                                # evaluate adversary effectiveness
                                pert_acc, orig_acc = eval_performance(model, inputs, adversaries, targets)
                                # sample_3D_images(model, inputs, adversaries, targets, classes)
                                
                                pert_acc = pert_acc.item() / 100.
                                orig_acc = orig_acc.item() / 100.

                                attack_success_rate = 1 - pert_acc
                                
                                # neuron coverage
                                covered_neurons, total_neurons, neuron_coverage_000 = eval_nc(model, adversaries, 0.00)
                                print('neuron_coverage_000:', neuron_coverage_000)
                                covered_neurons, total_neurons, neuron_coverage_020 = eval_nc(model, adversaries, 0.20)
                                print('neuron_coverage_020:', neuron_coverage_020)
                                covered_neurons, total_neurons, neuron_coverage_050 = eval_nc(model, adversaries, 0.50)
                                print('neuron_coverage_050:', neuron_coverage_050)
                                covered_neurons, total_neurons, neuron_coverage_075 = eval_nc(model, adversaries, 0.75)
                                print('neuron_coverage_075:', neuron_coverage_075)
                                
                                # inception score
                                preprocessed_advs = preprocess_3D_imgs(adversaries)
                                mean_is, std_is = inception_score(preprocessed_advs, is_cuda, is_batch_size, is_resize, is_splits)
                                print('inception_score:', mean_is)
                                
                                # fid score 
                                paths = [real_path, fake_path]
                                
                                # dimensionality = 64
                                target_num = 64
                                generate_imgs(inputs, real_path, target_num)
                                generate_imgs(adversaries, fake_path, target_num)
                                fid_score_64 = calculate_fid_given_paths(paths, fid_batch_size, fid_cuda, dims=64)
                                print('fid_score_64:', fid_score_64)
                                
                                # dimensionality = 2048
                                target_num = 2048
                                generate_imgs(inputs, real_path, target_num)
                                generate_imgs(adversaries, fake_path, target_num)
                                fid_score_2048 = calculate_fid_given_paths(paths, fid_batch_size, fid_cuda, dims=2048)
                                print('fid_score_2048:', fid_score_2048)

                                # output impoartiality
                                pert_output = model(adversaries)
                                y_pred = torch.argmax(pert_output, dim=1)
                                output_impartiality, y_pred_entropy, max_entropy = calculate_output_impartiality(targets, y_pred)
                                print('output_impartiality:', output_impartiality)
                                
                                out = {'timestamp': timestamp, 
                                       'attack': attack.__name__,
                                       'model': model_name, 
                                       'layer': layer_idx, 
                                       'regularization_weight': rw, 
                                       'confidence': c, 
                                       'adversaries': adversaries,
                                       'pert_acc':pert_acc, 
                                       'orig_acc': orig_acc,
                                       'attack_success_rate': attack_success_rate, 
                                       'neuron_coverage_000': neuron_coverage_000,
                                       'neuron_coverage_020': neuron_coverage_020,
                                       'neuron_coverage_050': neuron_coverage_050,
                                       'neuron_coverage_075': neuron_coverage_075,
                                       'inception_score': mean_is,
                                       'fid_score_64': fid_score_64,
                                       'fid_score_2048': fid_score_2048,
                                       'output_impartiality': output_impartiality}
                                
                                results.append(out)
                                
                                # save incremental outputs
                                with open(save_file_path, 'wb') as handle:
                                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            except Exception as e: 

                                print(str(traceback.format_exc()))
                                error_log.write("Failed on attack_detail {0}: {1}\n".format(str(attack_detail), str(traceback.format_exc())))

                            finally:

                                pass

if __name__ == '__main__':
    try:
        main()
    except Exception as e: 
        print(traceback.format_exc())