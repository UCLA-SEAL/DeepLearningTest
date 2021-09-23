import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import DatasetFolder, ImageFolder

import numpy as np
import pandas as pd
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import faulthandler
faulthandler.enable()

# provides a nice UI element when running in a notebook, otherwise use "import tqdm" only
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

# ============================================================================================== #
# ============================================================================================== #
# ============================================================================================== #
# ======================================= Misc Functions ======================================= #
# ============================================================================================== #
# ============================================================================================== #
# ============================================================================================== #

# https://github.com/pytorch/pytorch/issues/7284
def discretize(tensor, boundaries):
    result = torch.zeros_like(tensor, dtype=torch.int32)
    for boundary in boundaries:
        result += (tensor > boundary).int()
    return result

def convert_to_categorical(regression_labels, num_labels=25):
    min_val = np.min(regression_labels)
    max_val = np.max(regression_labels)
    bins = np.linspace(min_val - 1e-5, max_val + 1e-5, num_labels)
    return np.digitize(regression_labels, bins)

def calculate_output_impartiality(y_true, y_pred): 
    num_classes = len(y_true.unique())
    orig_classes, orig_counts = y_true.unique(return_counts=True) 
    
    # calculate maximum possible entropy from y_true
    max_diversity = torch.ones(num_classes) * (1./num_classes)
    max_entropy = torch.distributions.Categorical(probs=max_diversity).entropy().item()
    
    # calculate class counts
    class_counts = []
    for i in orig_classes:
        count = 0
        for pred in y_pred:
            if pred == i:
                count += 1
        class_counts.append(count)

    # calculate y_pred entropy
    class_counts = torch.tensor(class_counts, dtype=torch.float) 
    class_probs = class_counts / class_counts.sum()
    y_pred_entropy = torch.distributions.Categorical(probs=class_probs).entropy().item()
    
    output_bias = (max_entropy - y_pred_entropy) / max_entropy
    output_impartiality = 1 - output_bias
    return output_impartiality, y_pred_entropy, max_entropy

def extract_outputs(model, data, module):
    outputs = []      
    def hook(module, input, output):
        outputs.append(output)    
    handle = module.register_forward_hook(hook)     
    model(data)
    handle.remove()
    return torch.stack(outputs)

def norm_divergence_by_module(data, model, modules, device, regularizer_weight=None):
    """
    returns the kld between the activations of the specified layer and a uniform pdf
    """

    if not isinstance(modules, list):
        modules = [modules]

    data = torch.clamp(data, 0, 1)

    total_divergence = 0

    for module in modules: 
    
        # extract layer activations as numpy array
        # NOTE: torch.relu is added just in case the layer is not actually ReLU'd beforehand
        #       This is required for the summation and KL-Divergence calculation, otherwise nan
        layer_activations = torch.relu(torch.squeeze(extract_outputs(model=model, data=data, module=module)))
        
        # normalize over summation (to get a probability density)
        if len(layer_activations.size()) == 1:
            out_norm = (layer_activations / torch.sum(layer_activations)) + 1e-20 
        elif len(layer_activations.size()) == 2:
            out_norm = torch.sum(layer_activations, 0)
            out_norm = (out_norm / torch.sum(out_norm)) + 1e-20
        else:
            out_norm = (layer_activations / torch.sum(layer_activations)) + 1e-20 

        # create uniform tensor
        uniform_tensor = torch.ones(out_norm.shape).to(device)

        # normalize over summation (to get a probability density)
        uni_norm = uniform_tensor / torch.sum(uniform_tensor)
        
        # measure divergence between normalized layer activations and uniform distribution
        divergence = F.kl_div(input=out_norm.log(), target=uni_norm, reduction='sum')
        # divergence = F.kl_div(input=uni_norm.log(), target=out_norm, reduction='sum') 
        
        # default regularizer if not provided
        if regularizer_weight is None:
            regularizer_weight = 0.005 
            
        if divergence < 0:
            print('The divergence was technically less than 0', divergence, layer_activations, out_norm)
            torch.save(data, 'logs/data.pt')
            torch.save(out_norm, 'logs/out_norm.pt')
            torch.save(uni_norm, 'logs/uni_norm.pt')
            # return None

        total_divergence += divergence
    
    return regularizer_weight * total_divergence

def eval_performance(model, originals, adversaries, targets):
    pert_output = model(adversaries)
    orig_output = model(originals)

    pert_pred = torch.argmax(pert_output, dim=1)
    orig_pred = torch.argmax(orig_output, dim=1)

    pert_correct = pert_pred.eq(targets.data).sum()
    orig_correct = orig_pred.eq(targets.data).sum()

    pert_acc = 100. * pert_correct / len(targets)
    orig_acc = 100. * orig_correct / len(targets)

    print('Perturbed Accuracy: {}/{} ({:.0f}%)'.format(pert_correct, len(targets), pert_acc))
    print('Original Accuracy: {}/{} ({:.0f}%)'.format(orig_correct, len(targets), orig_acc))
    
    return pert_acc, orig_acc

def eval_performance_reg(model, originals, adversaries, targets, classes, dataset):
    pert_output = model(adversaries)
    orig_output = model(originals)
    
    # MSE
    
    mse = F.mse_loss(pert_output, targets)
    
    # Accuracy

    pert_pred = discretize(pert_output, dataset.boundaries).view(-1)
    orig_pred = discretize(orig_output, dataset.boundaries).view(-1)

    pert_correct = pert_pred.eq(classes.data).sum()
    orig_correct = orig_pred.eq(classes.data).sum()
    
    pert_acc = 100. * pert_correct / len(classes)
    orig_acc = 100. * orig_correct / len(classes)

    print('MSE:{:.4f}'.format(mse))
    print('Perturbed Accuracy: {}/{} ({:.0f}%)'.format(pert_correct, len(classes), pert_acc))
    print('Original Accuracy: {}/{} ({:.0f}%)'.format(orig_correct, len(classes), orig_acc))
    
    return mse, pert_acc, orig_acc

def eval_performance_reg2(model, originals, adversaries, targets, binned_targets, num_labels):
    pert_output = model(adversaries)
    orig_output = model(originals)
    
    # MSE
    
    mse = F.mse_loss(pert_output, targets).item()
    
    # Accuracy

    pert_pred = torch.tensor(convert_to_categorical(pert_output.detach().cpu().numpy(), num_labels)).long().view(-1).cuda()
    orig_pred = torch.tensor(convert_to_categorical(orig_output.detach().cpu().numpy(), num_labels)).long().view(-1).cuda()

    pert_correct = pert_pred.eq(binned_targets.data).sum()
    orig_correct = orig_pred.eq(binned_targets.data).sum()
    
    pert_acc = 100. * pert_correct / len(binned_targets)
    orig_acc = 100. * orig_correct / len(binned_targets)

    print('MSE: {:.4f}'.format(mse))
    print('Perturbed Accuracy: {}/{} ({:.0f}%)'.format(pert_correct, len(binned_targets), pert_acc))
    print('Original Accuracy: {}/{} ({:.0f}%)'.format(orig_correct, len(binned_targets), orig_acc))
    
    return mse, pert_acc, orig_acc

def sample_1D_images(model, originals, adversaries, targets, num_samples = 5):
    orig_inputs = originals.cpu().detach().numpy()
    adv_examples = adversaries.cpu().detach().numpy()
    pert_output = model(adversaries)
    orig_output = model(originals)
    pert_pred = torch.argmax(pert_output, dim=1)
    orig_pred = torch.argmax(orig_output, dim=1)
    plt.figure(figsize=(15,8))
    for i in range(1, num_samples+1):
        plt.subplot(2, num_samples, i)
        plt.imshow(np.squeeze(orig_inputs[i]), cmap='gray')  
        plt.title('true: {}'.format(targets[i].item()))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, num_samples, num_samples+i)
        plt.imshow(np.squeeze(adv_examples[i]), cmap='gray')
        plt.title('adv_pred: {} - orig_pred: {}'.format(pert_pred[i].item(), orig_pred[i].item()))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def sample_3D_images(model, originals, adversaries, targets, classes, num_samples = 5):
    orig_inputs = originals.cpu().detach().numpy()
    adv_examples = adversaries.cpu().detach().numpy()
    pert_output = model(adversaries)
    orig_output = model(originals)
    pert_pred = torch.argmax(pert_output, dim=1)
    orig_pred = torch.argmax(orig_output, dim=1)
    plt.figure(figsize=(15,8))
    for i in range(1, num_samples+1):
        plt.subplot(2, num_samples, i)
        plt.imshow(np.transpose(np.squeeze(orig_inputs[i]), (1, 2, 0)))  
        true_idx = targets[i].item()
        plt.title('true: {}'.format(classes[true_idx]))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, num_samples, num_samples+i)
        plt.imshow(np.transpose(np.squeeze(adv_examples[i]), (1, 2, 0)))  
        pred_idx = pert_pred[i].item()
        orig_idx = orig_pred[i].item()
        plt.title('adv_pred: {} - orig_pred: {}'.format(classes[pred_idx], classes[orig_idx]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def sample_3D_images_reg(model, originals, adversaries, targets, classes, data_loader, num_samples = 5):
    orig_inputs = originals.cpu().detach().numpy()
    orig_targets = targets.cpu().detach().numpy()
    orig_classes = classes.cpu().detach().numpy()
    adv_examples = adversaries.cpu().detach().numpy()
    pert_output = model(adversaries)
    orig_output = model(originals)
    disc_pert = discretize(pert_output, data_loader.boundaries)
    disc_orig = discretize(orig_output, data_loader.boundaries)
    plt.figure(figsize=(15,8))
    for i in range(1, num_samples+1):
        plt.subplot(2, num_samples, i)
        plt.imshow(np.transpose(np.squeeze(orig_inputs[i]), (1, 2, 0)))  
        plt.title('true: %.8f (%i)' % (targets[i], orig_classes[i]))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, num_samples, num_samples+i)
        plt.imshow(np.transpose(np.squeeze(adv_examples[i]), (1, 2, 0)))  
        plt.title('adv_pred: %.8f (%i) \n orig_pred: %.8f (%i)' % (pert_output[i], disc_pert[i], orig_output[i], disc_orig[i]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def generate_batch(dataset, num_per_class, device):
    '''
    creates a batch of inputs with a customizable number of instances for each class
    dataset       : torchvision.dataset
    num_per_class : iterable containing the desired counts of each class
                    example: torch.ones(num_classes) * 100
    '''
    
    def get_same_index(targets, label):
        '''
        Returns indices corresponding to the target label
        which the dataloader uses to serve downstream.
        '''
        label_indices = []
        for i in range(len(targets)):
            if targets[i] == label:
                label_indices.append(i)
        return label_indices

    data = []
    labels = []
    
    num_classes = len(np.unique(dataset.targets))
    
    for i in range(num_classes):
        
        target_indices = get_same_index(dataset.targets, i)
        class_batch_size = int(num_per_class[i])
        
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=class_batch_size, 
            sampler=SubsetRandomSampler(target_indices),
            shuffle=False)

        inputs, targets = next(iter(data_loader))

        data.append(inputs)
        labels.append(targets)

    inputs = torch.cat(data, dim=0).to(device)
    targets = torch.cat(labels, dim=0).to(device)
    
    return inputs, targets

def generate_batch_reg(dataset, num_per_class, device):
    '''
    creates a batch of inputs with a customizable number of instances for each class
    dataset       : torchvision.dataset
    num_per_class : iterable containing the desired counts of each class
                    example: torch.ones(num_classes) * 100
    '''
    
    def get_same_index(targets, label):
        '''
        Returns indices corresponding to the target label
        which the dataloader uses to serve downstream.
        '''
        label_indices = []
        for i in range(len(targets)):
            if targets[i] == label:
                label_indices.append(i)
        return label_indices

    all_data = []
    all_labels = []
    all_cats = []
    
    num_classes = len(np.unique(dataset.discrete_targets.cpu()))
    
    for i in range(num_classes):

        target_indices = get_same_index(dataset.discrete_targets, i)
        class_batch_size = int(num_per_class[i])
        
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=class_batch_size, 
            sampler=SubsetRandomSampler(target_indices),
            shuffle=False)

        if len(data_loader) > 0:
            inputs, targets, classes = next(iter(data_loader))

        all_data.append(inputs)
        all_labels.append(targets)
        all_cats.append(classes)

    inputs = torch.cat(all_data, dim=0).to(device)
    targets = torch.cat(all_labels, dim=0).to(device)
    classes = torch.cat(all_cats, dim=0).to(device)
    
    return inputs, targets, classes

def step_through_model(model, prefix=''):
    for name, module in model.named_children():
        path = '{}/{}'.format(prefix, name)
        if (isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)): # test for dataset
            yield (path, name, module)
        else:
            yield from step_through_model(module, path)

def get_model_layers(model):
    layer_dict = {}
    idx=1
    for (path, name, module) in step_through_model(model):
        layer_dict[path + '-' + str(idx)] = module
        idx += 1
    return layer_dict 

def get_dict_for_layer(dict, layer_name):
    return {k:v for k,v in dict.items() if layer_name in k[0]}

def get_pretrained_weights(model, device, directory="pretrained_models/mnist/", get_any=False):
    latest_model = None
    if get_any:
        prev_models = glob.glob(directory+'*.*')
    else:
        m_type = model.__class__.__name__
        prev_models = glob.glob(directory+'*'+ m_type +'*.*')
    if prev_models:
        latest_model = max(prev_models, key=os.path.getctime)
    if (latest_model is not None):  
        print('loading model', latest_model)
        model.load_state_dict(torch.load(latest_model, map_location=device))  
        return model
    else:
        print('no model found. train a new one.')
        return False

# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #
# ========================================= Data Loader ========================================= #
# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #

class car_loader(Dataset):

    def __init__(self, 
                 target_csv_file, 
                 img_dir, 
                 device,
                 transform=None, 
                 discretize_classes=True, 
                 num_classes=50):
        """
        Args:
            target_csv_file (string) : Path to the csv file with steering angles.
            img_dir (string)         : Directory with all the images.
        Returns:
            images                   : The images for training / inference.
            angles                   : The steering angle for each image
            classes                  : The discretized targets for the number of classes requested
        """
        self.targets = torch.tensor(pd.read_csv(target_csv_file)['steering_angle'].values, dtype=torch.float32).to(device)
        self.discretize_classes = discretize_classes
        self.discrete_targets = self.targets.clone()
        self.root_dir = img_dir
        self.img_paths = glob.glob(os.path.join(img_dir) + '/*.png')
        self.transform = transform
        self.num_classes = num_classes
        
        if discretize_classes:
            
            # https://github.com/pytorch/pytorch/issues/7284
            def discretize(tensor, boundaries):
                result = torch.zeros_like(tensor, dtype=torch.int32)
                for boundary in boundaries:
                    result += (tensor > boundary).int()
                return result
            
            min_bin = self.targets.min() # -1
            max_bin = self.targets.max() # 1
            self.boundaries = torch.linspace(min_bin, max_bin, num_classes).to(device)
            self.discrete_targets = discretize(self.targets, self.boundaries).int()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # images
        paths = self.img_paths[idx]
        images = imread(paths)
        
        if self.transform:
            images = self.transform(images)
        
        # angles
        angles = self.targets[idx]
            
        if self.discretize_classes:
            classes = self.discrete_targets[idx]
            sample = (images, angles, classes)
        else:
            sample = (images, angles)

        return sample