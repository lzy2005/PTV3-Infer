import torch
from collections import OrderedDict

state_dict_origin = torch.load("model_best_origin.pth")['state_dict']
state_dict = OrderedDict()

for key in state_dict_origin.keys():
	state_dict[key[7:]] = state_dict_origin[key]

torch.save(state_dict, "model_best.pth")