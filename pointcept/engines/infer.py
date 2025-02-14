import torch
import torch.nn.functional as F
import numpy as np
import os
from functools import partial
import random

from pointcept.utils.logger import get_root_logger
from pointcept.models.default import Segmentor
from pointcept.datasets.utils import collate_fn
from pointcept.datasets.defaults import IndoorDataset
from pointcept.utils.write import write_pcd

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

class Inferer:
	def __init__(self,cfg):
		set_seed(25326354)
		self.model = self.build_model()
		self.infer_loader = self.build_infer_loader(cfg['input_dir'])
		self.output_dir = cfg['output_dir']

	def build_model(self):
		model = Segmentor()
		model = model.cuda()
		return model

	def build_infer_loader(self,input_dir):
		infer_data = IndoorDataset(input_dir)

		infer_loader = torch.utils.data.DataLoader(
			infer_data,
			batch_size=1,
			shuffle=False,
			num_workers=1,
			pin_memory=True,
			sampler=None,
			collate_fn=collate_fn,
		)

		return infer_loader

	def infer(self):
		self.model.load_state_dict(torch.load("model_best.pth"))
		self.model.eval()
		for id, input_dict in enumerate(self.infer_loader):
			for key in input_dict.keys():
				if isinstance(input_dict[key], torch.Tensor):
					input_dict[key] = input_dict[key].cuda(non_blocking=True)
			with torch.no_grad():
				output_dict = self.model(input_dict)
				write_pcd(input_dict,output_dict,self.output_dir,id)