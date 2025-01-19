import torch
import torch.nn.functional as F
import numpy as np
import os
from functools import partial

from pointcept.utils.logger import get_root_logger
from pointcept.models.default import Segmentor
from pointcept.datasets.utils import collate_fn
from pointcept.datasets.defaults import IndoorDataset

class Inferer:
	def __init__(self,cfg):
		self.logger = get_root_logger(
			log_file=os.path.join(cfg['logger_dir'], "infer.log"),
			file_mode="w",
		)
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
		self.model.eval()
		for id, input_dict in enumerate(self.infer_loader):
			for key in input_dict.keys():
				if isinstance(input_dict[key], torch.Tensor):
					input_dict[key] = input_dict[key].cuda(non_blocking=True)
			with torch.no_grad():
				output_dict = self.model(input_dict)
				tensor_cpu = output_dict.pop("seg_logits").detach()
				tensor_cpu = F.softmax(tensor_cpu, dim=1)
				tensor_cpu = torch.multinomial(tensor_cpu, num_samples=1)
				tensor_cpu = tensor_cpu.to("cpu")
				with open(os.path.join(self.output_dir,'seg_out{}.pcd'.format(id)), "w") as file:
					file.write(f"# .PCD v0.7 - Point Cloud Data file format\n")
					file.write(f"VERSION 0.7\n")
					file.write(f"FIELDS segmentation\n")
					file.write(f"SIZE 4\n")
					file.write(f"TYPE F\n")
					file.write(f"COUNT 1\n")
					file.write(f"WIDTH {tensor_cpu.shape[0]}\n")
					file.write(f"HEIGHT 1\n")
					file.write(f"VIEWPOINT 0\n")
					file.write(f"POINTS {tensor_cpu.shape[0]}\n")
					file.write(f"DATA ascii\n")
					np.savetxt(file, tensor_cpu.numpy(), fmt='%.1f', delimiter=' ')