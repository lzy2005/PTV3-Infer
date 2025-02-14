# write the result of segmentation
import os.path
import random

import torch
import numpy
import torch.nn.functional as F
import open3d

to_rgb_dict = dict()
def to_rgb(id):
	if id in to_rgb_dict:
		return to_rgb_dict[id]
	else:
		to_rgb_dict[id] = random.getrandbits(24)
		return to_rgb_dict[id]

def write_pcd(input_dict, output_dict, write_dir, id):
	to_rgb_dict.clear()
	# output_dict = input_dict.pop("segment").detach().cpu().numpy()
	input_dict = input_dict.pop("coord").detach().to("cpu").numpy()
	output_dict = output_dict.pop("seg_logits").detach()
	output_dict = F.softmax(output_dict, dim=1)
	output_dict = torch.multinomial(output_dict, num_samples=1)
	output_dict = output_dict.to("cpu")
	output_dict = output_dict.numpy()

	with open(os.path.join(write_dir,f"seg{id}.pcd"),'w') as file:
		file.write("# .PCD v0.7 - Point Cloud Data file format\n")
		file.write("VERSION 0.7\n")
		file.write("FIELDS x y z rgb\n")
		file.write("SIZE 4 4 4 4\n")
		file.write("TYPE F F F U\n")
		file.write("COUNT 1 1 1 1\n")
		file.write(f"WIDTH {output_dict.shape[0]}\n")
		file.write("HEIGHT 1\n")
		file.write("VIEWPOINT 0 0 0 1 0 0 0\n")# TODO: Think about it
		file.write(f"POINTS {output_dict.shape[0]}\n")
		file.write("DATA ascii\n")
		for i in range(output_dict.shape[0]):
			file.write(f"{input_dict[i][0]} {input_dict[i][1]} {input_dict[i][2]} {to_rgb(output_dict[i][0])}\n")