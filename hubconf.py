import requests
import array as arr

import torch

from pos_taggers import cnn_pos_tagger

_local_run = False
_data_url = f'https://raw.githubusercontent.com/koleslena/sanskrit_nlp_models/main/output/{{}}_data.dat'

def cnn_pos_tagger_model(**kwargs):
	data = arr.array('i')

	model_name = cnn_pos_tagger.get_model_name()

	if _local_run:
		with open(f'output/{model_name}_data.dat', 'rb') as f:
			data.fromfile(f, 2) 
	else:
		response = requests.get(_data_url.format(model_name))
		file_content = response.content
		data.frombytes(file_content)

	model = cnn_pos_tagger.get_cnn_model(data[0], data[1], **kwargs)
	model.load_state_dict(torch.load(f'output/{model_name}.pth'))
	return model

def cnn_full_pos_tagger_model(**kwargs):
	data = arr.array('i')
	
	model_name = cnn_pos_tagger.get_model_name(True)

	if _local_run:
		with open(f'output/{model_name}_data.dat', 'rb') as f:
			data.fromfile(f, 2) 
	else:
		response = requests.get(_data_url.format(model_name))
		file_content = response.content
		data.frombytes(file_content)

	model = cnn_pos_tagger.get_cnn_model(data[0], data[1], **kwargs)
	model.load_state_dict(torch.load(f'output/{model_name}.pth'))
	return model


def main():
	model = cnn_full_pos_tagger_model()
	print(model)

if __name__ == "__main__":
	_local_run = True
	main()