from common.models_factory import load_segmenter_model_from_url, load_tagger_model_from_url
from sanskrit_tagger.device_util import get_device


def pos_tagger_model(version='latest', device=None, **kwargs):
	device = get_device(device)
	model = load_tagger_model_from_url(version, device, **kwargs)
	return model

def segmenter_model(version='latest', device=None, **kwargs):
	device = get_device(device)
	model = load_segmenter_model_from_url(version, device, **kwargs)
	return model

def main():
	model = segmenter_model()
	print(model)

if __name__ == "__main__":
	main()