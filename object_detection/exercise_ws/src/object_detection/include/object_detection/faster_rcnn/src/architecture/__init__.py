from .model import FasterRCNN, FasterRCNN_KF

def build_model(model_name):
	"""
	If any new model is defined,
	then import it above and add to the 
	"all_models" directory
	"""
	all_models = {"FasterRCNN": FasterRCNN,
				"FasterRCNN_KF": FasterRCNN_KF}

	return all_models[model_name]

