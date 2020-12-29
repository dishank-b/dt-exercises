from .dataloader import DuckieDataset

def build_dataset(data_name):
	all_datasets = {"DuckieDataset": DuckieDataset}
	return all_datasets[data_name]
