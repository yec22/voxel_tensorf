from .blender import BlenderDataset
from .dtu import DTUDataset

dataset_dict = {'blender': BlenderDataset,
                'dtu':DTUDataset}