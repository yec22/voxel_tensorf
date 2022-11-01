from .blender import BlenderDataset
from .dtu import DTUDataset
from .bmvs import BMVSDataset

dataset_dict = {'blender': BlenderDataset,
                'dtu': DTUDataset,
                'bmvs': BMVSDataset}