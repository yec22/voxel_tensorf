from .blender import BlenderDataset
from .tankstemple import TanksTempleDataset
from .dtu import DTUDataset


dataset_dict = {'blender': BlenderDataset,
               'tankstemple':TanksTempleDataset,
               'dtu':DTUDataset
               }