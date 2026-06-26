from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier

# Backbones
# Windows deployment note:
# The current transmission-line checkpoint only needs PT-v3m1.
# Importing older PointTransformer modules pulls optional CUDA extensions such
# as pointops, which are not required for this model and are hard to build on
# native Windows.
# from .sparse_unet import *
# from .point_transformer import *
# from .point_transformer_v2 import *
from .point_transformer_v3 import *
# from .stratified_transformer import *
# from .spvcnn import *

# OctFormer requires extra ocnn/dwconv dependencies; PTv3 reproduction does not need it.
# from .octformer import *

# from .swin3d import *

# Semantic Segmentation
# from .context_aware_classifier import *

# Instance Segmentation
# PointGroup requires optional pointgroup_ops; not needed for PTv3 semantic segmentation.
# from .point_group import *

# Pretraining
# from .masked_scene_contrast import *
# from .point_prompt_training import *
