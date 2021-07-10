# deepcluster - clustering of video files

This project aims to use the great work done in the *monkey_caput* repo on deep image clustering / segmentation and
apply it to frames extracted from videos.

Use the added requirements.txt file to install relevant requirements: ```pip install -r requirements.txt```

*Everything below this line is from the original [repo](https://github.com/anderzzz/monkey_caput) README*
---

## Original Repo - monkey_caput

Code used in fungi image analysis, supervised and unsupervised. Effort described in Towards Data Science,
see https://towardsdatascience.com/image-clustering-implementation-with-pytorch-587af1d14123 (no paywall).

The fungi image data is loaded and pre-procssed in `fungidata.py` in which the DataSet class is created through a
factory method. That includes full images, grid images, with or without ground-truth label or index in dataset. The
specific image dataset is presently proprietary, but can be recreated from the Danish fungi atlas,
see https://svampe.databasen.org

Image classification efforts are in files starting with `ic`. The template models for example are loaded
in `ic_template_models.py`. The auto-encoder is defined in `ae_deep.py` with a learner class in `ae_learner`. Local
Aggregation criterion is found in `cluster_utils` and its learner class in `la_learner`. The training inherits
from `_learner`. To use the implementation for another custom dataset, modify how `self.dataset` is set in `_learner`.
