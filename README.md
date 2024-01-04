# Augmentation data with semi-supervised methods

# The aim of the project

Small datasets pose challenges related to limited diversity in examples and potential class imbalance, which may lead to overfitting in neural networks. Data augmentation addresses these issues by introducing a range of transformations to the limited dataset, generating new examples while maintaining the distribution characteristics of the original dataset. Consequently, this augmentation expands the effective size of the datase (Wang, et al., [2017](https://arxiv.org/pdf/1712.04621.pdf?source=post_page)).

Data augmentation involves applying a series of conventional transformation techniques to the existing dataset, including operations such as flipping, rotation, translation, zooming in, and zooming out. This techniques are exclusively applied to the training data, leaving the test data unaltered.

<!--We aim to further improve the performance of our proposed method on regular and large datasets by applying data augmentation. Our proposed method has demonstrated to yield superior results compared to other learning techniques when using non-augmented datasets. -->

Our objective is to enhance the performance of our proposed methodology on both regular and large datasets through the implementation of data augmentation. The demonstrated superiority of our proposed method compared to other learning techniques is noteworthy, particularly in the context of non-augmented datasets, which are [Semi-supervised Classification with regular datasets](https://github.com/ekchacon/semi-supervised-regular-size-datasets.git) and [Semi-supervised Classification with large dataset](https://github.com/ekchacon/semi-supervised-large-size-dataset.git)
