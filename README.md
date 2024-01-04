# Augmentation data with semi-supervised methods

# The aim of the project

Small datasets pose challenges related to limited diversity in examples and potential class imbalance, which may lead to overfitting in neural networks. Data augmentation addresses these issues by introducing a range of transformations to the limited dataset, generating new examples while maintaining the distribution characteristics of the original dataset. Consequently, this augmentation expands the effective size of the datase (Wang, et al., [2017](https://arxiv.org/pdf/1712.04621.pdf?source=post_page)).

Data augmentation involves applying a series of conventional transformation techniques to the existing dataset, including operations such as flipping, rotation, translation, zooming in, and zooming out. This techniques are exclusively applied to the training data, leaving the test data unaltered.

<img width="550" alt="image" src="https://github.com/ekchacon/augmentation-data-with-semi-supervised-methods/assets/46211304/aa0c4a6b-5b93-4776-b43e-0dcd6731a736">


<!--We aim to further improve the performance of our proposed method on regular and large datasets by applying data augmentation. Our proposed method has demonstrated to yield superior results compared to other learning techniques when using non-augmented datasets. -->

Our objective is to enhance the performance of our proposed methodology on both regular and large datasets through the implementation of data augmentation. The demonstrated superiority of our proposed method compared to other learning techniques is noteworthy, particularly in the context of non-augmented datasets, which are [Semi-supervised Classification with regular datasets](https://github.com/ekchacon/semi-supervised-regular-size-datasets.git) and [Semi-supervised Classification with large dataset](https://github.com/ekchacon/semi-supervised-large-size-dataset.git)

# Dataset configuration for experiments

The data augmentation process was applied to the datasets of regular and large datasets, specifically to the training subsets of the Original subsets column. In the case of the MNIST dataset, the 100% of augmented training examples were split into 83.33% for pre-training (300k examples) and 16.67% for training ( 60k examples). The test subset remained unchanged. The same approach was employed for the FASHION and Quickdraw bitmap datasets as outlined by the the subsequent table.

| Dataset                                 | Augmented subsets      |
| :-------------------------------------- | :--------------------- |
| MNIST <br> (370k full)                  | 300k Pre-training      |
|                                         | 60k Training           |
|                                         | 10k Test               |
| FASHION <br> (370k full)                | 300k Pre-training      |
|                                         | 60k Training           |
|                                         | 10k Test               |
| Quickdraw bitmap <br>  (4,225,000 full) | 3,499,860 Pre-training |
|                                         | 700,140 Training       |
|                                         | 25k Test               |

# Experiment design

We selected the 16.67% (60k examples) of the training subset from the previous table and systematically reduced it to 0.33% to generate datasets of varying sizes for experimental purposes as the next table shows. We aim to assess the proposed method and alternative techniques under these datasets.

| AugMNIST | AugFASHION | AugQuickdraw | %      |
| :------- | :--------- | :----------- | :----- |
| 60012    | 60012      | 700140       | 16\.67 |
| 54000    | 54000      | 630000       | 15\.00 |
| 47988    | 47988      | 559860       | 13\.33 |
| 42012    | 42012      | 490140       | 11\.67 |
| 36000    | 36000      | 420000       | 10\.00 |
| 29988    | 29988      | 349860       | 8\.33  |
| 24012    | 24012      | 280140       | 6\.67  |
| 18000    | 18000      | 210000       | 5\.00  |
| 11988    | 11988      | 139860       | 3\.33  |
| 6012     | 6012       | 70140        | 1\.67  |
| 5400     | 5400       | 63000        | 1\.50  |
| 4788     | 4788       | 55860        | 1\.33  |
| 4212     | 4212       | 49140        | 1\.17  |
| 3600     | 3600       | 42000        | 1\.00  |
| 2988     | 2988       | 34860        | 0\.83  |
| 2412     | 2412       | 28140        | 0\.67  |
| 1800     | 1800       | 21000        | 0\.50  |
| 1188     | 1188       | 13860        | 0\.33  |

# Results

Initially, we compare the results of the self-training layer-wise method between the augmented datasets and their non-augmented counterparts. Subsequently, we provide a comparative analysis of the results obtained by all methods when using the augmented datasets exclusivel. 

## Results for MNIST dataset

## Results for FASHION dataset

## Results for Quickdraw Bitmap dataset

