# Semi-supervised classification with augmented datasets

# Datasets

The subsequent datasets comprise 28 x 28 pixel grayscale images.

The MNIST dataset encompasses handwritten digits (0 - 9) categorized into 10 classes, with 60,000 images designated for training and 10,000 for testing. 

The Fashion dataset also features 10 classes, comprising images of Zolando’s articles such as T-Shirts, Trousers, Bags, Ankle boots, and others. It includes 60,000 and 10,000 images for training and testing, respectively.

The Quickdraw bitmap dataset comprises 345 distinct drawing classes and a total of 50 million examples. For our experiments, we have selected a subset of 10 classes, containing 700,000 training examples and 25,000 test examples.

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

## Augmented compared to non-augmented datasets (MNIST, FASHION and Quickdraw)

The following figure presents the accuracy outcomes of the proposed method trained on the MNIST dataset and the Augmented MNIST dataset. Notably, the chart reveals that the method achieved higher performance with the Augmented MNIST dataset, surpassing the results obtained with the standard MNIST dataset during the initial portion of training labeled examples. However, as the percentage of the training dataset increased, the accuracy results for both datasets converged, demonstrating similarity in performance during the latter stages of training.

<img width="737" alt="image" src="https://github.com/ekchacon/augmentation-data-with-semi-supervised-methods/assets/46211304/7c97aa4b-c4b0-4b31-8207-5205af9fd411">

The performance of the proposed method when applied to the FASHION and Augmented FASHION datasets is presented in the subsequent figure. Overall, the method consistently exhibited higher accuracy with the Augmented FASHION dataset compared to the original FASHION dataset, irrespective of the percentage of labeled examples used for training

<img width="737" alt="image" src="https://github.com/ekchacon/augmentation-data-with-semi-supervised-methods/assets/46211304/c443dd5a-c29c-4406-91b2-89f97faa02da">

The performance of the self-training layer-wise approach when applied to the Quickdraw dataset and Augmented Quickdraw dataset is presented in the coming figure. Notably, both datasets exhibited a consistent increase in accuracy across various dataset sizes. However, the performance of the method with the Augmented Quickdraw dataset consistently outperformed that with the standard Quickdraw dataset.

<img width="738" alt="image" src="https://github.com/ekchacon/augmentation-data-with-semi-supervised-methods/assets/46211304/109397ce-5134-4f01-a67d-63e8dee2cf8b">

## All Methods with augmented datasets (MNIST, FASHION and Quickdraw)

We present a comparative analysis between the proposed self-training layer-wise method and other learning techniques when exclusively trained on the augmented dataset versions. Specifically, we compare self-training layer-wise with self-training, semi-supervised, and supervised methods across diverse scenarios involving varying labeled dataset sizes.

The subsequent figure presents a comprehensive analysis of our method’s performance and that of other techniques when trained on various labeled dataset sizes derived from the Augmented MNIST dataset. In the dataset size range of 0.33% to 1.67%, our method consistently outperforms the alternatives, achieving accuracy rates of approximately 94% and 97%, respectively. However, as the labeled dataset size increases to the range of 3.33% to 16.67%, our method’s accuracy results align closely with those of the alternative methods, typically hovering around 99%.

<img width="737" alt="image" src="https://github.com/ekchacon/augmentation-data-with-semi-supervised-methods/assets/46211304/e3a72b44-31db-4423-9a4c-b12e9551ab1b">

The Augmented FASHION dataset is utilized for comparative analysis of our proposed method against alternative approaches, as depicted in the following figure. Our self-training layer-wise methodology surpasses the alternatives within the dataset size range of 0.33% to 1.50%, achieving accuracy rates that climb from 72% to 81%. Although the accuracy result does not exhibit superiority at 1.67%, it maintains its advantage and steadily increases from 85% to just under 90% within the dataset size range of 3.33% to 16.67%.

<img width="738" alt="image" src="https://github.com/ekchacon/augmentation-data-with-semi-supervised-methods/assets/46211304/ce65de72-4f5e-4d9a-9000-029850afcb81">

The performance outcomes of the proposed method, in comparison to alternative methods, using the Augmented Quickdraw dataset are visually presented in the coming figure. Within the dataset size range of 0.33% to 1.67%, our method exhibits superior performance, achieving accuracy rates of 85% and slightly exceeding 88%, respectively. Similarly, our method maintains a slight advantage over the alternatives, with accuracy results of approximately 90% and just above 92% in the dataset size range of 3.33% to 16.67% within various labeled training scenarios.

<img width="737" alt="image" src="https://github.com/ekchacon/augmentation-data-with-semi-supervised-methods/assets/46211304/9a7adb64-1e1f-42bd-a55d-c47b7ae10617">
