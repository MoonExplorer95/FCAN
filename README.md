# Cross-Domain Classification based on Frequency Components Adaptation for Remote Sensing Images
Pytorch implementation of FCAN. 

Cross-domain scene classification, investigates how to transfer knowledge from labeled source domains to unlabeled target domain data to improve its classification performance. This task can reduce the labeling cost of remote sensing images and improve the generalization ability of the model. However, the huge distributional gap between labeled source domains and unlabeled target domains acquired by different scenes and different sensors are the core challenge. Existing cross-domain scene classification methods focus on designing better distributional alignment constraints, but are under-explored for fine-grained features. We propose a cross-domain scene classification method called Frequency Component Adaptation Network (FCAN), which considers low-frequency features and high-frequency features separately for more comprehensive adaptation. Specifically, the features are refined and aligned separately through a high-frequency feature enhancement module (HFE) and a low-frequency feature extraction module (LFE). We conducted extensive transfer experiments on 12 cross-scene tasks between the AID, CLRS, MLRSN, and RSSCN7 datasets, as well as 2 cross-sensor tasks between the NWPU-RESISC45 and NaSC-TG2 datasets, and the results show that FCAN can effectively improve the model’s performance for scene classification on unlabeled target domains compared to other methods.

## Usage
### Prerequisites
We experimented with python==3.8, pytorch==1.8.1, cudatoolkit==10.1. 

### Training
To reproduce results on cross-scene datasets,

```shell
bash cross_scene.sh
```

<!-- ## Citation
If you find our paper and code useful for your research, please consider citing
```bibtex
@article{sun2022domain,
    author    = {Sun, Tao and Lu, Cheng and Ling, Haibin},
    title     = {Domain Adaptation with Adversarial Training on Penultimate Activations},
    journal   = {AAAI Conference on Artificial Intelligence },
    year      = {2023}
}
``` -->