
# Seeing Earth at the Edge

## Adaptive inference for vision models in edge and remote sensing deployments 


Our goal is to investigate whether a vision transformer can reduce inference computation by adapting its execution depth to individual inputs, while preserving classification accuracy.

### Start here

**Experiments**
- [Notebook I — Static ViT baseline](I_static_vit_baseline.ipynb)
- [Notebook II — Adaptive ViT inference](II_adaptive_vit_inference.ipynb)

**Documentation**
- [Design decisions and limitations](docs/DESIGN_AND_LIMITATIONS.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)

We deliver our experiments in a set of two notebooks, and explain more details regarding the design and reproducibility of the experiments in additional documentation (in docs/). Relevant background literature papers are also attached (in literature/). Data is saved under data/, and the outputs from experiment runs are saved under runs/.

## Dataset
 
We use NWPU-RESISC45 for remote sensing scene
classification. Developed by the Northwestern Polytechnic University, NWPU-RESISC45 contains 31,500 RGB images, each 256 × 256 pixels, across 45 balanced classes with 700 images per class. Its classes include forests, harbours, industrial areas, agricultural patterns, and different residential densities. In this project, it will serve as our optical (RGB image) remote sensing scene-classification benchmark dataset.
The images come with scene-level labels and a consistent RGB format, meaning we can focus this project on investigating primarily the execution of the adaptive model inference without first needing to work on processing radar channels or developing some detection-annotation pre-processing pipelines.

Original dataset authors and access:
https://gcheng-nwpu.github.io/
See the "NWPU-RESISC45 dataset" entry under Datasets.

Dataset paper:
Gong Cheng, Junwei Han, and Xiaoqiang Lu.
"Remote Sensing Image Scene Classification: Benchmark and State of the Art."
Proceedings of the IEEE, 105(10), 1865–1883, 2017.
https://doi.org/10.1109/JPROC.2017.2675998

Image archive (direct download link from Hugging Face):
https://huggingface.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/NWPU-RESISC45.zip


Published split definitions:
https://github.com/google-research/google-research/tree/master/remote_sensing_representations

Train:
https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt

Validation:
https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt

Test:
https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt

We download these split manifests from the Google Research links,
with checksum-verified copies at the pinned mirror as a fallback.
It additionally divides the published validation partition into
model-selection and policy-validation subsets.

Dataset license: CC BY-NC 4.0, as specified by the dataset authors.
https://creativecommons.org/licenses/by-nc/4.0/

The actual dataset image files are not included in this repository for memory reasons.

## Methods for adaptive inference at the edge

We propose a vision transformer architecture with confidence-based processing depth adaptation during model inference. The data we are working with was deliberately constructed with substantial variation within classes and visual similarity between different classes. That makes it a reasonable setting for testing whether some images can be classified reliably at shallow depths while others benefit from further processing.

## Results

On the held-out NWPU-RESISC45 test set (6,300 images), confidence-guided early exits substantially reduced average inference computation while preserving most of the static model's accuracy.

| Configuration | Accuracy | GFLOPs / image |
|---|---:|---:|
| Static DeiT-Tiny (12 blocks) | 94.71% | 1.274 |
| Adaptive (confidence-guided exits) | 93.79% | 0.585 |

**Main result:** 54.13% less computation for a 0.92 percentage-point reduction in classification accuracy.

At identical average compute, confidence-guided routing also outperformed random exit assignment by 4.74 percentage points, demonstrating that selecting which images receive deeper processing matters.

See the [accuracy–computation frontier](figures/pareto.png) and [Notebook II](II_adaptive_vit_inference.ipynb) for the full evaluation, including fixed-depth comparisons and runtime measurements.





```text
                                                                /
                            _.----------._            [===]    /
                        _.-'  ######      '-._      .----. \[o]\
                      .'  ### ##         ###  '. .-'        /   [===]
                    /  ######          ########  \         /   \
                  /  ####               #########  \           ;
                 /  ###                 #   ##   #  \         v
                ;  ##                 #############  ;      .'
               |   ###                 ############## | .-'
               |   ######                ##########  .-'
            .- |   ########                  ### .--' |
          .'   |   ############               .-'###  |
         /      ;   ############         .---' ####  ;
       .'        \  ############    .---'      ###  /
       ;          \   ####### .----'           ##  /
       \            \  # .--'##                  /
        '---.___.-------' #                   .'
                        '-._ _            _.-'
                               '--------'
```




<3