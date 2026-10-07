# Adaptive Inference for Edge AI

Adaptive inference for edge AI in remote sensing - Vision transformer depth adaptation

## Dataset and provenance

We use NWPU-RESISC45 for remote sensing scene
classification from overhead imagery.

Original dataset authors and access:
https://gcheng-nwpu.github.io/
See the "NWPU-RESISC45 dataset" entry under Datasets.

Dataset paper:
Gong Cheng, Junwei Han, and Xiaoqiang Lu.
"Remote Sensing Image Scene Classification: Benchmark and State of the Art."
Proceedings of the IEEE, 105(10), 1865–1883, 2017.
https://doi.org/10.1109/JPROC.2017.2675998

Image archive:
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