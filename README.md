# ticl - Tabular In-Context Learning

This repository contains code for training and prediction of several models for tabular in-context learning, including **MotherNet**, **GAMformer** and **TabFlex**.
**MotherNet** is a hypernetwork foundational model (or conditional neural process) for tabular data classification that creates a small neural network.
**GAMformer** is a model trained to output an interpretable, additive model using in-context learning.
**TabFlex** is a extension of ``TabPFN``  using linear attention that overcomes the scaling limitations of ``TabPFN`` in terms of features, models and number of classes.

- [MotherNet](#MotherNet)
- [GAMformer](#GAMformer)
- [TabFlex](#TabFlex)

Both the architecture and the code in this repository is based on the [TabPFN](https://github.com/automl/TabPFN) by the [Freiburg AutoML group](https://www.automl.org/).

The repository includes code for training and prediction with these models, as well as links to checkpoints for the models used in our publications.

All models provided are research prototypes, shared for research use, and not meant for real-world applications. Responsibility for using the models contained in this repository, as well monitoring and assessing potential impact of the models lies with the user of the code.

# MotherNet

## Installation

It's recommended to use conda to create an environment using the provided environment file:

```
conda create -f environment.yml
```
Then install the package:
```
conda activate ticl
pip install -e .
```

## Getting started

A simple usage of the MotherNet sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ticl.prediction import MotherNetClassifier, EnsembleMeta

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# MotherNetClassifier encapsulates a single instantiation of the model.
# This will automatically download a model from blob storage

classifier = MotherNetClassifier(device='cpu')

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))

# Ensembling as described in the TabPFN paper an be performed using the EnsembleMeta wrapper
ensemble_classifier = EnsembleMeta(classifier)
# ...
```

### MotherNet Usage

MotherNet uses the same preprocessing as the TabPFN work it builds upon, but we found that using one-hot-encoding during inference improves accuracy.
Scaling of features is handled internally.

### Model Training
Full model training code is provided. Training ``MotherNet`` is possible with ``python fit_model.py mothernet``. A GPU ID can be specified with ``-g GPU_ID``. See the ``python fit_model.py mothernet -h`` and ``python fit_model.py -h`` for more options.
The results in the paper correspond to ``python fit_model.py mothernet -L 2``, though default values might change and no longer reflect the values in the paper.
Data-parallel Multi-GPU training is in principal supported using ``torchrun``.
By default, experiments are tracked using MLFlow if the ``MLFLOW_HOSTNAME`` environment variable is set. 

### Environment Prior (SCM/GP, minimal)
This repository also includes a minimal ``EnvironmentPrior`` in ``ticl/priors/environment_prior.py`` and exposes it through ``prior_type=environment_only``.

The input-side variables are explicitly partitioned for both SCM-style MLP and GP-style generators:
``[state | obs | action | noise | zero_pad]``, where ``obs`` is always sampled as a subset of ``state``.

Default randomization ranges in ``model_configs.py``:
- ``action_dim`` in ``[1, 30]``
- ``obs_dim`` in ``[1, 400]``
- ``state_dim`` in ``[1, 400]`` with ``obs_dim <= state_dim``
- ``noise_dim`` in ``[1, 64]``
- ``zero_pad_dim`` in ``[0, 400]``

The prior uses the same native hyperparameter sampling flow as other priors (``parse_distributions`` + ``sample_distributions``) and consumes ``single_eval_pos`` from the dataloader as part of rollout initialization.

It also provides a minimal differentiable rollout API for policy optimization:
- ``rollout_with_policy(...)`` for ``(s_t, a_t, r_t) -> a_{t+1} -> (s_{t+1}, r_{t+1})``
- ``policy_gradient_loss_from_rewards(...)`` for normalized reward objective with stability knobs (eps/clip/discount)

## Papers
This work is described in [MotherNet: A Foundational Hypernetwork for Tabular Classification](https://arxiv.org/pdf/2312.08598).
Please cite that work when using this code. As this work rests on the TabPFN work, I would suggest you also cite their [paper](https://arxiv.org/abs/2207.01848),
which also provides more background on the methodology.

# GAMformer

WIP

# TabFlex

Recent advances in the field of in-context learning (ICL) have demonstrated impressive performance for tabular classification, exemplified by TabPFN's success on small datasets. However, the quadratic complexity of the attention mechanism limits its applicability to larger datasets. To address this issue, we conduct a comprehensive comparison of popular scalable attention alternatives, including state-space models (SSMs) and linear attention mechanisms, revealing that the inherent causality of SSMs hinders ICL performance for large datasets, while linear attention preserves effectiveness. Leveraging these insights, we introduce TabFlex, a model based on linear attention that supports thousands of features and hundreds of classes, capable of handling datasets with millions of samples. Extensive experiments demonstrate that TabFlex is significantly faster than most existing methods while achieving top-two performance on small datasets among 25 baselines, with a 2xspeedup over TabPFN and a 1.5xspeedup over XGBoost. On large datasets, TabFlex remains efficient (e.g., approximately 5 seconds on the poker-hand dataset, which consists of millions of samples), while achieving relatively solid performance.

---

## **Step 1: Install Environment for TabFlex**

Create the Conda environment using the provided file:

   ```bash
   git clone https://github.com/microsoft/ticl
   conda env create -f ticl/tabflex_conda.yaml
   cd ../ticl
   pip install -e .
   ```

---

## **Step 2: Model Inference**

Below is an example of using TabFlex for logistic classification.

```python
from ticl.prediction.tabflex import TabFlex
import torch

# Generate synthetic dataset
X_train = torch.randn(300, 20)
coef = torch.randn(20) 
y_train = (X_train @ coef > 0).int()

X_test = torch.randn(50, 20)
y_test = (X_test @ coef > 0).int()

# Initialize and train TabFlex model
tabflex = TabFlex()
tabflex.fit(X_train, y_train)

# Make predictions
y_pred = tabflex.predict(X_test)

# Evaluate performance
acc = (torch.tensor(y_pred) == y_test).float().mean().item()
print(f"Accuracy: {acc:.4f}")
```

---

## **Step 3: Test TabFlex on Different Datasets**

To evaluate TabFlex on various datasets, use [TabZilla](https://github.com/yzeng58/tabzilla). Follow these steps:

1. Clone the TabZilla repository:
   ```bash
   git clone https://github.com/yzeng58/tabzilla
   ```
2. Follow the instructions in the TabZilla README.  
   - When specifying the `--model_name` parameter, set it to `tabflex`:
     ```bash
     --model_name tabflex
     ```

---



## License
Copyright 2022 Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter

Additions by Andreas Mueller, 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
