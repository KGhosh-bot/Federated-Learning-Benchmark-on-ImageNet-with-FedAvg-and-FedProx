# 🍎 Federated Learning Benchmark on ImageNet with FedAvg and FedProx

## 🌟 Project Overview

This repository presents an experimental analysis of two leading Federated Learning (FL) algorithms, **Federated Averaging (FedAvg)** and **Federated Proximal Averaging (FedProx)**, applied to a large-scale, complex dataset: **ImageNet (ILSVRC-2012)**.

While FedProx is typically validated on simpler benchmarks (e.g., CIFAR-10 or MNIST), this project aims to evaluate its performance and stability when dealing with the high dimensionality and complexity of ImageNet, particularly under challenging **Non-IID (Non-Independent and Identically Distributed)** data conditions.

## 🛠️ Key Technical Components

* **Model:** ResNet-18 (Pre-trained on ImageNet)
* **Dataset:** ImageNet (ILSVRC-2012 Validation Subset)
* **Framework:** PyTorch
* **FL Algorithms:** FedAvg and FedProx
* **Data Partitioning:** Dirichlet Distribution ($\alpha$) for simulating Non-IID client heterogeneity.

## 💡 Core Research Focus & Challenges

The core of this work lies in addressing the well-known **client drift** problem in FL caused by Non-IID data. Specifically, we investigate:

1.  **Performance Drop:** Quantifying the accuracy difference between IID ($\alpha=100$) and highly Non-IID ($\alpha=0.1$) distributions.
2.  **FedProx Efficacy:** Evaluating if the proximal term in FedProx effectively mitigates client drift compared to FedAvg on complex ImageNet features, given the known need for **more training rounds and extensive $\mu$ (proximal term weight) experimentation**.

---

## 📈 Results and Analysis

### Final Comparison Results

| Model | Non-IID ($\alpha=0.1$) Final Accuracy | IID ($\alpha=100$) Final Accuracy |
| :--- | :--- | :--- |
| **FedAvg** | **56.7%** | **58.66%** |
| **FedProx** | **56.18%** | **58.16%** |

### Critical Analysis of Findings

1.  **Impact of Data Heterogeneity:**
    * Both FedAvg and FedProx show a clear performance drop (approximately **2%**) when transitioning from the IID-like ($\alpha=100$) to the highly Non-IID ($\alpha=0.1$) distribution. This confirms the severity of the **client drift** issue when using complex, high-dimensional data like ImageNet.

2.  **FedProx Performance on ImageNet:**
    * Contrary to expectations based on simpler datasets, **FedProx did not provide a substantial performance improvement over FedAvg** under the tested setup. The final accuracies were highly competitive ($56.7\%$ vs. $56.2\%$).
    * This is a valuable finding, suggesting that for high-complexity models and datasets, the fixed proximal term ($\mu=0.01$) did not sufficiently mitigate client drift within the limited training budget (20 rounds, 5 epochs).

3.  **Future Work and Limitations:**
    * The lack of significant difference is likely a consequence of the **limited training resources**. ImageNet requires many more rounds and local epochs for convergence than were feasible for this initial benchmark.
    * The results are based on a single, default $\mu$ value. **Extensive hyperparameter tuning** of the proximal weight $\mu$ is required to unlock the potential benefits of the FedProx algorithm on such complex data.
    * Further experimentation with **larger training rounds (e.g., 100+ rounds)** and dynamic $\mu$ schedules is necessary to fully validate FedProx's effectiveness on production-scale, Non-IID image data.

---
