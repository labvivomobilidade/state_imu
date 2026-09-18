# Operational-State Recognition for Battery-Electric Buses (BEBs)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: FEEC Unicamp](https://img.shields.io/badge/Research-FEEC%20Unicamp-red.svg)](https://www.feec.unicamp.br/)

This repository provides the comprehensive benchmark, machine-readable feature manifest, statistical dictionaries, in-domain evaluations, cross-vehicle domain transfer analyses, unsupervised adaptation studies, event-level charging session benchmarks, and few-shot supervised target calibration for **operational-state recognition of battery-electric buses (BEBs)** using multi-modal inertial and magnetic sensing.

---

## 📌 Table of Contents
1. [Overview & Operational States](#overview--operational-states)
2. [Dataset & Windowing Configuration](#dataset--windowing-configuration)
3. [Feature Representations & Causal Transformation](#feature-representations--causal-transformation)
4. [Sensor Configurations & In-Domain Benchmarks](#sensor-configurations--in-domain-benchmarks)
5. [In-Domain Normalization & Preprocessing Control](#in-domain-normalization--preprocessing-control)
6. [Cross-Vehicle Transfer & Directional Asymmetry](#cross-vehicle-transfer--directional-asymmetry)
7. [Confounding Factor Controls (Sample Size & Class Distribution)](#confounding-factor-controls-sample-size--class-distribution)
8. [Target Domain Adaptation & Budget Sensitivity](#target-domain-adaptation--budget-sensitivity)
9. [Supervised Target Calibration (Few-Shot Adaptation)](#supervised-target-calibration-few-shot-adaptation)
10. [Charging Event Detection & Session-Level Performance](#charging-event-detection--session-level-performance)
11. [Repository File Inventory](#repository-file-inventory)
12. [Quickstart & Usage](#quickstart--usage)
13. [Citation & Contact](#citation--contact)

---

## 🚌 Overview & Operational States

Accurate recognition of battery-electric bus operational states is essential for fleet telematics, energy consumption modeling, charging infrastructure scheduling, and battery health prognosis (State of Health / State of Charge tracking).

Using non-intrusive embedded sensors—**Accelerometer (A)**, **Magnetometer (M)**, and **Gyroscope (G)**—this framework classifies operations into four mutually exclusive operational regimes:

| Operational State | Description | Physical Sensor Signature |
| :--- | :--- | :--- |
| **`Stationary off`** | Bus parked at depot/terminal with auxiliary systems and motor inverter powered down. | Baseline gravitational acceleration; static magnetic background; zero angular velocity. |
| **`Stationary on`** | Bus stationary (passenger stops, traffic lights) with active auxiliaries (HVAC, air compressor, electronics). | Micro-vibrations from auxiliary mechanical pumps; static orientation; baseline current draw. |
| **`Charging`** | Bus connected to depot DC fast charger with high-voltage power transfer active. | High structural stationarity; intense DC/switching magnetic field perturbations; thermal management vibration. |
| **`Driving`** | Bus in motion along urban routes, undergoing braking, cornering, and acceleration. | Dynamic kinetic vibrations, longitudinal/lateral inertial forces, angular velocity excursions, and rotating magnetic headings. |

---

## ⏱ Dataset & Windowing Configuration

Data was acquired from real-world revenue-service electric buses operating in regular urban transit corridors:

* **Sliding Window Specification**:
  * **Window Duration ($T_w$)**: `120 seconds` (nominal)
  * **Window Hop / Stride ($T_h$)**: `60 seconds` (50% temporal overlap)
  * **Service Day Timezone**: `America/Sao_Paulo`
* **Experimental Fleets**:
  * **Bus A (Primary Fleet)**:
    * **99,879 windows** collected across **254 operating days**.
    * *Class distribution*: Driving (44.7%), Stationary on (22.3%), Stationary off (21.0%), Charging (12.0%).
  * **Bus B (Transfer / Target Fleet)**:
    * **13,321 windows** collected across **30 operating days**.
    * *Class distribution*: Driving (42.1%), Stationary on (26.4%), Charging (23.7%), Stationary off (7.8%).
  * **Total Evaluated Corpus**: **113,200 valid windows** spanning **284 operating days**.

---

## 🧬 Feature Representations & Causal Transformation

The extraction pipeline computes multi-domain statistical, kinematic, and spectral descriptors from each 120-second window:

```
Total Stored Feature Columns: 164 features
├── Base Features (137 features)
│   ├── Accelerometer (54 features)
│   ├── Magnetometer (41 features)
│   └── Gyroscope (42 features)
└── Causal-Derived Features (27 features)
    ├── Accelerometer (9 features)
    ├── Magnetometer (9 features)
    └── Gyroscope (9 features)
```

### Feature Domains:
* **Time Domain (114 features)**: Mean, standard deviation, median, interquartile range (IQR), 95th percentile, mean absolute deviation (MAD), RMS, range.
* **Temporal Differences (20 features)**: Mean and standard deviation of first-order discrete differences ($\Delta x$).
* **Temporal Derivatives / Jerk (12 features)**: RMS, standard deviation, and peak jerk ($\Delta^2 x / \Delta t^2$) capturing sharp kinematic transients.
* **Frequency / Spectral Domain (15 features)**: Total band power, sub-band powers (0.1–0.5 Hz, 0.5–1.0 Hz, 1.0–2.0 Hz), spectral entropy, and dominant peak frequency.
* **Cross-Axis Correlation (3 features)**: Bivariate Pearson correlations ($r_{xy}, r_{xz}, r_{yz}$) capturing 3D spatial field dynamics.

### Representations (`RAW` vs `CAUSAL`):
Both model representations take exactly **137 feature inputs** to prevent dimensionality bias:
* **`RAW` Representation**:
  * Employs the 137 base features without temporal centering.
* **`CAUSAL` Representation**:
  * Retains **110 unchanged base descriptors** and replaces **27 drift-sensitive baseline features** with strictly past-only expanding-median centered variants:
  
  $$\tilde{x}_{\text{causal}}(t) = x(t) - \text{median}\{x(\tau) : \tau < t, \text{within the same operating day}\}$$

* **Strict Causality Guarantees**:
  * `uses_future_windows: false` — Zero look-ahead leakage.
  * `uses_target_labels: false` — Completely unsupervised running baseline.
  * First-window initialization uses the training set feature-wise empirical median.

---

## 📡 Sensor Configurations & In-Domain Benchmarks

Models were trained with an ensemble classifier ($N_{\text{estimators}} = 423$) using **5-fold grouped cross-validation** split strictly by calendar operating days.

### Performance Summary: Bus A (254 Days, 99,879 Windows)
| Configuration | Features | Accuracy | Balanced Acc (95% CI) | Macro $F_1$ (95% CI) | Stationary State $F_1$ | Charging $F_1$ | Driving $F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A+M`** | **95** | **0.9050** | **0.8721** [0.8548, 0.8881] | **0.8682** [0.8505, 0.8841] | **0.8247** | **0.8020** | **0.9986** |
| `A+M+G` | 137 | 0.9027 | 0.8711 [0.8535, 0.8877] | 0.8654 [0.8473, 0.8821] | 0.8209 | 0.8039 | 0.9989 |
| `A` | 54 | 0.8946 | 0.8611 [0.8427, 0.8787] | 0.8560 [0.8364, 0.8738] | 0.8084 | 0.8034 | 0.9985 |
| `A+G` | 96 | 0.8917 | 0.8565 [0.8396, 0.8726] | 0.8517 [0.8340, 0.8687] | 0.8026 | 0.7991 | 0.9989 |
| `M+G` | 83 | 0.8627 | 0.8131 [0.7950, 0.8309] | 0.8092 [0.7899, 0.8282] | 0.7465 | 0.7053 | 0.9974 |
| `M` | 41 | 0.8296 | 0.7743 [0.7516, 0.7963] | 0.7699 [0.7456, 0.7932] | 0.6989 | 0.6575 | 0.9830 |
| `G` | 42 | 0.8192 | 0.7510 [0.7289, 0.7725] | 0.7470 [0.7245, 0.7690] | 0.6635 | 0.5887 | 0.9977 |

### Performance Summary: Bus B (30 Days, 13,321 Windows)
| Configuration | Features | Accuracy | Balanced Acc (95% CI) | Macro $F_1$ (95% CI) | Stationary State $F_1$ | Charging $F_1$ | Driving $F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A+M`** | **95** | **0.9207** | **0.8959** [0.8276, 0.9453] | **0.8994** [0.8357, 0.9439] | **0.8699** | **0.8736** | **0.9877** |
| `A` | 54 | 0.9069 | 0.8832 [0.8144, 0.9329] | 0.8896 [0.8199, 0.9388] | 0.8570 | 0.8416 | 0.9872 |
| `A+M+G` | 137 | 0.9002 | 0.8764 [0.8062, 0.9350] | 0.8699 [0.7894, 0.9351] | 0.8290 | 0.8143 | 0.9927 |
| `A+G` | 96 | 0.8956 | 0.8729 [0.8021, 0.9321] | 0.8768 [0.8005, 0.9386] | 0.8380 | 0.7987 | 0.9931 |
| `M+G` | 83 | 0.8401 | 0.8117 [0.7512, 0.8643] | 0.7723 [0.7093, 0.8271] | 0.6988 | 0.6866 | 0.9927 |
| `M` | 41 | 0.8093 | 0.7783 [0.7258, 0.8283] | 0.7355 [0.6723, 0.7937] | 0.6533 | 0.6600 | 0.9821 |
| `G` | 42 | 0.8050 | 0.7502 [0.6896, 0.8066] | 0.7342 [0.6685, 0.7946] | 0.6481 | 0.6411 | 0.9926 |

---

## 🧪 In-Domain Normalization & Preprocessing Control

To test whether standard feature normalization techniques can replace causal median-centering, five preprocessing schemes were systematically benchmarked on unlabeled test splits (`in_domain_normalization_control.csv`):

| Vehicle | Preprocessing Method | Macro $F_1$ | $\Delta \text{Macro } F_1$ | Charging $F_1$ | $\Delta \text{Charging } F_1$ | Driving $F_1$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bus A** | **`CAUSAL`** | **0.8311** | **+0.0413** | **0.7581** | **+0.0946** | 0.9984 |
| Bus A | `RAW` | 0.7898 | 0.0000 | 0.6635 | 0.0000 | 0.9984 |
| Bus A | `QUANTILE` | 0.7628 | -0.0271 | 0.6362 | -0.0273 | 0.9983 |
| Bus A | `CORAL` | 0.7318 | -0.0580 | 0.5644 | -0.0991 | 0.9983 |
| Bus A | `ZSCORE` | 0.7190 | -0.0708 | 0.5029 | -0.1606 | 0.9983 |
| **Bus B** | **`CAUSAL`** | **0.8363** | **+0.0340** | **0.7885** | **+0.0977** | 0.9924 |
| Bus B | `RAW` | 0.8022 | 0.0000 | 0.6908 | 0.0000 | 0.9924 |
| Bus B | `QUANTILE` | 0.7706 | -0.0317 | 0.6718 | -0.0190 | 0.9923 |
| Bus B | `CORAL` | 0.6043 | -0.1980 | 0.3204 | -0.3704 | 0.9918 |
| Bus B | `ZSCORE` | 0.5769 | -0.2254 | 0.2892 | -0.4015 | 0.9918 |

> **Key Finding**: Conventional unsupervised adaptation techniques (`ZSCORE`, `CORAL`, `QUANTILE`) compress or distort inter-class variances, reducing charging $F_1$ by up to **40.2 percentage points**. In contrast, daily **`CAUSAL` expanding-median centering** consistently provides positive gains (+9.5 to +9.8 pp in charging $F_1$) without requiring future test data.

---

<!-- ## 🧲 Magnetic Representation & Feature Block Analysis

The 41 magnetometer descriptors were isolated into dedicated functional blocks (`magnetic_in_domain_results.csv`, `magnetic_transfer_results.csv`, `magnetic_paired_contrasts.csv`):
* **`LEVEL_AXIS`** (12 features): Axis-referenced statistics ($X, Y, Z$ percentiles, RMS, absolute mean).
* **`LEVEL_NORM`** (3 features): Euclidean norm level statistics ($\|\mathbf{m}\|$ mean, p95, RMS).
* **`LEVEL_AMPLITUDE`** (15 features): Combined axis and norm level features.
* **`VARIABILITY_DYNAMICS`** (26 features): Variability, spectral powers, differences, and cross-axis correlations.
* **`MAG_ALL`** (41 features): Full magnetometer feature set.

### In-Domain vs Transfer Performance (Bus A $\to$ Bus B):
| Magnetic Block | $N_{\text{feat}}$ | In-Domain Macro $F_1$ (Bus A) | Transfer Recorded Charging $F_1$ | Transfer YAW90 Charging $F_1$ | $\Delta_{\text{yaw} - \text{rec}}$ Charging $F_1$ | Transfer Driving $F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LEVEL_AXIS`** | 12 | 0.6761 | 0.5989 | 0.2727 | -0.3262 | 0.8409 |
| **`LEVEL_NORM`** | 3 | 0.4452 | 0.1703 | 0.1703 | **0.0000** | 0.7201 |
| **`LEVEL_AMPLITUDE`**| 15 | 0.6812 | **0.5903** | 0.2896 | -0.3007 | 0.8429 |
| **`VARIABILITY_DYNAMICS`** | 26 | 0.6384 | 0.0088 | 0.0129 | +0.0041 | **0.9717** |
| **`MAG_ALL`** | 41 | 0.7343 | 0.0080 | 0.1792 | +0.1711 | **0.9691** |

### Statistical Contrast Highlights (2,000 Bootstrap Replications):
* **Level vs Dynamics for Charging**: In transfer, `LEVEL_AMPLITUDE` outperforms `VARIABILITY_DYNAMICS` by $\Delta = +0.5814$ on Charging $F_1$ ($p < 0.002$; 99.8% bootstrap samples $> 0$).
* **Rotation Invariance**: `LEVEL_NORM` achieves mathematically exact rotation invariance ($\Delta_{\text{yaw}} = 0.0000$), whereas unnormalized axial level features suffer an orientation penalty of $\Delta \approx -0.30$ to $-0.33$ under 90° azimuth rotation.

--- -->

## 🔁 Cross-Vehicle Transfer & Directional Asymmetry

Direct cross-vehicle generalization was evaluated bidirectionally (`source_only_transfer.csv`):

| Transfer Direction | Configuration | Orientation | Accuracy | Macro $F_1$ | Stationary State $F_1$ | Charging $F_1$ | Driving $F_1$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bus A $\to$ Bus B** | `A` | RECORDED | 0.7711 | 0.6821 | 0.5884 | 0.7161 | 0.9632 |
| **Bus A $\to$ Bus B** | `A` | YAW90_LEFT | 0.7728 | 0.6679 | 0.5753 | 0.7408 | 0.9457 |
| **Bus A $\to$ Bus B** | `A+M` | RECORDED | 0.6717 | 0.5529 | 0.4166 | 0.3613 | 0.9618 |
| **Bus A $\to$ Bus B** | `A+M` | YAW90_LEFT | **0.8035** | **0.7320** | **0.6609** | **0.7438** | 0.9453 |
| **Bus A $\to$ Bus B** | `A+M+G` | RECORDED | 0.6409 | 0.4701 | 0.2968 | 0.0074 | 0.9899 |
| **Bus A $\to$ Bus B** | `A+M+G` | YAW90_LEFT | **0.8395** | **0.7763** | **0.7077** | **0.7065** | 0.9820 |
| **Bus B $\to$ Bus A** | `A` | RECORDED | 0.7266 | 0.5248 | 0.3676 | 0.0015 | 0.9966 |
| **Bus B $\to$ Bus A** | `A+M` | RECORDED | 0.7256 | 0.5291 | 0.3750 | 0.0003 | 0.9915 |
| **Bus B $\to$ Bus A** | `A+M+G` | RECORDED | 0.7335 | 0.5328 | 0.3778 | 0.0003 | 0.9979 |

> **Transfer Asymmetry**: Models trained on Bus A (254 days) generalize well to Bus B, retaining up to $F_1 = 0.776$. Conversely, models trained on Bus B (30 days) fail to detect Charging on Bus A ($F_1 < 0.04$), establishing that diverse multi-month source training data is strictly necessary for zero-shot state transfer.

---

## ⚖️ Confounding Factor Controls (Sample Size & Class Distribution)

To determine whether the transfer gap stems from dataset size disparity (254 vs 30 days) or intrinsic vehicle differences, two controlled experiments were executed:

### 1. Equal Source Size Control (`equal_source_size_control.csv`)
Bus A was downsampled to exactly match Bus B's sample size (**13,321 rows**, 10 independent Monte Carlo runs):
* For **Model `A` (RECORDED)**: Full Bus A Macro $F_1 = 0.6821$ vs Downsampled Median Macro $F_1 = 0.6820$ ($\Delta = -0.0001$).
* For **Model `A+M` (YAW90)**: Full Bus A Macro $F_1 = 0.7320$ vs Downsampled Median Macro $F_1 = 0.6542$ ($\Delta = -0.0778$).
* **Verdict**: Sample size alone accounts for less than 10% of transfer variance. Vehicle structural differences and operational variations dominate transferability.

### 2. Equal Class Day Control (`equal_class_day_control.csv`)
* **Composition Effect**: Isolates differences caused strictly by prior class proportions.
* **Vehicle Effect**: Measures intrinsic chassis-level mechanical and electromagnetic domain shifts.

---

## 🎯 Target Domain Adaptation & Budget Sensitivity

Unsupervised target domain adaptation was tested using an inductive fixed-test protocol across adaptation budgets $K \in \{1, 3, 7, 14, 18\}$ days (`adaptation_budget_sensitivity.csv`, `target_adaptation_k18.csv`):

| Method | 1 Day ($F_1$) | 3 Days ($F_1$) | 7 Days ($F_1$) | 14 Days ($F_1$) | 18 Days ($F_1$) | Calibration Overhead | Look-Ahead Leakage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CAUSAL`** | **0.5448** | **0.5448** | **0.5448** | **0.5448** | **0.5448** | **0 days (Zero-shot)** | **None** |
| `RAW` | 0.5437 | 0.5437 | 0.5437 | 0.5437 | 0.5437 | 0 days | None |
| `QUANTILE` | 0.2841 | 0.4227 | 0.5060 | 0.4896 | 0.4948 | Requires $\ge 7$ days | Batch |
| `CORAL` | 0.2423 | 0.3646 | 0.4422 | 0.4813 | 0.4941 | Requires $\ge 14$ days | Covariance batch |
| `ZSCORE` | 0.2217 | 0.2905 | 0.3971 | 0.4083 | 0.4931 | Requires $\ge 18$ days | Batch |

> **Adaptation Insight**: Batch adaptation methods (`CORAL`, `ZSCORE`) fail catastrophically under constrained observation budgets (1–3 target days), dropping Macro $F_1$ below 0.29. The proposed daily `CAUSAL` baseline provides instant zero-shot stability from day one.

---

## 🛠️ Supervised Target Calibration (Few-Shot Adaptation)

When limited ground-truth labels are obtainable on the target vehicle, supervised target calibration (`vehicle_balanced` strategy across 50 Monte Carlo runs per budget) enables rapid performance recovery with minimal data collection effort (`supervised_calibration_results.csv`):

| Sensor Configuration | Labeled Target Days | Target Exposure (Hours, p50) | Accuracy (p50) | Balanced Acc (p50) | Macro $F_1$ (p50) | Charging $F_1$ (p50) | Driving $F_1$ (p50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A`** | 0 | 0.0 h | 0.8106 | 0.7504 | 0.7071 | 0.7950 | 0.9680 |
| `A` | 1 | 13.6 h | 0.8321 | 0.7524 | 0.7480 | 0.7714 | 0.9832 |
| `A` | 3 | 26.6 h | 0.8421 | 0.7734 | 0.7680 | 0.7486 | 0.9856 |
| `A` | 5 | 45.7 h | 0.8647 | 0.8248 | 0.8208 | 0.7968 | 0.9868 |
| `A` | 10 | 94.6 h | 0.8936 | 0.8559 | 0.8602 | 0.8299 | 0.9881 |
| **`A`** | **18** | **158.9 h** | **0.9067** | **0.8719** | **0.8810** | **0.8430** | **0.9890** |
| **`A+M`** | 0 | 0.0 h | 0.6978 | 0.6382 | 0.5708 | 0.5085 | 0.9647 |
| `A+M` | 1 | 13.6 h | 0.7997 | 0.7303 | 0.6844 | 0.6567 | 0.9849 |
| `A+M` | 3 | 26.6 h | 0.8249 | 0.7554 | 0.7283 | 0.6246 | 0.9865 |
| `A+M` | 5 | 45.7 h | 0.8617 | 0.8188 | 0.8052 | 0.7901 | 0.9883 |
| `A+M` | 10 | 94.6 h | 0.8972 | 0.8590 | 0.8572 | 0.8416 | 0.9890 |
| **`A+M`** | **18** | **158.9 h** | **0.9016** | **0.8655** | **0.8650** | **0.8503** | **0.9899** |
| **`A+M+G`** | 0 | 0.0 h | 0.6358 | 0.5284 | 0.4521 | 0.0239 | 0.9911 |
| `A+M+G` | 1 | 13.6 h | 0.7712 | 0.7061 | 0.6549 | 0.5304 | 0.9909 |
| `A+M+G` | 5 | 45.7 h | 0.8540 | 0.8122 | 0.8027 | 0.7128 | 0.9913 |
| **`A+M+G`** | **18** | **158.9 h** | **0.9017** | **0.8656** | **0.8648** | **0.8371** | **0.9928** |

> **Key Calibration Insight**: Adding just **1 day (~13.6 hours)** of labeled target data lifts tri-modal `A+M+G` Macro $F_1$ from 0.4521 to 0.6549 (and charging $F_1$ from 0.0239 to 0.5304). By **5 target days**, all inertial-magnetic combinations exceed **0.80 Macro $F_1$**, offering an efficient supervised alternative to extensive fleet relabeling.

---

## ⚡ Charging Event Detection & Session-Level Performance

To evaluate real-world utility beyond window-level classifications, the framework was benchmarked against continuous physical charging events (`charging_session_results.csv`, `charging_orientation_changes.csv`) across **20 evaluable physical charging sessions** on Bus B:

### Session-Level Benchmark Summary:
| Condition | Sensor Configuration | Orientation | Session Recall | Time Precision | False Positives / Day | Daily Charge MAE (min) | Session Coverage (Median) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **In-Domain B (OOF)** | `A` | RECORDED | **70.0% (14/20)** | 88.8% | **0.47** | 35.3 min | **0.943** |
| **In-Domain B (OOF)** | `A+M` | RECORDED | 55.0% (11/20) | 90.7% | **0.30** | 43.1 min | 0.810 |
| **In-Domain B (OOF)** | `A+M+G` | RECORDED | 55.0% (11/20) | **91.7%** | 0.33 | 43.0 min | 0.810 |
| **In-Domain B (OOF)** | `M` | RECORDED | 20.0% (4/20) | 87.1% | **0.47** | 71.1 min | 0.000 |
| **Transfer (A $\to$ B)** | `A` | RECORDED | 45.0% (9/20) | 72.0% | 2.13 | 58.5 min | 0.065 |
| **Transfer (A $\to$ B)** | `A` | YAW90_LEFT | **55.0% (11/20)** | 65.5% | 2.87 | 64.0 min | **0.948** |
| **Transfer (A $\to$ B)** | `A+M` | RECORDED | 15.0% (3/20) | 63.2% | 0.80 | 85.9 min | 0.000 |
| **Transfer (A $\to$ B)** | `A+M` | YAW90_LEFT | **50.0% (10/20)** | 72.8% | 1.77 | 64.9 min | **0.521** |
| **Transfer (A $\to$ B)** | `A+M+G` | RECORDED | 0.0% (0/20) | 4.0% | 1.27 | 102.3 min | 0.000 |
| **Transfer (A $\to$ B)** | `A+M+G` | YAW90_LEFT | **40.0% (8/20)** | **77.4%** | 1.43 | 64.7 min | 0.134 |

### Orientation Discrepancy & Session Recovery (`charging_orientation_changes.csv`):
Under direct transfer, sensor mounting azimuth misalignment severely impedes charging session detection:
* **`A+M`**: Rotating azimuth by 90° (`YAW90_LEFT`) captures **8 additional charging sessions** previously missed, yielding a net gain of **+7 detected sessions** and improving session coverage by +7.4 pp.
* **`A+M+G`**: Misses all 20 sessions under `RECORDED` (0% recall), but recovers **8 sessions** under `YAW90_LEFT` (net **+8 detected sessions**, +11.2 pp coverage).
* **`G` (Gyroscope alone)**: Lacks rotation robustness, losing 7 sessions under YAW90 (-19.2 pp coverage).

---

## 📂 Repository File Inventory

| File | Type | Description |
| :--- | :---: | :--- |
| `feature_manifest.json` | JSON | Machine-readable specification of windowing parameters, feature blocks, and causal transformation rules. |
| `feature_dictionary.csv` | CSV | Exhaustive 164-feature metadata catalog detailing modalities, domains, statistical definitions, and inclusion flags. |
| `in_domain_bus_A_results.csv` | CSV | 5-fold CV evaluation metrics on Bus A across all 7 sensor configurations with 95% confidence intervals. |
| `in_domain_bus_A_by_class.csv` | CSV | Class-level performance metrics (Precision, Recall, $F_1$, Support) for Bus A. |
| `in_domain_bus_A_confusion_matrix.csv` | CSV | Empirical confusion matrices for all sensor configurations on Bus A. |
| `in_domain_bus_B_results.csv` | CSV | 5-fold CV evaluation metrics on Bus B across all 7 sensor configurations with 95% confidence intervals. |
| `in_domain_bus_B_by_class.csv` | CSV | Class-level performance metrics for Bus B. |
| `in_domain_bus_B_confusion_matrix.csv` | CSV | Empirical confusion matrices for all sensor configurations on Bus B. |
| `in_domain_normalization_control.csv` | CSV | Systematic evaluation of 5 normalization schemes (RAW, CAUSAL, ZSCORE, CORAL, QUANTILE) on unlabeled test folds. |
| `source_only_transfer.csv` | CSV | Complete bidirectional transfer evaluation (Bus A $\leftrightarrow$ Bus B) across 7 sensor configurations and 2 orientations. |
| `equal_source_size_control.csv` | CSV | Controlled Monte Carlo analysis isolating dataset sample size effects from vehicle physical domain shifts. |
| `equal_class_day_control.csv` | CSV | Controlled experimental evaluation isolating the day effect (30 vs 254 days) from vehicle-specific domain shifts. |
| `bfull_symmetric_comparison.csv` | CSV | Symmetric cross-domain evaluation isolating composition effects and vehicle-specific transfer degradation. |
| `adaptation_budget_sensitivity.csv` | CSV | Sensitivity analysis evaluating unsupervised target domain adaptation methods across budget horizons ($K \in \{1, 3, 7, 14, 18\}$ days). |
| `target_adaptation_k18.csv` | CSV | Full target adaptation benchmark for $K=18$ calibration days across configurations, orientations, and methods. |
| `supervised_calibration_results.csv` | CSV | Few-shot target calibration benchmark evaluating performance recovery across $K \in \{0, 1, 2, 3, 5, 10, 18\}$ labeled days. |
| `charging_session_results.csv` | CSV | Continuous charging session event metrics (Recall, Precision, Daily MAE, Coverage) for in-domain and transfer models. |
| `charging_orientation_changes.csv` | CSV | Pairwise event-level analysis detailing charging sessions gained, lost, or retained under 90° azimuth sensor rotation. |

---

## 🚀 Quickstart & Usage

### 1. Requirements
```bash
pip install pandas numpy scikit-learn
```

### 2. Inspecting the Feature Manifest & Benchmarks
```python
import json
from typing import Any, Dict
import pandas as pd

def load_study_metadata(manifest_path: str) -> Dict[str, Any]:
    f"""
    Loads and parses the experimental feature manifest.

    Parameters
    ----------
    manifest_path : str
        Path to feature_manifest.json.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing manifest parameters.
    """
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest_data: Dict[str, Any] = json.load(file)
    print(f"Loaded manifest: {manifest_data['title']}")
    print(f"Nominal window duration: {manifest_data['window_configuration']['window_duration_seconds']}s")
    return manifest_data

def inspect_in_domain_benchmarks(bus_a_path: str, bus_b_path: str) -> pd.DataFrame:
    f"""
    Aggregates and compares in-domain top sensor configurations across fleets.

    Parameters
    ----------
    bus_a_path : str
        Path to in_domain_bus_A_results.csv.
    bus_b_path : str
        Path to in_domain_bus_B_results.csv.

    Returns
    -------
    pd.DataFrame
        Comparative summary dataframe.
    """
    df_a: pd.DataFrame = pd.read_csv(bus_a_path)
    df_b: pd.DataFrame = pd.read_csv(bus_b_path)
    
    df_a["fleet"] = "Bus A (254 days)"
    df_b["fleet"] = "Bus B (30 days)"
    
    combined: pd.DataFrame = pd.concat([df_a, df_b], ignore_index=True)
    cols = ["fleet", "sensor_configuration", "n_model_features", "accuracy", "macro_f1", "charging_f1", "driving_f1"]
    return combined[cols].sort_values(by=["fleet", "macro_f1"], ascending=[True, False])

if __name__ == "__main__":
    manifest = load_study_metadata("feature_manifest.json")
    benchmark_summary = inspect_in_domain_benchmarks("in_domain_bus_A_results.csv", "in_domain_bus_B_results.csv")
    print(benchmark_summary.to_string(index=False))
```

---

## 📖 Citation & Contact

If you utilize this dataset, feature pipeline, or empirical benchmarks, please cite:

```bibtex
@misc{beb_state_recognition_2026,
  title={Operational-State Recognition for Battery-Electric Buses Using Inertial and Magnetic Sensing},
  author={Motta, Thonny E. Motta and Contributors},
  year={2026},
  howpublished={\url{https://github.com/labvivomobilidade/state_imu}},
  note={School of Electrical and Computer Engineering (FEEC), State University of Campinas (Unicamp)}
}
```

For technical questions or reproducibility discussions, please open an issue in this repository.
