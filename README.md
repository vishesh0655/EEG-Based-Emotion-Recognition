# EEG-Based Emotion Recognition Using Machine Learning

A machine learning-based project for recognizing emotional states from EEG (Electroencephalogram) signals using the DREAMER dataset.

## 📌 Project Overview

Emotion recognition from EEG signals is an important research area in Brain-Computer Interfaces (BCI), affective computing, and human-computer interaction.

This project focuses on extracting meaningful information from EEG signals and using machine learning algorithms to classify emotional states. The project uses the DREAMER dataset and applies signal preprocessing, windowing, frequency-domain feature extraction, feature engineering, normalization, and machine learning for emotion recognition.

The main objective is to investigate how effectively traditional machine learning and ensemble-based models can recognize emotional states from engineered EEG features.

## 🎯 Objectives

- Study EEG signals and their relationship with emotional states.
- Preprocess EEG recordings from the DREAMER dataset.
- Divide EEG recordings into consistent time windows.
- Extract frequency-domain features from EEG signals.
- Analyze Theta, Alpha, Beta, and Gamma frequency bands.
- Perform feature engineering to improve the representation of EEG data.
- Apply subject-level and trial-level normalization.
- Train and compare multiple machine learning models.
- Evaluate models using multiple performance metrics.
- Analyze model predictions using confusion matrices.
- Identify the most effective machine learning approach for the prepared EEG features.

## 📊 Dataset

The project uses the **DREAMER dataset**, which contains EEG recordings collected during emotional stimulation experiments.

### EEG Configuration

- Sampling frequency: **128 Hz**
- EEG channels: **14**
- Window size: **1 second**
- Samples per window: **128**
- Frequency bands:
  - Theta
  - Alpha
  - Beta
  - Gamma

The original DREAMER data was processed and converted into a structured format suitable for Python-based analysis and machine learning.

> **Dataset Notice:** The DREAMER dataset and any processed copies of the dataset are not included or redistributed in this repository. Users who wish to reproduce the experiments should obtain the dataset through the original source and comply with the applicable access and usage conditions.

## ⚙️ Methodology

The project consists of the following major stages:

### 1. EEG Data Preparation

The DREAMER dataset was examined and the required EEG recordings and emotional labels were extracted and organized into a structured format.

### 2. EEG Windowing

The EEG recordings were divided into **1-second windows**.

With a sampling frequency of 128 Hz:

```text
1 second = 128 EEG samples
