# Detecting Hallucinations in LLMs using Semantic Entropy

This project implements techniques from the 2024 *Nature* paper **"Detecting Hallucinations in Large Language Models using Semantic Entropy"**, presented at AJCAI 2024 in Melbourne, Australia. The approach leverages entropy-based uncertainty measures to identify hallucinations—instances where large language models (LLMs) generate unfaithful or unsupported content.

## 🚀 Objective

Detect hallucinations in LLM outputs to improve the safety and reliability of AI-generated information, with a focus on applications such as question answering.

## System Overview

* **Model:** DeBERTa-v3-xsmall fine-tuned for natural language inference (NLI), used to semantically cluster and compare generated responses.
* **Framework:** PyTorch and Hugging Face Transformers for model loading, tokenization, and inference.
* **Hardware:** Experiments run on GPUs — specifically, NVIDIA T4 GPUs (2x) for efficient computation and acceleration.

## 📚 Dataset

Experiments are conducted on **SQuAD 2.0**, a challenging reading comprehension dataset containing both answerable and unanswerable questions derived from Wikipedia. The validation set includes:

* Questions
* Context passages
* Multiple generated responses per question
* Reference answers annotated for correctness and answerability

This setup tests the system’s ability to detect hallucinations in complex QA scenarios and generalizes to other QA datasets such as **Natural Questions** and **TriviaQA**.

## Methods

### Uncertainty Estimation via Entropy

* **Cluster Assignment Entropy:** Measures uncertainty based on the entropy of semantic cluster assignment frequencies.
* **Predictive Entropy (MC Estimate):** Calculates entropy from averaged token log-likelihoods across generated samples.
* **Semantic Entropy (Rao-Blackwellized):** Aggregates token likelihoods within semantic clusters using log-sum-exp before computing entropy, yielding a refined uncertainty estimate.

### LogSumExp Aggregation

* Efficiently sums log-likelihoods of responses sharing the same semantic cluster to normalize probabilities in log space.

## Evaluation

* ROC curves and AUC scores evaluate the effectiveness of entropy-based methods in hallucination detection.
* The approach reliably distinguishes hallucinated, true, and neutral responses to identify unfaithful model generations.

---
