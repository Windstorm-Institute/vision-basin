# Entropy (MDPI) Submission Draft

# The Vision Basin: Cross-Modal Throughput Measurement Reveals Modality-Specific Information Extraction Rates

**Grant Lavell Whitmer III**

The Windstorm Institute, Fort Ann, NY 12827, USA; grantwhitmer3@gmail.com

---

## Abstract

Paper 7 established that the language-model throughput basin reflects training-data entropy minus exploitable hierarchical structure: BPT approximately equals source entropy minus f(structural depth). If this equation is general, different modalities should sit at different throughput levels. We test this prediction across three modalities - language, vision, and audio - using twelve pretrained models, from-scratch training, source-entropy measurement via lossless compression, and structural-destruction cascades. Each modality has its own characteristic throughput: language at 0.85-1.30 bits per source byte, generative vision at 1.33-1.36 bits per pixel, and real speech at 1.89 bits per mel-spectrogram dimension. A visual shuffling cascade reveals a spatial structural bonus of 0.69 bits per pixel. Music extracts 2.69 bits per mel dimension. Bits per pixel is not patch-size-invariant; visual source entropy is resolution-dependent. Version 2.1 adds a 112-million-parameter ViT-MAE trained from scratch at seven controlled-entropy image distributions across three random seeds, confirming the equation at the architectural level (Welch t = 249,994, p = 1.6e-11, Cohen's d = 204,119).

**Keywords:** vision basin; audio throughput; cross-modal comparison; masked autoencoder; shuffling cascade; patch-size invariance; source entropy; structural bonus; bits per source unit

---

## 1. Introduction

The first six papers in this series (Whitmer 2026a-f) established the throughput basin for language models. Paper 7 (Whitmer 2026g) demonstrated that the basin is data-driven and governed by the equation BPT approximately equals source\_entropy minus f(structural\_depth). Paper 8 tests whether this equation generalizes to vision and audio.

The full methods, results tables with 95 percent confidence intervals, cross-modal comparisons, and the v2.1 from-scratch verification round are in the accompanying paper.pdf in this repository.

---

## References

Whitmer III, G.L. (2026a-g). Papers 1-7, Windstorm Institute.

*This is the MDPI Entropy submission draft. See also: paper-aap-draft.md (AAP), paper-arxiv.tex (arXiv), paper-rsif-draft.md (Royal Society Interface), paper.pdf (published).*
