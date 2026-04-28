# The Vision Basin: Cross-Modal Throughput Measurement Reveals Modality-Specific Information Extraction Rates

Grant Lavell Whitmer III

The Windstorm Institute, Fort Ann, NY 12827, USA

Email: grantwhitmer3@gmail.com

ORCID: 0009-0007-3224-755X

---

## Abstract

Paper 7 demonstrated that the language-model throughput basin reflects training-data entropy minus exploitable hierarchical structure: BPT is approximately source entropy minus f(structural depth). If this equation is general, different modalities should sit at different throughput levels. We test this prediction across three modalities -- language, vision, and audio -- using twelve pretrained models, from-scratch training, source-entropy measurement via lossless compression, and structural-destruction cascades. Each modality has its own characteristic throughput: language at 0.85--1.30 bits per source byte, generative vision at 1.33--1.36 bits per pixel, and real speech at 1.89 bits per mel-spectrogram dimension. A visual shuffling cascade reveals a spatial structural bonus of 0.69 bits per pixel. Music extracts 2.69 bits per mel dimension. Bits per pixel is not patch-size-invariant and visual source entropy is resolution-dependent. A from-scratch verification round (v2.1) trains a 112-million-parameter ViT-MAE at seven controlled-entropy image distributions and confirms the equation at the architectural level. Version 2.2 (April 2026) adds a methodological-journey narrative (§1.3) tracing the seven rounds of follow-up and an adversarial-review-defense table (§4.4); Zenodo 10.5281/zenodo.19672828.

**Keywords:** vision basin; audio throughput; cross-modal comparison; masked autoencoder; source entropy; structural bonus

---

## 1. Introduction

Paper 7 established the language-model throughput basin is data-driven [7]. This paper tests whether the governing equation generalizes to vision and audio. The full methodology, cross-modal comparisons with 95 percent confidence intervals, the v2.1 from-scratch verification, and the v2.2 methodological-journey + adversarial-review-defense framework are in paper.pdf (current version v2.2).

*This is the Royal Society Interface submission draft. See also: paper-aap-draft.md, paper-arxiv.tex, paper-entropy-draft.md, paper.pdf.*
