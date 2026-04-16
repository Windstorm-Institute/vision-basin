# American Academic Publisher Draft

# The Vision Basin: Cross-Modal Throughput Measurement Reveals Modality-Specific Information Extraction Rates

**Grant Lavell Whitmer III**

The Windstorm Institute, Fort Ann, New York 12827, United States of America

Email: grantwhitmer3@gmail.com (Corresponding Author)

---

## Abstract

Paper 7 established that the language-model throughput basin reflects training-data entropy minus exploitable hierarchical structure: BPT is approximately source\_entropy minus f(structural\_depth). If this equation is general, different modalities should sit at different throughput levels. This paper tests that prediction across three modalities - language, vision, and audio - using twelve pretrained models, from-scratch training, source-entropy measurement via lossless compression, and structural-destruction cascades. Each modality has its own characteristic throughput: language at 0.85 to 1.30 bits per source byte (nine models, two corpora), generative vision at 1.33 to 1.36 bits per pixel (MAE reconstruction), and real speech at 1.89 bits per mel-spectrogram dimension (LJ Speech, 3,000 utterances, three seeds). A visual shuffling cascade on STL-10 reveals a spatial structural bonus of 0.69 bits per pixel. Music extracts 2.69 bits per mel dimension, substantially more than speech. Two methodological findings: bits per pixel is not patch-size-invariant (varies 4x across patch sizes 8x8 to 48x48), and visual source entropy is resolution-dependent. Version 2.1 adds a from-scratch verification round: a 112-million-parameter ViT-MAE trained at each of seven controlled-entropy image distributions across three random seeds confirms the equation at the architectural level. Welch's t-test for the uniform-noise versus uniform-color comparison: t = 249,994, p = 1.6e-11, Cohen's d = 204,119. This is the eighth paper in the Windstorm series and the first to generalize the throughput basin across sensory modalities.

**Keywords:** vision basin, audio throughput, cross-modal comparison, masked autoencoder, shuffling cascade, patch-size invariance, LJ Speech, source entropy, structural bonus, bits per source unit, from-scratch verification

---

## 1. Introduction

Paper 7 demonstrated that language models converge on approximately 4 bits per token (approximately 1 bit per source byte) because that is the compressible entropy of human language, not because of any architectural or thermodynamic constraint. The equation BPT is approximately source\_entropy minus f(structural\_depth) was confirmed for text across entropy levels 5 through 8 bits, at 92 million and 1.2 billion parameters, and under three different loss functions.

The natural prediction: if the basin is genuinely data-driven, then vision and audio - modalities with fundamentally different statistical structure - should sit at different throughput levels. This paper tests that prediction.

The full experimental methodology, results tables with 95 percent confidence intervals, and the v2.1 from-scratch verification round are detailed in the accompanying paper.pdf and in the Grand Slam Supplementary Materials PDF in this repository.

---

## References

See paper.pdf for full reference list. All code and data: github.com/Windstorm-Institute/vision-basin.

*This is the American Academic Publisher submission draft. For the ArXiv LaTeX version, see paper-arxiv.tex. For the MDPI Entropy version, see paper-entropy-draft.md. For the Royal Society Interface version, see paper-rsif-draft.md. For the full published PDF, see paper.pdf.*
