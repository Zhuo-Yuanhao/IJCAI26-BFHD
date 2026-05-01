# BFHD: Bidirectional Feature Harmonization Decomposition

This repository provides code and supplementary technical materials for our IJCAI paper:

**BFHD: Bidirectional Feature Harmonization Decomposition for Heterogeneous Clinical Assessments**

BFHD formulates clinical assessment harmonization as a bidirectional recoverability problem. Given paired observations from two heterogeneous assessment systems, the method identifies measurements that can be reliably translated in both directions under an application-defined feasibility tolerance, while separating non-translatable components.

## Repository status

This repository is under active preparation for the camera-ready release.

At the current stage, we provide the key implementation components required to reproduce the core BFHD mechanism, including:

- bidirectional feature harmonization with output-level gating;
- feasibility-based recoverable measurement selection.

Additional experiment scripts, data-processing utilities, and extended analysis code will be released progressively where permitted.

## Important note on data

This repository does **not** redistribute any ADNI data, subject-level records, derived ADNI features, or other access-controlled clinical data.

Experiments involving the Alzheimer's Disease Neuroimaging Initiative (ADNI) require users to obtain access through the official ADNI data access process and comply with the ADNI Data Use Agreement. Researchers who wish to reproduce ADNI-based experiments should apply for ADNI access independently and run the provided code on their locally authorized data.

For non-public clinical datasets used in the paper, access is subject to the corresponding data-use agreements and institutional approvals.

## Supplementary technical material

Additional implementation details, hyperparameter settings, experimental protocols, and extended results are provided in:

- `supplementary.pdf`  *(to be added / updated)*

The supplementary material is provided as additional technical documentation for reproducibility. It is not an official IJCAI proceedings appendix unless explicitly allowed by the conference.

## Code structure

```text
BFHD/
  README.md
  src/
    bfhd/
      ...
  configs/
    ...
  supplementary.pdf
  requirements.txt
```

The initial release focuses on the core BFHD components. Some project-specific training and data-processing code depends on access-controlled datasets or external project infrastructure and is therefore not included in this initial release.

## Usage
Example usage will be added as the repository is finalized.

The intended workflow is:

1. obtain the relevant dataset through the appropriate official data access process;
2. preprocess the data according to the protocol described in the paper and supplementary material;
3. run the BFHD training and feature-recoverability selection code on the locally authorized data.

## Citation

Citation information will be added once the official proceedings version is available.

If you use this repository before the official citation is available, please cite the paper as:

**BFHD: Bidirectional Feature Harmonization Decomposition for Heterogeneous Clinical Assessments**  
Yuanhao Zhuo, Zixi Qin, Ling Qin, and Wanqing Li  
Accepted to the 35th International Joint Conference on Artificial Intelligence (IJCAI 2026).

## Contact
For questions about the code or implementation, please open an issue in this repository.

