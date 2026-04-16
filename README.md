This repository constructs PII-related evaluation data and studies memorization behaviors of large language models, including **verbatim**, **associative**, **extractable memorization**, and **membership inference attacks**.

## Memorization Evaluation

### A. Verbatim memorization

Evaluate whether the model reproduces PII verbatim and compute target log-likelihood.

Run:

```
python verbatim_mem.py
```

------

### B. Associative memorization

Probe whether the model can infer PII from related attributes (e.g., name → email).

Run:

```
python asso_mem.py
```

This includes:

- Generation-based hit evaluation
- Target log-likelihood analysis
- Optional language-specific prompt templates

please check the template at templates/

------

### C. Extractable memorization

Assess whether PII can be systematically extracted from model outputs.

Run:

```
python extractable_mem.py
```
please check the template at templates/
------

## Membership Inference Attack

Membership inference experiments are conducted using the **mimir** framework.

Repository:

```
https://github.com/iamgroot42/mimir
```

Clone into this project:

```
git clone https://github.com/iamgroot42/mimir mimir
```

Refer to mimir’s documentation for configuration and execution details.
------
> **Ethical Note**  
> To minimize potential privacy risks, we do not publicly release the specific implementation used to parse PII-related data. A detailed description of the full pipeline is provided in the appendix of the paper.
