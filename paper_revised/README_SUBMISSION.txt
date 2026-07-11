Submission source package for The Journal of Supercomputing

The archive is deliberately flat: do not move any file into a subdirectory.

Compile sequence:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

Required local files:
  main.tex
  revised_refs.bib
  main.bbl
  sn-jnl.cls
  sn-basic.bst
  fig_kernel_scaling.pdf
  fig_system_scaling.pdf
  fig_svd_pareto.pdf
  fig_tradeoff.pdf
  fig_leakage.pdf
  fig_beir_utility.pdf

The compiled manuscript is supplied separately as
kurilenko_jsc_revised.pdf. The data/code reproducibility artifact is separate
from this publisher source archive.
