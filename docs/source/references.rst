References and Credits
======================

CausalBoundingEngine integrates several state-of-the-art algorithms and methods for causal effect bounding. We gratefully acknowledge the authors and contributors of the following works and implementations.

Algorithm References
--------------------

Autobound
~~~~~~~~~

**Authors**: Duarte, G., Finkelstein, N., Knox, D., Mummolo, J., & Shpitser, I.

**Reference**: Duarte, G., Finkelstein, N., Knox, D., Mummolo, J., & Shpitser, I. (2023). An Automated Approach to Causal Inference in Discrete Settings. *Journal of the American Statistical Association*, 1-12.

**DOI**: https://doi.org/10.1080/01621459.2023.2216909

**Description**: Autobound provides an automated approach to causal inference in discrete settings using linear programming optimization. The algorithm can handle complex causal graphs and provides tight bounds through systematic optimization.

Causaloptim
~~~~~~~~~~~

**Authors**: Sachs, M. C., Sjölander, A., & Gabriel, E. E.

**Repository**: https://github.com/sachsmc/causaloptim

**Reference**: Sachs, M. C., Sjölander, A., & Gabriel, E. E. (2022). A General Method for Deriving Tight Symbolic Bounds on Causal Effects. *Journal of Computational and Graphical Statistics*, 31(2), 496-510.

**Description**: Causaloptim uses symbolic computation to derive bounds on causal effects. It provides a general framework for bounding causal quantities in directed acyclic graphs with unmeasured confounders.

Entropy Bounds
~~~~~~~~~~~~~~

**Authors**: Jiang, Z., & Shpitser, I.

**Repository**: https://github.com/ziwei-jiang/Approximate-Causal-Effect-Identification-under-Weak-Confounding

**Reference**: Jiang, Z., & Shpitser, I. (2020). Approximate Causal Effect Identification under Weak Confounding. *Proceedings of the 37th International Conference on Machine Learning*, 4740-4750.

**Description**: Provides bounds on causal effects under weak confounding assumptions using entropy-based constraints. The method allows for partial identification when traditional approaches would fail.

Manski Bounds
~~~~~~~~~~~~~

**Author**: Manski, C. F.

**Reference**: Manski, C. F. (1990). Nonparametric Bounds on Treatment Effects. *The American Economic Review*, 80(2), 319-323.

**DOI**: https://www.jstor.org/stable/2006592

**Description**: Classical nonparametric bounds on treatment effects under minimal assumptions. These provide the most conservative bounds but require no additional assumptions beyond the observed data.

Tian-Pearl Bounds
~~~~~~~~~~~~~~~~~

**Authors**: Tian, J., & Pearl, J.

**Reference**: Tian, J., & Pearl, J. (2000). Probabilities of Causation: Bounds and Identification. *Annals of Mathematics and Artificial Intelligence*, 28(1-4), 287-313.

**Technical Report**: https://ftp.cs.ucla.edu/pub/stat_ser/r271-A.pdf

**Description**: Provides bounds on probabilities of causation, including the probability of necessity and sufficiency (PNS). The method extends classical bounds to handle more complex causal queries.

Zaffalon Bounds
~~~~~~~~~~~~~~~

**Authors**: Zaffalon, M., Antonucci, A., Cabañas, R., Huber, D., & Azzimonti, D.

**Dependencies**:
   - **CREMA**: https://github.com/IDSIA/crema
   - **CREDICI**: https://github.com/IDSIA/credici

**Reference**: Zaffalon, M., Antonucci, A., Cabañas, R., Huber, D., & Azzimonti, D. (2022). Bounding Counterfactuals under Selection Bias. *Proceedings of The 11th International Conference on Probabilistic Graphical Models*, 289-300.

**URL**: https://proceedings.mlr.press/v186/zaffalon22a/zaffalon22a.pdf

**Description**: Uses causal EM to compute bounds on causal effects. The implementation leverages Java libraries CREMA and CREDICI for probabilistic inference with imprecise probabilities.

Zhang-Bareinboim Bounds
~~~~~~~~~~~~~~~~~~~~~~~

**Authors**: Zhang, J., & Bareinboim, E.

**Reference**: Zhang, J., & Bareinboim, E. (2021). Bounding Causal Effects on Continuous Outcome. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(13), 12207-12215.

**Technical Report**: https://causalai.net/r61.pdf

**Description**: Provides bounds for causal effects in instrumental variable settings with continuous variables. The method handles compliance types and partial identification in IV designs.

Software Dependencies
---------------------

R Integration
~~~~~~~~~~~~~

Several algorithms require R integration through the `rpy2` package:

- **Causaloptim**: Requires R package `causaloptim`
- **R Statistical Computing**: https://www.r-project.org/

Java Integration
~~~~~~~~~~~~~~~~

Zaffalon bounds require Java libraries:

- **CREMA (CREdo Models Arithmetic)**: Credal network library
- **CREDICI**: Causal inference with credal networks
- **Institution**: Dalle Molle Institute for Artificial Intelligence (IDSIA), Switzerland

Python Libraries
~~~~~~~~~~~~~~~~~

Core dependencies:

- **NumPy**: Fundamental package for scientific computing
- **Pandas**: Data manipulation and analysis
- **SciPy**: Scientific computing library

Citation Guidelines
-------------------

If you use CausalBoundingEngine in your research, please cite the relevant algorithm papers based on which methods you employ:

For Autobound
~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{duarte2023automated,
     title={An Automated Approach to Causal Inference in Discrete Settings},
     author={Duarte, Guilherme and Finkelstein, Noam and Knox, Dean and Mummolo, Jonathan and Shpitser, Ilya},
     journal={Journal of the American Statistical Association},
     pages={1--12},
     year={2023},
     publisher={Taylor \& Francis}
   }

For Causaloptim
~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{sachs2022general,
     title={A General Method for Deriving Tight Symbolic Bounds on Causal Effects},
     author={Sachs, Michael C and Sj{\"o}lander, Arvid and Gabriel, Erin E},
     journal={Journal of Computational and Graphical Statistics},
     volume={31},
     number={2},
     pages={496--510},
     year={2022},
     publisher={Taylor \& Francis}
   }

For Entropy Bounds
~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @inproceedings{jiang2020approximate,
     title={Approximate Causal Effect Identification under Weak Confounding},
     author={Jiang, Ziwei and Shpitser, Ilya},
     booktitle={Proceedings of the 37th International Conference on Machine Learning},
     pages={4740--4750},
     year={2020}
   }

For Manski Bounds
~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{manski1990nonparametric,
     title={Nonparametric Bounds on Treatment Effects},
     author={Manski, Charles F},
     journal={The American Economic Review},
     volume={80},
     number={2},
     pages={319--323},
     year={1990},
     publisher={JSTOR}
   }

For Tian-Pearl Bounds
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{tian2000probabilities,
     title={Probabilities of Causation: Bounds and Identification},
     author={Tian, Jin and Pearl, Judea},
     journal={Annals of Mathematics and Artificial Intelligence},
     volume={28},
     number={1-4},
     pages={287--313},
     year={2000},
     publisher={Springer}
   }

For Zaffalon Bounds
~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @inproceedings{zaffalon2022bounding,
     title={Bounding Counterfactuals under Selection Bias},
     author={Zaffalon, Marco and Antonucci, Alessandro and Caba{\~n}as, Rafael and Huber, Denis and Azzimonti, Dario},
     booktitle={Proceedings of The 11th International Conference on Probabilistic Graphical Models},
     pages={289--300},
     year={2022},
     organization={PMLR},
     editors={Salmer{\'o}n, Antonio and Rum{\'\i}, Rafael},
     url={https://proceedings.mlr.press/v186/zaffalon22a/zaffalon22a.pdf}
   }

For Zhang-Bareinboim Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @inproceedings{zhang2021bounding,
     title={Bounding Causal Effects on Continuous Outcome},
     author={Zhang, Junzhe and Bareinboim, Elias},
     booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
     volume={35},
     number={13},
     pages={12207--12215},
     year={2021},
     month={May}
   }

Acknowledgments
---------------

We thank all the authors and contributors of the algorithms integrated into CausalBoundingEngine. Their groundbreaking work in causal inference has made this unified framework possible.

Special thanks to:

- The **UCLA Causality Lab** for foundational work in causal inference
- The **IDSIA** research institute for credal network implementations
- The **R Core Team** and package maintainers for statistical computing infrastructure
- The **Python scientific computing community** for essential libraries

License Compatibility
---------------------

All integrated algorithms and dependencies are used in accordance with their respective licenses. Users should ensure compliance with individual algorithm licenses when using CausalBoundingEngine in their projects.

For specific license information, please refer to:

- Individual algorithm repositories
- R package documentation
- Java library licenses (CREMA, CREDICI)
- Python package licenses

Contributing
------------

If you are an author of an algorithm used in CausalBoundingEngine and would like to update the citation information or add additional references, please submit a pull request or contact the maintainers.

For adding new algorithms, please include proper citation information and ensure all dependencies are clearly documented.
