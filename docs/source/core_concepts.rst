Core Concepts
==========

This guide provides a comprehensive overview of CausalBoundingEngine's concepts, scenarios, and usage patterns.

Causal Bounding
-------------

Causal inference often faces the challenge of unmeasured confounding - variables that affect both treatment and outcome but are not observed. When identification of causal effects is impossible, **causal bounding** provides a principled approach to determine the range of possible causal effects compatible with the observed data and assumptions.

CausalBoundingEngine focuses on two key causal quantities:

**Average Treatment Effect (ATE)**:
   The difference in expected outcomes between treated and untreated states:
   
   .. math:: ATE = E[Y(1)] - E[Y(0)]

**Probability of Necessity and Sufficiency (PNS)**:
   The probability that treatment is both necessary and sufficient for a positive outcome:
   
   .. math:: PNS = P(Y(1)=1, Y(0)=0)

Scenarios
---------

CausalBoundingEngine organizes algorithms by **scenarios** - different causal settings that determine which algorithms are applicable.
