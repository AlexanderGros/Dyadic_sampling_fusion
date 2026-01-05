This repository investigates the possibility to improve Automatic Modulation Classification (AMC) by using Dyadic filtering of IQ signals in combination with AI.

For additional information see the Poster : Poster_MdC_AG.pdf

The paper for citation can be found here: https://ieeexplore.ieee.org/document/11193926

The trained weights are too big for github, please contact me if you need them or want to collaborate.

The folders are not very well structured for the moment (lack of time), if you have any questions please ask.

The initial code can be found under: dyadic_s_fusion.py

The used Dataset is the Machine-Learning Challenge [CSPB.ML.2018] of Chad Spooners cyclostationary.blog.
The Dataset is originally composed of matlab binary .tim files, these have been aggregated and converted to the more common .h5 file called spooner_full.h5 in the code.

@misc{cyclo2,
author = {Spooner, Chad},
  title = {Dataset for the Machine-Learning Challenge [CSPB.ML.2018]},
  howpublished = {https://cyclostationary.blog/2019/02/15/data-set-for-the-machine-learning-challenge/},
  note = {Accessed: 10.02.2024}
}
