# ABiMap

BiMap layer for SPDNet-like networks that learns its output dimension 
jointly with the filter matrix via backpropagation. Two parallel BiMap 
projections of adjacent dimensions are interpolated along a log-Euclidean 
geodesic, with a learnable parameter driving expand and shrink transitions 
until the optimal subspace dimension is identified automatically.

Yacine Meftah, Marco Congedo, Laurent Bougrain  
Université de Lorraine, CNRS, LORIA — GIPSA-lab, Université Grenoble Alpes
