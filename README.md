# ABiMap

BiMap layer that learns its output dimension $m$ jointly with the weight matrix during training. 

Authors:

Yacine Meftah¹, Marco Congedo², Laurent Bougrain¹ ³

¹ Université de Lorraine, CNRS, LORIA, F-54000, Nancy, France  
² GIPSA-lab, Université Grenoble Alpes, CNRS, Grenoble-INP, Grenoble, France  
³ Sorbonne Université, ICM, CNRS, Inria, Inserm, Paris, France

## Key idea

At any point during training, ABiMap maintains two consecutive BiMap outputs:

- $P_{\text{hi}}$ : projection with $m$ filters → SPD matrix of size $m \times m$
- $P_{\text{lo}}$ : projection with $m-1$ filters → SPD matrix of size $(m-1) \times (m-1)$

Since $P_{\text{lo}}$ and $P_{\text{hi}}$ have different dimensions, $P_{\text{lo}}$ is first raised to dimension $m$ via **dimensionality transcending**, padding it with a unit eigenvalue:

$$P^{\uparrow}_{\text{lo}} = \begin{pmatrix} P_{\text{lo}} & 0 \\ 0 & 1 \end{pmatrix}$$

The two outputs are then interpolated along the **log-Euclidean geodesic**:

$$\gamma_{\text{LEM}}\left(P^{\uparrow}_{\text{lo}}, P_{\text{hi}}, \alpha\right) = \operatorname{Exp}\left( (1-\alpha) \operatorname{Log} P^{\uparrow}_{\text{lo}} + \alpha \operatorname{Log} P_{\text{hi}} \right)$$

where $\alpha$ is learned by backpropagation.