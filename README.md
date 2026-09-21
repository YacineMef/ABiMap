# ABiMap

BiMap layer that learns its output dimension $m$ jointly with the weight matrix during training. 

**Reference** : 

Learning the dimension of BiMap layers in SPD networks

Yacine Meftah¹, Marco Congedo², Laurent Bougrain¹ ³

¹ Université de Lorraine, CNRS, LORIA, F-54000, Nancy, France  
² GIPSA-lab, Université Grenoble Alpes, CNRS, Grenoble-INP, Grenoble, France  
³ Sorbonne Université, ICM, CNRS, Inria, Inserm, Paris, France

Paper link : https://hal.science/hal-05753369

## Key idea

At any point during training, ABiMap maintains two consecutive BiMap outputs:

- $P_{\text{hi}}$ : projection with $m$ filters → SPD matrix of size $m \times m$
- $P_{\text{lo}}$ : projection with $m-1$ filters → SPD matrix of size $(m-1) \times (m-1)$

Since $P_{\text{lo}}$ and $P_{\text{hi}}$ have different dimensions, $P_{\text{lo}}$ is first raised to dimension $m$ via **dimensionality transcending**, and the two outputs are then interpolated along the **log-Euclidean geodesic**:

$$\gamma_{\text{LEM}}\left(P^{\uparrow}_{\text{lo}}, P_{\text{hi}}, \alpha\right) = \exp\left( (1-\alpha) \log P^{\uparrow}_{\text{lo}} + \alpha \log P_{\text{hi}} \right)$$

where **the interpolation parameter $\alpha$ is learned by backpropagation**.

After each training step, $\alpha$ is checked against two thresholds to decide whether $m$ should change:

- **Expand**: if $\alpha$ rises above a high threshold, the additional filter is judged useful, $m$ is incremented.
- **Shrink**: if $\alpha$ falls below a low threshold, the additional filter is judged unnecessary, $m$ is decremented.

Over the course of training, this lets $m$ grow or shrink freely, settling on a particular dimension adapted to data by the end.

## Output 

Because $m$ can change throughout training, ABiMap always returns matrices padded to a fixed size $m_{\max}$ (via dimensionality transcending) to ensure dimension compatibility with the ensuing layers. **The input dimension of any layer placed after ABiMap must therefore be set to $m_{\max}$**. These extra weights can be pruned away after training (see paper for details).

This yields an SPDNet architecture equipped with a BiMap layer of adapted dimension $m^*$ and trained weights.


