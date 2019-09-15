# Esme
Esme is a python library that is used for topological data analysis of Graphs. It combines existing machine learning libraries
and libraries for persistence computation. Esme is used for systematically evaluating persistence diagram as features for graph classification. 

## Installation
 * Make sure the following libraries are installed: networkx, sklearn, [dionysus](https://github.com/mrzv/dionysus), and [sklearn_tda](https://github.com/MathieuCarriere/sklearn_tda)
 * Set python and pip directory correctly in the build.sh, and run build.sh. 
 
## Functionality

* Support common filtration on vertices such as degree, [ricci curvature](https://projecteuclid.org/euclid.tmj/1325886283), closeness centrality and also 
filtration functions on edges such as jaccard similarity, edge probability for commonly used graph classification [dataset](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

* Support sub/superlevel filtration and 0/1/extended-homology 

* Support computing persistence diagram and various kernels in parallel

* Various ways of featurization of computed persistence diagram, such as Persistence Landscape,
Sliced Wasserstein kernel, Persistence Scale Space kernel, Persistence Weighted Gaussian Kernel etc.

* Systematic evaluation using [sacred](https://github.com/IDSIA/sacred). Sacred is a tool to help you configure, organize, log and reproduce experiments developed at IDSIA.
 This is not mandatory but highly recommended if you want to evaluate different filtration functions and kernel methods for persistence diagram.

## Example
See Esme/examples. More examples will be added soon.
## Contact
If you have any questions, feel free to contact me via cai.507@osu.edu or submit an issue on github.
## Misc
The name of Esme comes from a short [story]((https://en.wikipedia.org/wiki/For_Esm%C3%A9%E2%80%94with_Love_and_Squalor)) by J. D. Salinger. It recounts a sergeant's meeting with a young girl before being sent into combat in World War II.