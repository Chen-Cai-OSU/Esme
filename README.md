# Esme
Esme is a python library that is used for topological data analysis of Graphs. It combines existing machine learning libraries
and libraries for persistence computation. Esme is used for systematically evaluating persistence diagram for graph classification. 

---
## Installation
 * Make sure the following libraries are installed: networkx, sklearn, [dionysus](https://github.com/mrzv/dionysus), and [sklearn_tda](https://github.com/MathieuCarriere/sklearn_tda)
 * Set python and pip directory in the build.sh, and run it 
 
## Functionality

* Given a function on graph vertices, Esme can compute the persistence diagram 

* Support common filtration on vertices such as degree, ricci curvature, closeness centrality and also 
filtration functions on edges such as jaccard similarity, edge probability

* Support sub/superlevel filtration and 0/1/extended-homology 

* Support detailed customization such as how to handle the aggregation of persistence diagrams, flip or not, 

* Various ways of featurization of computed persistence diagram, such as Persistence Landscape,
Sliced Wasserstein kernel, Persistence Scale Space kernel, etc.

* Systematic evaluation using [sacred](https://github.com/IDSIA/sacred). Sacred is a tool to help you configure, organize, log and reproduce experiments developed at IDSIA.
 This is not mandatory but highly recommended if you want to evaluate different filtration functions and kernel methods for persistence diagram.

## Example

## Contact
If you have any questions, feel free to contact me via chencai.math@gmail.com or submit an issue on github.