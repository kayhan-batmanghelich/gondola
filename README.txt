

INTRODUCTION
============

This software implements Generative-Discriminative Basis Learning (GONDOLA),
which is explained in detail in paper [1] and extended later on in paper [2].
Theoretical ideas are explained in these papers, but as a brief explanation,
GONDOLA provides a generative method to reduce the dimensionality of medical
images while using class labels. It produces basis vectors that are useful
for classification and also clinically interpretable. 
  
When provided with two sets of labeled images as input, the software outputs
features in Weka Format (.arff files [3]) and a MATLAB data file (.mat file).
The program can also save basis vectors as NIfTI-1 images. Scripts are provided
to find and build an optimal classifier using Weka. The software can also be
used for semi-supervised cases in which a number of subjects do not have class
labels (for an example, please see [1]).
 
If you find this software useful, please cite [1] and [2].


 
PACKAGE OVERVIEW
================

  Source Package
  --------------

  - BasisProject.cmake   Meta-data used by BASIS to configure the project.
  - CMakeLists.txt       Root CMake configuration file.
  - doc/                 Software documentation such as the user manual.
  - example/             Example input files.
  - src/                 Source code files.
  - test/                Implementation of software tests and corresponding data.

  - AUTHORS.txt          A list of the people who contributed to this software.
  - COPYING.txt          The copyright and license notices.
  - INSTALL.txt          Build and installation instructions.
  - README.txt           This readme file.


  Binary Package
  --------------

  Please refer to the INSTALL file for details on where the built executables
  and libraries, the auxiliary data, and the documentation files are installed.



LICENSING
=========

  See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.



INSTALLATION
============

  See build and installation instructions given in the INSTALL file.



DOCUMENTATION
=============

  See the software manual for details on the software including a demonstration
  of how to apply the software tools provided by this package.



REFERENCES
==========

  [1] K. N. Batmanghelich, B. Taskar, C. Davatzikos; Generative-Discriminative
      Basis Learning for Medical Imaging; IEEE Trans Med Imaging. 2012 Jan;31(1):51-69. 

  [2] K. N. Batmanghelich, B. Taskar, D. Ye, C. Davatzikos; Regularized Tensor Factorization
      for Multi-modality Medical Image Classification, MICCAI 2011, LNCS 6893, p17.

  [3] http://www.cs.waikato.ac.nz/ml/weka/
 
  [4] http://www.itk.org/
