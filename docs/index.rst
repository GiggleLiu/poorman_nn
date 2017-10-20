.. title:: Overview

Layers
===========
Layers is intended for scientific programming,
with complex number support and periodic boundary condition and most importantly,
extensible.

Our project repo is https://159.226.35.226/jgliu/PoorNN.git

Here I lists its main features

    * low abstraction
    * complex number support
    * periodic boundary convolution favored
    * numpy based, c/fortran boost
    * easy for extension, using interfaces to unify layers, which means layers are standalone modules.

But its cuda version is not realized yet.

Contents
	* :ref:`tutorial`: Tutorial containing instructions on how to get started with Layers.
	* :ref:`examples`: Example implementations of mnist networks.
	* :ref:`api`: The code documentation of Layers.

.. toctree::
	:maxdepth: 2
	:hidden:
	
	tutorials
	examples
	api
