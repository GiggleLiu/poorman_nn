# Poor Man's Neural Network lib
Neural Network framework for researchers.

## Warning
* under development, no cuda support for the momentum

## Features

* low abstraction
* complex number support
* periodic boundary convolution favored
* numpy based, c/fortran boost
* easy for extension, using interfaces to unify layers, which means layers are standalone modules.

## Install
clone this repository and run

```bash
    python setup.py install
```

## Run convolutional mnist test

```bash
    cd tests
    python test_mnist.py
```
