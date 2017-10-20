# Poor Man's Neural Network lib
Neural Network framework for researchers.

## Warning
* under development, no cuda support for this momentum

## Features

* low abstraction
* complex number support
* periodic boundary convolution favored
* numpy based, c/fortran boost
* easy for extension, using interfaces to unify layers, which means layers are standalone modules.

## Install
Clone this repository and run

```bash
    $ cd poornn
    $ pip install -r requirements.txt
    $ python setup.py install
```
**pygraphviz** is needed in order to activate network visualization. Its tutorial page is
http://pygraphviz.github.io/documentation/pygraphviz-1.4rc1/tutorial.html#start-up

## Run Tests
```bash
    $ pytest
```

Convolutional mnist test (need install tensorflow to provide mnist database)

```bash
    $ python -m poornn.tests.test_mnist.py
```

## Documentation
http://layers.readthedocs.io/en/latest/
