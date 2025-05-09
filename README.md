# Oliker-Prussner-Method

This Python package implements the Oliker-Prussner-Method [[1]](#1) for solving a two dimensional version of the Monge-Ampère-Equation:

$$\det \left(D^2 u(x,y)\right)=f(x,y)$$

This was originally a project as part of a bachelor's thesis and is currently being maintained even after its completion.
A copy of the thesis (german only) is included as a detailed documentation about the working principles of this method. 

We should also mention that this software does not implement the additional Newton-Iteration proposed by Oliker and Prussner in [[1]](#1), that can be used to gain an additional convergence speed up.
We only implemented the pure iteration sceme from Oliker and Prussner.

# How to use this software

We now give brief instructions on how to use and manipulate this software for your own purpose, including neccessary steps on how to set up python correctly.

When executing this software, be aware that the running time increases quickly with a growing number of grid points!
With as less as 1000 total grid points we began to see running times of several hours and by increasing the accuracy of the terminal criteria (the Monge-Ampère-Measure that shall be approximated) we even saw running times of multiple days.
We should also mention, that the code is currently not parallelized!
**THUS ONLY RUN BIG EXAMPLES WITH DECENT PREPARATION AND/OR CODE MODIFICATION!**

## Software parts

This GitHub repository contains an example file `main.py` for a basic workflow when solving the Monge-Ampère-Equation.

The functionallity of this package is distributed among the following four files (located in the `src` folder):

1. **`oliker_prussner_method.core`**: core functionalities for running the iteration sceme of the Oliker-Prussner-Method.
2. **`oliker_prussner_method.initialisation`**: Creates initial functions and calculating the desired initial measures for the iteration sceme from the right hand side function $f$.
3. **`oliker_prussner_metod.triangulation`**: Contains important functionallity for handling the triangulation and convex hull.
4. **`oliker_prussner_method.plotting`**: Contains helper functions and functionallity for plotting the results.

## Python installation
This package was originally developed with Python version `3.12.9` and should work with newer versions as well.

After installing a suitable python version ([Download here](https://www.python.org/downloads/)), you can simply download this package via `pip` (The example `main.py` is not included in the pip download!) and automatically install all dependencies:

```bash
python -m pip install oliker_prussner_method
```
You can also manually download the repository via git and install the package via pip into your Python environment:
```bash
mkdir path/to/new/folder
cd path/to/new/folder
git clone https://github.com/yannick-hoeffner/oliker_prussner_method .
python -m pip install -e .
```


## Basic workflow

An example for the use of this code is supplied via the `main.py` file.
You have to create an initial list of grid points, a boundary mask indicating which grid point is a boundary point and the function values modeling the boundary condition.
Then `oliker_prussner_method.initialisation.py` automatically calculates an initial triangulation and an intial function for the iteration sceme.

You then either have to supply the Monge-Ampère-Measure that the Oliker-Prussner-Method should approximate at every vertex or you can supply a python function for evaluating the right hand side function $f$. In the latter case `oliker_prussner_method.initialisation.py` provides functionallity for automatically calculating the Monge-Ampère-Measure at each vertex.

Now you simply need to call `oliker_prussner_method.core.oliker_prussner_sceme` with the gathered arrays to run the Oliker-Prussner-Method.

# References

- <a id="1"> [1] </a>  *Oliker, V.I., and Prussner, L.D* (1989).  **"On the Numerical Solution of the Equation ....-(....) = f and Its Discretizations, I.."** In: Numerische Mathematik 54.3 : 271-294. [http://eudml.org/doc/133318](http://eudml.org/doc/133318)
