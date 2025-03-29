# Oliker-Prussner-Method

This software implements the Oliker-Prussner-Method [[1]](#1) in Python for solving a two dimensional version of the Monge-Ampère-Equation:

$$\det \(D^2 u(x,y)\)=f(x,y)$$

This software was created as a bachelor thesis project. A copy of this thesis (german only) is included as a detailed documentation about the working principles of this method. 

We should also mention that this software does not implement the additional Newton-Iteration proposed by Oliker and Prussner in [[1]](#1), that can be used to gain an additional convergence speed up.
We only implemented the pure iteration sceme from Oliker and Prussner.

# How to use this software

We now give brief instructions on how to use and manipulate this software for your own purpose, including neccessary steps on how to set up python correctly.

When executing this software, be aware that the running time increases quickly with a growing number of grid points!
With as less as 1000 total grid points we began to see running times of several hours and by increasing the accuracy of the terminal criteria (the Monge-Ampère-Measure that shall be approximated) we even saw running times of multiple days. We should also mention, that the code is currently not parallelized!
**THUS ONLY RUN BIG EXAMPLES WITH DECENT PREPARATION AND/OR CODE MODIFICATION!**

## Software parts

This software consists of the follinwing files:

1. **`requirements.txt`**: Contains dependencies of other Python packages and is used for an automatic installation via `pip`.
2. **`main.py`**: Contains a python example of how to use the code.
3. **`core_src.oliker_prussner_core.py`**: core functionalities for running the iteration sceme of the Oliker-Prussner-Method.
4. **`core_src.oliker_prussner_coreinit.py`**: Creates initial functions and calculating the desired initial measures for the iteration sceme from the right hand side function $$f$$.
5. **`core_src.oliker_prussner_coretriag.py`**: Contains important functionallity for handling the triangulation and convex hull.
6. **`core_src.oliker_prussner_plot.py`**: Contains helper functions and functionallity for plotting the results.

## Python installation
Start by downloading the repository manually or with the git command:
```bash
git clone https://github.com/yannick-hoeffner/oliker_prussner_method
```
This software uses Python version `3.12.9`. 
It should work with newer/older versions if you can successfully install the needed python packages. 
After installing a suitable python version ([Download here](https://www.python.org/downloads/)), we recommend to use a virtual environment ([more information here](https://docs.python.org/3/tutorial/venv.html)) to prevent clashes with other installed python packages.

To create a virtual environment, execute the following three commands:
```bash
cd /path/to/cloned/repository
python -m venv .
./Scripts/activate
```
Now, the virtual environment will be created inside the downloaded repository and will be activated via the third command. 
We now can automatically install the correct version of the neccessary Python packages `numpy`, `numba`, `scipy` and `matplotlib` via `pip`:
```bash
pip install -r requirements.txt
```

## Basic workflow

An example for the use of this code is supplied via the `main.py` file.
You have to create an initial list of grid points, a boundary mask indicating which grid point is a boundary point and the function values modeling the boundary condition.
Then `core_src.oliker_prussner_coreinit.py` automatically calculates an initial triangulation and an intial function for the iteration sceme.

You then either have to supply the Monge-Ampère-Measure that the Oliker-Prussner-Method should approximate at every vertex or you can supply a python function for evaluating the right hand side function $$f$$. In the latter case `core_src.oliker_prussner_coreinit.py` provides functionallity for automatically calculating the Monge-Ampère-Measure at each vertex.

Now you simply need to call `core_src.oliker_prussner_core.oliker_prussner_sceme` with the gathered arrays to run the Oliker-Prussner-Method.

# References

- <a id="1"> [1] </a>  *Oliker, V.I., and Prussner, L.D* (1989).  **"On the Numerical Solution of the Equation ....-(....) = f and Its Discretizations, I.."** In: Numerische Mathematik 54.3 : 271-294. [http://eudml.org/doc/133318](http://eudml.org/doc/133318)
