# Oliker-Prussner-Method

This software implements the Oliker-Prussner-Method [[1]](#1) in Python for solving a two dimensional version of the Monge-Ampère-Equation:

$$\det \(D^2 u(x,y)\)=f(x,y)$$

This software was created as a bachelor thesis project. A copy of this thesis (german only) is included as a detailed documentation about the working principles of this method. The software is split up into three parts for creating an initial function and meshs to work with, running the Oliker-Prussner-Iteration and lastly plotting the results.

We should also mention that this software does not implement the additional Newton-Iteration proposed by Oliker and Prussner in [[1]](#1), that can be used to gain an additional convergence speed up.
We only implemented the pure iteration sceme from Oliker and Prussner.

# How to use this software

We now give brief instructions on how to use and manipulate this software for your own purpose, including neccessary steps on how to set up python correctly.

When executing this software, be aware that the running time increases quickly with a growing number of grid points!
With as less as 1000 total grid points we began to see running times of several hours and by increasing the accuracy of the terminal criteria (the Monge-Ampère-Measure that shall be approximated) we even saw running times of multiple days. We should also mention, that the code is currently not parallelized!
**THUS ONLY RUN BIG EXAMPLES WITH DECENT PREPARATION AND/OR CODE MODIFICATION!**

## Software parts

This software consists of the follinwing files:

1. **`requirements.txt`**: Contains dependencies of other Python modules and is used for an automatic installation via `pip`.
2. **`oliker_prussner_main.py`**: The main program for running the iteration sceme of the Oliker-Prussner-Method.
3. **`oliker_prussner_init.py`**: Creates meshes, initial functions and calculating the desired initial measures for the iteration sceme from the right hand side function $$f$$.
4. **`oliker_prussner_triangulation.py`**: Contains important functionallity for handling the triangulation and convex hull.
5. **`oliker_prussner_plot.py`**: Contains helper functions and functionallity for plotting the results.
6. **Four folders with prepared examples.**

## Python installation
Start by downloading the repository manually or with the git command:
```bash
git clone https://github.com/yannick-hoeffner/oliker_prussner_method
```
This software was initially created and tested with Python version `3.11.2`, but it should run with newer versions as well. 
After installing a suitable python version ([Download here](https://www.python.org/downloads/)), we recommend to use a virtual environment ([more information here](https://docs.python.org/3/tutorial/venv.html)) to prevent clashes with other installed python modules.

To create a virtual environment, execute the following three commands:
```bash
cd /path/to/cloned/repository
python -m venv .
./Scripts/activate
```
Now, the virtual environment will be created inside the downloaded repository and will be activated via the third command. 
We now can automatically install the correct version of the neccessary Python modules `numpy`, `numba`, `scipy` and `matplotlib` via `pip`:
```bash
pip install -r requirements.txt
```

## Mesh creation and initialization 

Before we can run the iteration sceme of the Oliker-Prussner-Method, we first have to create a triangular mesh to work with, as well as an initial function for starting the iteration and a desired pre defined solution to calculate the error of the output from the Oliker-Prussner-Method. The software contains four predefined example folders. If you want to create other meshes/initial states then use the following command:
```bash
python oliker_prussner_init.py <"circle"|"square"> <param1> <param2> <function> [<plot=False>]
```
### Parameter description

- **circle | square**: Select whether you want to create a mesh of the unit circle or of the square $$[-1,1]^2$$.
- **param1, param2**: specify the mesh creation.
  - for the quadratic mesh:
    - `param1`: Number of equidistant grid points per dimension on $$[-1,1]$$.
    - `param2`: Densification of the points towards the edge of the square (`param2=1` keeps equidistant mesh) using the formula:
      
      $$x_{\text{neu}} = \text{sign}(x_{\text{alt}}) \cdot |x_{\text{alt}}|^{\texttt{param2}}$$
      
      $$y_{\text{neu}} = \text{sign}(y_{\text{alt}}) \cdot |y_{\text{alt}}|^{\texttt{param2}}$$
  - for the unit circle:
    - `param1`: Number of grid points per circle.
    - `param2`: Number of equidistant placed circles inside the unit circle.
- **function**: specify, which function should be used to test the Oliker-Prussner-Method`). Currently there are five functions predefined:
    - **function = $1$**: $$u(x,y) = x^2+y^2$$
    - **function = $2$**: $$u(x,y) = \tan(x^2+y^2)$$
    - **function = $3$**: $$u(x,y) =1-|x+y|$$
    - **function = $4$**: $$u(x,y) = \exp(x^2+y^2)$$
    - **function = $5$**: $$u(x,y) = -\frac{\sin(x)\sin(y)}{\sin(x)+\sin(y)}$$
    - If you want to create yout own functions, then you have to update the functions `oliker-prussner-init.f` and `oliker-prussner-init.exactSolution`.
- **plot (optional)**: If `True`, the created initial triangulation and function will be plotted.

The four precreated folders of this repository were creeated using the following commands

```bash
python oliker_prussner_init.py "circle" 5 4 1
python oliker_prussner_init.py "circle" 6 4 2
python oliker_prussner_init.py "square" 9 1.0 3
python oliker_prussner_init.py "square" 13 0.5 5
```
The init program creates for every executation a folder with the following name sceme:
```
<circle|square>_<param1>_<param2>_<function>
```
This folder contains all neccessary files containing the mesh and function data, that will be needed by the iteration sceme to function properly.
## Execute the iteration sceme of the Oliker-Prussner-Method
After creating the initial mesh and start function, you can call the iteration sceme using the following comand:
```bash
python oliker_prussner_main.py <path_to_folder> <"solution"|"calculated"> [<plot=False>]
```

- **path_to_folder**: path to a folder that was created via `oliker-prussner-init`.
- **solution | calculated**: Select the measure to approximate:
  - `solution`: The exact Monge-Ampère-Measure of the desired solution (normaly unknown without knowing the solution to the Monge-Ampère-Equation).
  - `calculated`: The measure `\mu` calculated from the right hand side function `f`.
- **plot (optional)**: If `True`, the result will be plotted.

After the iteration finishes, a file called `latest-run.npz`, containing the results, will be created inside the given folder.

### Visualise the results
Now the `latest-run.npz` file can either be read out manually with NumPy or can be plotted with the command:
```bash
python oliker_prussner_plot.py <path_to_latest-run.npz>
```

## References

- <a id="1"> [1] </a>  *Oliker, V.I., and Prussner, L.D* (1989).  **"On the Numerical Solution of the Equation ....-(....) = f and Its Discretizations, I.."** In: Numerische Mathematik 54.3 : 271-294. [http://eudml.org/doc/133318](http://eudml.org/doc/133318)
