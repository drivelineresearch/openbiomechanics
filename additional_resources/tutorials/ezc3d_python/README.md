# OBP Tutorial #1: Accessing C3D files using Python and ezc3d

<img src="imgs/ez.png" alt="ezc3d logo" width="100"/> 

<img src="imgs/py.png" alt="python logo" width="100"/>

***This tutorial is an excerpt from a more exhaustive tutorial on using ezc3d that will be featured in Driveline Baseball's upcoming book on the biomechanics of pitching.***

# Getting Started

Below we will cover some basic exploration of c3d files using the wonderful ezc3d package in Python. We focus on Python implementation (specifically interactive Python via Jupyter Notebook) because Python and Jupyter are all free and well-maintained. 

If you do not have an Python IDE (interactive development environment) set up, you will need to install one. We use Jupyter notebooks via either Anaconda or VS Code. Many excellent guides have been writeen already so we will not write our own here.

A good starting point for setting up and getting familiar with Anaconda and Jupyter Notebooks can be found [here](https://sparkbyexamples.com/python/install-anaconda-jupyter-notebook/).
You can download VS Code [here](https://code.visualstudio.com/download). A guide to installing VS Code's Jupyter extension can be found [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). 

It does not matter whether you chose Anaconda or VS Code (or some other IDE), only that you have a way to interact with Python code that you are comfortable with and understand.

For additional examples beyond what we cover below, as well as tutorials for the MATLAB and C++ inclined among us, visit the [ezc3d GitHub](https://github.com/pyomeca/ezc3d) and check out their awesome paper here: [![DOI](https://joss.theoj.org/papers/10.21105/joss.02911/status.svg)](https://doi.org/10.21105/joss.02911)

Once you have your Python environment set up, make sure you install the ezc3d package using anaconda prompt:

```python
conda install -c conda-forge ezc3d
```

# The ezc3d C3D Object

A c3d file is read into python using the ezc3d command `ezc3d.c3d()`. The resulting object is an ezc3d c3d class. 
Luckily, this class functions a lot like a Python [dictionary](https://www.w3schools.com/python/python_dictionaries.asp) (i.e. it has keys and values) which allows us to examine its structure in greater detail.

All code below can be found in the Jupyter Notebook housed within this GitHub repo `(~\code\explore_c3d_object_structure.ipynb)`

# Exploring the C3D Object Structure

Let’s assign our c3d object to a variable `c`:

```python
# import package
import ezc3d

file = 'path to your c3d file'
# read in c3d file and assign to variable c
c = ezc3d.c3d(file)
```

Our variable `c` is now a c3d object with keys and values. We can examine its first level of structure using the command `c.keys()`:

```python
c = ezc3d.c3d(file)

c.keys()
# dict_keys(['header', 'parameters', 'data'])
```

The first level of structure for our c3d object contains three substructures: `header`, `parameters`, and `data`. We can access one level deeper within our c3d object by reusing the `keys()` command on one of our three substructures:

```python
c = ezc3d.c3d(<'path to your c3d file'>)

c.keys()
# dict_keys(['header', 'parameters', 'data'])

c['header'].keys()
# dict_keys(['points', 'analogs', 'events'])
```

We can access deeper and deeper levels of nesting by adding dictionary keys together:

```python
c = ezc3d.c3d(<'path to your c3d file'>)

c.keys()
# dict_keys(['header', 'parameters', 'data'])

c['header'].keys()
# dict_keys(['points', 'analogs', 'events'])

c['header']['points'].keys()
# dict_keys(['size', 'frame_rate', 'first_frame', 'last_frame'])
```

Eventually, we’ll run out of nested levels and arrive at the actual data. Once we get to the bottom level, the `keys()` command will no longer work:

```python
c['header']['points']['size'].keys()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[66], line 1
----> 1 c['header']['points']['size'].keys()

AttributeError: 'int' object has no attribute 'keys'
```

The above error means that whatever is in `c['header']['points']['size']` isn’t a dictionary. If we take away the `keys()` command, we can see what’s there:

 

```python
x = c['header']['points']['size']
print(x, type(x))
# 45 <class 'int'>
```

The above code shows that `c['header']['points']['size']` housed an integer; specifically, 45. This happens to be the number of markers in our markerset. This might be useful information so we can assign it to a variable with a more descriptive name than `x`:

```python
num_markers = c['header']['points']['size']
```

---

# Full C3D object structure:

# Top Level (c)

## header

### points

- size
- frame_rate
- first_frame
- last_frame

### analogs

- size
- frame_rate
- first_frame
- last_frame

### events

- size
- events_time
- events_label

## parameters

### POINT

- \_\_METADATA\_\_
- USED
- SCALE
- RATE
- DATA_START
- FRAMES
- UNITS
- LABELS
- DESCRIPTIONS

### ANALOG

- \_\_METADATA\_\_
- USED
- LABELS
- GEN_SCALE
- SCALE
- OFFSET
- GAIN
- UNITS
- RATE
- RATIO
- FORMAT
- DESCRIPTIONS
- BITS

### FORCE_PLATFORM

- \_\_METADATA\_\_
- USED
- TYPE
- ZERO
- CORNERS
- ORIGIN
- CHANNEL
- ZEROS
- CAL_MATRIX

### MANUFACTURER

- \_\_METADATA\_\_
- COMPANY
- SOFTWARE
- VERSION_LABEL

### SUBJECTS

- \_\_METADATA\_\_
- USED
- LABEL_PREFIXES
- NAMES

### FORCE_STRUCTURE

- \_\_METADATA\_\_
- USED

### EVENT

- \_\_METADATA\_\_
- USED

## data

### points

numpy array: 4 x number of markers x number of point frames

### meta_points

- residuals
- camera_masks

### analogs

numpy array: 1 x 6*number of force plates x number of analog frames