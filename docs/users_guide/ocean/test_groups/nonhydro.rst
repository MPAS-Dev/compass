.. _ocean_nonhydro:

nonhydro
==================

The ``ocean/nonhydro`` test group implements test cases from
`Vitousek et al. (2014) <http://dx.doi.org/10.1016/j.ocemod.2014.08.008>`_
to validate the nonhydrostatic capability of MPAS-O.

The test group includes 2 test cases. The first one describes a 
continuously-stratified internal seiche and is presented in section 5.3
in Vitousek et al. (2014). The second one describes internal solitary 
waves and is presented in section 5.5. All test cases have 4 steps:
``initial_state``, which defines the mesh and initial conditions for the model,
``hydro``, which performs time integration of the standard hydrostatic 
model, ``nonhydro``, which performs time integration of the nonhydrostaic 
version of MPAS-O, and finally ``visualize``, which compares the solutions
obtained with the hydrostatic and nonhydrostatic model.

stratified_seiche
-----------------

The domain is a 10mx10m box discretized in the horizontal with 64 cells and
in the vertical with 100 layers. The initial density depends on the hyperbolic
tangent of an internal cosine wave, and density contours are localized at the 
stratified interface. The set config options for this test case are: 

.. code-block:: cfg

    # config options for the horizontal grid
    [horizontal_grid]

    #Number of cells in the x-direction
    nx = 64

    #Number of cells in the y-direction
    ny = 4

    #Distance from two cell centers
    dc = 0.15625

    # config options for the seiche testcase
    [stratified_seiche]

    #Depth of the bottom of the ocean
    maxDepth = 16.0

    #Number of vertical levels
    nVertLevels = 100

    #Alpha in eos
    eos_linear_alpha = 0.2

    #Beta in eos
    eos_linear_beta = 0.8

    #Reference temperature
    eos_linear_Tref = 10.0

    #Reference salinity
    eos_linear_Sref = 35.0

    #Reference density
    eos_linear_densityref = 1000.0

    #Density difference
    deltaRho = 10.0

    #Interface thickness
    interfaceThick = 1.0

    #Wave amplitude
    amplitude = 0.1

    #Wavenumber
    wavenumber = 0.314159265358979

The hydrostatic and nonhydrostatic simulations are run for one period 
(50s) using RK4 with a time-step of 0.01s. A plot is produced at time 
t = 12s showing the horizontal and vertical velocity profiles 
normalized by their respective maxima at locations x=5m and x=0m, 
respectively. The nonhydrostatic profiles coincide with first-mode 
linearized eigenfunction analysis for this kind of problem presented 
in `Fringer et al. (2003) <https://doi.org/10.1017/S0022112003006189>`_

internal_wave
-------------

This test case approximates the evolution of solitary-like waves in the 
South China Sea. The domain has a length L of 300 km and depth H of 
2000 m, and it discretized with 1200 cells in the horizontal and 100 
layers in the vertical. The initial density depends on the hyperbolic
tangent of a Gaussian depression, and then evolves into a train of 
solitary-like waves. The set config options for this test case are:

.. code-block:: cfg

    # config options for the horizontal grid
    [horizontal_grid]

    #Number of cells in the x-direction
    nx = 1200

    #Number of cells in the y-direction
    ny = 4

    #Distance from two cell centers
    dc = 250.0

    # config options for the solitary wave testcase
    [solitary_wave]

    #Depth of the bottom of the ocean
    maxDepth = 2000.0

    #Number of vertical levels
    nVertLevels = 100

    #Alpha in eos 
    eos_linear_alpha = 0.2

    #Beta in eos
    eos_linear_beta = 0.8

    #Reference temperature
    eos_linear_Tref = 10.0

    #Reference salinity
    eos_linear_Sref = 35.0

    #Reference density
    eos_linear_densityref = 1000.0

    #Upper-layer depth
    h1 = 250.0

    #Density difference
    deltaRho = 1.0

    #Wave interface thickness
    interfaceThick = 200.0

    #Wave amplitude
    amplitude = 250.0

    #Wavelenght
    wavelenght = 15000.0

The hydrostatic and nonhydrostatic simulations are run for 40h
using split-explicit with a baroclinic time-step of 1min and a 
barotropic time-step of 1s. A plot at 40h is produced and shows 
that the nonhydrostatic result leads to a train of rank-ordered 
solitary-like internal gravity waves, whereas the hydrostatic model 
fails to capture correct physics.
