# MagnonNernstEff_CrI3
atomistic spin simulation of the intrinsic magnon spin Nernst effect (caused by magnon topology) in the single layer honeycomb vdW magnet CrI_3. Input and process/analysis files for VAMPIRE
See  arXiv:2409.15964v1

The Nernst effect describes the transverse response of a spin current (in our case pure magnon current) to a longitudinal temperature gradient. We simulate it in a finite system in order to observe the spin accumulation at the transverse edges. 

To do it in VAMPIRE I define many materials with different temepratures for the gradient. In data processing, the honeycomb is a bit of a problem since output data is stored as one large list (two columns: atom ID and value) and the positions in another (two columns: atom ID and position) -- but that is not sorted, and very unpredictable when using parallelization. Thus the hashing algorithm in one of the processing files.
Workflow:
1. For the input file: generate geofiles and material files automatically
2. Run simulation with sh for Ensemble average
3. Hash data
4. Process data (averages)
5. Plot and analysis

