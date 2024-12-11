This repository is a fork of the simulation code from the paper:

**Ring Attractor Dynamics in the *Drosophila* Central Brain**
Authors: Sung Soo Kim\*, Hervé Rouault\*, Shaul Druckmann†, Vivek Jayaraman†
\* Equal contributors
† Correspondence to: druckmanns@janelia.hhmi.org or vivek@janelia.hhmi.org
Science, 2017, doi: 10.1126/science.aal4835

This repository was made to replicate and explore the model for COGSCI 318 - Theoretical Neuroscience at ENS. Specifcially, we perform an ablation study to generate model predictions of the impact of dead neurons on bump activity in the network.

This work is contained in the updated files in the folder `ring_attractor`. The updated files contain the author's code, as well as the necessary changes to conduct this study. 

As in the original code. to run the parameter sweep, specify the type of connectivity model, __delta__ or __cosine__, and the (integer) degree offset of the input stimulus with a cli command: `./simuself {model} {input_angle}`.

However, we add the following functionality:

1. The ability to run a single simulation trial, specifying paramters. This is achieved by specifying the amp and width parameters in the cli command:
`./simuself {model} {input_angle} {amp} {width}`.

2. The ability to run a single simulation trial with a range of neurons turned off. This is achieved by specifying amp, width, the --damaged flag, the degree offset from 0 of the center of the damaged neuron region, and the number of damaged neurons. i.e., 
`./simuself {model} {input_angle} {amp} {widht} --damaged {center_offset_angle} {num_neurons}`.

3. The ability to run a parameter sweep with damaged neurons. This is achieved by setting the --damaged flag, the degree offset from 0 of the center of the damaged neurons, and the number of damaged neuron region. i.e.,
`./simuself {model} {input_angle} --damaged {center_offset_angle} {num_neurons}`.

The single trial simulations produce a file called `single_run_activity.dat` which contains the activity of each neuron at each time step during a trial.
Both parameter sweeps output to the terminal, however they can be saved via a `>` command, i.e., `./simuself {model} {input_angle} > filename.txt`

`plotting.ipynb` contains the code to generate the plots used in our presentation. The videos of the activity one can generate are particularily fun. 

**A word about CMakeLists.txt**

The version of this file contained in this branch has been jerry-rigged to run on my local machine. It will likely not work via a naive download. To return it to its original state, simply uncomment lines 35 and 36, and comment out lines 39-53 in ring_attractor/CMakeLists.txt. Also, comment out line 32 in ring_attractor/src/CMakeLists.txt


The folder `bump_sampling` contains the code the authors used to fit multiple bumps on the ring. This was outside the scope of this project, and thus remains unchanged.

