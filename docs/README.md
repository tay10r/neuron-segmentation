Neuron Segmentation
===================

This project is for training a U-net model to segment the soma and neurites of a neuron.

<p align="center">
    <img src="banner.png">
</p>

This project is unique in that the imagery is from a simulated fluorescent confocal microscope and the neuron geometry
is from a collection of digitally reconstructed neurons. This allows the model to be trained with physically based
parameters, such as the Z step distance.  The sister project for simulating microscopes can be found
[here](https://github.com/tay10r/neuroscope). We plan on expanding the capabilities of the simulation to include phase
contrast microscopy and more physically-based parameterization of confocal fluorescent microscopes. We source the neuron
 models from [NeuroMorpho](neuromorpho.org) and support reading standardized SWC files.

### Project Structure

```
| - notebooks/
|   | - 00_Data_Generation.ipynb  # Use this notebook first, to generate the training data.
|   | - 01_Model_Training.ipynb   # This trains a variant of U-net to segment the neurons.
| - docs/
|   | - README.md                 # This file.
| - code/                         # A directory containing the source code for the demo web page.
| - demo/                         # The pre-built demo web page
```

### Setup with HP AI Studio

We use HP AI Studio to generate data and train models. This allows us to iterate quickly between tweaking physical
parameters of the microscope and training the model. If you're familiar with how to setup and run Jupyter, MLFlow, and
Tensorboard, you may also set those up and run the code with your own configuration. To get started with HP AI Studio:

| Step | Instructions                         |
|------|--------------------------------------|
|    1 | Create a new project in HP AI Studio |
|    2 | Set the Git Repository URL to `https://github.com/tay10r/neuron-segmentation.git` |
|    3 | Choose the Git Local Folder so that the cloned repo has somewhere to live. |
|    4 | Proceed to Create a Workspace and select the Deep Learning GPU template. |
|    5 | Name the workspace (for example: *Neuron Segmentation Workspace*) |
|    6 | Open a new Terminal in Jupyter and navigate to the repo at `neuron-segmentation` |
|    7 | Install Embree (for the simulator): `conda install conda-forge::embree`
|    8 | Install Python Dependencies: `pip3 install -r requirements.txt` |

### Running Project

The project is meant to be run using the notebooks.
The first thing you'll want to do is run the setup instructions above.
Then, you may run the first notebook, `00_Data_Generation.ipynb` in order to generate training and validation data.
Next, you may run the training notebook, `01_Model_Training.ipynb` in order to train a model.
Finally, you may run the deployment notebook, `02_Deploy.ipynb` in order to deploy the model to a local MLFlow server.
If you are running this in HP AI Studio, you may also use the deployed model and demo web page after running the last notebook.