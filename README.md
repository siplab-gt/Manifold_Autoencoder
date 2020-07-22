# Manifold Autoencoder
This is code associated with Connor, Marissa, and Christopher Rozell. "Representing Closed Transformation Paths in Encoded Network Latent Space." AAAI. 2020.

## Requirements
- torch
- scipy
- numpy
- cv2
- sklearn
- six
- glob

## Code structure

### Main Manifold Autoencoder Function
`manifold_autoencoder_closedPaths.py` is the main fule to run when training the manifold autoencoder mode. The argparser at the beginning of the file shows the possible parameters to select for a run. Two noteable parameters are the `data_use` parameter which specifies which dataset you'll be working with and the `training_phase` parameter which specifies which phase of training to run. The options for `data_use` are `concen_circle`, `rotDigits`, and `gait` corresponding to the tests from the paper. The `training_phase` options are `AE_train`, `transOpt_train`, and `finetune` corresponding to the three training phases described in Figure 2 in the paper.

### Auxillary functions
- `covNetModel.py` - Defines the convolutional networks used in the MNIST experiments
- `fullyConnectedLargeModel.py` - Defines the larger fully connected network used in the gait experiments
- `fullyConnectedModel.py` - Defines the fully connected network used in the concentric circle experiments
- `trans_opt_objectives.py` - functions used to infer transport operator coefficients
- `transOptModel.py` - defines the transport operator neural network layer
- `utils.py` - contains functions for reading in and generating data as well as visualizing outputs

## Guide to parameters for experiments
*Concentric circle experiment*:
- Autoencoder Training Phase: batch_size: 64, learning rate phi: 1e-4
- Transport Operator Training Phase: batch_size: 10, learning rate psi: 5, zeta: 1e-4, gamma: 0.005, M: 4
- Fine-tuning Phase: batch_size: 10, learning rate phi: 1e-4, learning rate psi: 5e-4, zeta: 0.0, gamma: 0.0, lambda: 1000, M: 1

*Rotated MNIST experiment*:
- Autoencoder Training Phase: batch_size: 64, learning rate phi: 1e-4
- Transport Operator Training Phase: batch_size: 32, learning rate psi: 0.01, zeta: 0.01, gamma: 8e-5, M: 10
- Fine-tuning Phase: batch_size: 32, learning rate phi: 0.005, learning rate psi: 1, zeta: 0.0, gamma: 0.0, lambda: 10, M: 1

*Gait experiment*:
- Autoencoder Training Phase: batch_size: 64, learning rate phi: 5e-4
- Transport Operator Training Phase: batch_size: 32, learning rate psi: 0.005, zeta: 0.05, gamma: 1e-4, M: 10


   

