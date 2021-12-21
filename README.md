# sequence-models

My personal notes on sequence modelling using deep learning. Contains a mixture of theory in `notes` and implementations in `src` and/or `notebooks`.

- `src`: Python modules containing models and helper functions
- `notebooks`: Implementations and demos. Not all models necessarily will have notebooks. Look at `src/seq` for a full list.
- `notes`: My personal notes on sequence models
- `figures`: Pictures for `notes` and `notebooks`
## Installation

If you're running on Windows, you can try using the `environment.yml` included with the project. This can be a bit flaky with complicated dependencies, at least in my experience. Otherwise, go to the pytorch [installation GUI](https://pytorch.org/) for the appropriate command, and then once pytorch is installed, then use the `environment.yml` as a guide for the rest of the dependencies. You can delete problematic dependencies if you encounter errors, and also remove the build information if you're on a different platform. [Use mamba!](https://github.com/mamba-org/mamba)