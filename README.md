# sequence-models

My personal notes on sequence modelling using deep learning. Contains a mixture of theory in `notes` and implementations in `src` and/or `notebooks`.

- `src`: Python modules containing models and helper functions
- `notebooks`: Implementations and demos. For more complex architectures, such as the transformer, notebooks can be considered as entry points to the code in `src`, and so may be quite minimal. See `src` for implementation details, and `notes` for theory.
- `notes`: My personal notes on sequence models
- `figures`: Pictures for `notes` and `notebooks`

## Installation (CPU only)

This project's environment was set up for the CPU only, as I'm focusing on understanding architecture rather than performance here. 

If you're running on Windows, you can [try using](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) the `environment.yml` included with the project. This can be a bit flaky with complicated dependencies, at least in my experience. Otherwise, go to the pytorch [installation GUI](https://pytorch.org/) for the appropriate command, and then once pytorch is installed, then use the `environment.yml` as a guide for the rest of the dependencies. You can delete problematic dependencies if you encounter errors. [Use mamba!](https://github.com/mamba-org/mamba)

### Updating the `environment.yml`

```
conda env export --from-history > environment.yml
```