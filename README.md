# CombineTF2

Perform complex profile binned maximum likelihood fits by exploiting state-of-the-art differential programming. 
Computations are based on the tensorflow 2 library and scipy minimizers with multithreading support on CPU (FIXME: and GPU).
Implemted approximations in the limit of large sample size to simplify intensive computations.

## Setup

CombineTF2 can be run within a comprehensive singularity (recommended) or in an environment set up by yourself. 

### In singularity
The singularity includes a comprehensive set of packages. 
It also comes with custom optimized builds that for example enable numpy and scipy to be run with more than 64 threads (the limit in the standard build).
Activate the singularity image (to be done every time before running code)
```bash
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```

### In custom environment
The only non-standard package is 
```
https://pypi.org/project/narf-ioutils/0.1.1/
```

### Get the code
```bash
MY_GIT_USER=$(git config user.github)
git clone git@github.com:$MY_GIT_USER/combinetf2.git
cd WRemnants/
git remote add upstream git@github.com:WMass/combinetf2.git
```

Get updates from the central repository (and main branch)
```bash
git pull upstream main
git push origin main
```

## Making the input tensor

An example can be found in ```tests/make_tensor.py```. 

### Symmetrization
By default, systematic variations are asymmetric. 
However, defining only symmetric variations can be beneficial as a fully symmetric tensor has reduced memory consumption, simplifications in the likelihood function in the fit, and is usually numerically more stable. 
Different symmetrization options are supported:
 * "average": TBD
 * "conservative": TBD
 * "linear": TBD
 * "quadratic": TBD
If a systematic variation is added by providing a single histogram, the variation is mirrored. 

## Running the fit

For example:
```bash
python scripts/fit.py input_tensor.hdf5 -o results/fitresult.hdf5 -t 0 --doImpacts --doGlobalImpacts --binByBinStat --saveHists --computeHistErrors
```

## Fit diagnostics

Nuisance parameter impacts:
```bash
python scripts/printImpacts.py results/fitresult.hdf5
```

## Contributing to the code

We use linters. Activate git pre-commit hooks (only need to do this once when checking out)
```
git config --local include.path ../.gitconfig
```
