# CtrlFuzz
This is the artifact of submission:"CtrlFuzz: A Controllable Diffusion-based Fuzz Testing for Deep Neural Networks via Coverage-aware Manifold Guidance".  

## Implementations

In this work , we present the CtrlFuzz, a controllable, manifold-aware diffusion model-based fuzz testing tool for DNNs. 
CtrlFuzz accomplishes two significant advances:
- it enhances the controllability of generation-based CGF via directing test inputs along with targeted manifold;
- it integrates the strengths of mutation-based and generation-based CGF, facilitating semantically meaningful and distribution-aligned test case synthesis. 

## Installation

- The implementation was based on Python 3.10.14, Opencv 4.10, Numpy 1.23.5, etc. 

## Usage
- Usage:src/train.py. Training models. 
- Usage:src/infer.py. Testcase generation.
- Usage:src/manifold_build.py. Building latent feature space.
