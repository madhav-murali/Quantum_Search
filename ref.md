# Theoretical Background and References

This document provides academic references and theoretical background for the quantum computations, models, and datasets used in this repository.

## Quantum Encodings

### Angle Encoding
**Concept**: Angle encoding embeds classical data into quantum states by using rotation gates (typically $R_X$ or $R_Y$) where the rotation angle is determined by the input feature value.

**References**:
- LaRose, R., & Coyle, B. (2020). Robust data encodings for quantum classifiers. *Physical Review A*, *102*(3), 032420.
- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*. Springer.

### Amplitude Encoding
**Concept**: Amplitude encoding maps a vector of $N$ features to the amplitudes of a quantum state of $\lceil \log_2 N \rceil$ qubits. This allows for exponential compression of data potential, but requires complex state preparation circuits.

**References**:
- Möttönen, M., Vartiainen, J. J., Bergholm, V., & Salomaa, M. M. (2004). Transformation of quantum states using uniformly controlled rotations. *Physical Review Letters*, *93*(13), 130502.
- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*. Springer.

### IQP Encoding
**Concept**: Instantaneous Quantum Polynomial (IQP) encoding uses a circuit of Hadamard gates and diagonal gates to embed data. It is conjecturally hard to simulate classically and is used to create "quantum-enhanced" feature spaces that are hard for classical kernels to approximate.

**References**:
- Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, *567*(7747), 209-212.

## Quantum Ansatz & Layers

### Variational Quantum Classifier (VQC) / Strongly Entangling Layers
**Concept**: A VQC uses a parameterized quantum circuit (PQC) followed by measurement. The "Strongly Entangling Layers" ansatz consists of single-qubit rotations and entangling gates (like CNOTs) arranged to maximize entanglement capability and expressibility.

**References**:
- Schuld, M., Bocharov, A., Svore, K. M., & Wiebe, N. (2020). Circuit-centric quantum classifiers. *Physical Review A*, *101*(3), 032308.
- Benedetti, M., Lloyd, E., Sack, S., & Fiorentini, M. (2019). Parameterized quantum circuits as machine learning models. *Quantum Science and Technology*, *4*(4), 043001.

### QAOA (Quantum Approximate Optimization Algorithm)
**Concept**: Originally designed for combinatorial optimization, the QAOA structure (alternating cost and mixer Hamiltonians) is adapted here as an ansatz for variational learning.

**References**:
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). *A quantum approximate optimization algorithm*. arXiv. https://arxiv.org/abs/1411.4028

### Quantum LSTM (QLSTM)
**Concept**: A hybrid quantum-classical extension of the Long Short-Term Memory (LSTM) network where the classical linear layers in the LSTM cell are replaced or augmented by Variational Quantum Circuits (VQCs).

**References**:
- Chen, S. Y. C., Yoo, S., & Fang, Y. L. L. (2020). *Quantum long short-term memory*. arXiv. https://arxiv.org/abs/2009.01783

## Classical Backbones & Datasets

### ResNet (Residual Networks)
**Concept**: Deep convolutional networks using skip connections to allow training of very deep models. Used here as a feature extractor.

**References**:
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770-778).

### Vision Transformer (ViT)
**Concept**: Adaptation of the Transformer architecture for image recognition, processing images as sequences of patches.

**References**:
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). *An image is worth 16x16 words: Transformers for image recognition at scale*. arXiv. https://arxiv.org/abs/2010.11929

### EuroSAT Dataset
**Concept**: A dataset for land use and land cover classification based on Sentinel-2 satellite images.

**References**:
- Helber, P., Bischke, B., Dengel, A., & Biedermann, S. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, *12*(7), 2217-2226.
