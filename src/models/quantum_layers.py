import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

class QuantumLayer(nn.Module):
    """
    Generic Quantum Layer supporting different encodings and ansatzes.
    """
    def __init__(self, n_qubits, n_layers, encoding='angle', ansatz='vqc', device='default.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding.lower()
        self.ansatz = ansatz.lower()
        
        self.dev = qml.device(device, wires=n_qubits)
        
        self.dev = qml.device(device, wires=n_qubits)
        
        # Define weight shapes based on ansatz
        if self.ansatz == 'vqc':
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        elif self.ansatz == 'basic':
            weight_shapes = {"weights": (n_layers, n_qubits)}
        elif self.ansatz == 'hardware_efficient':
            weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        elif self.ansatz == 'qaoa':
             weight_shapes = {"weights": (n_layers, 2, n_qubits)}
        elif self.ansatz == 'pqc':
             # Custom Parameterized Quantum Circuit with arbitrary rotations and CZ entanglement
             weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        else:
             # Default generic shape
             weight_shapes = {"weights": (n_layers, 2 * n_qubits)}

        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Encoding
            self._apply_encoding(inputs)
            
            # Ansatz
            self._apply_ansatz(weights)
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = circuit
        self.weight_shapes = weight_shapes
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def _apply_encoding(self, inputs):
        if self.encoding == 'angle':
            # Angle encoding: RY rotations
            # Inputs shape should be (batch, n_qubits)
            for i in range(self.n_qubits):
                qml.RX(inputs[:, i], wires=i) # Using RX as per prompt or RY
        elif self.encoding == 'amplitude':
            # Amplitude encoding
            # Inputs must be normalized and size 2^n
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_qubits), normalize=True, pad_with=0.)
        elif self.encoding == 'iqp':
            # IQP Encoding
            qml.IQPEmbedding(features=inputs, wires=range(self.n_qubits), n_repeats=1)
        elif self.encoding == 'molecular':
            # Molecular-inspired encoding: 3 parameters per qubit (RX, RY, RZ) capturing 3D correlations
            for i in range(self.n_qubits):
                if inputs.shape[1] > i:
                    qml.RX(inputs[:, i], wires=i)
                if inputs.shape[1] > i + self.n_qubits:
                    qml.RY(inputs[:, i + self.n_qubits], wires=i)
                if inputs.shape[1] > i + 2*self.n_qubits:
                    qml.RZ(inputs[:, i + 2*self.n_qubits], wires=i)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _apply_ansatz(self, weights):
        if self.ansatz == 'vqc':
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        elif self.ansatz == 'basic':
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
        elif self.ansatz == 'hardware_efficient':
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                # Circular Entanglement
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        elif self.ansatz == 'qaoa':
            # Simplified QAOA-inspired layer
            for l in range(self.n_layers):
                # Cost Hamiltonian (ZZ interactions)
                for i in range(self.n_qubits):
                     next_i = (i + 1) % self.n_qubits
                     qml.CNOT(wires=[i, next_i])
                     qml.RZ(weights[l, 0, i], wires=next_i)
                     qml.CNOT(wires=[i, next_i])
                
                # Mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qml.RX(weights[l, 1, i], wires=i)
        elif self.ansatz == 'pqc':
            # Custom PQC: full 3-axis rotation per qubit + CZ hardware entanglement structure
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)
                for i in range(self.n_qubits):
                    qml.CZ(wires=[i, (i + 1) % self.n_qubits])
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz}")

    def forward(self, x):
        return self.qlayer(x)
    
    def save_quantum_weights(self, path):
        """
        Save quantum circuit weights for transfer learning.
        
        Args:
            path (str): Path to save weights
        """
        checkpoint = {
            'weights': self.qlayer.weights.data,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'encoding': self.encoding,
            'ansatz': self.ansatz,
            'weight_shapes': self.weight_shapes
        }
        torch.save(checkpoint, path)
        print(f"Quantum weights saved to {path}")
    
    def load_quantum_weights(self, path, strict=True):
        """
        Load pre-trained quantum circuit weights for transfer learning.
        
        Args:
            path (str): Path to load weights from
            strict (bool): If True, verify compatibility of architecture
        
        Raises:
            ValueError: If architecture mismatch and strict=True
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        if strict:
            # Verify compatibility
            if checkpoint['n_qubits'] != self.n_qubits:
                raise ValueError(f"Qubit mismatch: expected {self.n_qubits}, got {checkpoint['n_qubits']}")
            if checkpoint['n_layers'] != self.n_layers:
                raise ValueError(f"Layer mismatch: expected {self.n_layers}, got {checkpoint['n_layers']}")
            if checkpoint['encoding'] != self.encoding:
                raise ValueError(f"Encoding mismatch: expected {self.encoding}, got {checkpoint['encoding']}")
            if checkpoint['ansatz'] != self.ansatz:
                raise ValueError(f"Ansatz mismatch: expected {self.ansatz}, got {checkpoint['ansatz']}")
        
        # Load weights
        self.qlayer.weights.data = checkpoint['weights']
        print(f"Quantum weights loaded from {path}")




class QLSTM(nn.Module):
    """
    Quantum LSTM.
    """
    def __init__(self, input_size, hidden_size, n_qubits=4, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        # Factorized quantum layers for LSTM gates
        self.clayer_in = torch.nn.Linear(input_size + hidden_size, n_qubits)
        self.vqc = QuantumLayer(n_qubits, n_layers, encoding='angle', ansatz='vqc')
        
        # Project back to 4 * hidden_size for gates
        self.clayer_out = torch.nn.Linear(n_qubits, 4 * hidden_size)

    def forward(self, x, init_states=None):
        # x: (Batch, Seq, Features)
        # Sequence processing
        
        batch_size, seq_len, _ = x.size()
        h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device), 
                    torch.zeros(batch_size, self.hidden_size).to(x.device)) if init_states is None else init_states
        
        hidden_seq = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate
            combined = torch.cat([x_t, h_t], dim=1)
            
            # Reduce to qubits
            q_in = self.clayer_in(combined)
            
            # Quantum pass
            q_out = self.vqc(q_in) # shape: (batch, n_qubits)
            
            # Expand to gates
            gates = self.clayer_out(q_out)
            
            # Split
            i_t, f_t, g_t, o_t = gates.chunk(4, 1)
            
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(1))
            
        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)
