import torch
import torch.nn as nn
from .quantum_layers import QuantumLayer, QLSTM

class HybridGeoModel(nn.Module):
    """
    Hybrid Quantum-Classical Model for Geospatial Classification.
    """
    def __init__(self, backbone, feature_dim, n_classes, n_qubits=4, n_qlayers=1, 
                 encoding='angle', ansatz='vqc', q_type='standard', 
                 standard_dim=None, quantum_weights_path=None, freeze_quantum=False,
                 multistage=False):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.multistage = multistage
        self.n_qubits = n_qubits
        self.q_type = q_type.lower()
        self.standard_dim = standard_dim
        
        # Determine projector output dimension
        if standard_dim is not None:
            # Use standardized features for transfer learning
            self.projector_out = standard_dim
        elif encoding == 'amplitude':
            self.projector_out = 2**n_qubits
        else:
            self.projector_out = n_qubits
        
        # Feature projector (backbone -> projector_out)
        self.projector = nn.Linear(feature_dim, self.projector_out)
        
        if self.q_type == 'standard':
            # Determine quantum input dimension
            if encoding == 'amplitude':
                q_input_dim = 2**n_qubits
            elif encoding == 'molecular':
                q_input_dim = 3 * n_qubits
            else:
                q_input_dim = n_qubits
            
            # Add adapter if using standard_dim and it differs from quantum input
            if standard_dim is not None and standard_dim != q_input_dim:
                self.quantum_adapter = nn.Linear(standard_dim, q_input_dim)
            else:
                self.quantum_adapter = nn.Identity()
            
            # Create quantum layer
            self.quantum_layer = QuantumLayer(n_qubits, n_qlayers, encoding=encoding, ansatz=ansatz)
            
            # Load pre-trained quantum weights if provided
            if quantum_weights_path is not None:
                self.quantum_layer.load_quantum_weights(quantum_weights_path)
                print(f"Loaded quantum weights from {quantum_weights_path}")
            
            # Freeze quantum layer if requested
            if freeze_quantum:
                for param in self.quantum_layer.parameters():
                    param.requires_grad = False
                print("Quantum layer parameters frozen")
            
            # Quantum output usually [expval(Z_0), ..., expval(Z_n)] -> size n_qubits
            self.classifier = nn.Linear(n_qubits, n_classes)
            
        elif self.q_type == 'qlstm':
            # Simplified: Use QLSTM on the projected features
            q_input_dim = standard_dim if standard_dim is not None else self.projector_out
            self.quantum_adapter = nn.Identity()
            self.quantum_layer = QLSTM(input_size=q_input_dim, hidden_size=n_qubits, n_qubits=n_qubits)
            self.classifier = nn.Linear(n_qubits, n_classes) 
            
        else:
            # Fallback to classical-only from projector
            self.quantum_adapter = nn.Identity()
            self.quantum_layer = nn.Identity()
            self.classifier = nn.Linear(self.projector_out, n_classes)

        # Auxiliary classifier for multistage training
        if self.multistage:
            self.aux_classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        
        # 1. Classical Backbone
        features = self.backbone(x) # (B, feature_dim)
        
        # 1.5 Auxiliary output for multistage classification
        aux_out = None
        if self.multistage and self.training:
            aux_out = self.aux_classifier(features)
        
        # 2. Projector
        x_proj = self.projector(features) # (B, projector_out)
        
        # 3. Quantum Layer
        if self.q_type == 'standard':
            # Apply quantum adapter for dimension matching
            x_adapted = self.quantum_adapter(x_proj)
            
            # Normalize for amplitude encoding if needed
            if hasattr(self, 'quantum_layer') and hasattr(self.quantum_layer, 'encoding'):
                if self.quantum_layer.encoding == 'amplitude':
                    norm = torch.norm(x_adapted, dim=1, keepdim=True) + 1e-8
                    x_adapted = x_adapted / norm
            
            x_q = self.quantum_layer(x_adapted)
            
        elif self.q_type == 'qlstm':
            # Reshape to sequence of length 1
            x_seq = x_proj.unsqueeze(1) # (B, 1, projector_out)
            x_seq_out, (h_n, c_n) = self.quantum_layer(x_seq)
            x_q = h_n # Last hidden state (B, hidden_size)
            
            x_q = x_proj
            
        # 4. Classifier
        out = self.classifier(x_q)
        
        if self.multistage and self.training:
            return out, aux_out
        return out

