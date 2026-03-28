import torch
import sys
import os
from src.models.backbones import BackboneFactory
from src.models.hybrid_model import HybridGeoModel

if len(sys.argv) < 3:
    print("Usage: extract_weights.py <input_ckpt> <output_weights>")
    sys.exit(1)

input_ckpt = sys.argv[1]
output_weights = sys.argv[2]

print(f"Loading {input_ckpt}...")
ckpt = torch.load(input_ckpt, map_location='cpu')
config = ckpt['config']
args = ckpt['args']

dataset_name = args.get('dataset', 'EuroSAT')
if dataset_name.lower() == "eurosat":
    n_classes = 10
    in_channels = 13 if args.get('bands') == 'ALL' else 3
elif dataset_name.lower() == "siri-whu":
    n_classes = 12
    in_channels = 3
elif dataset_name.lower() == "uc_m_luc":
    n_classes = 21
    in_channels = 3
else:
    n_classes = 10
    in_channels = 3

backbone, feature_dim = BackboneFactory.create(config['backbone'], pretrained=False, in_channels=in_channels)

model = HybridGeoModel(
    backbone=backbone,
    feature_dim=feature_dim,
    n_classes=n_classes,
    n_qubits=args['n_qubits'],
    n_qlayers=args['q_depth'],
    encoding=args.get('encoding') or config.get('encoding'),
    ansatz=args.get('ansatz') or config.get('ansatz'),
    q_type=config.get('q_type', 'standard'),
    standard_dim=config.get('standard_dim', None),
    multistage=config.get('multistage', False)
)

model.load_state_dict(ckpt['model_state_dict'])

os.makedirs(os.path.dirname(output_weights), exist_ok=True)
model.quantum_layer.save_quantum_weights(output_weights)
print("Extraction complete!")
