# DentScanNet

DentScanNet is a lightweight deep learning framework for real-time annotation of periodontal ultrasound images. The model predicts three point landmarks and three anatomical regions from intraoral ultrasound frames.

## Predicted Features

DentScanNet predicts the following six outputs:

### Point landmarks
- **GM**: Gingival Margin
- **CEJ**: Cemento-Enamel Junction
- **ABC**: Alveolar Bone Crest

### Anatomical regions
- **TOOTH**
- **BONE**
- **GINGIVA**

The model can also compute periodontal ultrasound indices from predicted landmarks:

- **iGR**: CEJ-to-GM distance
- **iGH**: GM-to-ABC distance
- **iABL**: CEJ-to-ABC distance

## Repository Structure

DentScanNet/
├── model_dentscannet.py              # DentScanNet model architecture
├── dentscannet_data.py               # Data loading and preprocessing
├── train_dentscannet.py              # Training script
├── live_dentscannet_realtime.py      # Real-time video annotation script
├── README.md
└── requirements.txt
