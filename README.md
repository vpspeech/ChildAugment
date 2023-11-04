# LPC_SWP_FEP-BWP
Codes for LPC Segmental Warping Perturbations (LPC-SWP) and Formant Energy Bandwidth (FEP-BWP) Perturbations

# How to use:

Step-1: Download the files and copy them to your SpeechBrain directory (https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec)

Step-2: Install dependencies listed in requirements.txt

Step-3: Run the following command to train ECAPA-TDNN using LPC-SWP and FEP-BWP augmentations
   
        python train_speaker_embeddings_pitch_mod_vtlp_lpcswp_bwp_fep_v2.py hparams/train_ecapa_tdnn.yaml

# License
We modify and publish the ECAPA-TDNN script in SpeechBrain under the Apache License, version 2.0.

# References:
We have utilized the utility functions in the Functions.py from https://github.com/hcy71o/LPC_Speech_Synthesis
