# LPC_SWP_FEP-BWP
Codes for LPC Segmental Warping Perturbations (LPC-SWP) and Formant Energy Bandwidth (FEP-BWP) Perturbations proposed in our under-review JASA paper: 
      # Vishwanath Pratap Singh, Md Sahidullah, Tomi Kinnunen, "ChildAugment: Data Augmentation Methods for Zero-Resource Children’s Speaker Verification", The Journal of the Acoustical Society of America (under review).

# How to use:

Step-1: Download the files and copy them to your SpeechBrain directory (https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec)

Step-2: Install dependencies listed in requirements.txt

Step-3: Run the following command to train ECAPA-TDNN using LPC-SWP and FEP-BWP augmentations
   
        python train_speaker_embeddings_pitch_mod_vtlp_lpcswp_bwp_fep_v2.py hparams/train_ecapa_tdnn.yaml

# License
We modify and publish the ECAPA-TDNN script in SpeechBrain under the Apache License, version 2.0. 
SpeechBrain is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Note that this project has no connection to the Apache Foundation, other than that we use the same license terms.

# References:
We have utilized the utility functions in the Functions.py from https://github.com/hcy71o/LPC_Speech_Synthesis

We modify the ECAPA-TDNN training script published in Speechbrain (https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py)
