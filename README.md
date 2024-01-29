# License for using CSLU Protocols
Users can utilize the protocol under the strict licensing terms specified at https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/cslu.pdf, limiting its use exclusively to non-commercial linguistic education and research. Additionally, users are required to independently license LDC2007S18 directly from the Linguistic Data Consortium (LDC) at https://catalog.ldc.upenn.edu/LDC2007S18 before employing the protocol.

# Project

This work was partially supported by the Academy of Finland (Decision No. 349605, project [“SPEECHFAKES”]( https://uefconnect.uef.fi/en/group/speechfakes-generalized-voice-anti-spoofing-and-voice-biometrics/)).

# Citation
@book{singhchildaugment, <br>
author = "Vishwanath Pratap Singh and Md Sahidullah and Tomi Kinnunen", <br>
title = "ChildAugment: Data Augmentation Methods for Zero-Resource Children's Speaker Verification", <br>
publisher = "the journal of acoustical society of America ({JASA}) (under review)",<br>
year = 2024 }


# CSLU Evaluation and Developmental List

1. **CSLU_Trial_Finetune_Metadata/test_utterances_good_42k_abs:** Contains 42521 utterances from 451 Girls and  542 Boys speaker. We use these utterances as an Evaluation set for preparing the trial pairs for evaluating ASV systems.
2. **CSLU_Trial_Finetune_Metadata/dev_utterances_good_5k_abs:** Contains 5004 utterances from 60 Girls and  60 Boys speaker. We use these utterances as a Developmental set for finetuning the scoring methods.

# CSLU Trial List

We share the trail list containing gender and age information in **CSLU_Trial_Finetune_Metadata/trial_list_combined_with_age_gender_abs** from CSLU kids corpus (https://catalog.ldc.upenn.edu/LDC2007S18) under under the Apache License, version 2.0. We do not share any audio from CSLU Kids corpus. We only provide the metadata in the trial list containing the list of utterances, age, and gender information. Under Apache License, Version 2.0 the metadata is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

# Documentation on CSLU Trial List:

**<your_cslu_dir>:** Replace with the path of your downloaded CSLU Kids data directory

**utterance-1:** enrollment utterance

**utterance-2:** test utterance

**type-1:** linguistic content of utterance-1 [word: single word, sent: sentence, alphanum: alpha numeric sentence]

**type-2:** linguistic content of utterance-2

**target/non-target:** 0: non-target pair, 1: target pair

**age-1:** grade of speaker in utterance-1

**age-2:** grade of speaker in utterance-2

**gender-1:** gender of speaker in utterance-1

**gender-2:** gender of speaker in utterance-2

# Preparing SpeechBrain Style Trial

SpeechBrain supports the specific data directory structure. We provide the script for preparing SpeechBrain style data directory structure below:

1. cd CSLU_Trial_Finetune_Metadata
2. python prepare_vox_style_dir.py
3. python prepare_sb_style_trial.py > trial_list_combined_with_age_gender_vox_style
4. Now, **trial_list_combined_with_age_gender_vox_style** can be used for obtaining the EER in SpeechBrain

# LPC_SWP_FEP-BWP

Codes for LPC Segmental Warping Perturbations (LPC-SWP) and Formant Energy Bandwidth (FEP-BWP) Perturbations proposed in our under-review JASA paper: 

**Vishwanath Pratap Singh, Md Sahidullah, Tomi Kinnunen, "ChildAugment: Data Augmentation Methods for Zero-Resource Children’s Speaker Verification", The Journal of the Acoustical Society of America (under review).**

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
