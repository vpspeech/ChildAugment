import os
import torchaudio.transforms as T
import shutil

with open("test_utterances_good_42k") as f11:
    meta = f11.readlines()

for lines in meta:
    line=lines.strip()
    spkid=line.split("/")[-2]
    uttid = line.split("/")[-1]
    p4="cslu_vox_style/wav/" +spkid + "/1"
    #print(p4)
    try:
        os.makedirs(p4)
    except:
        pass
    shutil.copy(line,p4)
    print(spkid + "/1/" + uttid)
