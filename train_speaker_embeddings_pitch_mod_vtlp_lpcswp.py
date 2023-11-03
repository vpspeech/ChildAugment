#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import librosa
import random
import torch
import torchaudio
import speechbrain as sb
import nlpaug.augmenter.audio as naa
import math
from scipy.signal import lfilter, hamming
import cmath
import copy
from Functions import *
from scipy.linalg import solve_toeplitz, toeplitz
from sympy import Symbol
#from librosa.effects import pitch_shift as pisi
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
#sys.path.append('/home/vsingh/speechbrain/recipes/VoxCeleb/SpeakerRec')
#from pitch_pertub import pitch_perturb1
import numpy as np
#def pitch_perturb1(wavt):
#    wava = wavt.detach().cpu().numpy()
#    wavpm = []
#    for i in range(len(wava)):
#        pstep = random.uniform(7.0, 17.0)
#        print(pstep)
#        print(np.asarray(wava[i]))
#        new_y = pisi(np.asarray(wava[i]),16000,pstep)
#        wavpm.append(new_y)
#    wavpm_t = torch.as_tensor(wavpm).float().to(self.device)
#    return wavpm_t
def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = eval(modulename + "." + shortname)
global_imports("librosa.effects","pitch_shift",True)
#def pitch_perturb1():
#    global_imports("librosa.effects","pitch_shift",True)
#    #wava = wavt.detach().cpu().numpy()
#    wavpm = []
#    y, sr = librosa.load('/home/vsingh/speechbrain/recipes/VoxCeleb/SpeakerRec/00001.wav')
#    for i in range(8):
#        pstep = random.uniform(7.0, 17.0)
#        print(len(y))
#        new_y = pitch_shift(y, sr=16000, n_steps=pstep)
#        wavpm.append(new_y)
#    wavpm_t = torch.as_tensor(wavpm).float().to(self.device)
#    return wavpm_t
class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """
     
    def Levinson(w_sig,p):
        r_list = [Autocor(w_sig,i) for i in range(p)]
        b_list = [Autocor(w_sig,i) for i in range(1,p+1)]
        LPC = solve_toeplitz((r_list,r_list),b_list)
        return LPC 
    def make_matrix_X(x, p):
        n = len(x)
        # [x_n, ..., x_1, 0, ..., 0]
        xz = np.concatenate([x[::-1], np.zeros(p)])
        X = np.zeros((n - 1, p))
        for i in range(n - 1):
            offset = n - 1 - i
            X[i, :] = xz[offset : offset + p]
        return X
    def Levinson1(x, p):
        b = x[1:]
        X = make_matrix_X(x, p)
        a = np.linalg.lstsq(X, b.T)[0]
        e = b - np.dot(X, a)
        g = np.var(e)
        #print(a,e)
        return [a, g]
#    '''Get prediction, residual signal'''
    def residual(windowed_signal, p):

        LPC = Levinson(windowed_signal,p)
        #print(LPC)
        length = len(windowed_signal)
        prediction = np.zeros((length))
        win_sig = np.pad(windowed_signal, p)[:-p]
        for k in range(length):
            prediction[k] = np.sum(win_sig[k:k+p][::-1]*LPC)
        error = windowed_signal - prediction
        return prediction, error
#'''Get prediction, residual error for whole signal'''
    def prediction(signal, window, p, overlap = 0.5):
    
    #'''padding'''
        shift = int(len(window)*overlap)
        if len(signal) % shift != 0:                    
            pad = np.zeros(shift - (len(signal) % shift))  
            new_signal = np.append(signal, pad)                  
        else:
            new_signal = signal
        index = (len(new_signal) // shift) -1 
    
    #'''make array'''
        whole_prediction = np.zeros((len(new_signal)),dtype = np.float64)
        whole_error = np.zeros((len(new_signal)),dtype = np.float64)
    
        for i in range(index):
            win_sig = new_signal[i*shift:i*shift+len(window)]*window #windowing
            prediction, error = residual(win_sig, p)
            whole_prediction[i*shift:i*shift+len(window)] += prediction
            whole_error[i*shift:i*shift+len(window)] += error
    
        return whole_prediction, whole_error

    def c_find(*roots):
        x = Symbol('x')
        whole =1
        for root in roots:
            whole *=(x-root)
        print(whole, 'f(x) =',whole.expand())
        
    def get_formant_num(rts,ang):
        count = 0
        if ang > 0:
            A = np.poly(rts)
            w, h = scipy.signal.freqz([1],A,worN = 2048, fs = 16000)
            num = int(np.ceil(((ang/3.14)*2048)))
            for i in range(1,num+1,1):
                forward_slope = abs(h[i+1]) -  abs(h[i])
                back_slope = abs(h[i]) -  abs(h[i-1])
                if forward_slope >=0 and back_slope < 0:
                    count = count + 1
                else:
                    if abs(20*np.log(abs(h[i])/abs(h[i+1]))) > 3:
                        count = count + 1
        return count     

#'''Get prediction, residual signal'''
    def residual(windowed_signal, p):
    
        LPC = Levinson(windowed_signal,p)
        length = len(windowed_signal)
        prediction = np.zeros((length))
        win_sig = np.pad(windowed_signal, p)[:-p]
        for k in range(length):
            prediction[k] = np.sum(win_sig[k:k+p][::-1]*LPC)
        error = windowed_signal - prediction
        #print(LPC)
        return prediction, error,LPC      

    def synthesis_slpcw(signal,sr, window, p, overlap, c=0.68, th=0.3):
    
        length = len(window)
    
    #'''padding'''
        shift = int(length*overlap)
        if len(signal) % shift != 0:                    
            pad = np.zeros(shift - (len(signal) % shift))  
            new_signal = np.append(signal, pad)                  
        else:
            new_signal = signal
        
        index = [j*shift for j in range(len(new_signal)//shift-1)]
    
    #'''pitch detection'''
        pitch_contour = medianfilter(Pitch_detect(
            new_signal, np.ones(length), sr, overlap=overlap, center_clip=c, th_=th))
    
    #'''voiced region index'''
        voiced_idx = [shift*i for i in range(len(pitch_contour)) if pitch_contour[i]>50]
    
        syn_signal = np.zeros((len(new_signal)))
 
    #'''make excitation'''
        print(len(index))
        count=0
        for idx in index: #for each window
            w_sig = new_signal[idx:idx+len(window)]*window
            if idx in voiced_idx: #Voiced
                try:   
                    A1 =  Levinson(w_sig,p) #residual(w_sig, p)
                #print(A1)
                #print(G) 
                    A2 = np.insert(-A1, 0, 1) 
           #print(A1) 
                    error1 = lfilter([1], A2, new_signal[idx:idx+len(window)])#w_sig)
                    G = np.var(error1) 
                    rts = numpy.roots(A1)
                    warp1 = np.random.uniform(1.2,1.6)
                    warp2 = np.random.uniform(1.2,min(warp1,1.4))
                    warp3 = np.random.uniform(1.1,min(warp2,1.3))
                    warp3 = np.random.uniform(1.0,min(warp3,1.1))
                    for i in range(len(rts)):
                        mag = abs(rts[i])
                        ##if magnitude very close to unit circle
                        if mag > 0.98:
                            mag = mag*0.9
                        angle = cmath.phase(rts[i])
                        if round(angle,2)%3.14 !=0: 
                            if int(get_formant_num(rts,ang)) == 1:
                                angle_new = angle * warp1
                            if int(get_formant_num(rts,ang)) == 2:
                                angle_new = angle * warp2
                            if int(get_formant_num(rts,ang)) == 3:
                                angle_new = angle * warp3
                            if int(get_formant_num(rts,ang)) == 4:
                                angle_new = angle * warp4    
                        else:
                            angle_new = angle  
                        rts[i] = complex(mag*cmath.cos(angle_new),mag*cmath.sin(angle_new));
                    A_new = np.poly(rts) 
                    tmp = lfilter([G],A_new,error1)
                    w_sig1 = copy.deepcopy(w_sig)
                    Flse=0 
                    for kkk in range(len(tmp)):
                        if(abs(tmp[kkk]) < 5):
                            w_sig1[kkk] = tmp[kkk]
                            Flse=0
                        else:
                            Flse=1
                    if Flse ==0:
                        result = w_sig1
                    else:
                        result = w_sig
                        count = count + 1  
           #print("this",result) 
                except:
                    result=w_sig 
                    count = count + 1        
           #print("that", result)
            else: #Unvoiced
                count = count + 1 
                result=w_sig
           #print("that", result) 
        #'''overlap-and-add'''
            syn_signal[idx:idx+length] += np.real(result)*np.hamming(length)
        #print(count)    
        return syn_signal
####################################################
####################################################
    def swp(self,wavt):
        wava = wavt.detach().cpu().numpy()
        wav_swp = []
        for i in range(len(wava)):
            #print(len(y))
            new_y = synthesis_slpcw(signal, 16000, np.hamming(256), p, 0.5, 0.6, 0.3)
            wav_swp.append(new_y)
        wav_swp = np.array(wavpm)
        wav_swpt = torch.as_tensor(wavpm).float().to(self.device)
        if wav_swpt.shape[1] > wavt.shape[1]:
            wav_swpt = wav_swpt[:, 0 : wavt.shape[1]]
        else:
            zero_sig = torch.zeros_like(wavt)
            zero_sig[:, 0 : wav_swp.shape[1]] = wav_swpt
            wav_swpt = zero_sig
        return wav_swpt


    def pitch_perturb1(self,wavt):
        wava = wavt.detach().cpu().numpy()
        wavpm = []
        for i in range(len(wava)):
            pstep = random.uniform(4.0, 15.0)
            #print(len(y))
            new_y = pitch_shift(wava[i], sr=16000, n_steps=pstep)
            wavpm.append(new_y)
        wavpm = np.array(wavpm)    
        wavpm_t = torch.as_tensor(wavpm).float().to(self.device)
        if wavpm_t.shape[1] > wavt.shape[1]:
            wavpm_t = wavpm_t[:, 0 : wavt.shape[1]]
        else:
            zero_sig = torch.zeros_like(wavt)
            zero_sig[:, 0 : wavpm_t.shape[1]] = wavpm_t
            wavpm_t = zero_sig
        return wavpm_t

        return wavpm_t
    def vtlp(self,wavt):
        wava = wavt.detach().cpu().numpy()
        wavpm = []
        for i in range(len(wava)):
            vtlp_aug = naa.VtlpAug(sampling_rate=16000, factor=(0.8, 1.2), zone=(0.0, 1.0), coverage=1.0, fhi=4800,)
            new_y_vtlp = vtlp_aug.augment(wava[i])
            wavpm.append(new_y_vtlp[0])
        wavpm = np.array(wavpm)
        wavpm_t = torch.as_tensor(wavpm).float().to(self.device)
        if wavpm_t.shape[1] > wavt.shape[1]:
            wavpm_t = wavpm_t[:, 0 : wavt.shape[1]]
        else:
            zero_sig = torch.zeros_like(wavt)
            zero_sig[:, 0 : wavpm_t.shape[1]] = wavpm_t
            wavpm_t = zero_sig
        return wavpm_t

        return wavpm_t
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        #print(wavs, wavs[1], wavs[2]) 
        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_pp=self.pitch_perturb1(wavs)
            wavs_vtlp=self.vtlp(wavs)
            wavs_swp=self.swp(wavs) 
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            wavs_aug_tot.append(wavs_pp)
            wavs_aug_tot.append(wavs_vtlp)
            wavs_aug_tot.append(wavs_swp)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            #print(self.n_augment)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        
#######################################################################
        #feats_tmp1 = feats.detach().cpu().numpy()
        #feats_tmp2 = feats.detach().cpu().numpy()
        #feats_tmp = np.concatenate((feats_tmp1, feats_tmp2), axis=0)
        #feats2 = torch.from_numpy(feats_tmp).float().to(self.device)

        #np.savetxt('feats.out', feats_tmp, fmt='%f') can't save a 3d array either convert in n 2d array and write 1 by one
        #print(len(feats_tmp),len(feats_tmp[0]),len(feats_tmp[11]), len(feats_tmp[0][0]))
        #print(feats2)
        #lens = torch.cat([lens] * 2)
#########################################################################
        feats = self.modules.mean_var_norm(feats, lens)
        
        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = torch.cat([spkid] * self.n_augment, dim=0)
            #spkid = torch.cat([spkid] * 2, dim=0)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
