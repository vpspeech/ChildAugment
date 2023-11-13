import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
torch.manual_seed(1)
import librosa.display, librosa
import scipy
from Functions import *
import sys
import numpy
import wave
import math
from scipy.signal import lfilter, hamming
import cmath
import copy


def Levinson( w_sig,p):
        r_list = [Autocor(w_sig,i) for i in range(p)]
        b_list = [Autocor(w_sig,i) for i in range(1,p+1)]
        LPC = solve_toeplitz((r_list,r_list),b_list)
        return LPC
def make_matrix_X( x, p):
        n = len(x)
        # [x_n, ..., x_1, 0, ..., 0]
        xz = np.concatenate([x[::-1], np.zeros(p)])
        X = np.zeros((n - 1, p))
        for i in range(n - 1):
            offset = n - 1 - i
            X[i, :] = xz[offset : offset + p]
        return X
def Levinson1( x, p):
        b = x[1:]
        X = make_matrix_X(x, p)
        a = np.linalg.lstsq(X, b.T)[0]
        e = b - np.dot(X, a)
        g = np.var(e)
        return [a, g]
#    '''Get prediction, residual signal'''
def residual( windowed_signal, p):

        LPC = Levinson(windowed_signal,p)
        length = len(windowed_signal)
        prediction = np.zeros((length))
        win_sig = np.pad(windowed_signal, p)[:-p]
        for k in range(length):
            prediction[k] = np.sum(win_sig[k:k+p][::-1]*LPC)
        error = windowed_signal - prediction
        return prediction, error
#'''Get prediction, residual error for whole signal'''
def prediction( signal, window, p, overlap = 0.5):

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

def get_formant_num( rts,ang):
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
def residual( windowed_signal, p):

        LPC = Levinson(windowed_signal,p)
        length = len(windowed_signal)
        prediction = np.zeros((length))
        win_sig = np.pad(windowed_signal, p)[:-p]
        for k in range(length):
            prediction[k] = np.sum(win_sig[k:k+p][::-1]*LPC)
        error = windowed_signal - prediction
        return prediction, error,LPC

def synthesis_slpcw_fep_bwp1( signal,sr, window, p, overlap, c=0.68, th=0.3):
        length = len(window)
    #'''padding'''
        shift = int(length*overlap)
        if len(signal) % shift != 0:
            pad = np.zeros(shift - (len(signal) % shift))
            new_signal = np.append(signal, pad)
        else:
            new_signal = signal

        index = [j*shift for j in range(len(new_signal)//shift-1)]
        syn_signal = np.zeros((len(new_signal)))
        count=0
        for idx in index: #for each window
            w_sig = new_signal[idx:idx+len(window)]*window
            energy = np.sum(new_signal[idx:idx+len(window)]*new_signal[idx:idx+len(window)])
            if energy > 5:  # later will be extended for voice frame only using pitch
                try:
                    A1 =  Levinson(w_sig,p) #residual(w_sig, p)
                    A2 = np.insert(-A1, 0, 1)
                    error1 = lfilter([1], A2, new_signal[idx:idx+len(window)])#w_sig)
                    G = np.var(error1)
                    rts = numpy.roots(A1)
                    mag_warp = np.random.uniform(0.9,1.1)
                    warp1 = np.random.uniform(1.2,1.6)
                    warp2 = np.random.uniform(1.2,min(warp1,1.4))
                    warp3 = np.random.uniform(1.1,min(warp2,1.3))
                    warp4 = np.random.uniform(1.0,min(warp3,1.1))
                    for i in range(len(rts)):
                        mag = abs(rts[i])
                        ##if magnitude very close to unit circle
                        if mag > 0.98:
                            mag = mag*0.9
                        mag = mag * mag_warp
                        angle = cmath.phase(rts[i])
                        ang = angle
                        if round(angle,2)%3.14 !=0:
                            if int(get_formant_num(rts,ang)) == 1:
                                angle_new = angle * warp1
                            elif int(get_formant_num(rts,ang)) == 2:
                                angle_new = angle * warp2
                            elif int(get_formant_num(rts,ang)) == 3:
                                angle_new = angle * warp3
                            elif int(get_formant_num(rts,ang)) == 4:
                                angle_new = angle * warp4
                        else:
                            angle_new = angle
                        rts[i] = complex(mag*cmath.cos(angle_new),mag*cmath.sin(angle_new));
                    A_new = np.poly(rts)
                    tmp = lfilter([G],A_new,error1)
                    w_sig1 = copy.deepcopy(w_sig)
                    Flse=0
                    for kkk in range(len(tmp)):
                        if(abs(tmp[kkk]) < 1.5):
                            w_sig1[kkk] = tmp[kkk]
                            Flse=0
                        else:
                            Flse=1
                    if Flse ==0:
                        result = w_sig1
                    else:
                        result = w_sig
                        count = count + 1
                except:
                    result=w_sig
                    count = count + 1
            else: #Unvoiced
                count = count + 1
                result=w_sig
        #'''overlap-and-add'''
            syn_signal[idx:idx+length] += result*np.hamming(length)
        return syn_signal


def synthesis_slpcw1( signal,sr, window, p, overlap, c=0.68, th=0.3):
        length = len(window)
    #'''padding'''
        shift = int(length*overlap)
        if len(signal) % shift != 0:
            pad = np.zeros(shift - (len(signal) % shift))
            new_signal = np.append(signal, pad)
        else:
            new_signal = signal

        index = [j*shift for j in range(len(new_signal)//shift-1)]
        syn_signal = np.zeros((len(new_signal)))
        count=0
        for idx in index: #for each window
            w_sig = new_signal[idx:idx+len(window)]*window
            energy = np.sum(new_signal[idx:idx+len(window)]*new_signal[idx:idx+len(window)])
            if energy > 5: #later will be extended for voiced frame only using pitch 
                try:
                    A1 =  Levinson(w_sig,p) #residual(w_sig, p)
                    A2 = np.insert(-A1, 0, 1)
                    error1 = lfilter([1], A2, new_signal[idx:idx+len(window)])#w_sig)
                    G = np.var(error1)
                    rts = numpy.roots(A1)
                    mag_warp = np.random.uniform(0.9,1.1)
                    warp1 = np.random.uniform(1.2,1.6)
                    warp2 = np.random.uniform(1.2,min(warp1,1.4))
                    warp3 = np.random.uniform(1.1,min(warp2,1.3))
                    warp4 = np.random.uniform(1.0,min(warp3,1.1))
                    for i in range(len(rts)):
                        mag = abs(rts[i])
                        ##if magnitude very close to unit circle
                        if mag > 0.98:
                            mag = mag*0.9    
                        angle = cmath.phase(rts[i])
                        ang = angle
                        if round(angle,2)%3.14 !=0:
                            if int(get_formant_num(rts,ang)) == 1:
                                angle_new = angle * warp1
                            elif int(get_formant_num(rts,ang)) == 2:
                                angle_new = angle * warp2
                            elif int(get_formant_num(rts,ang)) == 3:
                                angle_new = angle * warp3
                            elif int(get_formant_num(rts,ang)) == 4:
                                angle_new = angle * warp4
                        else:
                            angle_new = angle
                        rts[i] = complex(mag*cmath.cos(angle_new),mag*cmath.sin(angle_new));
                    A_new = np.poly(rts)
                    tmp = lfilter([G],A_new,error1)
                    w_sig1 = copy.deepcopy(w_sig)
                    Flse=0
                    for kkk in range(len(tmp)):
                        if(abs(tmp[kkk]) < 1.5):
                            w_sig1[kkk] = tmp[kkk]
                            Flse=0
                        else:
                            Flse=1
                    if Flse ==0:
                        result = w_sig1
                    else:
                        result = w_sig
                        count = count + 1
                except:
                    result=w_sig
                    count = count + 1
            else: #Unvoiced
                count = count + 1
                result=w_sig
        #'''overlap-and-add'''
            syn_signal[idx:idx+length] += np.real(result)*np.hamming(length)
        return syn_signal    

def synthesis_slpcw( signal,sr, window, p, overlap, c=0.68, th=0.3):
        length = len(window)
    #'''padding'''
        shift = int(length*overlap)
        if len(signal) % shift != 0:
            pad = np.zeros(shift - (len(signal) % shift))
            new_signal = np.append(signal, pad)
        else:
            new_signal = signal

        index = [j*shift for j in range(len(new_signal)//shift-1)]
        syn_signal = np.zeros((len(new_signal)))
        count=0
        for idx in index: #for each window
            w_sig = new_signal[idx:idx+len(window)]*window
            energy = np.sum(new_signal[idx:idx+len(window)]*new_signal[idx:idx+len(window)])
            #print(idx)
            if energy > 5: #later will be extended for voiced frame only Voiced
                try:
                    A1 =  Levinson(w_sig,p) #residual(w_sig, p)
                    A2 = np.insert(-A1, 0, 1)
                    error1 = lfilter([1], A2, new_signal[idx:idx+len(window)])#w_sig)
                    G = np.var(error1)
                    rts = numpy.roots(A1)
                    mag_warp = np.random.uniform(0.9,1.1)
                    warp1 = np.random.uniform(1.2,1.4)
                    warp2 = np.random.uniform(1.2,min(warp1,1.4))
                    warp3 = np.random.uniform(1.1,min(warp2,1.3))
                    warp4 = np.random.uniform(1.0,min(warp3,1.1))
                    for i in range(len(rts)):
                        mag = abs(rts[i])
                        ##if magnitude very close to unit circle
                        if mag > 0.18:
                            mag = mag*0.09   
                        angle = cmath.phase(rts[i])
                        ang = angle
                        if round(angle,2)%3.14 !=0:
                            if int(get_formant_num(rts,ang)) == 1:
                                angle_new = angle * warp1
                            elif int(get_formant_num(rts,ang)) == 2:
                                angle_new = angle * warp2
                            elif int(get_formant_num(rts,ang)) == 3:
                                angle_new = angle * warp3
                            elif int(get_formant_num(rts,ang)) == 4:
                                angle_new = angle * warp4
                            else: 
                                angle_new = angle        
                        else:
                            angle_new = angle
                        rts[i] = complex(mag*cmath.cos(angle_new),mag*cmath.sin(angle_new));
                    A_new = np.poly(rts)
                    A_new2 = np.insert(-A_new, 0, 1)    
                    w,h = scipy.signal.freqz([0.05], A_new2, worN=length, fs = sr)
                    #plt.plot(np.log(abs(h)))
                    F_excitation = np.fft.fft(error1, length)  
                    F_result = F_excitation*h
                    result = np.fft.ifft(F_result, length)
                except:
                    result=w_sig
                    count = count + 1
            else: #Unvoiced
                count = count + 1
                result=w_sig
        #'''overlap-and-add'''
            syn_signal[idx:idx+length] += np.real(result)*np.hamming(length)
        return syn_signal

def synthesis_slpcw_fep_bwp( signal,sr, window, p, overlap, c=0.68, th=0.3):
        length = len(window)
    #'''padding'''
        shift = int(length*overlap)
        if len(signal) % shift != 0:
            pad = np.zeros(shift - (len(signal) % shift))
            new_signal = np.append(signal, pad)
        else:
            new_signal = signal

        index = [j*shift for j in range(len(new_signal)//shift-1)]
        syn_signal = np.zeros((len(new_signal)))
        count=0
        for idx in index: #for each window
            w_sig = new_signal[idx:idx+len(window)]*window
            energy = np.sum(new_signal[idx:idx+len(window)]*new_signal[idx:idx+len(window)])
            #print(idx)
            if energy > 10: #later will be extended for voiced frame only Voiced
                try:
                    A1 =  Levinson(w_sig,p) #residual(w_sig, p)
                    A2 = np.insert(-A1, 0, 1)
                    error1 = lfilter([1], A2, new_signal[idx:idx+len(window)])#w_sig)
                    G = np.var(error1)
                    rts = numpy.roots(A1)
                    mag_warp = np.random.uniform(0.9,1.1)
                    warp1 = np.random.uniform(1.2,1.4)
                    warp2 = np.random.uniform(1.2,min(warp1,1.4))
                    warp3 = np.random.uniform(1.1,min(warp2,1.3))
                    warp4 = np.random.uniform(1.0,min(warp3,1.1))
                    for i in range(len(rts)):
                        mag = abs(rts[i])
                        ##if magnitude very close to unit circle
                        if mag > 0.18:
                            mag = mag*0.09
                        mag = mag * mag_warp    
                        angle = cmath.phase(rts[i])
                        ang = angle
                        if round(angle,2)%3.14 !=0:
                            if int(get_formant_num(rts,ang)) == 1:
                                angle_new = angle * warp1
                            elif int(get_formant_num(rts,ang)) == 2:
                                angle_new = angle * warp2
                            elif int(get_formant_num(rts,ang)) == 3:
                                angle_new = angle * warp3
                            elif int(get_formant_num(rts,ang)) == 4:
                                angle_new = angle * warp4
                            else: 
                                angle_new = angle        
                        else:
                            angle_new = angle
                        rts[i] = complex(mag*cmath.cos(angle_new),mag*cmath.sin(angle_new));
                    A_new = np.poly(rts)
                    A_new2 = np.insert(-A_new, 0, 1)    
                    w,h = scipy.signal.freqz([0.05], A_new2, worN=length, fs = sr)
                    #plt.plot(np.log(abs(h)))
                    F_excitation = np.fft.fft(error1, length)
                    F_result = F_excitation*h
                    result = np.fft.ifft(F_result, length)
                    #print("here")
                except:
                    result=w_sig
                    count = count + 1
            else: #Unvoiced
                count = count + 1
                result=w_sig
        #'''overlap-and-add'''
            syn_signal[idx:idx+length] += np.real(result)*np.hamming(length)
        return syn_signal

def do_nothing(signal,sr, window, p, overlap, c=0.68, th=0.3):
        length = len(window)
    #'''padding'''
        shift = int(length*overlap)
        if len(signal) % shift != 0:
            pad = np.zeros(shift - (len(signal) % shift))
            new_signal = np.append(signal, pad)
        else:
            new_signal = signal
        index = [j*shift for j in range(len(new_signal)//shift-1)]
        syn_signal = np.zeros((len(new_signal)))
        for idx in index: #for each window
            w_sig = new_signal[idx:idx+len(window)]*window
            result=w_sig
        #'''overlap-and-add'''
            syn_signal[idx:idx+length] += np.real(result)*np.hamming(length)
        return syn_signal
