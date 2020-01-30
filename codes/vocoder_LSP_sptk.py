#!/usr/bin/env python

'''
Python3 functions for a vocoder using MGC-LSP and MGLSADF

Written by Tamas Gabor CSAPO <csapot@tmit.bme.hu>
based on HTS data/Makefile and Training.pl from http://hts.sp.nitech.ac.jp
requires SPTK 3.8 or above from http://sp-tk.sourceforge.net
or install SPTK using 'apt install sptk'
2017 Jan 9


### example usage ###
from vocoder_LSP_sptk import encode, decode

basefilename = '2017jan03_alexa_003_sentence_0'
basefilename_out = basefilename + '_synthesized'

minF0 = 50
maxF0 = 300
frameLength = 512
frameShift = 110 # 10 ms
Fs_new = 11025

# Parameters of mel-cepstrum
order = 12
alpha = 0.42
stage = 3

(mgc_lsp_coeff, lf0) = encode(basefilename, Fs_new, frameLength, frameShift, order, alpha, stage, minF0, maxF0)
x_synthesized = decode(mgc_lsp_coeff, lf0, basefilename_out, Fs_new, frameLength, frameShift, order, alpha, stage)

'''

import subprocess
from subprocess import run
import numpy as np
import scipy.io.wavfile as io_wav




def encode(basefilename, Fs_new = 11025, frlen = 512, frshft = 200, order = 12, alpha = 0.42, stage = 3, minF0 = 50, maxF0 = 400):
    
    # from HTS Makefile
    # $(X2X) +sf $${raw} | $(PITCH) -H $(UPPERF0) -L $(LOWERF0) -p $(FRAMESHIFT) -s $${SAMPKHZ} -o 2 > lf0/$${base}.lf0
    #
    # $(X2X) +sf $${raw} | \
    # $(FRAME) -l $(FRAMELEN) -p $(FRAMESHIFT) | \
    # $(WINDOW) -l $(FRAMELEN) -L $(FFTLEN) -w $(WINDOWTYPE) -n $(NORMALIZE) | \
    # $(MGCEP) -a $(FREQWARP) -c $(GAMMA) -m $(MGCORDER) -l $(FFTLEN) -e 1.0E-08 -o 4 | \
    # $(LPC2LSP) -m $(MGCORDER) -s $${SAMPKHZ} $${GAINOPT} -n $(FFTLEN) -p 8 -d 1.0E-08 > mgc/$${base}.mgc; \
    
    # calculate MGC-LSP
    command = 'sox ' + basefilename + '.wav' + ' -t raw -r ' + str(Fs_new) + ' - ' + ' | x2x +sf | ' + \
              'sptk frame -l ' + str(frlen) + ' -p ' + str(frshft) + ' | ' + \
              'sptk window -l ' + str(frlen) + ' -L ' + str(frlen) + ' -w 0 -n 1 | ' + \
              'sptk mgcep -a ' + str(alpha) + ' -c ' + str(stage) + ' -m ' + str(order) + ' -l ' + str(frlen) + ' -e 1.0E-08 -o 4 | ' + \
              'sptk lpc2lsp -m ' + str(order) + ' -s ' + str(Fs_new / 1000) + ' -n ' + str(frlen) + ' -p 8 -d 1.0E-08 > ' + basefilename + '.mgclsp'
    # print(command)
    run(command, shell=True)
    
    # estimate pitch using SWIPE
    command = 'sox ' + basefilename + '.wav' + ' -t raw -r ' + str(Fs_new) + ' - ' + ' | x2x +sf | ' + \
              'sptk pitch -a 1 -H ' + str(maxF0) + ' -L ' + str(minF0) + ' -p ' + str(frshft) + ' -s ' + str(Fs_new / 1000) + ' -o 2 > ' + basefilename + '.lf0'
    # print(command)
    run(command, shell=True)
    
    # read files for output
    mgc_lsp_coeff = np.fromfile(basefilename + '.mgclsp', dtype=np.float32).reshape(-1, order + 1)
    lf0 = np.fromfile(basefilename + '.lf0', dtype=np.float32)
    
    return (mgc_lsp_coeff, lf0)

def decode(mgc_lsp_coeff, lf0, basefilename_out, Fs = 11050, frlen = 512, frshft = 200, order = 12, alpha = 0.42, stage = 3):
    
    # from HTS Training.pl / gen_wave
    #
    # MGC-LSPs -> MGC coefficients
    # $line = "$LSPCHECK -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " $lgopt -c -r 0.1 -g -G 1.0E-10 $mgc | ";
    # $line .= "$LSP2LPC -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " $lgopt | ";
    # $line .= "$MGC2MGC -m " . ( $ordr{'mgc'} - 1 ) . " -a $fw -c $gm -n -u -M " . ( $ordr{'mgc'} - 1 ) . " -A $fw -C $gm " . " > $gendir/$base.c_mgc";
    # shell($line);
    #
    # $line = "$SOPR -magic -1.0E+10 -EXP -INV -m $sr -MAGIC 0.0 $lf0 > $gendir/${base}.pit";
    #
    # $line = "$EXCITE -n -p $fs $gendir/$base.pit | ";
    # $line .= "$DFS -b $lfil | $VOPR -a $gendir/$base.unv | ";
    # $line .= "$MGLSADF -P 5 -m " . ( $ordr{'mgc'} - 1 ) . " -p $fs -a $fw -c $gm $mgc | ";
    # $line .= "$X2X +fs -o > $gendir/$base.raw";
    # shell($line);
    
    # write files for SPTK
    mgc_lsp_coeff.astype('float32').tofile(basefilename_out + '.mgclsp')
    lf0.astype('float32').tofile(basefilename_out + '.lf0')
    
    # MGC-LSPs -> MGC coefficients
    command = 'sptk lspcheck -m ' + str(order) + ' -s ' + str(Fs / 1000) + ' -c -r 0.1 -g -G 1.0E-10 ' + basefilename_out + '.mgclsp' + ' | ' + \
              'sptk lsp2lpc -m '  + str(order) + ' -s ' + str(Fs / 1000) + ' | ' + \
              'sptk mgc2mgc -m '  + str(order) + ' -a ' + str(alpha) + ' -c ' + str(stage) + ' -n -u ' + \
                      '-M '  + str(order) + ' -A ' + str(alpha) + ' -C ' + str(stage) + ' > ' + basefilename_out + '.mgc'
    # print(command)
    run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # MGLSADF synthesis based on pitch and MGC coefficients
    command = 'sptk sopr -magic -1.0E+10 -EXP -INV -m ' + str(Fs) + ' -MAGIC 0.0 ' + basefilename_out + '.lf0' + ' | ' + \
              'sptk excite -n -p ' + str(frshft) + ' | ' + \
              'sptk mglsadf -P 5 -m ' + str(order) + ' -p ' + str(frshft) + ' -a ' + str(alpha) + ' -c ' + str(stage) + ' ' + basefilename_out + '.mgc' + ' | ' + \
              'sptk x2x +fs -o | sox -c 1 -b 16 -e signed-integer -t raw -r ' + str(Fs) + ' - -t wav -r ' + str(Fs) + ' ' + basefilename_out + '.wav'
    # print(command)
    run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # read file for output
    (Fs_out, x_synthesized) = io_wav.read(basefilename_out + '.wav')
    
    return x_synthesized


