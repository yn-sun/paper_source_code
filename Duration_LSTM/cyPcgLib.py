#!/usr/bin/python
import wave,struct,numpy,numpy.fft
#import scipy.signal,matplotlib.pyplot
import numpy as np
from numpy import interp, mean, std, fft, angle, argsort
from scipy.signal import butter, lfilter
import sys
import random
import cyMetricLib
import os
#From spy###############################################
def wavread(wfname,stime=0,etime=0):
    '''
    Returns the contents of the wave file and its sampling frequency.
    Input to the function is the path to the wave file

    wfname is the input wave file path.
    The input for stime and etime should be seconds.

    wavread also reads a segment of the wave file.
    stime and etime determines for what sections in the wave file
    should the content be read. By default they are set to zero.
    When both stime and etime are zero the function reads the whole
    wave file.
    '''
    wavfp = wave.open(wfname,'r')
    fs = wavfp.getframerate() #to obtain the sampling frequency
    sig = []
    wavnframes = wavfp.getnframes()
    sframe = int(stime * fs)                               #input is in seconds and so they are to be convereted
    wavfp.setpos(sframe)                                   #to frame count
    eframe = int(etime * fs)
    if eframe == 0:
        eframe = wavnframes
    for i in range(sframe,eframe):                          #In wave format 2 bytes are used to represent one value
        hexv = wavfp.readframes(1)                          #readframes is an iterative function which reads 2 bytes
        sig.append(float(struct.unpack('h',hexv)[0]))
    wavfp.close()
    return numpy.array(sig,dtype='float'),fs

def wavwrite(sig,fs,wfname):
    '''
    Following function is used to create a wave file provided
    the signal with its sampling frequency. It creates a
    standard wave audio, PCM, 16 bit, mono with fs Hertz.
    It takes two types of formats
    1. If the amplitude ranges from -1 to 1 it scales the
       amplitudes of the wave files
    2. Else it converts the floats to integers and then writes
       the wave file

    Suggested normalization would be as follows:
    wav = wav/(0.0001 + max(wav))
    '''
    if max(sig) <= 1 and min(sig) >= -1:
        print('[Warning]: Scaling signal magnitudes')
        sampwidth = 2 #nunber of bytes required to store an integer value
        max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)
        sig = sig * max_amplitude
    sig = numpy.array(sig,dtype='int')
    wavfp = wave.open(wfname,'w')
    wavfp.setparams((1,2,fs,sig.size,'NONE','not compressed')) #setting the params for the wave file
    for i in sig:                                              #params: nchannels, sampwidth(bytes), framerate, nframes, comptype, compname
        hvalue = struct.pack('h',i)                            #only accepts 2 bytes => hex value
        wavfp.writeframes(hvalue)
    wavfp.close()

def dumpfeats(feats,fname):
    """
    To dump the feature files in a file.
    Input Parameters:
    feats  :  Feature Vectors
    fname  :  File name
    """

    fp = open(fname,'w')
    for i in range(0,len(feats)):
        feats_str = ''
        for j in range(0,len(feats[i])):
            feats_str = feats_str + ' ' + str(feats[i][j])
        feats_str = feats_str.strip()
        fp.write(feats_str+'\n')
    fp.close()

def deltac(feats,DELWIN=2,ACCWIN=2):
    """
    Following function computes the delta and delta coefficients
    given a sequence of vectors. Most speech applications perform
    well with delta and delta coefficients. Following is the
    equation to compute the delta and delta coefficients

                    WIN
          d[t] = SUMMATION k * (c[t+k] - c[t-k])
                   k = 1
                 -------------------------------
                               WIN
                       2 * SUMMATION k^2
                              k = 1

     WIN can be DELWIN or ACCWIN. DELWIN is the window to compute
     the delta coefficients and ACCWIN is the window to compute the
     acceleration coefficients.

     Note: Acceleration coefficients are computed using the delta
     coefficients
    """
    dac = feats[:] #to store the original feats + delta + acceleration coefficients
    delta = [] #stores only the delta coefficients
    ac = [] #acceration coefficients

    for i in range(0,DELWIN): #appending start and ending coefficients
        feats = numpy.vstack([feats,feats[-1]])
        feats = numpy.vstack([feats[0],feats])

    dnorm = numpy.arange(1,DELWIN+1) #to normlaize the delta coefficients
    dnorm = 2 * sum(dnorm * dnorm)

    for i in range(DELWIN,len(feats)-DELWIN):
        d = numpy.zeros(len(feats[i])) #stores the delta coefficients
        for j in range(1,DELWIN+1):
            d += j * (feats[i+j] - feats[i-j])
        d = d/dnorm
        delta.append(d)

    dac = numpy.concatenate((dac,delta),1)

    for i in range(0,ACCWIN): #appending start and ending coefficients
        delta = numpy.vstack([delta,delta[-1]])
        delta = numpy.vstack([delta[0],delta])

    dnorm = numpy.arange(1,ACCWIN+1)
    dnorm = 2 * sum(dnorm * dnorm)

    for i in range(ACCWIN,len(delta)-ACCWIN):
        d = numpy.zeros(len(delta[i])) #stores the acceleration coefficients
        for j in range(1,ACCWIN+1):
            d += j * (delta[i+j] - delta[i-j])
        d = d/dnorm
        ac.append(d)
    return numpy.concatenate((dac,ac),1)



class Cepstrum:
    """
    Following class consists of functions to generate the
    cepstral coefficients given a speech signal.
    Following are the functions it supports
    1. Cepstral Coefficients
    2. LP Cepstral Coefficients
    """

    def __init__(self):
        self.lpco = LPC()

    def cep(self,sig,corder,L=0):
        """
        Following function require a windowed signal along
        with the cepstral order.
        Following is the process to calculate the cepstral
        coefficients

                 x[n] <--> X[k] Fourier Transform of the
                                signal
                 c = ifft(log(|X[k]|)
             where c is the cepstral coefficients

        For the liftering process two procedures were implemented
        1. Sinusoidal
        c[m] = (1+(L/2)sin(pi*n/L))c[m]
        2. Linear Weighting

        By default it gives linear weighted cepstral coefficients.
        To obtain raw cepstral coefficients input L=None.

        For any other value of L it performs sinusoidal liftering.

        Note: The input signal is a windowed signal,
        typically hamming or hanning
        """
        c = numpy.fft.ifft(numpy.log(numpy.abs(numpy.fft.fft(sig))))
        c = numpy.real(c)
        if len(c) < corder:
            print('[Warning]: Lenght of the windowed signal < cepstral order')
            print('[Warning]: cepstral order set to length of the windowed signal')
            corder = len(sig)

        if L == None:           #returning raw cepstral coefficients
            return c[:corder]
        elif L == 0:      #returning linear weighted cepstral coefficients
            for i in range(0,corder):
                c[i] = c[i] * (i+1)
            return c[:corder]

        #cep liftering process as given in HTK
        for i in range(0,corder):
            c[i] = (1 + (float(L)/2) * numpy.sin((numpy.pi * (i+1))/L)) * c[i]
        return c[:corder]

    def cepfeats(self,sig,fs,wlent,wsht,corder,L=None):
        """
        Following function is used to generate the cepstral coefficients
        given a speech signal. Following are the input parameters
        1. sig: input signal
        2. fs: sampling frequency
        3. wlent: window length
        4. wsht: window shift
        5. corder: cepstral order
        """
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000

        sig = numpy.append(sig[0],numpy.diff(sig))
        sig = sig + 0.001
        ccs = []
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            c = self.cep(smooth_signal,corder,L)
            ccs.append(c)
        return numpy.array(ccs)

    def lpccfeats(self,sig,fs,wlent,wsht,lporder,corder,L=None):
        '''
        Following fucntion in turn calls the lpccfeats from
        the LPC class to obtain the cepstral coefficients.
        Following are the input parameters
        sig: input speech signal
        fs: its sampling frequency
        wlent: window length in milli seconds
        wsht: window shift in milli seconds.
        lporder: self explanatory
        corder: no. of cepstral coefficients
        (read the lpcc documentation for the description about
        the features)
        L(ceplifter) to lift the cepstral coefficients. Read the
        documentation for the LPCC for the liftering process

        Function returns only the cepstral coefficients
        '''
        Gs,nEs,lpcs,lpccs = self.lpco.lpccfeats(sig,fs,wlent,wsht,lporder,corder,L)
        return lpccs


class FIRFilters:
    '''Class consists of building the following FIR filters
       a. Low Pass Filter
       b. High Pass Filter
       c. Band Pass Filter
       d. Band Reject Filter

       In general the filter is denoted as follows

                 jw               -jw            -jmw
          jw  B(e)    b[0] + b[1]e + .... + b[m]e
       H(e) = ---- = ------------------------------------
                 jw               -jw            -jnw
              A(e)    a[0] + a[2]e + .... + a[n]e


       Where the roots of the numerator is denoted as zeros
       and that of the denominator as poles. In FIR filters
       are represented only via the zeros. Hence we only
       compute the coefficients "b" as shown in the above
       equation.

       Class also consists of funtions to plot filters to view
       the frequency response
    '''
    def __init__(self):
        pass

    def low_pass(self,M,cfreq,wtype='blackmanharris'):
        """
        Following are the required parameters by the low pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreq is the cutoff frequency. Make sure that the
           cfreq lies between 0 and 1 with 1 being Nyquist
           frequency.
        3. wtype is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
        """
        lb = scipy.signal.firwin(M,cutoff=cfreq,window=wtype)
        return lb

    def high_pass(self,M,cfreq,wtype='blackmanharris'):
        """
        Following are the required parameters by the high pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreq is the cutoff frequency. Make sure that the
           cfreq lies between 0 and 1 with 1 being Nyquist
           frequency.
        3. win_type is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
       The high pass filter is obtained by first obtaining the impulse
       response of a low pass filter. A more detail explanation is given
       Scientists and Engineers Guide to Digital Signal Processing,
       chapter 14-16.
       """
        lb = self.low_pass(M,cfreq,wtype) #to obtain the impulse response using the low pass filter
                                                  #and then reversing
        hb = -1 * lb
        hb[M/2] = 1 + hb[M/2]
        return hb

    def band_reject(self,M,cfreqb,cfreqe,wtype='blackmanharris'):
        """
        Following are the required parameters by the high pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreqb and cfreqe are the frequency ranges that are to be suppressed.
           Make sure that the cfreqb and cfreqe lies between 0 and 1 with 1
           being Nyquist frequency.
        3. wtype is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
       The band reject filter is obtained by first obtaining by combining the
       low pass filter and the high pass filter responses. A more detail explanation
       is given Scientists and Engineers Guide to Digital Signal Processing,
       chapter 14-16.
       """
        lb = self.low_pass(M,cfreqb,wtype) #coefficients from the low pass filter
        hb = self.high_pass(M,cfreqe,wtype) #coefficients from the high pass filter

        brb = lb + hb
        return brb

    def band_pass(self,M,cfreqb,cfreqe,wtype='blackmanharris'):
        """
        Following are the required parameters by the high pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreqb and cfreqe are the frequency ranges that are to be captured.
           Make sure that the cfreqb and cfreqe lies between 0 and 1, with 1
           being Nyquist frequency.
        3. wtype is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
       The band pass filter is obtained by using the band reject filter. A more
       detail explanation is given Scientists and Engineers Guide to Digital
       Signal Processing, chapter 14-16.
       """

        brb = self.band_reject(M,cfreqb,cfreqe,wtype)
        bpb = -1 * brb
        bpb[M/2] = 1 + bpb[M/2]
        return bpb

    def fsignal(self,sig,b):
        """
        Following function outputs the filtered signal
        using the FIR filter coefficients b.
        """
        fsig = scipy.signal.lfilter(b,[1],sig)
        M = len(b)          #fir filters has a delay of (M-1)/2
        fsig[0:(M-1)/2] = 0 #setting the delay values to zero
        return fsig


    def plotResponse(self,b):
        """
        Following function plots the amplitude and phase response
        given the impulse response of the FIR filter. The impulse
        response of the FIR filter is nothing but the numerator (b)
        of the transfer function.
        """
        w,h = scipy.signal.freqz(b)
        h_db = 20.0 * numpy.log10(numpy.abs(h))
        ph_angle = numpy.unwrap(numpy.angle(h))

        fig = matplotlib.pyplot.figure()
        subp1 = fig.add_subplot(3,1,1)
        subp1.text(0.05,0.95,'Frequency Response',transform=subp1.transAxes,fontsize=16,fontweight='bold',va='top')
        subp1.plot(w/max(w),numpy.abs(h))
        subp1.set_ylabel('Magnitude')


        subp2 = fig.add_subplot(3,1,2)
        subp2.text(0.05,0.95,'Frequency Response',transform=subp2.transAxes,fontsize=16,fontweight='bold',va='top')
        subp2.plot(w/max(w),h_db)
        subp2.set_ylabel('Magnitude (DB)')
        subp2.set_ylim(-150, 5)

        subp3 = fig.add_subplot(3,1,3)
        subp3.text(0.05,0.95,'Phase',transform=subp3.transAxes,fontsize=16,fontweight='bold',va='top')
        subp3.plot(w/max(w),ph_angle)
        subp3.set_xlabel('Normalized Frequency')
        subp3.set_ylabel('Angle (radians)')

        fig.show()


class SignalMath:
    '''
    Contains quite commonly used mathematical operations
    needed for signal processing techniques.
    '''
    def __init__(self):
        pass

    def movavg(self,sig,fs,wlent):
        '''
        The following function is a mathematical representation for
        a moving average. The input to the function is the signal and
        the window length in milli seconds.
        Following is the mathematical equation for the moving average.

        y[n] = 1/(2*N+1)sum(sig[i+n]) for i = -N to +N

        y[n], sig[n] are both discrete time signals

        sig is the signal and wlent is the window length in milli seconds
        '''
        sigLen=len(sig)
        wlenf = (wlent * fs)/1000
        window = numpy.array([1] * wlenf)
        avg = numpy.convolve(sig,window,mode='full')
        #print "avg.shape", avg.shape
        avg = avg[(window.size/2) - 1:avg.size - (window.size/2)]
        norm = numpy.convolve(window,numpy.array([1] * avg.size),mode='full')
        norm = norm[(window.size/2) - 1:norm.size - (window.size/2)]
        #norm=norm[0:len(avg)]
        return numpy.divide(avg[0:sigLen],norm[0:sigLen])


class ZeroFreqFilter:
    '''
    Containg functions required to obtain the Zero Frequncy Filtered Signal.
    Following is the procedure to obtain the Zero Frequency Filtered Signal:

    Let s[n] be the input signal then

    1. Bias Removal:

            x[n] = s[n] - s[n-1]

       Used to remove any dc or low frequency bias that might have been
       captured during recording.

    2. Zero Frequency Filtering:
       Pass the signal twice through a zero frequency resonator. That is
       containes two poles on the unit circle with zero frequency.

           y1[n] = -1 * SUMMATION( a(k) * y[n-k] ) + x[n]   where k = 1,2
                           k
           y2[n] = -1 * SUMMATION( a(k) * y2[n-k] ) + y1[n] where k = 1,2
                           k
                        where a(1) = -2 and a(2) = 1

       The above two operations can be obtained by finding the cumulative
       sum of the signal x[n] four times.

    3. Trend Removal:

           y[n] = y2[n] - (moving average of y2[n] of windlow lenght n)

       The moving average function is clrealy mentioned in the Math class.
       Window length n is important. The choice of the window length is not
       very critical as long as it is in the range of about 1 to 2 times
       the average pitch period.
    '''
    def __init__(self):
        self.sm = SignalMath()

    def getZFFSignal(self,sig,fs,wlent=30,wsht=20,mint=3):
        '''
        Following function returns the Zero Frequency Filtered Signal.
        Following are the steps involved in generating the Zero
        Frequency Filtered Signal:
        1. Bias Removal
        2. Zero Frequency Resonator
        3. Average Pitch Estimation
        4. Trend Removal

        sig is the samples in the wave file.
        fs is the sampling frequency in Hz.
        wlent is the window length in milli-seconds.
        wsht is the window shift in milli-seconds.
        mint is the minimum pitch value in milli-seconds.
        '''
        sig = sig - (sum(sig)/sig.size)
        dsig = numpy.diff(sig) #bias removal
        dsig = numpy.append(dsig,dsig[-1])
        dsig = numpy.divide(dsig,max(abs(dsig))) #normalization
        csig = numpy.cumsum(numpy.cumsum(numpy.cumsum(numpy.cumsum(dsig)))) #zero frequency resonator
        wlenpt = self.__avgpitch(dsig,fs,wlent,wsht,mint) #estimating the average pitch
        wlenpf = int((wlenpt * fs)/1000) #converting pitch in milli seconds to pitch in number of samples
        tr = numpy.subtract(csig,self.sm.movavg(csig,fs,wlenpt)) #trend removal
        tr = numpy.subtract(tr,self.sm.movavg(tr,fs,wlenpt)) #trend removal
        tr = numpy.subtract(tr,self.sm.movavg(tr,fs,wlenpt)) #trend removal
        tr = numpy.subtract(tr,self.sm.movavg(tr,fs,wlenpt)) #trend removal
        for i in range(dsig.size - (wlenpf*3) - 1,dsig.size): # To remove the trailing samples. Without removing the trailing samples
            tr[i] = 0                                        # we cannot view the ZFF signal as they have huge values
        tr = numpy.divide(tr,max(abs(tr)))                   # normalizing
        return tr

    def __avgpitch(self,sig,fs,wlent,wsht,mint=3):
        '''
        Gets the average pitch from the input signal. The window length
        is for the remove trend function. Window length anything between
        pitch and twice the pitch should be adequate.

        sig is the samples in the wave file.
        fs is the sampling frequency in Hz.
        wlent is the window length in milli-seconds.
        wsht is the window shift in milli-seconds.
        mint is the minimum pitch value in milli-seconds.
        '''
        wlenf = (wlent * fs)/1000
        wshtf = (wsht * fs)/1000
        nof = (sig.size - wlenf)/wshtf
        pitch = []
        for i in range(0,nof):#block processing for obtaining pitch from each of the window
            sigblock = sig[i*wshtf:i*wshtf + wlenf]
            pitch.append(self.__getpitch(sigblock,fs,mint))
        pitch = numpy.array(pitch)
        pbins = numpy.arange(3,20,2)#min pitch is 3 msec and maximum is 20 msec. The bin rages from 3-5,5-7,..
        phist = numpy.histogram(pitch,bins=pbins)[0]#plotting histogram for each of the pitch values
        prange_index = 0                            #to see the most commonly occuring pitch value
        prange_count = 0
        for i in range(0,phist.size):
            if phist[i] > prange_count:
                prange_count = phist[i]
                prange_index = i
        avgpitch = (pbins[prange_index] + pbins[prange_index+1])/2#finding the average pitch value
        return avgpitch

    def __getpitch(self,sig,fs,mint=3):
        '''
        To find the pitch given a speech signal of some window.
        sig is the samples in the wave file.
        fs is the sampling frequency in Hz.
        mint is the minimum pitch value in milli-seconds.
        '''
        minf = (mint * fs)/1000 #to convert into number of frames/samples
        cor = numpy.correlate(sig,sig,mode='full')
        cor = cor[cor.size/2:]#auto correlation is symmetric about the y axis.
        cor = cor/(max(abs(cor)) + 0.0001)#normalizing the auto correlation values
        dcor = numpy.diff(cor)#finding diff
        dcor[0:minf] = 0#setting values of the frames below mint to be zero
        locmax = numpy.array([1] * dcor.size) * (dcor > 0)
        locmin = numpy.array([1] * dcor.size) * (dcor <= 0)
        locpeaks = numpy.array([2] * (dcor.size - 1)) * (locmax[0:locmax.size - 1] + locmin[1:locmin.size] == 2)#to get the positive peaks
        maxi,maxv = self.__getmax(cor,locpeaks)
        return (maxi * 1000.0)/fs

    def __getmax(self,src,peaks):
        '''
        To get peak which has the maximum value.
        '''
        maxi = 0
        maxv = 0
        for i in range(0,peaks.size):           #diff values will be one sample less than the original signal
            if src[i] > maxv and peaks[i] == 2: #consider only the diff values in the for loop
                maxi = i
                maxv = src[i]
        return maxi,maxv

class SegmentSpeech:
    '''
    Following class provides functions, which are useful segment speech.
    All the functions utilize the Zero Frequency Filtered signal to
    segment the speech.

    Functions of format segmentxc represents a function to segment
    the speech into x categories.
    '''
    def __init__(self):
        self.sm = SignalMath()

    def vnv(self,zfs,fs,theta=2.0,wlent=30):
        '''
        To obtain the voiced regions in the speech segment.
        Following are the input parameters
        1. zfs is the Zero Frequency Filtered Signal
        2. fs is the sampling rate
        3. wlent is the window length required for the moving average.
        '''
        zfse = 1.0 * zfs * zfs #squaring each of the samples: to find the ZFS energy.
        zfse_movavg = numpy.sqrt(self.sm.movavg(zfse,fs,wlent)) #averaging across wlent window

        zfse_movavg = zfse_movavg/max(zfse_movavg) #normalzing
        avg_energy = sum(zfse_movavg)/zfse_movavg.size #average energy across all the window.
        voicereg = zfse_movavg * (zfse_movavg >= avg_energy/theta) #selecting segments whose energy is higher than the average.
        return voicereg

    def vnvNoMax(self, zfs, fs, theta=2.0, wlent=30):
        '''
        To obtain the voiced regions in the speech segment.
        Following are the input parameters
        1. zfs is the Zero Frequency Filtered Signal
        2. fs is the sampling rate
        3. wlent is the window length required for the moving average.
        '''
        zfse = 1.0 * zfs * zfs  # squaring each of the samples: to find the ZFS energy.

        zfse_movavg = numpy.sqrt(self.sm.movavg(zfse, fs, wlent))  # averaging across wlent window

        #zfse_movavg = zfse_movavg / max(zfse_movavg)  # normalzing
        #avg_energy = sum(zfse_movavg) / zfse_movavg.size  # average energy across all the window.
        #voicereg = zfse_movavg * (
        #zfse_movavg >= avg_energy / theta)  # selecting segments whose energy is higher than the average.
        return zfse
    def vnv_shannon_energy(self, zfs, fs, theta=2.0, wlent=30):
        '''
        To obtain the voiced regions in the speech segment.
        Following are the input parameters
        1. zfs is the Zero Frequency Filtered Signal
        2. fs is the sampling rate
        3. wlent is the window length required for the moving average.
        '''
        #zfse = 1.0 * zfs * zfs  # squaring each of the samples: to find the ZFS energy.
        zfse=-1.0 *zfs * zfs*numpy.log( 1.0 *zfs * zfs+0.000001)#shannon energy
        zfse_movavg = numpy.sqrt(self.sm.movavg(zfse, fs, wlent))  # averaging across wlent window
        # zfse_movavg = zfse_movavg / max(zfse_movavg)  # normalzing
        # avg_energy = sum(zfse_movavg) / zfse_movavg.size  # average energy across all the window.
        # voicereg = zfse_movavg * (zfse_movavg >= avg_energy / theta)  # selecting segments whose energy is higher than the average.
        return zfse
    def vnv_shannon_entropy(self, zfs, fs, theta=2.0, wlent=30):
        '''
        To obtain the voiced regions in the speech segment.
        Following are the input parameters
        1. zfs is the Zero Frequency Filtered Signal
        2. fs is the sampling rate
        3. wlent is the window length required for the moving average.
        '''
        #zfse = 1.0 * zfs * zfs  # squaring each of the samples: to find the ZFS energy.
        #zfse=-1.0 *zfs * zfs*numpy.log( 1.0 *zfs * zfs+0.000001)#shannon energy
        zfs=zfs
        zfse =-numpy.absolute(zfs) * numpy.log(numpy.absolute(zfs)+0.000001)
        zfse_movavg = numpy.sqrt(self.sm.movavg(zfse, fs, wlent))  # averaging across wlent window
        zfse_movavg = zfse_movavg / max(zfse_movavg)  # normalzing
        avg_energy = sum(zfse_movavg) / zfse_movavg.size  # average energy across all the window.
        voicereg = zfse_movavg * (zfse_movavg >= avg_energy / theta)  # selecting segments whose energy is higher than the average.
        return voicereg
    def __zerocross(self,zfs):
        '''
        To obtain the postive zero crossing from the
        Zero Frequency Filtered Signal.
        '''
        zc = numpy.array([1]) * (zfs >= 0)
        dzc = numpy.diff(zc)
        zcst = numpy.diff(zfs) * (dzc > 0)
        return numpy.append(zcst,0)

    def getGCI(self,zfs,voicereg):
        '''
        To obtain the Glottal Closure Instants and the strength of
        excitations. We obtain the Strength of excitation by taking
        the derivative at postive zero crossing from the ZFS.
        Voiced regions are used to remove spurious GCI.
        '''
        #voicereg=voicereg[0:len(zfs)]
        gci = self.__zerocross(zfs) * (voicereg > 0)#only considering the GCI in regions where it was
        return gci                   #detected as voiced regions by the vnv function

    def segment2c(self,gci,fs,sildur=300):
        '''
        The following function returns a two category segmentation of
        speech, speech and silence region

        Following are the input parametrs
        1. gci is the glottal closure instants.
        2. sildur is the minimum duration without gci to
        classify the segment as sil
        '''
        stime = 0#starting time
        etime = 0#end time
        wtime = (gci.size * 1000.0)/fs#total time of the wave file
        wlab = []#array containing the lab information for the wave file
        i = 0
        vflag = False#flag to keep track of voiced sounds

        while i < gci.size:
            if gci[i] > 0:
                etime = (i * 1000.0)/fs
                if (etime - stime) >= sildur: #to check whether its a silence region or not
                    if stime == 0 and etime == wtime:
                        stime = stime
                        etime = etime
                    elif stime == 0:
                        stime = stime
                        etime = etime - 50
                    elif etime == wtime:
                        stime = stime + 50
                        etime = etime
                    else:
                        stime = stime + 50
                        etime = etime - 50
                    wlab.append((stime,etime,'SIL')) #to make sure that the end unvoiced sounds are
                    stime = etime                    #not classified as silence
                else:
                    stime = etime
            i += 1

        #fixing the trailing silence. Because the trailing silence might not end with an epoch.
        if etime != wtime:
            if (wtime - etime) >= sildur:
                wlab.append((etime+50,wtime,'SIL'))

        #some times there might not be any silence in the wave file
        if len(wlab) == 0:
            wlab.append((0,wtime,'SPH'))

        #fixing the missing time stamps
        #the above loop only puts time stamps for the silence regions
        cwlab = []

        for i in range(0,len(wlab)):
            tlab = wlab[i]
            stime = tlab[0]

            etime = tlab[1]
            if len(cwlab) == 0:
                if stime == 0:
                    cwlab.append(tlab[:])
                else:
                    cwlab.append((0,stime,'SPH'))
                    cwlab.append(tlab[:])
            else:
                if cwlab[-1][1] == stime:
                    cwlab.append(tlab[:])
                else:
                    cwlab.append((cwlab[-1][1],stime,'SPH'))
                    cwlab.append(tlab[:])
        if wlab[-1][1] != wtime:
            cwlab.append((wlab[-1][1],wtime,'SPH'))
        return cwlab

    def segmentvnvc(self,gci,fs,uvdur=18):
        '''
        The following function returns a three category segmentation of
        speech, namely, voiced(VOI), unvoiced(UNV).

        Following are the input parametrs
        1. gci is the glottal closure instants.
        2. fs is the sampling rate.
        3. uvdur is the maximum duration between any two GCI
        to classify them at VOI or else they would be
        classified as UNV.
        '''

        stime = 0#start time
        etime = 0#end time
        ptime = stime#to keep track of previous time
        wtime = (gci.size * 1000.0)/fs#total time of the wave file
        wlab = []#array containing the time stamps of the wave file
        i = 0
        vflag = False#to keey track of voiced regions

        while i < gci.size:
            if gci[i] > 0:
                etime = (i * 1000.0)/fs
                if (etime - ptime) < uvdur:#to check whether VOICED or not
                    ptime = etime          #if its more than uvdur then it is UNVOICED
                    vflag = True
                else:                      #to tag it as UNVOICED
                    if vflag:
                        wlab.append((stime,ptime,'VOI'))
                        vflag = False
                    wlab.append((ptime,etime,'UNV'))
                    stime = etime
                    ptime = stime
            i += 1
        #fixing the trailing tags
        if etime != wtime:
            if vflag:
                wlab.append((stime,etime,'VOI'))
            wlab.append((etime,wtime,'UNV'))
        return wlab


    def segment3c(self,gci,fs,uvdur=18,sildur=300):
        '''
        The following function returns a three category segmentation of
        speech, namely, voiced(VOI), unvoiced(UNV) and silence segments(SIL).

        Following are the input parametrs
        1. gci is the glottal closure instants.
        2. fs is the sampling rate.
        3. uvdur is the maximum duration between any two GCI
        to classify them at VOI or else they would be
        classified as UNV.
        4. sildur is the minimum duration without gci to
        classify the segment as SIL.
        '''

        stime = 0#start time
        etime = 0#end time
        ptime = stime#to keep track of previous time
        wtime = (gci.size * 1000.0)/fs#total time of the wave file
        wlab = []#array containing the time stamps of the wave file
        i = 0
        vflag = False#to keey track of voiced regions

        while i < gci.size:
            if gci[i] > 0:
                etime = (i * 1000.0)/fs
                if (etime - ptime) < uvdur:#to check whether VOICED or not
                    ptime = etime          #if its more than uvdur then it is UNVOICED
                    vflag = True
                elif (etime - ptime) > sildur:#to tag it as SILENCE
                    if vflag:
                        wlab.append((stime,ptime,'VOI'))
                        vflag = False
                    wlab.append((ptime,etime,'SIL'))
                    stime = etime
                    ptime = stime
                else:#to tag it as UNVOICED
                    if vflag:
                        wlab.append((stime,ptime,'VOI'))
                        vflag = False
                    wlab.append((ptime,etime,'UNV'))
                    stime = etime
                    ptime = stime
            i += 1
        #fixing the trailing tags
        #one assumption made is that the trailing speech is most likely
        #contains the SILENCE
        if etime != wtime:
            if vflag:
                wlab.append((stime,etime,'VOI'))
            wlab.append((etime,wtime,'SIL'))
        return wlab

class LPC:
    '''
    Following class consists functions relating to Linear Prediction
    Analysis.

    The methodology adopted is as given in Linear Prediction:
    A Tutorial Review by John Makhoul.
    This is a preliminary version and the funtions are bound
    to change.

    Following are the functions present
    1. lpc: to calculate the linear prediction coefficients
    2. lpcfeats: to extract the linear prediction coefficients
                 from a wav signal
    3. lpcc: to calculate the cepstral coefficients
    4. lpccfeats: to extract the cepstral coefficients from a
                  wav signal
    5. lpresidual: to obtain the lp residual signal.
    6. residual: computes the residual of a windowed signal
    7. plotSpec: to plot the power spectrum of a signal and its
                 lp spectrum
    '''
    def __init__(self):
        pass

    def __autocor(self,sig,lporder):
        '''
        To calculate the AutoCorrelation Matrix from the input signal.
        Auto Correlation matrix is defined as follows:


              N-i-1
        r[i] = sum s[n]s[n+i] for all n
                n

        Following are the input parameters:
        sig: input speech signal
        lporder: self explanatory
        '''
        r = numpy.zeros(lporder + 1)
        for i in range(0,lporder + 1):
            for n in range(0,len(sig) - i):
                r[i] += (sig[n] * sig[n+i])
        return r

    def lpc(self,sig,lporder):
        '''
        Levinson-Durbin algorithm was implemented for
        calculating the lp coefficients. Please refer
        to the Linear Prediction: A Tutorial Review by
        John Makhoul

        Following are the input parameters:
        sig: input signal (of fixed window).
        lporder: self explanatory

        Output of the function is the following:
        1. Gain (E[0])
        2. Normalized Error (E[lporder]/E[0])
        3. LP Coefficients

        Function returns the following
        1. G: Gain
        2. nE: normalized error E[p]/E[0]
        3. LP Coefficients
        '''
        r = self.__autocor(sig,lporder) #Autocorrelation coefficients
        a = numpy.zeros(lporder + 1) #to store the a(k)
        b = numpy.zeros(lporder + 1) #to store the previous values of a(k)
        k = numpy.zeros(lporder + 1) #PARCOR coefficients
        E = numpy.zeros(lporder + 1)
        E[0] = r[0] #Energy of the signal
        for i in range(1,lporder+1):
            Sum = 0.0
            for j in range(1,i):
                Sum += (a[j] * r[i-j])
            k[i] = -(r[i] + Sum)/E[i-1]
            a[i] = k[i]
            for j in range(1,i):
                b[j] = a[j]
            for j in range(1,i):
                a[j] = b[j] + (k[i] * b[i-j])
            E[i] = (1.0 - (k[i]**2)) * E[i-1]
        a[0] = 1
        nE = E[lporder]/E[0] #normalized error
        G = r[0] #gain parameter
        for i in range(1,lporder+1):
            G += a[i] * r[i]
        return G,nE,a #G is the gain

    def lpcfeats(self,sig,fs,wlent,wsht,lporder):
        '''
        Extract the LPC features from the wave file.
        Following are the input parameters
        sig: input speech signal
        fs: its sampling frequency
        wlent: window length in milli seconds
        wsht: window shift in milli seconds.
        lporder: self explanatory

        Function returnes the following
        1. G: Gain for each frame
        2. nE: Normalized error for each frame
        3. LP coefficients for each frame.
        '''
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000

        sig = numpy.append(sig[0],numpy.diff(sig))
        sig = sig + 0.001
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        lpcs = [] #to store the lp coefficients
        nEs = [] #normalized errors
        Gs = [] #gain values

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            G,nE,a = self.lpc(smooth_signal,lporder)
            lpcs.append(a)
            nEs.append(nE)
            Gs.append(G)
        return numpy.array(Gs),numpy.array(nEs),numpy.array(lpcs)

    def residual(self,sig,a):
        '''
        Returns the error signal provided a set of LP
        coefficients. Error computation is as follows

        e[n] = s[n] + sum a[k]s[n-k] for k = 1,2,..p
                       k
             = s[n] * a[k] convolution of the signal + parameters

        Following are the input parameters
        1. sig: input signal. Consider using rectangular
                windowed signal, even though the LP
                coefficients were computed on hamming windowed
                signal
        2. a: LP coefficients.
        '''
        residual = numpy.convolve(sig,a,mode='full')
        residual = residual[len(a)/2 - 1 : len(sig) - (len(a)/2)]
        return residual

    def lpresidual(self,sig,fs,wlent,wsht,lporder):
        '''
        Computes the LP residual for a given speech signal.
        Signal is windowed using hamming window and LP
        coefficients are computed. Using these LP coefficients
        and the original signal residual for each frame of
        window length wlent is computed (see function error_signal)

        Following are the input parameters:
        1. sig: input speech signal
        2. fs: sampling rate of the signal
        3. wlent: window length in milli seconds
        4. wsht: window shift in milli seconds
        5. lporder: self explanatory.
        '''
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000

        sig = numpy.append(sig[0],numpy.diff(sig)) #to remove the dc
        sig = sig + 0.0001 #to make sure that there are no zeros in the signal
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        residual_signal = numpy.zeros(len(sig))
        residual_index = 0

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            G,nE,a = self.lpc(smooth_signal,lporder)
            er = self.__ersignal(window_signal,a)

            for i in range(0,wshf):
                residual_signal[residual_index] = er[i]
                residual_index += 1
        return residual_signal

    def lpcc(self,G,a,corder,L=0):
        """
        Following function returns the cepstral coefficients.
        Following are the input parameters
        1. G is the gain (energy of the signal)
        2. a are the lp coefficients
        3. corder is the cepstral order
        4. L (ceplifter) to lift the cepstral values
        The output of the function will the set of cepstral coefficients
        with the first value log(G). So the number of cepstral
        coefficients will one more than the corder.

        For the liftering process two procedures were implemented
        1. Sinusoidal
        c[m] = (1+(L/2)sin(pi*n/L))c[m]
        2. Linear Weighting

        By default it gives linear weighted cepstral coefficients.
        To obtain raw cepstral coefficients input L=None.

        For any other value of L it performs sinusoidal liftering.

        Note that number of cepstral coefficients can be more than
        lporder. Generally it is suggested that corder = (3/2)lporder
        """
        c = numpy.zeros(corder+1)
        c[0] = numpy.log(G)
        p = len(a) -1 #lp order + 1, a[0] = 1

        if corder <= p: #calculating if the corder is less than the lp order
            for m in range(1,corder+1):
                c[m] = a[m]
                for k in range(1,m):
                    c[m] -= (float(k)/m) * c[k] * a[m-k]
        else:
            for m in range(1,p+1):
                c[m] = a[m]
                for k in range(1,m):
                    c[m] -= (float(k)/m) * c[k] * a[m-k]

            for m in range(p+1,corder+1):
                for k in range((m-p),m):
                    c[m] -= (float(k)/m) * c[k] * a[m-k]

        if L == None:           #returning raw cepstral coefficients
            return c
        elif L == 0:      #returning linear weighted cepstral coefficients
            for i in range(1,corder+1):
                c[i] = c[i] * i
            return c

        #cep liftering process as given in HTK
        for i in range(1,corder+1):
            c[i] = (1 + (float(L)/2) * numpy.sin((numpy.pi * i)/L)) * c[i]

        return c

    def lpccfeats(self,sig,fs,wlent,wsht,lporder,corder,L=0):
        '''
        Computes the LPCC coefficients from the wave file.
        Following are the input parameters
        sig: input speech signal
        fs: its sampling frequency
        wlent: window length in milli seconds
        wsht: window shift in milli seconds.
        lporder: self explanatory
        corder: no. of cepstral coefficients
        (read the lpcc documentation for the description about
        the features)
        L(ceplifter) to lift the cepstral coefficients. Read the
        documentation for the LPCC for the liftering process

        Function returns the following
        1. G: Gain for each of the frames
        2. nE: Normalized errors for each of the frames
        3. LP Coefficients for each of the frames
        4. LP Cepstral Coefficients for each of the frames
        '''
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000

        sig = numpy.append(sig[0],numpy.diff(sig)) #to remove dc
        sig = sig + 0.001 #making sure that there are no zeros in the signal
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        lpcs = [] #to store the lp coefficients
        lpccs = []
        nEs = [] #normalized errors
        Gs = [] #gain values

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            G,nE,a = self.lpc(smooth_signal,lporder)
            c = self.lpcc(G,a,corder,L)
            lpcs.append(a)
            lpccs.append(c)
            nEs.append(nE)
            Gs.append(G)
        return numpy.array(Gs),numpy.array(nEs),numpy.array(lpcs),numpy.array(lpccs)

    def plotSpec(self,sig,G,a,res=0):
        """
        The following function plots the power spectrum of the wave
        signal along with the lp spectrum. This function is primary
        to analyse the lp spectrum
        Input for the function is as follows:
        sig is the input signal
        G is the gain
        a are the lp coefficients
        res is the resolution factor which tell the number of zeros
        to be appended before performing fft on the inverse filter

        Following funcion provides the power spectrum for the following
        Power spectrum of the signal

                          s[n]    <------->     S(w)
                           P(w) = 20 * log(|S(w)|)


                             P'(w) = 20 * log(G /|A(w)|)
               where A(w) is the inverse filter and is defined as follows

                                           p              -jkw
                             A(z) = 1 + SUMMATION a[k] * e
                                           k=1

        The xaxis in the plots give the frequencies in the rage 0-1,
        where 1 represents the nyquist frequency
        """
        for i in range(0,res):
            a = numpy.insert(a,-1,0) #appending zeros for better resolution
        fftA = numpy.abs(numpy.fft.fft(a))
        Gs = numpy.ones(len(a)) * G
        P1 = 10 * numpy.log10(Gs/fftA)
        P1 = P1[0:len(P1)/2] #power spectrum of the lp spectrum

        P = 10 * numpy.log10(numpy.abs(numpy.fft.fft(sig))) #power spectrum of the signal
        P = P[:len(P)/2]

        x = numpy.arange(0,len(P))
        x = x/float(max(x))

        matplotlib.pyplot.subplot(2,1,1)
        matplotlib.pyplot.title('Power Spectrum of the Signal')
        matplotlib.pyplot.plot(x,P)
        matplotlib.pyplot.xlabel('Frequency')
        matplotlib.pyplot.ylabel('Amplitude (dB)')
        matplotlib.pyplot.subplot(2,1,2)
        matplotlib.pyplot.title('LP Spectrum of the Signal')
        matplotlib.pyplot.plot(x,P1)
        matplotlib.pyplot.xlabel('Frequency')
        matplotlib.pyplot.ylabel('Amplitude (dB)')
        matplotlib.pyplot.show()


class MFCC:
    """
    Following class consists of functions to extract the following
    feature coefficients from the speech signal
    1. Mel Frequency Cepstral Coefficients (MFCC)
    2. Mel Spectrum (MELSPEC)
    3. Log Mel Spectrum values (FBANK)
    The implementation is similar to that of the HTK document. For
    further information please refer to the HTK Manual by Steve Young.

    A detail description of the procedure is given in Comparative
    Evaluation of Various MFCC Implementations on the Speaker
    Verification Task by Todor Ganchev, Nikos Fakotakis and
    George Kokkinakis
    """

    def __init__(self,fs,NOFB,LOFREQ=None,HIFREQ=None,N=512):
        """
        Input Parameters:
        fs       : input sampling frequency
        NOFB     : Number of filter banks
        N        : N point DFT
        LOFREQ   : Lower cutoff frequency
        HIFREQ   : Higher cutoff frequency

        Note: Following are the standard cutoff frequencies:
        1. fs = 8000 Hz, LOFREQ = 300, HIFREQ = 3400 #given in HTK manual
        2. fs = 16000 Hz, LOFREQ = 133.33334, HIFREQ=6855.4976 #default parameters in sphinx3
        """
        self.fs = float(fs)
        self.NOFB = NOFB
        self.N = N
        if LOFREQ == None:
            self.LOFREQ = 0

        if LOFREQ > self.fs/2 or LOFREQ < 0:
            print('LOFREQ is not in permissable range')
            print('Resetting LOFREQ to 0')
            LOFREQ = 0

        self.LOFREQ = LOFREQ

        if HIFREQ == None:
            self.HIFREQ = self.fs/2

        if HIFREQ > self.fs/2 or HIFREQ < 0:
            print('LOFREQ is not in permissable range')
            print('Resetting HIFREQ to nyquist frequency')
            HIFREQ = self.fs/2

        self.HIFREQ = HIFREQ

        if self.LOFREQ >= self.HIFREQ:
            print('Bad frequency ranges given')
            sys.exit(1)

        self.__melfb()

    def __mel(self,linf):
        return (2595.0 * numpy.log10(1.0 + (linf/700.0)))

    def __melinv(self,melf):
        return (700.0 * (numpy.power(10.0,melf/2595.0) - 1))

    def __melfb(self):
        LOFREQ_MEL = self.__mel(self.LOFREQ)
        HIFREQ_MEL = self.__mel(self.HIFREQ)

        melwidth = (HIFREQ_MEL - LOFREQ_MEL)/(self.NOFB + 1)

        melfilters = []
        for i in range(0,self.NOFB + 2):                   # generating filter banks in mel frequency domain
            melfilters.append(LOFREQ_MEL + (i * melwidth)) #all the centers of mel filters are equally spaced
        melfilters = numpy.array(melfilters)

        linfilters = self.__melinv(melfilters) #filters mapped to linear frequency range
        fb = (linfilters * self.N)/self.fs #discritising the frequency range

        self.LON = fb[0]   #min and max N values are stored to compute the energy
        self.HIN = fb[-1]  # of the signal

        self.H = numpy.array([[0.0] * self.N for i in xrange(self.NOFB + 2)]) # for easy computation we have used NOFB + 2

        for i in range(1,self.NOFB + 1):
            for k in range(0,self.N):
                if k <= fb[i-1]:
                    self.H[i][k] = 0
                elif fb[i-1] <= k <= fb[i]:
                    self.H[i][k] = (k - fb[i-1])/(fb[i] - fb[i-1])
                elif fb[i] <= k <= fb[i+1]:
                    self.H[i][k] = (fb[i+1] - k)/(fb[i+1] - fb[i])
                else:
                    self.H[i][k] = 0

        #Note: The first column of the transfer function H are all zeros

    def melspec(self,sig):
        """
        Computes the Mel Spectrum values for the windowed speech signal.
        Computation of the Mel Spectrum is as follows:

                               N - 1
                       X[i] = SUMMATION |S(k)| * H[i][k]
                                 k=0
        where H is weighting function.
        Input Parameters:
        sig : windowed speech signal
        """
        X = numpy.zeros(self.NOFB+1)
        for i in range(1,len(X)):
            X[i] = sum(numpy.abs(numpy.fft.fft(sig,self.N)) * self.H[i])
        return X[1:]

    def fbank(self,sig):
        """
        Computes the log of Mel Spectrum. Check melspec function for
        more information
        Input Parameters:
        sig: windowed signal
        """
        return numpy.log(self.melspec(sig))

    def melfilt(self,sig,corder,L):
        """
        Calculate the cepstral coefficients
        Input parameters:
        sig    : windowed speech signal
        corder : cepstral order
        L      : to lift the cepstral values. The output of the
        function will the set of cepstral (optional) coefficients
        with the first value log(E), where E is the energy of the
        signal
        So the number of cepstral coefficients will one more than
        the corder.

        For the liftering process two procedures were implemented
        1. Sinusoidal
        c[m] = (1+(L/2)sin(pi*n/L))c[m]
        2. Linear Weighting

        By default it gives linear weighted cepstral coefficients.
        To obtain raw cepstral coefficients input L=None.

        For any other value of L it performs sinusoidal liftering.
        """
        mfcc = numpy.zeros(corder + 1)
        mspec = numpy.append([0],self.fbank(sig))  #appending zeros for easy compuataion
        for j in range(1,len(mfcc)):      #performing dct
            for i in range(1,self.NOFB+1):
                mfcc[j] += mspec[i] * numpy.cos(j * (i - 0.5) * numpy.pi/self.NOFB)
        mfcc = mfcc * numpy.sqrt(2.0/self.NOFB)

        #computing C0
        mfcc[0] = sum(mspec[1:self.NOFB+1]) * numpy.sqrt(2.0/self.NOFB)

        if L == None:           #returning raw cepstral coefficients
            return mfcc
        elif L == 0:      #returning linear weighted cepstral coefficients
            for i in range(1,corder+1):
                mfcc[i] = mfcc[i] * i
            return mfcc

        #cep liftering process as given in HTK
        for i in range(1,corder+1):
            mfcc[i] = (1 + (float(L)/2) * numpy.sin((numpy.pi * i)/L)) * mfcc[i]

        return mfcc


    def melspecfeats(self,sig,wlent,wsht):
        """
        Calulcates the Mel Spectrum values from an input speech signal.
        Hamming window is applied on each of the frames.
        Input Parameters:
        sig   : speech signal
        wlent : window length in milli seconds
        wsht  : window shift in milli seconds
        """
        wlenf = (wlent * self.fs)/1000
        wshf = (wsht * self.fs)/1000

        sig = numpy.append(sig[0],numpy.diff(sig)) #to remove dc
        sig = sig + 0.001 #making sure that there are no zeros in the signal
        noFrames = int((len(sig) - wlenf)/wshf) + 1

        mspec = []

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            mspec.append(self.melspec(smooth_signal))
        mspec = numpy.array(mspec)
        return mspec

    def fbankfeats(self,sig,wlent,wsht):
        """
        Calulcates the Log Mel Spectrum values from an input speech signal.
        Hamming window is applied on each of the frames. The naming
        convention is smilar to HTK feature parameters.
        Input Parameters:
        sig   : speech signal
        wlent : window length in milli seconds
        wsht  : window shift in milli seconds
        """
        return numpy.log(self.melspecfeats(sig,wlent,wsht))

    def melfiltfeats(self,sig,wlent,wsht,corder,L=0):
        """
        Calulcates the Mel Filter Cepstral Coefficients from an input
        speech signal. Hamming window is applied on each of the frames.
        Input Parameters:
        sig   : speech signal
        wlent : window length in milli seconds
        wsht  : window shift in milli seconds
        """
        wlenf = (wlent * self.fs)/1000
        wshf = (wsht * self.fs)/1000

        sig = numpy.append(sig[0],numpy.diff(sig)) #to remove dc
        sig = sig + 0.001 #making sure that there are no zeros in the signal
        noFrames = int((len(sig) - wlenf)/wshf) + 1

        mfcc = []

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            mfcc.append(self.melfilt(smooth_signal,corder,L))
        mfcc= numpy.array(mfcc)
        return mfcc

def lowpass(signal, Fs, fc=400,order=4,plot=False):
    '''
    Filter out the really low frequencies, default is below 50Hz
    '''

    # have some predefined parameters
    rp = 5  # minimum ripple in dB in pass-band
    rs = 60   # minimum attenuation in dB in stop-band
    n = order   # order of the filter
    type = 'butter'

    # normalized cut-off frequency
    wc = 2. * fc / Fs

    # design the filter
    from scipy.signal import iirfilter, lfilter, freqz
    b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='lowpass', ftype=type)

    # plot frequency response of filter if requested
    if (plot):
        import matplotlib.pyplot as plt
        w, h = freqz(b, a)

        plt.figure()
        plt.title('Digital filter frequency response')
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.title('Digital filter frequency response')
        plt.ylabel('Amplitude Response [dB]')
        plt.xlabel('Frequency (rad/sample)')
        plt.grid()

    # apply the filter
    signal = lfilter(b, a, signal.copy())

    return signal
def highpass(signal, Fs, fc=20, order=4,plot=False):
    '''
    Filter out the really low frequencies, default is below 50Hz
    '''

    # have some predefined parameters
    rp = 5  # minimum ripple in dB in pass-band
    rs = 60   # minimum attenuation in dB in stop-band
    n = 4    # order of the filter
    type = 'butter'

    # normalized cut-off frequency
    wc = 2. * fc / Fs

    # design the filter
    from scipy.signal import iirfilter, lfilter, freqz
    b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='highpass', ftype=type)

    # plot frequency response of filter if requested
    if (plot):
        import matplotlib.pyplot as plt
        w, h = freqz(b, a)

        plt.figure()
        plt.title('Digital filter frequency response')
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.title('Digital filter frequency response')
        plt.ylabel('Amplitude Response [dB]')
        plt.xlabel('Frequency (rad/sample)')
        plt.grid()

    # apply the filter
    signal = lfilter(b, a, signal.copy())

    return signal
#spike removal
class Filter:

    filtered_data = []
    lycoord = []
    lxcoord = []
    rycoord = []
    rxcoord = []

    def __init__(self, xcoord, data, TraceNo, sample_interval):
        self.xcoord = xcoord
        self.data = data
        self.TraceNo = TraceNo
        self.sample_interval = sample_interval

    def linear_interpolation(self, markerList):
        #This will linearly interpolate between 2 chosen data points
        if len(markerList) != 0:
            marker_points = sorted(markerList[-2:])
        xrange = []
        time = [self.xcoord[int(round(marker_points[0]))], self.xcoord[int(round(marker_points[1]))]]
        b = self.xcoord[int(round(marker_points[0]))]
        while b <= self.xcoord[int(round(marker_points[1]))]:
            xrange.append(b)
            b = b + self.sample_interval[self.TraceNo-1]*10e-7
        amplitude = [self.data[int(round(marker_points[0]))], self.data[int(round(marker_points[1]))]]
        linearinterp = interp(xrange, time, amplitude)
        a = int(round(marker_points[0]))
        i = 0
        print(len(self.data))
        while a <= marker_points[1]:
            self.data[a] = linearinterp[i]
            a = a + 1
            i = i + 1
        Filter.filtered_data = self.data

    def mean_removal(self, markerList):
        if len(markerList) > 1:
            marker_points = sorted(markerList[-2:])
            me = mean(self.data[(int(round(marker_points[0]))-1):(int(round(marker_points[1])))])
            a = int(round(marker_points[0]))
            i = 0
            while a <= marker_points[1]:
                self.data[a] = self.data[a] - me
                a = a + 1
                i = i + 1

        if len(markerList) == 1:
            marker_points = markerList[0]
            me = self.data[int(round(marker_points))]
            self.data[int(round(marker_points))] = self.data[int(round(marker_points))] - self.data[int(round(marker_points))]

        Filter.filtered_data = [x for x in self.data]

    def offset_removal(self):

        average = mean(self.data)
        standardDeviation = std(self.data)

        s = 0
        k = 75

        foundl = 0
        foundr = 1

        initialOffset = 0
        finalOffset = 0

        offset = []

        #Offset detection algorithm
        if all([values > 1.8*standardDeviation for values in self.data[0:200]]):
            if initialOffset == 0:
                offset.append(0)
                foundl = 1
                foundr = 0
                initialOffset = 1

        for i in range(int(round(len(self.data)/75))):
            pAverage = abs(mean(self.data[s:k]))

            if foundl == 0:
                if pAverage > 1.8*standardDeviation:
                    for j in range(len(self.data[s:k])):
                        if self.data[s+j] > 1.5*standardDeviation:
                            offset.append(s+j)
                            foundl = 1
                            foundr = 0
                            break

            if foundr == 0:
                if pAverage < 1.8*standardDeviation:
                    for j in range(len(self.data[s:k])):
                        if self.data[s+j] < 1.5*standardDeviation:
                            offset.append(s+j)
                            foundr = 1
                            foundl = 0
                            break

            k = k + 75
            s = s + 75

        if all([x > 1.8*standardDeviation for x in self.data[-200:]]):
            if finalOffset == 0:
                offset.append(len(self.data))
                finalOffset = 1

        """Offset removal algorithm"""

        l = 5
        p = 0

        b = [x for x in self.data]

        stdDeviation = std(b)

        index = []

        for x in range(len(b)):
            if abs(average - abs(b[x])) > stdDeviation:
                index.append(x)

        for e in range(len(index)):
            del b[index[e]]
            index = [x - 1 for x in index]

        realAverage = mean(b)
        self.data = [x - realAverage for x in self.data]

        l = 5
        p = 0

        for x in range(len(offset)-1):
            for i in range(int(round(len(self.data[offset[x]:offset[x+1]]))/5)):
                pAverage2 = mean(self.data[offset[x]+p:offset[x]+l])
                self.data[offset[x]+p:offset[x]+l] = [y - pAverage2 for y in self.data[offset[x]+p:offset[x]+l]]
                l = l + 5
                p = p + 5

        Filter.filtered_data = [x for x in self.data]

    def bandpass_filter(self, lowfreq, highfreq):
        nyq = 0.5 * 1/(self.sample_interval[self.TraceNo-1]*10e-7) #Butter function works with Nyquist frecuencies for corner frequencies
        low = float(lowfreq) / nyq #Normalized lowcut frequency
        high = float(highfreq) / nyq #Normalized highcut frequency
        b, a = butter(2, [low, high], btype='band') #Coefficients for an order 2 Butterworth filter
        filtered_data = lfilter(b, a, self.data)
        Filter.filtered_data = filtered_data

    def spike_detect(self, maxstdtimes, minstdtimes, minxcoord, maxxcoord):
        me = mean(self.data)
        stdev = std(self.data)
        #self.data = [x - me for x in self.data]

        found = 0
        lycoord = []
        lxcoord = []
        rycoord = []
        rxcoord = []

        minxcoord = int(minxcoord)*100
        maxxcoord = int(maxxcoord)*100

        if minxcoord != 0:
            me = mean(self.data[minxcoord:maxxcoord])
            stdev = std(self.data[minxcoord:maxxcoord])
            for i in range(len(self.data[minxcoord:maxxcoord])-2):  #For each value in data
                if found == 0:
                    if abs(me-abs(self.data[minxcoord+i+2])) > float(maxstdtimes)*stdev: #find values who differ from the mean more than two times the std (spikes)
                        lycoord.append(self.data[minxcoord+i]) #Store those values
                        lxcoord.append(self.xcoord[minxcoord+i])
                        found = 1
                elif found == 1:
                    if abs(me-abs(self.data[minxcoord+i+1])) < float(minstdtimes)*stdev: #find values who differ from the mean more than two times the std (spikes)
                        rycoord.append(self.data[minxcoord+i+2]) #Store those values
                        rxcoord.append(self.xcoord[minxcoord+i+2])
                        found = 0
        else:
            ll = 0
            kk = 125
            for t in range(int(round(len(self.data)/125))):
                partialAverage = mean(self.data[ll:kk])
                partialStdev = std(self.data[ll:kk])
                for i in range(len(self.data[ll:kk])-2):  #For each value in data
                    if found == 0:
                        if abs(partialAverage-abs(self.data[ll+i+2])) > float(maxstdtimes)*partialStdev: #find values who differ from the mean more than two times the std (spikes)
                            lycoord.append(self.data[ll+i]) #Store those values
                            lxcoord.append(self.xcoord[ll+i])
                            found = 1
                    elif found == 1:
                        if abs(partialAverage-abs(self.data[ll+i+1])) < float(minstdtimes)*partialStdev: #find values who differ from the mean more than two times the std (spikes)
                            rycoord.append(self.data[ll+i+2]) #Store those values
                            rxcoord.append(self.xcoord[ll+i+2])
                            found = 0
                ll = ll + 125
                kk = kk + 125

        while len(lxcoord) > len(rxcoord):
            del lxcoord[-1]
            del lycoord[-1]

        while len(rxcoord) > len(lxcoord):
            del rxcoord[-1]
            del rycoord[-1]

        return lxcoord, lycoord, rxcoord, rycoord, stdev, me

    def de_spike(self, lxcoord, lycoord, rxcoord, rycoord):

        for i in range(len(lxcoord)):
            """Linear interpolation for spikes thinner than 8 points; mean removal for wider spikes. A second pass may be required"""
            xrange = []
            time = [lxcoord[i], rxcoord[i]]
            b = lxcoord[i]
            while b <= rxcoord[i]:
                xrange.append(b)
                b = b + self.sample_interval[self.TraceNo-1]*10e-7
            amplitude = [lycoord[i], rycoord[i]]
            linearinterp = interp(xrange, time, amplitude)
            a = lxcoord[i]
            c = 0
            """If spike width < 8, then linearly interpolate between its corners"""
            """Else, if spike width > 8, it may still contain valid data. Remove the mean value of that data and then remove the 2 resulting spikes
            with a linear interpolation"""
            if len(xrange) <= 8:
                while a <= rxcoord[i]:
                    self.data[int(round(a*100))] = linearinterp[c]
                    a = a + self.sample_interval[self.TraceNo-1]*10e-7
                    c = c + 1
            elif len(xrange) > 8:
                """Mean & trend removal"""
                me2 = mean(self.data[int(round(lxcoord[i]*100))+3:int(round(rxcoord[i]*100))-3])
                self.data[int(round(lxcoord[i]*100))+3:int(round(rxcoord[i]*100))-3] = [x - me2 for x in self.data[int(round(lxcoord[i]*100))+3:int(round(rxcoord[i]*100))-3]]

                """Left spike linear interpolation"""
                lxrange = []
                ltime = [lxcoord[i], lxcoord[i]+3*self.sample_interval[self.TraceNo-1]*10e-7]
                l = lxcoord[i]
                while l <= (lxcoord[i]+3*self.sample_interval[self.TraceNo-1]*10e-7):
                    lxrange.append(l)
                    l = l + self.sample_interval[self.TraceNo-1]*10e-7
                lamplitude = [self.data[int(round(lxcoord[i]*100))], self.data[int(round((lxcoord[i]+3*self.sample_interval[self.TraceNo-1]*10e-7)*100))]]
                leftlinearinterp = interp(lxrange, ltime, lamplitude)
                a = lxcoord[i]
                m = 0
                while a <= (lxcoord[i]+3*self.sample_interval[self.TraceNo-1]*10e-7):
                    self.data[int(round(a*100))] = leftlinearinterp[m]
                    a = a + self.sample_interval[self.TraceNo-1]*10e-7
                    m = m + 1

                """Right spike linear interpolation"""
                rxrange = []
                rtime = [rxcoord[i]-3*self.sample_interval[self.TraceNo-1]*10e-7, rxcoord[i]]
                r = rxcoord[i]-3*self.sample_interval[self.TraceNo-1]*10e-7
                while r <= rxcoord[i]:
                    rxrange.append(r)
                    r = r + self.sample_interval[self.TraceNo-1]*10e-7
                ramplitude = [self.data[int(round((rxcoord[i]-3*self.sample_interval[self.TraceNo-1]*10e-6)*100))], self.data[int(round(rxcoord[i]*100))]]
                rightlinearinterp = interp(rxrange, rtime, ramplitude)
                a = rxcoord[i]-3*self.sample_interval[self.TraceNo-1]*10e-7
                m = 0
                while a <= rxcoord[i]:
                    self.data[int(round(a*100))] = rightlinearinterp[m]
                    a = a + self.sample_interval[self.TraceNo-1]*10e-7
                    m = m + 1

        """When one of those really wide spikes with offset is present, mean value will be wrong after the first pass because of the removal of the spike's offset.
        This whas discovered because closing and opening the program again fixed the error, because class GraphicPlot in mod_plotter automatically removes the mean of the whole
        data set. To avoid restarting the program each time this happens, this function now removes the mean value after deleting spikes"""
        """mean_value = mean(self.data)
        self.data = [x - mean_value for x in self.data]"""

        Filter.filtered_data = self.data

        """Housekeeping"""
        del self.lycoord[:]
        del self.rycoord[:]
        del self.lxcoord[:]
        del self.rxcoord[:]


    def fft(self):
        """When the input a is a time-domain signal, A = fft(a), then np.abs(A) is its amplitude spectrum
        and np.angle(A) is its phase spectrum"""
        A = fft.fft(self.data)
        amplitude = abs(A)
        phase = angle(A)
        """fftfreq computes the frequencies associated with the signal's samplerate"""
        frequency = fft.fftfreq(len(self.data), self.sample_interval[self.TraceNo-1]*10e-7)
        index = argsort(frequency)
        return amplitude, phase, frequency, index
######up From spy###############################################
#For evaluate the predict sequence
#PCG,from predict value 11000111110111 get the 1 median positon
def pre_predict1010(y_predict):
    s1s2position = []
    positiveBundaries = np.nonzero(np.diff(y_predict) == 1)[0] + 1#+1,because len(diff (a))==len(a)-1
    nagetiveBoundaries = np.nonzero(np.diff(y_predict) == -1)[0]  #+1 will cause error

    print("positiveBundaries",positiveBundaries)
    print("nagetiveBoundaries",nagetiveBoundaries)
    print("The position ")
    print(zip(positiveBundaries,nagetiveBoundaries))
    print("The middle points of 1111 in array")
    for i in zip(positiveBundaries, nagetiveBoundaries):
        s1s2position.append(i[0] + (i[1] - i[0]) / 2)
    print(s1s2position)
    return s1s2position
#PCG,from predict value 11000111110111 get the 1 median positon
def pre_predict0123(y_predict):
    s1s2position = []
    positiveBundaries = np.nonzero(np.diff(y_predict) == 1)[0] + 1#+1,because len(diff (a))==len(a)-1
    nagetiveBoundaries = np.nonzero(np.diff(y_predict) == -1)[0]  #+1 will cause error

    print("positiveBundaries",positiveBundaries)
    print("nagetiveBoundaries",nagetiveBoundaries)
    print("The position ")
    print(zip(positiveBundaries,nagetiveBoundaries))
    print("The middle points of 1111 in array")
    for i in zip(positiveBundaries, nagetiveBoundaries):
        s1s2position.append(i[0] + (i[1] - i[0]) / 2)
    print(s1s2position)
    return s1s2position
#Lable the feature###################################################################
#get the start_pos and end_pos from the s1,s2 position and duration
def extendPositionFromS1S2PosByNormal(s1Pos,s2Pos,durationArray):
    s1Mean=durationArray[0][0]
    s2Mean=durationArray[2][0]
    systoleMean=durationArray[1][0]
    diastoleMean=durationArray[3][0]

    s1Std=durationArray[0][1]
    s2Std=durationArray[2][1]
    systoleStd=durationArray[1][1]
    diastoleStd=durationArray[3][1]

    s1Dur = random.normalvariate(s1Mean,s1Std)
    s2Dur = random.normalvariate(s2Mean, s2Std)
    sysDur = random.normalvariate(systoleMean, systoleStd)
    diaDur = random.normalvariate(diastoleMean, diastoleStd)

    s1StartPos=int(s1Pos-s1Dur/2)
    s1EndPos=int(s1Pos+s1Dur/2)
    s2StartPos =int( s2Pos -s2Dur / 2)
    s2EndPos = int(s2Pos + s2Dur / 2)
    diastoleEndpos=int(s2EndPos+diastoleMean)
    print("s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos",s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos)
    return s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos
def extendPositionFromS1S2PosByScale(s1Pos,s2Pos,durationArray):
    s1Mean=durationArray[0][0]
    s2Mean=durationArray[2][0]
    systoleMean=durationArray[1][0]
    diastoleMean=durationArray[3][0]

    s1Mean=durationArray[0][0]
    s2Mean=durationArray[2][0]
    systoleMean=durationArray[1][0]
    diastoleMean=durationArray[3][0]

    s1Std=durationArray[0][1]
    s2Std=durationArray[2][1]
    systoleStd=durationArray[1][1]
    diastoleStd=durationArray[3][1]

    s1Dur = random.normalvariate(s1Mean,s1Std)
    s2Dur = random.normalvariate(s2Mean, s2Std)
    sysDur = random.normalvariate(systoleMean, systoleStd)
    diaDur = random.normalvariate(diastoleMean, diastoleStd)

    s1StartPos=int(s1Pos-s1Dur/2)
    s1EndPos=int(s1Pos+s1Dur/2)
    s2StartPos =int( s2Pos -s2Dur / 2)
    s2EndPos = int(s2Pos + s2Dur / 2)
    diastoleEndpos=int(s2EndPos+diastoleMean)
    print("s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos",s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos)
    return s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos
#get the start_pos and end_pos from the s1,s2 position and duration
def extendPositionFromS1S2PosByMean(s1Pos,s2Pos,durationArray):
    s1Mean=durationArray[0][0]
    s2Mean=durationArray[2][0]
    systoleMean=durationArray[1][0]
    diastoleMean=durationArray[3][0]

    s1Std=durationArray[0][1]
    s2Std=durationArray[2][1]
    systoleStd=durationArray[1][1]
    diastoleStd=durationArray[3][1]

    s1Dur = random.normalvariate(s1Mean,s1Std)
    s2Dur = random.normalvariate(s2Mean, s2Std)
    sysDur = random.normalvariate(systoleMean, systoleStd)
    diaDur = random.normalvariate(diastoleMean, diastoleStd)

    s1StartPos=int(s1Pos-s1Mean/2)
    s1EndPos=int(s1Pos+s1Mean/2)
    s2StartPos =int( s2Pos -s2Mean / 2)
    s2EndPos = int(s2Pos + s2Mean / 2)
    diastoleEndpos=int(s2EndPos+diastoleMean)
    #print "s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos",s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos
    return s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos

def LabelFromS1S2PositionInAll(Label,s1StartPos,s1EndPos,s2StartPos,s2EndPos,diastoleEndpos):

    Label[s1StartPos:s1EndPos] = 1  #s1 label is 1
    Label[s1EndPos:s2StartPos] = 0 #systole label is 2
    Label[s2StartPos:s2EndPos] = 1  #s2 label is 3
    #Label[s1StartPos:s1EndPos] = 0  #diastole label is 0
    return Label,s1StartPos,diastoleEndpos
#label 1010
def LabelFromS1S2PositionInAll1010(Label, s1StartPos, s1EndPos, s2StartPos, s2EndPos, diastoleEndpos):
    Label[s1StartPos:s1EndPos] = 1  # s1 label is 1
    Label[s1EndPos:s2StartPos] = 0  # systole label is 2
    Label[s2StartPos:s2EndPos] = 1  # s2 label is 3
    # Label[s1StartPos:s1EndPos] = 0  #diastole label is 0
    return Label, s1StartPos, diastoleEndpos
#label 1020
def LabelFromS1S2PositionInAll1020(Label, s1StartPos, s1EndPos, s2StartPos, s2EndPos, diastoleEndpos):
    Label[s1StartPos:s1EndPos] = 1  # s1 label is 1
    Label[s1EndPos:s2StartPos] = 0  # systole label is 2
    Label[s2StartPos:s2EndPos] = 2  # s2 label is 3
    # Label[s1StartPos:s1EndPos] = 0  #diastole label is 0
    return Label, s1StartPos, diastoleEndpos
#label 1230
def LabelFromS1S2PositionInAll1230(Label, s1StartPos, s1EndPos, s2StartPos, s2EndPos, diastoleEndpos):
    Label[s1StartPos:s1EndPos] = 1  # s1 label is 1
    Label[s1EndPos:s2StartPos] = 2  # systole label is 2
    Label[s2StartPos:s2EndPos] = 3  # s2 label is 3
    # Label[s1StartPos:s1EndPos] = 0  #diastole label is 0
    return Label, s1StartPos, diastoleEndpos
def LabelFromS1S2PositionInAll0123(Label, s1StartPos, s1EndPos, s2StartPos, s2EndPos, diastoleEndpos):
    Label[s1StartPos:s1EndPos] = 0  # s1 label is 1
    Label[s1EndPos:s2StartPos] = 1  # systole label is 2
    Label[s2StartPos:s2EndPos] = 2  # s2 label is 3
    # Label[s1StartPos:s1EndPos] = 0  #diastole label is 0
    return Label, s1StartPos, diastoleEndpos

import argparse
import h5py
from sklearn.utils import shuffle
import cyEcgLib
from cyEcgLib import SearchS1S2MultiSample
class CLSTM():
    outputClass=4
    def to_categorical(self,y, Tick, nb_classes=None):
        '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
        2D->3D
        # Arguments
            y: class vector to be converted into a matrix
            nb_classes: total number of classes

        # Returns
            A binary matrix representation of the input.
        '''
        if not nb_classes:
            nb_classes = np.max(y.shape[2]) + 1
        Y = np.zeros((len(y), Tick, nb_classes))

        for i in range(len(y)):
            for j in range(Tick):
                Y[i, j, y[i, j]] = 1.
        return Y
    def ReadRawData(self,dataName,inputTickdim,inputFeatureDim,outputTick):
        # Load data and shaffle data
        file = h5py.File(dataName, 'r')

        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        # with open('S1S2Dict.pickle','rb')as f:
        #     dataAll=pickle.load(f)
        #     (trainData, trainDataLabel, testData, testDataLabel)=dataAll
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            numData = len(data)
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], outputTick))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        # generate valide data
        numSplit = int(round(1 * len(trainData)))
        X_train = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_valid = trainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test = testData
        y_test = testLable

        # reshape data
        # print(inputTickdim,inputFeatureDim,)
        # print(X_train.shape)
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = y_train.reshape((-1, outputTick, 1))
        Y_valid = y_valid.reshape((-1, outputTick, 1))
        Y_test = y_test.reshape((-1, outputTick, 1))

        Y_train = self.to_categorical(y_train, outputTick, self.outputClass)
        Y_valid = self.to_categorical(y_valid, outputTick, self.outputClass)
        Y_test = self.to_categorical(y_test, outputTick, self.outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    #get data#######################################################################
    def resampFromLongFeatureNoEquel(self,Feature, Label, label_length,
                                     feature_length, number=1000):

        segFeature = []
        segLabel = []
        start_index = np.random.randint(0, len(Label) - label_length - 1,
                                        (1, number))  # generate n mumber start index for segmenting feature
        # print start_index
        for i in start_index[0]:
            segFeature.append(Feature[i * 32:i * 32 + feature_length])
            segLabel.append(Label[i:i + label_length])
        #print(len(segFeature), len(segLabel))
        return segFeature, segLabel

    def getS1S2DataList(self,NameList, EnvelopDict, LabelDict, secondLength=4, sampleNumber=200,sample_dim=1):
        SAMPLE_LABEL_LENGTH=secondLength*50
        SAMPLE_FEATURE_LENGTH=secondLength*1600
        SAMPLE_DIM=sample_dim
        # initialnize
        fs = 50
        heartSoundList = []
        heartSoundLabelList = []
        s1s2List = []
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envFeature = envFeature[:, 6]
            envLabel = LabelDict[fileName_pre]
            #print("envFeature.shape,envLabel.shape", envFeature.shape, envLabel.shape)

            setFeature, segLabel = self.resampFromLongFeatureNoEquel(envFeature, envLabel, label_length=SAMPLE_LABEL_LENGTH,
                                                                feature_length=SAMPLE_FEATURE_LENGTH,
                                                                number=sampleNumber)
            # print len(setFeature),len(segLabel)
            heartSoundList = heartSoundList + setFeature

            heartSoundLabelList = heartSoundLabelList + segLabel
            #print(len(heartSoundList), len(heartSoundLabelList))
            #print("setFeature.shape", np.asarray(setFeature).shape)
        s1s2List = np.asarray(s1s2List).reshape(1, -1)
        #print s1s2List
        heartSoundList = np.reshape(heartSoundList, (-1, SAMPLE_FEATURE_LENGTH, SAMPLE_DIM))
        heartSoundLabelList = np.reshape(heartSoundLabelList, (-1, SAMPLE_LABEL_LENGTH, 1))
        # print(heartSoundList.shape,heartSoundLabelList.shape)
        return heartSoundList, heartSoundLabelList, s1s2List

    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    def filterFun(self, sig, fs=300, lowPass=1, highPass=150):
        filtecg = self.butter_bandpass_filter(sig, lowPass, highPass, fs)
        return filtecg
    def getS1S2DataList_Filter(self,NameList, EnvelopDict, LabelDict, secondLength=4, sampleNumber=200,sample_dim=4):
        SAMPLE_LABEL_LENGTH=secondLength*50
        SAMPLE_FEATURE_LENGTH=secondLength*1600
        SAMPLE_DIM=sample_dim
        # initialnize
        fs = 1600
        heartSoundList = []
        heartSoundLabelList = []
        s1s2List = []
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envFeature = envFeature[:, 6]
            envLabel = LabelDict[fileName_pre]
            #print("envFeature.shape,envLabel.shape", envFeature.shape, envLabel.shape)
            envFeatureFilter=[]
            if(sample_dim==4):
                sig_filter1 = self.filterFun(envFeature, fs, 20, 45)
                sig_filter2 = self.filterFun(envFeature, fs, 45, 80)
                sig_filter3 = self.filterFun(envFeature, fs, 80, 200)
                sig_filter4 = self.filterFun(envFeature, fs, 5, 200)
                envFeatureFilter.append(sig_filter1)
                envFeatureFilter.append(sig_filter2)
                envFeatureFilter.append(sig_filter3)
                envFeatureFilter.append(sig_filter4)
                envFeatureFilter = np.asarray(envFeatureFilter)
            elif(sample_dim==1):
                envFeatureFilter.append(envFeature)
                envFeatureFilter = np.asarray(envFeatureFilter)
            envFeatureFilter=envFeatureFilter.reshape(sample_dim,-1)
            #print(envFeatureFilter.shape)
            setFeature, segLabel = self.resampFromLongFeatureNoEquel(envFeatureFilter.T, envLabel, label_length=SAMPLE_LABEL_LENGTH,
                                                                feature_length=SAMPLE_FEATURE_LENGTH,
                                                                number=sampleNumber)
            # print len(setFeature),len(segLabel)
            heartSoundList = heartSoundList + setFeature

            heartSoundLabelList = heartSoundLabelList + segLabel
            #print(len(heartSoundList), len(heartSoundLabelList))
            #print("setFeature.shape", np.asarray(setFeature).shape)
        s1s2List = np.asarray(s1s2List).reshape(1, -1)
        #print s1s2List
        heartSoundList = np.reshape(heartSoundList, (-1, SAMPLE_FEATURE_LENGTH, SAMPLE_DIM))
        heartSoundLabelList = np.reshape(heartSoundLabelList, (-1, SAMPLE_LABEL_LENGTH, 1))
        print(heartSoundList.shape, heartSoundLabelList.shape)
        return heartSoundList, heartSoundLabelList, s1s2List

    def getRawData(self,trainNameList,testNameList,EnvelopDict,LabelDict,secondLength,sampleNumber,inputTickdim,inputFeatureDim,outputTick,isFilter=True,addDuration=False):
        if(isFilter==True):
            trainData, trainLabel, Trains1s2List =self.getS1S2DataList_Filter(trainNameList, EnvelopDict, LabelDict, secondLength, sampleNumber,sample_dim=inputFeatureDim)
            testData, testLable, Tests1s2List = self.getS1S2DataList_Filter(testNameList, EnvelopDict, LabelDict, secondLength, sampleNumber,sample_dim=inputFeatureDim)
        else:
            trainData, trainLabel, Trains1s2List =self.getS1S2DataList(trainNameList, EnvelopDict, LabelDict, secondLength, sampleNumber)
            testData, testLable, Tests1s2List = self.getS1S2DataList(testNameList, EnvelopDict, LabelDict, secondLength, sampleNumber)
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            numData = len(data)
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], outputTick))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        #print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        # generate valide data
        numSplit = int(round(0.9 * len(trainData)))
        X_train = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_valid = trainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test = testData
        y_test = testLable


        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = y_train.reshape((-1, outputTick, 1))
        Y_valid = y_valid.reshape((-1, outputTick, 1))
        Y_test = y_test.reshape((-1, outputTick, 1))

        Y_train = self.to_categorical(y_train, outputTick, self.outputClass)
        Y_valid = self.to_categorical(y_valid, outputTick, self.outputClass)
        Y_test = self.to_categorical(y_test, outputTick, self.outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    def getS1S2DataList_Filter_duration(self,NameList, EnvelopDict, LabelDict,HeartRateDict, secondLength=4, sampleNumber=200,sample_dim=4):
        SAMPLE_LABEL_LENGTH=secondLength*50
        SAMPLE_FEATURE_LENGTH=secondLength*1600
        SAMPLE_DIM=sample_dim
        # initialnize
        fs = 1600
        heartSoundList = []
        heartSoundLabelList = []
        s1s2List = []
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envFeature = envFeature[:, 6]
            envLabel = LabelDict[fileName_pre]
            HeartRateDictScaled = cyEcgLib.HeartRateDict_scale(HeartRateDict)
            addFeature = np.zeros((envFeature.shape[0], 2))
            for i in range(envFeature.shape[0]):
                addFeature[i][0] = HeartRateDictScaled[fileName_pre][0]  # heart rate
                addFeature[i][1] = HeartRateDictScaled[fileName_pre][1]  # heart duration

            #print("envFeature.shape,envLabel.shape", envFeature.shape, envLabel.shape)
            envFeatureFilter=[]
            sig_filter1 = self.filterFun(envFeature, fs, 20, 45)
            sig_filter2 = self.filterFun(envFeature, fs, 45, 80)
            sig_filter3 = self.filterFun(envFeature, fs, 80, 200)
            sig_filter4 = self.filterFun(envFeature, fs, 5, 200)
            envFeatureFilter.append(sig_filter1)
            envFeatureFilter.append(sig_filter2)
            envFeatureFilter.append(sig_filter3)
            envFeatureFilter.append(sig_filter4)
            envFeatureFilter = np.asarray(envFeatureFilter)

            envFeatureFilter=envFeatureFilter.reshape(4,-1)

            envFeatureFilter = np.envFeatureFilter((envFeatureFilter, addFeature))
            #print(envFeatureFilter.shape)
            setFeature, segLabel = self.resampFromLongFeatureNoEquel(envFeatureFilter.T, envLabel, label_length=SAMPLE_LABEL_LENGTH,
                                                                feature_length=SAMPLE_FEATURE_LENGTH,
                                                                number=sampleNumber)
            # print len(setFeature),len(segLabel)
            heartSoundList = heartSoundList + setFeature

            heartSoundLabelList = heartSoundLabelList + segLabel
            #print(len(heartSoundList), len(heartSoundLabelList))
            #print("setFeature.shape", np.asarray(setFeature).shape)
        s1s2List = np.asarray(s1s2List).reshape(1, -1)
        #print s1s2List
        heartSoundList = np.reshape(heartSoundList, (-1, SAMPLE_FEATURE_LENGTH, SAMPLE_DIM))
        heartSoundLabelList = np.reshape(heartSoundLabelList, (-1, SAMPLE_LABEL_LENGTH, 1))
        print(heartSoundList.shape, heartSoundLabelList.shape)
        return heartSoundList, heartSoundLabelList, s1s2List

    def getRawData_duration(self,trainNameList,testNameList,EnvelopDict,LabelDict,HeartRateDict,secondLength,sampleNumber,inputTickdim,inputFeatureDim,outputTick,isFilter=True,addDuration=False):
        if(addDuration==True):
            trainData, trainLabel, Trains1s2List =self.getS1S2DataList_Filter_duration(trainNameList, EnvelopDict,HeartRateDict, LabelDict, secondLength, sampleNumber,sample_dim=inputFeatureDim)
            testData, testLable, Tests1s2List = self.getS1S2DataList_Filter_duration(testNameList, EnvelopDict, LabelDict, HeartRateDict,secondLength, sampleNumber,sample_dim=inputFeatureDim)
        else:
            trainData, trainLabel, Trains1s2List =self.getS1S2DataList(trainNameList, EnvelopDict, LabelDict, secondLength, sampleNumber)
            testData, testLable, Tests1s2List = self.getS1S2DataList(testNameList, EnvelopDict, LabelDict, secondLength, sampleNumber)


        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            numData = len(data)
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], outputTick))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        #print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        # generate valide data
        numSplit = int(round(0.9 * len(trainData)))
        X_train = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_valid = trainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test = testData
        y_test = testLable


        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = y_train.reshape((-1, outputTick, 1))
        Y_valid = y_valid.reshape((-1, outputTick, 1))
        Y_test = y_test.reshape((-1, outputTick, 1))

        Y_train = self.to_categorical(y_train, outputTick, self.outputClass)
        Y_valid = self.to_categorical(y_valid, outputTick, self.outputClass)
        Y_test = self.to_categorical(y_test, outputTick, self.outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    def get_arguments(self,nb_epoch,batch_size,inputTickdim,inputFeatureDim,outputTick,outputClass,SaveName,TrainLoadWeights,TrainLoadWeightsName):
        def _str_to_bool(s):
            """Convert string to bool (in argparse context)."""
            if s.lower() not in ['true', 'false']:
                raise ValueError('Argument needs to be a '
                                 'boolean, got {}'.format(s))
            return {'true': True, 'false': False}[s.lower()]

        parser = argparse.ArgumentParser(description='Pcg dectect S1 S2')
        parser.add_argument('--epoch', type=int, default=nb_epoch,
                            help='How many wav files to process at once.')
        parser.add_argument('--batch_size', type=int, default=batch_size,
                            help='How many wav files to process at once.')
        parser.add_argument('--input_length', type=int, default=inputTickdim,
                            help='How many wav files to process at once.')
        parser.add_argument('--input_dim', type=int, default=inputFeatureDim,
                            help='How many wav files to process at once.')
        parser.add_argument('--output_length', type=int, default=outputTick,
                            help='How many wav files to process at once.')
        parser.add_argument('--output_dim', type=int, default=outputClass,
                            help='How many wav files to process at once.')
        parser.add_argument('--SaveName', type=str, default=SaveName,
                            help='Save weight Name')
        parser.add_argument('--Train', type=_str_to_bool, default=True,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')

        parser.add_argument('--TrainLoadWeights', type=_str_to_bool, default=TrainLoadWeights,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')
        parser.add_argument('--TrainLoadWeightsName', type=str, default=TrainLoadWeightsName,
                            help='SavePreName')
        return parser.parse_args()

    def oneHotToNumber3D(self,returnResult):
        predict=np.zeros((returnResult.shape[0],returnResult.shape[1]))
        for i in range(returnResult.shape[0]):
            for j in range(returnResult.shape[1]):
                predict[i][j]=np.argmax(returnResult[i][j])
        return predict

    def saveEpochWeight(self, SaveName, model, i):
        import os
        import logging
        if os.path.isdir(SaveName) == False:
            os.mkdir(SaveName)
        json_string = model.to_json()
        open(SaveName + '/' + SaveName + '.json', 'w').write(json_string)
        model.save_weights(SaveName + '/-' + str(i) + '-' '.h5')

    def saveEpochLog(self, SaveName, inputSecond, allScore1,allScore2):
        import os
        import logging
        if os.path.isdir(SaveName) == False:
            os.mkdir(SaveName)
        if (len(allScore1) > 0):
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                                filename='ExperimentRecord.log', filemode='a')

            logging.info(str(inputSecond)+'seconds input length'+'tolerante windows=100 ms' + str(allScore1)+'\n '+'tolerante windows=60 ms' + str(allScore2))
            logging.info('\n')


    def getF1score(self,model, X_test, Y_test, tolerate=5,convolution=True):
        X_sample = X_test
        Y_sample = Y_test
        print("Model Predict Input Data")
        returnResult = model.predict(X_sample, batch_size=200)
        print(returnResult.shape)

        predictLabel =  self.oneHotToNumber3D(returnResult)
        targetLabel = self.oneHotToNumber3D(Y_sample)
        directoryName = "Predict_s1s2position"
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)

        predictAll = predictLabel
        targetAll = targetLabel
        pretm = cyMetricLib.HeartSoundEvaluation()
        ###test all
        p_all, p_s1_all, p_s2_all = pretm.get_all_pscore(predictAll, targetAll, tolerate,convolution)
        s_all, s_s1_all, s_s2_all = pretm.get_all_sscore(predictAll, targetAll, tolerate,convolution)
        print("s_all", s_all)
        print("p_all", p_all)

        p = p_s1_all;
        s = s_s1_all;
        f_score_s1 = 2 * p * s / (p + s)
        print("f_s1_score", f_score_s1)

        p = p_s2_all;
        s = s_s2_all;
        f_score_s2 = 2 * p * s / (p + s)
        print("f_s2_score", f_score_s2)
        p = p_all;
        s = s_all;
        f_score = 2 * p * s / (p + s)
        print("f_score", f_score)
        return [s_all,p_all,f_score_s1,f_score_s2,f_score]
    def getF1score2(self,predictLabel, targetLabel, tolerate=5,convolution=False):
        predictAll = predictLabel
        targetAll = targetLabel
        pretm = cyMetricLib.HeartSoundEvaluation()
        ###test all
        p_all, p_s1_all, p_s2_all = pretm.get_all_pscore(predictAll, targetAll, tolerate,convolution)
        s_all, s_s1_all, s_s2_all = pretm.get_all_sscore(predictAll, targetAll, tolerate,convolution)
        print("s_all", s_all)
        print("p_all", p_all)

        p = p_s1_all;
        s = s_s1_all;
        if((p+s)<0.001):
            f_score_s1 = 2 * p * s / 0.001 #avoid  ZeroDivisionError: float division by zero
        else:
            f_score_s1 = 2 * p * s / (p + s)
        print("f_s1_score", f_score_s1)

        p = p_s2_all;
        s = s_s2_all;
        if((p+s)<0.001):
            f_score_s2 = 2 * p * s / 0.001#avoid  ZeroDivisionError: float division by zero
        else:
            f_score_s2 = 2 * p * s / (p + s)
        print("f_s2_score", f_score_s2)
        p = p_all;
        s = s_all;
        if((p+s)<0.001):
            f_score = 2 * p * s / 0.001#avoid  ZeroDivisionError: float division by zero
        else:
            f_score = 2 * p * s / (p + s)
        print("f_score", f_score)
        return [s_all,p_all,f_score_s1,f_score_s2,f_score]
class DLSTM():
    def saveEpochWeight(self, SaveName, model, i):
        import os
        import logging
        if os.path.isdir(SaveName) == False:
            os.mkdir(SaveName)
        json_string = model.to_json()
        open(SaveName + '/' + SaveName + '.json', 'w').write(json_string)
        model.save_weights(SaveName + '/-' + str(i) + '-' '.h5')

    def saveEpochLog(self, SaveName, inputSecond, allScore1,allScore2):
        import os
        import logging
        if os.path.isdir(SaveName) == False:
            os.mkdir(SaveName)
        if (len(allScore1) > 0):
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                                filename= 'ExperimentRecord.log', filemode='a')

            logging.info(str(inputSecond)+'seconds input length'+'tolerante windows=100 ms' + str(allScore1)+'\n '+'tolerante windows=60 ms' + str(allScore2))
            logging.info('\n')
    outputClass=4
    def to_categorical(self,y, Tick, nb_classes=None):
        '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
        2D->3D
        # Arguments
            y: class vector to be converted into a matrix
            nb_classes: total number of classes

        # Returns
            A binary matrix representation of the input.
        '''
        if not nb_classes:
            nb_classes = np.max(y.shape[2]) + 1
        Y = np.zeros((len(y), Tick, nb_classes))

        for i in range(len(y)):
            for j in range(Tick):
                Y[i, j, y[i, j]] = 1.
        return Y
    def ReadRawData(self,dataName,inputTickdim,inputFeatureDim,outputTick):
        # Load data and shaffle data
        file = h5py.File(dataName, 'r')

        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        # with open('S1S2Dict.pickle','rb')as f:
        #     dataAll=pickle.load(f)
        #     (trainData, trainDataLabel, testData, testDataLabel)=dataAll
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            numData = len(data)
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], outputTick))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        # generate valide data
        numSplit = int(round(0.9 * len(trainData)))
        X_train = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_valid = trainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test = testData
        y_test = testLable

        # reshape data
        # print(inputTickdim,inputFeatureDim,)
        # print(X_train.shape)
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = y_train.reshape((-1, outputTick, 1))
        Y_valid = y_valid.reshape((-1, outputTick, 1))
        Y_test = y_test.reshape((-1, outputTick, 1))

        Y_train = self.to_categorical(y_train, outputTick, self.outputClass)
        Y_valid = self.to_categorical(y_valid, outputTick, self.outputClass)
        Y_test = self.to_categorical(y_test, outputTick, self.outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    #get data#######################################################################
    def getS1S2DataList_SixFeature(self,NameList, EnvelopDict, LabelDict, HeartRateDict,secondLength=4, sampleNumber=200):
        SAMPLE_DIM = 6
        sampleLength=secondLength*50
        # initialnize
        fs = 50
        heartSoundList = []
        heartSoundLabelList = []
        s1s2List = []
        # recurrent for all file
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envLabel = LabelDict[fileName_pre]
            HeartRateDictScaled = cyEcgLib.HeartRateDict_scale(HeartRateDict)
            # Add feature
            addFeature = np.zeros((envFeature.shape[0], 2))
            for i in range(envFeature.shape[0]):
                addFeature[i][0] = HeartRateDictScaled[fileName_pre][0]  # heart rate
                addFeature[i][1] = HeartRateDictScaled[fileName_pre][1]  # heart duration
            #print "envFeature", envFeature.shape, addFeature.shape
            envFeature = np.hstack((envFeature, addFeature))

            #print "envFeature.shape,envLabel.shape", envFeature.shape, envLabel.shape
            ms = SearchS1S2MultiSample()
            setFeature, segLabel = ms.resampFromLongFeature(envFeature, envLabel, sampleLength=sampleLength,
                                                            sampleNumber=sampleNumber)
            # connect Two List
            heartSoundList = heartSoundList + setFeature
            heartSoundLabelList = heartSoundLabelList + segLabel
            #print len(heartSoundList), len(heartSoundLabelList)
            #print "setFeature.shape", len(heartSoundList)
        s1s2List = np.asarray(s1s2List).reshape(1, -1)
        #print s1s2List
        #print s1s2List
        heartSoundList = np.reshape(heartSoundList, (-1, sampleLength, SAMPLE_DIM))
        heartSoundLabelList = np.reshape(heartSoundLabelList, (-1, sampleLength, 1))
        return heartSoundList, heartSoundLabelList, s1s2List

    def getData_SixFeature(self,trainNameList,testNameList,EnvelopDict,LabelDict,HeartRateDict,secondLength,sampleNumber,inputTickdim,inputFeatureDim,outputTick):

        trainData, trainLabel, Trains1s2List =self.getS1S2DataList_SixFeature(trainNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
        testData, testLable, Tests1s2List = self.getS1S2DataList_SixFeature(testNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            numData = len(data)
            #print(np.asarray(data).shape)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], outputTick))  # reshape and translate into interger
            #print(data.shape, lable.shape)
            return data, lable

        #print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        # generate valide data
        numSplit = int(round(0.9 * len(trainData)))
        X_train = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_valid = trainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test = testData
        y_test = testLable


        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = y_train.reshape((-1, outputTick, 1))
        Y_valid = y_valid.reshape((-1, outputTick, 1))
        Y_test = y_test.reshape((-1, outputTick, 1))

        Y_train = self.to_categorical(y_train, outputTick, self.outputClass)
        Y_valid = self.to_categorical(y_valid, outputTick, self.outputClass)
        Y_test = self.to_categorical(y_test, outputTick, self.outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def getS1S2DataList_FourFeature(self, NameList, EnvelopDict, LabelDict, HeartRateDict, secondLength=4, sampleNumber=200):
        SAMPLE_DIM = 4
        #initialnize
        fs = 50
        sampleLength=secondLength*fs
        heartSoundList=[]
        heartSoundLabelList =[]
        s1s2List=[]
        #recurrent for all file
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envLabel = LabelDict[fileName_pre]
            HeartRateDictScaled = cyEcgLib.HeartRateDict_scale(HeartRateDict)
            #Add feature
            addFeature=np.zeros((envFeature.shape[0],2))
            for i in range(envFeature.shape[0]):
                addFeature[i][0]=HeartRateDictScaled[fileName_pre][0]#heart rate
                addFeature[i][1]=HeartRateDictScaled[fileName_pre][1]#heart duration
            #print "envFeature", envFeature.shape,addFeature.shape
            #envFeature=np.hstack((envFeature,addFeature))

            #print "envFeature.shape,envLabel.shape",envFeature.shape,envLabel.shape
            ms = SearchS1S2MultiSample()
            setFeature,segLabel=ms.resampFromLongFeature(envFeature, envLabel, sampleLength=sampleLength, sampleNumber=sampleNumber)
            #connect Two List
            heartSoundList=heartSoundList+setFeature
            heartSoundLabelList = heartSoundLabelList + segLabel
            #print len(heartSoundList), len(heartSoundLabelList)
            #print "setFeature.shape", len(heartSoundList)
        s1s2List=np.asarray(s1s2List).reshape(1,-1)
        #print s1s2List
        heartSoundList=np.reshape(heartSoundList,(-1,sampleLength,SAMPLE_DIM))
        heartSoundLabelList=np.reshape(heartSoundLabelList,(-1,sampleLength,1))
        return heartSoundList,heartSoundLabelList,s1s2List
    def getData_FourFeature(self,trainNameList,testNameList,EnvelopDict,LabelDict,HeartRateDict,secondLength,sampleNumber,inputTickdim,inputFeatureDim,outputTick):

        trainData, trainLabel, Trains1s2List =self.getS1S2DataList_FourFeature(trainNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
        testData, testLable, Tests1s2List = self.getS1S2DataList_FourFeature(testNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            numData = len(data)
            #print(np.asarray(data).shape)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], outputTick))  # reshape and translate into interger
            #print(data.shape, lable.shape)
            return data, lable

        #print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        # generate valide data
        numSplit = int(round(1 * len(trainData)))
        X_train = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_valid = trainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test = testData
        y_test = testLable


        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = y_train.reshape((-1, outputTick, 1))
        Y_valid = y_valid.reshape((-1, outputTick, 1))
        Y_test = y_test.reshape((-1, outputTick, 1))

        Y_train = self.to_categorical(y_train, outputTick, self.outputClass)
        Y_valid = self.to_categorical(y_valid, outputTick, self.outputClass)
        Y_test = self.to_categorical(y_test, outputTick, self.outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    def getS1S2DataList_5Feature_1(self,NameList, EnvelopDict, LabelDict, HeartRateDict,secondLength=4, sampleNumber=200):
        SAMPLE_DIM = 5
        sampleLength=secondLength*50
        # initialnize
        fs = 50
        heartSoundList = []
        heartSoundLabelList = []
        s1s2List = []
        # recurrent for all file
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envLabel = LabelDict[fileName_pre]
            HeartRateDictScaled = cyEcgLib.HeartRateDict_scale(HeartRateDict)
            # Add feature
            addFeature = np.zeros((envFeature.shape[0], 1))
            for i in range(envFeature.shape[0]):
                addFeature[i][0] = HeartRateDictScaled[fileName_pre][0]  # heart rate
            #print "envFeature", envFeature.shape, addFeature.shape
            envFeature = np.hstack((envFeature, addFeature))

            #print "envFeature.shape,envLabel.shape", envFeature.shape, envLabel.shape
            ms = SearchS1S2MultiSample()
            setFeature, segLabel = ms.resampFromLongFeature(envFeature, envLabel, sampleLength=sampleLength,
                                                            sampleNumber=sampleNumber)
            # connect Two List
            heartSoundList = heartSoundList + setFeature
            heartSoundLabelList = heartSoundLabelList + segLabel
            #print len(heartSoundList), len(heartSoundLabelList)
            #print "setFeature.shape", len(heartSoundList)
        s1s2List = np.asarray(s1s2List).reshape(1, -1)
        #print s1s2List
        #print s1s2List
        heartSoundList = np.reshape(heartSoundList, (-1, sampleLength, SAMPLE_DIM))
        heartSoundLabelList = np.reshape(heartSoundLabelList, (-1, sampleLength, 1))
        return heartSoundList, heartSoundLabelList, s1s2List
    def getS1S2DataList_5Feature_2(self,NameList, EnvelopDict, LabelDict, HeartRateDict,secondLength=4, sampleNumber=200):
        SAMPLE_DIM = 5
        sampleLength=secondLength*50
        # initialnize
        fs = 50
        heartSoundList = []
        heartSoundLabelList = []
        s1s2List = []
        # recurrent for all file
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envLabel = LabelDict[fileName_pre]
            HeartRateDictScaled = cyEcgLib.HeartRateDict_scale(HeartRateDict)
            # Add feature
            addFeature = np.zeros((envFeature.shape[0], 1))
            for i in range(envFeature.shape[0]):
                addFeature[i][0] = HeartRateDictScaled[fileName_pre][1]  # heart rate
            #print "envFeature", envFeature.shape, addFeature.shape
            envFeature = np.hstack((envFeature, addFeature))

            #print "envFeature.shape,envLabel.shape", envFeature.shape, envLabel.shape
            ms = SearchS1S2MultiSample()
            setFeature, segLabel = ms.resampFromLongFeature(envFeature, envLabel, sampleLength=sampleLength,
                                                            sampleNumber=sampleNumber)
            # connect Two List
            heartSoundList = heartSoundList + setFeature
            heartSoundLabelList = heartSoundLabelList + segLabel
            #print len(heartSoundList), len(heartSoundLabelList)
            #print "setFeature.shape", len(heartSoundList)
        s1s2List = np.asarray(s1s2List).reshape(1, -1)
        #print s1s2List
        #print s1s2List
        heartSoundList = np.reshape(heartSoundList, (-1, sampleLength, SAMPLE_DIM))
        heartSoundLabelList = np.reshape(heartSoundLabelList, (-1, sampleLength, 1))
        return heartSoundList, heartSoundLabelList, s1s2List
    def getS1S2DataList_8Feature(self,NameList, EnvelopDict, LabelDict, HeartRateDict,secondLength=4, sampleNumber=200):
        SAMPLE_DIM = 7
        sampleLength=secondLength*50
        # initialnize
        fs = 50
        heartSoundList = []
        heartSoundLabelList = []
        s1s2List = []
        # recurrent for all file
        for fileName_pre in NameList:
            envFeature = EnvelopDict[fileName_pre].T
            envLabel = LabelDict[fileName_pre]
            HeartRateDictScaled = cyEcgLib.HeartRateDict_scale(HeartRateDict)
            # Add feature
            addFeature = np.zeros((envFeature.shape[0], 3))
            for i in range(envFeature.shape[0]):
                addFeature[i][0] = HeartRateDictScaled[fileName_pre][0]  # heart rate
                addFeature[i][1] = HeartRateDictScaled[fileName_pre][1]  # heart rate
                addFeature[i][2] = float(1.0)/HeartRateDictScaled[fileName_pre][0]-HeartRateDictScaled[fileName_pre][1] -0.09 # heart rate
                #addFeature[i][3] = HeartRateDictScaled[fileName_pre][1]
            #print "envFeature", envFeature.shape, addFeature.shape
            envFeature = np.hstack((envFeature, addFeature))

            #print "envFeature.shape,envLabel.shape", envFeature.shape, envLabel.shape
            ms = SearchS1S2MultiSample()
            setFeature, segLabel = ms.resampFromLongFeature(envFeature, envLabel, sampleLength=sampleLength,
                                                            sampleNumber=sampleNumber)
            # connect Two List
            heartSoundList = heartSoundList + setFeature
            heartSoundLabelList = heartSoundLabelList + segLabel
            #print len(heartSoundList), len(heartSoundLabelList)
            #print "setFeature.shape", len(heartSoundList)
        s1s2List = np.asarray(s1s2List).reshape(1, -1)
        #print s1s2List
        #print s1s2List
        heartSoundList = np.reshape(heartSoundList, (-1, sampleLength, SAMPLE_DIM))
        heartSoundLabelList = np.reshape(heartSoundLabelList, (-1, sampleLength, 1))
        return heartSoundList, heartSoundLabelList, s1s2List
    def getData_5Feature(self,trainNameList,testNameList,EnvelopDict,LabelDict,HeartRateDict,secondLength,sampleNumber,inputTickdim,inputFeatureDim,outputTick,addDuration=1):
        if(addDuration==1):
            trainData, trainLabel, Trains1s2List =self.getS1S2DataList_5Feature_1(trainNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
            testData, testLable, Tests1s2List = self.getS1S2DataList_5Feature_1(testNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
        elif(addDuration==2):
            trainData, trainLabel, Trains1s2List =self.getS1S2DataList_5Feature_2(trainNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
            testData, testLable, Tests1s2List = self.getS1S2DataList_5Feature_2(testNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
        elif(addDuration==3):
            trainData, trainLabel, Trains1s2List =self.getS1S2DataList_8Feature(trainNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)
            testData, testLable, Tests1s2List = self.getS1S2DataList_8Feature(testNameList, EnvelopDict, LabelDict,HeartRateDict, secondLength, sampleNumber)

        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            numData = len(data)
            #print(np.asarray(data).shape)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], outputTick))  # reshape and translate into interger
            #print(data.shape, lable.shape)
            return data, lable

        #print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        # generate valide data
        numSplit = int(round(0.9 * len(trainData)))
        X_train = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_valid = trainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test = testData
        y_test = testLable


        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = y_train.reshape((-1, outputTick, 1))
        Y_valid = y_valid.reshape((-1, outputTick, 1))
        Y_test = y_test.reshape((-1, outputTick, 1))

        Y_train = self.to_categorical(y_train, outputTick, self.outputClass)
        Y_valid = self.to_categorical(y_valid, outputTick, self.outputClass)
        Y_test = self.to_categorical(y_test, outputTick, self.outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def get_arguments(self,nb_epoch,batch_size,inputTickdim,inputFeatureDim,outputTick,outputClass,SaveName,TrainLoadWeights,TrainLoadWeightsName):
        def _str_to_bool(s):
            """Convert string to bool (in argparse context)."""
            if s.lower() not in ['true', 'false']:
                raise ValueError('Argument needs to be a '
                                 'boolean, got {}'.format(s))
            return {'true': True, 'false': False}[s.lower()]

        parser = argparse.ArgumentParser(description='Pcg dectect S1 S2')
        parser.add_argument('--epoch', type=int, default=nb_epoch,
                            help='How many wav files to process at once.')
        parser.add_argument('--batch_size', type=int, default=batch_size,
                            help='How many wav files to process at once.')
        parser.add_argument('--input_length', type=int, default=inputTickdim,
                            help='How many wav files to process at once.')
        parser.add_argument('--input_dim', type=int, default=inputFeatureDim,
                            help='How many wav files to process at once.')
        parser.add_argument('--output_length', type=int, default=outputTick,
                            help='How many wav files to process at once.')
        parser.add_argument('--output_dim', type=int, default=outputClass,
                            help='How many wav files to process at once.')
        parser.add_argument('--SaveName', type=str, default=SaveName,
                            help='Save weight Name')
        parser.add_argument('--Train', type=_str_to_bool, default=True,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')

        parser.add_argument('--TrainLoadWeights', type=_str_to_bool, default=TrainLoadWeights,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')
        parser.add_argument('--TrainLoadWeightsName', type=str, default=TrainLoadWeightsName,
                            help='SavePreName')
        return parser.parse_args()

    def oneHotToNumber3D(self,returnResult):
        predict=np.zeros((returnResult.shape[0],returnResult.shape[1]))
        for i in range(returnResult.shape[0]):
            for j in range(returnResult.shape[1]):
                predict[i][j]=np.argmax(returnResult[i][j])
        return predict

    def getF1score(self,model, X_test, Y_test, tolerate=5,convolution=False):
        X_sample = X_test
        Y_sample = Y_test
        print("Model Predict Input Data")
        returnResult = model.predict(X_sample, batch_size=200)
        print(returnResult.shape)

        predictLabel =  self.oneHotToNumber3D(returnResult)
        targetLabel = self.oneHotToNumber3D(Y_sample)
        directoryName = "Predict_s1s2position"
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)

        predictAll = predictLabel
        targetAll = targetLabel
        pretm = cyMetricLib.HeartSoundEvaluation()
        ###test all
        p_all, p_s1_all, p_s2_all = pretm.get_all_pscore(predictAll, targetAll, tolerate,convolution)
        s_all, s_s1_all, s_s2_all = pretm.get_all_sscore(predictAll, targetAll, tolerate,convolution)
        print("p_all, p_s1_all, p_s2_all", p_all, p_s1_all, p_s2_all)
        print("s_all, s_s1_all, s_s2_all", s_all, s_s1_all, s_s2_all)
        try:
            p = p_s1_all;
            s = s_s1_all;
            f_score_s1 = 2 * p * s / (p + s)
            print("f_s1_score", f_score_s1)

            p = p_s2_all;
            s = s_s2_all;
            f_score_s2 = 2 * p * s / (p + s)
            print("f_s2_score", f_score_s2)
            p = p_all;
            s = s_all;
            f_score = 2 * p * s / (p + s)
            print("f_score", f_score)
        except:
            s_all=0
            p_all=0
            f_score_s1=0
            f_score_s2=0
            f_score=0
        return [s_all,p_all,f_score_s1,f_score_s2,f_score]