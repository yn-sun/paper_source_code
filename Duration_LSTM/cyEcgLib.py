#-*- coding:utf8 -*-
import numpy as np
import scipy
from scipy import signal, fft, arange
import sys
import matplotlib.patches as patches
from numpy import NaN, Inf, arange, isscalar, asarray, array
#from mne.filter import band_pass_filter
#from mne.utils import sum_squared
import numpy
from scipy.signal import resample, hilbert, butter, filtfilt, lfilter
import librosa
import scipy.io as scio
from numpy import interp, mean, std, fft, angle, argsort
from scipy.signal import butter, lfilter
import h5py
def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    #PEAKDET Detect peaks in a vector
    #        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    #        maxima and minima ("peaks") in the vector V.
    #        MAXTAB and MINTAB consists of two columns. Column 1
    #        contains indices in V, and column 2 the found values.
    #
    #        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    #        in MAXTAB and MINTAB are replaced with the corresponding
    #        X-values.
    #
    #        A point is considered a maximum peak if it has the maximal
    #        value, and was preceded (to the left) by a value lower by
    #        DELTA.

    # Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    # This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)
from matplotlib import cbook

class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""
    def __init__(self, artists, tolerance=5, offsets=(-20, 20),
                 template='x: %0.2f\ny: %0.2f', display_all=False):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be
            selected.
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "template" is the format string to be used. Note: For compatibility
            with older versions of python, this uses the old-style (%)
            formatting specification.
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless.
        """
        self.template = template
        self.offsets = offsets
        self.display_all = display_all
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))

        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect('pick_event', self)

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(self.template, xy=(0, 0), ha='right',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        annotation.set_visible(False)
        return annotation

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            annotation.xy = x, y
            annotation.set_text(self.template % (x, y))
            annotation.set_visible(True)
            event.canvas.draw()


def baseline(data, rate):
    '''
    input data array (list) and rate value (capture rate in seconds), returns the baserate average and the shifted data array.
    using the first 0.2 s of data, find average of all data values. this becomes the baseline.
    this means that the first 0.5 seconds of data MUST be clean and not include and event.
    baseline is used later in burst measurements
    import numpy as np
    '''

    index = 0.2 / rate  # find the index of the first 0.2 seconds of data
    baserate = np.mean(data[:index])  # average all points to obtain baserate

    datashift = []

    # may be able to use np.subtract(data,base) instead, but this seems to work correctly.
    for x in data:
        foo = (x - baserate)
        datashift.append(foo)

    return baserate, datashift


def burstduration(time, data, baserate, threshperc=0.2, clusterperc=0.006):
    """
    threshperc is the percentage of the baseline that the threshold will be above; clusterperc should be changed based on the type of data (default for ECG is 0.006)
    baserate needs to be calculated and passed into this argument
    data should already be transformed, smoothed, and baseline shifted (shifting is technically optional, but it really doesn't matter)
    returns the lists of start times, end times, and duration times
    """

    if len(time) != len(data):  # logic check, are the number of data points and time points the same?
        sys.exit('You cannot have more time points than there are data points. Get that sorted, buddy.')

    burst_start = []  # empty array for burst start
    burst_end = []  # empty array for burst end
    burst_duration = []  # empty array to calculate burst durration

    threshold = baserate * threshperc  # calculate the point at which a event is considered a peak

    burst = False  # burst flag, should start not in a burst

    index = -1
    for point in data:  # for each data point in the set
        index = index + 1
        # print index, "=", t.clock()

        if burst == False and point > threshold:  # if we are not in a burst already, the value is higher than the threshold, AND the last burst didn't end less than .2 ms ago
            tpoint = time[index]  # find the actual time given teh time index
            burst_start.append(tpoint)  # add the time at point as the begining of the burst
            burst = True  # burst flag, we are now in a burst

        if burst == True and point <= threshold:  # if we are in a burst and the point falls below the threshold

            if len(burst_end) == 0 or len(burst_start) == 0:  # if this is the first end
                tpoint = time[index]  # find the actual time given teh time index
                burst_end.append(tpoint)  # add the time at point as the end of the burst
                burst = False  # burst flag, we are now out of the burst

            else:
                tpoint = time[index]  # find the actual time given teh time index
                burst_end.append(tpoint)  # add the time at point as the end of the burst
                burst = False  # burst flag, we are now out of the burst
                if burst_start[-1] < (burst_end[
                                          -2] + clusterperc):  # if the new burst is too close to the last one, delete the second to last end and the last start
                    del burst_end[-2]
                    del burst_start[-1]

    if burst == True and len(burst_start) == 1 + len(burst_end):  # we exit the for loop but are in a burst
        del burst_start[-1]  # delete the last burst start time

    if len(burst_start) != len(burst_end):
        sys.exit(
            'Unexpectedly, the number of burst start times and end times are not equal. Seeing as this is physically impossible, I quit the program for you. Begin hunting for the fatal flaw. Good luck!')

    # print t.clock(), "- start duration array"
    for foo in burst_start:  # for each burst
        index = burst_start.index(foo)
        duration = burst_end[index] - burst_start[
            index]  # calculate duration by subtracting the start time from the end time, found by indexing
        burst_duration.append(duration)  # add the burst duration to the duration list
    # print t.clock(), "-end duration array"

    return burst_start, burst_end, burst_duration


def interburstinterval(burst_start, burst_end):
    """
    this function is used to find the inter-burst interval.
    this is defined as the difference between the last end and the new start time
    Dependent on numpy, burst_start, and burst_end
    """

    ibi = []

    for end in burst_end[:-1]:  # for each end time except the last one
        tindex = burst_end.index(end)  # find the start time index
        start = burst_start[tindex + 1]  # find start time
        ibi.append(start - end)  # subtract the old end time from the start time

    return ibi


def ttotal(burst_start):
    """
    find time from start to start, called the interburst interval. Input array must be a list of numbers (float).
    """

    ttotal = []  # empty array for ttot to go into

    for time in burst_start[1:]:  # for each start time, starting with the second one
        s2time = burst_start.index(time)
        s2 = burst_start[s2time - 1]
        meas = time - s2  # measure the interval by subtracting
        ttotal.append(meas)  # append the measurement to the ttotal array
    return ttotal  # return array


def burstarea(data, time, burst_start, burst_end, dx=10):
    """
    integral, area under curve of each burst. Use start and end times to split the y values into short lists.
    need the time array to do this
    """
    from scipy.integrate import simps, trapz  # import the integral functions

    time = list(time)  # time array must be a list for the indexting to work.

    burst_area = []  # empty array for the areas to go into

    for i in np.arange(len(burst_start)):  # for each index in the start array
        end = time.index(burst_end[
                             i])  # using the value at each i in the burst_end array, index in the time array to get the time index. this will be the same index # as the data array
        start = time.index(burst_start[i])
        area = trapz(data[start:end], x=time[start:end], dx=dx)  # find area using the trapz function, but only
        burst_area.append(area)

    return burst_area

#find time from r peak to r peak
def rrinterval(maxptime):
    """
    find time from r peak to r peak, called the R-R interval. Input array must be a list of numbers (float).
    """
    maxptime = maxptime.tolist()

    rrint = []  # empty array for ttot to go into

    for time in maxptime[1:]:  # for each r peak, starting with the second one
        s2time = maxptime.index(time)
        s2 = maxptime[s2time - 1]
        meas = time - s2  # measure the interval by subtracting
        rrint.append(meas)  # append the measurement to the ttotal array
    return rrint  # return array


def burst_peaks(data, burst_start, burst_end, time=0, delta=0.04, rate=0.00025):
    """
    given the burst start and end time, along with the looser peak detect, calculate the number of peaks in a given burst
    dependent on burstduration, numpy, sys, and peakdet
    default delta is 0.04, based on ECG data, seems to catch QRS and sometimes P and/or T
    data MUST be transformed, but not shifted. peakdet will reject it if it has negative values
    time varible is the value of time at the the first index of the time array. You can input it in as time=TIME_ARRAY[0]
    rate is the collection rate of the data set.
    """

    if len(burst_start) != len(burst_end):  # logic check, before we start
        sys.exit(
            'Unexpectedly, the number of burst start times and end times are not equal. Seeing as this is physically impossible, I quit the program for you. Begin hunting for the fatal flaw. Good luck!')

    if (len(burst_start) == 0) or (len(burst_end) == 0):  # logic check, are they empty lists
        sys.exit('One of the input arrays is an empty list')

    maxtab, mintab = peakdet(data, delta)  # generate the maxtab array of times and heights

    maxptime = maxtab[:, 0]  # extract the time column
    maxptime = (np.multiply(maxptime, rate) + time)  # convert from index to ms

    index = 0  # initalize burst index to zero
    burstpeaknum = []  # empty array for where the number of peaks will go. index # = burst #
    burstpeaks = []  # empty tuple of peak locations. index # = burst #
    plist = []  # make blank plist

    for time in maxptime:  # for each peak found in maxtab. remember that maxtab has multipule columns, so only take time.

        if index == (len(burst_start) - 1) and time > burst_end[index]:
            burstpeaks.append(plist)  # put list of peaks in burst peaks list
            burstpeaknum.append(len(plist))  # add count of peaks to burst peak num list

        if time > burst_end[-1]:
            break

        if burst_start[index] < time < burst_end[index]:  # if the peak is within the duration of the burst
            plist.append(time)

            if time == maxptime[-1]:  # if the current peak is the last one
                burstpeaks.append(plist)  # put list of peaks in burst peaks list
                burstpeaknum.append(len(plist))  # add count of peaks to burst peak num list

        if index < (len(burst_start) - 1) and time > burst_start[
                    index + 1]:  # if the next peak is actually in the NEXT burst AND it isn't the last burst
            burstpeaks.append(plist)  # put list of peaks in burst peaks list
            burstpeaknum.append(len(plist))  # add count of peaks to burst peak num list
            index = (index + 1)  # go to the next burst
            plist = []  # make a new list for the new peak
            plist.append(time)  # add this timepoint to the new list

    return burstpeaknum, burstpeaks, maxtab
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    # except ValueError, msg:
    #     raise ValueError("window_size and order have to be of type int")
    except ValueError:
        print("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#庞加莱
def poincare(data_array):
    """
    Given a 1d array of data, create a Poincare plot along with the SD1 and SD2 parameters
    usees matplotlib.patches.Ellipse to create the fit elipse
    equations are derived from Brennan and
    http://www.mif.pg.gda.pl/homepages/graff/Tzaa/Guzik_geometry_asymetry.pdf cause OMG THIS MAKES SENSE NOW
    """

    x = data_array[:(len(data_array) - 1)]
    y = data_array[1:]

    xc = np.mean(x)
    yc = np.mean(y)

    # SD1 = np.sqrt((1/len(x)) * sum(((x-y) - np.mean(x-y))^2)/2)
    # SD2 = np.sqrt((1/len(x)) * sum(((x+y) - np.mean(x+y))^2)/2)

    SD1 = 0
    SD2 = 0

    for i in np.arange(len(x)):
        d1 = np.power(abs((x[i] - xc) - (y[i] - yc)) / np.sqrt(2), 2)
        SD1 = SD1 + d1

        d2 = np.power(abs((x[i] - xc) + (y[i] - yc)) / np.sqrt(2), 2)
        SD2 = SD2 + d2

    SD1 = np.sqrt(SD1 / len(x))
    SD2 = np.sqrt(SD2 / len(x))

    return x, y, xc, yc, SD1, SD2

#这个是重mne里面来的
def qrs_detector( ecg,sfreq, thresh_value=0.2, levels=2.5, n_thresh=3, tstart=0):
    """Detect QRS component in ECG channels.

    QRS is the main wave on the heart beat.

    Parameters
    ----------
    sfreq : float
        Sampling rate
    ecg : array
        ECG signal
    thresh_value : float | str
        qrs detection threshold. Can also be "auto" for automatic
        selection of threshold.
    levels : float
        number of std from mean to include for detection
    n_thresh : int
        max number of crossings
    tstart : float
        Start detection after tstart seconds.
    Returns
    -------
    events : array
        Indices of ECG peaks
    """
    win_size = int(round((60.0 * sfreq) / 120.0))
    ecg_abs = np.abs(ecg)
    init = int(sfreq)

    n_samples_start = int(sfreq * tstart)
    ecg_abs = ecg_abs[n_samples_start:]

    n_points = len(ecg_abs)

    maxpt = np.empty(3)
    maxpt[0] = np.max(ecg_abs[:init])
    maxpt[1] = np.max(ecg_abs[init:init * 2])
    maxpt[2] = np.max(ecg_abs[init * 2:init * 3])

    init_max = np.mean(maxpt)

    if thresh_value == 'auto':
        thresh_runs = np.arange(0.3, 1.1, 0.05)
    else:
        thresh_runs = [thresh_value]

    # Try a few thresholds (or just one)
    clean_events = list()
    for thresh_value in thresh_runs:
        thresh1 = init_max * thresh_value
        numcross = list()
        time = list()
        rms = list()
        ii = 0
        while ii < (n_points - win_size):
            window = ecg_abs[ii:ii + win_size]
            if window[0] > thresh1:
                max_time = np.argmax(window)
                time.append(ii + max_time)
                nx = np.sum(np.diff(((window > thresh1).astype(np.int) ==
                                     1).astype(int)))
                numcross.append(nx)
                rms.append(np.sqrt(sum_squared(window) / window.size))
                ii += win_size
            else:
                ii += 1

        if len(rms) == 0:
            rms.append(0.0)
            time.append(0.0)
        time = np.array(time)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_thresh = rms_mean + (rms_std * levels)
        b = np.where(rms < rms_thresh)[0]
        a = np.array(numcross)[b]
        ce = time[b[a < n_thresh]]

        ce += n_samples_start
        clean_events.append(ce)

    # pick the best threshold; first get effective heart rates
    rates = np.array([60. * len(cev) / (len(ecg) / float(sfreq))
                      for cev in clean_events])

    # now find heart rates that seem reasonable (infant through adult athlete)
    idx = np.where(np.logical_and(rates <= 160., rates >= 40.))[0]
    if len(idx) > 0:
        ideal_rate = np.median(rates[idx])  # get close to the median
    else:
        ideal_rate = 80.  # get close to a reasonable default
    idx = np.argmin(np.abs(rates - ideal_rate))
    clean_events = clean_events[idx]
    return clean_events
#自己写的找最大点的,这个是由心音或者心电图的特点来写的函数
#输入：音频数据，采样率
#输出：list of max value position
def GetCutRecorder(envmax,fs):
    # give a signal envmax ,can get the serial recorder
    cut_recorder=[]
    max_first_position=np.argmax(envmax[0:fs])
    iterator=max_first_position
    t_envmax=envmax.T
    while iterator<len(envmax):#find from the first pitch
        cut_recorder.append(iterator)
        if len(envmax)>iterator+fs:
            start_pos=int(iterator+fs/2)
            end_pos=int(iterator+1.2*fs)
        else:
            break
        max_local_position=np.argmax(t_envmax[start_pos:end_pos])
        iterator=max_local_position+start_pos
    cut_recorder = np.asarray(cut_recorder)
    return cut_recorder
#input:R_start,R_end,fs:sample rate,baseline：sinal average value
#ouput:T_max,T_end
def TWaveEndDet_sinal(sinal,R_start,R_end,fs,baseLine):
    if baseLine==0:
        baseLine=baserate = np.mean(sinal[0:2*fs])
    start_pos=int(R_start+round(fs/10))
    #print start_pos
    #end_pos=int(R_end-round(fs/10))
    end_pos = int(R_end - round(fs / 6))
    max_position=np.argmax(sinal[start_pos:end_pos])
    max_local_position=start_pos+max_position
    #print "max_local_position,start_pos,end_pos", max_local_position,start_pos, end_pos
    for i in range(max_local_position,end_pos):#from max positon ,to find the value = baseline
        if sinal[i]<baseLine:#第一次小于baseLine的位置
            break
    T_end=i;
    T_max=max_local_position
    return T_max,T_end;
#得到列表TendList,get the number of Tend is len(RList)-1
def TWaveEndDet(sinal,RList,fs,baseLine=0):
    TMaxList=[]
    TEndList=[]
    for i in range(0,len(RList)-1):
        #print i
        T_max, T_end=TWaveEndDet_sinal(sinal,RList[i],RList[i+1],fs,baseLine)
        TMaxList.append(T_max)
        TEndList.append(T_end)
    return TMaxList,TEndList
#得到列表TendList,and avoid,get the number of Tend is len(RList)
def TWaveEndDet2(sinal,RList,fs,baseLine=0):
    TMaxList=[]
    TEndList=[]
    for i in range(0,len(RList)):
        #print i
        if i==len(RList)-1: #avoid RList[i+1] is out of bounds of axis 0,the R_end is  end of sinal
            T_max, T_end = TWaveEndDet_sinal(sinal, RList[i], len(sinal), fs, baseLine)
        else:
            T_max, T_end=TWaveEndDet_sinal(sinal,RList[i],RList[i+1],fs,baseLine)
        TMaxList.append(T_max)
        TEndList.append(T_end)
    return TMaxList,TEndList
#get s1 position from ecg RpeakPosition
#seach s1  s1+- sigma the position of maxi mum value of the Hilber envelope within this search windows was marked as center of s2 sound
#sinal is vnv data
def getS1FromEcg(sinalEnvolope,RpeakPosition,sigma=300):#150ms 2000*0.15=30
    start_pos = int(RpeakPosition)
    end_pos =int(RpeakPosition + sigma)
    max_local_position = np.argmax(sinalEnvolope[start_pos:end_pos])
    s1Position = max_local_position + start_pos
    return s1Position
#get s2 position from ecg TWavePosition,the method is differet from s1
#seach s1  s1+- sigma the position of maxi mum value of the Hilber envelope within this search windows was marked as center of s2 sound
#sinal is vnv data
def getS2FromEcg(sinalEnvolope,TWavePosition,sigma=100):
    # start_pos = TWavePosition-sigma
    start_pos = int(TWavePosition)
    end_pos =int( TWavePosition + sigma)
    max_local_position = np.argmax(sinalEnvolope[start_pos:end_pos])
    s2Position = max_local_position + start_pos
    return s2Position
#get s1 position List
def getS1ListFromEcg(sinalEnvolope,RpeakPositionList,sigma=300):#150ms 2000*0.15=30
    s1List=[]
    for i in RpeakPositionList:
        s1List.append(getS1FromEcg(sinalEnvolope,i,sigma))

    return s1List
#get s2 position List
def getS2ListFromEcg(sinalEnvolope,TWavePositionList,sigma=100):
    s2List = []
    for i in TWavePositionList:
        s2List.append(getS2FromEcg(sinalEnvolope,i,sigma))
    return s2List
#--envelop-------------------------------------------------------------------
#----------------------------------------------------------------------------
def _normalize(array):
    """
    Zero mean unit variance array normalization
    Args:
        array : numpy array
    Returns:
        norm : numpy array
            array with normalized values
    """
    norm = array - np.mean(array)
    norm = norm / np.std(array)
    return norm
def getEnergyEnvelope(signal,fs):
    signal=_normalize(signal)
    signal = signal / max(signal)
    return signal*signal
def getShannonEnergyEnvelope(signal,fs):
    signal = _normalize(signal)
    signal=signal/max(signal)
    ShannonEnergy = -1.0 * signal * signal * numpy.log(1.0 * signal * signal + 0.000001)
    return ShannonEnergy
def getShannonEntropyEnvelope(signal,fs):
    signal = _normalize(signal)
    signal = signal / max(signal)
    ShannonEntropy = -1.0 * signal *  numpy.log(1.0 * signal  + 0.000001)  #shannon energy
    return ShannonEntropy

def movavg(sig, fs, wlent):
    '''
    The following function is a mathematical representation for
    a moving average. The input to the function is the signal and
    the window length in milli seconds.
    Following is the mathematical equation for the moving average.

    y[n] = 1/(2*N+1)sum(sig[i+n]) for i = -N to +N

    y[n], sig[n] are both discrete time signals

    sig is the signal and wlent is the window length in milli seconds
    '''
    sigLen = len(sig)
    wlenf = (wlent * fs) / 1000
    window = numpy.array([1] * wlenf)
    avg = numpy.convolve(sig, window, mode='full')
    # print "avg.shape", avg.shape
    avg = avg[(window.size / 2) - 1:avg.size - (window.size / 2)]
    norm = numpy.convolve(window, numpy.array([1] * avg.size), mode='full')
    norm = norm[(window.size / 2) - 1:norm.size - (window.size / 2)]
    # norm=norm[0:len(avg)]
    return numpy.divide(avg[0:sigLen], norm[0:sigLen])

def getEnergyEnvelopeVnv( zfs, fs, theta=2.0, wlent=30):
    '''
    To obtain the voiced regions in the speech segment.
    Following are the input parameters
    1. zfs is the Zero Frequency Filtered Signal
    2. fs is the sampling rate
    3. wlent is the window length required for the moving average.
    '''
    zfse = 1.0 * zfs * zfs  # squaring each of the samples: to find the ZFS energy.
    zfse_movavg = np.sqrt(movavg(zfse, fs, wlent))  # averaging across wlent window

    zfse_movavg = zfse_movavg / max(zfse_movavg)  # normalzing
    avg_energy = sum(zfse_movavg) / zfse_movavg.size  # average energy across all the window.
    voicereg = zfse_movavg * (
    zfse_movavg >= avg_energy / theta)  # selecting segments whose energy is higher than the average.
    return voicereg

def getOneSetEnvelope(signal,fs):
    # Some magic number defaults, FFT window and hop length
    N_FFT = 2048
    # We use a hop of 512 here so that the HPSS spectrogram input
    # matches the default beat tracker parameters
    HOP_LENGTH = 512
    onset_env = librosa.onset.onset_strength(y=signal,
                                             sr=fs,
                                             hop_length=HOP_LENGTH,
                                             n_fft=N_FFT,
                                             aggregate=np.median)
    return onset_env
def getOneSetMulEnvelope(signal,fs):
    # Some magic number defaults, FFT window and hop length
    N_FFT = 2048

    # We use a hop of 512 here so that the HPSS spectrogram input
    # matches the default beat tracker parameters
    HOP_LENGTH = 512
    onset_env = librosa.onset.onset_strength_multi(y=signal,
                                             sr=fs,
                                             hop_length=HOP_LENGTH,
                                             n_fft=N_FFT,
                                             aggregate=np.median)
    return onset_env
def getHilbertEnvelope(signal,fs=2000):
    hl = scipy.signal.hilbert(signal)
    hilbert_envelope = np.abs(hl)
    return hilbert_envelope
def getHarmonicEnvelope(x, fs=1000, f_LPF=8, order=1):
    """
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_LPF : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 8 Hz
    Returns:
        time : numpy array
    """
    b_low, a_low = butter(order, 2 * f_LPF / fs, 'lowpass',analog=False,output='ba')
    #b_low, a_low = butter(order, 2 * f_LPF / fs, 'low')
    print(b_low,a_low)
    he = np.exp(filtfilt(b_low, a_low, np.log(np.abs(hilbert(x)))))
    return he
def dtw_envelope(x, fs=1000, f_LPF=20, order_LPF=5):
    envelope = getHarmonicEnvelope(x, fs, f_LPF, order_LPF)
    return envelope
def readEnvelopeMatData(Path='AMatdata/wavEnvelope50hz.mat'):
    dataFile = Path
    wavdata = scio.loadmat(dataFile)
    return wavdata
#using examples:
#wavdata=cyEcgLib.readEnvelopeMatData(Path='AMatdata/wavEnvelope50hz.mat')
# for i in wavdata:
#     print i
#
# wavdata = cyEcgLib.readEnvelopeMatToDict(Path='AMatdata/wavEnvelope50hz.mat')
# wavWqrs=cyEcgLib.readWqrsMatToDict()
# for i in wavWqrs:
#     print i ,wavWqrs[i].T
def readEnvelopeMatToDict(Path='AMatdata/wavEnvelope50hz.mat'):
    print(" mat file")
    dataFile = Path
    wavdata = scio.loadmat(dataFile)

    wavEnvelope = wavdata['wavEnvelope']  # get the
    wavName = wavdata['wavName']
    wavDuration_distributions= wavdata['wavDuration_distributions']
    wavDuration_distributionsMaxMin= wavdata['wavDuration_distributionsMaxMin']
    wavHeartRateSchmidt= wavdata['wavHeartRateSchmidt']
    EnvelopDict = {}
    HeartRateDict= {}
    Duration_distributionsDict= {}
    Duration_distributionsMaxMinDict= {}
    for i in range(len(wavEnvelope)):
        name = wavName[i][0][0].split('.')[0] #get the name string ::"a0009.wav"
        #print name

        Envelope = wavEnvelope[i][0]
        HeartRateSchmidt = wavHeartRateSchmidt[i][0]
        Duration_distributions = wavDuration_distributions[i][0]
        Duration_distributionsMaxMin = wavDuration_distributionsMaxMin[i][0]

        EnvelopDict[str(name)] = Envelope.T            #4 envelop feature

        HeartRateDict[str(name)] = HeartRateSchmidt     #heart and sys interval
        Duration_distributionsDict[str(name)] = Duration_distributions  #s1 mean std s2 mean std,sys mean std, di mean std
        Duration_distributionsMaxMinDict[str(name)] = Duration_distributionsMaxMin#s1 max min s2 max min,sys max min, di max min

    return EnvelopDict,HeartRateDict,Duration_distributionsDict,Duration_distributionsMaxMinDict


#load label mat data from matlab
def readLabelToDict(Path='AMatdata/wavLabel50hz.mat'):
    print(" mat file")
    dataFile = Path
    d = scio.loadmat(dataFile)
    # Simple auxiliary function to load .mat files without
    #  the unnecesary MATLAB keys and squeezing unnecesary dimensions
    del d['__globals__']
    del d['__header__']
    del d['__version__']
    #wavdata = {k: d[k].squeeze() for k in d}
    wavdata=d
    wavName = wavdata['wavName']
    wavLabel = wavdata['wavLabel']  # get the

    LabelDict = {}
    #print(wavName)
    for i in range(len(wavLabel)):
        name = wavName[i][0][0].split('.')[0] #get the name string ::"a0009.wav"
        #print name
        labelData=wavLabel[i][0][0] - 1
        for i in range(len(labelData)):#delete  value=255,value =i+1
            if labelData[i]==255:
                labelData[i]=labelData[i-1]
        LabelDict[str(name)] =labelData
    return LabelDict
def readLabelToDict2(Path='AMatdata/wavLabel50hz.mat'):
    print(" mat file")
    dataFile = Path
    d = scio.loadmat(dataFile)
    # Simple auxiliary function to load .mat files without
    #  the unnecesary MATLAB keys and squeezing unnecesary dimensions
    del d['__globals__']
    del d['__header__']
    del d['__version__']
    #wavdata = {k: d[k].squeeze() for k in d}
    wavdata=d
    wavName = wavdata['wavName']
    wavLabel = wavdata['wavLabel']  # get the

    LabelDict = {}
    #print(wavName)
    for i in range(len(wavLabel)):
        name = wavName[i][0][0].split('.')[0] #get the name string ::"a0009.wav"
        #print name
        labelData=wavLabel[i][0] - 1
        for i in range(len(labelData)):#delete  value=255,value =i+1
            if labelData[i]==255:
                labelData[i]=labelData[i+1]
        LabelDict[str(name)] =labelData.T
        print(labelData.shape)
    return LabelDict
def custom_loadmat(file):
    """
    Simple auxiliary function to load .mat files without
    the unnecesary MATLAB keys and squeezing unnecesary
    dimensions
    """
    d = scio.loadmat(file)
    del d['__globals__']
    del d['__header__']
    del d['__version__']
    d = {k: d[k].squeeze() for k in d}
    return d
###############################################################
#SCALE THE HeartRateDict_scale reading from the function "readEnvelopeMatToDict"
# EX :outputDict=HeartRateDict_scale(dict)
def HeartRateDict_scale(dict):
    outputDict={}
    list=[]
    for str in dict:
        list.append(dict[str].flatten())#1*1*2,flatten,1*2

    array=np.asarray(list).T
    #print array.shape
    mean_heartReat=np.mean(array[0])
    std_heartReat=np.std(array[0])

    mean_sistole_mean=np.mean(array[1])
    std_sistole=np.std(array[1])

    #print("mean_heart mean,std",mean_heartReat,std_heartReat)
    #print("mean_heart mean,std", mean_sistole_mean, std_sistole)
    for str in dict:
        outputDict[str]=[float(dict[str].flatten()[0]-mean_heartReat)/std_heartReat,float(dict[str].flatten()[1]-mean_sistole_mean)/std_sistole]

    return outputDict

def readWqrsMatToDict(Path='AMatdata/wavWqrs.mat'):
    print( " read Wqrs mat file")
    dataFile = Path
    wavdata = scio.loadmat(dataFile)

    wavWqrs = wavdata['wavWqrs']  # get the
    wavName = wavdata['wavName']

    WqrsDict = {}

    for i in range(len(wavWqrs)):
        name = wavName[i][0][0].split('.')[0] #get the name string ::"a0009.wav"
        #print name

        Wqrs = wavWqrs[i][0]

        WqrsDict[str(name)] = Wqrs            #wqrs dict

    return WqrsDict
def readGqrsMatToDict(Path='AMatdata/wavWqrs.mat'):
    print( " read Wqrs mat file")
    dataFile = Path
    wavdata = scio.loadmat(dataFile)

    wavGqrs = wavdata['wavWqrs']  # get the
    wavName = wavdata['wavName']

    GqrsDict = {}

    for i in range(len(wavGqrs)):
        name = wavName[i][0][0] #get the name string ::"a0009.wav"
        #print name

        Wqrs = wavGqrs[i][0]
        GqrsDict[str(name)] = Wqrs            #wqrs dict
    return GqrsDict
#row 1:S1--row 2 :systole---row 3:S2--row4:diastole
def DurationDistributionsDict_toArray(OneDistributionsDictItem):#ex:Duration_distributionsDict['a0002']
    array=np.ones((4,2))
    for i in range(4):
        for j in range(2):
            array[i,j]=OneDistributionsDictItem[i][j].flatten()[0]
    return array


def extract(file):
    with open(file, 'r') as f:
        content = f.read().strip().split('\n')
    return content


def custom_loadmat(file):
    """
    Simple auxiliary function to load .mat files without
    the unnecesary MATLAB keys and squeezing unnecesary
    dimensions
    """
    d = scipy.io.loadmat(file)
    del d['__globals__']
    del d['__header__']
    del d['__version__']
    d = {k: d[k].squeeze() for k in d}

def to_categorical_2dTo3d(y, Tick,nb_classes=None):

    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    2D->3D
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y.shape[2])+1
    Y = np.zeros((len(y), Tick,nb_classes))

    for i in range(len(y)):
        for j in range(Tick):
            Y[i, j,y[i,j]] = 1.
    return Y
class SearchS1S2MultiSample:
    # feature size==label size,random sample short feature(length=60-200) and label from a file feature and label.
    def resampFromLongFeature(self,Feature, Label, sampleLength=60, sampleNumber=1000):
        segFeature = []
        segLabel = []
        start_index = np.random.randint(0, len(Feature) - sampleLength, (1, sampleNumber))  # generate n mumber start index for segmenting feature
        # print start_index
        for i in start_index[0]:
            segFeature.append(Feature[i:i + sampleLength])
            segLabel.append(Label[i:i + sampleLength])
        #print len(segFeature), len(segLabel)
        return segFeature, segLabel

    # get the train set and test set List from dict
    def getTheTrainSetAndTestSetList(self,LabelDict):
        List = []
        for i in LabelDict:
            List.append(i)
        return List

    # some position is error ,we will discard these file
    def getTheTrainSetAndTestSetList_DiscartError(self,LabelDict, removeList):
        List = self.getTheTrainSetAndTestSetList(LabelDict)
        for i in removeList:
            removeItem = 'a0' + i
            try:
                List.remove(removeItem)
                print("delete item sucess:" + removeItem)
            except:
                print("delete error:" + removeItem)
                # print(len(List))
        return List
    def getTheTrainSetAndTestSetList_DiscartError_conv2(self,List, removeList):
        for i in removeList:
            removeItem = i
            try:
                List.remove(removeItem)
                print("delete item sucess:" + removeItem)
            except:
                print("delete error:" + removeItem)
                # print(len(List))
        return List
    # split List to train and list from train_size
    def splitTrainTest(self,List, train_size=0.5):
        from sklearn.utils import shuffle
        shuffle_List = shuffle(List)
        len_all = len(shuffle_List)
        split_number = int(len_all * train_size)

        trainNameList = shuffle_List[0:split_number]
        testNameList = shuffle_List[split_number:]
        return trainNameList, testNameList
    def saveH5(self,fileName,TrainList, TrainLabelList, TestList, TestLabelList):
        file = h5py.File(fileName, 'w')
        print(TrainList.shape, TrainLabelList.shape, TestList.shape, TestLabelList.shape)
        file.create_dataset('TrainList', data=TrainList)
        file.create_dataset('TrainLabelList', data=TrainLabelList)
        file.create_dataset('TestList', data=TestList)
        file.create_dataset('TestLabelList', data=TestLabelList)
        #file.create_dataset('Trains1s2List', data=Trains1s2List)
        #file.create_dataset('Tests1s2List', data=Tests1s2List)
        file.close()
class FIRFilters:
    '''
    This file defines the main physical constants of the system
    '''

    # Speed of sound c=343 m/s
    c = 343.

    # distance to the far field
    ffdist = 10.

    # cut-off frequency of standard high-pass filter
    fc_hp = 300.

    # tolerance for computations
    eps = 1e-10

    def to_16b(self, signal):
        '''
        converts float 32 bit signal (-1 to 1) to a signed 16 bits representation
        No clipping in performed, you are responsible to ensure signal is within
        the correct interval.
        '''
        return ((2 ** 15 - 1) * signal).astype(np.int16)

    def clip(self, signal, high, low):
        '''
        Clip a signal from above at high and from below at low.
        '''
        s = signal.copy()

        s[np.where(s > high)] = high
        s[np.where(s < low)] = low

        return s

    def normalize(selfs, signal):
        sig = signal.copy()
        mean = np.mean(sig)
        std = np.std(sig)
        filtecg = (sig - mean) / std
        return filtecg

    def normalize_DivMax(self, signal, bits=None):
        '''
        normalize to be in a given range. The default is to normalize the maximum
        amplitude to be one. An optional argument allows to normalize the signal
        to be within the range of a given signed integer representation of bits.
        '''

        s = signal.copy()

        s /= np.abs(s).max()

        # if one wants to scale for bits allocated
        if bits is not None:
            s *= 2 ** (bits - 1)
            s = self.clip(signal, 2 ** (bits - 1) - 1, -2 ** (bits - 1))
        return s

    def angle_from_points(self, x1, x2):

        return np.angle((x1[0, 0] - x2[0, 0]) + 1j * (x1[1, 0] - x2[1, 0]))

    def normalize_pwr(self, sig1, sig2):
        '''
        normalize sig1 to have the same power as sig2
        '''

        # average power per sample
        p1 = np.mean(sig1 ** 2)
        p2 = np.mean(sig2 ** 2)

        # normalize
        return sig1.copy() * np.sqrt(p2 / p1)

    def lowpass(self, signal, Fs, fc=fc_hp, plot=False):
        '''
        Filter out the really low frequencies, default is below 50Hz
        '''

        # have some predefined parameters
        rp = 5  # minimum ripple in dB in pass-band
        rs = 60  # minimum attenuation in dB in stop-band
        n = 4  # order of the filter
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

    def highpass(self, signal, Fs, fc=fc_hp, plot=False):
        '''
        Filter out the really low frequencies, default is below 50Hz
        '''

        # have some predefined parameters
        rp = 5  # minimum ripple in dB in pass-band
        rs = 60  # minimum attenuation in dB in stop-band
        n = 4  # order of the filter
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

    def time_dB(self, signal, Fs, bits=16):
        '''
        Compute the signed dB amplitude of the oscillating signal
        normalized wrt the number of bits used for the signal
        '''

        import matplotlib.pyplot as plt

        # min dB (least significant bit in dB)
        lsb = -20 * np.log10(2.) * (bits - 1)

        # magnitude in dB (clipped)
        pos = self.clip(signal, 2. ** (bits - 1) - 1, 1.) / 2. ** (bits - 1)
        neg = -self.clip(signal, -1., -2. ** (bits - 1)) / 2. ** (bits - 1)

        mag_pos = np.zeros(signal.shape)
        Ip = np.where(pos > 0)
        mag_pos[Ip] = 20 * np.log10(pos[Ip]) + lsb + 1

        mag_neg = np.zeros(signal.shape)
        In = np.where(neg > 0)
        mag_neg[In] = 20 * np.log10(neg[In]) + lsb + 1

        plt.plot(np.arange(len(signal)) / float(Fs), mag_pos - mag_neg)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [dB]')
        plt.axis('tight')
        plt.ylim(lsb - 1, -lsb + 1)

        # draw ticks corresponding to decibels
        div = 20
        n = int(-lsb / div) + 1
        yticks = np.zeros(2 * n)
        yticks[:n] = lsb - 1 + np.arange(0, n * div, div)
        yticks[n:] = -lsb + 1 - np.arange((n - 1) * div, -1, -div)
        yticklabels = np.zeros(2 * n)
        yticklabels = range(0, -n * div, -div) + range(-(n - 1) * div, 1, div)
        plt.setp(plt.gca(), 'yticks', yticks)
        plt.setp(plt.gca(), 'yticklabels', yticklabels)

        plt.setp(plt.getp(plt.gca(), 'ygridlines'), 'ls', '--')
