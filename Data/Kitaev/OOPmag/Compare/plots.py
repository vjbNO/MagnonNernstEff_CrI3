import numpy as np
import glob
#import scipy
import sys
from scipy.optimize import curve_fit
from matplotlib import colors
import time
#from concurrent.functions import ProcessPoolExecutor
import matplotlib.pyplot as plt
font = {'size'   : 15}
import matplotlib as mpl
mpl.rc('font', **font)

def plotTopView(d,title1):
    fig,ax = plt.subplots()#2,1,sharex=True)
    #divnorm1=colors.TwoSlopeNorm(vmin=np.amin(d), vcenter=0., vmax=np.amax(d))
    im = ax.imshow(np.transpose(d),interpolation='none')#,norm=divnorm1)
    plt.ylabel('y (D)')
    plt.xlabel('x (D)')
    #fig.suptitle(title) 
    plt.title(title1)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tick_params(top='off', bottom='off', left='on', right='off', labelleft='on', labelbottom='on')
    cbar = plt.colorbar(im,ax=ax,aspect=10,shrink=0.3)
    cbar.set_ticks([np.amin(d),0,np.amax(d)])
    cbar.set_ticklabels([str(round(np.amin(d),2))+'/fs',0,str(round(np.amax(d),2))+'/fs'])
    plt.tight_layout()

def plotDiffBetweenTempGrads(d1,d2,whichgrad):
    fig1,ax = plt.subplots()
    
    
    ax.set_ylabel(r'y',fontsize=20)
    ax.set_xlabel(r'x',fontsize=20)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    
    s1 = np.transpose(d1)
    if (whichgrad == 'xgrad'):
        s2 = np.copy(np.transpose(d2))
        s2 = np.roll(s2[:,::-1],1)
        for y in range(len(s2)):
            if (y%2 == 0):
                s2[y,:] = np.roll(s2[y,:],1)
        #ax.set_title(r'$\mu((\nabla T)_x<0) - \mu((\nabla T)_x>0)$')
    elif (whichgrad == 'ygrad'):
        s2 = np.copy(np.transpose(d2))
        s2 = np.roll(s2[::-1,:],1,axis=0)
        #ax.set_title(r'$\mu((\nabla T)_y<0) - \mu((\nabla T)_y>0)$')
    else:
        print('which gradient?')
        stop
    sig = np.subtract(s1,s2)
    plotsig = sig# np.sign(sig)*sig**2

    divnorm=colors.TwoSlopeNorm(vmin=np.amin(plotsig), vcenter=0., vmax=np.amax(plotsig))
    im = ax.imshow(sig,cmap='seismic',norm=divnorm,aspect=1,interpolation='none')
    cbar = plt.colorbar(im,ax=ax,aspect=10,shrink=0.4)#,label=r'$\Delta \mu$ (1/fs)')
    cbar.ax.set_title(r'$\Delta \mu^z$')
    cbar.set_ticks([np.amin(plotsig),0,np.amax(plotsig)])
    cbar.set_ticklabels([str(round(np.amin(sig),1)),0,str(round(np.amax(sig),1))])
      
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    

    fig2,ax2 = plt.subplots(figsize=(7,3))
    if (whichgrad=='xgrad'):
        longdir = np.arange(len(sig))
        s,e=int(len(sig)/4),int(3*len(sig)/4)
        mean1 = np.mean(sig[:,s:e],axis=1)
        ax2.set_xlabel('y',fontsize=20,rotation=180)
        ax2.set_ylabel(r'$\langle \Delta\mu^z \rangle_x$',fontsize=25)
    else:
        longdir = np.arange(len(sig[0]))
        s,e=int(len(sig[0])/4),int(3*len(sig[0])/4)
        mean1 = np.mean(sig[s:e,:],axis=0)
        ax2.set_xlabel('x',fontsize=20)
        ax2.set_ylabel(r'$\langle \Delta\mu^z \rangle_y$',fontsize=25)

    ax2.grid()
    m1 = np.mean(mean1[10:-10])
    plt.plot(longdir,mean1-m1,color='black',linewidth=5)
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    plt.subplots_adjust(left=0.17,right=0.9)
    ax2.set_ylim(-0.12,0.12)

    return fig1,fig2

def CalcDiff(d1,d2):
    #make it simpler, just take difference at edge
    sig1 = np.transpose(d1)
    upperEdge = np.mean(sig1[-1,:])
    lowerEdge = np.mean(sig1[0,:])

    sig2 = np.transpose(d2)

    upperEdge2 = np.mean(sig2[-1,:])
    lowerEdge2 = np.mean(sig2[0,:])

    print('edge signal difference up minus down, for neg pos grad', upperEdge-lowerEdge,upperEdge2-lowerEdge2)
   
def plotTopViewBoth(d1,d2,graddir):
    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
    d1[d1==0]=np.nan
    d2[d2==0]=np.nan
    #fig.suptitle(r'$\Delta\mu = \mu(D<0)-\mu(D=0)$')
    #ax1.set_title(r'$(\nabla T)_x<0$')
    #ax2.set_title(r'$(\nabla T)_x>0$')
    ax1.set_ylabel('y (D)')
    ax2.set_ylabel('y (D)')
    ax2.set_xlabel('x (D)')
    d1,d2 = np.transpose(d1),np.transpose(d2)

    if (graddir=='x'):
        axis = 0
        L = len(d1)
    elif (graddir=='y'):
        axis = 1
        L = len(d1[0])
    means1 = np.nanmean(d1,axis=axis)
    meanmatrix1 = np.repeat(np.expand_dims(means1, axis=axis), repeats=L, axis=axis)
    sig1 = np.subtract(meanmatrix1,d1)
    sig1[sig1==np.nan]=0
    mi,ma=0.3*np.amin(sig1),0.3*np.amax(sig1)
    divnorm1=colors.TwoSlopeNorm(vmin=mi, vcenter=0., vmax=ma)
    im1 = ax1.imshow(sig1,cmap='seismic',norm=divnorm1)
       
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    
    cbar = plt.colorbar(im1,ax=ax1,aspect=10,shrink=0.4,label=r'$\Delta \mu$ (1/fs)')
    cbar.set_ticks([mi,0,ma])
    cbar.set_ticklabels([str(round(mi,2)),0,str(round(ma,2))])
    
    means2 = np.nanmean(d2,axis=axis)
    meanmatrix2 = np.repeat(np.expand_dims(means2, axis=axis), repeats=L, axis=axis)
    sig2 = np.subtract(meanmatrix2,d2)
    sig2[sig2==np.nan]=0
    mi2,ma2 = 0.3*np.amin(sig2),0.3*np.amax(sig2)
    divnorm2=colors.TwoSlopeNorm(vmin=mi2, vcenter=0., vmax=ma2)
    im2 = ax2.imshow(sig2,cmap='seismic',norm=divnorm2)
    
    cbar = plt.colorbar(im2,ax=ax2,aspect=10,shrink=0.4,label=r'$\Delta \mu$ (1/fs)')
    cbar.set_ticks([mi2,0,ma2])
    cbar.set_ticklabels([str(round(mi2,2)),0,str(round(ma2,2))])
    
    #plt.tight_layout()

def plotTopViewWithDiff(d1,d2):
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    
    #fig.suptitle(r'$\Delta\mu = \mu(D<0)-\mu(D=0)$')
    #ax1.set_title(r'$(\nabla T)_x<0$')
    #ax2.set_title(r'$(\nabla T)_x>0$')
    #ax3.set_title(r'$(\nabla T)_x<0 - (\nabla T)_x>0$')
    ax1.set_ylabel('y (D)')
    ax2.set_ylabel('y (D)')
    ax3.set_ylabel('y (D)')
    ax3.set_xlabel('x (D)')
    
    means1 = np.mean(d1,axis=1)
    meanmatrix1 = np.repeat(np.expand_dims(means1, axis=1), repeats=len(d1[0]), axis=1)
    sig1 = np.subtract(np.transpose(meanmatrix1),np.transpose(d1))
    print('sig1',np.amin(sig1),np.amax(sig1))
    divnorm1=colors.TwoSlopeNorm(vmin=0.3*np.amin(sig1), vcenter=0., vmax=0.3*np.amax(sig1))
    im1 = ax1.imshow(sig1,cmap='seismic',norm=divnorm1)
       
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    cbar = plt.colorbar(im1,ax=ax1,aspect=10,shrink=0.4,label=r'$\Delta \mu$ (1/fs)')
    cbar.set_ticks([0.3*np.amin(sig1),0,0.3*np.amax(sig1)])
    cbar.set_ticklabels([str(round(0.3*np.amin(sig1),2)),0,str(round(0.3*np.amax(sig1),2))])
    
    means2 = np.mean(d2,axis=1)
    meanmatrix2 = np.repeat(np.expand_dims(means2, axis=1), repeats=len(d2[0]), axis=1)
    sig2 = np.subtract(np.transpose(meanmatrix2),np.transpose(d2))
    print('sig2',np.amin(sig2),np.amax(sig2))
    divnorm2=colors.TwoSlopeNorm(vmin=np.amin(sig2), vcenter=0., vmax=np.amax(sig2))
    im2 = ax2.imshow(sig2,cmap='seismic',norm=divnorm2)
    #divnorm2=colors.TwoSlopeNorm(vmin=np.amin(d2), vcenter=0., vmax=np.amax(d2))
    #im2 = ax2.imshow(np.transpose(d2),cmap='seismic',norm=divnorm2)
       
    cbar = plt.colorbar(im2,ax=ax2,aspect=10,shrink=0.4,label=r'$\Delta \mu$ (1/fs)')
    cbar.set_ticks([np.amin(sig2),0,np.amax(sig2)])
    cbar.set_ticklabels([str(round(np.amin(sig2),2)),0,str(round(np.amax(sig2),2))])
    
    sig3 = np.subtract(sig2,sig1)
    divnorm3=colors.TwoSlopeNorm(vmin=np.amin(sig3), vcenter=0., vmax=np.amax(sig3))
    im3 = ax3.imshow(sig3,cmap='seismic',norm=divnorm3)
    cbar = plt.colorbar(im3,ax=ax3,aspect=10,shrink=0.4,label=r'$\Delta \mu$ (1/fs)')
    cbar.set_ticks([np.amin(sig3),0,np.amax(sig3)])
    cbar.set_ticklabels([str(round(np.amin(sig3),2)),0,str(round(np.amax(sig3),2))])

def giveAccumulatedSignal(dat,howmany):
    d = np.transpose(dat) #to be coherent with the plotting above. axis 0 is the y direction, axis 1 the x direction
    d2 = np.copy(d)
    for y in range(len(d2)): #shift every other row to level out difference that comes from projecting the honeycomb into the matrix. makes a difference when reading from right vs left
        if (y%2 == 0):
            d2[y,:] = np.roll(d2[y,:],1)
    #print('data shape:',d.shape)
    leftedge,rightedge,upperedge,loweredge = [],[],[],[]
    countleft,countright=0,0
    ll=0
    while countleft<howmany:
        s = np.sum(d[:,ll])
        #print('sum in the line',s)
        if (np.abs(s)>1e-5):
            leftedge.append(np.sum(d[:,ll]))
            countleft +=1
        ll +=1
        s=0
    lr=0
    while countright<howmany:
        s = np.sum(d2[:,-lr])
        #print('sum in the line',s)
        if (np.abs(s)>1e-5):
            rightedge.append(np.sum(d2[:,-lr]))
            countright +=1
        lr +=1
        s=0
    #print('for the left signal, ',ll, 'lines were needed to integrate ', howmany,' lines')
    #print('for the right signal, ',lr,' lines were needed')

    left = np.mean(leftedge)
    right = np.mean(rightedge)
    lu=0
    countupper,countlower = 0,0
    while countupper<howmany:
        s = np.sum(d[lu,:])
        #print('sum in the line',s)
        if (np.abs(s)>1e-5):
            upperedge.append(np.sum(d[lu,:]))
            countupper +=1 
        lu+=1
    ld=0
    while countlower<howmany:
        s = np.sum(d[-ld,:])
        #print('sum in the line',s)
        if (np.abs(s)>1e-5):
            loweredge.append(np.sum(d[-ld,:]))
            countlower +=1
        ld +=1
    upper = np.mean(upperedge)
    lower = np.mean(loweredge)
    #print('for the upper signal, ',lu,' lines were needed')
    #print('for the lower signal, ',ld,' lines were needed')

   
    #print('left,right,upper,lower:',left,right,upper,lower)
    #print('right/left, upper/lower',right/left,lower/upper)
    print('left-right,upper-lower')
    print(left-right,upper-lower)


def PlotTransverseAverage(d1,d2,d3,d4):
    fig,(ax1,ax2) = plt.subplots(2,1)
    d1[d1==0]=np.nan
    d2[d2==0]=np.nan
    d3[d3==0]=np.nan
    d4[d4==0]=np.nan
    #fig.suptitle(r'$\Delta\mu = \mu(D<0)-\mu(D=0)$')
    #ax1.set_title(r'$(\nabla T)_x<0$')
    #ax2.set_title(r'$(\nabla T)_x>0$')
    ax1.set_ylabel(r'$\langle\mu\rangle$')
    ax2.set_ylabel(r'$\langle\mu\rangle$')
    ax1.set_xlabel('y (D)')
    ax2.set_xlabel('x (D)')
    d1,d2 = np.transpose(d1),np.transpose(d2)
    d3,d4 = np.transpose(d3),np.transpose(d4)

    
    axis = 1
    #s,e=int(len(d1[0,:])/4),int(3*len(d1[0,:])/4)
    #print(s,e)
    means1 = np.nanmean(d1[:,:],axis=axis)
    means1 = 0.5*(means1+np.roll(means1,1))
    m1 = np.nanmean(means1[10:-10])
    ax1.plot(np.arange(len(means1)),means1-m1,label='Neg Grad x',color='red',marker='o')
    m1 = np.nanmean(means1[10:-10])
    #ax1.hlines(m1,0,len(means1),color='grey')
    ax1.set_ylim(-0.02,0.02)
    means2 = np.nanmean(d2[:,:],axis=axis)
    means2 = 0.5*(means2+np.roll(means2,1))
    m2 = np.nanmean(means2[10:-10])
    ax1.plot(np.arange(len(means2)),means2-m2,label='Pos Grad x',color='black',marker='s')
    #m2 = np.nanmean(means2[10:-10])
    #ax1.hlines(m2,0,len(means2),color='navy')
    ax1.legend(loc='upper center')
    ax1.grid()

    axis = 0
    #s,e=int(len(d1[:,0])/4),int(3*len(d1[:,0])/4)
    #print(s,e)
    means3 = np.nanmean(d3[:,:],axis=axis)
    #means3 = 0.5*(means3+np.roll(means3,1))
    m3 = np.nanmean(means3[10:-10])
    ax2.plot(np.arange(len(means3)),means3-m3,label='Neg Grad y',color='red',marker='o')
    #m3 = np.nanmean(means3[10:-10])
    #ax2.hlines(m3,0,len(means3),color='pink')
    ax2.set_ylim(-0.02,0.02)
    means4 = np.nanmean(d4[:,:],axis=axis)
    means4 = 0.5*(means4+np.roll(means4,1))
    m4 = np.nanmean(means4[10:-10])
    ax2.plot(np.arange(len(means4)),means4-m4,label='Pos Grad y',color='black',marker='s')
    #m4 = np.nanmean(means4[10:-10])
    #ax2.hlines(m4,0,len(means4),color='violet')
    ax2.legend(loc='upper center')
    ax2.grid()
    plt.subplots_adjust(hspace=0.5)

if __name__ == '__main__':
    PosXgrad = np.loadtxt('Data/PosXgrad_TimeAvMu.txt') 
    NegXgrad = np.loadtxt('Data/NegXgrad_TimeAvMu.txt')
    PosYgrad = np.loadtxt('Data/PosYgrad_TimeAvMu.txt') 
    NegYgrad = np.loadtxt('Data/NegYgrad_TimeAvMu.txt')
    '''
    data = [NegXgrad,PosXgrad,NegYgrad,PosYgrad]
    names = ['NegXgrad','PosXgrad','NegYgrad','PosYgrad']
    for d in range(len(data)):
        dataset = data[d]
        print(names[d])
        giveAccumulatedSignal(dataset,10)
    
    PosXgradstd = np.loadtxt('Data/PosXgrad_stddev_TimeAvMu.txt') 
    NegXgradstd = np.loadtxt('Data/NegXgrad_stddev_TimeAvMu.txt')
    PosYgradstd = np.loadtxt('Data/PosYgrad_stddev_TimeAvMu.txt') 
    NegYgradstd = np.loadtxt('Data/NegYgrad_stddev_TimeAvMu.txt')
    datastd = [NegXgradstd,PosXgradstd,NegYgradstd,PosYgradstd]
    namesstd = ['NegXgrad std','PosXgrad std','NegYgrad std','PosYgrad std']
    for d in range(len(datastd)):
        dataset = datastd[d]
        print(namesstd[d])
        giveAccumulatedSignal(dataset,10)
    stop'''
    fig1,fig2 = plotDiffBetweenTempGrads(PosXgrad,NegXgrad,'xgrad')
    fig1.savefig('Plots/Kitaev_OOP_PosMinusNegXgrad_MuZ.pdf')
    fig2.savefig('Plots/Kitaev_OOP_PosMinusNegXgrad_MuZav.pdf')
    
    
    fig3,fig4=plotDiffBetweenTempGrads(PosYgrad,NegYgrad,'ygrad')
    fig3.savefig('Plots/Kitaev_OOP_PosMinusNegYgrad_MuZ.pdf')
    fig4.savefig('Plots/Kitaev_OOP_PosMinusNegYgrad_MuZav.pdf')

    stop
    PlotTransverseAverage(NegXgrad,PosXgrad,NegYgrad,PosYgrad)
    plt.savefig('Plots/Kitaev_IP_transverseav_bothGrads.pdf')
    
