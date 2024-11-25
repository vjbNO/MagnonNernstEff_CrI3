import numpy as np
import sys
import matplotlib.pyplot as plt
font = {'size'   : 20}
import matplotlib as mpl
import glob
mpl.rc('font', **font)

dt = 0.1e-12 #100fs output time step
a = 1e-9 #does not matter since we normalize anyways after

dataUpCenter = np.sort(glob.glob('ProcessedData/Ycomp*UpCenter*'))
dataDownCenter = np.sort(glob.glob('ProcessedData/Ycomp*DownCenter*'))
dataUpR = np.sort(glob.glob('ProcessedData/Ycomp*UpRight*'))
dataDownR = np.sort(glob.glob('ProcessedData/Ycomp*DownRight*'))
'''
alldata = [dataUpCenter,dataDownCenter,dataUpR,dataDownR]

for d in range(len(alldata)):
    data = alldata[d]
    print(d)
    tlen = len(data)
    example = np.loadtxt(data[0])
    Ny = example.shape[0]
    Nx = example.shape[1]
    bigmatrix = np.zeros((tlen,Ny,Nx))
    rawdata = 0
    for t in range(tlen):
        rawdata=np.loadtxt(data[t])  
        bigmatrix[t,:,:]= rawdata
        
    
    Xallsig = np.zeros((Nx,tlen,Ny)) #for each x position FT along the y direction
    for x in range(Nx):
        Xallsig[x,:,:] = np.fft.fftshift(np.abs(np.fft.fft2(bigmatrix[:,:,x])))

    print('first FT along the Y direction')
    # for the bulk: average over all but edges
    Ysighere = np.mean(Xallsig,axis=0) #for the bulk signal cut off edges
    Ysig = np.fft.fftshift(Ysighere,axes=1)
    if (d==0):
        bulksigY = np.copy(Ysig)
    if (d>0):
        bulksigY = np.add(bulksigY,Ysig) #should sum up all the contributions
    
    # for edges only take certain
    #s1=0
    #for l in range(6):
    if (d==2):
        sigLeftLine = np.fft.fftshift(np.abs(np.fft.fft2(bigmatrix[:,:,1])))
        
        sigLeftLine = np.fft.fftshift(sigLeftLine,axes=1)
        np.savetxt('Ycomp_SigLeftEdge.txt',sigLeftLine)
   
    
        sigRightLine = np.fft.fftshift(np.abs(np.fft.fft2(bigmatrix[:,:,-1])))
     
        sigRightLine = np.fft.fftshift(sigRightLine,axes=1)
        np.savetxt('Ycomp_SigRightEdge.txt',sigRightLine)

    print('now FT along the X direction')
    Xallsig = np.zeros((Ny,tlen,Nx)) #for each line along y FT along the x direction
    for y in range(Ny):
        Xallsig[y,:,:] = np.fft.fftshift(np.abs(np.fft.fft2(bigmatrix[:,y,:])))
 

    sighereX = np.mean(Xallsig,axis=0)
    sigX = np.fft.fftshift(sighereX,axes=1)
    if (d==0):
        bulksigX = np.copy(sigX)
    if (d>0):
        bulksigX = np.add(bulksigX,sigX) #should sum up all the contributions
    
    if (d==0):
        sigUpperLine = np.fft.fftshift(np.abs(np.fft.fft2(bigmatrix[:,-1,:])))
        sigUpperLine = np.fft.fftshift(sigUpperLine,axes=1)
        np.savetxt('Ycomp_SigUpperEdge.txt',sigUpperLine)
    
    if (d==1):
        sigLowerLine = np.fft.fftshift(np.abs(np.fft.fft2(bigmatrix[:,2,:])))
        sigLowerLine = np.fft.fftshift(sigLowerLine,axes=1)
        np.savetxt('Ycomp_SigLowerEdge.txt',sigLowerLine)
    
    

np.savetxt('Ycomp_BulkSignal_alongX.txt',bulksigX)
np.savetxt('Ycomp_BulkSignal_alongY.txt',bulksigY)

'''
names = ['SigLeftEdge','SigRightEdge','SigUpperEdge','SigLowerEdge']#,'BulkSignal_alongX','BulkSignal_alongY']

for s in range(len(names)):
    sig = np.loadtxt('Ycomp_'+names[s]+'.txt')
    fig,ax = plt.subplots()
    tlen = len(sig)
    Nx = len(sig[0])
    hbar = 6.582e-16 #eV/Hz
    freq = 2*np.pi*np.fft.fftfreq(tlen,dt)*hbar*1000
    ks = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Nx,a))
    kmax = ks[-1]*a#2*np.pi*ks[int(0.5*len(ks))]
    fmax = np.abs(freq[int(0.5*len(freq))]) 
    result = [sig[i] for i in range(int(0.5*tlen),tlen)]
    #plt.ylabel(r'$E^{\mathrm{DM}}_{q\sigma}$ (meV)')
    if (s==0 or s==1):
        plt.xlabel(r'$q_y\tilde{a}$')
        plt.ylabel(r'$E^{\mathrm{DM}}_{q_y}$ (meV)')
    else:
        plt.xlabel(r'$q_xa$')
        plt.ylabel(r'$E^{\mathrm{DM}}_{q_x}$ (meV)')
    plt.subplots_adjust(0.2,0.2,0.9,0.9)
    result = np.log(result)
    plt.xticks([0,np.pi/2,np.pi],[r'$0$',r'$\pi/2$',r'$\pi$'])
    plt.imshow(result,extent=[0,kmax*1.02,0,fmax],aspect='auto',origin='lower',interpolation='bilinear',cmap='viridis',vmin=0.3*np.amax(result),vmax=np.amax(result))
    plt.savefig('Plots/DM_IP_NegXgrad_'+names[s]+'.pdf')
    plt.close()
    sig=0
