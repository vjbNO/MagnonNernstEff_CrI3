import numpy as np
import sys
import matplotlib.pyplot as plt
font = {'size'   : 15}
import matplotlib as mpl
import glob
mpl.rc('font', **font)

dt = 0.1e-12 #100fs output time step
a = 1e-9 #does not matter since we normalize anyways after

dataUpCenter = np.sort(glob.glob('ProcessedData/*UpCenter*'))
#dataDownCenter = np.sort(glob.glob('ProcessedData/*DownCenter*'))
dataUpR = np.sort(glob.glob('ProcessedData/*UpRight*'))
#dataDownR = np.sort(glob.glob('ProcessedData/*DownRight*'))

#leftdata = [dataUpCenter,dataDownCenter]
#right = [dataUpR,dataDownR]

bothdata = [dataUpCenter,dataUpR]
#updown center will be for left edge, up down right will be for right edge 

#for left edge
for d in range(2):
    data = bothdata[d]
    print(d)
    tlen = len(data)
    example = np.loadtxt(data[0])
    Ny = example.shape[0]
    Nx = example.shape[1]
    bigmatrix = np.zeros((tlen,Ny,Nx))

    for t in range(tlen):
        rawdata=np.loadtxt(data[t]) 
        #if (d==0):
        bigmatrix[t,:,:]= rawdata[:,:]
        #else:
        #    bigmatrix[t,:,:]= rawdata[:,-3:]

    for l in range(Nx):
        s = np.fft.fftshift(np.abs(np.fft.fft2(bigmatrix[:,:,l])))
        if (l==0):
            Line = np.copy(s)
            BulkLine = np.zeros(Line.shape)
            OuterLine = np.zeros(Line.shape)
        else:
            if (l>5 and l<Nx-5):
                BulkLine = np.add(Line,s)
            if (l<3):
                OuterLine = np.add(Line,s)

    Line = np.fft.fftshift(Line,axes=1)
    if (d==0):
        np.savetxt('SigBulk_sublatticeA.txt',BulkLine)
        np.savetxt('SigLeftEdge.txt',OuterLine)
    if (d==1):
        np.savetxt('SigBulk_sublatticeB.txt',BulkLine)
        np.savetxt('SigRightEdge.txt',OuterLine)

    s=0
    Line = 0
    BulkLine = 0
    OuterLine = 0


names = ['SigLeftEdge','SigRightEdge','SigBulk_sublatticeA','SigBulk_sublatticeB']

for s in range(len(names)):
    sig = np.loadtxt(names[s]+'.txt')
    fig,ax = plt.subplots()
    tlen = len(sig)
    Nx = len(sig[0])
    hbar = 6.582e-16 #eV/Hz
    freq = 2*np.pi*np.fft.fftfreq(tlen,dt)*hbar*1000
    ks = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Nx,a))
    kmax = ks[-1]*a#2*np.pi*ks[int(0.5*len(ks))]
    fmax = np.abs(freq[int(0.5*len(freq))]) 
    result = [sig[i] for i in range(int(0.5*tlen),tlen)]
    plt.ylabel(r'$E^{DM}_{q\sigma}$ (meV)')
    #plt.xlim(-kmax,kmax)
    plt.xlabel('k(a)')
    #plt.xticks([-kmax/2,-3*kmax/8,0,3*kmax/8,kmax/2],['X','P corresp. K', 'Gamma', 'P corresp. K', 'X'])
    #plt.xticks([0,0.34*kmax,kmax/2],['Gamma', 'P', 'X'])
    plt.imshow(np.log(result),extent=[0,kmax,0,fmax],aspect='auto',origin='lower',interpolation='bilinear')
    plt.savefig('Plots/DM_OOP_NegXgrad_'+names[s]+'.pdf')
    plt.close()
