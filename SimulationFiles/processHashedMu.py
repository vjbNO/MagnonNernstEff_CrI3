import numpy as np
import glob


def timeAv(data,start):
    L = len(data)
    d0 = np.loadtxt(data[0])
    Nx,Ny = len(d0),len(d0[0])
    bigmatrix = np.zeros((L-start,Nx,Ny))

    for l in range(L-start):
        bigmatrix[l] = np.loadtxt(data[l])
        print(l)

    timeav = np.mean(bigmatrix[start:,:,:],axis=0)
    timeavstd = np.std(bigmatrix[start:,:,:],axis=0)
    
    return timeav,timeavstd

def processMu():
    comps_z = np.sort(glob.glob('ProcessedData/SpinPumpingMuz_run_0_Hexdata_*'))
    timeav,timeavstd = timeAv(comps_z,0) #has been equilibrated for 500ps already
    np.savetxt('ProcessedData/TimeAvMu_run_0_xcomp.txt',timeav)
    np.savetxt('ProcessedData/TimeAvMu_stddev_run_0_xcomp.txt',timeavstd)

if __name__=='__main__':
    processMu()


