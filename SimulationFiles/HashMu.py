import scipy as sp
import numpy as np
import time
import pandas as pd
import glob
import heapq
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

run = '0' #sys.argv[1]

def Definecenters(pos):
        shortdy = (pos[2,1]-pos[1,1])/2
        longdy = (pos[3,1]-pos[0,1])/2
        dx = (pos[4,0]-pos[0,0])/2
        firstcenter = [pos[0,0],pos[1,1]+shortdy] #[pos[0,0]-dx/4+1,pos[1,1]+shortdy/2+1] #first center
        HexDistX = 2*dx 
        HexDistY = 2*longdy+2*shortdy 
        lastcenter = [-1*firstcenter[0],-1*firstcenter[1]]
        # Number of hexagons
        Nx = int((lastcenter[0]-firstcenter[0])/HexDistX)
        Ny = int((lastcenter[1]-firstcenter[1])/HexDistY)
        
        
        padding = 3

        HexPositions = np.mgrid[firstcenter[0]-padding*HexDistX:lastcenter[0]+padding*HexDistX:HexDistX,firstcenter[1]-padding*HexDistY:lastcenter[1]+padding*HexDistY:HexDistY]
        HexPosOffset = np.mgrid[firstcenter[0]+dx-padding*HexDistX:lastcenter[0]+dx+padding*HexDistX:HexDistX,firstcenter[1]+longdy+shortdy-padding*HexDistY:lastcenter[1]+longdy+shortdy+padding*HexDistY:HexDistY]
        #plt.scatter(pos[:,0].astype(int),pos[:,1].astype(int))
        #plt.scatter(HexPositions[0,:],HexPositions[1,:])
        #plt.scatter(HexPosOffset[0,:],HexPosOffset[1,:])
        #plt.show()
        #stop
        return Nx,Ny,HexPositions,HexPosOffset,dx,shortdy,longdy,padding

def FindData(Nx,Ny,pos_swap,HexPositions,HexPosOffset,dx,dy1,dy2,coorddata,padding):
    maxY = int(2*Ny)+2*padding
    HexData = np.zeros((Nx+2*padding,maxY+2*padding,2)) #add white (zeros) buffer layer to be sure to capture everything
    #print(HexData.shape)
    
    e=[]
    #left to right
    for nx in range(Nx+2*padding):#range(Nx):
        for ny in range(maxY+2*padding): 
            if (ny%2==0):
                #print('normal')
                center = HexPositions[:,nx,int(ny/2)]
            else:
                #print('offset')
                center = HexPosOffset[:,nx,int(ny/2)]
            #print('nx,ny',nx,ny)
            mu = []
            x2,y2 = center[0],center[1] 
            #print('center:',x2,y2)
            positions2 = [(np.round(x2-dx,0),np.round(y2-dy1,0)),(np.round(x2-dx,0),np.round(y2+dy1,0)),(np.round(x2,0),np.round(y2+dy2,0)),(np.round(x2+dx,0),np.round(y2+dy1,0)),(np.round(x2+dx,0),np.round(y2-dy1,0)),(np.round(x2,0),np.round(y2-dy2,0))]
            count=0

            for pos in positions2:
                if pos in pos_swap:
                    mup = (pos_swap[pos])
                    mu.append(mup)
                    count +=1
            positions = []
            if (count>0): # (np.sum(np.abs(mu)-1e-10)>0):
                HexData[nx,ny,0]=np.sum(mu)
            HexData[nx,ny,1]=count
       
    #plt.imshow(HexData[:,:,1])
    #plt.savefig('counts.pdf')
    return HexData


def task(dataname): #what comes in is a singular data file name (string)
        #print('step:',timestep)
        #dataname = datalist[timestep] 
        print('now working on dataset ',dataname)
        
        coorddata = np.loadtxt('./atoms-spin-pumping-coords.data',skiprows=1)#sys.argv[2],skiprows=1)
        
        savedir = './ProcessedData/' #sys.argv[1]
        
        realtime = (dataname.split('-')[3]).split('.')[0] #extract the time stamp from the file name
        print('realtime:',realtime)
        rawdata = np.loadtxt(dataname,skiprows=1)
        #print(type(rawdata),rawdata)
        
        #coords = {}
        data = {}

        for i in range(len(coorddata)): #mux is 'key', coords are 'value'. Make mux unique with small marker, because if keys are the same, the item gets replaced!
                data[rawdata[i,0]*(1+1e-15*i)] = (np.round(coorddata[i,0],0),np.round(coorddata[i,1],0))

        orderedY = dict(sorted(data.items(), key=lambda item: item[1][1]))
        orderedYX = dict(sorted(data.items(), key=lambda item: item[1][0]))

        # swap ID and position
        def get_swap_dict(d):
                return {v: k for k, v in d.items()}

        pos_swap = get_swap_dict(orderedYX) #now coords are the keys and mux are values

        Nx,Ny,HexPositions,HexPosOffset,dx,dy1,dy2,padding = Definecenters(coorddata)

        HexData = FindData(Nx,Ny,pos_swap,HexPositions,HexPosOffset,dx,dy1,dy2,coorddata,padding)
        
        np.savetxt(savedir+'SpinPumpingMuz_run_'+run+'_Hexdata_time_'+realtime+'.txt',HexData[:,:,0])
        #print('counts:',HexData[:,:,1])
        #plt.imshow(np.transpose(HexData[:,:,1]))
        #plt.show()
        #stop

def main():
        
        #savedir = './ProcessedData/' #sys.argv[1]
        #coorddata = np.loadtxt('./atoms-spin-pumping-coords.data',skiprows=1)#sys.argv[2],skiprows=1)
        datalist = np.sort(glob.glob('atoms-spin-pumping-000*.data'))# ['./AllTerms_1000x100p5_50psMuResolution/NegTorque/run_1/atoms-spin-pumping-00000010.data']#sys.argv[3:]
        
        #task('atoms-spin-pumping-00000001.data')
        #stop
        
        with ProcessPoolExecutor(25) as executor:
                executor.map(task,datalist)

if __name__ == '__main__':
        main()
