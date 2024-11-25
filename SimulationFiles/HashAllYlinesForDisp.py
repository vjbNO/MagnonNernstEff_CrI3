import scipy as sp
import numpy as np
import time
import pandas as pd
import glob
import heapq
import sys
#import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def Definecenters(pos):
        shortdy = (pos[2,1]-pos[1,1])/2
        longdy = (pos[3,1]-pos[0,1])/2
        dx = (pos[4,0]-pos[0,0])/2
        firstcenter = [pos[0,0]-dx/4+1,pos[1,1]+shortdy/2+1] #first center
        HexDistX = 2*dx 
        HexDistY = 2*longdy+2*shortdy 
        lastcenter = np.amax(pos[:,0]),np.amax(pos[:,1])-longdy #pos[-2,-2]-dx/4,pos[-1,-1]+shortdy/2 # [-1*firstcenter[0],-1*firstcenter[1]]
        # Number of hexagons
        Nx = int((lastcenter[0]-firstcenter[0])/HexDistX)-1 # int cuts off the decimal
        Ny = int((lastcenter[1]-firstcenter[1])/HexDistY)
        
        padding = 1

        HexPositions = np.mgrid[firstcenter[0]-padding*HexDistX:lastcenter[0]+padding*HexDistX:HexDistX,firstcenter[1]-padding*HexDistY:lastcenter[1]+padding*HexDistY:HexDistY]
        HexPosOffset = np.mgrid[firstcenter[0]+dx-padding*HexDistX:lastcenter[0]+dx+padding*HexDistX:HexDistX,firstcenter[1]+longdy+shortdy-padding*HexDistY:lastcenter[1]+longdy+shortdy+padding*HexDistY:HexDistY]
        #plt.scatter(pos[:,0].astype(int),pos[:,1].astype(int))
        #plt.scatter(HexPositions[0,:],HexPositions[1,:])
        #plt.scatter(HexPosOffset[0,:],HexPosOffset[1,:])
        #plt.show()
        #stop
        return Nx,Ny,HexPositions,HexPosOffset,dx,shortdy,longdy,padding

def FindData(Nx,Ny,pos_swap,HexPositions,HexPosOffset,dx,dy1,dy2,coorddata,padding):
    maxY = int(2*Ny)+int(2*padding)
    #HexData = np.zeros((Nx+2*padding,maxY+2*padding,2)) #add white (zeros) buffer layer to be sure to capture everything
    Downline,Upline,UpPeakLine,DownPeakLine = np.zeros((maxY+2*padding,Nx+2*padding)),np.zeros((maxY+2*padding,Nx+2*padding)),np.zeros((maxY+2*padding,Nx+2*padding)), np.zeros((maxY+2*padding,Nx+2*padding)) 
    e=[]
    #left to right
    for nx in range(Nx+2*padding):#range(Nx):
        for ny in range(maxY+2*padding): 
            if (ny%2==1):
                center = HexPositions[:,nx,int(ny/2)] 
            else:
                center = HexPosOffset[:,nx,int(ny/2)]
            mu = []
            x,y = center[0],center[1]  
            #print('center',x,y)
            #positions2 = [(np.round(x2-dx,0),np.round(y2-dy1,0)),(np.round(x2-dx,0),np.round(y2+dy1,0)),(np.round(x2,0),np.round(y2+dy2,0)),(np.round(x2+dx,0),np.round(y2+dy1,0)),(np.round(x2+dx,0),np.round(y2-dy1,0)),(np.round(x2,0),np.round(y2-dy2,0))]
            #positionsDownLine = [(np.round(x2-dx,0),np.round(y2-dy1,0)),(np.round(x2+dx,0),np.round(y2-dy1,0))]
            #positionsUpLine = [(np.round(x2-dx,0),np.round(y2+dy1,0)),(np.round(x2+dx,0),np.round(y2+dy1,0))]
            

            count1=0
            count2=0
            count3=0
            count4=0

            def find(x,y,count1,count2,count3,count4):
                #DownLineL = (np.round(x-dx,0),np.round(y-dy1,0))
                DownLineR = (np.round(x+dx,0),np.round(y-dy1,0))
                #UpLineL = (np.round(x-dx,0),np.round(y+dy1,0))
                UpLineR = (np.round(x+dx,0),np.round(y+dy1,0))
                DownPeak = (np.round(x,0),np.round(y-dy2,0))
                UpPeak = (np.round(x,0),np.round(y+dy2,0))
                #print('looking at positions DLL,DLR,ULL,ULR,DP,UP:',DownLineL,DownLineR,UpLineL,UpLineR,DownPeak,UpPeak)
                
                #if DownLineL in pos_swap:
                #    mup = (pos_swap[DownLineL])
                #    Downline[ny,2*nx] = mup
                #    count1 +=1
                if DownLineR in pos_swap:
                    mup = (pos_swap[DownLineR])
                    Downline[ny,nx] = mup
                    count1 +=1
                
                #if UpLineL in pos_swap:
                #    mup = (pos_swap[UpLineL])
                #    Upline[ny,2*nx] = mup
                #    count2 +=1
                if UpLineR in pos_swap:
                    mup = (pos_swap[UpLineR])
                    Upline[ny,nx] = mup
                    count2 +=1

                if DownPeak in pos_swap:
                    mup = (pos_swap[DownPeak])
                    DownPeakLine[ny,nx]=mup
                    count3 +=1
                if UpPeak in pos_swap:
                    mup = (pos_swap[UpPeak])
                    UpPeakLine[ny,nx]=mup
                    count4 +=1

                return count1,count2,count3,count4

            count1,count2,count3,count4 = find(x,y,count1,count2,count3,count4)
            '''
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x,y-1,count1,count2,count3,count4)
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x-1,y,count1,count2,count3,count4)
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x,y+1,count1,count2,count3,count4)
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x+1,y,count1,count2,count3,count4)
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x-1,y-1,count1,count2,count3,count4)
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x+1,y+1,count1,count2,count3,count4)
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x+1,y-1,count1,count2,count3,count4)
            if (count1<2 or count2<2 or count3<1 or count4<1):
                count1,count2,count3,count4 = find(x-1,y+1,count1,count2,count3,count4)
            '''
            #print('counts1 to count4:',count1,count2,count3,count4)
            #stop
            positions = []
            #if (np.abs(np.sum(mu))>0.0):
            #    HexData[nx,ny,0]=np.mean(mu)
            #HexData[nx,ny,1]=count
    #print('counts:',HexData[:,:,1])
    return Downline,Upline,DownPeakLine,UpPeakLine


def task(dataname): #what comes in is a singular data file name (string)
        #print('step:',timestep)
        #dataname = datalist[timestep] 
        print('now working on dataset ',dataname)
        
        c = np.loadtxt('data/atoms-coords.data',skiprows=1)#sys.argv[2],skiprows=1)
        coorddata = c[:,2:4]
        
        savedir = './ProcessedData/' #sys.argv[1]
        
        realtime = (dataname.split('-')[1]).split('.')[0] #extract the time stamp from the file name
        #print('realtime:',realtime)

        rawdata = np.loadtxt(dataname,skiprows=1)
        #print(type(rawdata),rawdata)
        
        #coords = {}
        dataX,dataY,dataZ = {},{},{}

        for i in range(len(coorddata)): #mux is 'key', coords are 'value'. Make mux unique with small marker, because if keys are the same, the item gets replaced!
                #dataX[rawdata[i,0]*(1+1e-15*i)] = (np.round(coorddata[i,0],0),np.round(coorddata[i,1],0))
                dataY[rawdata[i,1]*(1+1e-15*i)] = (np.round(coorddata[i,0],0),np.round(coorddata[i,1],0))
                #dataZ[rawdata[i,2]*(1+1e-15*i)] = (np.round(coorddata[i,0],0),np.round(coorddata[i,1],0))

        #orderedY_x = dict(sorted(dataX.items(), key=lambda item: item[1][1]))
        #orderedYX_x = dict(sorted(dataX.items(), key=lambda item: item[1][0]))
        orderedY_y = dict(sorted(dataY.items(), key=lambda item: item[1][1]))
        orderedYX_y = dict(sorted(dataY.items(), key=lambda item: item[1][0]))
        #orderedY_z = dict(sorted(dataZ.items(), key=lambda item: item[1][1]))
        #orderedYX_z = dict(sorted(dataZ.items(), key=lambda item: item[1][0]))

        # swap ID and position
        def get_swap_dict(d):
                return {v: k for k, v in d.items()}

        #pos_swap_x = get_swap_dict(orderedYX_x) #now coords are the keys and mux are values
        pos_swap_y = get_swap_dict(orderedYX_y) #now coords are the keys and mux are values
        #pos_swap_z = get_swap_dict(orderedYX_z) #now coords are the keys and mux are values

        Nx,Ny,HexPositions,HexPosOffset,dx,dy1,dy2,padding = Definecenters(coorddata)
        
        
        #plt.scatter(HexPositions[0,:,:],HexPositions[1,:,:],color='black')
        #plt.scatter(HexPosOffset[0,:,:],HexPositions[1,:,:],color='blue')
        #plt.scatter(coorddata[:,0],coorddata[0,:],color='red')
        #plt.show()

        DownLine_ycomp,UpLine_ycomp,DownPeakLine_ycomp,UpPeakLine_ycomp = FindData(Nx,Ny,pos_swap_y,HexPositions,HexPosOffset,dx,dy1,dy2,coorddata,padding)
       
        np.savetxt(savedir+'Ycomp_time_'+realtime+'_UpRight.txt',UpLine_ycomp)
        np.savetxt(savedir+'Ycomp_time_'+realtime+'_DownRight.txt',DownLine_ycomp)
        np.savetxt(savedir+'Ycomp_time_'+realtime+'_UpCenter.txt',UpPeakLine_ycomp)
        np.savetxt(savedir+'Ycomp_time_'+realtime+'_DownCenter.txt',DownPeakLine_ycomp)
        #np.savetxt(savedir+'run_'+run+'_Y_time_'+realtime+'.txt',HexData_yUp[:,:,0])
        #np.savetxt(savedir+'run_'+run+'_Z_time_'+realtime+'.txt',HexData_zUp[:,:,0])
        #print('counts:',HexData[:,:,1])
        #plt.imshow(np.transpose(HexData_x[:,:,1]))
        #plt.show()
        #stop

def main(): 
        #savedir = './ProcessedData/' #sys.argv[1]
        #coorddata = np.loadtxt('./atoms-spin-pumping-coords.data',skiprows=1)#sys.argv[2],skiprows=1)
        datalist = np.sort(glob.glob('data/spins-000*.data'))# ['./AllTerms_1000x100p5_50psMuResolution/NegTorque/run_1/atoms-spin-pumping-00000010.data']#sys.argv[3:]
        #print(datalist)
        #task(datalist[1])
        #stop
        
        with ProcessPoolExecutor(25) as executor:
                executor.map(task,datalist)

if __name__ == '__main__':
        main()
