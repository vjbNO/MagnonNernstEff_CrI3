import numpy as np

names = ['PosXgrad','NegYgrad','PosYgrad']
for n in names:
    for run in range(5):
        d = np.loadtxt('../'+n+'/Transport/run_'+str(run+1)+'/ProcessedData/TimeAvMu_run_0_xcomp.txt')
        if (run==0):
            alldata = np.zeros((5,d.shape[0],d.shape[1]))
            alldata[run,:,:]=d
        else:
            alldata[run,:,:]=d
    savedata = np.mean(alldata,axis=0)
    np.savetxt('Data/'+n+'_TimeAvMu.txt',savedata)
