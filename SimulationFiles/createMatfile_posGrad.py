import matplotlib.pyplot as plt
import numpy as np

Nx = 80
Start = 20
End = 61
Nprofile = End-Start

alpha = 0.001
Ndamp = Start
Tmax = 3.6
Tmin = 0.0036

J1 = 3.6e-22 
BQ = 0 #1.7033e-22
aniso = 3.923e-23

fig,axs = plt.subplots(2,1,sharex=True)

f = open('ManyMat_PosGradient.mat','w+')


f.write('material:num-materials='+str(Nx)+'\n')

for n in range(1,Nx+1): #VAMPIRE is 1-indexed
    #f.write(f'material['+str(n)+']:material-name='+str(n)+' \n')
    if (n < Start):
        T = Tmin
        f.write(f'material['+str(n)+']:damping-constant='+str(alpha)+'\n')
        f.write(f'material['+str(n)+']:temperature='+str(T)+' \n')
    elif (n > End):
        T = Tmax
        f.write(f'material['+str(n)+']:damping-constant='+str(alpha)+'\n')
        f.write(f'material['+str(n)+']:temperature='+str(T)+' \n')
    
    elif (n >= Start and n <= End):
        T = Tmin+(n-Start)* (Tmax-Tmin)/Nprofile 
        f.write(f'material['+str(n)+']:damping-constant='+str(alpha)+'\n')
        f.write(f'material['+str(n)+']:temperature='+str(T)+' \n')
   
    else: 
        print('dont know what to do with n ',n)

    axs[0].scatter(n,alpha,color='blue')
    axs[1].scatter(n,T,color='black',marker='.',s=10)

    f.write(f'material['+str(n)+']:atomic-spin-moment=3 !muB \n')
    f.write(f'material['+str(n)+']:uniaxial-anisotropy-direction=0,0,1 \n')
    f.write(f'material['+str(n)+']:uniaxial-anisotropy-constant='+str(aniso)+' \n')
    f.write(f'material['+str(n)+']:initial-spin-direction=1,0,0 \n')
    
    #exchange constants; all the non coupled:
    for m in range(1,Nx+1):
        if(m!=n and m!=n-1 and m!=n+1):
            f.write(f'material['+str(n)+']:exchange-matrix['+str(m)+']=0\n')
    #now set couplings
    
    #with itself
    f.write(f'material['+str(n)+']:exchange-matrix['+str(n)+']='+str(J1)+' \n')
    
    #with neighbors left and right:
    if (n>1):
        f.write(f'material['+str(n)+']:exchange-matrix['+str(n-1)+']='+str(J1)+'\n')
    if (n<Nx):    
        f.write(f'material['+str(n)+']:exchange-matrix['+str(n+1)+']='+str(J1)+'\n')
    
    f.write(f'material['+str(n)+']:geometry-file=./geofiles/'+str(n)+'.geo \n \n')


f.close()


axs[1].set_xlabel('n along x')
axs[0].set_ylabel('damping')
axs[1].set_ylabel('temperature (K)')
plt.savefig('TempAndDampingProfile_posGrad.pdf')
plt.show()

