N = 80 #number of temp steps along the x direction #100 is max set by VAMPIRE

m=1
for n in range(N): 
    f = open('geofiles/'+str(m)+'.geo','w+')
    f.write('4 \n')
    f.write(f'{float((n)/N):.5f}'+' '+f'0.0 \n')
    f.write(f'{float((n+1)/N):.5f}'+' '+f'0.0 \n')
    f.write(f'{float((n+1)/N):.5f}'+' '+f'1.0 \n')
    f.write(f'{float((n)/N):.5f}'+' '+f'1.0 \n')
    f.close()
    m+=1


