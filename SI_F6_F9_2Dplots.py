import numpy as np
import matplotlib.pyplot as plt

#THIS CODE PLOTS 3 VECTOR FLOWS FOR A 2 SPECIES SYSTEM UNDER COOPERATION, COMPETITION OR BOTH
# G AGUADÉ-GORGORIÓ

# DEFINING PARAMETERS:

choose=2
#COOP=1, COMP=2, BOTH=3

SP=2

# Intraspecies parameters: d, g, Aii, Bii
var = 1.1

meand=0.1
vard=var 

meang=1.0  
varg=var

meanAii=0.5
varAii=var

meanBii = 0.1
varBii= var

#Intra-species arrays:
d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=SP)
g=np.random.lognormal(mean=np.log(meang), sigma=np.log(varg), size=SP)
Aii=np.random.lognormal(mean=np.log(meanAii), sigma=np.log(varAii), size=SP)
Bii=np.random.lognormal(mean=np.log(meanBii), sigma=np.log(varBii), size=SP)

#interaction matrices to play with:
varinter = 1.01
varA = varinter
varB = varinter

################ STREAMS

#INDEPENDENT################################################################################
meanA = 0.1
meanB = 0.0000000001
A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #cooperation matrix, all interactions uniform
np.fill_diagonal(A,Aii) 
B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) #cooperation matrix, all interactions uniform
np.fill_diagonal(B,Bii) 

w = 4.0
y,x = np.mgrid[-0:w:100j, -0:w:100j]

if choose==1:
    U = x* ( (A[0,0]*x/(g[0]+x)) + (A[0,1]*y/(g[1]+y)) - d[0] - (B[0,0]*x) )
    V = y* ( (A[1,0]*x/(g[0]+x)) + (A[1,1]*y/(g[1]+y)) - d[1] - (B[1,1]*y) )    
elif choose==2:
    U = x* ( (A[0,0]*x/(g[0]+x)) - d[0] - (B[0,0]*x) - (B[0,1]*y) )
    V = y* ( (A[1,1]*y/(g[1]+y)) - d[1] - (B[1,1]*y) - (B[1,0]*x) )  
else:
    U = x* ( (A[0,0]*x/(g[0]+x)) + (A[0,1]*(y/g[1]+y)) - d[0] - (B[0,0]*x) - (B[0,1]*y) )
    V = y* ( (A[1,0]*x/(g[0]+x)) + (A[1,1]*y/(g[1]+y)) - d[1] - (B[1,1]*y) - (B[1,0]*x) )  

#LOW INT################################################################################

yy,xx = np.mgrid[-0:w:100j, -0:w:100j]

meanA = 0.01
meanB = 0.04
A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #cooperation matrix, all interactions uniform
np.fill_diagonal(A,Aii) 
B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) #cooperation matrix, all interactions uniform
np.fill_diagonal(B,Bii) 

if choose==1:
    UU = xx* ( (A[0,0]*xx/(g[0]+xx)) + (A[0,1]*yy/(g[1]+yy)) - d[0] - (B[0,0]*xx) )
    VV = yy* ( (A[1,0]*xx/(g[0]+xx)) + (A[1,1]*yy/(g[1]+yy)) - d[1] - (B[1,1]*yy) )    
elif choose==2:
    UU = xx* ( (A[0,0]*xx/(g[0]+xx)) - d[0] - (B[0,0]*xx) - (B[0,1]*yy) )
    VV = yy* ( (A[1,1]*yy/(g[1]+yy)) - d[1] - (B[1,1]*yy) - (B[1,0]*xx) )  
else:
    UU = xx* ( (A[0,0]*xx/(g[0]+xx)) + (A[0,1]*yy/(g[1]+yy)) - d[0] - (B[0,0]*xx) - (B[0,1]*yy) )
    VV = yy* ( (A[1,0]*xx/(g[0]+xx)) + (A[1,1]*yy/(g[1]+yy)) - d[1] - (B[1,1]*yy) - (B[1,0]*xx) ) 

#HIGH INT################################################################################
yyy,xxx = np.mgrid[-0:w:100j, -0:w:100j]
meanA = 0.05
meanB = 10.5
A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #cooperation matrix, all interactions uniform
np.fill_diagonal(A,Aii) 
B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) #cooperation matrix, all interactions uniform
np.fill_diagonal(B,Bii) 

if choose==1:
    UUU = xxx* ( (A[0,0]*xxx/(g[0]+xxx)) + (A[0,1]*yyy/(g[1]+yyy)) - d[0] - (B[0,0]*xxx) )
    VVV = yyy* ( (A[1,0]*xxx/(g[0]+xxx)) + (A[1,1]*yyy/(g[1]+yyy)) - d[1] - (B[1,1]*yyy) )    
elif choose==2:
    UUU = xxx* ( (A[0,0]*xxx/(g[0]+xxx)) - d[0] - (B[0,0]*xxx) - (B[0,1]*yyy) )
    VVV = yyy* ( (A[1,1]*yyy/(g[1]+yyy)) - d[1] - (B[1,1]*yyy) - (B[1,0]*xxx) )  
else:
    UUU = xxx* ( (A[0,0]*xxx/(g[0]+xxx)) + (A[0,1]*yyy/(g[1]+yyy)) - d[0] - (B[0,0]*xxx) - (B[0,1]*yyy) )
    VVV = yyy* ( (A[1,0]*xxx/(g[0]+xxx)) + (A[1,1]*yyy/(g[1]+yyy)) - d[1] - (B[1,1]*yyy) - (B[1,0]*xxx) )  
    


fig,(ax1, ax2 , ax3) = plt.subplots(1, 3, figsize=(20,4))

#  Varying density along a streamline
ax1.streamplot(x, y, U, V, density=[1.0, 1.0], color=(.55,.70,.73))
ax2.streamplot(xx, yy, UU, VV, density=[1.0, 1.0], color=(.55,.70,.73))
ax3.streamplot(xxx, yyy, UUU, VVV, density=[1.0, 1.0], color=(.55,.70,.73))

plt.show()

