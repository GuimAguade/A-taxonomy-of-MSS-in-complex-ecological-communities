import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
import random


# THIS CODE GENERATES COMMUNITIES WITH RANDOM INITIAL CONDITIONS UNDER INCREASING MUTUALISM
# TO OBSERVE A LOCAL-TO-GLOBAL TRANSITION (FIGURE 2A).
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

reps=500 # Number of initial conditions tested for each system
SP = 50 # Number of species in the initial species pool
EPS=10.**-5 #Threshold of immigration
Max_init =10.0 #Total initial abundance, distributed randomly across species
PRECISION = 2 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

# DEFINING PARAMETERS: 
varintra = 1.1 #intra-species heterogeneity
varinter = 1.1 #interaction strength heterogeneity

# Intraspecies parameters: d, g, Aii
meand=0.1
vard=varintra 
meang=1.0 
varg=varintra
meanAii=0.5
varAii=varintra
meanBii = 0.1
varBii= varintra

# Intra-species arrays:
d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=SP)
g=np.random.lognormal(mean=np.log(meang), sigma=np.log(varg), size=SP)
Aii=np.random.lognormal(mean=np.log(meanAii), sigma=np.log(varAii), size=SP)
Bii=np.random.lognormal(mean=np.log(meanBii), sigma=np.log(varBii), size=SP)

# Set interaction heterogeneity
varA = varinter
varB = varinter  

#No competition:
B=np.zeros((SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
np.fill_diagonal(B,Bii) #we keep the diagonal also random!

# What we explore:
meanAmin = 0.0000001
meanAmax=0.01
meanAfrac=100


##################################################################################
startclock = timer()

meanAlist=[]

Biomass=np.zeros((meanAfrac,reps))
Survivors=np.zeros((meanAfrac,reps))
Sppbiomass=np.zeros((meanAfrac,reps,SP))

for i in range(0,meanAfrac): # For each mean(A) value:
    
    meanA=meanAmin + (i*((meanAmax - meanAmin)/(float(meanAfrac))) )
    
    print("avg A: ", meanA," of ",meanAmax)
    
    A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
    meanAlist.append(np.mean(A)) #compute avg interaction strength before we introduce the diagonal terms.
    np.fill_diagonal(A,Aii)

    for j in range(0,reps): # Perform experiment many times

        # One simulation: 
        def run(
                 S, #Species number
                 tmax=10000, #maximum time
                 EPS=EPS,**kwargs
                     ):
             def eqs(t,x):
                 dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                 dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                 return dx                 
             
             #Generate random initial conditions of different width
             width = np.random.uniform(0,2)
             x0 = [m*width for m in np.random.random(SP)]
             sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
             time=sol.t
             trajectories=sol.y
             return time, trajectories
    
        # End of simulation: write spp trajectories
        time,trajectories = run(SP)
            
        # Count final biomass and surviving spp
        abundances=sum(trajectories[:,-1])
        survivors = len([m for m in trajectories[:,-1] if m>0.01])
        
        for h in range(SP):
            Sppbiomass[i,j,h]=trajectories[h,-1]
        Biomass[i,j]=abundances
        Survivors[i,j]=survivors


endclock = timer()
print("Program runtime", endclock - startclock)


################## COOP PLOTS ###############################

fig,((ax1, ax2 , ax3)) = plt.subplots(1, 3, figsize=(15,4))


for i in range(0,reps):
    ax1.scatter(meanAlist, Biomass[:,i],marker="o", s=20, color="grey", alpha=0.1)
ax1.set_title('Total biomass')
ax1.set(xlabel='avg coop', ylabel='Final Biomass')


for i in range(0,reps):
    ax2.scatter(meanAlist, Survivors[:,i],marker="o", s=20, color="grey", alpha=0.1)
ax2.set_title('Surviving species')
ax2.set(xlabel='avg coop', ylabel='Surviving species')
ax2.set_xlim(-0.0005, meanAmax)

for i in range(0,reps):
    for j in range(SP):
        ax3.scatter(meanAlist, Sppbiomass[:,i,j],marker="o", s=20, color="grey", alpha=0.1)     
ax3.set_title('Single-spp abundance')
ax3.set(xlabel='avg coop', ylabel='species abundance')


#plt.show()

fig.tight_layout()
nom = "F2A.png" 
plt.savefig(nom, format='png')
plt.close()



