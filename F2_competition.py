import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
import random

# THIS CODE GENERATES COMMUNITIES WITH RANDOM INITIAL CONDITIONS UNDER INCREASING COMPETITION
# TO OBSERVE A LOCAL-TO-MUTUAL EXCLUSION TRANSITION (FIGURE 2B).
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

reps=500 # Number of initial conditions tested for each system
SP = 50 # Number of species in the initial species pool
EPS=10.**-20 #Threshold of immigration
Max_init =1.5 #Total initial abundance, distributed randomly across species
PRECISION = 0 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

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

#Intra-species arrays:
d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=SP)
g=np.random.lognormal(mean=np.log(meang), sigma=np.log(varg), size=SP)
Aii=np.random.lognormal(mean=np.log(meanAii), sigma=np.log(varAii), size=SP)
Bii=np.random.lognormal(mean=np.log(meanBii), sigma=np.log(varBii), size=SP)

# interaction heterogeneity
varA = varinter
varB = varinter  

# No mutualism:
A=np.zeros((SP,SP)) #np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) 
np.fill_diagonal(A,Aii)


# What we explore:
meanBmin = 0.15
meanBmax=0.25
meanBfrac=100


##################################################################################
startclock = timer()

meanBlist=[]
Biomass=np.zeros((meanBfrac,reps))
Survivors=np.zeros((meanBfrac,reps))
Sppbiomass=np.zeros((meanBfrac,reps,SP))

for i in range(0,meanBfrac): # For each mean(B) value:
    
    meanB= 0.00001 + meanBmin + i*((meanBmax-meanBmin)/(float(meanBfrac))) 

      
    print("i: ", i," of ",meanBfrac)
    
    B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
    meanBlist.append(np.mean(B)) #compute avg interaction strength before we introduce the diagonal terms.
    np.fill_diagonal(B,Bii)
    

    for j in range(0,reps): #Perform experiment many times

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
             
             width = np.random.uniform(0,2)
             x0 = [m*width for m in np.random.random(SP)]
             sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
             time=sol.t
             trajectories=sol.y
             return time, trajectories
    
        # End of simulation: write spp trajectories
        time,trajectories = run(SP)
            
        # Count final biomass and surviving spp
        abundances = sum(trajectories[:,-1])
        survivors  = len([m for m in trajectories[:,-1] if m>0.1])
        
        for h in range(SP):
            Sppbiomass[i,j,h]=trajectories[h,-1]
        Biomass[i,j]=abundances
        Survivors[i,j]=survivors


endclock = timer()
print("Program runtime", endclock - startclock)



################## COOP PLOTS ###############################

fig,((ax1, ax2 , ax3)) = plt.subplots(1, 3, figsize=(8,1))


for i in range(0,reps):
    ax1.scatter(meanBlist, Biomass[:,i],marker="o", s=20, color="grey", alpha=0.1)
ax1.set_title('Total biomass')
ax1.set(xlabel='avg comp', ylabel='Final Biomass')


for i in range(0,reps):
    ax2.scatter(meanBlist, Survivors[:,i],marker="o", s=10, color="grey", alpha=0.1)
ax2.set_title('Surviving species')
ax2.set(xlabel='avg comp', ylabel='Surviving species')
#ax2.set_xlim(meanBmin-0.0005, meanBmax)
#ax2.set_ylim(-1,10)

for i in range(0,reps):
    for j in range(SP):
        ax3.scatter(meanBlist, Sppbiomass[:,i,j],marker="o", s=20, color="grey", alpha=0.1)     
ax3.set_title('Single-spp abundance')
ax3.set(xlabel='avg comp', ylabel='species abundance')


#plt.show()

fig.tight_layout()
nom = "F2_competition.png" 
plt.savefig(nom, format='png')
#plt.show()
plt.close()

