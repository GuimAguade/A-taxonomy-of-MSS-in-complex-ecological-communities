import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
import random

# THIS CODE GENERATES COMMUNITIES WITH RANDOM INITIAL CONDITIONS UNDER INCREASING INTERACTION MATRIX HETEROGENEITY
# TO OBSERVE A LOCAL-TO-CLIQUES TRANSITION (FIGURE 2C).
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

reps=1000 # Number of initial conditions tested for each system
SP = 50 # Number of species in the initial species pool
EPS=10.**-20 #Threshold of immigration

# DEFINING PARAMETERS:
varintra = 1.1 #intra-species heterogeneity

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


# What we explore: Increasing heterogeneity (logarithm of the Gaussian)
varmin = 1.0000001
varmax = 1.3
meanAfrac=100

# Admit weak (and balanced, see SI II.E.1) interaction strengths that will become more and more heterogeneous
meanA = 0.1
meanB = 0.025

##################################################################################
startclock = timer()

varlist=[]


Biomass=np.zeros((meanAfrac,reps))
Survivors=np.zeros((meanAfrac,reps))
Sppbiomass=np.zeros((meanAfrac,reps,SP))

for i in range(0,meanAfrac):
    
    var=varmin + (i*((varmax - varmin)/(float(meanAfrac))) )
    varlist.append(var)
    
    print("i: ", i," of ",meanAfrac)
    
    varA = var
    varB = var 

    A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
    np.fill_diagonal(A,Aii)
    
    B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
    np.fill_diagonal(B,Bii)
   
    
    for j in range(0,reps): #Perform experiment many times
        #print(j," of ",reps)
        # One simulation: 
        def run(
                 S, #Species number
                 tmax=10000, #maximum time
                 EPS=EPS,**kwargs
                     ):
             def eqs(t,x):
                 dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                 #dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                 return dx                 
             
             # EXAMPLE OF A DIRICHLET APPROACH TO GENERATING RANDOM INITIAL CONDITIONS (SEE SI I.C.1) 
             # Generate random initial conditions, but summing X:
             vals = np.random.default_rng().dirichlet(np.ones(SP), size=None) #Dirichlet distr: generate an array of SP random floats that sum 1
             width = np.random.uniform(0,1*SP)
             k_nums = [v*width for v in vals]  #multiply each by N so that they sum N
             x0 = k_nums 
                        
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

varlistlog = []

for i in range(len(varlist)):
    varlistlog.append(np.log(varlist[i]))
    

################## COOP PLOTS ###############################

fig,((ax1, ax2 , ax3)) = plt.subplots(1, 3, figsize=(15,4))


for i in range(0,reps):
    ax1.scatter(varlist, Biomass[:,i],marker="o", s=20, color="grey", alpha=0.1)
ax1.set_title('Total biomass')
ax1.set(xlabel='std dev', ylabel='Final Biomass')


for i in range(0,reps):
    ax2.scatter(varlist, Survivors[:,i],marker="o", s=20, color="grey", alpha=0.1)
ax2.set_title('Surviving species')
ax2.set(xlabel='std dev', ylabel='Surviving species')
#ax2.set_xlim(-0.0005, varmax)

for i in range(0,reps):
    ax3.scatter(varlistlog, Survivors[:,i],marker="o", s=20, color="grey", alpha=0.1)
ax3.set_title('Surviving species')
ax3.set(xlabel='std dev', ylabel='Surviving species')


#plt.show()

fig.tight_layout()
nom = "cliques10.png" 
plt.savefig(nom, format='png')
#plt.show()
plt.close()



