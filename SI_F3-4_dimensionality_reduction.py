import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
import random

# THIS CODE EXPLORES THE DIMENSIONALITY REDUCTION TECHNIQUES DEVELOPED IN Gao et al., 2016
# TO TEST THE UNIVERSALITY OF THE GLOBAL BISTABILITY REGIME
# BOTH THE PRESENCE OF LOCAL STATES AND HETEROGENEITY DISMANTLE THE SHARP ALL-OR-NOTHING TRANSITION
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

reps=500
SP = 20
EPS=10.**-5 #Threshold of immigration
Max_init =10.0 #Total initial abundance, distributed randomly across species
PRECISION = 6 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

# DEFINING PARAMETERS:
varintra = 1.1
varinter = 1.1
# Intraspecies parameters: d, g, Aii
meand=0.4
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

#no competition:
B=np.zeros((SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
np.fill_diagonal(B,Bii) #we keep the diagonal also random!

# What we explore:
meanAmax=0.05
meanAfrac=100


##################################################################################
startclock = timer()

meanAlist=[]

alphalist=[]
Biomass_elist=[]
xelist=[]
Bunstlist=[]
xunstlist=[]

Biomass=np.zeros((meanAfrac,reps))
Survivors=np.zeros((meanAfrac,reps))
Sppbiomass=np.zeros((meanAfrac,reps,SP))

for i in range(0,meanAfrac):
    
    if i==0:
        meanA=0.000000001*(meanAmax/(float(meanAfrac)))
    else:
        meanA=i*(meanAmax/(float(meanAfrac))) 

      
    print("avg A: ", meanA," of ",meanAmax)
    
    A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
    meanAlist.append(np.mean(A)) #compute avg interaction strength before we introduce the diagonal terms.
    np.fill_diagonal(A,Aii)
    
    #### GAO METRICS
    
    alpha_e=np.mean(np.matmul(A,A))/np.mean(A)
    
    #threshold happening at
    alphacrit = (np.mean(Bii)*np.mean(g)) + np.mean(d) + (2*np.sqrt( np.mean(Bii)*np.mean(g)*np.mean(d) )) 
    if alpha_e<alphacrit:
        alphalist.append(alpha_e)
        Biomass_elist.append(0.0)
        xunstlist.append(0.0)
        Bunstlist.append(0.0)
        xelist.append(0.0)
    else:
        alphalist.append(alpha_e)
        x_effect = (0.5/np.mean(Bii))* (alpha_e - (np.mean(Bii)*np.mean(g)) - np.mean(d) + np.sqrt( ((alpha_e - (np.mean(Bii)*np.mean(g)) - np.mean(d))**2) - (4*np.mean(Bii)*np.mean(d)*np.mean(g)) ))
        x_unst = (0.5/np.mean(Bii))* (alpha_e - (np.mean(Bii)*np.mean(g)) - np.mean(d) - np.sqrt( ((alpha_e - (np.mean(Bii)*np.mean(g)) - np.mean(d))**2) - (4*np.mean(Bii)*np.mean(d)*np.mean(g)) ))
        Biomass_elist.append( SP* x_effect)
        xunstlist.append(x_unst)
        Bunstlist.append(SP*x_unst)
        xelist.append(x_effect)
    
    ######
    
    
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
                 dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                 return dx                 
             
             #Generate random initial conditions, but summing Xinit:
             x0 = [m for m in np.random.random(SP)]
             
             #Uniform divided (same as Dirichlet in the end?)
             #valors = [np.random.random() for i in range(SP)]
             #suma = sum(valors)
             #x0 = [ kappa*Xinit/float(suma) for kappa in valors ]
             
             sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
             time=sol.t
             trajectories=sol.y
             return time, trajectories
    
        # End of simulation: write spp trajectories
        time,trajectories = run(SP)
            
        # Count final biomass and surviving spp
        abundances=sum(trajectories[:,-1])
        survivors = len([m for m in trajectories[:,-1] if m>EPS])
        
        for h in range(SP):
            Sppbiomass[i,j,h]=trajectories[h,-1]
        Biomass[i,j]=abundances
        Survivors[i,j]=survivors


endclock = timer()
print("Program runtime", endclock - startclock)


################## COOP PLOTS ###############################

fig,((ax1, ax2 , ax3),(ax4, ax5 , ax6)) = plt.subplots(2, 3, figsize=(20,6))


for i in range(0,reps):
    ax1.scatter(meanAlist, Biomass[:,i],marker="o", s=10, color="grey", alpha=0.7)
ax1.plot(meanAlist, Biomass_elist, label='Sum of eff. abundances')
ax1.plot(meanAlist, Bunstlist, label='AE threshold ', linestyle="--")
ax1.set_title('Total biomass')
ax1.legend()
ax1.set(xlabel='avg coop', ylabel='Final Biomass')


for i in range(0,reps):
    ax2.scatter(meanAlist, Survivors[:,i],marker="o", s=10, color="grey", alpha=0.7)
ax2.set_title('Surviving species')
ax2.legend()
ax2.set(xlabel='avg coop', ylabel='Surviving species')


for i in range(0,reps):
    for j in range(SP):
        ax3.scatter(meanAlist, Sppbiomass[:,i,j],marker="o", s=10, color="grey", alpha=0.7)     
ax3.plot(meanAlist, xelist, label='Single-spp eff. abundances')
ax3.plot(meanAlist, xunstlist, label='AE_i threshold ', linestyle="--")
ax3.set_title('Single-spp abundance')
ax3.legend()
ax3.set(xlabel='avg coop', ylabel='species abundance')


#plt.show()

################## alpha_e plots ###############################

#fig,((ax1, ax2 , ax3)) = plt.subplots(1, 3, figsize=(20,3))


for i in range(0,reps):
    ax4.scatter(alphalist, Biomass[:,i],marker="o", s=10, color="grey", alpha=1.0)
ax4.plot(alphalist, Biomass_elist, label='Gao weighted average')
ax4.legend()
ax4.set(xlabel='alpha_e', ylabel='Final Biomass')


for i in range(0,reps):
    ax5.scatter(alphalist, Survivors[:,i],marker="o", s=10, color="grey", alpha=1.0)
ax5.set(xlabel='alpha_e', ylabel='Surviving species')



for i in range(0,reps):
    for j in range(SP):
        ax6.scatter(alphalist, Sppbiomass[:,i,j],marker="o", s=10, color="grey", alpha=1.0) 
ax6.plot(alphalist, xelist, label='Gao expected single-spp abundance')
ax6.set(xlabel='alpha_e', ylabel='species abundance')



fig.tight_layout()
nom = "local_global.png" 
plt.savefig(nom, format='png')
#plt.show()
plt.close()

