import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
import random

# THIS CODE EXPLORES THE DIMENSIONALITY REDUCTION TECHNIQUES DEVELOPED IN Gao et al., 2016
# FOR COMPETITIVE INTERACTION SCHEMES

# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

reps=1000
SP = 20
EPS=10.**-20 #Threshold of immigration
Max_init =1.5 #Total initial abundance, distributed randomly across species
PRECISION = 0 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

# DEFINING PARAMETERS:
varintra = 1.1
varinter = 1.00001

# Intraspecies parameters: d, g, Aii
meand=0.1
vard=varintra 
meang=1.0 
varg=varintra
meanAii=1.5
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

meanA = 0.0
A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) 
np.fill_diagonal(A,Aii)
print(A)

# What we explore:
meanBmin = 0.0001
meanBmax=0.15
meanBfrac=30


##################################################################################
startclock = timer()

meanBlist=[]

betalist=[]
Biomass_elist=[]
xelist=[]
Bunstlist=[]
xunstlist=[]

Biomass=np.zeros((meanBfrac,reps))
Survivors=np.zeros((meanBfrac,reps))
Sppbiomass=np.zeros((meanBfrac,reps,SP))

for i in range(0,meanBfrac):
    
    meanB= 0.00001 + meanBmin + i*((meanBmax-meanBmin)/(float(meanBfrac))) 

      
    print("i: ", i," of ",meanBfrac)
    
    B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) #cooperation matrix, all interactions uniform between 0 and coop
    meanBlist.append(np.mean(B)) #compute avg interaction strength before we introduce the diagonal terms.
    #beta_e=np.mean(np.matmul(B,B))/np.mean(B)
    np.fill_diagonal(B,Bii)
    
    #### GAO METRICS
    
    beta_e=np.mean(np.matmul(B,B))/np.mean(B)
    
    #threshold happening at
    betacrit = (1/np.mean(g)) * (np.mean(Aii) + np.mean(d) - (2*np.sqrt(np.mean(Aii)*np.mean(d))) ) #PLUS OR MINUS???
    if beta_e>betacrit:
        betalist.append(beta_e)
        Biomass_elist.append(0.0)
        xunstlist.append(0.0)
        Bunstlist.append(0.0)
        xelist.append(0.0)
    else:
        betalist.append(beta_e)
        x_effect = (0.5/beta_e)* (np.mean(Aii) - (beta_e*np.mean(g)) - np.mean(d) + np.sqrt( ((np.mean(Aii) - (beta_e*np.mean(g)) - np.mean(d))**2) - (4*beta_e*np.mean(d)*np.mean(g)) ))
        x_unst = (0.5/beta_e)* (np.mean(Aii) - (beta_e*np.mean(g)) - np.mean(d) - np.sqrt( ((np.mean(Aii) - (beta_e*np.mean(g)) - np.mean(d))**2) - (4*beta_e*np.mean(d)*np.mean(g)) ))
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
             
             x0 = [m*0.8 for m in np.random.random(SP)]
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

fig,((ax1, ax2 , ax3),(ax4, ax5 , ax6)) = plt.subplots(2, 3, figsize=(20,6))


for i in range(0,reps):
    ax1.scatter(meanBlist, Biomass[:,i],marker="o", s=10, color="grey", alpha=0.5)
ax1.plot(meanBlist, Biomass_elist, label='Sum of eff. abundances')
ax1.plot(meanBlist, Bunstlist, label='AE threshold ', linestyle="--")
ax1.set_title('Total biomass')
ax1.legend()
ax1.set(xlabel='avg comp', ylabel='Final Biomass')


for i in range(0,reps):
    ax2.scatter(meanBlist, Survivors[:,i],marker="o", s=10, color="grey", alpha=0.5)
ax2.set_title('Surviving species')
ax2.set(xlabel='avg comp', ylabel='Surviving species')
ax2.legend()

for i in range(0,reps):
    for j in range(SP):
        ax3.scatter(meanBlist, Sppbiomass[:,i,j],marker="o", s=10, color="grey", alpha=0.5)     
ax3.plot(meanBlist, xelist, label='Single-spp eff. ab.')
ax3.plot(meanBlist, xunstlist, label='AE_i threshold ', linestyle="--")
ax3.set_title('Single-spp abundance')
ax3.set(xlabel='avg comp', ylabel='species abundance')
#ax3.legend()

#plt.show()

################## alpha_e plots ###############################

#fig,((ax1, ax2 , ax3)) = plt.subplots(1, 3, figsize=(20,3))


for i in range(0,reps):
    ax4.scatter(betalist, Biomass[:,i],marker="o", s=10, color="grey", alpha=1.0)
ax4.plot(betalist, Biomass_elist, label='Gao weighted average')
ax4.legend()
ax4.set(xlabel='alpha_e', ylabel='Final Biomass')


for i in range(0,reps):
    ax5.scatter(betalist, Survivors[:,i],marker="o", s=10, color="grey", alpha=1.0)
ax5.set(xlabel='alpha_e', ylabel='Surviving species')



for i in range(0,reps):
    for j in range(SP):
        ax6.scatter(betalist, Sppbiomass[:,i,j],marker="o", s=10, color="grey", alpha=1.0) 
ax6.plot(betalist, xelist, label='Gao expected single-spp abundance')
ax6.set(xlabel='alpha_e', ylabel='species abundance')


plt.show()

fig.tight_layout()
nom = "local_mutual_excl.png" 
plt.savefig(nom, format='png')
#plt.show()
plt.close()

