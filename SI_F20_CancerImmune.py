import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
    
# CODE TO EXPLORE THE STATISTICS OF STATES UNDER COMP + COOP FOR THE CANCER-IMMUNE MODEL (Garay and Lefever, 1978)
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanAstart=-0.002
meanAmax=0.008

varAstart=0.0
varAmax=0.05

frac=10

reps=50
S = 50

temps1 = 500
temps2 = 50

EPS=10.**-20 #Threshold of immigration

SI_threshold = 0.05 # if two species differ by more than this, they are not the same state

# DEFINING PARAMETERS:
varintra = 1.1
varinter = 1.00000001

# Intraspecies parameters: d, mu

meand = 0.25
vard = varintra
d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=S)

meanmu = 1.4
varmu = varintra
mu=np.random.lognormal(mean=np.log(meanmu), sigma=np.log(varmu), size=S)

g = 1

I = np.identity(S)


##################################################################################

# Interaction range:

meanAlist = []
varAlist = []



Numstates = np.zeros((frac, frac)) # Number of different observed states: len(FreqAB)
Avg_Surv = np.zeros((frac, frac))   # Average surviving species: weighted (FreqAB) avg of Surviving list.
Frac_cycles = np.zeros((frac, frac)) # number of runs that end in non-stable behavior

z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    varA = varAstart + (z*(varAmax-varAstart)/float(frac)) + 0.000001
    varAlist.append(varA)
    
    i=0
    while i<frac:
        
        meanA= meanAstart + (i*(meanAmax-meanAstart)/float(frac))

        A=np.random.normal(meanA, varA, size=(S,S)) #COOP matrix, all interactions uniform between 0 and coop    
        if z==0:
            meanAlist.append(np.mean(A))
        np.fill_diagonal(A,-d)
    
        StatAB = []     # For given A and B: A matrix with the different states observed
        FreqAB = []     # For given A and B: An array, how often each state is reached
        SurvAB = []     # For given A and B: An array, the number of species surviving in each state
        num_unst = 0
        
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) - ( x*mu / (g + np.dot(I,x) ) ) 
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
                #Generate random initial conditions, but summing Xinit:
                x0 = [v for v in np.random.random(S)]           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            time,trajectories = run(S)
            
            finalstate = [m for m in trajectories[:,-1]] 
            #cleanstate = list(np.around(np.array(finalstate),PRECISION)) #APPROXIMATION 1: NO DECIMALS TO AVOID OVER-PRECISE STATE DIFFERENTIATION
            alivestate = [m for m in finalstate if m>0.01] #APPROXIMATION 2 (SIMILAR): ONLY CONSIDER SPECIES THAT ARE BEYOND 1.0 ABUNDANCE (TO AVOID EFFECTS OF MIGRATION?)
 
            #################### CYCLE CHECK: RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT #########################
            
            def run(S,tmax=temps2,EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) - ( x*mu / (g + np.dot(I,x) ) ) 
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
                #Solve the system of equations:
                x0 = finalstate           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            timeplus,trajectoriesplus = run(S)
            finalstateplus = [m for m in trajectoriesplus[:,-1]] 
            
            ###########################################################################################################
            
            # DO WE SEE A NEW STATE, AN ALREADY-SEEN STATE, OR CYCLING/CHAOTIC BEHAVIOUR?
            
            # 1: SEE IF STATE IS STABLE IN TIME
            
            diff = 0.0
            for spp in range(S):
                if abs ( finalstate[spp] - finalstateplus[spp] ) > SI_threshold:
                    diff += 1
                    break
                            
            if diff > 0: # if two species have been deemed different, this state is not stable!
                num_unst+=1
                SurvAB.append(len(alivestate)) #We do not measure data from non-stable results
          
          # 2: IF STABLE, MEASURE THE SIMILARITY INDEX WITH ALL PREVIOUS STATES TO SEE IF NEW.
            else:
                state = 0
                
                while state<len(StatAB): # compare all states with new "cleanstate":
                    
                    SI = 0.0
                    for spp in range(S):
                        if ( abs ( StatAB[state][spp] - finalstate[spp] ) ) > SI_threshold:
                            SI+=1
                            break
                            
                    if SI == 0.0: # this is the same state, no species are different!
                        FreqAB[state]+=1 #Increase frequency by one
                        SurvAB.append(len(alivestate)) #count it as a (repeated) diversity
                        state += 2*reps # no need to compare with other states, we already found ours!
                    
                    else: # this is not the same state, compare it with another one
                        state+=1
                
                if state < 2*reps : #We have reached the end of the loop without finding the same state: the state is new!                     
                    StatAB.append(finalstate)
                    SurvAB.append(len(alivestate))
                    FreqAB.append(1)            
                
                


        #End of REPS: we have explored the same parameters many times, avg what we've seen
        Numstates[z,i] = len(FreqAB) 
        
        Frac_cycles[z,i] = num_unst / float(reps)
        
        #Heatmap 2: What is the average number of surviving spp?
        real_surv = []
        for i1 in range(len(SurvAB)):
            for i2 in range(len(FreqAB)):
                real_surv.append(SurvAB[i1]) # We are creating the real list of events: how many times each is observed (like a weighted avg)
        Avg_Surv[z,i] = np.mean(real_surv)

        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)

################## PLOTS ###############################

#Figure 1: number of states

fig,(ax1,ax2,ax3)= plt.subplots(1,3,figsize=(24,8))

im1 = ax1.imshow(Numstates)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in varAlist]
# Show all ticks and label them with the respective list entries
ax1.set_xticks(np.arange(len(meanAlist)))
ax1.set_yticks(np.arange(len(varAlist)))
ax1.set_xticklabels(stra)
ax1.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax1.invert_yaxis()
ax1.invert_xaxis()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
ax1.set_ylabel('avg competition')
ax1.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax1.set_title("Number of observed states")




im2 = ax2.imshow(Frac_cycles, cmap="cividis")
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in varAlist]
# Show all ticks and label them with the respective list entries
ax2.set_xticks(np.arange(len(meanAlist)))
ax2.set_yticks(np.arange(len(varAlist)))
ax2.set_xticklabels(stra)
ax2.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax2.invert_yaxis()
ax2.invert_xaxis()
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.set_ylabel('avg competition')
ax2.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax2.set_title("Fraction of cycling simulations/outgrowth")






im3 = ax3.imshow(Avg_Surv)
#ax1.plot(meanAlist, Bcritplus, label="bistab. threshold", linewidth=2, color="white")
#ax1.plot(meanAlist, Bcritminus, label="bistab. threshold", linewidth=2, color="white", linestyle="--")
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in varAlist]
# Show all ticks and label them with the respective list entries
ax3.set_xticks(np.arange(len(meanAlist)))
ax3.set_yticks(np.arange(len(varAlist)))
ax3.set_xticklabels(stra)
ax3.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax3.invert_yaxis()
ax3.invert_xaxis()
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')
ax3.set_ylabel('avg competition')
ax3.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax3.set_title("Avg surviving species")


fig.tight_layout()
nom = "GL_005_g1_GLOBAL.png" 
plt.savefig(nom, format='png')
plt.close()

