import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
    
# CODE TO EXPLORE THE STATISTICS OF STATES UNDER COMP + COOP / 17-nov-22

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanBstart = 0.001
meanBmax=0.09

varBstart=1.000000001
varBmax=1.5

temps1 = 1000
temps2 = 50

meanBfrac=50
meanAfrac = meanBfrac

reps=100
SP = 50

EPS=10.**-20 #Threshold of immigration
PRECISION = 2 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES
SI_threshold = SP * 0.01

# DEFINING PARAMETERS:
varintra = 1.1
varinter = 1.00001

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

meanA = 0.2

A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #COOP matrix, all interactions uniform between 0 and coop    
np.fill_diagonal(A,Aii)

##################################################################################
startclock = timer()

# Interaction range:

varBlist = []

varBfrac=meanBfrac
meanBlist = []

Numstates = np.zeros((meanAfrac, meanBfrac)) # Number of different observed states: len(FreqAB)
Avg_Surv = np.zeros((meanAfrac, meanBfrac))   # Average surviving species: weighted (FreqAB) avg of Surviving list.
Frac_cycles = np.zeros((meanAfrac, meanBfrac)) # number of runs that end in non-stable behavior

for z in range(0,varBfrac):
    
    varB= varBstart + (z*(varBmax-varBstart)/float(varBfrac))
    print("varB: ", varB," of ",varBmax)
    varBlist.append(np.log(varB))
    
    for i in range(0,meanBfrac):
        
        meanB= meanBstart + (i*(meanBmax-meanBstart)/float(meanBfrac))

        B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) 
        if z==0:
            meanBlist.append(np.mean(B))
        np.fill_diagonal(B,Bii)
    
        D=np.zeros(SP+1) #Species distribution: how many times do we see "n" species? (from 0 to S)
        
        StatAB = []     # For given A and B: A matrix with the different states observed
        FreqAB = []     # For given A and B: An array, how often each state is reached
        SurvAB = []     # For given A and B: An array, the number of species surviving in each state
        num_unst = 0
        
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
                #Generate random initial conditions, but summing Xinit:
                x0 = [v for v in np.random.random(SP)]           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            time,trajectories = run(SP)
            
            finalstate = [m for m in trajectories[:,-1]] 
            cleanstate = list(np.around(np.array(finalstate),PRECISION)) #APPROXIMATION 1: NO DECIMALS TO AVOID OVER-PRECISE STATE DIFFERENTIATION
            alivestate = [m for m in finalstate if m>0.5] #APPROXIMATION 2 (SIMILAR): ONLY CONSIDER SPECIES THAT ARE BEYOND 1.0 ABUNDANCE (TO AVOID EFFECTS OF MIGRATION?)
 
            #################### CYCLE CHECK: RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT #########################
            
            def run(S,tmax=temps2,EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
                #Solve the system of equations:
                x0 = finalstate           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            timeplus,trajectoriesplus = run(SP)
            finalstateplus = [m for m in trajectoriesplus[:,-1]] 
            
            ###########################################################################################################
            
            # DO WE SEE A NEW STATE, AN ALREADY-SEEN STATE, OR CYCLING/CHAOTIC BEHAVIOUR?
            
            # 1: SEE IF STATE IS STABLE IN TIME
            
            diff = 0.0
            for spp in range(SP):
                diff += 2 * abs ( finalstate[spp] - finalstateplus[spp] ) / (finalstate[spp]+finalstateplus[spp]+0.01)
                            
            if diff > 1*SI_threshold: # if Similarity index is higher than threshold, the state is not stable
                num_unst+=1
                SurvAB.append(len(alivestate)) #We do not measure data from non-stable results
          
          # 2: IF STABLE, MEASURE THE SIMILARITY INDEX WITH ALL PREVIOUS STATES TO SEE IF NEW.
            else:
                state = 0
                SI_perturb = 0.0
                while state<len(StatAB): # compare all states with new "cleanstate":
                    SI = 0.0
                    for spp in range(SP):
                        SI += 2* abs ( StatAB[state][spp] - cleanstate[spp] ) / (StatAB[state][spp] + cleanstate[spp] + 0.01) #Add the Euclidean distance
                    
                    # IF SI IS SMALLER THAN THRESHOLD, THIS IS THE SAME STATE
                    if SI<SI_threshold:
                        FreqAB[state]+=1 #Increase frequency by one
                        SurvAB.append(len(alivestate)) #count it as a (repeated) diversity
                        state = len(StatAB)+10 # Stop the search
                    #ELSE, KEEP ON SEARCHING
                    else:
                        state+=1
                
                if state==len(StatAB): #If we have reached the end of the loop, the state is new.                     
                    StatAB.append(cleanstate)
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
        
        
endclock = timer()
print("Program runtime", endclock - startclock)

################## PLOTS ###############################

#Figure 1: number of states

fig,(ax1,ax2,ax3)= plt.subplots(1,3,figsize=(24,8))


im1 = ax1.imshow(Numstates)
#ax1.plot(meanAlist, Bcritplus, label="bistab. threshold", linewidth=2, color="white")
#ax1.plot(meanAlist, Bcritminus, label="bistab. threshold", linewidth=2, color="white", linestyle="--")
stra = ["{:.3f}".format(i) for i in meanBlist]
strb = ["{:.3f}".format(i) for i in varBlist]
# Show all ticks and label them with the respective list entries
ax1.set_xticks(np.arange(len(meanBlist)))
ax1.set_yticks(np.arange(len(varBlist)))
ax1.set_xticklabels(stra)
ax1.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax1.invert_yaxis()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
ax1.set_ylabel('std dev competition')
ax1.set_xlabel('avg competition')
plt.gca().invert_yaxis()
ax1.set_title("Number of observed states")


im2 = ax2.imshow(Frac_cycles, cmap="cividis")
stra = ["{:.3f}".format(i) for i in meanBlist]
strb = ["{:.3f}".format(i) for i in varBlist]
# Show all ticks and label them with the respective list entries
ax2.set_xticks(np.arange(len(meanBlist)))
ax2.set_yticks(np.arange(len(varBlist)))
ax2.set_xticklabels(stra)
ax2.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax2.invert_yaxis()
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.set_ylabel('std dev competition')
ax2.set_xlabel('avg competition')
plt.gca().invert_yaxis()
ax2.set_title("Fraction of cycling simulations")

im3 = ax3.imshow(Avg_Surv)
#ax1.plot(meanAlist, Bcritplus, label="bistab. threshold", linewidth=2, color="white")
#ax1.plot(meanAlist, Bcritminus, label="bistab. threshold", linewidth=2, color="white", linestyle="--")
stra = ["{:.3f}".format(i) for i in meanBlist]
strb = ["{:.3f}".format(i) for i in varBlist]
# Show all ticks and label them with the respective list entries
ax3.set_xticks(np.arange(len(meanBlist)))
ax3.set_yticks(np.arange(len(varBlist)))
ax3.set_xticklabels(stra)
ax3.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax3.invert_yaxis()
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')
ax3.set_ylabel('std dev competition')
ax3.set_xlabel('avg competition')
plt.gca().invert_yaxis()
ax3.set_title("Avg surviving species")


fig.tight_layout()
nom = "Bunin_inzoom.png" 
plt.savefig(nom, format='png')
plt.close()


