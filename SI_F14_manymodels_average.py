import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
    
# THIS CODE EXPLORES THE NUMBER OF STATES AND FRACTION OF UNSTABLE SIMULATIONS FOR THE ALLEE EFFECT MODEL
# CONSIDERING THAT EACH SQUARE (EACH A,B PAIR) CONTAINS 50 MODELS INSTEAD OF ONE,
# ALLOWING FOR AN AVERAGE COMPARISON OF THE FRACTION OF UNSTABLE STATES (EMERGENCE OF CLIQUES)
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanAmax=0.5
meanBmax=0.15
meanAfrac=100

models = 50 # For each A and B, how many different models (random matrices generated) do we test and avg?

reps=100
SP = 50

EPS=10.**-20 #Threshold of immigration
Xinit = SP * 0.1 #Total initial abundance, distributed randomly across species
PRECISION = 5 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES
thresh = 5 # biomass threshold beyond which states are different (cycling warning)

# DEFINING PARAMETERS:
varintra = 1.1
varinter = 1.5

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

##################################################################################
startclock = timer()

# Interaction range:

meanAlist = []

meanBfrac=meanAfrac
meanBlist = []

Bcritplus = []
Bcritminus = []

mod_Numstates = np.zeros((meanAfrac, meanBfrac, models)) # Number of different observed states: len(FreqAB)
mod_Numcycles = np.zeros((meanAfrac, meanBfrac, models)) # Number of different observed cycles
mod_Avg_Surv = np.zeros((meanAfrac, meanBfrac, models))   # Average surviving species: weighted (FreqAB) avg of Surviving list.
mod_Avg_Biom = np.zeros((meanAfrac, meanBfrac, models))   # Average biomass: weighteed avg of Biomassess list.
mod_Var_Surv = np.zeros((meanAfrac, meanBfrac, models))   # Variance of the surviving species number
mod_Var_Biom = np.zeros((meanAfrac, meanBfrac, models))   # Variance of the biomassess.
mod_Avg_biom = np.zeros((meanAfrac, meanBfrac, models))   # Average single-species biomass: weighteed avg of biomassess list.
mod_Var_biom = np.zeros((meanAfrac, meanBfrac, models))   # variance of single-species biomass: weighteed avg of biomassess list.
mod_Frac_cycles = np.zeros((meanAfrac, meanBfrac, models)) 

for mod in range(models):
    print("model:", mod+1, " of ", models)
    for z in range(0,meanBfrac):
        meanB= z*meanBmax/float(meanBfrac)+0.0001
        
        B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) 
        if mod==0:
            meanBlist.append(np.mean(B))
        np.fill_diagonal(B,Bii) #we keep the diagonal also random!
    
        meanA=0.0000001
        for i in range(0,meanAfrac):
            meanA= i*meanAmax/float(meanAfrac)+0.0001

            A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP)) #COOP matrix, all interactions uniform between 0 and coop    
            if mod==0 and z==0:
                meanAlist.append(np.mean(A))
                      
            np.fill_diagonal(A,Aii)
    
            D=np.zeros(SP+1) #Species distribution: how many times do we see "n" species? (from 0 to S)
        
            StatAB = []     # For given A and B: A matrix with the different states observed    
            FreqAB = []     # For given A and B: An array, how often each state is reached
            BiomAB = []     # For given A and B: An array, the biomass of each state
            SurvAB = []     # For given A and B: An array, the number of species surviving in each state
            avg_biomAB = []
            var_biomAB = []
            num_cycles = []
            cycle_list = []
            
            for j in range(0,reps): #Perform experiment many times
                #print(j," of ",reps)
                # One simulation: 
                def run(
                         S, #Species number
                         tmax=500, #maximum time
                         EPS=EPS,**kwargs
                             ):

                     def eqs(t,x):
                         dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                         dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                         return dx
                     #Generate random initial conditions, but summing Xinit:
                     x0 = [v*1.0 for v in np.random.random(SP)]           
                     sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                     time=sol.t
                     trajectories=sol.y
                     return time, trajectories
    
                # Simulation: write spp trajectories
                time,trajectories = run(SP)
            
                finalstate = [m for m in trajectories[:,-1]] 
                cleanstate = list(np.around(np.array(finalstate),PRECISION)) #APPROXIMATION 1: NO DECIMALS TO AVOID OVER-PRECISE STATE DIFFERENTIATION
                alivestate = [m for m in finalstate if m>0.9] #APPROXIMATION 2 (SIMILAR): ONLY CONSIDER SPECIES THAT ARE BEYOND 1.0 ABUNDANCE (TO AVOID EFFECTS OF MIGRATION?)
 
                #################### CYCLE CHECK: RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT #########################
                def run(
                         S, #Species number
                         tmax=100, #maximum time
                         EPS=EPS,**kwargs
                             ):

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
            
                ###########################################################################################################
            
                finalstateplus = [m for m in trajectoriesplus[:,-1]] 
                cleanstateplus = list(np.around(np.array(finalstate),PRECISION)) #APPROXIMATION 1: NO DECIMALS TO AVOID OVER-PRECISE STATE DIFFERENTIATION
                alivestateplus = [m for m in finalstate if m>0.9] #APPROXIMATION 2 (SIMILAR): ONLY CONSIDER SPECIES THAT ARE BEYOND 1.0 ABUNDANCE (TO AVOID EFFECTS OF MIGRATION?)
            
                ###########################################################################################################
            
                # DO WE SEE A NEW STATE, AN ALREADY-SEEN STATE, OR CYCLING/CHAOTIC BEHAVIOUR?
            
                diff = abs( sum(finalstate) - sum(finalstateplus) )
            
                if diff > thresh:
                    num_cycles.append(1)
                    
                    # CYCLE METRICS:
                    biomass_trajectory = []
                    for tem in range(len(timeplus)):
                        biomass_trajectory.append(sum(trajectoriesplus[:,tem])) #an array with total abundances.
                    
                    # the cycle index is the distance between max and min total biomass
                    cycle_index = np.around( max(biomass_trajectory)-min(biomass_trajectory), 1)
                
                    if cycle_index not in cycle_list:
                        cycle_list.append(cycle_index) #this will tell us how many different cycles
                        BiomAB.append(sum(alivestate))
                        SurvAB.append(len(alivestate))
                    
                    
                elif cleanstate not in StatAB: # See if the state is new (COSTLY!)
                    StatAB.append(cleanstate)
                    #print(cleanstate)
                    BiomAB.append(sum(alivestate))
                    SurvAB.append(len(alivestate))
                    FreqAB.append(1)
                    if len(cleanstate)>0:
                        avg_biomAB.append(np.mean(cleanstate)) # the average single-species abundance
                        var_biomAB.append(np.var(cleanstate))  # the variance in single-species abundances: how different species are in one state?
                    else:   
                        avg_biomAB.append(0) # the average single-species abundance
                        var_biomAB.append(0)  # the variance in single-species abundances: how different species are in one state?
                else: #if all of the above fails, it means the state has already been observed
                    #Find position of state in list:
                    index = StatAB.index(cleanstate) #COST????
                    #Increase frequency by one
                    FreqAB[index]+=1
    

            #End of REPS: we have explored the same parameters many times, avg what we've seen
            #print("For A:", meanA, "and B:", meanB)
            #print("Observed states are:")
            #print(StatAB)
            #Heatmap 1: How many states in this A,B region? We should balance with the number of REPS performed?!
            mod_Numstates[z,i] = len(FreqAB)
            mod_Numcycles[z,i] = len(cycle_list) 
            mod_Frac_cycles[z,i,mod] = len(num_cycles) / float(reps)
        
            #Heatmap 2: What is the average number of surviving spp?
            real_surv = []
            for i1 in range(len(SurvAB)):
                for i2 in range(len(FreqAB)):
                    real_surv.append(SurvAB[i1]) # We are creating the real list of events: how many times each is observed (like a weighted avg)
            mod_Avg_Surv[z,i,mod] = np.mean(real_surv)

            #Heatmap 3: What is their biomass?
            real_biom = []
            for i1 in range(len(BiomAB)):
                for i2 in range(len(FreqAB)):
                    real_biom.append(BiomAB[i1]) # We are creating the real list of events: how many times each is observed (like a weighted avg)
            mod_Avg_Biom[z,i,mod] = np.mean(real_biom)
        
            #Heatmap 4: How different are the states in terms of survivors?
            mod_Var_Surv[z,i,mod] = np.var(real_surv)
        
            #Heatmap 5: How different are the states in terms of biomasses?
            mod_Var_Biom[z,i,mod] = np.var(real_biom)
        
            #Heatmap 6: What is the average single-species biomass?
            real_singlesppbiom = []
            for i1 in range(len(avg_biomAB)):
                for i2 in range(len(FreqAB)):
                    real_singlesppbiom.append(avg_biomAB[i1])        
            mod_Avg_biom[z,i,mod] = np.mean(real_singlesppbiom)
        
            #Heatmap 7: What is the avg variance of single-species biomass?
            real_singlesppbiomvar = []
            for i1 in range(len(var_biomAB)):
                for i2 in range(len(FreqAB)):
                    real_singlesppbiomvar.append(var_biomAB[i1])        
            mod_Var_biom[z,i,mod] = np.mean(real_singlesppbiomvar) #in all states of a given A,B, how different species are in average?
        

#Average over models:

Numstates = np.zeros((meanAfrac, meanBfrac)) # Number of different observed states: len(FreqAB)
Numcycles = np.zeros((meanAfrac, meanBfrac)) # Number of different observed cycles
Avg_Surv = np.zeros((meanAfrac, meanBfrac))   # Average surviving species: weighted (FreqAB) avg of Surviving list.
Avg_Biom = np.zeros((meanAfrac, meanBfrac))   # Average biomass: weighteed avg of Biomassess list.
Var_Surv = np.zeros((meanAfrac, meanBfrac))   # Variance of the surviving species number
Var_Biom = np.zeros((meanAfrac, meanBfrac))   # Variance of the biomassess.
Avg_biom = np.zeros((meanAfrac, meanBfrac))   # Average single-species biomass: weighteed avg of biomassess list.
Var_biom = np.zeros((meanAfrac, meanBfrac))   # variance of single-species biomass: weighteed avg of biomassess list.
Frac_cycles = np.zeros((meanAfrac, meanBfrac)) 

for i in range(meanAfrac):
    for j in range(meanBfrac):
        Numstates[i,j] = sum(mod_Numstates[i,j,:])/float(models)
        Numcycles[i,j] = sum(mod_Numcycles[i,j,:])/float(models)        
        Avg_Surv[i,j] = sum(mod_Avg_Surv[i,j,:])/float(models)
        Avg_Biom[i,j] = sum(mod_Avg_Biom[i,j,:])/float(models)
        Var_Surv[i,j] = sum(mod_Var_Surv[i,j,:])/float(models)
        Var_Biom[i,j] = sum(mod_Var_Biom[i,j,:])/float(models)
        Avg_biom[i,j] = sum(mod_Avg_biom[i,j,:])/float(models)
        Var_biom[i,j] = sum(mod_Var_biom[i,j,:])/float(models)
        Frac_cycles[i,j] = sum(mod_Frac_cycles[i,j,:])/float(models)
        
endclock = timer()
print("Program runtime", endclock - startclock)

################## PLOTS ###############################

#Figure 1: number of states

fig,(ax1,numcycles, cycles)= plt.subplots(1,3,figsize=(15,5))

im1 = ax1.imshow(Numstates)
#ax1.plot(meanAlist, Bcritplus, label="bistab. threshold", linewidth=2, color="white")
#ax1.plot(meanAlist, Bcritminus, label="bistab. threshold", linewidth=2, color="white", linestyle="--")
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
ax1.set_xticks(np.arange(len(meanAlist)))
ax1.set_yticks(np.arange(len(meanBlist)))
ax1.set_xticklabels(stra)
ax1.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax1.invert_yaxis()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
ax1.set_ylabel('avg competition')
ax1.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax1.set_title("Number of observed states")


imcycles = numcycles.imshow(Numcycles)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
numcycles.set_xticks(np.arange(len(meanAlist)))
numcycles.set_yticks(np.arange(len(meanBlist)))
numcycles.set_xticklabels(stra)
numcycles.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(numcycles.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
numcycles.invert_yaxis()
divider = make_axes_locatable(numcycles)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(imcycles, cax=cax, orientation='vertical')
numcycles.set_ylabel('avg predation')
numcycles.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
numcycles.set_title("Number of observed cycles")

im2 = cycles.imshow(Frac_cycles, cmap="cividis")
#ax1.plot(meanAlist, Bcritplus, label="bistab. threshold", linewidth=2, color="white")
#ax1.plot(meanAlist, Bcritminus, label="bistab. threshold", linewidth=2, color="white", linestyle="--")
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
cycles.set_xticks(np.arange(len(meanAlist)))
cycles.set_yticks(np.arange(len(meanBlist)))
cycles.set_xticklabels(stra)
cycles.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(cycles.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
cycles.invert_yaxis()
divider = make_axes_locatable(cycles)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
cycles.set_ylabel('avg competition')
cycles.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
cycles.set_title("Fraction of cycling simulations")


fig.tight_layout()
nom = "50modlong_sigma%1.2f.png" % varinter
plt.savefig(nom, format='png')
plt.close()


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
#Figure 2: statistics of states

fig,((ax2, ax3, ax4) ,(ax5, ax6, ax7)) = plt.subplots(2, 3, figsize=(20,10))


###################################################################

im2 = ax2.imshow(Avg_Surv)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
ax2.set_xticks(np.arange(len(meanAlist)))
ax2.set_yticks(np.arange(len(meanBlist)))
ax2.set_xticklabels(stra)
ax2.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
divider = make_axes_locatable(ax2)
ax2.invert_yaxis()
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.set_ylabel('avg competition')
ax2.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax2.set_title("Average surviving species")

###################################################################33

im3=ax3.imshow(Avg_Biom)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
ax3.set_xticks(np.arange(len(meanAlist)))
ax3.set_yticks(np.arange(len(meanBlist)))
ax3.set_xticklabels(stra)
ax3.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
divider = make_axes_locatable(ax3)
ax3.invert_yaxis()
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')
ax3.set_ylabel('avg competition')
ax3.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax3.set_title("Average total Biomass")

###################################################################33

im4=ax4.imshow(Avg_biom)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
ax4.set_xticks(np.arange(len(meanAlist)))
ax4.set_yticks(np.arange(len(meanBlist)))
ax4.set_xticklabels(stra)
ax4.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax4.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
divider = make_axes_locatable(ax4)
ax4.invert_yaxis()
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im4, cax=cax, orientation='vertical')
ax4.set_ylabel('avg competition')
ax4.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax4.set_title("Avg of single-species biomass averages inside a state")

###################################################################33

im5=ax5.imshow(Var_Surv)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
ax5.set_xticks(np.arange(len(meanAlist)))
ax5.set_yticks(np.arange(len(meanBlist)))
ax5.set_xticklabels(stra)
ax5.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
divider = make_axes_locatable(ax5)
ax5.invert_yaxis()
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im5, cax=cax, orientation='vertical')
ax5.set_ylabel('avg competition')
ax5.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax5.set_title("Variance of surviving species")

###################################################################33

im6=ax6.imshow(Var_Biom)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
ax6.set_xticks(np.arange(len(meanAlist)))
ax6.set_yticks(np.arange(len(meanBlist)))
ax6.set_xticklabels(stra)
ax6.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax6.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
divider = make_axes_locatable(ax6)
ax6.invert_yaxis()
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im6, cax=cax, orientation='vertical')
ax6.set_ylabel('avg competition')
ax6.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax6.set_title("Variance of total biomass")

###################################################################

im7=ax7.imshow(Var_biom)
stra = ["{:.3f}".format(i) for i in meanAlist]
strb = ["{:.3f}".format(i) for i in meanBlist]
# Show all ticks and label them with the respective list entries
ax7.set_xticks(np.arange(len(meanAlist)))
ax7.set_yticks(np.arange(len(meanBlist)))
ax7.set_xticklabels(stra)
ax7.set_yticklabels(strb)
# Rotate the tick labels and set their alignment.
plt.setp(ax7.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
divider = make_axes_locatable(ax7)
ax7.invert_yaxis()
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im7, cax=cax, orientation='vertical')
ax7.set_ylabel('avg competition')
ax7.set_xlabel('avg cooperation')
plt.gca().invert_yaxis()
ax7.set_title("Avg of single-species biomass variances: How different species are inside a state")
fig.tight_layout()

nom = "50modlong_statistics_sigma%1.2f.png" % varinter
plt.savefig(nom, format='png')
plt.close()

