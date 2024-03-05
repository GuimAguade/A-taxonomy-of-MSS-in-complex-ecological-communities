import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

# CODE TO EXPLORE THE STATISTICS OF THE 3 FAMILIES OF STATES

####################### PARAMETERS AND SIMULATION VALUES ##############################

temps1 = 800

species_experiments = 5

reps=2000
systems=10

EPS=10.**-20 #Threshold of immigration
PRECISION = 2 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

SI_threshold = 0.03

# DEFINING PARAMETERS:
varintra = 1.1
# Intraspecies parameters: d, g, Aii
meand=0.1
vard=varintra 
meang=1.0 
varg=varintra
meanAii=0.5
varAii=varintra
meanBii = 0.1
varBii= varintra

SPlist = []

Species = [2,4,6, 8, 10]

##################################################################################

# interaction heterogeneity


##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
startclock = timer()

meanAlist = []
meanBlist = []
sigmalist = []
Numstates = []
Avg_Surv = []
Frac_cycles = []
Color = []

sys = 0

while sys < systems: # 4 systems, 5 species pools each, 5000 simulations each
    print("system: ", sys+1, " of ", systems)
    
    varinter = 1.0001 #LOW FOR GLOBAL AND LOCAL, HIGH FOR CLIQUES
    
    if sys < 40:
        # GENERATE FIXED A AND B
        chooser = 4
        print("system type: ", chooser)
        if chooser == 1: #GLOBAL STATE
            meanA = abs(np.random.normal(0.3,0.001))
            meanB = abs(np.random.normal(0.0001,0.000001))           
        if chooser == 2: #MUTUAL EXCLUSION
            meanA = abs(np.random.normal(0.3,0.0001))
            meanB = abs(np.random.normal(0.15,0.00001))
        if chooser == 3: #CLIQUES
            meanA = abs(np.random.normal(0.3,0.001))
            meanB = abs(np.random.normal(0.05,0.0001))
            varinter = 2.0
        if chooser == 4: # LOCAL STATES
            meanA = abs(np.random.normal(0.00000001,0.00000001))
            meanB = meanA
    else:
        chooser = 3
        meanA = abs(np.random.normal(0.2,0.01))
        meanB = abs(np.random.normal(0.05,0.001))
        varinter = 2.0
            
    z = 0
    while z < species_experiments:
        
        SP = Species[z]
        print("species pool: ", SP)
        SI_threshold = SP * 0.01

        # GENERATE RANDOM INTRA-SPP PARAMETERS
       
        d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=SP)
        g=np.random.lognormal(mean=np.log(meang), sigma=np.log(varg), size=SP)
        Aii=np.random.lognormal(mean=np.log(meanAii), sigma=np.log(varAii), size=SP)
        Bii=np.random.lognormal(mean=np.log(meanBii), sigma=np.log(varBii), size=SP)
        
        Color.append(chooser)
        varA = varinter
        varB = varinter  
        meanAlist.append(meanA)
        meanBlist.append(meanB)
        sigmalist.append(varinter)
 
        B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(SP,SP)) 
        np.fill_diagonal(B,Bii) 

        A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(SP,SP))
        np.fill_diagonal(A,Aii)
        
            
        StatAB = []     # For given A and B: A matrix with the different states observed
        FreqAB = []     # For given A and B: An array, how often each state is reached
        SurvAB = []     # For given A and B: An array, the number of species surviving in each state

        num_unst=0.0    
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 

            def run(S,tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
                    #Solve the system of equations:
             
                #Random normal IC's with increasing variance: we start at 0.0
                x0 = np.random.random(SP)          
                 
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            time,trajectories = run(SP)
            finalstate = [m for m in trajectories[:,-1]] 
            cleanstate = list(np.around(np.array(finalstate),PRECISION)) #APPROXIMATION 1: NO DECIMALS TO AVOID OVER-PRECISE STATE DIFFERENTIATION
            alivestate = [m for m in finalstate if m>0.05] #APPROXIMATION 2 (SIMILAR): ONLY CONSIDER SPECIES THAT ARE BEYOND 1.0 ABUNDANCE (TO AVOID EFFECTS OF MIGRATION?)
            
            #################### CYCLE CHECK: RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT #########################
            def run(
                     S, #Species number
                     tmax=20, #maximum time
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
            finalstateplus = [m for m in trajectoriesplus[:,-1]] 
            cleanstateplus = list(np.around(np.array(finalstate),PRECISION)) #APPROXIMATION 1: NO DECIMALS TO AVOID OVER-PRECISE STATE DIFFERENTIATION
            alivestateplus = [m for m in finalstate if m>0.05] #APPROXIMATION 2 (SIMILAR): ONLY CONSIDER SPECIES THAT ARE BEYOND 1.0 ABUNDANCE (TO AVOID EFFECTS OF MIGRATION?)
            
            ###########################################################################################################
            
            # DO WE SEE A NEW STATE, AN ALREADY-SEEN STATE, OR CYCLING/CHAOTIC BEHAVIOUR?
            
            # 1: SEE IF STATE IS STABLE IN TIME
            
            diff = 0.0
            for spp in range(SP):
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
                    for spp in range(SP):
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
            
           
        SPlist.append(SP)           
        Numstates.append(len(FreqAB)) 
        Frac_cycles.append(num_unst / float(reps))
        #Heatmap 2: What is the average number of surviving spp?
        Avg_Surv.append(np.mean(SurvAB))
        
        z+=1
    
    sys +=1
 

        
endclock = timer()
print("Program runtime", endclock - startclock)


#LISTS AS COLUMNS

rows = zip(SPlist, Color, meanAlist, meanBlist, sigmalist, Numstates, Frac_cycles, Avg_Surv)

import csv

with open('low_diversity_scaling.csv', 'w') as f:
    writer = csv.writer(f)
    row1 = ["SPlist", "Color", "A", "B", "sigma", "Omega", "F", "D"] 
    writer.writerow(row1)
    for row in rows:
        writer.writerow(row)



import matplotlib.pyplot as plt
import csv
import numpy as np
from pandas import * 



# Open the file in 'r' mode, not 'rb'
csv_file = open('low_diversity_scaling.csv','r')

# reading CSV file
data = read_csv("low_diversity_scaling.csv")
 
# converting column data to list
SP = data['SPlist'].tolist()
Color = data['Color'].tolist()
meanA_list = data['A'].tolist()
meanB_list = data['B'].tolist()
Omega = data['Omega'].tolist()
Unstable = data['F'].tolist()
Diversities = data['D'].tolist()

#Retrieve each color to generate the mean:

# For each row with Color, each SP, generate the mean.

Species = [2,4,6,8,10]

grocSP = []
grocO = []
grocD = []
blueSP = []
blueO= []
blueD= []
greenSP= []
greenO= []
greenD= []
purpleSP= []
purpleO= []
purpleD= []

for i in range(len(Color)):
    if Color[i] == 4:
        grocSP.append(SP[i])
        grocO.append(Omega[i])
        grocD.append(Diversities[i])
    if Color[i] == 3:
        greenSP.append(SP[i])
        greenO.append(Omega[i])
        greenD.append(Diversities[i])
    if Color[i] == 2:
        blueSP.append(SP[i])
        blueO.append(Omega[i])
        blueD.append(Diversities[i])
    if Color[i] == 1:
        purpleSP.append(SP[i])
        purpleO.append(Omega[i])
        purpleD.append(Diversities[i])
        
# Now, the mean of all values with same SP number        
    
meangrocO = [0,0,0,0,0]
meangrocD = [0,0,0,0,0]

meanblueO= [0,0,0,0,0]
meanblueD= [0,0,0,0,0]

meangreenO= [0,0,0,0,0]
meangreenD= [0,0,0,0,0]

meanpurpleO= [0,0,0,0,0]
meanpurpleD= [0,0,0,0,0]

for i in range(len(grocSP)): # for each element that is yellow, we add 
    z=0
    while z < (len(Species)):
        if grocSP[i]==Species[z]:
            meangrocO[z] += grocO[i] / (len(grocSP) / 5.0)
            meangrocD[z] += grocD[i] / (len(grocSP) / 5.0)
            z += 10 # get out of loop, no need to compare it with others 
        else:
            z +=1

for i in range(len(blueSP)): # for each element that is yellow, we add 
    z=0
    while z < (len(Species)):
        if blueSP[i]==Species[z]:
            meanblueO[z] += blueO[i] / (len(blueSP) / 5.0)
            meanblueD[z] += blueD[i] / (len(blueSP) / 5.0)
            z += 10 # get out of loop, no need to compare it with others 
        else:
            z +=1

for i in range(len(greenSP)): # for each element that is yellow, we add 
    z=0
    while z < (len(Species)):
        if greenSP[i]==Species[z]:
            meangreenO[z] += greenO[i] / (len(greenSP) / 5.0)
            meangreenD[z] += greenD[i] / (len(greenSP) / 5.0)
            z += 10 # get out of loop, no need to compare it with others 
        else:
            z +=1
            
for i in range(len(purpleSP)): # for each element that is yellow, we add 
    z=0
    while z < (len(Species)):
        if purpleSP[i]==Species[z]:
            meanpurpleO[z] += purpleO[i] / (len(purpleSP) / 5.0)
            meanpurpleD[z] += purpleD[i] / (len(purpleSP) / 5.0)
            z += 10 # get out of loop, no need to compare it with others 
        else:
            z +=1          



fitN = []
fitEXP = []
fitNexp = []
fit2N = []
for z in range(len(Species)):
    fitN.append(Species[z])
    fitEXP.append(Species[z] + np.exp(0.13*Species[z]))
    fitNexp.append(np.power(2,Species[z]))
# PLOT 2D 

fig, (ax3, ax4) = plt.subplots(1,2, figsize=(10,4))

ax3.scatter(SP, Omega,  s=20, c = "#fde725", alpha = 0.5)
#ax3.plot(Species, meanpurpleO, c = "#440154", alpha = 1)
#ax3.plot(Species, fitN, c = "gray", ls = "--", alpha = 1)
#ax3.plot(Species, fitEXP, c = "black", ls = "--", alpha = 1)
ax3.plot(Species, fitNexp, c = "black", ls = "--", alpha = 1)
#ax3.plot(Species, meanblueO, c = "#31688e", alpha = 1)
#ax3.plot(Species, meangreenO, c = "#35b779", alpha = 1)
ax3.plot(Species, meangrocO, c = "#fde725", alpha = 1)
ax3.set_xlabel('Species pool')
ax3.set_ylabel('Omega')
ax3.set_yscale('log')

ax4.scatter(SP, Diversities, s=20, c = Color, alpha = 0.5)
ax4.plot(Species, meanpurpleD, c = "#440154", alpha = 1)
ax4.plot(Species, meanblueD, c = "#31688e", alpha = 1)
ax4.plot(Species, meangreenD, c = "#35b779", alpha = 1)
ax4.plot(Species, meangrocD, c = "#fde725", alpha = 1)
ax4.set_xlabel('Species pool')
ax4.set_ylabel('Diversity')
ax4.set_yscale('log')



fig.tight_layout()
#nom = "fingerprints_high.png" 
#plt.savefig(nom, format='png')
plt.show()
#plt.close()



exit()

















