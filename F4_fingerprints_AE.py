import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D    

# THIS CODE EXPLORES A MULTIPLICITY OF OBSERVABLE TRAITS (FINGERPRINTS) FOR THE ALLEE EFFECT MODEL
# BY RANDOMLY GENERATING A,B,sigma SYSTEMS AND EXPLORING THEIR FINAL STATES
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ############################

# EXPLORED STD DEVIATION VALUES
sigma_init = 1.00001
sigma_max = 3.0001
points = 50

perturb_min = 0.001
perturb_max = 3.0 # Maximum standard deviation of a normally distributed random perturbation

# EXPLORED Aij AND Bij RANGES # see F3 main text
Amin = 0.0
Amax = 0.5
Bmin = 0.0
Bmax = 0.14
sigmamin = 1.10001
sigmamax = 3.00001

run_time = 10000 

SP = 50

reps= 300
rounds=400

EPS=10.**-20 #Threshold of imm igration
PRECISION = 2 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES
SI_threshold = SP * 0.01 # HERE WE TEST FOR THE RELATIVE VERSION OF THE DISSIMILARITY INDEX (SEE SI I.D.2, see other fingerprints codes for the absolute dissimilarity index approach)

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

##################################################################################

startclock = timer()


#SYSTEM METRICS:
sigma_list = []
meanA_list = []
meanB_list = []

#STATES OBSERVED IN A GIVEN SYSTEM (and fraction of unstable runs)
Numstates = []
Numstates_repeat = []
Frac_unst = []

#DIVERSITIES OF THESE STATES
Diversities = []

#JACCARD DISTANCES TO PERTURBED STATES
YesNoFrac = []
YesNoFrac_repeat = []
Jaccards = []
Overlaps = []
JaccDiv = []
Sigmas = []
JaccSig = []
SigmaSee = []
Extinctions = []

# EACH ROUND = A RANDOMLY GENERATED (A,B) COUPLE
num_unst=0
total = 0
for ro in range(rounds):
    print("round: ", ro+1, " of ", rounds)
    
    # EACH POINT = A SIGMA VALUE ASSIGNED TO THAT RANDOM A,B COUPLE, BETWEEN SIGMA-INIT AND SIGMA-MAX (to control we ony explore both homog. and het.)
    for z in range(points):
        sigma = sigma_init + (z*(sigma_max-sigma_init)/float(points))

      # GENERATE RANDOM INTRA-SPP PARAMETERS
       
        d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=SP)
        g=np.random.lognormal(mean=np.log(meang), sigma=np.log(varg), size=SP)
        Aii=np.random.lognormal(mean=np.log(meanAii), sigma=np.log(varAii), size=SP)
        Bii=np.random.lognormal(mean=np.log(meanBii), sigma=np.log(varBii), size=SP)
        
        # GENERATE RANDOM A AND B
        meanA = np.random.uniform(Amin,Amax)
        meanB = np.random.uniform(Bmin,Bmax)

        B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(sigma), size=(SP,SP)) 
        np.fill_diagonal(B,Bii) 
   
        A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(sigma), size=(SP,SP))
        np.fill_diagonal(A,Aii)
        
        StatAB = []     # For given A and B: A matrix with the different states observed
        FreqAB = []     # For given A and B: An array, how often each state is reached
        SurvAB = []     # For given A and B: An array, the number of species surviving in each state
        JaccAB = []     # For given A and B: An array, the list of Jaccard distances observed after perturbing each state
        JaccDivAB =  []
        sigma_perturb_AB = []
        JaccSigAB = []
        SigmaSeeAB = []
        OverlapAB = []
        YesNoAB = []
        

        #print("The system I explore:")
        #print(meanA,meanB,sigma)
        # FOR THIS GIVEN A,B,sigma SYSTEM: PERFORM MANY RUNS AND COUNT STATES AND METRICS
        for j in range(0,reps): 
            meanA_list.append(meanA)
            meanB_list.append(meanB)
            sigma_list.append(sigma) 

            transitions = 0
            total+=1

            def run(S,tmax=run_time, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
             
                #Random uniform i.cc. 
                x0 = np.random.random(SP)        
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
    
            # Simulation: write spp trajectories
            time,trajectories = run(SP)
            # trajectories[i,t] is the abundance of species i at time t: trajectories[:,-1] is the final abundance of all spp.
            
            #Data 1: CREATE MATRIX OF OBSERVED STATES: IF STATE IS NEW: ADD IT.
            #Data 2: Count how often that state is reached
            #Data 3: Count number of survivors 
            
            finalstate = [m for m in trajectories[:,-1]] 
            cleanstate = list(np.around(np.array(finalstate),PRECISION)) # NO DECIMALS TO AVOID OVER-PRECISE STATE COMPARISONS
            alivestate = [m for m in finalstate if m>0.01] #FOR EFFICIENCY PURPOSES: LIST OF ONLY SURVIVING SPP.'S ABUNDANCES
            
            #################### STABILITY CHECK: RUN MORE TIME AND OBSERVE IF STATE REMAINS THE SAME #########################
            def run(
                     S, #Species number
                     tmax=int(run_time/100), # Run a shorter time-lapse
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
            
            #################### PERTURBATION: PERTURB RANDOMLY AND MEASURE FINAL TURNOVER #########################
            
            # RANDOM PERTURBATION
            
            sigma_perturb = np.random.uniform(perturb_min,perturb_max)
            perturbed_state = [np.maximum(0,m+n) for m,n in zip(finalstate, np.random.normal(0.0,sigma_perturb,SP))]        
            
            # count real extinctions
            extinctions = 0
            for conta in range(SP):
                # if species is present in one state but not the other, add +1
                if finalstate[conta]>0.01 and perturbed_state[conta]==0.0:
                    extinctions +=1
                    
            Extinctions.append(extinctions)
            
            
            def run(
                     S, #Species number
                     tmax=run_time, # Run a shorter time-lapse
                     EPS=EPS,**kwargs
                         ):

                 def eqs(t,x):
                     dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                     dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                     return dx
                 #Solve the system of equations:
                 x0 = perturbed_state           
                 sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                 time=sol.t
                 trajectories=sol.y
                 return time, trajectories
    
            # Simulation: write spp trajectories
            time_perturb,trajectories_perturb = run(SP)
            final_perturb = [m for m in trajectories_perturb[:,-1]] 
            cleanstate_perturb = list(np.around(np.array(final_perturb),PRECISION)) # NO DECIMALS TO AVOID OVER-PRECISE STATE COMPARISONS
            
            ###########################################################################################################
            
            # DO WE SEE A NEW STATE, AN ALREADY-SEEN STATE, OR CYCLING/CHAOTIC BEHAVIOUR?
            
            # 1: SEE IF STATE IS STABLE IN TIME
            
            diff = 0.0
            for spp in range(SP):
                diff += 2 * abs ( finalstate[spp] - finalstateplus[spp] ) / (finalstate[spp]+finalstateplus[spp]+0.01)
                            
            if diff > 4*SI_threshold: # if Similarity index is much higher than threshold, the state is not stable
                num_unst+=1
                SurvAB.append(np.nan) #We do not measure data from non-stable results
                JaccAB.append(np.nan)  #We do not measure data from non-stable results
                JaccDivAB.append(np.nan)
                JaccSigAB.append(np.nan)
                SigmaSeeAB.append(np.nan)
                OverlapAB.append(np.nan)
                sigma_perturb_AB.append(np.nan) 
            # 2: IF STABLE, MEASURE THE SIMILARITY INDEX WITH ALL PREVIOUS STATES 
            
            else:
                state = 0
                SI_perturb = 0.0
                while state<len(StatAB): # compare all states with new "cleanstate":
                    SI = 0.0
                    SI_perturb = 0.0
                    for spp in range(SP):
                        SI += 2* abs ( StatAB[state][spp] - cleanstate[spp] ) / (StatAB[state][spp] + cleanstate[spp] + 0.01) #Add the Euclidean distance
                        SI_perturb += 2* abs ( cleanstate_perturb[spp] - cleanstate[spp] ) / (cleanstate_perturb[spp] + cleanstate[spp] + 0.01)
                    
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
                
                # HAS THE STATE TRANSITIONED INTO ANOTHER ONE AFTER THE PERTURBATION?
                
                if SI_perturb > SI_threshold: 
                    
                    sigma_perturb_AB.append(sigma_perturb)
                     
                    transitions +=1.0
                    
                    # Jaccard distance: difference in present and absent species between states
                    distance = 0.0
                    overlap = 0.0
                    for spp in range(SP):
                        # if species is present in one state but not the other, add +1
                        if cleanstate[spp]>0.01 and cleanstate_perturb[spp]<0.01:
                            distance +=1
                        elif cleanstate[spp]<0.01 and cleanstate_perturb[spp]>0.01:
                            distance +=1                    
                        elif cleanstate[spp]>0.01 and cleanstate_perturb[spp]>0.01:
                            overlap +=1
                    
                    JaccAB.append(distance)
                    OverlapAB.append(overlap)
                    
                    if len(alivestate)>0:
                        JaccDivAB.append(distance/len(alivestate))
                    else:
                        JaccDivAB.append(0) # Because the state is the same x=0, no turnover.    
                    JaccSigAB.append(distance/sigma_perturb)
                
                    SigmaSeeAB.append(perturb_max - sigma_perturb) # If there has been a transition, save the necessary sigma that generated it.
                else:
                    JaccAB.append(np.nan)
                    OverlapAB.append(np.nan)
                    JaccDivAB.append(np.nan)
                    JaccSigAB.append(np.nan)
                    SigmaSeeAB.append(np.nan)
                    sigma_perturb_AB.append(np.nan)    
                    
        #End of repetitions for single (A,B,sigma): compute what we've seen
        #print("States I have seen:")
        #print(len(FreqAB))
        Numstates.append(len(FreqAB))
        for repe in range(len(SurvAB)):
            Numstates_repeat.append(len(FreqAB)) #Atribute the same number of elements than the rest of list of lists (Omega, Omega, ...)
        Frac_unst.append(float(num_unst/reps))
        #print("Diversities I have seen of these states")
        #print(SurvAB)
        Diversities.append(SurvAB) #Diversities is a list of lists
        Jaccards.append(JaccAB) # Jaccards is a list of lists
        JaccDiv.append(JaccDivAB)
        Sigmas.append(sigma_perturb_AB)
        JaccSig.append(JaccSigAB)
        SigmaSee.append(SigmaSeeAB)
        Overlaps.append(OverlapAB)
        if len(FreqAB)>0:
            YesNoFrac.append(transitions / len(FreqAB))
            for repe in range(len(SurvAB)):
                YesNoFrac_repeat.append(transitions / len(FreqAB))
        else:
            YesNoFrac.append(np.nan)
            for repe in range(len(SurvAB)):
                YesNoFrac_repeat.append(np.nan)
            
##################################################################################
##################################################################################
##################################################################################
##################################################################################

print("Final run, unstable states: ", num_unst, "total: ", total)





##################################################################################
##################################################################################
##################################################################################
##################################################################################

#SAVE DATA INTO FILE TO PLOT DIFFERENT VERSIONS

#undo list of lists, each as a single list
import itertools

# RESULTS LISTS:

# Sigma, A, B, Omega, YesNo, D, J, O, J/D, Sigmatrans, J/Sigmatrans, SigmaMax - Sigmatrans

flatsigma_list = sigma_list 
flatmeanA_list = meanA_list 
flatmeanB_list = meanB_list 

flatNumstates_repeat = Numstates_repeat 
flatYesNoFrac_repeat = YesNoFrac_repeat 

flatDiversities = list(itertools.chain.from_iterable(Diversities)) 
flatJaccards = list(itertools.chain.from_iterable(Jaccards)) 
flatOverlaps = list(itertools.chain.from_iterable(Overlaps)) 
flatJaccDiv = list(itertools.chain.from_iterable(JaccDiv)) 
flatSigmas = list(itertools.chain.from_iterable(Sigmas)) 
flatJaccSig = list(itertools.chain.from_iterable(JaccSig)) 
flatSigmaSee = list(itertools.chain.from_iterable(SigmaSee)) 

#LISTS AS COLUMNS

rows = zip(flatsigma_list, flatmeanA_list, flatmeanB_list, flatNumstates_repeat, flatYesNoFrac_repeat, Extinctions, flatDiversities, flatJaccards, flatOverlaps, flatJaccDiv, flatSigmas, flatJaccSig, flatSigmaSee)


import csv

with open('SI001_F004_200_200_randsigma.csv', 'w') as f:
    writer = csv.writer(f)
    row1 = ["Sigma", "A", "B", "Omega", "YesNoFrac", "Ext","D", "J", "O", "J/D", "SigmaP", "J/SigmaP", "SigmaMax - SigmaP"]
    writer.writerow(row1)
    for row in rows:
        writer.writerow(row)


exit()
