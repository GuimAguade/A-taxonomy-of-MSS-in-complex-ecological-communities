import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D    

# THIS CODE EXPLORES A MULTIPLICITY OF OBSERVABLE TRAITS (FINGERPRINTS) FOR THE GENERALIZED LOTKA-VOLTERRA MODEL
# BY RANDOMLY GENERATING A,B,sigma SYSTEMS AND EXPLORING THEIR FINAL STATES
# G AGUADÉ-GORGORIÓ


####################### PARAMETERS AND SIMULATION VALUES ############################


points = 10

perturb_min = 0.001
perturb_max = 3.0 #standard deviation of a normally distributed random perturbation

# EXPLORED Aij AND Bij RANGES
Amin = -0.5
Amax = -0.25

sigmamin = 0.0
sigmamax = 0.5


run_time = 10000 

SP = 50

reps = 300
rounds = 400

EPS=10.**-20 #Threshold of imm igration
#PRECISION = 5 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES
SI_threshold = 0.03

# DEFINING PARAMETERS:
varintra = 1.1

# Intraspecies parameters: d

meand = 0.25
vard = varintra

##################################################################################

startclock = timer()


#SYSTEM METRICS:
sigma_list = []
meanA_list = []


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
        sigma = np.random.uniform(sigmamin,sigmamax)
        
        # GENERATE RANDOM INTRA-SPP PARAMETERS
       
        d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=SP)
        
        # GENERATE RANDOM A AND B
        meanA = np.random.uniform(Amin,Amax)
        A=np.random.normal(meanA, sigma, size=(SP,SP))
        np.fill_diagonal(A,-d)
        
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
        
        # FOR THIS GIVEN A,B,sigma SYSTEM: PERFORM MANY RUNS AND COUNT STATES AND METRICS
        for j in range(0,reps): 
            meanA_list.append(meanA)

            sigma_list.append(sigma) 

            transitions = 0
            total+=1

            def run(S,tmax=run_time, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) 
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
            #cleanstate = list(np.around(np.array(finalstate),PRECISION)) # NO DECIMALS TO AVOID OVER-PRECISE STATE COMPARISONS
            alivestate = [m for m in finalstate if m>0.01] #FOR EFFICIENCY PURPOSES: LIST OF ONLY SURVIVING SPP.'S ABUNDANCES
            
            #################### STABILITY CHECK: RUN MORE TIME AND OBSERVE IF STATE REMAINS THE SAME #########################
            def run(
                     S, #Species number
                     tmax=int(run_time/50), # Run a shorter time-lapse
                     EPS=EPS,**kwargs
                         ):

                 def eqs(t,x):
                     dx = x*(1+np.dot(A,x)) 
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
                     dx = x*(1+np.dot(A,x)) 
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
            #cleanstate_perturb = list(np.around(np.array(final_perturb),PRECISION)) # NO DECIMALS TO AVOID OVER-PRECISE STATE COMPARISONS
            
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

                while state<len(StatAB): # compare all states with new "cleanstate":

                    SI = 0

                    for spp in range(SP):
                        if ( abs ( StatAB[state][spp] - finalstate[spp] ) ) > SI_threshold:
                            SI+=1
                            break
                    
                    # IF SI IS SMALLER THAN THRESHOLD FOR ALL SPECIES, THIS IS THE SAME STATE
                    if SI==0:
                        FreqAB[state]+=1 #Increase frequency by one
                        SurvAB.append(len(alivestate)) #count it as a (repeated) diversity
                        state += 2*reps # Stop the search
                    
                    #ELSE, KEEP ON SEARCHING
                    else:
                        state+=1
                
                if state < 2*reps: #We have reached the end of the loop without finding the same state: the state is new!                    
                    StatAB.append(finalstate)
                    SurvAB.append(len(alivestate))
                    FreqAB.append(1)            
                
                # HAS THE STATE TRANSITIONED INTO ANOTHER ONE AFTER THE PERTURBATION?
                
                SI_perturb=0
                
                for spp in range(SP):
                    if ( abs ( final_perturb[spp] - finalstate[spp] ) ) > SI_threshold:
                        SI_perturb += 1
                        break
                 
                if SI_perturb > 0: 
                    
                    sigma_perturb_AB.append(sigma_perturb)
                     
                    transitions +=1.0
                    
                    # Jaccard distance: difference in present and absent species between states
                    distance = 0.0
                    overlap = 0.0
                    for spp in range(SP):
                        # if species is present in one state but not the other, add +1
                        if finalstate[spp]>0.01 and final_perturb[spp]<0.01:
                            distance +=1
                        elif finalstate[spp]<0.01 and final_perturb[spp]>0.01:
                            distance +=1                    
                        elif finalstate[spp]>0.01 and final_perturb[spp]>0.01:
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

print("unstable states: ", num_unst, "total: ", total)







#SAVE DATA INTO FILE TO PLOT DIFFERENT VERSIONS

#undo list of lists, each as a single list
import itertools

# RESULTS LISTS:

# Sigma, A, B, Omega, YesNo, D, J, O, J/D, Sigmatrans, J/Sigmatrans, SigmaMax - Sigmatrans



flatsigma_list = sigma_list 
flatmeanA_list = meanA_list 

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

rows = zip(flatsigma_list, flatmeanA_list, flatNumstates_repeat, flatYesNoFrac_repeat, Extinctions, flatDiversities, flatJaccards, flatOverlaps, flatJaccDiv, flatSigmas, flatJaccSig, flatSigmaSee)


import csv

with open('GLV_FPS.csv', 'w') as f:
    writer = csv.writer(f)
    row1 = ["Sigma", "A", "Omega", "YesNoFrac", "Ext","D", "J", "O", "J/D", "SigmaP", "J/SigmaP", "SigmaMax - SigmaP"]
    writer.writerow(row1)
    for row in rows:
        writer.writerow(row)


exit()
