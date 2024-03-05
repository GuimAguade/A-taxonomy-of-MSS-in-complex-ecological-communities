import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
        
# CODE TO EXPLORE THE STATISTICS OF STATES UNDER COMP + COOP FOR THE GENERALIZED LOTKA-VOLTERRA MODEL (Barbier et al., 2018)
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

Sstart = 5
#Smax = 500

meanA=-1.0

varA=0.2

frac=200

reps=100


temps1 = 1000
temps2 = 100

EPS=10.**-20 #Threshold of immigration

#PRECISION = 5 #NUMBER OF DECIMAL POINTS CONSIDERED FOR SPECIES ABUNDANCE - AVOID OVER-PRECISE STATES

SI_threshold = 0.001

# DEFINING PARAMETERS:

# Intraspecies parameters: d

meand = 1.0
vard = 0.01



##################################################################################

# Interaction range:

Slist = []

Frac_cycles = [] # number of runs that end in non-stable behavior

z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    S = Sstart + z
    Slist.append(S)
    
    d=np.random.normal(meand,vard,S)
    
    i=0
    while i<1:
        
        num_unst = 0
        exp_warning=0
        
        for j in range(0,reps): #Perform experiment many times
            #print(j," of ",reps)
            # One simulation: 
            
            A=np.random.normal(meanA, varA, size=(S,S)) #COOP matrix, all interactions uniform between 0 and coop    
            np.fill_diagonal(A,-d)
            
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x)) 
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
                            
            #################### CYCLE CHECK: RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT #########################
            
            def run(S,tmax=temps2,EPS=EPS,**kwargs):
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
        
          
        
        Frac_cycles.append( 1 - (num_unst / float(reps) ) )
        
        

        i+=1
    z+=5    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)

################## PLOTS ###############################

#Figure 1: Error


# Scatter plot
plt.plot(Slist, Frac_cycles)
plt.xlabel('N')
plt.ylabel('Fraction of stable simulations')

# Adding a vertical dashed line at varBunin
#plt.axvline(x=varBunin, color='gray', linestyle='--')

# Set major ticks for x-axis at intervals of 0.2 and minor ticks at intervals of 0.05
#plt.xticks(np.arange(0, 5*varBunin, 0.2))
#plt.gca().xaxis.set_minor_locator(plt.FixedLocator(np.arange(0, 5*varBunin, 0.05)))

# Set major ticks for y-axis at intervals of 0.2 and minor ticks at intervals of 0.05
#plt.yticks(np.arange(0, 1, 0.1))
#plt.gca().yaxis.set_minor_locator(plt.FixedLocator(np.arange(0, 1, 0.05)))

# Adding more detailed gridlines
#plt.grid(color='lightgray', linestyle='-', linewidth=0.5, which='both')

plt.tight_layout()
nom = "fraction_unstable_cliques.png"
plt.savefig(nom, format='png')
plt.close()

