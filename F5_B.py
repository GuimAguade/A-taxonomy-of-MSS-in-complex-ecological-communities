import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# THIS SIMPLE CODE EXPLORES THE TRAJECTORIES FOR A SPECIES-RICH COMMUNITY UNDER THE ALLEE EFFECT MODEL
# IT GENERATES RANDOM INITIAL CONDITIONS, PLOTS THE TIME EVOLUTION, AND THEN PERTURBS THE FINAL STATE AND 
# REPEATS THE PROCESS UP TO SIX TIMES - AS A SIMPLE VISUALIZATION OF POTENTIALLY DIFFERENT STABLE STATES
# G AGUADÉ-GORGORIÓ

mean_perturbation = 1.0
sigma_perturbation = 2.5

temps = 1000

EPS=10.**-20 #Threshold of immigration

S=50

# DEFINING PARAMETERS:
varintra = 1.01
varinter = 2.0

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
d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=S)
g=np.random.lognormal(mean=np.log(meang), sigma=np.log(varg), size=S)
Aii=np.random.lognormal(mean=np.log(meanAii), sigma=np.log(varAii), size=S)
Bii=np.random.lognormal(mean=np.log(meanBii), sigma=np.log(varBii), size=S)

# interaction heterogeneity
varA = varinter
varB = varinter  

meanA = 0.3
A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(S,S)) 
np.fill_diagonal(A,Aii)

meanB = 0.04
B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(S,S)) 
np.fill_diagonal(B,Bii)

def calculate_diversity(trajectories, threshold=0.01):
    # Count elements larger than the threshold for each time point
    diversity = [np.sum(timestep > threshold) for timestep in trajectories.T]
    return diversity
    
#####################################################################################################
# 1
#####################################################################################################

def run(S, tmax=temps, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
        dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        return dx

    x0=[v*2 for v in np.random.random(S)]  
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time=sol.t
    trajectories=sol.y
    return time, trajectories

time,trajectories = run(S)   

finalstate1 = [m for m in trajectories[:,-1]]

# perturb one species

perturbed_state = finalstate1

# the perturbed has to be present: 

# Iterate through the list using enumerate to get both index and value
for chosen1, value in enumerate(finalstate1):
    if value > 0.01:
        break  # Stop the loop once the condition is met

perturbation=np.random.lognormal(mean=np.log(mean_perturbation), sigma=np.log(sigma_perturbation))

# find allee effect of that species:

#AE = (Aii[chosen1] - (Bii[chosen1]*g[chosen1]) - d[chosen1]- np.sqrt( (Aii[chosen1] - (Bii[chosen1]*g[chosen1]) - d[chosen1])*(Aii[chosen1] - (Bii[chosen1]*g[chosen1]) - d[chosen1]) - (4*Bii[chosen1]*g[chosen1]*d[chosen1])  ) ) / (2*Bii[chosen1])

perturbed_state[chosen1] = 0# AE - 0.1 # we decrease the abundance of one species

# DIVERSITY

div1 = calculate_diversity(trajectories, threshold=0.01)


#####################################################################################################
# 2
#####################################################################################################

def run(S, tmax=temps, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
        dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        return dx

    x0=perturbed_state
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time2,trajectories2 = run(S)

t2 = [m+time[-1] for m in time2]

finalstate2 = [m for m in trajectories2[:,-1]]

# perturb one species

perturbation=np.random.lognormal(mean=np.log(mean_perturbation), sigma=np.log(sigma_perturbation))

perturbed_state2 = finalstate2

# the perturbed has to be present: 

# Iterate through the list using enumerate to get both index and value
for chosen2, value in enumerate(finalstate2):
    if value > 0.01:
        if chosen2 == chosen1:
            continue
        else:
            break  # Stop the loop once the condition is met

# find allee effect of that species:

# AE = (Aii[chosen2] - (Bii[chosen2]*g[chosen2]) - d[chosen2]- np.sqrt( (Aii[chosen2] - (Bii[chosen2]*g[chosen2]) - d[chosen2])*(Aii[chosen2] - (Bii[chosen2]*g[chosen2]) - d[chosen2]) - (4*Bii[chosen2]*g[chosen2]*d[chosen2])  ) ) / (2*Bii[chosen2])

perturbed_state2[chosen2] = 0 # AE + 0.1 # we decrease the abundance of one species


# DIVERSITY

div2 =calculate_diversity(trajectories2, threshold=0.01)

#####################################################################################################
# 3
#####################################################################################################

def run(S, tmax=temps, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
        dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        return dx

    x0=perturbed_state2
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time3,trajectories3 = run(S)

t3 = [m+t2[-1] for m in time3]

# DIVERSITY

div3 = calculate_diversity(trajectories3, threshold=0.01)

finalstate3 = [m for m in trajectories3[:,-1]]


#####################################################################################################
#####################################################################################################

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

for i in range(len(trajectories)):
    ax1.plot(time,trajectories[i], linestyle = "-", color="gray", alpha=0.5)
ax1.plot(time,trajectories[chosen1],linestyle="-", color="firebrick", alpha=1)     
ax1.plot(time,trajectories[chosen2],linestyle="-", color="rebeccapurple", alpha=1)  
for i in range(len(trajectories2)):
    ax1.plot(t2,trajectories2[i], linestyle = "-", color="gray", alpha=0.5)
ax1.plot(t2,trajectories2[chosen1],linestyle="-", color="firebrick", alpha=1)  
ax1.plot(t2,trajectories2[chosen2],linestyle="-", color="rebeccapurple", alpha=1)  
for i in range(len(trajectories3)):
    ax1.plot(t3,trajectories3[i], linestyle = "-", color="gray", alpha=0.5)
ax1.plot(t3,trajectories3[chosen1],linestyle="-", color="firebrick", alpha=1)  
ax1.plot(t3,trajectories3[chosen2],linestyle="-", color="rebeccapurple", alpha=1)  

#ax2 = ax1.twinx()

ax2.plot(time,div1, linestyle = "--", color="black") 
ax2.plot(t2, div2, linestyle = "--", color="black")
ax2.plot(t3, div3, linestyle = "--", color="black")
    
ax2.set_ylim(0, ax2.get_ylim()[1])  # Setting the lower limit to 0


    
plt.show()
