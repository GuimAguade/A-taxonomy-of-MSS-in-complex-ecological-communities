import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# THIS SIMPLE CODE EXPLORES THE TRAJECTORIES FOR A SPECIES-RICH COMMUNITY UNDER THE ALLEE EFFECT MODEL
# IT GENERATES RANDOM INITIAL CONDITIONS, PLOTS THE TIME EVOLUTION, AND THEN PERTURBS THE FINAL STATE AND 
# REPEATS THE PROCESS UP TO SIX TIMES - AS A SIMPLE VISUALIZATION OF POTENTIALLY DIFFERENT STABLE STATES
# G AGUADÉ-GORGORIÓ

EPS=10.**-20 #Threshold of immigration

S=50

# DEFINING PARAMETERS:
varintra = 1.1
varinter = 1.1

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

meanB = 0.05
B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(S,S)) 
np.fill_diagonal(B,Bii)


#####################################################################################################
#####################################################################################################

def run(S, tmax=10000, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
        dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        return dx

    x0=[v*1.0 for v in np.random.random(S)]  
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time=sol.t
    trajectories=sol.y
    return time, trajectories

time,trajectories = run(S)   

finalstate1 = [m for m in trajectories[:,-1]]
perturbation = np.random.normal(0.0,0.1,S)
perturbed_state = [abs(m+n) for m,n in zip(finalstate1, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=5, EPS=EPS,**kwargs ):

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
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state2 = [abs(m+n) for m,n in zip(finalstate2, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=5, EPS=EPS,**kwargs ):

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

#####################################################################################################
#####################################################################################################

finalstate3 = [m for m in trajectories3[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state3 = [abs(m+n) for m,n in zip(finalstate3, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=5, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
        dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        return dx

    x0=perturbed_state3
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time4,trajectories4 = run(S)

t4 = [m+t3[-1] for m in time4]

finalstate4 = [m for m in trajectories4[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state4 = [abs(m+n) for m,n in zip(finalstate4, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=5, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
        dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        return dx

    x0=perturbed_state4
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time5,trajectories5 = run(S)

t5 = [m+t4[-1] for m in time5]

#####################################################################################################
#####################################################################################################

finalstate5 = [m for m in trajectories5[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state5 = [abs(m+n) for m,n in zip(finalstate5, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=5, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
        dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        return dx

    x0=perturbed_state5
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time6,trajectories6 = run(S)

t6 = [m+t5[-1] for m in time6]

finalstate6 = [m for m in trajectories6[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state6 = [abs(m+n) for m,n in zip(finalstate6, perturbation)]

#####################################################################################################
#####################################################################################################

for i in range(len(trajectories)):
    plt.plot(time,trajectories[i]) 
for i in range(len(trajectories2)):
    plt.plot(t2, trajectories2[i])
for i in range(len(trajectories3)):
    plt.plot(t3, trajectories3[i])
for i in range(len(trajectories4)):
    plt.plot(t4, trajectories4[i])
for i in range(len(trajectories5)):
    plt.plot(t5, trajectories5[i])    
for i in range(len(trajectories6)):
    plt.plot(t6, trajectories6[i])  
    
plt.show()
