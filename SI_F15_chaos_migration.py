import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# THIS CODE EXPLORES, FOR BOTH THE ALLEE EFFECT AND THE GENERALIZED LOTKA-VOLTERRA MODEL, HOW THE FRACTION OF CHAOTIC
# SIMULATIONS DECAYS IN TIME, FOR SYSTEMS WITH AND WITHOUT EXTERNAL MIGRATION, TO OBSERVE HOW MIGRATION ALLOWS 
# FLUCTUATIONS TO PERSIST (Roy et al., 2020)
# G AGUADÉ-GORGORIÓ

EPS=10.**-10 #Threshold of immigration

S = 50

SI_threshold = 0.05

systems = 800 # 100 of each

temps= 1000
extra = 50

passes = 20


GLV = np.zeros(passes)
GLVnom = np.zeros(passes)
AE = np.zeros(passes)
AEnom = np.zeros(passes)

repsGLV = 0
repsAE = 0
repsGLVnom = 0
repsAEnom = 0

for sys in range(systems):
    
    print(sys, "out of ", systems)
    #####################################################################################################
    #####################################################################################################
    
    choose = np.random.randint(1,5)
    if choose == 1: #GLV
        repsGLV+=1.0            
    elif choose == 2: #GLV NO MIGRATION
        repsGLVnom+=1.0       
    elif choose == 3: #AE
        repsAE+=1.0                  
    elif choose == 4: #AE NO MIGRATION
        repsAEnom+=1.0
                
    if choose == 1 or choose==2: #GLV
        meanA = -0.07
        varA = 0.05
        meand = 0.25
        vard = 1.1 
        d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=S)
        A=np.random.normal(meanA, varA, size=(S,S))    
        np.fill_diagonal(A,-d)
        
    elif choose == 3 or choose==4: #AE
        varintra = 1.1
        varinter = 2.0
        meand=0.1
        vard=varintra 
        meang=1.0  
        varg=varintra
        meanAii=0.5
        varAii=varintra
        meanBii = 0.1
        varBii= varintra
        d=np.random.lognormal(mean=np.log(meand), sigma=np.log(vard), size=S)
        g=np.random.lognormal(mean=np.log(meang), sigma=np.log(varg), size=S)
        Aii=np.random.lognormal(mean=np.log(meanAii), sigma=np.log(varAii), size=S)
        Bii=np.random.lognormal(mean=np.log(meanBii), sigma=np.log(varBii), size=S)
        varA = varinter
        varB = varinter  
        meanA = 0.5
        meanB = 0.05
        A=np.random.lognormal(mean=np.log(meanA), sigma=np.log(varA), size=(S,S)) 
        np.fill_diagonal(A,Aii)
        B=np.random.lognormal(mean=np.log(meanB), sigma=np.log(varB), size=(S,S)) 
        np.fill_diagonal(B,Bii)
                  
       

    for p in range(passes):
        if p==0:
            init = np.random.random(S)
        else: 
            init = finalstate
        
        def run(S, tmax=temps, EPS=EPS,**kwargs):
            if choose == 1: #GLV
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x))
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
            elif choose == 2: #GLV NO MIGRATION
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x))
                    return dx
            elif choose == 3: #AE
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx            
            elif choose == 4: #AE NO MIGRATION
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    return dx
            x0=init
            sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
            time=sol.t
            trajectories=sol.y
            return time, trajectories

        time,trajectories = run(S)
        finalstate = [m for m in trajectories[:,-1]] 
        
        # EXTRA TIME:
           
        init = finalstate
        def run(S, tmax=extra, EPS=EPS,**kwargs):
            if choose == 1: #GLV
                def eqs(t,x):
                    dx = x*np.dot(A,x)
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx
            elif choose == 2: #GLV NO MIGRATION
                def eqs(t,x):
                    dx = x*np.dot(A,x)
                    return dx
            elif choose == 3: #AE
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    dx[x<=EPS]=np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
                    return dx            
            elif choose == 4: #AE NO MIGRATION
                def eqs(t,x):
                    dx = (x*np.dot(A,x/(g + x)) ) - (d*x) - (x*(np.dot(B,x)))
                    return dx
            x0= init
            sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
            time=sol.t
            trajectories=sol.y
            return time, trajectories

        timeplus,trajectoriesplus = run(S)
        finalstateplus = [m for m in trajectoriesplus[:,-1]] 
        #COMPARE: STABLE OR VARYING?
        
        diff = 0.0
        for spp in range(S):
            if abs ( finalstate[spp] - finalstateplus[spp] ) > SI_threshold:
                diff += 1
                break
                            
        if diff > 0: # if two species have been deemed different, this state is not stable!
            if choose == 1: #GLV
                GLV[p] +=1 
            elif choose == 2: #GLV NO MIGRATION
                GLVnom[p]+=1
            elif choose == 3: #AE
                AE[p] +=1
            elif choose == 4: #AE NO MIGRATION   
                AEnom[p] +=1
        
        
Time = []

for p in range(passes):
    Time.append((p+1)*1000)
    GLV[p] = GLV[p]/repsGLV
    GLVnom[p] = GLVnom[p]/repsGLVnom
    AE[p] = AE[p]/repsAE
    AEnom[p] = AEnom[p]/repsAEnom
    
plt.plot(Time,GLV, color="gray", alpha=1)
plt.plot(Time,GLVnom, color="blue", alpha=1)
plt.plot(Time,AE, color="black", alpha=1)
plt.plot(Time,AEnom, color="red", alpha=1) 

plt.show()

plt.plot(Time,GLV, color="gray", alpha=1)
plt.plot(Time,GLVnom, color="blue", alpha=1)
plt.plot(Time,AE, color="black", alpha=1)
plt.plot(Time,AEnom, color="red", alpha=1) 
nom = "unst.png"
plt.savefig(nom, format='png')
plt.close()


#LISTS AS COLUMNS

rows = zip(Time, GLV, GLVnom, AE, AEnom)

import csv

with open('chaos.csv', 'w') as f:
    writer = csv.writer(f)
    row1 = ["T", "GLV", "GLVnom", "AE", "AEnom"] 
    writer.writerow(row1)
    for row in rows:
        writer.writerow(row)




exit()

    
