# Let us consider an airport with total number of passengers = 89170/day. We are considering the passenger departure process that consists of 4 stages
# i.e. Booking, Check-in, Security, Boarding. For each stage let's consider the probability of detecting fraud passengers as p1, p2, p3, p4 resp. Here,
# we are assuming that once a passenger is detected fraud, he can not re-enter the process. But since every passenger is not threat, so we will consider
# that 1% of the total passengers are threat, hence simulation will be done only on 891.70~892 passengers

# Now, we define efficiency of the process as its capability to detect more and more number of fraud passengers.
# Mathematically, we define it as [1-(1-p1)*(1-p2)*(1-p3)*(1-p4)]*100 percent.

# Firsly, we are considering a case where baseline proabilities of detection are given. We then calculate the efficiency. Now, we ask the question:
# Given the cost associated with change in probability, C_i for each stage i, If we want to increase the efficiency by say x percent,
# what should be the corresponding probabilities of detection at each stage?

import random
import math
import numpy as np
from pulp import *

n = int(input('Enter the initial number of iterations'))

d_threshold =[0.005,0.003,0.001,0.002]   # to calculate the number of simulation runs  required, we need an acceptable value of the standard deviation of the estimator

def generate_probabilities(n):
    p = [0.1,0.2,0.9,0.4]           #baseline proabilities of detection at each stage
    efficiency = (1-(1-p[0])*(1-p[1])*(1-p[2])*(1-p[3]))*100
    print('Initial Efficiency = ',efficiency)

    c1 = 20     # cost associated with change in p1 = p[0]
    c2 = 40     # cost associated with change in p2 = p[1]
    c3 = 70     # cost associated with change in p3 = p[2]
    c4 = 10     # cost associated with change in p4 = p[3]

    increment = 0.01   # required increase in efficiency
    print("Required increase in efficiency = ",increment*100,"%")

    probability = [i for i in range(1,5)]
    prob_vector = [p]
    cost_vector = []

#step 1 : Booking Tickets

    for h in range(n):
        p = [0.1,0.2,0.9,0.4]
        efficiency = 1-(1-p[0])*(1-p[1])*(1-p[2])*(1-p[3])
        
        a = np.random.randint(1,4)       # random integer,a,chosen between {1,..4} to work on p_a
        
        prob = LpProblem("DES",LpMinimize)
        x = LpVariable.dicts("x",probability,0,1,LpContinuous)

        prob += c1*x[1] + c2*x[2] + c3*x[3] + c4*x[4]       # total cost associated with the change in p_a
        alpha = np.random.uniform(0,1)

        if a == 1:
            prob += 1-(1-p[0]-x[1])*(1-p[1])*(1-p[2])*(1-p[3]) ==  efficiency +increment*alpha
            efficiency = efficiency + increment*alpha
        elif a == 2:
            prob += 1-(1-p[0])*(1-p[1]-x[2])*(1-p[2])*(1-p[3]) == efficiency +increment*alpha
            efficiency = efficiency + increment*alpha
        elif a == 3:
            prob += 1-(1-p[0])*(1-p[1])*(1-p[2]-x[3])*(1-p[3]) == efficiency +increment*alpha
            efficiency = efficiency + increment*alpha
        else:
            prob += 1-(1-p[0])*(1-p[1])*(1-p[2])*(1-p[3]-x[4]) == efficiency +increment*alpha
            efficiency = efficiency + increment*alpha

        prob.solve()
        
        if(LpStatus[prob.status] == 'Infeasible'):
            continue

        p_change = []  # stores the x_i != 0 i.e. the change in p_i if it is non-zero

        for v in prob.variables():
            if v.varValue != 0 :
                p_change.append(v.varValue)

        for i in range(1,5):     # updates the p_i
            if a == i:
                p[a-1] = p[a-1]+p_change[0]
            else:
                p[i-1] = p[i-1]

#step 2 : Check-in
        b = np.random.randint(1,4)          # random integer,b (not equals to a),chosen between {1,..4} to work on p_b
        while b == a:    
            b = np.random.randint(1,4)

        prob1 = LpProblem("DES1",LpMinimize)
        y = LpVariable.dicts("y",probability,0,1,LpContinuous)

        prob1+= c1*y[1] + c2*y[2] + c3*y[3] + c4*y[4]    # total cost associated with the change in p_b

        beta = np.random.uniform(0,1-alpha)

        if b == 1:
            prob1 += 1-(1-p[0]-y[1])*(1-p[1])*(1-p[2])*(1-p[3]) ==  efficiency+ increment*beta
            efficiency = efficiency + increment*beta
        elif b == 2:
            prob1 += 1-(1-p[0])*(1-p[1]-y[2])*(1-p[2])*(1-p[3]) ==  efficiency+ increment*beta
            efficiency = efficiency + increment*beta
        elif b == 3:
            prob1 += 1-(1-p[0])*(1-p[1])*(1-p[2]-y[3])*(1-p[3]) ==  efficiency+ increment*beta
            efficiency = efficiency + increment*beta
        else:
            prob1 += 1-(1-p[0])*(1-p[1])*(1-p[2])*(1-p[3]-y[4]) == efficiency +increment*beta
            efficiency = efficiency + increment*beta

        prob1.solve()
        if(LpStatus[prob1.status]=='Infeasible'):
            continue

        p_change_1 = []       # stores the x_i != 0 i.e. the change in p_i if it is non-zero

        for v in prob1.variables():
            if v.varValue != 0 :
                p_change_1.append(v.varValue)

        for i in range(1,5):     # updates the p_i
            if b == i:
                p[b-1] = p[b-1] + p_change_1[0]
            else:
                p[i-1] = p[i-1]

#step 3 : Security check
        c = np.random.randint(1,4)    # random integer,c (not equals to a and b),chosen between {1,..4} to work on p_c
        while c == a or c == b:
            c = np.random.randint(1,4)

        prob2 = LpProblem("DES2",LpMinimize)
        z = LpVariable.dicts("z",probability,0,1,LpContinuous)

        prob2 += c1*z[1] + c2*z[2] + c3*z[3] + c4*z[4]       # total cost associated with the change in p_c

        gamma = np.random.uniform(0,1-alpha-beta)

        if c == 1:
            prob2 += 1-(1-p[0]-z[1])*(1-p[1])*(1-p[2])*(1-p[3]) ==  efficiency+ increment*gamma
            efficiency = efficiency + increment*gamma
        elif c == 2:
            prob2 += 1-(1-p[0])*(1-p[1]-z[2])*(1-p[2])*(1-p[3]) ==  efficiency+ increment*gamma
            efficiency = efficiency + increment*gamma
        elif c == 3:
            prob2 += 1-(1-p[0])*(1-p[1])*(1-p[2]-z[3])*(1-p[3]) ==  efficiency+ increment*gamma
            efficiency = efficiency + increment*gamma
        else:
            prob2 += 1-(1-p[0])*(1-p[1])*(1-p[2])*(1-p[3]-z[4]) == efficiency +increment*gamma
            efficiency = efficiency + increment*gamma

        prob2.solve()
        
        if(LpStatus[prob2.status] == 'Infeasible'):
            continue

        p_change_2 = []     # stores the x_i != 0 i.e. the change in p_i if it is non-zero

        for v in prob2.variables():
            if v.varValue != 0 :
                p_change_2.append(v.varValue)

        for i in range(1,5):        # updates the p_i
            if c == i:
                p[c-1] = p[c-1]+p_change_2[0]
            else:
                p[i-1] = p[i-1]

#step 4 : Boarding
        for i in range(1,5):         # random integer,d (not equals to a,b and c),chosen between {1,..4} to work on p_d
            if i != a and i != b and i != c:
                d = i

        prob3 = LpProblem("DES3",LpMinimize)
        w = LpVariable.dicts("w",probability,0,1,LpContinuous)

        prob3 += c1*w[1]+c2*w[2]+c3*w[3]+c4*w[4]        # total cost associated with the change in p_d

        delta = 1-alpha-beta-gamma

        if d == 1:
            prob3 += 1-(1-p[0]-w[1])*(1-p[1])*(1-p[2])*(1-p[3]) ==  efficiency+ increment*delta
            efficiency = efficiency + increment*delta
        elif d == 2:
            prob3 += 1-(1-p[0])*(1-p[1]-w[2])*(1-p[2])*(1-p[3]) ==  efficiency+ increment*delta
            efficiency = efficiency + increment*delta
        elif d == 3:
            prob3 += 1-(1-p[0])*(1-p[1])*(1-p[2]-w[3])*(1-p[3]) ==  efficiency+ increment*delta
            efficiency = efficiency + increment*delta
        else:
            prob3 += 1-(1-p[0])*(1-p[1])*(1-p[2])*(1-p[3]-w[4]) == efficiency +increment*delta
            efficiency = efficiency + increment*delta

        prob3.solve()

        if(LpStatus[prob3.status]=='Infeasible'):
            continue

        p_change_3 = []      # stores the x_i != 0 i.e. the change in p_i if it is non-zero

        for v in prob3.variables():
            if v.varValue != 0 :
                p_change_3.append(v.varValue)

        for i in range(1,5):        # updates the p_i
            if d==i:
                p[d-1] = p[d-1]+p_change_3[0]
            else:
                p[i-1] = p[i-1]

        total_cost = value(prob.objective) + value(prob1.objective)+value(prob2.objective) + value(prob3.objective)

        prob_vector.append(p)   # append feasible (p1,p2,p3,p4) that gives desired efficiency
        cost_vector.append(total_cost)     # append the cost associated with new (p1,p2,p3,p4)
    return [prob_vector,cost_vector]

prob_gen = generate_probabilities(n)
prob_vector=prob_gen[0]

# Now, once these probabilities are generated, we will run our code until we get S/(no.of runs)^0.5 < d_threshold for each stage, 
# where S is the standard deviation of the probabilities obtained from above for each individual stage.

#Firstly S/(no.of runs)^0.5 is calculated for each stage.
def d_generation(prob_vector):
    p_1_only=[]
    for i in range(len(prob_vector)):
        p_1_only.append(prob_vector[i][0])
    std1=np.std(p_1_only)
    d1_check=n*std1/((n-1)*np.sqrt(n))
    #print('d1_check',d1_check)

    p_2_only=[]
    for i in range(len(prob_vector)):
        p_2_only.append(prob_vector[i][1])
    std2=np.std(p_2_only)
    d2_check=n*std2/((n-1)*np.sqrt(n))

    p_3_only=[]
    for i in range(len(prob_vector)):
        p_3_only.append(prob_vector[i][2])
    std3=np.std(p_3_only)
    d3_check=np.sqrt(n)*std3/(n-1)

    p_4_only=[]
    for i in range(len(prob_vector)):
        p_4_only.append(prob_vector[i][3])
    std4=np.std(p_4_only)
    d4_check=np.sqrt(n)*std4/(n-1)
    return [d1_check,d2_check,d3_check,d4_check]

d = d_generation(prob_vector)
d1 = d[0]
d2 = d[1]
d3 = d[2]
d4 = d[3]

# Now, the above mentioned condition is checked and code is run till we meet the required condition.

while d1 >= d_threshold[0] or d2 >= d_threshold[1] or d3 >= d_threshold[2] or d4 >= d_threshold[3]:
    n=n+1
    prob_gen = generate_probabilities(n)
    prob_vector=prob_gen[0]
    d = d_generation(prob_vector)
    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    d4 = d[3]
print('Total number of simulation runs required = ', n)
index_min_cost = 0       # the index of the (p1,p2,p3,p4) which gives minimum cost for desired efficiency
for i in range(len(prob_gen[1])):
    if prob_gen[1][i] == min(prob_gen[1]):
        index_min_cost += i
prob_min_cost = prob_gen[0][index_min_cost+1]
print("2)   (p1,p2,p3,p4) which gives minimum cost is ",prob_min_cost," and the cost associated with the change is ",min(prob_gen[1]))

# Secondly, we address the question of actually verifying the hypothesis made i.e. the above (p1,p2,p3,p4)
# which we get gives us desired efficiency. To do this, we filter the number of passengers detected at each stage i 
# according to p_i and at the end, we calculate (1-number of passengers not detected at boarding)*100. If this equals 
# the desired efficiency, then we say that our hypothesis is true.

# Assumption: Number of passengers detected at stage i ~ binomial(N,1-p_i)

# Scenario 1 : When all the passengers are mainstream passengers i.e. no transit passengers

N = 0.01*89170     # Total number of passengers at the airport
run = n   # Number of runs in simulation
def airport_no_transit(N,run,p):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]
    
# step 1 : booking
    a = []  #stores number of not detected items

    for i in range(0,run):        
        a.append(np.random.binomial(N,1-p1,1))

    not_detected_booking = np.mean(a)

    detected_booking = N - not_detected_booking

# step 2 : check-in
    b = []  #stores number of not detected items

    for i in range(0,run):
        b.append(np.random.binomial(not_detected_booking,1-p2,1))

    not_detected_checkin = np.mean(b)

    detected_checkin = not_detected_booking-not_detected_checkin

# step 3 : security_checkpoint:
    c = []  #stores number of not detected items

    for i in range(0,run):
        c.append(np.random.binomial(not_detected_checkin,1-p3,1))

    not_detected_security = np.mean(c)

    detected_security = not_detected_checkin-not_detected_security

# step 4: boarding gate
    d = []  #stores number of not detected items

    for i in range(0,run):
        d.append(np.random.binomial(not_detected_security,1-p4,1))

    not_detected_boarding = np.mean(d)

    detected_boarding = not_detected_security-not_detected_boarding
 
    x = [not_detected_booking,not_detected_checkin,not_detected_security,not_detected_boarding]
    y = [detected_booking,detected_checkin,detected_security,detected_boarding]
    return [x,y]


airport_no_transit(N,run,prob_min_cost)
x = airport_no_transit(N,run,prob_min_cost)
print("In scenario 1 i.e. Mainstream Passengers, when p= ",prob_min_cost,": ")
print("3)   Efficiency for mainstream passengers in scenario 1 is ", 100-(x[0][3]/N)*100)
print("4)   Undetected passengers out of ",N," passengers are ",x[0])
print("5)   Detected passengers out of ",N," passengers are ",x[1])

# Scenario 2 : When there is a fraction of transit passengers at the airport

proportion_transit_passengers = 0.20  # Proportion of transit passengers outr of N passengers
N_non_transit = int((1-proportion_transit_passengers)*N)
N_transit = N - N_non_transit

airport_no_transit(N_non_transit,run,prob_min_cost)
x = airport_no_transit(N_non_transit,run,prob_min_cost)
print("In scenario 2 i.e. Transit Passengers, when p= ",prob_min_cost,": ")
print("6)   Efficiency for non-transit passengers is ", 100-(x[0][3]/N_non_transit)*100)
print("7)   Undetected non-transit passengers out of ",N_non_transit," non-transit passengers are ",x[0])
print("8)   Detected non-transit passengers out of ",N_non_transit," non-transit passengers are ",x[1])

# Now, we define efficiency of the process for transit passengers as its capability to detect more and more number of fraud transit passengers.
# Mathematically, we define it as [1-(1-p3)*(1-p4)]*100 percent because transit passengers donot book tickets or check-in at the airport in question.

# So now, we are considering a case where baseline proabilities of detection are given. We then calculate the efficiency. Now, we ask the question:
# Given the cost associated with change in probability, C_i for each stage i (security and boarding), If we want to increase the efficiency by say x percent,
# what should be the corresponding probabilities of detection at each stage(security and booking)?

n = int(input('Enter the initial number of iterations'))

d_threshold =[0.001,0.002]    # to calculate the number of simulation runs  required, we need an acceptable value of the standard deviation of the estimator
    
def generate_probabilities_transit(n):
    p = [0.1,0.2,0.9,0.4]      #baseline proabilities of detection at each stage
    efficiency = (1-(1-p[2])*(1-p[3]))*100
    print('Initial Efficiency = ',efficiency)

    c3 = 70     # cost associated with change in p3
    c4 = 10     # cost associated with change in p4

    increment = 0.01     # required increase in efficiency
    print("Required increase in efficiency = ",increment*100,"%")

    probability = [i for i in range(1,3)]
    prob_vector = [p]
    cost_vector = []

#step 1 : Security
    for h in range(n):
        p = [0.9,0.4]
        efficiency = 1-(1-p[0])*(1-p[1])
        
        a = np.random.randint(1,2)    # random integer,a,chosen between {1,..4} to work on p_a
        
        prob = LpProblem("DES",LpMinimize)
        x = LpVariable.dicts("x",probability,0,1,LpContinuous)

        prob += c3*x[1] + c4*x[2]         # total cost associated with the change in p_a
        alpha=np.random.uniform(0,1)
        
        if a == 1:
            prob += 1-(1-p[0]-x[1])*(1-p[1]) == efficiency + increment*alpha
            efficiency = efficiency + increment*alpha
        else:
            prob += 1-(1-p[0])*(1-p[1]-x[2]) == efficiency + increment*alpha
            efficiency = efficiency + increment*alpha

        prob.solve()
        
        if(LpStatus[prob.status] == 'Infeasible'):
            continue

        p_change = []    # stores the x_i != 0 i.e. the change in p_i if it is non-zero

        for v in prob.variables():
            if v.varValue != 0 :
                p_change.append(v.varValue)

        for i in range(1,3):       # updates the p_i
            if a == i:
                p[a-1] = p[a-1] + p_change[0]
            else:
                p[i-1] = p[i-1]

#step 2 : Boarding
    for i in range(1,3):     # random integer,b (not equals to a),chosen between {1,..4} to work on p_b
        if i != a:
            b = i

        prob1 = LpProblem("DES1",LpMinimize)
        y = LpVariable.dicts("y",probability,0,1,LpContinuous)

        prob1 += c3*y[1] + c4*y[2]       # total cost associated with the change in p_b

        beta = 1-alpha

        if b == 1:
            prob1 += 1-(1-p[0]-y[1])*(1-p[1]) ==  efficiency + increment*beta
            efficiency = efficiency + increment*beta
        else:
            prob1 += 1-(1-p[0])*(1-p[1]-y[2]) == efficiency + increment*beta
            efficiency = efficiency + increment*beta

        prob1.solve()
        
        if(LpStatus[prob1.status] == 'Infeasible'):
            #print("infeasible")
            continue

        p_change_1 = []       # stores the x_i != 0 i.e. the change in p_i if it is non-zero

        for v in prob1.variables():
            if v.varValue != 0 :
                p_change_1.append(v.varValue)

        for i in range(1,3):       # updates the p_i
            if b == i:
                p[b-1] = p[b-1] + p_change_1[0]
            else:
                p[i-1] = p[i-1]

        total_cost = value(prob.objective) + value(prob1.objective)

        prob_vector.append(p)         # append feasible (p3,p4) that gives desired efficiency
        cost_vector.append(total_cost)         # append the cost associated with new (p1,p2,p3,p4)
    return [prob_vector,cost_vector]

prob_gen = generate_probabilities_transit(n)
prob_vector=prob_gen[0]

# Now, once these probabilities are generated, we will run our code until we get S/(no.of runs)^0.5 < d_threshold for each stage, 
# where S is the standard deviation of the probabilities obtained from above for each individual stage.

#Firstly S/(no.of runs)^0.5 is calculated for each stage.

def d_generation_transit(prob_vector):
    p_1_only=[]
    for i in range(len(prob_vector)):
        p_1_only.append(prob_vector[i][0])
    std1=np.std(p_1_only)
    d1_check=n*std1/((n-1)*np.sqrt(n))
    #print('d1_check',d1_check)

    p_2_only=[]
    for i in range(len(prob_vector)):
        p_2_only.append(prob_vector[i][1])
    std2=np.std(p_2_only)
    d2_check=n*std2/((n-1)*np.sqrt(n))
    #print('d2_check',d2_check)
    return [d1_check,d2_check]

d = d_generation_transit(prob_vector)
d1 = d[0]
d2 = d[1]

# Now, the above mentioned condition is checked and code is run till we meet the required condition.

while d1 >= d_threshold[0] or d2 >= d_threshold[1]:
    n=n+1
    prob_gen = generate_probabilities_transit(n)
    prob_vector=prob_gen[0]
    d = d_generation_transit(prob_vector)
    d1 = d[0]
    d2 = d[1]
print('Total number of simulation runs required = ', n)
index_min_cost = 0            # the index of the (p1,p2,p3,p4) which gives minimum cost for desired efficiency
for i in range(len(prob_gen[1])):
    if prob_gen[1][i] == min(prob_gen[1]):
        index_min_cost += i
prob_min_cost = prob_gen[0][index_min_cost+1]
print("9)   (p3,p4) which gives minimum cost is ",prob_min_cost," and the cost associated with the change is ",min(prob_gen[1]))

# Now, we address the question of actually verifying the hypothesis made i.e. the above (p3,p4)
# which we get gives us desired efficiency. To do this, we filter the number of passengers detected at each stage i(security 
# and boarding) according to p_i and at the end, we calculate (1-number of passengers not detected at boarding)*100. 
# If this equals the desired efficiency, then we say that our hypothesis is true.

# Assumption: Number of passengers detected at stage i ~ binomial(N,1-p_i)

def airport_transit(N,run,p):
    p3 = p[0]
    p4 = p[1]
# stage 1 : security_checkpoint:
    c = []  #stores number of not detected items

    for i in range(0,run):
        c.append(np.random.binomial(N,1-p3,1))

    not_detected_security = np.mean(c)

    detected_security = N - not_detected_security

# stage 2 : boarding gate
    d = []  #stores number of not detected items

    for i in range(0,run):
        d.append(np.random.binomial(not_detected_security,1-p4,1))

    not_detected_boarding = np.mean(d)

    detected_boarding = not_detected_security-not_detected_boarding

    x = [not_detected_security,not_detected_boarding]
    y = [detected_security,detected_boarding]
    return [x,y]

run = n
airport_transit(N_transit,run,prob_min_cost)
x = airport_transit(N_transit,run,prob_min_cost)

print("In scenario 2, when p= ",prob_min_cost,": ")
print("10)   Efficiency for non-transit passengers is ", 100-(x[0][1]/N_transit)*100)
print("11)   Undetected transit passengers out of ",N_transit," transit passengers are ",x[0])
print("12)   Detected transit passengers out of ",N_transit," transit passengers are ",x[1])
