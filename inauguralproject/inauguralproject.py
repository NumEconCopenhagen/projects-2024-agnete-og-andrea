from types import SimpleNamespace
import numpy as np
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ExchangeEconomyClass():

    # MODEL SET UP

    '''
    We initialize the model, set baseline parameters and create solution variables. 
    We also define the utility and demand functions for both consumers. 
    We define the market clearing conditions.
    
    '''

    def __init__(self):

        par = self.par = SimpleNamespace()

        par.alpha = 1/3
        par.beta = 2/3

        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

        sol = self.sol = SimpleNamespace()

        sol.x1 = np.nan
        sol.x2 = np.nan
        sol.p = np.nan

    def utility_A(self,x1A,x2A):

        par = self.par 

        return x1A** par.alpha * x2A** (1- par.alpha)

    def utility_B(self,x1B,x2B):

        par = self.par

        return x1B ** par.beta * x2B **(1-par.beta)
    

    def demand_A(self,p1):

        par = self.par 

        if not isinstance(p1, (int, float)):
            p1 = float(p1)

        x1A = par.alpha * ((p1*par.w1A+par.w2A)/p1)  
        x1A = max(0, min(1, x1A))  
        x1B = 1 - x1A
        x2A = (1-par.alpha) * (p1*par.w1A+par.w2A)
        x2B = max(0, min(1,1 - x2A))  

        return np.array([x1A, x2A])

    def demand_B(self,p1):

        par = self.par

        if not isinstance(p1, (int, float)):
            p1 = float(p1)

        x1B = par.beta * ((p1*par.w1B+par.w2B)/p1)
        x1B = max(0, min(1, x1B))  
        x2B = (1 - par.beta) * (p1*par.w1B+par.w2B)
        x2A = max(0, min(1,1 - x2B))  

        return np.array([x1B, x2B])

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1, eps2
    
    # QUESTION 1

    '''
    1. 
    We define the C set by first setting N = 75 and creating the ranges for x1A and x2A. 
    We then create the other three constraints that the set needs to satisfy. 
    We ensure that the method returns the values for x1A and x2A that are in the set.
    2. 
    We create and plot the figure with all allocations in the set C as well as a dot with the initial endowments.

    '''

    def Cset(self):

        par = self.par
        par.N = 75

        x1A_range = np.arange(0, 1 + 1/par.N, 1/par.N)
        x2A_range = np.arange(0, 1 + 1/par.N, 1/par.N)

        logical_condition_1 = lambda x1A, x2A: self.utility_A(x1A, x2A) >= self.utility_A(par.w1A, par.w2A)
        logical_condition_2 = lambda x1B, x2B: self.utility_B(x1B, x2B) >= self.utility_B(par.w1B, par.w2B)
        logical_condition_3 = lambda x1A, x1B: x1B == 1 - x1A
        logical_condition_4 = lambda x2A, x2B: x2B == 1 - x2A

        values = [(x1A, x2A) for x1A in x1A_range for x2A in x2A_range if logical_condition_1(x1A, x2A) and logical_condition_2(1-x1A, 1-x2A) and logical_condition_3(x1A, 1-x1A) and logical_condition_4(x2A, 1-x2A)]

        return values
    
    def edgeworth(self):

        par = self.par

        w1bar = 1.0
        w2bar = 1.0

        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment') 
        xes = [] 
        yes = [] 
        for x in self.Cset():
            xes.append(x[0]) 
            yes.append(x[1])
        ax_A.scatter(xes,yes,marker="s", color='green',label='Pareto Pairs')

        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0));
          
    
    # QUESTION 2

    '''
    1.
    We define the P1 set as a range for p1.
    2. 
    We define the errors for all p1 in P1 from the existing market clearing method.
    
    '''

    def P1set(self):

        par = self.par

        p1_range = [] 
        for i in range(par.N+1):
            p1_range.append(0.5 + 2*i/par.N)

        return p1_range
    
    def Q2errors(self, p1_range):

        errors = []

        for p1 in p1_range:

            self.eps1, self.eps2 = self.check_market_clearing(p1)
            errors.append((self.eps1, self.eps2))

        return errors
    
    # QUESTION 3

    '''
    We use the Nelder-Mead method to find the p1 that minimizes the errors in the market clearing condition, 
    resulting in the market clearing price.
    
    '''

    def market_clearing_price(self):

        sol = self.sol

        obj_fun = lambda x: np.sum(np.abs(self.check_market_clearing(x)))
        init_p = [1.0]

        res = optimize.minimize(obj_fun, init_p, method='Nelder-Mead')

        p1 = res.x[0]
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        sol.p = p1
        sol.x1 = x1A
        sol.x2 = x2A

        print(f'The market clearing price is {sol.p:.3f}')


    # QUESTION 4A 

    '''
    We maximize utility for A in the already defined set P1 by creating a for loop.
    In the for loop, the allocation for consumer A is defined from consumer B's demand. 
    We set the natural bounds for x1A and x2A, and then find the utility maximizing allocation.  

    '''

    def optimal_allocation_4a(self, print_output=True):

        p1_range = self.P1set() 

        max_utility = float('-inf')

        optimal_allocation = None

        for p1 in p1_range:

            x1A, x2A = 1-self.demand_B(p1)  
            
            if 0 <= x1A <= 1 and 0 <= x2A <= 1:

                utility = self.utility_A(x1A, x2A)

                if utility > max_utility:

                    max_utility = utility
                    optimal_allocation = (x1A, x2A)

        x1A_rounded = round(optimal_allocation[0], 3)
        x2A_rounded = round(optimal_allocation[1], 3)

        if print_output:
            print("The optimal allocation for question 4a is x1A, x2A:")

        return x1A_rounded, x2A_rounded
     

    # QUESTION 4B

    '''
    1.
    We create a similar for loop, where the range for p1 is unrestricted (set to 1000). 
    The allocation is again set from consumer B's demand with the same bounds. 
    2. 
    We find the optimal allocation for A given the found optimal price for good 1 and consumer B's demand
    with p2 set as numeraire.  
    
    '''

    def utility_maximization_4b(self):

        max_utility = float('-inf')

        p1_opt_4b = None

        for p1 in range(1, 1000):

            x1A, x2A = 1-self.demand_B(p1)

            if 0 <= x1A <= 1 and 0 <= x2A <= 1:

                utility = self.utility_A(x1A, x2A)

                if utility > max_utility:

                    max_utility = utility
                    p1_opt_4b = p1

            return p1_opt_4b, 1   

    def optimal_allocation_4b(self, print_output=True):

        p1_4b, _ = self.utility_maximization_4b()
        x1A_un, x2A_un = 1-self.demand_B(p1_4b)

        x1A_rounded = round(x1A_un, 3)
        x2A_rounded = round(x2A_un, 3)

        if print_output:
            print('Optimal allocation for question 4b is x1A, x2A:')

        return x1A_rounded, x2A_rounded


    # QUESTION 5A

    '''
    We maximize consumer A's utility by creating a for loop for values for x1A and x2A in set C 
    and returning the optimal values. To do this we call on the already defined set C. 
    '''
    
    def utility_maximization_5a(self):

        set_C = self.Cset()

        max_utility = float('-inf')

        x1A_optimal, x2A_optimal = None, None

        for x1A, x2A in set_C:

            utility_A = self.utility_A(x1A, x2A)

            if utility_A > max_utility:
                
                max_utility = utility_A
                x1A_optimal, x2A_optimal = x1A, x2A
                
        return x1A_optimal, x2A_optimal
    
    def optimal_allocation_5a(self, print_output=True):

        x1A_optimal, x2A_optimal = self.utility_maximization_5a()

        x1A_rounded = round(x1A_optimal, 3)
        x2A_rounded = round(x2A_optimal, 3)

        if print_output:
            print("Optimal allocation for question 5a is x1A, x2A:")

        return x1A_rounded, x2A_rounded

    # QUESTION 5B

    '''
    We maximize consumer A's utility with scipy optimize minimize, and thus use negative utility function.
    We define the objective function, constraints and bounds as stated in question 5b and use endownments as initial guess.
    '''

    def utility_maximization_5b(self):
        
        par = self.par
        sol = self.sol

        def obj_fun(x):
            x1A, x2A = x
            return -self.utility_A(x1A, x2A) 

        def constraint(x):
            x1A, x2A = x
            x1B = 1 - x1A
            x2B = 1 - x2A
            return self.utility_B(x1B, x2B) - self.utility_B(par.w1B, par.w2B)

        bounds = [(0, 1), (0, 1)]
        initial_guess = [par.w1A, par.w2A]
        constraints = [{'type': 'ineq', 'fun': constraint}]

        result = minimize(obj_fun, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        x1A_optimal = sol.x1 = result.x[0]
        x2A_optimal = sol.x2 = result.x[1]

        return x1A_optimal, x2A_optimal
    
    def optimal_allocation_5b(self, print_output=True):
        
        x1A_optimal, x2A_optimal = self.utility_maximization_5b()
        x1A_rounded = round(x1A_optimal, 3)
        x2A_rounded = round(x2A_optimal, 3)

        if print_output:
            print("The optimal allocation for question 5b is x1A, x2A:")

        return x1A_rounded, x2A_rounded

    # QUESTION 6A

    '''
    We maximize the aggregated utility by creating two for loops in the appropriate ranges for x1A and x2A.
    For simplicity, we only solve for x1A, x2A as we know that the total endowments are equal to one. 
    '''
 
    def utility_maximization_6a(self):

        max_utility = float('-inf')

        x1A_optimal, x2A_optimal = None, None

        for x1A in np.linspace(0, 1, 100):

            for x2A in np.linspace(0, 1, 100):

                utility_6a = self.utility_A(x1A, x2A)+self.utility_B(1-x1A,1-x2A)
                
                if utility_6a > max_utility:

                    max_utility = utility_6a

                    x1A_optimal, x2A_optimal = x1A, x2A

        return x1A_optimal, x2A_optimal
    
    def optimal_allocation_6a(self):

        x1A_optimal, x2A_optimal = self.utility_maximization_6a()
        x1A_rounded = round(x1A_optimal, 3)
        x2A_rounded = round(x2A_optimal, 3)

        return x1A_rounded, x2A_rounded
    
    # QUESTION 6B

    '''
    We create and plot the figure by calling on all the optimal allocations in the previous questions. 
    We plot them all into the edgeworth box from question 1.
    '''

    def figure_6b(self):

        par = self.par
   
        w1bar = 1.0
        w2bar = 1.0

        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='Initial endowment') 

        xes = [] 
        yes = [] 
        for x in self.Cset():
            xes.append(x[0]) 
            yes.append(x[1])

        ax_A.scatter(xes,yes,marker="s", color='green',label='Pareto Pairs') 

        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')
        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        x1A_optimal_4a, x2A_optimal_4a = self.optimal_allocation_4a(print_output=False)
        x1A_optimal_4b, x2A_optimal_4b = self.optimal_allocation_4b(print_output=False) 
        x1A_optimal_5a, x2A_optimal_5a = self.optimal_allocation_5a(print_output=False) 
        x1A_optimal_5b, x2A_optimal_5b = self.optimal_allocation_5b(print_output=False) 
        x1A_optimal_6a, x2A_optimal_6a = self.optimal_allocation_6a() 

        ax_A.scatter(x1A_optimal_4a,x2A_optimal_4a,marker='s',color='red',label='4a') 
        ax_A.scatter(x1A_optimal_4b,x2A_optimal_4b,marker='s',color='yellow',label='4b') 
        ax_A.scatter(x1A_optimal_5a,x2A_optimal_5a,marker='s',color='purple',label='5a') 
        ax_A.scatter(x1A_optimal_5b,x2A_optimal_5b,marker='s',color='turquoise',label='5b') 
        ax_A.scatter(x1A_optimal_6a,x2A_optimal_6a,marker='s',color='blue',label='6a') 

        ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0));
    
    # QUESTION 7
    
    '''
    1. 
    We define the set W by setting a random seed, the number of elements, and determining the distributions.
    We create a list with the pairs of endowments.
    2. 
    We plot the set by unpacking the set to create the axes.
    '''

    def Wset(self):
        
        np.random.seed(42)
        num_elements = 50

        w1A = np.random.uniform(0, 1, num_elements)
        w2A = np.random.uniform(0, 1, num_elements)

        W = list(zip(w1A, w2A))

        return W
    
    def plot_Wset(self):

        W_x = [pair[0] for pair in self.Wset()]
        W_y = [pair[1] for pair in self.Wset()]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(W_x, W_y, color='blue', alpha=0.5)
        plt.title('Random Set W with 50 Elements')
        plt.xlabel('ωA1')
        plt.ylabel('ωA2')
        plt.grid(True)
        plt.show()
    
    # QUESTION 8

    '''
    1. We calculate total demand given the endowments. We define possible allocations as allocations given endowments in the set W.
    We create new demand functions, where the endowments are variables, not baseline parameters. We then create the market clearing
    conditions given the new demand functions. We then find the market clearing allocations for each w_A using the Nelder-Mead method in a for loop
    over the pairs of endowments by minimizing the error.
    2. 
    We then plot the market equilibrium allocation for each w_A in the set W. 
    '''

    def market_clearing_allocation(self, endowment):

        demand_A = self.demand_A(endowment[0])
        demand_B = self.demand_B(endowment[0])

        total_demand_1A = demand_A[0]
        total_demand_2A = demand_A[1]
        total_demand_1B = demand_B[0]
        total_demand_2B = demand_B[1]

        x1A = total_demand_1A
        x2A = total_demand_2A

        x1B = total_demand_1B
        x2B = total_demand_2B

        return x1A, x2A, x1B, x2B
    
    def all_market_clearing_allocations(self):
        W = self.Wset()
        allocations = []
        for endowment in W:
            allocation = self.market_clearing_allocation(endowment)
            allocations.append(allocation)
        return allocations
    
    def newdemand_A(self, p1, w1A, w2A):
        par = self.par
        return par.alpha*(p1*w1A+w2A)/p1, (1-par.alpha)*(p1*w1A+w2A)

    def newdemand_B(self, p1, w1A, w2A):
        par = self.par
        w1B = 1 - w1A
        w2B = 1 - w2A
        return par.beta*(p1*w1B+w2B)/p1, (1-par.beta)*(p1*w1B+w2B)

    def market_clearing_Q8(self, p1, w1A, w2A):
        x1A, x2A = self.newdemand_A(p1, w1A, w2A)
        x1B, x2B = self.newdemand_B(p1, w1A, w2A)
        eps1 = x1A - w1A + x1B - (1 - w1A)
        return eps1
    
    def optimize(self):
        initial_guess = 0.5
        bounds = [(0, np.inf)]
        W = self.Wset()
        allocations_Q8 = []

        for w1A, w2A in W:
            result = optimize.minimize(
                lambda x: np.abs(self.market_clearing_Q8(x, w1A, w2A)),
                initial_guess,
                method='Nelder-Mead',
                bounds=bounds
            )
            allocations_Q8.append(tuple(map(float, self.newdemand_A(result.x[0], w1A, w2A))))

        return allocations_Q8

    def figure_8(self):

        w1bar = 1.0
        w2bar = 1.0

        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        allocations_Q8 = self.optimize()
        x1_allocations_Q8, x2_allocations_Q8 = zip(*allocations_Q8)
        ax_A.scatter(x1_allocations_Q8,x2_allocations_Q8,marker='o', alpha=0.5, color='blue',label='Allocations')

        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0));
    

    