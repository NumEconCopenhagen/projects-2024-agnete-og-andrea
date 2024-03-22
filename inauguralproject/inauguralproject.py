from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ExchangeEconomyClass():

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):

        par = self.par 

        u_A = x1A** par.alpha * x2A** (1- par.alpha)

        return u_A

    def utility_B(self,x1B,x2B):

        par = self.par

        return x1B ** par.beta * x2B **(1-par.beta)
    

    def demand_A(self,p1):

        par = self.par 

        if not isinstance(p1, (int, float)):
            p1 = float(p1)

        x1A = par.alpha * ((p1*par.w1A+par.w2A)/p1)  
        x1A = max(0, min(1, x1A))  # Ensure x1B is between 0 and 1
        x1B = 1 - x1A
        x2A = (1-par.alpha) * (p1*par.w1A+par.w2A)
        x2B = max(0, min(1,1 - x2A))  # Ensure x2B is between 0 and 1

        return np.array([x1A, x2A])

    def demand_B(self,p1):

        par = self.par

        if not isinstance(p1, (int, float)):
            p1 = float(p1)

        x1B = par.beta * ((p1*par.w1B+par.w2B)/p1)
        x1B = max(0, min(1, x1B))  # Ensure x1A is between 0 and 1
        x2B = (1 - par.beta) * (p1*par.w1B+par.w2B)
        x2A = max(0, min(1,1 - x2B))  # Ensure x2A is between 0 and 1

        return np.array([x1B, x2B])

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    #Question 1: defining the C set

    def Cset(self):

        par = self.par
        
        par.N = 75
        
        #defining the range for x1A, x2A

        x1A_range = np.arange(0, 1 + 1/par.N, 1/par.N)
        x2A_range = np.arange(0, 1 + 1/par.N, 1/par.N)

        #creating the conditions for the subset C

        logical_condition_1 = lambda x1A, x2A: self.utility_A(x1A, x2A) >= self.utility_A(par.w1A, par.w2A)
        logical_condition_2 = lambda x1B, x2B: self.utility_B(x1B, x2B) >= self.utility_B(par.w1B, par.w2B)
        logical_condition_3 = lambda x1A, x1B: x1B == 1 - x1A
        logical_condition_4 = lambda x2A, x2B: x2B == 1 - x2A

        #creating the set C as a list of Pareto pairs of x1A, x2A

        values = [(x1A, x2A) for x1A in x1A_range for x2A in x2A_range if logical_condition_1(x1A, x2A) and logical_condition_2(1-x1A, 1-x2A) and logical_condition_3(x1A, 1-x1A) and logical_condition_4(x2A, 1-x2A)]

        return values
    
    #Question 2: defining P1 set

    def P1set(self):

        par = self.par

        p1_range = {0.5+2* i / par.N for i in range(par.N+1)}

        return p1_range
    
    def Q2errors(self, p1_range):

        errors = []

        for p1 in p1_range:

            self.eps1, self.eps2 = self.check_market_clearing(p1)
            errors.append((self.eps1, self.eps2))

        return errors
    
    #Question 2 & Question 3: Market clearing price

    def market_clearing_price(self):

        p1_range = range(1, 100)
        errors = self.Q2errors(p1_range)

        market_clearing_price_index = np.argmin(np.abs(errors))

        market_clearing_price = list(p1_range)[market_clearing_price_index]

        return market_clearing_price

    #Question 4a: 

    def optimal_allocation_4a(self):

        p1_range = self.P1set() # Use the P1set for generating p1 values

        max_utility = float('-inf')

        optimal_allocation = None

        for p1 in p1_range:

            x1A, x2A = self.demand_A(p1)  # Extract x1A and x2A from demand_A
            # Check if x1A and x2A are within the valid range

            if 0 <= x1A <= 1 and 0 <= x2A <= 1:

                utility = self.utility_A(x1A, x2A)

                if utility > max_utility:

                    max_utility = utility
                    optimal_allocation = (x1A, x2A)

        x1A_rounded = round(optimal_allocation[0], 3)
        x2A_rounded = round(optimal_allocation[1], 3)

        return x1A_rounded, x2A_rounded            

    #Question 4b:

    def max_utility_prices_unrestricted(self):

        max_utility = float('-inf')

        max_p1_optimal_unrestricted = None
    
        # Iterate over positive values for p1

        for p1 in range(1, 1000):

            # Calculate the demand for goods A using p1

            x1A, x2A = self.demand_A(p1)

            if 0 <= x1A <= 1 and 0 <= x2A <= 1:  # Check feasibility

            # Calculate the utility for goods A using the demand

                utility = self.utility_A(x1A, x2A)
        
            # Check if the utility is greater than the current maximum utility

                if utility > max_utility:

                # Update the maximum utility and the corresponding p1

                    max_utility = utility
                    max_p1_optimal_unrestricted = p1
    
            # Return the optimal p1 value that maximizes utility

            return max_p1_optimal_unrestricted, 1  # Since p2 is the numeraire

    def optimal_allocation_4b(self):

        # Get the utility maximizing price of good 1 (p1) for any positive p1

        p1_optimal_4b, _ = self.max_utility_prices_unrestricted()

        # Find the allocation of (x1A, x2A) for the utility maximizing price of good 1 for any positive p1

        x1A_optimal_unrestricted, x2A_optimal_unrestricted = self.demand_A(p1_optimal_4b)

        x1A_rounded = round(x1A_optimal_unrestricted, 3)
        x2A_rounded = round(x2A_optimal_unrestricted, 3)

        return x1A_rounded, x2A_rounded

    # Question 5a
    def optimal_allocation_5a(self):

        # Get the utility maximizing allocation of (x1A, x2A) in set C

        x1A_optimal, x2A_optimal = self.utility_maximization_Cset()

        x1A_rounded = round(x1A_optimal, 3)
        x2A_rounded = round(x2A_optimal, 3)

        return x1A_rounded, x2A_rounded

    # Question 5b

    def optimal_allocation_5b(self):

        # Get the utility maximizing allocation of (x1A, x2A) without restrictions

        x1A_optimal, x2A_optimal = self.utility_maximization_no_restrictions_B()

        x1A_rounded = round(x1A_optimal, 3)
        x2A_rounded = round(x2A_optimal, 3)

        return x1A_rounded, x2A_rounded

    # Additional method for utility maximization in set C

    def utility_maximization_Cset(self):

        # Get the set C

        set_C = self.Cset()

        # Initialize maximum utility

        max_utility = float('-inf')

        # Initialize optimal allocation

        x1A_optimal, x2A_optimal = None, None

        # Iterate over each allocation in set C

        for x1A, x2A in set_C:

            # Calculate utility for agent A

            utility_A = self.utility_A(x1A, x2A)

            # Update optimal allocation if utility is higher

            if utility_A > max_utility:
                
                max_utility = utility_A
                x1A_optimal, x2A_optimal = x1A, x2A
                
        return x1A_optimal, x2A_optimal

    # Additional method for utility maximization without restrictions, ensuring B's utility is not worse than initial endowment

    def utility_maximization_no_restrictions_B(self):

        # Initialize maximum utility

        max_utility = float('-inf')

        # Initialize optimal allocation

        x1A_optimal, x2A_optimal = None, None

        # Iterate over each possible allocation in [0, 1] X [0, 1]

        for x1A in np.linspace(0, 1, 100):

            for x2A in np.linspace(0, 1, 100):

                # Ensure that B is not worse off than the initial endowment

                if self.utility_B(1 - x1A, 1 - x2A) >= self.utility_B(self.par.w1B, self.par.w2B):

                    # Calculate utility for agent A

                    utility_A = self.utility_A(x1A, x2A)

                    # Update optimal allocation if utility is higher

                    if utility_A > max_utility:

                        max_utility = utility_A
                        x1A_optimal, x2A_optimal = x1A, x2A

        return x1A_optimal, x2A_optimal
    
        #Question 6a

    def utility_maximization_6a(self):

        # Initialize maximum utility

        max_utility = float('-inf')

        # Initialize optimal allocation

        x1A_optimal, x2A_optimal = None, None

        # Iterate over each possible allocation in [0, 1] X [0, 1]

        for x1A in np.linspace(0, 1, 100):

            for x2A in np.linspace(0, 1, 100):

                # Calculate utility for agent A

                utility_6a = self.utility_A(x1A, x2A)+self.utility_B(1-x1A,1-x2A)

                # Update optimal allocation if utility is higher

                if utility_6a > max_utility:

                    max_utility = utility_6a

                    x1A_optimal, x2A_optimal = x1A, x2A

        return x1A_optimal, x2A_optimal
    
    def optimal_allocation_6a(self):

        # Get the utility maximizing allocation of (x1A, x2A) without restrictions

        x1A_optimal, x2A_optimal = self.utility_maximization_6a()

        return x1A_optimal, x2A_optimal
    
    #Question 7: creating the set W
    
    def Wset(self):
        
        par=self.par
        # Set the seed for reproducibility
        np.random.seed(42)

        # Number of elements in the set
        num_elements = 50

        # Generate random values for ωA1 and ωA2
        w1A = np.random.uniform(0, 1, num_elements)
        w2A = np.random.uniform(0, 1, num_elements)

        # Create a set W with pairs (ωA1, ωA2)
        W = list(zip(w1A, w2A))

        return W
    
    #Question 8:

    def market_clearing_allocation(self, endowment):
        par = self.par

        # Calculate the total endowments for goods 1 and 2
        total_endowment_1 = par.w1A + par.w1B
        total_endowment_2 = par.w2A + par.w2B

        # Calculate the total demand for goods 1 and 2
        demand_A = self.demand_A(endowment[0])
        demand_B = self.demand_B(endowment[0])

        # Extract individual demand components
        total_demand_1A = demand_A[0]
        total_demand_2A = demand_A[1]
        total_demand_1B = demand_B[0]
        total_demand_2B = demand_B[1]

        # Calculate the allocations for agent A
        x1A = total_demand_1A
        x2A = total_demand_2A

        # Calculate the allocations for agent B
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
        return self.par.alpha*(p1*w1A+w2A)/p1, (1-self.par.alpha)*(p1*w1A+w2A)

    def newdemand_B(self, p1, w1A, w2A):
        w1B = 1 - w1A
        w2B = 1 - w2A
        return self.par.beta*(p1*w1B+w2B)/p1, (1-self.par.beta)*(p1*w1B+w2B)

    def market_clearing_Q8(self, p1, w1A, w2A):
        x1A, x2A = self.newdemand_A(p1, w1A, w2A)
        x1B, x2B = self.newdemand_B(p1, w1A, w2A)
        eps1 = x1A - w1A + x1B - (1 - w1A)
        return eps1
    
    def optimize(self):
        initial_guess = 0
        bounds = [(0, np.inf)]
        W = self.Wset()
        allocations_Q8 = []

        for w1A, w2A in W:
            res = optimize.minimize(
                lambda x: np.abs(self.market_clearing_Q8(x, w1A, w2A)),
                initial_guess,
                method='Nelder-Mead',
                bounds=bounds
            )
            allocations_Q8.append(tuple(map(float, self.newdemand_A(res.x[0], w1A, w2A))))
            

        return allocations_Q8
    

    