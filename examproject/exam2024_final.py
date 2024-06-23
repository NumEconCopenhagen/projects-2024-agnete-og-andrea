import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import pandas as pd
from scipy.optimize import minimize

class ProductionEconomy:

    # model set up
    def __init__(self):
        '''
        We initialize the model by setting up the parameters given in the problem.
        '''
        par = self.par = SimpleNamespace()

        # parameters
        par.A = 1.0  # productivity parameter
        par.gamma = 0.5  # labor elasticity
        par.alpha = 0.3  # consumption share of good 1
        par.nu = 1.0  # disutility of labor parameter
        par.epsilon = 2.0  # labor supply elasticity
        par.kappa = 0.1  # SCC parameter

    # optimal firm behavior
    '''
    In the following methods, we define the equations that make up the firms' optimal behaviour.
    '''
    def optimal_labor_demand(self, w, p):

        par = self.par

        labor_demand = ((p * par.A * par.gamma) / w) ** (1 / (1 - par.gamma))
        return labor_demand
    
    def optimal_output(self, w, p):

        par = self.par

        l = self.optimal_labor_demand(w, p)
        return par.A * l ** par.gamma
    
    def optimal_profit(self, w, p):

        par = self.par

        l = self.optimal_labor_demand(w, p)
        return ((1 - par.gamma) / par.gamma) * w * l
   
    # optimal consumer behavior
    '''
    In the following methods, we define the consumer problem, 
    which is a constrained utility maximization problem, where she solves for optimal labor supply.
    '''
    def utility(self, l, w, p1, p2, tau, T):

        par = self.par

        I = w * l + T + self.optimal_profit(w, p1) + self.optimal_profit(w, p2)
        c1 = par.alpha * I / p1
        c2 = (1 - par.alpha) * I / (p2 + tau)
        return - (par.alpha * np.log(c1) + (1 - par.alpha) * np.log(c2) - par.nu * (l ** (1 + par.epsilon)) / (1 + par.epsilon))
    
    def budget_constraint(self, l, w, p1, p2, tau, T):

        par = self.par

        I = w * l + T + self.optimal_profit(w, p1) + self.optimal_profit(w, p2)
        c1 = par.alpha * I / p1
        c2 = (1 - par.alpha) * I / (p2 + tau)
        return p1 * c1 + (p2 + tau) * c2 - I
    
    def optimal_labor_supply(self, l, w, p1, p2, tau, T):

        par = self.par

        T = 0.0
        I = w * l + T + self.optimal_profit(w, p1) + self.optimal_profit(w, p2)
        c1 = par.alpha * I / p1
        c2 = (1 - par.alpha) * I / (p2 + tau)
        return -(np.log((c1)**par.alpha * (c2)**(1-par.alpha))-par.nu*((l)**(1+par.epsilon))/(1+par.epsilon)) # set as a negative


    # QUESTION 1

    def consumer_problem(self, p1_range, p2_range, w, tau, T):
        '''
        We start by solving the consumer's problem for the given range of prices. 
        Specifically, we minimize the optimal labor supply function with the scipy.optimize minimize.  
        '''

        ell_initial = 1
        optimal_labor = np.zeros((len(p1_range), len(p2_range)))

        for i, p1 in enumerate(p1_range):
            for j, p2 in enumerate(p2_range):

                result = minimize(self.optimal_labor_supply, ell_initial, args=(w, p1, p2, tau, T), bounds=[(0, None)])
                optimal_labor[i, j] = result.x[0]
        
        return optimal_labor

    '''
    In order to check market clearing, we need to find the optimal consumption for the two goods for all
    the optimal labor supply given all the combinations of prices. We also find the output for all the prices.
    '''
    def consumption(self, optimal_labor, p1_range, p2_range, w, tau, T):

        par = self.par
        T=0.0

        c1 = np.zeros_like(optimal_labor)
        c2 = np.zeros_like(optimal_labor)

        for i in range(len(p1_range)):
            for j in range(len(p2_range)):

                ell = optimal_labor[i, j]
                I = w * ell + T + self.optimal_profit(w, p1_range[i]) + self.optimal_profit(w, p2_range[j])
                c1[i, j] = par.alpha * I / p1_range[i]
                c2[i, j] = (1 - par.alpha) * I / (p2_range[j] + tau)

        return c1, c2
    
    def output(self, p_range, w):

        optimal_output = np.zeros(len(p_range))

        for i, p in enumerate(p_range):
            optimal_output[i] = self.optimal_output(w, p)
        return optimal_output
    
    '''
    We now set the ranges to the linspaces given in the problem.
    We draw of the found optimal consumption and output of the two goods markets.
    We take advantage of Walras' law and only check two of the three markets (not checking labor market). 
    '''

    def question_1(self):

        p1_range = np.linspace(0.1, 2.0, 10)
        p2_range = np.linspace(0.1, 2.0, 10)

        optimal_labor = self.consumer_problem(p1_range, p2_range, w=1.0, tau=0, T=0)
        c1, c2 = self.consumption(optimal_labor, p1_range, p2_range, w=1.0, tau=0, T=0)
        y1 = self.output(p1_range, w=1.0)
        y2 = self.output(p2_range, w=1.0)

        market_1 = np.isclose(y1, c1)
        market_2 = np.isclose(y2, c2)

        if market_1.all():
                print("The market for good 1 clears.")
        else:
                print("The market for good 1 does not clear.")
            
        if market_2.all():
                print("The market for good 2 clears.")
        else:
                print("The market for good 2 does not clear.")

    # QUESTION 2

    '''
    We now find equilibrium prices again by using Walras' Law. We therefore compute the excess demand of both goods.
    The equilibrium condition is that the sum of the excess goods should be equal to zero. 
    We then find the eq. prices by using the Nelder-Mead optimizer.
    '''

    def excess_demand_good1(self, p1, p2, w, tau, T):

        optimal_labor = self.consumer_problem([p1], [p2], w, tau, T)
        optimal_consumption = self.consumption(optimal_labor, [p1], [p2], w, tau, T)[0][0][0]
        return self.output([p1], w)[0] - optimal_consumption

    def excess_demand_good2(self, p1, p2, w, tau, T):

        optimal_labor = self.consumer_problem([p1], [p2], w, tau, T)
        optimal_consumption = self.consumption(optimal_labor, [p1], [p2], w, tau, T)[1][0][0]
        return self.output([p2], w)[0] - optimal_consumption

    def equilibrium_condition(self, p, w, tau, T):

        p1, p2 = p
        return [self.excess_demand_good1(p1, p2, w, tau, T), self.excess_demand_good2(p1, p2, w, tau, T)]

    def equilibrium_prices(self, w, tau, T, initial_guess=[1.0, 1.0]):

        result = minimize(lambda p: sum(np.abs(self.equilibrium_condition(p, w, tau, T))), initial_guess, method='Nelder-Mead')
        p1, p2 = result.x
        return p1, p2
    
    def question_2(self):

        p1_eq, p2_eq = self.equilibrium_prices(w=1, tau=0, T=0)
        print(f"Equilibrium prices: p1 = {p1_eq}, p2 = {p2_eq}")

    # QUESTION 3

    '''
    We now want to maximize the social welfare function with regards to the co2 tax. 
    We do this with a for loop after defining the social welfare function.
    We define the utility maximization problem for the consumer such that tau is a variable, and T is a function of tau.
    We make sure the consumption of good 2 and the optimal labor is found again without the initial values of tau and T. 
    '''

    def social_welfare_function(self, w, tau, T):

        par = self.par

        p1, p2 = self.equilibrium_prices(w, tau, T)
        optimal_labor = self.consumer_problem([p1], [p2], w, tau, T)
        c1, c2 = self.consumption(optimal_labor, [p1], [p2], w, tau, T)
        T = tau * c2[0, 0]  # Update T based on tau and c2
        utility_value = self.utility(optimal_labor[0, 0], w, p1, p2, tau, T)
        SWF = utility_value - par.kappa * self.optimal_output(w, p2)

        return SWF, c2[0, 0]  # Return SWF and c2 for updating T

    def max_SWF(self, w, tau_range):

        max_SWF = -np.inf
        optimal_tau = None
        implied_T = None

        for tau in tau_range:
            T = 0.0
            T_new = T

            for _ in range(100):  
                SWF, c2 = self.social_welfare_function(w, tau, T)
                T_new = tau * c2
                if np.isclose(T, T_new, atol=1e-6):  
                    break
                T = T_new

            if SWF > max_SWF:
                max_SWF = SWF
                optimal_tau = tau
                implied_T = T_new

        return max_SWF, optimal_tau, implied_T
    
    def question_3(self):
        
        w = 1.0
        tau_range = np.linspace(0.0, 1.0, 50)
        max_SWF, optimal_tau, implied_T = self.max_SWF(w, tau_range)

        print("The optimal CO2 tax is:", optimal_tau)
        print("The maximum social welfare given the optimal CO2 tax is:", max_SWF)
        print("The implied T is:", implied_T)

class CareerChoiceModel():
    def __init__(self, par):
        """Define the parameters"""
        self.J = par.J  # Number of career choices
        self.N = par.N  # Number of graduates
        self.K = par.K  # Number of draws
        self.sigma = par.sigma  # Standard deviation of the normal distribution
        self.v = par.v  # Deterministic components of utility
        self.epsilon = None  # Placeholder for the epsilon values
        self.expected_utility = None  # Placeholder for expected utility
        self.average_realized_utility = None  # Placeholder for average realized utility
        self.career_choices = None  # Initialize career_choices attribute

    ##QUESTION 1:
    def simulate_epsilon(self):
        """We simulate K draws for each career j from a normal distribution with mean 0 and standard deviation sigma"""
        np.random.seed(0)  # Set seed 
        self.epsilon = np.random.normal(0, self.sigma, (self.J, self.K))

    def calculate_expected_utility(self):
        """Calculate the expected utility for each career j. Since the expected value of epsilon = 0, expected utility is v_j"""
        return self.v

    def calculate_average_realized_utility(self):
        """The average realized utility for each career j: Add the simulated epsilon values to v_j and take the mean of this"""
        self.realized_utility = self.v[:, np.newaxis] + self.epsilon
        return np.mean(self.realized_utility, axis=1)

    def run_simulation(self):
        """Run the entire simulation process and return results."""
        self.simulate_epsilon()
        avg_expected_utility = self.calculate_expected_utility()
        avg_realized_utility = self.calculate_average_realized_utility()

        return avg_expected_utility, avg_realized_utility
    
    def question_1(self):
        '''Output for question 1'''
        # Run the simulation
        avg_expected_utility, avg_realized_utility = self.run_simulation()

        # Output the results
        print("Average expected utility for each career choice j:")
        for j in range(self.J):
            print(f"Career {j + 1}: {avg_expected_utility[j]}")

        print("\nAverage realized utility for each career choice j:")
        for j in range(self.J):
            print(f"Career {j + 1}: {avg_realized_utility[j]}")

    ##QUESTION 2:
    def simulate_friends_epsilon(self, F_i):
        """Simulate the friends' noise terms:"""
        np.random.seed(10)  # Set seed 
        return np.random.normal(0, self.sigma, (self.J, F_i, self.K))

    def simulate_personal_epsilon(self):
        """Simulate the graduates' personal noise terms:"""
        np.random.seed(123)  # Set seed 
        return np.random.normal(0, self.sigma, (self.N, self.J, self.K))

    def prior_expected_utility(self, F_i):
        """Calculate the prior expected utility for each career track."""
        # Simulate friends' noise terms:
        friends_epsilon = self.simulate_friends_epsilon(F_i)
        
        # Ensure friends_epsilon is properly sliced
        friends_epsilon = friends_epsilon[:, :F_i, :self.K]
        
        # Calculate the prior expected utility
        prior_expected_utility = self.v[:, np.newaxis] + friends_epsilon.mean(axis=1)
        
        return prior_expected_utility
    
    def career_with_max_prior_utility(self):
        """Choose the career track with the highest prior expected utility for each graduate."""
        chosen_careers = np.zeros((self.N, self.K), dtype=int)

        for i in range(self.N):
            F_i = i + 1
            prior_expected_utility = self.prior_expected_utility(F_i)

            for k in range(self.K):
                chosen_career = np.argmax(prior_expected_utility[:, k]) + 1  # Adjust to (1, 2, 3)
                chosen_careers[i, k] = chosen_career

        self.career_choices = chosen_careers
        return chosen_careers
    
    def store_results(self):
        """Store the chosen careers, prior expectations, and realized values."""
        career_choices = np.zeros((self.N, self.K), dtype=int)
        prior_expectations = np.zeros((self.N, self.K))
        realized_values = np.zeros((self.N, self.K))

        personal_epsilon = self.simulate_personal_epsilon()

        for i in range(self.N):
            F_i = i + 1 # As Python indexing starts at 0
            prior_expected_utility = self.prior_expected_utility(F_i)
            epsilon_i = personal_epsilon[i]

            for k in range(self.K):
                chosen_career = np.argmax(prior_expected_utility[:, k]) + 1
                career_choices[i, k] = chosen_career
                prior_expectations[i, k] = prior_expected_utility[chosen_career - 1, k]
                realized_values[i, k] = self.v[chosen_career - 1] + epsilon_i[chosen_career - 1, k]

        self.career_choices=career_choices

        return career_choices, prior_expectations, realized_values

    def visualize_results(self, career_choices, prior_expectations, realized_values):
        """Visualize the results."""
        # Share of graduates choosing each career
        career_share = np.zeros(self.J)
        for j in range(1, self.J + 1):
            career_share[j - 1] = np.mean(career_choices == j)

        # Average prior expected utility
        avg_prior_expectation = np.mean(prior_expectations, axis=1)

        # Average realized utility
        avg_realized_utility = np.mean(realized_values, axis=1)

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Career share plot
        axs[0].bar(np.arange(1, self.J + 1), career_share)
        axs[0].set_title('Figure Q2.1: Share of Graduates Choosing Each Career')
        axs[0].set_xlabel('Career')
        axs[0].set_ylabel('Share')

        # Average prior expected utility plot
        axs[1].bar(np.arange(1, self.N + 1), avg_prior_expectation)
        axs[1].set_title('Figure Q2.2: Average Prior Expected Utility')
        axs[1].set_xlabel('Graduate')
        axs[1].set_ylabel('Utility')

        # Average realized utility plot
        axs[2].bar(np.arange(1, self.N + 1), avg_realized_utility)
        axs[2].set_title('Figure Q2.3: Average Ex Post Realized Utility')
        axs[2].set_xlabel('Graduate')
        axs[2].set_ylabel('Utility')

        plt.tight_layout()
        plt.show()


    def question_2(self):
        '''Output for question 2'''
        # Run simulation to simulate epsilon and calculate utilities
        self.run_simulation()

        # Determine chosen careers based on max prior expected utility
        chosen_careers = self.career_with_max_prior_utility()

        # Visualize the chosen careers:
        career_choices, prior_expectations, realized_values = self.store_results()

        self.visualize_results(career_choices, prior_expectations, realized_values)

    ##QUESTION 3:
    def expected_utility_with_switching_cost(self, F_i, chosen_career, switching_cost):
        """Calculate the prior expected utility with switching cost for each career track."""
        q3_prior_expected_utility = np.zeros(self.J)  # Initialize an array for prior expected utility
        
        # Apply switching cost conditionally
        for j in range(self.J):
            if j == chosen_career - 1:  # Adjust for zero-based index
                q3_prior_expected_utility[j] = self.v[j]  # No switching cost for the chosen career
            else:
                q3_prior_expected_utility[j] = self.v[j] - switching_cost  # Apply switching cost for other careers
        
        return q3_prior_expected_utility

    def store_results_with_switching_cost(self, switching_cost):
        """Store the chosen careers, prior expectations, and realized values with switching cost."""
        career_choices_q3 = np.zeros((self.N, self.K), dtype=int)
        q3_prior_expectations = np.zeros((self.N, self.K))
        q3_realized_values = np.zeros((self.N, self.K))

        # Ensure career_choices is populated
        if self.career_choices is None:
            self.career_with_max_prior_utility() 

        personal_epsilon = self.simulate_personal_epsilon()

        for i in range(self.N):
            F_i = i + 1  # As Python indexing starts at 0

            for k in range(self.K):
                chosen_career = self.career_choices[i, k]  # Chosen career from previous step
                q3_prior_expected_utility = self.expected_utility_with_switching_cost(F_i, chosen_career, switching_cost)

                chosen_career_q3 = np.argmax(q3_prior_expected_utility) + 1  # Adjust the careers to (1, 2, 3) instead of (0, 1, 2)
                career_choices_q3[i, k] = chosen_career_q3

                # Store prior expectation and realized value
                q3_prior_expectations[i, k] = q3_prior_expected_utility[chosen_career_q3 - 1]
                if chosen_career_q3 == chosen_career:  # If chosen career remains the same
                    q3_realized_values[i, k] = self.v[chosen_career_q3 - 1] + personal_epsilon[i, chosen_career_q3 - 1, k]
                else:
                    q3_realized_values[i, k] = self.v[chosen_career_q3 - 1] + personal_epsilon[i, chosen_career_q3 - 1, k] - switching_cost # Switching cost is subtracted if initial career != final career

        # Update instance variables
        self.career_choices_q3 = career_choices_q3
        self.q3_prior_expectations = q3_prior_expectations
        self.q3_realized_values = q3_realized_values

        return career_choices_q3, q3_prior_expectations, q3_realized_values

    def visualize_results_with_switching_cost(self, career_choices_q3, q3_prior_expectations, q3_realized_values, switching_cost):
        """Visualize the results with switching cost included."""
        # Share of graduates choosing each career
        career_share = np.zeros(self.J)
        for j in range(1, self.J + 1):
            career_share[j - 1] = np.mean(career_choices_q3 == j)

        # Average prior expected utility
        avg_prior_expectation = np.mean(q3_prior_expectations, axis=1)

        # Average realized utility
        avg_realized_utility = np.mean(q3_realized_values, axis=1)

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Career share plot
        axs[0].bar(np.arange(1, self.J + 1), career_share)
        axs[0].set_title('Figure Q3.1: Share of Graduates Choosing Each Career')
        axs[0].set_xlabel('Career')
        axs[0].set_ylabel('Share')

        # Average prior expected utility plot
        axs[1].bar(np.arange(1, self.N + 1), avg_prior_expectation)
        axs[1].set_title('Figure Q3.2: Average Prior Expected Utility with Switching Cost')
        axs[1].set_xlabel('Graduate')
        axs[1].set_ylabel('Utility')

        # Average realized utility plot
        axs[2].bar(np.arange(1, self.N + 1), avg_realized_utility)
        axs[2].set_title('Figure Q3.3: Average Ex Post Realized Utility with Switching Cost')
        axs[2].set_xlabel('Graduate')
        axs[2].set_ylabel('Utility')

        plt.tight_layout()
        plt.show()

    def numeric_solutions_table(self, career_choices_q3, q3_prior_expectations, q3_realized_values):
        """Generate and return table that shows the career choice, expected utility, and realized utility for each graduate"""
        results_table = pd.DataFrame(index=np.arange(1, self.N + 1))
        results_table['Career Choice'] = career_choices_q3[:, 0]  # Assuming displaying the first draw choice
        results_table['Avg Prior Expected Utility'] = np.mean(q3_prior_expectations, axis=1)
        results_table['Avg Ex Post Realized Utility'] = np.mean(q3_realized_values, axis=1)

        return results_table
  
    def initial_to_other_career_share(self, initial_career):
        """Calculate the share of graduates initially in one career who switch to another career."""
        switch_count = 0
        initial_career_count = 0

        """Count the number who stays in same career and number of career switches"""
        for i in range(self.N): # For each graduate
            for k in range(self.K): # For each simulation
                current_career = self.career_choices[i, k]
                final_career = self.career_choices_q3[i, k]

                if current_career == initial_career:
                    initial_career_count += 1
                    if current_career != final_career:
                        switch_count += 1

        if initial_career_count > 0:
            initial_to_other_share = switch_count / initial_career_count
        else:
            initial_to_other_share = 0.0

        return initial_to_other_share

    def table_career_switch(self):
        """Calculate the share of graduates initially in each career who switch to another career."""
        shares = {}

        for j in range(1, self.J + 1):
            share = self.initial_to_other_career_share(j)
            shares[f'Career {j}'] = share

        # Print the shares in a table:
        print("\nTable Q3.2: Share of graduates initially in each career switching to another career:")
        for career, share in shares.items():
            print(f"{career}: {share:.2%}")

        return shares
    
    def question_3(self, switching_cost=1):
        """Output for question 3"""
        career_choices = self.career_with_max_prior_utility()
        career_choices_q3, q3_prior_expectations, q3_realized_values = self.store_results_with_switching_cost(switching_cost)

        # Print the stored results
        print("\nThe new optimal career choice for each i, k:")
        print("===========================================")
        for i in range(self.N):
            print(f"Graduate {i + 1}:")
            for k in range(self.K):
                chosen_career_q3 = career_choices_q3[i, k]
                prior_expectation_q3 = q3_prior_expectations[i, k]
                realized_value_q3 = q3_realized_values[i, k]

                print(f"  Draw {k + 1}:")
                print(f"    Chosen Career: {chosen_career_q3}")
                print(f"    Prior Expectation: {prior_expectation_q3:.2f}")
                print(f"    Realized Value: {realized_value_q3:.2f}")
            print()

        # Figures:
        self.visualize_results_with_switching_cost(career_choices_q3, q3_prior_expectations, q3_realized_values, switching_cost)

        # Print the table:
        numeric_solutions_table = self.numeric_solutions_table(career_choices_q3, q3_prior_expectations, q3_realized_values)
        print('\nTable Q3.1: The table shows the average, optimal career choice for each i, the prior expected utility, and ex post realized utility for new optimal career choices: \n')
        print(numeric_solutions_table)

        # Print the share of graduates that switch career
        self.table_career_switch()
    
class BarycentricInterpolation:

    def __init__(self):
        """
        We initiale with the given random number generator. This provides us with random coordinates in the unit square.
        We also define the functions given and initiale the points. 
        """
        self.rng = np.random.default_rng(2024)
        self.X = self.rng.uniform(size=(50, 2))
        self.y = self.rng.uniform(size=(2,))
        self.A, self.B, self.C, self.D = None, None, None, None

        # for question 3 and 4
        self.f = lambda x: x[0] * x[1]
        self.F = np.array([self.f(x) for x in self.X])

    def barycentric_coordinates(self, A, B, C, y):
        """
        Building block I: We define the barycentric coordinates of the point y.
        """
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2

        return r1, r2, r3
    

    def find_points(self, y):
        """
        Building block II: We compute the points A, B, C, and D as the minimizing arguments of the Euclidian distance,
        """
        X = self.X
        A = B = C = D = None
        min_A = min_B = min_C = min_D = float('inf')
        
        for point in X:
            dist = np.linalg.norm(point - y)
            if point[0] > y[0] and point[1] > y[1] and dist < min_A:
                A, min_A = point, dist
            if point[0] > y[0] and point[1] < y[1] and dist < min_B:
                B, min_B = point, dist
            if point[0] < y[0] and point[1] < y[1] and dist < min_C:
                C, min_C = point, dist
            if point[0] < y[0] and point[1] > y[1] and dist < min_D:
                D, min_D = point, dist
        
        return A, B, C, D

    def interpolate(self, y):
        """
        If the points safisfy the conditions, we then use the coordinates to find the function 
        values at the point y. If they are not found, the method returns nan.
        We do this for both triangles.
        """
        A, B, C, D = self.find_points(y)
        if A is None or B is None or C is None or D is None:
            return np.nan
        
        # triangle ABC
        r1, r2, r3 = self.barycentric_coordinates(A, B, C, y)
        if 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1:
            return r1 * self.F[np.where((self.X == A).all(axis=1))[0][0]] + \
                   r2 * self.F[np.where((self.X == B).all(axis=1))[0][0]] + \
                   r3 * self.F[np.where((self.X == C).all(axis=1))[0][0]]
        
        # triangle CDA
        r1, r2, r3 = self.barycentric_coordinates(C, D, A, y)
        if 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1:
            return r1 * self.F[np.where((self.X == C).all(axis=1))[0][0]] + \
                   r2 * self.F[np.where((self.X == D).all(axis=1))[0][0]] + \
                   r3 * self.F[np.where((self.X == A).all(axis=1))[0][0]]
        
        return np.nan

    def question1(self):
        """
        We print the found values of A, B, C and D for point y. We also plot a figure that show:
        - the random set X in the unit square
        - the point y
        - triangle ABC and CDA.
        """
        self.A, self.B, self.C, self.D = self.find_points(self.y)
        print("A:", self.A)
        print("B:", self.B)
        print("C:", self.C)
        print("D:", self.D)
        
        plt.scatter(self.X[:, 0], self.X[:, 1], label='Data points')
        plt.scatter(self.y[0], self.y[1], color='red', label='y')
        if self.A is not None and self.B is not None and self.C is not None:
            plt.plot([self.A[0], self.B[0], self.C[0], self.A[0]], [self.A[1], self.B[1], self.C[1], self.A[1]], 'r-', label='Triangle ABC')
        if self.C is not None and self.D is not None and self.A is not None:
            plt.plot([self.C[0], self.D[0], self.A[0], self.C[0]], [self.C[1], self.D[1], self.A[1], self.C[1]], 'g-', label='Triangle CDA')
        plt.legend()
        plt.title("Points and Triangles")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    def question2(self):
        """
        We return the coordinates for the two triangles given the found points and the point y. 
        We also determine which triangle the point is in by checking the conditions for the coordinates.
        """
        r_ABC = self.barycentric_coordinates(self.A, self.B, self.C, self.y) if self.A is not None and self.B is not None and self.C is not None else (None, None, None)
        r_CDA = self.barycentric_coordinates(self.C, self.D, self.A, self.y) if self.C is not None and self.D is not None and self.A is not None else (None, None, None)
        print("Barycentric coordinates for triangle ABC:", r_ABC)
        print("Barycentric coordinates for triangle CDA:", r_CDA)
        
        if r_ABC and all(r is not None and 0 <= r <= 1 for r in r_ABC):
            print("y is inside the triangle ABC")
        elif r_CDA and all(r is not None and 0 <= r <= 1 for r in r_CDA):
            print("y is inside the triangle CDA")
        else:
            print("y is outside both triangles")

    def question3(self):
        """
        We find the approximate value from the interpolation and the true value from the given function and point y.
        """
        approx_f_y = self.interpolate(self.y)
        true_f_y = self.f(self.y)
        print("Approximated f(y):", approx_f_y)
        print("True f(y):", true_f_y)

    def question4(self):
        """
        We do the same as in question 3, but use the given tuple of coordinates, instead of the original point y.
        """
        Y = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.8, 0.2), (0.5, 0.5)]
        for y in Y:

            approx_f_y = self.interpolate(y)
            true_f_y = self.f(y)
            print(f"Approximated f(y) = {approx_f_y}, True f(y) = {true_f_y}")
            print()