from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, Math

def Solow_analytical():

    k = sm.symbols('k')
    A = sm.symbols('A')
    alpha = sm.symbols('alpha')
    delta = sm.symbols('delta')
    s = sm.symbols('s')
    

    #Define production function in per capita terms:
    f=A*k**alpha

    #The SS equation:
    ss = sm.Eq(k,(s*f+(1-delta)*k))

    #Solves the steady-state equation for the steady-state value of capital (k_ss) using SymPy's solve function and takes the first solution (assuming there's only one solution), storing it in the variable kss.
    kss = sm.solve(ss,k)[0]

    print('The steady state equation for k is:')

    display(kss)

class Solow_model():

    def __init__(self):
        # Initialize parameter values as attributes of the class
        self.alpha = 0.3
        self.delta = 0.1
        self.s = 0.2
        self.A = 2.0
        self.k_min = 0
        self.k_max = 14
        self.ts_length = 100
        self.t_min = 0
        self.t_max = self.ts_length
        self.result = 0
        self.ts_1 = []
        self.newA = [self.A for _ in range(self.t_max+1)]

    def run(self):
        # Define the production function f(k)=k^α using a lambda function
        f = lambda k: self.A * k**self.alpha
        
        # Define an objective function obj_kss that represents the difference between the left-hand side and right-hand side of the steady-state equation.
        # This objective function calculates the deviation from steady state for a given k_ss
        obj_kss = lambda kss: kss - (self.s * f(kss) + (1 - self.delta) * kss)
        
        # Use the root_scalar function to find the root of the obj_kss function, which corresponds to the steady-state value of capital (k_ss)
        result = optimize.root_scalar(obj_kss, bracket=[0.1, 100], method='brentq') 
        
        # Print the steady-state value of capital (k_ss) found by the optimization routine
        print('The steady state for k is', result.root) 
        #return result.root

    def visual(self, k, A=99):
        # Use the parameter values stored as attributes of the class to calculate visual
        if (A == 99): A=self.A 
        return A * self.s * k**self.alpha + (1 - self.delta) * k

    def plot45(self):

        # Generate grid for capital values
        xgrid = np.linspace(self.k_min, self.k_max, 12000)

        # Create figure and axes
        fig, ax = plt.subplots()

        # Set limits for x-axis and y-axis
        ax.set_xlim(self.k_min, self.k_max)
        ax.set_ylim(self.k_min, self.k_max)  # Adjust y-axis limits same as x-axis

        # Calculate visual values using the production function
        visual_values = self.visual(xgrid)

        # Calculate integer ticks for both axes
        x_ticks = np.arange(self.k_min, self.k_max + 1, 1)
        y_ticks = np.arange(self.k_min, self.k_max + 1, 1)

        # Set integer ticks for both axes
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Label for the production function
        lb = r'$k_{t+1} = sAk^{\alpha} + (1 - \delta)k$'

        # Plot the production function
        ax.plot(xgrid, visual_values, lw=2, alpha=0.6, label=lb)
        ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='$45^{\circ}$')

        # Calculate steady-state capital
        kstar = ((self.s * self.A) / self.delta)**(1/(1 - self.alpha))

        # Plot steady-state point
        ax.plot(kstar, kstar, 'go', ms=10, alpha=0.6)
        ax.annotate(r'$k^* = {:.2f}$'.format(kstar),
                    xy=(kstar, kstar),
                    xycoords='data',
                    xytext=(-40, -60),
                    textcoords='offset points',
                    fontsize=14,
                    arrowprops=dict(arrowstyle="->"))

        # Add legend
        ax.legend(loc='upper left', frameon=False, fontsize=12)

        # Set labels for axes
        ax.set_xlabel('$k_t$', fontsize=12)
        ax.set_ylabel('$k_{t+1}$', fontsize=12)

        # Show plot
        plt.show()


    def simulate(self, k_ini_values, A=[], dograph=True):
        
        # Calculate steady-state capital
        k_star = (self.s * self.A / self.delta)**(1/(1-self.alpha))
        ymin, ymax = 0, (k_star+5)

        # Create figure and axes
        if dograph:
            fig, ax = plt.subplots(figsize=[11, 5])

            # Adjust x-axis limits dynamically based on the time period
            ax.set_xlim(self.t_min, self.ts_length - 1)  # Adjust x-axis limits

            # Set y-axis limits
            ax.set_ylim(ymin, ymax)

            # Initialize time series array
            ts = np.zeros(self.ts_length)

            # Simulate and plot time series
            for k_init in k_ini_values:
                ts[0] = k_init
                for t in range(1, self.ts_length):
                    ts[t] = self.visual(ts[t-1], A=self.newA[t])
                ax.plot(np.arange(self.ts_length), ts, '-o', ms=4, alpha=0.6,
                       label=r'$k_0=%g$' %k_init)
                ax.plot(np.arange(self.ts_length), np.full(self.ts_length, k_star),
                    alpha=0.6, color='red', label=r'$k^*$')
        
            ax.legend(fontsize=10)

        # Set labels for axes
            ax.set_xlabel(r'$t$', fontsize=14)  # Change x-axis label
            ax.set_ylabel(r'$k_t$', fontsize=14)
            plt.show()
        else:
            ts = np.zeros(self.ts_length)

            # Simulate and plot time series
            for k_init in k_ini_values:
               ts[0] = k_init
            for t in range(1, self.ts_length):
                   ts[t] = self.visual(ts[t-1], A=self.newA[t])
        
        self.ts_1 = ts
            # Show plot
            

    def plot_combined(self, s_values):
        
        # Generate grid for capital values
        xgrid = np.linspace(self.k_min, self.k_max, 12000)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=[10, 6])

        # Plot the 45-degree line
        ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='$45^{\circ}$')

        # Plot each scenario
        for s_val in s_values:
            # Set new savings rate value
            self.s = s_val

            # Calculate steady-state capital
            k_star = ((self.s * self.A) / self.delta) ** (1 / (1 - self.alpha))

            # Calculate visual values using the production function
            visual_values = self.visual(xgrid)

            # Plot the production function
            ax.plot(xgrid, visual_values, label=f's={s_val}')

            # Plot a point for the k_star value and add label
            ax.plot(k_star, k_star, 'go', ms=10, alpha=0.6)
            ax.text(k_star, k_star, f'$k^*={k_star:.2f}$', fontsize=10, va='bottom', ha='right')

        # Set limits for x-axis and y-axis
        ax.set_xlim(self.k_min, 14)
        ax.set_ylim(self.k_min, 14)

        # Set integer ticks for both axes
        ax.set_xticks(np.arange(self.k_min, 14 + 1, 1))
        ax.set_yticks(np.arange(self.k_min, 14 + 1, 1))

        # Add legend
        ax.legend(loc='upper left', frameon=False, fontsize=10)

        # Set labels for axes
        ax.set_xlabel('$k_t$', fontsize=12)
        ax.set_ylabel('$k_{t+1}$', fontsize=12)

        # Show plot
        plt.title('Solow Model Dynamics for Different Savings Rates')
        plt.show()

    def plot_technology_shock_combined(self):
       
        # Original model with A=2
        original_model = Solow_model()
        original_model.run()  # Calculate steady-state capital
        
        # Shocked model with A=2.5
        shocked_model = Solow_model()
        shocked_model.A = 2.5
        for i in range(0, 100):
            shocked_model.newA[i] = 2.5
        shocked_model.run()  # Calculate steady-state capital

        # Plot the original transition path (k_0=5)
        original_model.simulate([5],dograph=False)
        k_star_original = original_model.result

        # Plot the shocked transition path (k_0=5)
        shocked_model.simulate([5],dograph=False)
        k_star_shocked = shocked_model.result
        
        # Calculate steady-state capital for the two models 
        k_star_original = (original_model.s * original_model.A / original_model.delta)**(1/(1-original_model.alpha))
        ymin, ymax = 0, (k_star_original+5)
        k_star_shocked = (shocked_model.s * shocked_model.A / shocked_model.delta)**(1/(1-shocked_model.alpha))
        ymin, ymax = 0, (k_star_original+5)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=[11, 5])

        # Adjust x-axis limits dynamically based on the time period
        ax.set_xlim(self.t_min, self.ts_length - 1)  # Adjust x-axis limits
        

        # Set y-axis limits
        ax.set_ylim(ymin, ymax)

        # Initialize time series array
        ts = np.zeros(self.ts_length)
        self.ts_1 = ts

        # Simulate and plot time series
        ax.plot(np.arange(self.ts_length), shocked_model.ts_1, '-o', ms=4, alpha=0.6,
                    label="shocked")
        ax.plot(np.arange(self.ts_length), original_model.ts_1, '-o', ms=4, alpha=0.6,
                    label="original")
        ax.plot(np.arange(self.ts_length), np.full(self.ts_length, k_star_original),
                alpha=0.6, color='red', label=r'$k^*$')
        ax.plot(np.arange(self.ts_length), np.full(self.ts_length, k_star_shocked),
                alpha=0.6, color='green', label=r'$k^*_{shock}$')
        ax.legend(fontsize=10)

        # Set labels for axes
        ax.set_xlabel(r'$t$', fontsize=14)  # Change x-axis label
        ax.set_ylabel(r'$k_t$', fontsize=14)

        # Show plot
        plt.show()


"""Extensions:"""

def Solow_chap8_analytical():
    k, alpha, delta, phi, s, n = sm.symbols('k alpha delta phi s n')

    # Step 3: Define the steady-state equation
    k_next = (1 / (1 + n)) * k * (s * k**(alpha - 1) + (1 - delta))**(1 - phi)

    # Step 4: Solve for the steady-state value of k
    kss = sm.solve(sm.Eq(k_next, k), k)

    kss_simplified = sm.simplify(kss[1])  # Selecting the non-trivial solution

    # Step 6: Display the solution
    print("Steady-state value of k:")
    display(kss_simplified)

    #Substituting into y:
    y_ss=kss_simplified**alpha
    print("Substituting the steady-state value of k into the SS equation for y yields the following steady-state value of y:")
    display(y_ss)


class SolowModelWithTechProgress():

    def __init__(self, alpha=0.3, delta=0.1, s=0.2, n=0.01, phi=0.5, k_min=0, k_max=4, ts_length=100):
        self.alpha = alpha
        self.delta = delta
        self.s = s
        self.n = n
        self.phi = phi
        self.k_min = k_min
        self.k_max = k_max
        self.ts_length = ts_length
        self.newA = [self.alpha for _ in range(self.ts_length+1)]

    def production_function(self, k):
        return k**self.alpha

    def steady_state_equation(self, k):
        lhs = k
        rhs = (1 / (1 + self.n)) * k * (self.s * k**(self.alpha - 1) + (1 - self.delta))**(1 - self.phi)
        return lhs - rhs

    def find_steady_state(self):
        result = optimize.root_scalar(self.steady_state_equation, bracket=[0.1, 100], method='brentq')
        k_steady_state = result.root
        y_steady_state = self.production_function(k_steady_state)

        #print('The steady state for k is', k_steady_state) 
        #print('The steady state for y is', y_steady_state) 

        return k_steady_state, y_steady_state
    
    def visual(self, k, A=99):
        if A == 99:
            A = self.alpha
        return (1 / (1 + self.n)) * k * (self.s * k**(self.alpha - 1) + (1 - self.delta))**(1 - self.phi)
    
    def plot45(self):
        xgrid = np.linspace(self.k_min, self.k_max, 12000)
        fig, ax = plt.subplots()

        ax.set_xlim(self.k_min, self.k_max)
        ax.set_ylim(self.k_min, self.k_max)

        visual_values = self.visual(xgrid)

        x_ticks = np.arange(self.k_min, self.k_max + 1, 1)
        y_ticks = np.arange(self.k_min, self.k_max + 1, 1)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        lb = r'$k_{t+1} = \frac{1}{1 + n} \cdot k_{t} \cdot \left(s \cdot k_{t}^{\alpha - 1} + (1 - \delta)\right)^{1 - \phi}$'

        ax.plot(xgrid, visual_values, lw=2, alpha=0.6, label=lb)
        ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='$45^{\circ}$')

        k_star, _ = self.find_steady_state()
        ax.plot(k_star, k_star, 'go', ms=10, alpha=0.6)
        ax.annotate(r'$k^* = {:.2f}$'.format(k_star),
                    xy=(k_star, k_star),
                    xycoords='data',
                    xytext=(-40, -60),
                    textcoords='offset points',
                    fontsize=14,
                    arrowprops=dict(arrowstyle="->"))

        ax.legend(loc='upper left', frameon=False, fontsize=12)
        ax.set_xlabel('$k_t$', fontsize=12)
        ax.set_ylabel('$k_{t+1}$', fontsize=12)

        plt.show()

    def simulate(self, k_ini_values, A=[], dograph=True):
        k_star, _ = self.find_steady_state()
        ymin, ymax = 0, (k_star + 5)

        if dograph:
            fig, ax = plt.subplots(figsize=[11, 5])
            ax.set_xlim(self.k_min, self.ts_length - 1)
            ax.set_ylim(ymin, ymax)

            ts = np.zeros(self.ts_length)

            for k_init in k_ini_values:
                ts[0] = k_init
                for t in range(1, self.ts_length):
                    ts[t] = self.visual(ts[t-1], A=self.newA[t])
                ax.plot(np.arange(self.ts_length), ts, '-o', ms=4, alpha=0.6,
                        label=r'$k_0=%g$' % k_init)
                ax.plot(np.arange(self.ts_length), np.full(self.ts_length, k_star),
                        alpha=0.6, color='red', label=r'$k^*$')

            ax.legend(fontsize=10)
            ax.set_xlabel(r'$t$', fontsize=14)
            ax.set_ylabel(r'$k_t$', fontsize=14)
            plt.title('The transition of capital back to steady state')
            plt.show()
        else:
            ts = np.zeros(self.ts_length)
            for k_init in k_ini_values:
                ts[0] = k_init
                for t in range(1, self.ts_length):
                    ts[t] = self.visual(ts[t-1], A=self.newA[t])
            self.ts_1 = ts

    def plot_combined(self, s_values):
        xgrid = np.linspace(self.k_min, self.k_max, 12000)
        fig, ax = plt.subplots(figsize=[10, 6])
        ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='$45^{\circ}$')

        for s_val in s_values:
            self.s = s_val
            k_star, _ = self.find_steady_state()
            visual_values = self.visual(xgrid)
            ax.plot(xgrid, visual_values, label=f's={s_val}')
            ax.plot(k_star, k_star, 'go', ms=10, alpha=0.6)
            ax.text(k_star, k_star, f'$k^*={k_star:.2f}$', fontsize=10, va='bottom', ha='right')

        ax.set_xlim(self.k_min, 4)
        ax.set_ylim(self.k_min, 4)
        ax.set_xticks(np.arange(self.k_min, 4 + 1, 1))
        ax.set_yticks(np.arange(self.k_min, 4 + 1, 1))
        ax.legend(loc='upper left', frameon=False, fontsize=10)
        ax.set_xlabel('$k_t$', fontsize=12)
        ax.set_ylabel('$k_{t+1}$', fontsize=12)
        plt.title('Solow Model Dynamics for Different Savings Rates')
        plt.show()

    def plot_technology_shock_combined(self):
        original_model = SolowModelWithTechProgress(alpha=self.alpha, delta=self.delta, s=self.s, n=self.n, phi=self.phi)
        original_model.find_steady_state()

        shocked_model = SolowModelWithTechProgress(alpha=self.alpha, delta=self.delta, s=self.s, n=self.n, phi=self.phi)
        shocked_model.alpha = 0.4
        for i in range(0, self.ts_length):
            shocked_model.newA[i] = 0.4
        shocked_model.find_steady_state()

        original_model.simulate([5], dograph=False)
        k_star_original = original_model.find_steady_state()[0]

        shocked_model.simulate([5], dograph=False)
        k_star_shocked = shocked_model.find_steady_state()[0]

        ymin, ymax = 0, (k_star_original + 5)

        fig, ax = plt.subplots(figsize=[11, 5])
        ax.set_xlim(self.k_min, self.ts_length - 1)
        ax.set_ylim(ymin, ymax)

        ax.plot(np.arange(self.ts_length), shocked_model.ts_1, '-o', ms=4, alpha=0.6, label="shocked")
        ax.plot(np.arange(self.ts_length), original_model.ts_1, '-o', ms=4, alpha=0.6, label="original")
        ax.plot(np.arange(self.ts_length), np.full(self.ts_length, k_star_original), alpha=0.6, color='red', label=r'$k^*$')
        ax.plot(np.arange(self.ts_length), np.full(self.ts_length, k_star_shocked), alpha=0.6, color='green', label=r'$k^*_{shock}$')
        plt.title('Technology Shock to the Solow Model with Endogenous Technological Progress')
        ax.legend(fontsize=10)
        ax.set_xlabel(r'$t$', fontsize=14)
        ax.set_ylabel(r'$k_t$', fontsize=14)
        plt.show()

def SolowHuman_analytical_capital():

    k = sm.symbols('k')
    A = sm.symbols('A')
    h = sm.symbols('h')
    alpha = sm.symbols('alpha')
    delta = sm.symbols('delta')
    phi = sm.symbols('phi')
    s_k = sm.symbols('s_k')
    s_h = sm.symbols('s_h')
    g = sm.symbols('g')
    n = sm.symbols('n')

    #Define production function in per capita, technology adjusted terms:
    f=k**alpha * h**phi

    #The capital accumulation equation:
    ss = sm.Eq(k,((s_k*f+(1-delta)*k)/((1+n)*(1+g))))

    #Solves the steady-state equation for the steady-state value of capital (k_ss) using SymPy's solve function and takes the first solution (assuming there's only one solution), storing it in the variable kss.
    kss_k = sm.solve(ss,k)[0]

    print('The steady state equation for k is:')

    display(kss_k)

def SolowHuman_analytical_humancapital():

    k = sm.symbols('k')
    A = sm.symbols('A')
    h = sm.symbols('h')
    alpha = sm.symbols('alpha')
    delta = sm.symbols('delta')
    phi = sm.symbols('phi')
    s_k = sm.symbols('s_k')
    s_h = sm.symbols('s_h')
    g = sm.symbols('g')
    n = sm.symbols('n')

    #Define production function in per capita, technology adjusted terms:
    f=k**alpha * h**phi

    #The capital accumulation equation:
    ss = sm.Eq(h,((s_h*f+(1-delta)*h)/((1+n)*(1+g))))

    #Solves the steady-state equation for the steady-state value of capital (k_ss) using SymPy's solve function and takes the first solution (assuming there's only one solution), storing it in the variable kss.
    kss_h = sm.solve(ss,h)[0]

    print('The steady state equation for h is:')

    display(kss_h)


def SolowHuman_analytical_combined():
    # Symbols
    k, h, alpha, delta, phi, s_k, s_h, g, n = sm.symbols('k h alpha delta phi s_k s_h g n')

    # Production function
    f = k**alpha * h**phi

    # Capital accumulation equation
    ss_k = sm.Eq(k, (s_k*f + (1-delta)*k)/((1+n)*(1+g)))

    # Human capital accumulation equation
    ss_h = sm.Eq(h, (s_h*f + (1-delta)*h)/((1+n)*(1+g)))

    # Substitute steady-state expressions into each other's equations
    ss_k_combined = ss_k.subs({k: sm.solve(ss_k, k)[0], h: sm.solve(ss_h, h)[0]})
    ss_h_combined = ss_h.subs({k: sm.solve(ss_k, k)[0], h: sm.solve(ss_h, h)[0]})

    print('The combined steady state equation for k is:')
    display(ss_k_combined)

    print('The combined steady state equation for h is:')
    display(ss_h_combined)

def SolowHuman_analytical_combined_test():
    # Symbols
    k, h, alpha, delta, phi, s_k, s_h, g, n = sm.symbols('k h alpha delta phi s_k s_h g n')

    # Production function
    f = k**alpha * h**phi

    # Capital accumulation equation
    ss_k = sm.Eq(k, (s_k * f + (1 - delta) * k) / ((1 + n) * (1 + g)))

    # Human capital accumulation equation
    ss_h = sm.Eq(h, (s_h * f + (1 - delta) * h) / ((1 + n) * (1 + g)))

    # Solve for steady-state k and h
    kss_expr = sm.solve(ss_k, k)[0]
    hss_expr = sm.solve(ss_h, h)[0]

    # Substitute h from hss_expr into kss_expr
    kss_combined = kss_expr.subs(h, hss_expr)
    hss_combined = hss_expr.subs(k, kss_expr)

    # Simplify the expressions
    kss_combined_simplified = sm.simplify(kss_combined)
    hss_combined_simplified = sm.simplify(hss_combined)

    print('The combined steady state equation for k is:')
    display(kss_combined_simplified)

    print('The combined steady state equation for h is:')
    display(hss_combined_simplified)

    isolated_k = sm.solve(k,kss_combined_simplified)
    print("The analytical solution to k:")
    display(isolated_k)
    isolated_h = sm.solve(h,hss_combined_simplified)
    print("The analytical solution to h:")
    display(isolated_h)



class SolowModelwithHumanCapital:
    def __init__(self):
        # Initialize parameter values as attributes of the class
        self.alpha = 0.3
        self.phi = 0.5
        self.delta = 0.1
        self.sk = 0.2  # Savings rate for capital (initial)
        self.sh = 0.1  # Savings rate for human capital (initial)
        self.A = 2.0
        self.n = 0.01  # Labor growth rate
        self.g = 0.02  # Technology growth rate

    def run(self):
        # Calculate the steady-state values for capital and human capital
        k_star = (((self.sk ** (1 - self.phi)) * (self.sh ** self.phi)) /
                  (self.n + self.g + self.delta + self.n*self.g)) ** (1 / (1 - self.alpha - self.phi))
        h_star = (((self.sk ** self.alpha) * (self.sh ** (1 - self.alpha))) /
                  (self.n + self.g + self.delta + self.n*self.g)) ** (1 / (1 - self.alpha - self.phi))

        # Print the steady-state values for capital and human capital
        print('Steady-state value for capital (k*):', k_star)
        print('Steady-state value for human capital (h*):', h_star)

    def production_function(self, k, h, A, L):
        return (k ** self.alpha) * (h ** self.phi) * ((A * L) ** (1 - self.alpha - self.phi))

    def transition_eq_capital(self, k, h, sk=None):
        sk = sk if sk is not None else self.sk
        return sk * k ** self.alpha * h ** self.phi - (self.n + self.g + self.delta + self.n*self.g) * k

    def transition_eq_human_capital(self, k, h, sh=None):
        sh = sh if sh is not None else self.sh
        return sh * k ** self.alpha * h ** self.phi - (self.n + self.g + self.delta + self.n*self.g) * h

    def plot_phase_diagram_transition_equations(self):
        # Create a grid of capital and human capital values
        k_vals = np.linspace(0.01, 2, 1000)
        h_vals = np.linspace(0.01, 2, 1000)
        K, H = np.meshgrid(k_vals, h_vals)

        # Compute the transition equations for initial savings rate (self.sk, self.sh)
        z_capital = self.transition_eq_capital(K, H)
        z_human_capital = self.transition_eq_human_capital(K, H)

        # Plot the phase diagram for initial savings rate
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(K, H, z_capital, levels=[0], colors='blue')
        ax.contour(K, H, z_human_capital, levels=[0], colors='red')

        # Add labels and title
        ax.set_xlabel('Capital ($k_{t}$)')
        ax.set_ylabel('Human Capital ($h_{t}$)')
        ax.set_title('Phase Diagram with Transition Equations')

        # Add legend
        ax.plot([], [], color='blue', label='Transition Equation for Capital ($s_k$={})'.format(self.sk))
        ax.plot([], [], color='red', label='Transition Equation for Human Capital ($s_h$={})'.format(self.sh))
        ax.legend()

        plt.show()

    def plot_phase_diagram_with_higher_savings(self, sk_high):
        # Create a grid of capital and human capital values
        k_vals = np.linspace(0.01, 5, 1000)
        h_vals = np.linspace(0.01, 5, 1000)
        K, H = np.meshgrid(k_vals, h_vals)

        # Compute the transition equation for higher savings rate (sk_high)
        z_capital_high = self.transition_eq_capital(K, H, sk=sk_high)

        # Plot the phase diagram with higher savings rate
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(K, H, z_capital_high, levels=[0], colors='green', linestyles='dashed')  # Use green color for the new curve

        # Overlay the existing contours for comparison
        z_capital = self.transition_eq_capital(K, H)  # Compute transition equation with initial savings rate
        z_human_capital = self.transition_eq_human_capital(K, H)  # Compute transition equation for human capital

        ax.contour(K, H, z_capital, levels=[0], colors='blue')
        ax.contour(K, H, z_human_capital, levels=[0], colors='red')

        # Add labels and title
        ax.set_xlabel('Capital ($k_{t}$)')
        ax.set_ylabel('Human Capital ($h_{t}$)')
        ax.set_title('Phase Diagram with Transition Equations')

        # Add legend
        ax.plot([], [], color='blue', label='Transition Equation for Capital ($s_k$={})'.format(self.sk))
        ax.plot([], [], color='red', label='Transition Equation for Human Capital ($s_h$={})'.format(self.sh))
        ax.plot([], [], color='green', linestyle='dashed', label='Transition Equation for Capital ($s_k$={})'.format(sk_high))
        ax.legend()

        plt.show()

    def plot_phase_diagram_with_higher_human_capital_savings(self, sh_high):
        # Create a grid of capital and human capital values
        k_vals = np.linspace(0.01, 10, 1000)
        h_vals = np.linspace(0.01, 10, 1000)
        K, H = np.meshgrid(k_vals, h_vals)

        # Compute the transition equation for higher human capital savings rate (sh_high)
        z_human_capital_high = self.transition_eq_human_capital(K, H, sh=sh_high)

        # Plot the phase diagram with higher human capital savings rate
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(K, H, z_human_capital_high, levels=[0], colors='purple', linestyles='dashed')  # Use purple color for the new curve

        # Overlay the existing contours for comparison
        z_capital = self.transition_eq_capital(K, H)  # Compute transition equation with initial savings rate
        z_human_capital = self.transition_eq_human_capital(K, H)  # Compute transition equation for human capital

        ax.contour(K, H, z_capital, levels=[0], colors='blue')
        ax.contour(K, H, z_human_capital, levels=[0], colors='red')

        # Add labels and title
        ax.set_xlabel('Capital ($k_{t}$)')
        ax.set_ylabel('Human Capital ($h_{t}$)')
        ax.set_title('Phase Diagram with Transition Equations')

        # Add legend
        ax.plot([], [], color='blue', label='Transition Equation for Capital ($s_k$={})'.format(self.sk))
        ax.plot([], [], color='red', label='Transition Equation for Human Capital ($s_h$={})'.format(self.sh))
        ax.plot([], [], color='purple', linestyle='dashed', label='Transition Equation for Human Capital ($s_h$={})'.format(sh_high))
        ax.legend()

        plt.show()

    def plot_phase_diagram_technology_shock(self, new_g):
        # Save the original technology growth rate
        original_g = self.g

        # Set the new technology growth rate
        self.g = new_g

        # Create a grid of capital and human capital values
        k_vals = np.linspace(0.01, 2, 1000)
        h_vals = np.linspace(0.01, 2, 1000)
        K, H = np.meshgrid(k_vals, h_vals)

        # Compute the transition equations for the new parameter values
        z_capital = self.transition_eq_capital(K, H)
        z_human_capital = self.transition_eq_human_capital(K, H)

        # Plot the phase diagram with the new parameter values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(K, H, z_capital, levels=[0], colors='blue')
        ax.contour(K, H, z_human_capital, levels=[0], colors='red')

        # Add labels and title
        ax.set_xlabel('Capital ($k_{t}$)')
        ax.set_ylabel('Human Capital ($h_{t}$)')
        ax.set_title('Phase Diagram with Transition Equations (After Technology Shock)')

        # Add legend
        ax.plot([], [], color='blue', label='Transition Equation for Capital')
        ax.plot([], [], color='red', label='Transition Equation for Human Capital')
        ax.legend()

        # Reset the technology growth rate to the original value
        self.g = original_g

        plt.show()