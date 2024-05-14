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
                alpha=0.6, color='green', label=r'$k^*$_shock')
        ax.legend(fontsize=10)

        # Set labels for axes
        ax.set_xlabel(r'$t$', fontsize=14)  # Change x-axis label
        ax.set_ylabel(r'$k_t$', fontsize=14)

        # Show plot
        plt.show()


"""Extensions:"""

def SolowHuman_analytical_capital():

    k = sm.symbols('k')
    A = sm.symbols('A')
    h = sm.symbols('h')
    alpha = sm.symbols('alpha')
    delta = sm.symbols('delta')
    phi = sm.symbols('phi')
    sk = sm.symbols('sk')
    sh = sm.symbols('sh')
    g = sm.symbols('g')
    n = sm.symbols('n')

    #Define production function in per capita, technology adjusted terms:
    f=k**alpha * h**phi

    #The capital accumulation equation:
    ss = sm.Eq(k,((sk*f+(1-delta)*k)/((1+n)*(1+g))))

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
    sk = sm.symbols('sk')
    sh = sm.symbols('sh')
    g = sm.symbols('g')
    n = sm.symbols('n')

    #Define production function in per capita, technology adjusted terms:
    f=k**alpha * h**phi

    #The capital accumulation equation:
    ss = sm.Eq(h,((sh*f+(1-delta)*h)/((1+n)*(1+g))))

    #Solves the steady-state equation for the steady-state value of capital (k_ss) using SymPy's solve function and takes the first solution (assuming there's only one solution), storing it in the variable kss.
    kss_h = sm.solve(ss,h)[0]

    print('The steady state equation for h is:')

    display(kss_h)


def SolowHuman_analytical_combined():
    # Symbols
    k, h, alpha, delta, phi, sk, sh, g, n = sm.symbols('k h alpha delta phi sk sh g n')

    # Production function
    f = k**alpha * h**phi

    # Capital accumulation equation
    ss_k = sm.Eq(k, (sk*f + (1-delta)*k)/((1+n)*(1+g)))

    # Human capital accumulation equation
    ss_h = sm.Eq(h, (sh*f + (1-delta)*h)/((1+n)*(1+g)))

    # Substitute steady-state expressions into each other's equations
    ss_k_combined = ss_k.subs({k: sm.solve(ss_k, k)[0], h: sm.solve(ss_h, h)[0]})
    ss_h_combined = ss_h.subs({k: sm.solve(ss_k, k)[0], h: sm.solve(ss_h, h)[0]})

    print('The combined steady state equation for k is:')
    display(ss_k_combined)

    print('The combined steady state equation for h is:')
    display(ss_h_combined)


class SolowModelwithHumanCapital:
    def __init__(self):
        # Initialize parameter values as attributes of the class
        self.alpha = 0.3
        self.phi = 0.2
        self.delta = 0.1
        self.sk = 0.2  # Savings rate for capital
        self.sh = 0.2  # Savings rate for human capital
        self.A = 2.0
        self.n = 0.02  # Labor growth rate
        self.g = 0.02  # Technology growth rate

    def run(self):
        # Calculate the steady-state values for capital and human capital
        #k_star = ((self.sk * self.A) / self.delta) ** (1 / (1 - self.alpha))
        #h_star = ((self.sh * self.A) / self.delta) ** (1 / (1 - self.phi))

        k_star = (((self.sk ** (1-self.phi))*(self.sh**self.phi)) / (self.n + self.g + self.delta + self.n*self.g))**(1/(1-self.alpha-self.phi))
        h_star = (((self.sk ** self.alpha) * (self.sh ** (1 - self.alpha))) / (self.n + self.g + self.delta + self.n * self.g)) ** (1 / (1 - self.alpha - self.phi))

        # Print the steady-state values for capital and human capital
        print('Steady-state value for capital (k*):', k_star)
        print('Steady-state value for human capital (h*):', h_star)

    def production_function(self, k, h, A, L):
        return (k ** self.alpha) * (h ** self.phi) * ((A * L) ** (1 - self.alpha - self.phi))

    def transition_eq_capital(self, k, h):
        return self.sk * k ** self.alpha * h ** self.phi - (self.n + self.g + self.delta + self.n * self.g) * h

    def transition_eq_human_capital(self, k, h):
        return self.sh * k ** self.alpha * h ** self.phi - (self.n + self.g + self.delta + self.n * self.g) * k

    def plot_phase_diagram_transition_equations(self):
        # Create a grid of capital and human capital values
        k_vals = np.linspace(0.01, 5, 1000)
        h_vals = np.linspace(0.01, 5, 1000)
        K, H = np.meshgrid(k_vals, h_vals)

        # Compute the transition equations
        z_capital = self.transition_eq_capital(K, H)
        z_human_capital = self.transition_eq_human_capital(K, H)

        # Plot the phase diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(K, H, z_capital, levels=[0], colors='blue')
        ax.contour(K, H, z_human_capital, levels=[0], colors='red')

        # Add labels and title
        ax.set_xlabel('Capital ($k_{t}$)')
        ax.set_ylabel('Human Capital ($h_{t}$)')
        ax.set_title('Phase Diagram with Transition Equations')

        # Add legend
        ax.plot([], [], color='blue', label='Transition Equation for Capital')
        ax.plot([], [], color='red', label='Transition Equation for Human Capital')
        ax.legend()

        plt.show()
