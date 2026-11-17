
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== choose product: 'Ozempic' or 'Wegovy' or 'Cavalitide' =====
#drug = 'Ozempic'
#drug = 'Wegovy'
drug = 'Cavalitide'


# ==========================================
# PRODUCT-SPECIFIC PARAMETERS
# ==========================================

products = {
    'Ozempic': dict(Favailable=0.89,        # bioavailability
                    ka=0.029,               # absorption constant in 1/h
                    D=2.0,                  # dosage in mg
                    interval=168,           # in hours
                    n_doses=15,             
                    depot='subcutaneous'),  # delivery method

    'Wegovy': dict(Favailable=0.01,
                   ka=0.0029,
                   D=25,
                   interval=24,
                   n_doses=100,
                   depot='gut (oral)'),

    'Cavalitide': dict(Favailable=0.01,
                       ka=0.006,
                       D=21,
                       interval=24,
                       n_doses=100,
                       depot='gut (oral)')
}

p = products[drug]

Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']


# ==========================================
# SHARED MODEL PARAMETERS
# ==========================================

ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # free fraction in blood
dt = 0.1

# ==========================================
# PHARMACOKINETIC MODEL
# ==========================================

def ODEfunc(t, y, ka):
    Asc, Cc = y
    dAsc = -ka*Asc            # Asc: depot amount (mg)
    dCc = ka*Asc/V - ke*Cc    # Cc: blood concentration (mg/L)
    return [dAsc, dCc]


def simulate(Favailable, ka, D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    and blood concentration (nmol/L)."""
    t_out, Asc_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))
    for i in range(n_doses):
        y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)

        # CHANGED: solve_ivp replaces the explicit Euler loop
        sol = solve_ivp(ODEfunc, (t_seg[0], t_seg[-1]), y, t_eval=t_seg,
                        args=(ka,), rtol=1e-6)
        y_seg = sol.y

        t_out.append(t_seg)
        Asc_out.append(y_seg[0])
        Cc_out.append(y_seg[1])
        y = y_seg[:, -1].copy()         # carry state into next interval
    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)
    Cc_nM = Cc_out * 1e-3 / MW * 1e9
    return t, Asc_out, Cc_nM


# ==========================================
# PLOT 1: PHARMACODYNAMICS
# COMPARE ALL THREE DRUGS
# ==========================================

Kd1 = 0.17       # nM, GLP-1
L1 = 0.015       # nM, GLP-1 concentration

# Kd2 values for each drug
Kd2_values = {
    'Ozempic': 0.36,       # nM, semaglutide
    'Wegovy': 0.36,        # nM, semaglutide
    'Cavalitide': 0.077    # nM, Cavalitide
}


plt.figure(figsize=(10, 6))

for drug_name, p in products.items():

    # Load drug-specific parameters
    Favailable = p['Favailable']
    ka = p['ka']
    D = p['D']
    interval = p['interval']
    n_doses = p['n_doses']
    dt = 0.1
    # Run pharmacokinetic simulation
    t_drug, Asc_drug, Cc_drug = simulate(Favailable, ka, D, n_doses, interval, dt)

    # Free blood concentration (nM)
    L2 = Ffree * Cc_drug

    # Drug-specific dissociation constant
    Kd2 = Kd2_values[drug_name]

    # Calculate fractional GLP1R activation
    Fact = (
        (Kd2 * L1 + Kd1 * L2) /
        (Kd1 * Kd2 + Kd2 * L1 + Kd1 * L2)
    )

    # Plot pharmacodynamic response
    plt.plot(
        t_drug / 24,
        Fact,
        label=f"{drug_name} ({p['D']}mg)  (Kd2 = {Kd2} nM)"
    )


plt.xlabel("Time (days)")
plt.ylabel("Active Fraction of GLP1R")

plt.title("GLP1R Active Fraction Over Time: Drug Comparison")

plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)

plt.show()

# ==========================================
# PLOT 2: DOSAGE ADJUSTMENT PLOT
# ==========================================

t, Asc, Cc = simulate(
       0.01, 0.006 ,21, 100, 24, 0.1
    ) # predicts blood concentration over time for original dose of 21mg cavalitde

t_adj, Asc_adj, Cc_adj = simulate(
        0.01, 0.006,28, 100, 24, 0.1
    ) # predicts blood concentration over time for adjusted dose of 28mg cavalitde

L2 = Ffree * Cc
L2_adj = Ffree * Cc_adj

Kd2 = 0.077
Fact = (
        (Kd2 * L1 + Kd1 * L2) /
        (Kd1 * Kd2 + Kd2 * L1 + Kd1 * L2)
    )
Fact_adj = (
        (Kd2 * L1 + Kd1 * L2_adj) /
        (Kd1 * Kd2 + Kd2 * L1 + Kd1 * L2_adj)
    )
plt.plot(
        t_drug / 24,
        Fact,
        label=f"Cavalitide (21mg)"
    )
plt.plot(
        t_drug / 24,
        Fact_adj,
        label=f"Adjusted Cavalitide dose (28mg)"
    )

plt.axhline(y=.9, label="Target (0.9)", linestyle='dotted')

plt.xlabel("Time (days)")
plt.ylabel("Active Fraction of GLP1R")

plt.title("GLP1R Active Fraction Over Time: Drug Dosage Comparison")

plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)

plt.show()