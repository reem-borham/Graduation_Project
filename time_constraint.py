from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import  StatevectorSampler as Sampler

# ============================================
# Create optimization problems
# ============================================
qp = QuadraticProgram(name="multi_hour_energy_optimization")

time_slots = ["morning", "afternoon"]

# Real battery fractions (used for cost)
battery_fraction = {
    "low": 0.25,
    "mid": 0.50,
    "high": 0.75
}

# Integer units (used for constraints)
battery_units = {
    "low": 1,
    "mid": 2,
    "high": 3
}

# ============================================
# Variables
# ============================================
for t in time_slots:
    for level in battery_fraction:
        qp.binary_var(name=f"battery_{level}_{t}")

    qp.binary_var(name=f"grid_{t}")
    qp.binary_var(name=f"ac_off_{t}")

# ============================================
# Objective function
# ============================================
GRID_COST = 0.15
BATTERY_WEAR = 0.05
CARBON_WEIGHT = 0.3
GRID_CARBON = 0.4
AC_SAVINGS = 0.20

linear_costs = {}

for t in time_slots:
    linear_costs[f"grid_{t}"] = GRID_COST + CARBON_WEIGHT * GRID_CARBON
    linear_costs[f"ac_off_{t}"] = -AC_SAVINGS

    for level, frac in battery_fraction.items():
        linear_costs[f"battery_{level}_{t}"] = BATTERY_WEAR * frac

qp.minimize(linear=linear_costs)

# ============================================
# Constraints
# ============================================

# (1) at most one battery level each time slot
for t in time_slots:
    qp.linear_constraint(
        linear={f"battery_{level}_{t}": 1 for level in battery_fraction},
        sense="<=",
        rhs=1,
        name=f"one_battery_level_{t}"
    )

# (2) Must have battery or grid
for t in time_slots:
    terms = {f"battery_{level}_{t}": 1 for level in battery_fraction}
    terms[f"grid_{t}"] = 1

    qp.linear_constraint(
        linear=terms,
        sense="==",
        rhs=1,
        name=f"must_have_power_{t}"
    )

# (3) Total battery budget (integer coefficients)
total_battery_terms = {}

for t in time_slots:
    for level, units in battery_units.items():
        total_battery_terms[f"battery_{level}_{t}"] = units

qp.linear_constraint(
     linear=total_battery_terms,
     sense="<=",
     rhs=4,        # 100% battery = 4 units
     name="total_battery_budget"
    )
print(qp.prettyprint())

# ============================================
# Solve with QAOA
# ============================================
print("Building QAOA...")
qaoa = QAOA(
    sampler=Sampler(),
    optimizer=COBYLA(maxiter=3),
    reps=1
)
print(qaoa)
print("Creating optimizer...")
optimizer = MinimumEigenOptimizer(qaoa)
print("hello")

print("Binary variables:", qp.get_num_binary_vars())
print("Linear constraints:", qp.get_num_linear_constraints())
print(qp.prettyprint())
result = optimizer.solve(qp)


# ============================================
# Display Results
# ============================================

print("\n========== SOLUTION ==========")

for t in time_slots:
    print(f"\n{t.upper()}")

    chosen = None

    for level in battery_fraction:
        if result.variables_dict[f"battery_{level}_{t}"] == 1:
            chosen = level

    print(f"Battery Level : {chosen}")

    print(
        f"Grid          : {'YES' if result.variables_dict[f'grid_{t}'] else 'NO'}"
    )

    print(
        f"AC Off        : {'YES' if result.variables_dict[f'ac_off_{t}'] else 'NO'}"
    )

print("\nObjective value:", result.fval)