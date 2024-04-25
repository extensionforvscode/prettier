from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Define the structure of the Bayesian Network
model = BayesianNetwork([('e', 'm'), ('i', 'm'), ('i', 's'), ('m', 'a')])

'''
                    Student
                      |
                      v
exam ---> marks <--- IQ
           |
           v
      admission
'''

# Define Conditional Probability Distributions (CPDs)

cpd_e = TabularCPD(variable='e', variable_card=2, values=[[0.6], [0.4]])
cpd_i = TabularCPD(variable='i', variable_card=2, values=[[0.7], [0.3]])
cpd_m = TabularCPD(variable='m', variable_card=2,
                   evidence=['e', 'i'],
                   values=[[0.9, 0.6, 0.7, 0.1],
                           [0.1, 0.4, 0.3, 0.9]],
                   evidence_card=[2, 2])
cpd_s = TabularCPD(variable='s', variable_card=2,
                   evidence=['i'],
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence_card=[2])
cpd_a = TabularCPD(variable='a', variable_card=2,
                   evidence=['m'],
                   values=[[0.8, 0.1],
                           [0.2, 0.9]],
                   evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_e, cpd_i, cpd_m, cpd_s, cpd_a)

# Check if the model is consistent
print("Model is consistent:", model.check_model())

# Print CPDs
for cpd in model.get_cpds():
    print("CPD for {}: \n{}".format(cpd.variable, cpd))

# Doing exact inference using Variable Elimination
from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

# Calculate the probability of admission given marks=1
print("\nProbability of admission given marks=1:")
print(infer.query(variables=['a'], evidence={'m': 1}).values[0])

# Calculate the probability of marks given exam=0 and IQ=1
print("\nProbability of marks given exam=0 and IQ=1:")
print(infer.query(variables=['m'], evidence={'e': 0, 'i': 1}).values)

