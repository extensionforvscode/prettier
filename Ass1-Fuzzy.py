import numpy as np

# Union Intersection and Complement
X = [1,2,3,4,5]
A = {1:0.2,2:0.8,3:0.6,4:0.3,5:0.7}
B = {1:0.5,2:0.6,3:0.1,4:0.2,5:0.9}

def union(X,A,B):
    temp ={}
    for i in X:
        temp[i] = max(A[i],B[i])
    return temp
def intersection(X,A,B):
    temp ={}
    for i in X:
        temp[i] = min(A[i],B[i])
    return temp
def complement(A):
    temp ={}
    for i in X:
        temp[i] = 1-A[i]
    return temp
print("Union of A and B : "+ str(union(X,A,B)))
print("Intersection of A and B : "+ str(intersection(X,A,B)))
print("Complement of A : "+ str(complement(A)))

# Cartesian Product
def fuzzy_cartesian_product(A, B):
    A_matrix = np.array(list(A.values()))
    B_matrix = np.array(list(B.values()))
    AXB = np.zeros((A_matrix.shape[0], B_matrix.shape[0]))
    for i, a_val in enumerate(A_matrix):
        for j, b_val in enumerate(B_matrix):
            AXB[i][j] = min(a_val, b_val)
    return AXB

AXB = fuzzy_cartesian_product(A, B)

print("Fuzzy Cartesian Product (A × B):")
print(AXB)
def max_min_composition(RA, RB):
    RA = np.array(RA)
    RB = np.array(RB)

    X, Y = RA.shape
    Y2, Z = RB.shape

    if Y != Y2:
        print("Number of columns in RA must equal number of rows in RB")

    Result = np.zeros((X, Z))

    for i in range(X):
        for j in range(Z):
            max_val = 0
            for k in range(Y):
                temp = min(RA[i][k], RB[k][j])
                if temp > max_val:
                    max_val = temp
            Result[i][j] = max_val

    return Result

RA = np.array([[0.2, 0.8, 0.6],
               [0.5, 0.3, 0.9]])

RB = np.array([[0.7, 0.4],
               [0.1, 0.8],
               [0.5, 0.2]])

Result = max_min_composition(RA, RB)

print("Max–Min Composition (RA ○ RB):")
print(Result)

"""
================================================================================
REAL-WORLD SCENARIO: MEDICAL DIAGNOSIS SYSTEM
================================================================================

PROBLEM CONTEXT:
In healthcare, symptoms and diseases rarely have binary (yes/no) relationships.
A patient may have "slightly elevated" temperature, "moderate" chest pain, or 
"somewhat irregular" heartbeat. Traditional hard computing requires precise 
thresholds (e.g., fever = temperature > 38°C), which can miss nuanced cases.

SCENARIO APPLICATION:
- Set A: Patient symptoms with membership degrees (0.0 to 1.0)
  Example: {Fever: 0.8, Cough: 0.6, Fatigue: 0.3, Headache: 0.7}
- Set B: Disease indicators with membership degrees
  Example: {Flu: 0.5, COVID-19: 0.9, Common Cold: 0.2, Pneumonia: 0.6}

FUZZY OPERATIONS USED:
1. UNION: Combining symptoms from multiple observations
   - When symptoms overlap across different test results
   
2. INTERSECTION: Finding common symptom patterns
   - Identifying symptoms present in both morning and evening check-ups
   
3. COMPLEMENT: Identifying absence of symptoms
   - Understanding what symptoms are NOT present helps rule out diseases
   
4. CARTESIAN PRODUCT: Symptom-Disease relationship mapping
   - Maps each symptom to each disease with correlation strength
   - Creates a comprehensive diagnostic matrix
   
5. MAX-MIN COMPOSITION: Multi-stage diagnosis
   - First relation: Symptoms → Test Results
   - Second relation: Test Results → Diseases
   - Composition: Direct Symptoms → Diseases pathway

SOFT COMPUTING ADVANTAGES OVER HARD COMPUTING:
================================================================================

1. HANDLING UNCERTAINTY & IMPRECISION:
   - Soft: Represents "moderate fever" as membership 0.6
   - Hard: Requires exact temperature value; misses borderline cases
   - Critical: Medical symptoms are inherently vague and patient-reported

2. PARTIAL TRUTH REPRESENTATION:
   - Soft: A patient can have 0.7 membership in "high risk" and 0.4 in "low risk"
   - Hard: Binary classification forces patients into single category
   - Critical: Early disease stages show mixed indicators

3. SMOOTH TRANSITIONS:
   - Soft: Gradual transition between health states (healthy → borderline → sick)
   - Hard: Sudden jumps at thresholds can cause misdiagnosis
   - Critical: Disease progression is continuous, not discrete

4. EXPERT KNOWLEDGE INTEGRATION:
   - Soft: Easily encodes doctor's linguistic terms ("slightly elevated", "very high")
   - Hard: Requires precise mathematical models that may not capture expertise
   - Critical: Medical expertise is often expressed in fuzzy linguistic terms

5. HANDLING INCOMPLETE DATA:
   - Soft: Can process partial symptom information with reduced certainty
   - Hard: Often fails or gives incorrect results with missing data
   - Critical: Not all tests may be available immediately

6. CONTEXT-AWARE DECISION MAKING:
   - Soft: Same symptom severity interpreted differently for different age groups
   - Hard: Rigid rules don't adapt to patient context
   - Critical: Age, weight, medical history affect diagnosis

KEY PARAMETERS ENABLING SOFT COMPUTING BENEFITS:
================================================================================
- Membership Functions: Map real values to [0,1] degrees of belonging
- Fuzzy Operations (min/max): Preserve gradual transitions
- Composition Rules: Chain multiple uncertain relationships
- Threshold Flexibility: Adjust sensitivity without rewriting entire system

CRITICAL CONSIDERATIONS:
================================================================================
1. MEMBERSHIP FUNCTION DESIGN:
   - Must accurately reflect real-world phenomena
   - Requires domain expert input for validity
   - Incorrect design leads to poor decisions

2. COMPUTATIONAL OVERHEAD:
   - Fuzzy operations more computationally expensive than crisp logic
   - Trade-off between precision and processing speed
   - Important in real-time critical systems

3. RULE BASE EXPLOSION:
   - Large number of variables → exponential growth in rules
   - Careful feature selection and rule optimization needed

4. INTERPRETABILITY:
   - Results must be explainable to medical professionals
   - Membership degrees should have clear medical meaning
   - Regulatory compliance requires transparent reasoning

5. VALIDATION CHALLENGES:
   - Difficult to validate against crisp ground truth
   - Requires expert agreement on membership degrees
   - Clinical trials needed to prove effectiveness

6. INTEGRATION WITH EXISTING SYSTEMS:
   - Legacy systems use crisp values
   - Need defuzzification for interfacing with traditional software
   - Data format conversions may introduce errors

PRACTICAL IMPACT:
================================================================================
✓ Early Detection: Catches borderline cases missed by threshold-based systems
✓ Reduced False Negatives: Gradual transitions prevent sudden classification errors
✓ Patient-Specific Care: Adapts to individual symptom patterns
✓ Decision Support: Assists doctors with nuanced, explainable recommendations
✓ Risk Assessment: Provides probability-like certainty measures

CONCLUSION:
Fuzzy logic excels in medical diagnosis because healthcare data is inherently
uncertain, imprecise, and linguistically expressed. It bridges the gap between
precise mathematical models and real-world medical practice.
"""