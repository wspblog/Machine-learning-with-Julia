using Optim   # Julia Optimizer package 
using PyCall  # to call Python optimization routines

# Import Python routines
@pyimport sklearn.linear_model as lm
 

##Load functions for this exercise
include("sigmoid.jl")
include("costFunction.jl")

# define functions for optimization
function costFunc(theta::Vector)
	m = length(y) # number of training examples
	iz = ones(m,1)
	hX = sigmoid(X * theta)
	izz = ones(size(hX))
	J = 1/m * ((-y'* log(hX)) - (iz - y)'*log(izz - (hX)))

	return J
end

##########################################################################################
# ------- Part 1  ----------------------
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
print("Loading data\n")
data = readcsv("ex2data1.txt")
X = data[:, 1:2] 
y = data[:, 3]

##########################################################################################
#= ------ TO DO - Fix plotting in Julia 
# ------ Part 1: Plotting -----------
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print("Plotting data with + indicating (y = 1) examples and o 
         'indicating (y = 0) examples.\n")
plotData(X, y)
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
=#

##########################################################################################

## ------- Part 2: Compute Cost and Gradient ============
print("\nComputing Initial cost and theta\n")
m, n = size(X)
# Add intercept term to x and X_test
X = [ones(m, 1) X]

# Initialize fitting parameters
initial_theta = zeros(n + 1, 1)

# Compute and display initial cost and gradient
icost, igrad = costFunction(initial_theta, X, y)

#Julia printf macro requires individual arguments to be specified
@printf "\nCost at initial theta (zeros): %f \n" icost[]
@printf "\nGradient at initial theta (zeros): %f %f %f \n" igrad[1] igrad[2] igrad[3]

##########################################################################################
# Use imported Python optimization functions
print("Using Python Sklearn Logistic Regression Routines")

logreg = lm.LogisticRegression(C=1e5)
res = logreg[:fit](X, y)

# Thetas
print("\nIntercept = \n", res[:intercept_])
print("\n Theta = \n", res[:coef_])

#Predictions using Fitted Log Reg
print("\nLog Reg score \n",logreg[:score](X,y) * 100)

print("\nPredict score for individual\n", testscore)
testscore = [1 45 85] 
pred = res[:predict_proba](testscore)
print("\nProbability - Class 0 , Class 1 \n",pred * 100)

pred1 = res[:predict](testscore)
print("\nProbability Student will qualtfy \n",pred1)

##########################################################################################
#=
# Plot Decision Boundary
plotDecisionBoundary(theta, X, y)

# Put some labels 
hold on
# Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

# Specified in plot order
legend('Admitted', 'Not admitted')
hold off

fprintf('\nProgram paused. Press enter to continue.\n')
pause
=#