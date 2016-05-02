#=MATLAB functions written in Julia
based on Machine Learning exercises
=#

##Packages Used
#using Gadfly #Plot Package
#using Winston #2D Plot Package

#using Convex

##########################################################################################
#User defined functions
#include("computeCost.jl")
function computeCost(X, y, theta)
	m = length(y) # number of training examples	
	J = 1/(2*m)*(sum(((X * theta) - y).^2))
end

#include("gradientDescent.jl")
function gradientDescent(X, y, theta, alpha, num_iters)
	m = length(y) # number of training examples
	J_history = zeros(num_iters, 1)

	for iter = 1:num_iters
	    theta = theta - (alpha/m) * (((X * theta) - y)' * X)' 
	#   % Save the cost J in every iteration    
    	J_history[iter, 1] = computeCost(X, y, theta)
	end

	return theta, J_history
end

##########################################################################################
##Data set up
data = readcsv("ex1data1.txt")
X = data[:, 1]
y = data[:, 2]
m = length(y)  #number of training examples

#Gradient Descent
X = [ones(m, 1) data[:,1]]    # Add a column of ones to x
theta = zeros(2, 1)           # initialize fitting parameters

#Some gradient descent settings
print("Gradient Descent")
iterations = 1500
alpha = 0.01

#Compute and display initial cost
initcost = computeCost(X, y, theta)
print("\nInitial cost = ", initcost)

#Run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print("\nOptimal Theta = ", theta)
print("\nLast cost ", J_history[1500])

##########################################################################################
# Predict values
# Predict values for population sizes of 35,000 and 70,000
predict1 = [1 3.5] * theta
print("\n\nFor population = 35,000, we predict a profit of \n",
    predict1*10000)
predict2 = [1 7] * theta
print("\n\nFor population = 70,000, we predict a profit of \n",
    predict2*10000)
    
#########################################################################################
#=
# Solution using Convex, SCS solver
# Create a (column vector) variable of size n x 1.
theta = Variable(2)

# The problem is to minimize ||Ax - b||^2 subject to x >= 0
# This can be done by: minimize(objective, constraints)
problem = minimize(sumsquares(X * theta - y))

# Solve the problem by calling solve!
solve!(problem)

# Check the status of the problem
problem.status # :Optimal, :Infeasible, :Unbounded etc.

# Get the optimum value
problem.optval
mincost = problem.optval/(2 * m)

print("\nOptimal Theta from Convex solver\n", theta.value)
print("\nMinimum cost = \n", mincost)

=#