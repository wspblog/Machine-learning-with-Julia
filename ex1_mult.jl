##Multivariate regression exercise in Julia
##Packages Used
#using Gadfly #Plot Package

##Load functions for this program
include("featuresNormalize.jl")
include("gradientDescentMulti.jl")
include("computeCostMulti.jl")
include("normalEqn.jl")

## Part 0: Initialization 
##Load data
data = readcsv("ex1data2.txt")
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

print("First 10 examples from the dataset: \n")
#@printf(" x = [%.0f %.0f], y = %.0f \n", X[1:10,:], y[1:10,:]')
print("\nFirst 10 X data \n", X[1:10,:])
print("\nFirst 10 y data \n", y[1:10,:])


##Part 1: Feature Normalization 
#Normalize features
print("\nNormalizing features\n")
X_norm, mu_norm, sigma_norm = featureNormalize(X)

## ------- Part 2: Gradient Descent ------------
print("Running gradient descent ...\n")
# Choose some alpha value
alpha = 0.1
num_iters = 400
X = [ones(m, 1) X_norm]

# Init Theta and Run Gradient Descent 
theta = zeros(3, 1)
t_J_History, theta_fin = gradientDescentMulti(X, y, theta, alpha, num_iters)

print("\nCost minimal = ", t_J_History[end],"\n")
print("Optimal theta = ","\n",theta_fin,"\n")


#Plot cost vs number of iterations
#plot(x=1:num_iters,y=t_J_History,Geom.line)

#----Predict price for 1650 sq ft house with 3 rooms (check if data has been normalized)
pr0 = theta_fin[1,1] * 1 
pr1 = theta_fin[2,1] * ((1650 - mu_norm[1,1])/sigma_norm[1,1])
pr2 = theta_fin[3,1] * ((3 - mu_norm[1,2])/sigma_norm[1,2])
price1 = pr0 + pr1 + pr2

print("\nPredicted price of a 1650 sq-ft, 3 br house ") 
print("\n(using gradient descent):\n", price1)

#normalize required value first - vectorized operations
prx = ([1650, 3]' - mu_norm)./sigma_norm 
price2 = theta_fin' * [1; prx']

## ================ Part 3: Normal Equations ================
#Note that feature scaling does not apply for normal equations
#Reload data
data = readcsv("ex1data2.txt")
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

# Add intercept term to X
X = [ones(m, 1) X]

# Calculate the parameters from the normal equation
theta_norm = normalEqn(X, y)
print("\nNormal Equation theta =\n",theta_norm)

# Predict price for 1650 sq ft with 3 room
price3 = theta_norm' * [1, 1650, 3]

print("\nPredicted price of a 1650 sq-ft, 3 br house ") 
print("\n(using normal equation):\n", price3)
 