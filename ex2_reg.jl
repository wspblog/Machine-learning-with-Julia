using PyCall  # to call Python optimization routines

# Import Python routines
@pyimport sklearn.linear_model as lm
@pyimport sklearn.preprocessing as pprep

##########################################################################################
## Functions for this program

function costFunctionReg(theta, X, y, lambda)
	m = length(y) # number of training examples
	iz = ones(m,1)
	ixx = eye(size(X,2))
	ixx[1,1] = 0
	
	hX = sigmoid(X * theta)
	izz = ones(size(hX))
	J = 1/m * ((-y'* log(hX)) - (iz - y)'*log(izz - (hX)))
	J += (lambda/(2*m))*sum((ixx * theta)' * theta)
	
	grad = 1/m * (((hX - y)' * X)') + (1/m * lambda * (ixx * theta))

	return J, grad
end

# Sigmoid Function
function sigmoid(z)
	g = 1.0 ./ (1.0 + exp(-(z)))
end

##########################################################################################
## Load Data
print("Loading data\n")
data = readcsv("ex2data2.txt")
X = data[:, 1:2] 
y = data[:, 3]

##########################################################################################
# Regularised Logistic Regression
# Add Polynomial Features using Scikit Sklearn, 
# Normalize training data using sklearn preprocessing.normalize

poly = pprep.PolynomialFeatures(6)
XX = poly[:fit_transform](X)
XXNorm = pprep.normalize(XX,axis=0)

##########################################################################################
# Calculate initial cost function value for zero weights, and regularization lambda = 1.0

m, n = size(XX) #Note XX already include X0 column
initial_theta = zeros(n, 1)
vlambda = 1.0

init_cost, init_grad = costFunctionReg(initial_theta, XX, y, vlambda)

print("\nCost at initial theta (zeros): \n", init_cost)

##########################################################################################
# Use imported Python optimization functions
print("\nUsing Python Sklearn Logistic Regression Routines")

logreg = lm.LogisticRegression(C=1e5)
res = logreg[:fit](XX, y)

# Weights
#print("\nIntercept = \n", res[:intercept_])
#print("\n Theta = \n", res[:coef_])

#Predictions using Fitted Log Reg
print("\nRegularization parameter lambda = ", 1/1e5)
@printf "\nLog Reg score on training data %8f2 \n" logreg[:score](XX,y) * 100

