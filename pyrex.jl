#Calling Python routines from Julia
using PyCall

#=
@pyimport math
math.sin(math.pi / 4) - sin(pi / 4)  # returns 0.0
=#

@pyimport sklearn.linear_model as lm
XX = reshape([1:10],(10,1))
lm.LinearRegression()[:fit](XX, linspace(0,200,10))[:predict]([3])
