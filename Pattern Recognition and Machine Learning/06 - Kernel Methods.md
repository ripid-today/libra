# 06 - Kernel Methods
*Pages 291-324 from Pattern Recognition and Machine Learning*

---
**Page 291**
274
5. NEURAL NETWORKS
x1
xD
θ1
θM
θ
t
p(t|x)
Figure 5.20
The mixture density network can represent general conditional probability densities p(t|x)
by considering a parametric mixture model for the distribution of t whose parameters are
determined by the outputs of a neural network that takes x as its input vector.
the outputs of a conventional neural network that takes x as its input. The structure
of this mixture density network is illustrated in Figure 5.20. The mixture density
network is closely related to the mixture of experts discussed in Section 14.5.3. The
principle difference is that in the mixture density network the same function is used
to predict the parameters of all of the component densities as well as the mixing co-
efﬁcients, and so the nonlinear hidden units are shared amongst the input-dependent
functions.
The neural network in Figure 5.20 can, for example, be a two-layer network
having sigmoidal (‘tanh’) hidden units. If there are L components in the mixture
model (5.148), and if t has K components, then the network will have L output unit
activations denoted by aπ
k that determine the mixing coefﬁcients πk(x), K outputs
denoted by aσ
k that determine the kernel widths σk(x), and L × K outputs denoted
by aµ
kj that determine the components µkj(x) of the kernel centres µk(x). The total
number of network outputs is given by (K + 2)L, as compared with the usual K
outputs for a network, which simply predicts the conditional means of the target
variables.
The mixing coefﬁcients must satisfy the constraints
K

k=1
πk(x) = 1,
0 ⩽πk(x) ⩽1
(5.149)
which can be achieved using a set of softmax outputs
πk(x) =
exp(aπ
k)
K
l=1 exp(aπ
l )
.
(5.150)
Similarly, the variances must satisfy σ2
k(x) ⩾0 and so can be represented in terms
of the exponentials of the corresponding network activations using
σk(x) = exp(aσ
k).
(5.151)
Finally, because the means µk(x) have real components, they can be represented


---
**Page 292**
5.6. Mixture Density Networks
275
directly by the network output activations
µkj(x) = aµ
kj.
(5.152)
The adaptive parameters of the mixture density network comprise the vector w
of weights and biases in the neural network, that can be set by maximum likelihood,
or equivalently by minimizing an error function deﬁned to be the negative logarithm
of the likelihood. For independent data, this error function takes the form
E(w) = −
N

n=1
ln
 k

k=1
πk(xn, w)N

tn|µk(xn, w), σ2
k(xn, w)


(5.153)
where we have made the dependencies on w explicit.
In order to minimize the error function, we need to calculate the derivatives of
the error E(w) with respect to the components of w. These can be evaluated by
using the standard backpropagation procedure, provided we obtain suitable expres-
sions for the derivatives of the error with respect to the output-unit activations. These
represent error signals δ for each pattern and for each output unit, and can be back-
propagated to the hidden units and the error function derivatives evaluated in the
usual way. Because the error function (5.153) is composed of a sum of terms, one
for each training data point, we can consider the derivatives for a particular pattern
n and then ﬁnd the derivatives of E by summing over all patterns.
Because we are dealing with mixture distributions, it is convenient to view the
mixing coefﬁcients πk(x) as x-dependent prior probabilities and to introduce the
corresponding posterior probabilities given by
γk(t|x) =
πkNnk
K
l=1 πlNnl
(5.154)
where Nnk denotes N (tn|µk(xn), σ2
k(xn)).
The derivatives with respect to the network output activations governing the mix-
ing coefﬁcients are given by
Exercise 5.34
∂En
∂aπ
k
= πk −γk.
(5.155)
Similarly, the derivatives with respect to the output activations controlling the com-
ponent means are given by
Exercise 5.35
∂En
∂aµ
kl
= γk
µkl −tl
σ2
k

.
(5.156)
Finally, the derivatives with respect to the output activations controlling the compo-
nent variances are given by
Exercise 5.36
∂En
∂aσ
k
= −γk
∥t −µk∥2
σ3
k
−1
σk

.
(5.157)


---
**Page 293**
276
5. NEURAL NETWORKS
Figure 5.21
(a) Plot of the mixing
coefﬁcients πk(x) as a function of
x for the three kernel functions in a
mixture density network trained on
the data shown in Figure 5.19. The
model has three Gaussian compo-
nents, and uses a two-layer multi-
layer perceptron with ﬁve ‘tanh’ sig-
moidal units in the hidden layer, and
nine outputs (corresponding to the 3
means and 3 variances of the Gaus-
sian components and the 3 mixing
coefﬁcients). At both small and large
values of x, where the conditional
probability density of the target data
is unimodal, only one of the ker-
nels has a high value for its prior
probability, while at intermediate val-
ues of x, where the conditional den-
sity is trimodal, the three mixing co-
efﬁcients have comparable values.
(b) Plots of the means µk(x) using
the same colour coding as for the
mixing coefﬁcients.
(c) Plot of the
contours of the corresponding con-
ditional probability density of the tar-
get data for the same mixture den-
sity network.
(d) Plot of the ap-
proximate conditional mode, shown
by the red points, of the conditional
density.
0
1
0
1
(a)
0
1
0
1
(b)
(c)
0
1
0
1
0
1
0
1
(d)
We illustrate the use of a mixture density network by returning to the toy ex-
ample of an inverse problem shown in Figure 5.19. Plots of the mixing coefﬁ-
cients πk(x), the means µk(x), and the conditional density contours corresponding
to p(t|x), are shown in Figure 5.21. The outputs of the neural network, and hence the
parameters in the mixture model, are necessarily continuous single-valued functions
of the input variables. However, we see from Figure 5.21(c) that the model is able to
produce a conditional density that is unimodal for some values of x and trimodal for
other values by modulating the amplitudes of the mixing components πk(x).
Once a mixture density network has been trained, it can predict the conditional
density function of the target data for any given value of the input vector. This
conditional density represents a complete description of the generator of the data, so
far as the problem of predicting the value of the output vector is concerned. From
this density function we can calculate more speciﬁc quantities that may be of interest
in different applications. One of the simplest of these is the mean, corresponding to
the conditional average of the target data, and is given by
E [t|x] =

tp(t|x) dt =
K

k=1
πk(x)µk(x)
(5.158)


---
**Page 294**
5.7. Bayesian Neural Networks
277
where we have used (5.148). Because a standard network trained by least squares
is approximating the conditional mean, we see that a mixture density network can
reproduce the conventional least-squares result as a special case. Of course, as we
have already noted, for a multimodal distribution the conditional mean is of limited
value.
We can similarly evaluate the variance of the density function about the condi-
tional average, to give
Exercise 5.37
s2(x)
=
E

∥t −E[t|x]∥2 |x
	
(5.159)
=
K

k=1
πk(x)
⎧
⎨
⎩σ2
k(x) +
'''''µk(x) −
K

l=1
πl(x)µl(x)
'''''
2⎫
⎬
⎭(5.160)
where we have used (5.148) and (5.158). This is more general than the corresponding
least-squares result because the variance is a function of x.
We have seen that for multimodal distributions, the conditional mean can give
a poor representation of the data. For instance, in controlling the simple robot arm
shown in Figure 5.18, we need to pick one of the two possible joint angle settings
in order to achieve the desired end-effector location, whereas the average of the two
solutions is not itself a solution. In such cases, the conditional mode may be of
more value. Because the conditional mode for the mixture density network does not
have a simple analytical solution, this would require numerical iteration. A simple
alternative is to take the mean of the most probable component (i.e., the one with the
largest mixing coefﬁcient) at each value of x. This is shown for the toy data set in
Figure 5.21(d).
5.7. Bayesian Neural Networks
So far, our discussion of neural networks has focussed on the use of maximum like-
lihood to determine the network parameters (weights and biases). Regularized max-
imum likelihood can be interpreted as a MAP (maximum posterior) approach in
which the regularizer can be viewed as the logarithm of a prior parameter distribu-
tion. However, in a Bayesian treatment we need to marginalize over the distribution
of parameters in order to make predictions.
In Section 3.3, we developed a Bayesian solution for a simple linear regression
model under the assumption of Gaussian noise. We saw that the posterior distribu-
tion, which is Gaussian, could be evaluated exactly and that the predictive distribu-
tion could also be found in closed form. In the case of a multilayered network, the
highly nonlinear dependence of the network function on the parameter values means
that an exact Bayesian treatment can no longer be found. In fact, the log of the pos-
terior distribution will be nonconvex, corresponding to the multiple local minima in
the error function.
The technique of variational inference, to be discussed in Chapter 10, has been
applied to Bayesian neural networks using a factorized Gaussian approximation


---
**Page 295**
278
5. NEURAL NETWORKS
to the posterior distribution (Hinton and van Camp, 1993) and also using a full-
covariance Gaussian (Barber and Bishop, 1998a; Barber and Bishop, 1998b). The
most complete treatment, however, has been based on the Laplace approximation
(MacKay, 1992c; MacKay, 1992b) and forms the basis for the discussion given here.
We will approximate the posterior distribution by a Gaussian, centred at a mode of
the true posterior. Furthermore, we shall assume that the covariance of this Gaus-
sian is small so that the network function is approximately linear with respect to the
parameters over the region of parameter space for which the posterior probability is
signiﬁcantly nonzero. With these two approximations, we will obtain models that
are analogous to the linear regression and classiﬁcation models discussed in earlier
chapters and so we can exploit the results obtained there. We can then make use of
the evidence framework to provide point estimates for the hyperparameters and to
compare alternative models (for example, networks having different numbers of hid-
den units). To start with, we shall discuss the regression case and then later consider
the modiﬁcations needed for solving classiﬁcation tasks.
5.7.1
Posterior parameter distribution
Consider the problem of predicting a single continuous target variable t from
a vector x of inputs (the extension to multiple targets is straightforward). We shall
suppose that the conditional distribution p(t|x) is Gaussian, with an x-dependent
mean given by the output of a neural network model y(x, w), and with precision
(inverse variance) β
p(t|x, w, β) = N(t|y(x, w), β−1).
(5.161)
Similarly, we shall choose a prior distribution over the weights w that is Gaussian of
the form
p(w|α) = N(w|0, α−1I).
(5.162)
For an i.i.d. data set of N observations x1, . . . , xN, with a corresponding set of target
values D = {t1, . . . , tN}, the likelihood function is given by
p(D|w, β) =
N

n=1
N(tn|y(xn, w), β−1)
(5.163)
and so the resulting posterior distribution is then
p(w|D, α, β) ∝p(w|α)p(D|w, β).
(5.164)
which, as a consequence of the nonlinear dependence of y(x, w) on w, will be non-
Gaussian.
We can ﬁnd a Gaussian approximation to the posterior distribution by using the
Laplace approximation. To do this, we must ﬁrst ﬁnd a (local) maximum of the
posterior, and this must be done using iterative numerical optimization. As usual, it
is convenient to maximize the logarithm of the posterior, which can be written in the


---
**Page 296**
5.7. Bayesian Neural Networks
279
form
ln p(w|D) = −α
2 wTw −β
2
N

n=1
{y(xn, w) −tn}2 + const
(5.165)
which corresponds to a regularized sum-of-squares error function. Assuming for
the moment that α and β are ﬁxed, we can ﬁnd a maximum of the posterior, which
we denote wMAP, by standard nonlinear optimization algorithms such as conjugate
gradients, using error backpropagation to evaluate the required derivatives.
Having found a mode wMAP, we can then build a local Gaussian approximation
by evaluating the matrix of second derivatives of the negative log posterior distribu-
tion. From (5.165), this is given by
A = −∇∇ln p(w|D, α, β) = αI + βH
(5.166)
where H is the Hessian matrix comprising the second derivatives of the sum-of-
squares error function with respect to the components of w. Algorithms for comput-
ing and approximating the Hessian were discussed in Section 5.4. The corresponding
Gaussian approximation to the posterior is then given from (4.134) by
q(w|D) = N(w|wMAP, A−1).
(5.167)
Similarly, the predictive distribution is obtained by marginalizing with respect
to this posterior distribution
p(t|x, D) =

p(t|x, w)q(w|D) dw.
(5.168)
However, even with the Gaussian approximation to the posterior, this integration is
still analytically intractable due to the nonlinearity of the network function y(x, w)
as a function of w. To make progress, we now assume that the posterior distribution
has small variance compared with the characteristic scales of w over which y(x, w)
is varying. This allows us to make a Taylor series expansion of the network function
around wMAP and retain only the linear terms
y(x, w) ≃y(x, wMAP) + gT(w −wMAP)
(5.169)
where we have deﬁned
g = ∇wy(x, w)|w=wMAP .
(5.170)
With this approximation, we now have a linear-Gaussian model with a Gaussian
distribution for p(w) and a Gaussian for p(t|w) whose mean is a linear function of
w of the form
p(t|x, w, β) ≃N 
t|y(x, wMAP) + gT(w −wMAP), β−1
.
(5.171)
We can therefore make use of the general result (2.115) for the marginal p(t) to give
Exercise 5.38
p(t|x, D, α, β) = N

t|y(x, wMAP), σ2(x)

(5.172)


---
**Page 297**
280
5. NEURAL NETWORKS
where the input-dependent variance is given by
σ2(x) = β−1 + gTA−1g.
(5.173)
We see that the predictive distribution p(t|x, D) is a Gaussian whose mean is given
by the network function y(x, wMAP) with the parameter set to their MAP value. The
variance has two terms, the ﬁrst of which arises from the intrinsic noise on the target
variable, whereas the second is an x-dependent term that expresses the uncertainty
in the interpolant due to the uncertainty in the model parameters w. This should
be compared with the corresponding predictive distribution for the linear regression
model, given by (3.58) and (3.59).
5.7.2
Hyperparameter optimization
So far, we have assumed that the hyperparameters α and β are ﬁxed and known.
We can make use of the evidence framework, discussed in Section 3.5, together with
the Gaussian approximation to the posterior obtained using the Laplace approxima-
tion, to obtain a practical procedure for choosing the values of such hyperparameters.
The marginal likelihood, or evidence, for the hyperparameters is obtained by
integrating over the network weights
p(D|α, β) =

p(D|w, β)p(w|α) dw.
(5.174)
This is easily evaluated by making use of the Laplace approximation result (4.135).
Exercise 5.39
Taking logarithms then gives
ln p(D|α, β) ≃−E(wMAP) −1
2 ln |A| + W
2 ln α + N
2 ln β −N
2 ln(2π) (5.175)
where W is the total number of parameters in w, and the regularized error function
is deﬁned by
E(wMAP) = β
2
N

n=1
{y(xn, wMAP) −tn}2 + α
2 wT
MAPwMAP.
(5.176)
We see that this takes the same form as the corresponding result (3.86) for the linear
regression model.
In the evidence framework, we make point estimates for α and β by maximizing
ln p(D|α, β). Consider ﬁrst the maximization with respect to α, which can be done
by analogy with the linear regression case discussed in Section 3.5.2. We ﬁrst deﬁne
the eigenvalue equation
βHui = λiui
(5.177)
where H is the Hessian matrix comprising the second derivatives of the sum-of-
squares error function, evaluated at w = wMAP. By analogy with (3.92), we obtain
α =
γ
wT
MAPwMAP
(5.178)


---
**Page 298**
5.7. Bayesian Neural Networks
281
where γ represents the effective number of parameters and is deﬁned by
Section 3.5.3
γ =
W

i=1
λi
α + λi
.
(5.179)
Note that this result was exact for the linear regression case. For the nonlinear neural
network, however, it ignores the fact that changes in α will cause changes in the
Hessian H, which in turn will change the eigenvalues. We have therefore implicitly
ignored terms involving the derivatives of λi with respect to α.
Similarly, from (3.95) we see that maximizing the evidence with respect to β
gives the re-estimation formula
1
β =
1
N −γ
N

n=1
{y(xn, wMAP) −tn}2.
(5.180)
As with the linear model, we need to alternate between re-estimation of the hyper-
parameters α and β and updating of the posterior distribution. The situation with
a neural network model is more complex, however, due to the multimodality of the
posterior distribution. As a consequence, the solution for wMAP found by maximiz-
ing the log posterior will depend on the initialization of w. Solutions that differ only
as a consequence of the interchange and sign reversal symmetries in the hidden units
Section 5.1.1
are identical so far as predictions are concerned, and it is irrelevant which of the
equivalent solutions is found. However, there may be inequivalent solutions as well,
and these will generally yield different values for the optimized hyperparameters.
In order to compare different models, for example neural networks having differ-
ent numbers of hidden units, we need to evaluate the model evidence p(D). This can
be approximated by taking (5.175) and substituting the values of α and β obtained
from the iterative optimization of these hyperparameters. A more careful evaluation
is obtained by marginalizing over α and β, again by making a Gaussian approxima-
tion (MacKay, 1992c; Bishop, 1995a). In either case, it is necessary to evaluate the
determinant |A| of the Hessian matrix. This can be problematic in practice because
the determinant, unlike the trace, is sensitive to the small eigenvalues that are often
difﬁcult to determine accurately.
The Laplace approximation is based on a local quadratic expansion around a
mode of the posterior distribution over weights. We have seen in Section 5.1.1 that
any given mode in a two-layer network is a member of a set of M!2M equivalent
modes that differ by interchange and sign-change symmetries, where M is the num-
ber of hidden units. When comparing networks having different numbers of hid-
den units, this can be taken into account by multiplying the evidence by a factor of
M!2M.
5.7.3
Bayesian neural networks for classiﬁcation
So far, we have used the Laplace approximation to develop a Bayesian treat-
ment of neural network regression models. We now discuss the modiﬁcations to


---
**Page 299**
282
5. NEURAL NETWORKS
this framework that arise when it is applied to classiﬁcation. Here we shall con-
sider a network having a single logistic sigmoid output corresponding to a two-class
classiﬁcation problem. The extension to networks with multiclass softmax outputs
is straightforward. We shall build extensively on the analogous results for linear
Exercise 5.40
classiﬁcation models discussed in Section 4.5, and so we encourage the reader to
familiarize themselves with that material before studying this section.
The log likelihood function for this model is given by
ln p(D|w) =

n
= 1N {tn ln yn + (1 −tn) ln(1 −yn)}
(5.181)
where tn ∈{0, 1} are the target values, and yn ≡y(xn, w). Note that there is no
hyperparameter β, because the data points are assumed to be correctly labelled. As
before, the prior is taken to be an isotropic Gaussian of the form (5.162).
The ﬁrst stage in applying the Laplace framework to this model is to initialize
the hyperparameter α, and then to determine the parameter vector w by maximizing
the log posterior distribution. This is equivalent to minimizing the regularized error
function
E(w) = −ln p(D|w) + α
2 wTw
(5.182)
and can be achieved using error backpropagation combined with standard optimiza-
tion algorithms, as discussed in Section 5.3.
Having found a solution wMAP for the weight vector, the next step is to eval-
uate the Hessian matrix H comprising the second derivatives of the negative log
likelihood function. This can be done, for instance, using the exact method of Sec-
tion 5.4.5, or using the outer product approximation given by (5.85). The second
derivatives of the negative log posterior can again be written in the form (5.166), and
the Gaussian approximation to the posterior is then given by (5.167).
To optimize the hyperparameter α, we again maximize the marginal likelihood,
which is easily shown to take the form
Exercise 5.41
ln p(D|α) ≃−E(wMAP) −1
2 ln |A| + W
2 ln α + const
(5.183)
where the regularized error function is deﬁned by
E(wMAP) = −
N

n=1
{tn ln yn + (1 −tn) ln(1 −yn)} + α
2 wT
MAPwMAP
(5.184)
in which yn ≡y(xn, wMAP). Maximizing this evidence function with respect to α
again leads to the re-estimation equation given by (5.178).
The use of the evidence procedure to determine α is illustrated in Figure 5.22
for the synthetic two-dimensional data discussed in Appendix A.
Finally, we need the predictive distribution, which is deﬁned by (5.168). Again,
this integration is intractable due to the nonlinearity of the network function. The


---
**Page 300**
5.7. Bayesian Neural Networks
283
Figure 5.22
Illustration of the evidence framework
applied to a synthetic two-class data set.
The green curve shows the optimal de-
cision boundary, the black curve shows
the result of ﬁtting a two-layer network
with 8 hidden units by maximum likeli-
hood, and the red curve shows the re-
sult of including a regularizer in which
α is optimized using the evidence pro-
cedure, starting from the initial value
α = 0. Note that the evidence proce-
dure greatly reduces the over-ﬁtting of
the network.
−2
−1
0
1
2
−2
−1
0
1
2
3
simplest approximation is to assume that the posterior distribution is very narrow
and hence make the approximation
p(t|x, D) ≃p(t|x, wMAP).
(5.185)
We can improve on this, however, by taking account of the variance of the posterior
distribution. In this case, a linear approximation for the network outputs, as was used
in the case of regression, would be inappropriate due to the logistic sigmoid output-
unit activation function that constrains the output to lie in the range (0, 1). Instead,
we make a linear approximation for the output unit activation in the form
a(x, w) ≃aMAP(x) + bT(w −wMAP)
(5.186)
where aMAP(x) = a(x, wMAP), and the vector b ≡∇a(x, wMAP) can be found by
backpropagation.
Because we now have a Gaussian approximation for the posterior distribution
over w, and a model for a that is a linear function of w, we can now appeal to the
results of Section 4.5.2. The distribution of output unit activation values, induced by
the distribution over network weights, is given by
p(a|x, D) =

δ

a −aMAP(x) −bT(x)(w −wMAP)

q(w|D) dw
(5.187)
where q(w|D) is the Gaussian approximation to the posterior distribution given by
(5.167). From Section 4.5.2, we see that this distribution is Gaussian with mean
aMAP ≡a(x, wMAP), and variance
σ2
a(x) = bT(x)A−1b(x).
(5.188)
Finally, to obtain the predictive distribution, we must marginalize over a using
p(t = 1|x, D) =

σ(a)p(a|x, D) da.
(5.189)


---
**Page 301**
284
5. NEURAL NETWORKS
−2
−1
0
1
2
−2
−1
0
1
2
3
−2
−1
0
1
2
−2
−1
0
1
2
3
Figure 5.23
An illustration of the Laplace approximation for a Bayesian neural network having 8 hidden units
with ‘tanh’ activation functions and a single logistic-sigmoid output unit. The weight parameters were found using
scaled conjugate gradients, and the hyperparameter α was optimized using the evidence framework. On the left
is the result of using the simple approximation (5.185) based on a point estimate wMAP of the parameters,
in which the green curve shows the y = 0.5 decision boundary, and the other contours correspond to output
probabilities of y = 0.1, 0.3, 0.7, and 0.9. On the right is the corresponding result obtained using (5.190). Note
that the effect of marginalization is to spread out the contours and to make the predictions less conﬁdent, so
that at each input point x, the posterior probabilities are shifted towards 0.5, while the y = 0.5 contour itself is
unaffected.
The convolution of a Gaussian with a logistic sigmoid is intractable. We therefore
apply the approximation (4.153) to (5.189) giving
p(t = 1|x, D) = σ 
κ(σ2
a)bTwMAP

(5.190)
where κ(·) is deﬁned by (4.154). Recall that both σ2
a and b are functions of x.
Figure 5.23 shows an example of this framework applied to the synthetic classi-
ﬁcation data set described in Appendix A.
Exercises
5.1
(⋆⋆) Consider a two-layer network function of the form (5.7) in which the hidden-
unit nonlinear activation functions g(·) are given by logistic sigmoid functions of the
form
σ(a) = {1 + exp(−a)}−1 .
(5.191)
Show that there exists an equivalent network, which computes exactly the same func-
tion, but with hidden unit activation functions given by tanh(a) where the tanh func-
tion is deﬁned by (5.59). Hint: ﬁrst ﬁnd the relation between σ(a) and tanh(a), and
then show that the parameters of the two networks differ by linear transformations.
5.2
(⋆) www
Show that maximizing the likelihood function under the conditional
distribution (5.16) for a multioutput neural network is equivalent to minimizing the
sum-of-squares error function (5.11).


---
**Page 302**
Exercises
285
5.3
(⋆⋆)
Consider a regression problem involving multiple target variables in which it
is assumed that the distribution of the targets, conditioned on the input vector x, is a
Gaussian of the form
p(t|x, w) = N(t|y(x, w), Σ)
(5.192)
where y(x, w) is the output of a neural network with input vector x and weight
vector w, and Σ is the covariance of the assumed Gaussian noise on the targets.
Given a set of independent observations of x and t, write down the error function
that must be minimized in order to ﬁnd the maximum likelihood solution for w, if
we assume that Σ is ﬁxed and known. Now assume that Σ is also to be determined
from the data, and write down an expression for the maximum likelihood solution
for Σ. Note that the optimizations of w and Σ are now coupled, in contrast to the
case of independent target variables discussed in Section 5.2.
5.4
(⋆⋆)
Consider a binary classiﬁcation problem in which the target values are t ∈
{0, 1}, with a network output y(x, w) that represents p(t = 1|x), and suppose that
there is a probability ϵ that the class label on a training data point has been incorrectly
set. Assuming independent and identically distributed data, write down the error
function corresponding to the negative log likelihood. Verify that the error function
(5.21) is obtained when ϵ = 0. Note that this error function makes the model robust
to incorrectly labelled data, in contrast to the usual error function.
5.5
(⋆) www
Show that maximizing likelihood for a multiclass neural network model
in which the network outputs have the interpretation yk(x, w) = p(tk = 1|x) is
equivalent to the minimization of the cross-entropy error function (5.24).
5.6
(⋆) www
Show the derivative of the error function (5.21) with respect to the
activation ak for an output unit having a logistic sigmoid activation function satisﬁes
(5.18).
5.7
(⋆) Show the derivative of the error function (5.24) with respect to the activation ak
for output units having a softmax activation function satisﬁes (5.18).
5.8
(⋆)
We saw in (4.88) that the derivative of the logistic sigmoid activation function
can be expressed in terms of the function value itself. Derive the corresponding result
for the ‘tanh’ activation function deﬁned by (5.59).
5.9
(⋆) www
The error function (5.21) for binary classiﬁcation problems was de-
rived for a network having a logistic-sigmoid output activation function, so that
0 ⩽y(x, w) ⩽1, and data having target values t ∈{0, 1}. Derive the correspond-
ing error function if we consider a network having an output −1 ⩽y(x, w) ⩽1
and target values t = 1 for class C1 and t = −1 for class C2. What would be the
appropriate choice of output unit activation function?
5.10
(⋆) www
Consider a Hessian matrix H with eigenvector equation (5.33). By
setting the vector v in (5.39) equal to each of the eigenvectors ui in turn, show that
H is positive deﬁnite if, and only if, all of its eigenvalues are positive.


---
**Page 303**
286
5. NEURAL NETWORKS
5.11
(⋆⋆) www
Consider a quadratic error function deﬁned by (5.32), in which the
Hessian matrix H has an eigenvalue equation given by (5.33). Show that the con-
tours of constant error are ellipses whose axes are aligned with the eigenvectors ui,
with lengths that are inversely proportional to the square root of the corresponding
eigenvalues λi.
5.12
(⋆⋆) www
By considering the local Taylor expansion (5.32) of an error function
about a stationary point w⋆, show that the necessary and sufﬁcient condition for the
stationary point to be a local minimum of the error function is that the Hessian matrix
H, deﬁned by (5.30) with w = w⋆, be positive deﬁnite.
5.13
(⋆)
Show that as a consequence of the symmetry of the Hessian matrix H, the
number of independent elements in the quadratic error function (5.28) is given by
W(W + 3)/2.
5.14
(⋆) By making a Taylor expansion, verify that the terms that are O(ϵ) cancel on the
right-hand side of (5.69).
5.15
(⋆⋆) In Section 5.3.4, we derived a procedure for evaluating the Jacobian matrix of a
neural network using a backpropagation procedure. Derive an alternative formalism
for ﬁnding the Jacobian based on forward propagation equations.
5.16
(⋆)
The outer product approximation to the Hessian matrix for a neural network
using a sum-of-squares error function is given by (5.84). Extend this result to the
case of multiple outputs.
5.17
(⋆) Consider a squared loss function of the form
E = 1
2

{y(x, w) −t}2 p(x, t) dx dt
(5.193)
where y(x, w) is a parametric function such as a neural network. The result (1.89)
shows that the function y(x, w) that minimizes this error is given by the conditional
expectation of t given x. Use this result to show that the second derivative of E with
respect to two elements wr and ws of the vector w, is given by
∂2E
∂wr∂ws
=

∂y
∂wr
∂y
∂ws
p(x) dx.
(5.194)
Note that, for a ﬁnite sample from p(x), we obtain (5.84).
5.18
(⋆) Consider a two-layer network of the form shown in Figure 5.1 with the addition
of extra parameters corresponding to skip-layer connections that go directly from
the inputs to the outputs. By extending the discussion of Section 5.3.2, write down
the equations for the derivatives of the error function with respect to these additional
parameters.
5.19
(⋆) www
Derive the expression (5.85) for the outer product approximation to
the Hessian matrix for a network having a single output with a logistic sigmoid
output-unit activation function and a cross-entropy error function, corresponding to
the result (5.84) for the sum-of-squares error function.


---
**Page 304**
Exercises
287
5.20
(⋆) Derive an expression for the outer product approximation to the Hessian matrix
for a network having K outputs with a softmax output-unit activation function and
a cross-entropy error function, corresponding to the result (5.84) for the sum-of-
squares error function.
5.21
(⋆⋆⋆) Extend the expression (5.86) for the outer product approximation of the Hes-
sian matrix to the case of K > 1 output units. Hence, derive a recursive expression
analogous to (5.87) for incrementing the number N of patterns and a similar expres-
sion for incrementing the number K of outputs. Use these results, together with the
identity (5.88), to ﬁnd sequential update expressions analogous to (5.89) for ﬁnding
the inverse of the Hessian by incrementally including both extra patterns and extra
outputs.
5.22
(⋆⋆)
Derive the results (5.93), (5.94), and (5.95) for the elements of the Hessian
matrix of a two-layer feed-forward network by application of the chain rule of cal-
culus.
5.23
(⋆⋆) Extend the results of Section 5.4.5 for the exact Hessian of a two-layer network
to include skip-layer connections that go directly from inputs to outputs.
5.24
(⋆) Verify that the network function deﬁned by (5.113) and (5.114) is invariant un-
der the transformation (5.115) applied to the inputs, provided the weights and biases
are simultaneously transformed using (5.116) and (5.117). Similarly, show that the
network outputs can be transformed according (5.118) by applying the transforma-
tion (5.119) and (5.120) to the second-layer weights and biases.
5.25
(⋆⋆⋆) www
Consider a quadratic error function of the form
E = E0 + 1
2(w −w⋆)TH(w −w⋆)
(5.195)
where w⋆represents the minimum, and the Hessian matrix H is positive deﬁnite and
constant. Suppose the initial weight vector w(0) is chosen to be at the origin and is
updated using simple gradient descent
w(τ) = w(τ−1) −ρ∇E
(5.196)
where τ denotes the step number, and ρ is the learning rate (which is assumed to be
small). Show that, after τ steps, the components of the weight vector parallel to the
eigenvectors of H can be written
w(τ)
j
= {1 −(1 −ρηj)τ} w⋆
j
(5.197)
where wj = wTuj, and uj and ηj are the eigenvectors and eigenvalues, respectively,
of H so that
Huj = ηjuj.
(5.198)
Show that as τ →∞, this gives w(τ) →w⋆as expected, provided |1 −ρηj| < 1.
Now suppose that training is halted after a ﬁnite number τ of steps. Show that the


---
**Page 305**
288
5. NEURAL NETWORKS
components of the weight vector parallel to the eigenvectors of the Hessian satisfy
w(τ)
j
≃w⋆
j
when
ηj ≫(ρτ)−1
(5.199)
|w(τ)
j | ≪|w⋆
j |
when
ηj ≪(ρτ)−1.
(5.200)
Compare this result with the discussion in Section 3.5.3 of regularization with simple
weight decay, and hence show that (ρτ)−1 is analogous to the regularization param-
eter λ. The above results also show that the effective number of parameters in the
network, as deﬁned by (3.91), grows as the training progresses.
5.26
(⋆⋆) Consider a multilayer perceptron with arbitrary feed-forward topology, which
is to be trained by minimizing the tangent propagation error function (5.127) in
which the regularizing function is given by (5.128). Show that the regularization
term Ωcan be written as a sum over patterns of terms of the form
Ωn = 1
2

k
(Gyk)2
(5.201)
where G is a differential operator deﬁned by
G ≡

i
τi
∂
∂xi
.
(5.202)
By acting on the forward propagation equations
zj = h(aj),
aj =

i
wjizi
(5.203)
with the operator G, show that Ωn can be evaluated by forward propagation using
the following equations:
αj = h′(aj)βj,
βj =

i
wjiαi.
(5.204)
where we have deﬁned the new variables
αj ≡Gzj,
βj ≡Gaj.
(5.205)
Now show that the derivatives of Ωn with respect to a weight wrs in the network can
be written in the form
∂Ωn
∂wrs
=

k
αk {φkrzs + δkrαs}
(5.206)
where we have deﬁned
δkr ≡∂yk
∂ar
,
φkr ≡Gδkr.
(5.207)
Write down the backpropagation equations for δkr, and hence derive a set of back-
propagation equations for the evaluation of the φkr.


---
**Page 306**
Exercises
289
5.27
(⋆⋆) www
Consider the framework for training with transformed data in the
special case in which the transformation consists simply of the addition of random
noise x →x + ξ where ξ has a Gaussian distribution with zero mean and unit
covariance. By following an argument analogous to that of Section 5.5.5, show that
the resulting regularizer reduces to the Tikhonov form (5.135).
5.28
(⋆) www
Consider a neural network, such as the convolutional network discussed
in Section 5.5.6, in which multiple weights are constrained to have the same value.
Discuss how the standard backpropagation algorithm must be modiﬁed in order to
ensure that such constraints are satisﬁed when evaluating the derivatives of an error
function with respect to the adjustable parameters in the network.
5.29
(⋆) www
Verify the result (5.141).
5.30
(⋆) Verify the result (5.142).
5.31
(⋆) Verify the result (5.143).
5.32
(⋆⋆) Show that the derivatives of the mixing coefﬁcients {πk}, deﬁned by (5.146),
with respect to the auxiliary parameters {ηj} are given by
∂πk
∂ηj
= δjkπj −πjπk.
(5.208)
Hence, by making use of the constraint 
k πk = 1, derive the result (5.147).
5.33
(⋆)
Write down a pair of equations that express the Cartesian coordinates (x1, x2)
for the robot arm shown in Figure 5.18 in terms of the joint angles θ1 and θ2 and
the lengths L1 and L2 of the links. Assume the origin of the coordinate system is
given by the attachment point of the lower arm. These equations deﬁne the ‘forward
kinematics’ of the robot arm.
5.34
(⋆) www
Derive the result (5.155) for the derivative of the error function with
respect to the network output activations controlling the mixing coefﬁcients in the
mixture density network.
5.35
(⋆)
Derive the result (5.156) for the derivative of the error function with respect
to the network output activations controlling the component means in the mixture
density network.
5.36
(⋆)
Derive the result (5.157) for the derivative of the error function with respect to
the network output activations controlling the component variances in the mixture
density network.
5.37
(⋆)
Verify the results (5.158) and (5.160) for the conditional mean and variance of
the mixture density network model.
5.38
(⋆)
Using the general result (2.115), derive the predictive distribution (5.172) for
the Laplace approximation to the Bayesian neural network model.


---
**Page 307**
290
5. NEURAL NETWORKS
5.39
(⋆) www
Make use of the Laplace approximation result (4.135) to show that the
evidence function for the hyperparameters α and β in the Bayesian neural network
model can be approximated by (5.175).
5.40
(⋆) www
Outline the modiﬁcations needed to the framework for Bayesian neural
networks, discussed in Section 5.7.3, to handle multiclass problems using networks
having softmax output-unit activation functions.
5.41
(⋆⋆)
By following analogous steps to those given in Section 5.7.1 for regression
networks, derive the result (5.183) for the marginal likelihood in the case of a net-
work having a cross-entropy error function and logistic-sigmoid output-unit activa-
tion function.


---
**Page 308**
6
Kernel
Methods
In Chapters 3 and 4, we considered linear parametric models for regression and
classiﬁcation in which the form of the mapping y(x, w) from input x to output y
is governed by a vector w of adaptive parameters. During the learning phase, a
set of training data is used either to obtain a point estimate of the parameter vector
or to determine a posterior distribution over this vector. The training data is then
discarded, and predictions for new inputs are based purely on the learned parameter
vector w. This approach is also used in nonlinear parametric models such as neural
networks.
Chapter 5
However, there is a class of pattern recognition techniques, in which the training
data points, or a subset of them, are kept and used also during the prediction phase.
For instance, the Parzen probability density model comprised a linear combination
Section 2.5.1
of ‘kernel’ functions each one centred on one of the training data points. Similarly,
in Section 2.5.2 we introduced a simple technique for classiﬁcation called nearest
neighbours, which involved assigning to each new test vector the same label as the
291


---
**Page 309**
292
6. KERNEL METHODS
closest example from the training set. These are examples of memory-based methods
that involve storing the entire training set in order to make predictions for future data
points. They typically require a metric to be deﬁned that measures the similarity of
any two vectors in input space, and are generally fast to ‘train’ but slow at making
predictions for test data points.
Many linear parametric models can be re-cast into an equivalent ‘dual represen-
tation’ in which the predictions are also based on linear combinations of a kernel
function evaluated at the training data points. As we shall see, for models which are
based on a ﬁxed nonlinear feature space mapping φ(x), the kernel function is given
by the relation
k(x, x′) = φ(x)Tφ(x′).
(6.1)
From this deﬁnition, we see that the kernel is a symmetric function of its arguments
so that k(x, x′) = k(x′, x). The kernel concept was introduced into the ﬁeld of pat-
tern recognition by Aizerman et al. (1964) in the context of the method of potential
functions, so-called because of an analogy with electrostatics. Although neglected
for many years, it was re-introduced into machine learning in the context of large-
margin classiﬁers by Boser et al. (1992) giving rise to the technique of support
vector machines. Since then, there has been considerable interest in this topic, both
Chapter 7
in terms of theory and applications. One of the most signiﬁcant developments has
been the extension of kernels to handle symbolic objects, thereby greatly expanding
the range of problems that can be addressed.
The simplest example of a kernel function is obtained by considering the identity
mapping for the feature space in (6.1) so that φ(x) = x, in which case k(x, x′) =
xTx′. We shall refer to this as the linear kernel.
The concept of a kernel formulated as an inner product in a feature space allows
us to build interesting extensions of many well-known algorithms by making use of
the kernel trick, also known as kernel substitution. The general idea is that, if we have
an algorithm formulated in such a way that the input vector x enters only in the form
of scalar products, then we can replace that scalar product with some other choice of
kernel. For instance, the technique of kernel substitution can be applied to principal
component analysis in order to develop a nonlinear variant of PCA (Sch¨olkopf et al.,
Section 12.3
1998). Other examples of kernel substitution include nearest-neighbour classiﬁers
and the kernel Fisher discriminant (Mika et al., 1999; Roth and Steinhage, 2000;
Baudat and Anouar, 2000).
There are numerous forms of kernel functions in common use, and we shall en-
counter several examples in this chapter. Many have the property of being a function
only of the difference between the arguments, so that k(x, x′) = k(x −x′), which
are known as stationary kernels because they are invariant to translations in input
space. A further specialization involves homogeneous kernels, also known as ra-
dial basis functions, which depend only on the magnitude of the distance (typically
Section 6.3
Euclidean) between the arguments so that k(x, x′) = k(∥x −x′∥).
For recent textbooks on kernel methods, see Sch¨olkopf and Smola (2002), Her-
brich (2002), and Shawe-Taylor and Cristianini (2004).


---
**Page 310**
6.1. Dual Representations
293
6.1. Dual Representations
Many linear models for regression and classiﬁcation can be reformulated in terms of
a dual representation in which the kernel function arises naturally. This concept will
play an important role when we consider support vector machines in the next chapter.
Here we consider a linear regression model whose parameters are determined by
minimizing a regularized sum-of-squares error function given by
J(w) = 1
2
N

n=1

wTφ(xn) −tn
2 + λ
2 wTw
(6.2)
where λ ⩾0. If we set the gradient of J(w) with respect to w equal to zero, we see
that the solution for w takes the form of a linear combination of the vectors φ(xn),
with coefﬁcients that are functions of w, of the form
w = −1
λ
N

n=1

wTφ(xn) −tn

φ(xn) =
N

n=1
anφ(xn) = ΦTa
(6.3)
where Φ is the design matrix, whose nth row is given by φ(xn)T. Here the vector
a = (a1, . . . , aN)T, and we have deﬁned
an = −1
λ

wTφ(xn) −tn

.
(6.4)
Instead of working with the parameter vector w, we can now reformulate the least-
squares algorithm in terms of the parameter vector a, giving rise to a dual represen-
tation. If we substitute w = ΦTa into J(w), we obtain
J(a) = 1
2aTΦΦTΦΦTa −aTΦΦTt + 1
2tTt + λ
2 aTΦΦTa
(6.5)
where t = (t1, . . . , tN)T. We now deﬁne the Gram matrix K = ΦΦT, which is an
N × N symmetric matrix with elements
Knm = φ(xn)Tφ(xm) = k(xn, xm)
(6.6)
where we have introduced the kernel function k(x, x′) deﬁned by (6.1). In terms of
the Gram matrix, the sum-of-squares error function can be written as
J(a) = 1
2aTKKa −aTKt + 1
2tTt + λ
2 aTKa.
(6.7)
Setting the gradient of J(a) with respect to a to zero, we obtain the following solu-
tion
a = (K + λIN)−1 t.
(6.8)


---
**Page 311**
294
6. KERNEL METHODS
If we substitute this back into the linear regression model, we obtain the following
prediction for a new input x
y(x) = wTφ(x) = aTΦφ(x) = k(x)T (K + λIN)−1 t
(6.9)
where we have deﬁned the vector k(x) with elements kn(x) = k(xn, x). Thus we
see that the dual formulation allows the solution to the least-squares problem to be
expressed entirely in terms of the kernel function k(x, x′). This is known as a dual
formulation because, by noting that the solution for a can be expressed as a linear
combination of the elements of φ(x), we recover the original formulation in terms of
the parameter vector w. Note that the prediction at x is given by a linear combination
Exercise 6.1
of the target values from the training set. In fact, we have already obtained this result,
using a slightly different notation, in Section 3.3.3.
In the dual formulation, we determine the parameter vector a by inverting an
N × N matrix, whereas in the original parameter space formulation we had to invert
an M × M matrix in order to determine w. Because N is typically much larger
than M, the dual formulation does not seem to be particularly useful. However, the
advantage of the dual formulation, as we shall see, is that it is expressed entirely in
terms of the kernel function k(x, x′). We can therefore work directly in terms of
kernels and avoid the explicit introduction of the feature vector φ(x), which allows
us implicitly to use feature spaces of high, even inﬁnite, dimensionality.
The existence of a dual representation based on the Gram matrix is a property of
many linear models, including the perceptron. In Section 6.4, we will develop a dual-
Exercise 6.2
ity between probabilistic linear models for regression and the technique of Gaussian
processes. Duality will also play an important role when we discuss support vector
machines in Chapter 7.
6.2. Constructing Kernels
In order to exploit kernel substitution, we need to be able to construct valid kernel
functions. One approach is to choose a feature space mapping φ(x) and then use
this to ﬁnd the corresponding kernel, as is illustrated in Figure 6.1. Here the kernel
function is deﬁned for a one-dimensional input space by
k(x, x′) = φ(x)Tφ(x′) =
M

i=1
φi(x)φi(x′)
(6.10)
where φi(x) are the basis functions.
An alternative approach is to construct kernel functions directly. In this case,
we must ensure that the function we choose is a valid kernel, in other words that it
corresponds to a scalar product in some (perhaps inﬁnite dimensional) feature space.
As a simple example, consider a kernel function given by
k(x, z) =

xTz
2 .
(6.11)


---
**Page 312**
6.2. Constructing Kernels
295
−1
0
1
−1
−0.5
0
0.5
1
−1
0
1
0
0.25
0.5
0.75
1
−1
0
1
0
0.25
0.5
0.75
1
−1
0
1
0
0.02
0.04
−1
0
1
0
0.02
0.04
−1
0
1
0
0.02
0.04
Figure 6.1
Illustration of the construction of kernel functions starting from a corresponding set of basis func-
tions. In each column the lower plot shows the kernel function k(x, x′) deﬁned by (6.10) plotted as a function of
x for x′ = 0, while the upper plot shows the corresponding basis functions given by polynomials (left column),
‘Gaussians’ (centre column), and logistic sigmoids (right column).
If we take the particular case of a two-dimensional input space x = (x1, x2) we
can expand out the terms and thereby identify the corresponding nonlinear feature
mapping
k(x, z)
=

xTz
2 = (x1z1 + x2z2)2
=
x2
1z2
1 + 2x1z1x2z2 + x2
2z2
2
=
(x2
1,
√
2x1x2, x2
2)(z2
1,
√
2z1z2, z2
2)T
=
φ(x)Tφ(z).
(6.12)
We see that the feature mapping takes the form φ(x) = (x2
1,
√
2x1x2, x2
2)T and
therefore comprises all possible second order terms, with a speciﬁc weighting be-
tween them.
More generally, however, we need a simple way to test whether a function con-
stitutes a valid kernel without having to construct the function φ(x) explicitly. A
necessary and sufﬁcient condition for a function k(x, x′) to be a valid kernel (Shawe-
Taylor and Cristianini, 2004) is that the Gram matrix K, whose elements are given by
k(xn, xm), should be positive semideﬁnite for all possible choices of the set {xn}.
Note that a positive semideﬁnite matrix is not the same thing as a matrix whose
elements are nonnegative.
Appendix C
One powerful technique for constructing new kernels is to build them out of
simpler kernels as building blocks. This can be done using the following properties:


---
**Page 313**
296
6. KERNEL METHODS
Techniques for Constructing New Kernels.
Given valid kernels k1(x, x′) and k2(x, x′), the following new kernels will also
be valid:
k(x, x′)
=
ck1(x, x′)
(6.13)
k(x, x′)
=
f(x)k1(x, x′)f(x′)
(6.14)
k(x, x′)
=
q (k1(x, x′))
(6.15)
k(x, x′)
=
exp (k1(x, x′))
(6.16)
k(x, x′)
=
k1(x, x′) + k2(x, x′)
(6.17)
k(x, x′)
=
k1(x, x′)k2(x, x′)
(6.18)
k(x, x′)
=
k3 (φ(x), φ(x′))
(6.19)
k(x, x′)
=
xTAx′
(6.20)
k(x, x′)
=
ka(xa, x′
a) + kb(xb, x′
b)
(6.21)
k(x, x′)
=
ka(xa, x′
a)kb(xb, x′
b)
(6.22)
where c > 0 is a constant, f(·) is any function, q(·) is a polynomial with nonneg-
ative coefﬁcients, φ(x) is a function from x to RM, k3(·, ·) is a valid kernel in
RM, A is a symmetric positive semideﬁnite matrix, xa and xb are variables (not
necessarily disjoint) with x = (xa, xb), and ka and kb are valid kernel functions
over their respective spaces.
Equipped with these properties, we can now embark on the construction of more
complex kernels appropriate to speciﬁc applications. We require that the kernel
k(x, x′) be symmetric and positive semideﬁnite and that it expresses the appropriate
form of similarity between x and x′ according to the intended application. Here we
consider a few common examples of kernel functions. For a more extensive discus-
sion of ‘kernel engineering’, see Shawe-Taylor and Cristianini (2004).
We saw that the simple polynomial kernel k(x, x′) = 
xTx′2 contains only
terms of degree two.
If we consider the slightly generalized kernel k(x, x′) =

xTx′ + c2 with c > 0, then the corresponding feature mapping φ(x) contains con-
stant and linear terms as well as terms of order two. Similarly, k(x, x′) =

xTx′M
contains all monomials of order M. For instance, if x and x′ are two images, then
the kernel represents a particular weighted sum of all possible products of M pixels
in the ﬁrst image with M pixels in the second image. This can similarly be gener-
alized to include all terms up to degree M by considering k(x, x′) =

xTx′ + c
M
with c > 0. Using the results (6.17) and (6.18) for combining kernels we see that
these will all be valid kernel functions.
Another commonly used kernel takes the form
k(x, x′) = exp

−∥x −x′∥2/2σ2
(6.23)
and is often called a ‘Gaussian’ kernel. Note, however, that in this context it is
not interpreted as a probability density, and hence the normalization coefﬁcient is


---
**Page 314**
6.2. Constructing Kernels
297
omitted. We can see that this is a valid kernel by expanding the square
∥x −x′∥2 = xTx + (x′)Tx′ −2xTx′
(6.24)
to give
k(x, x′) = exp 
−xTx/2σ2
exp 
xTx′/σ2
exp 
−(x′)Tx′/2σ2
(6.25)
and then making use of (6.14) and (6.16), together with the validity of the linear
kernel k(x, x′) = xTx′. Note that the feature vector that corresponds to the Gaussian
kernel has inﬁnite dimensionality.
Exercise 6.11
The Gaussian kernel is not restricted to the use of Euclidean distance. If we use
kernel substitution in (6.24) to replace xTx′ with a nonlinear kernel κ(x, x′), we
obtain
k(x, x′) = exp

−1
2σ2 (κ(x, x) + κ(x′, x′) −2κ(x, x′))

.
(6.26)
An important contribution to arise from the kernel viewpoint has been the exten-
sion to inputs that are symbolic, rather than simply vectors of real numbers. Kernel
functions can be deﬁned over objects as diverse as graphs, sets, strings, and text doc-
uments. Consider, for instance, a ﬁxed set and deﬁne a nonvectorial space consisting
of all possible subsets of this set. If A1 and A2 are two such subsets then one simple
choice of kernel would be
k(A1, A2) = 2|A1∩A2|
(6.27)
where A1 ∩A2 denotes the intersection of sets A1 and A2, and |A| denotes the
number of subsets in A. This is a valid kernel function because it can be shown to
correspond to an inner product in a feature space.
Exercise 6.12
One powerful approach to the construction of kernels starts from a probabilistic
generative model (Haussler, 1999), which allows us to apply generative models in a
discriminative setting. Generative models can deal naturally with missing data and
in the case of hidden Markov models can handle sequences of varying length. By
contrast, discriminative models generally give better performance on discriminative
tasks than generative models. It is therefore of some interest to combine these two
approaches (Lasserre et al., 2006). One way to combine them is to use a generative
model to deﬁne a kernel, and then use this kernel in a discriminative approach.
Given a generative model p(x) we can deﬁne a kernel by
k(x, x′) = p(x)p(x′).
(6.28)
This is clearly a valid kernel function because we can interpret it as an inner product
in the one-dimensional feature space deﬁned by the mapping p(x). It says that two
inputs x and x′ are similar if they both have high probabilities. We can use (6.13) and
(6.17) to extend this class of kernels by considering sums over products of different
probability distributions, with positive weighting coefﬁcients p(i), of the form
k(x, x′) =

i
p(x|i)p(x′|i)p(i).
(6.29)


---
**Page 315**
298
6. KERNEL METHODS
This is equivalent, up to an overall multiplicative constant, to a mixture distribution
in which the components factorize, with the index i playing the role of a ‘latent’
variable. Two inputs x and x′ will give a large value for the kernel function, and
Section 9.2
hence appear similar, if they have signiﬁcant probability under a range of different
components. Taking the limit of an inﬁnite sum, we can also consider kernels of the
form
k(x, x′) =

p(x|z)p(x′|z)p(z) dz
(6.30)
where z is a continuous latent variable.
Now suppose that our data consists of ordered sequences of length L so that
an observation is given by X = {x1, . . . , xL}. A popular generative model for
sequences is the hidden Markov model, which expresses the distribution p(X) as a
Section 13.2
marginalization over a corresponding sequence of hidden states Z = {z1, . . . , zL}.
We can use this approach to deﬁne a kernel function measuring the similarity of two
sequences X and X′ by extending the mixture representation (6.29) to give
k(X, X′) =

Z
p(X|Z)p(X′|Z)p(Z)
(6.31)
so that both observed sequences are generated by the same hidden sequence Z. This
model can easily be extended to allow sequences of differing length to be compared.
An alternative technique for using generative models to deﬁne kernel functions
is known as the Fisher kernel (Jaakkola and Haussler, 1999). Consider a parametric
generative model p(x|θ) where θ denotes the vector of parameters. The goal is to
ﬁnd a kernel that measures the similarity of two input vectors x and x′ induced by the
generative model. Jaakkola and Haussler (1999) consider the gradient with respect
to θ, which deﬁnes a vector in a ‘feature’ space having the same dimensionality as
θ. In particular, they consider the Fisher score
g(θ, x) = ∇θ ln p(x|θ)
(6.32)
from which the Fisher kernel is deﬁned by
k(x, x′) = g(θ, x)TF−1g(θ, x′).
(6.33)
Here F is the Fisher information matrix, given by
F = Ex

g(θ, x)g(θ, x)T	
(6.34)
where the expectation is with respect to x under the distribution p(x|θ). This can
be motivated from the perspective of information geometry (Amari, 1998), which
considers the differential geometry of the space of model parameters. Here we sim-
ply note that the presence of the Fisher information matrix causes this kernel to be
invariant under a nonlinear re-parameterization of the density model θ →ψ(θ).
Exercise 6.13
In practice, it is often infeasible to evaluate the Fisher information matrix. One
approach is simply to replace the expectation in the deﬁnition of the Fisher informa-
tion with the sample average, giving
F ≃1
N
N

n=1
g(θ, xn)g(θ, xn)T.
(6.35)


---
**Page 316**
6.3. Radial Basis Function Networks
299
This is the covariance matrix of the Fisher scores, and so the Fisher kernel corre-
sponds to a whitening of these scores. More simply, we can just omit the Fisher
Section 12.1.3
information matrix altogether and use the noninvariant kernel
k(x, x′) = g(θ, x)Tg(θ, x′).
(6.36)
An application of Fisher kernels to document retrieval is given by Hofmann (2000).
A ﬁnal example of a kernel function is the sigmoidal kernel given by
k(x, x′) = tanh

axTx′ + b

(6.37)
whose Gram matrix in general is not positive semideﬁnite. This form of kernel
has, however, been used in practice (Vapnik, 1995), possibly because it gives kernel
expansions such as the support vector machine a superﬁcial resemblance to neural
network models. As we shall see, in the limit of an inﬁnite number of basis functions,
a Bayesian neural network with an appropriate prior reduces to a Gaussian process,
thereby providing a deeper link between neural networks and kernel methods.
Section 6.4.7
6.3. Radial Basis Function Networks
In Chapter 3, we discussed regression models based on linear combinations of ﬁxed
basis functions, although we did not discuss in detail what form those basis functions
might take. One choice that has been widely used is that of radial basis functions,
which have the property that each basis function depends only on the radial distance
(typically Euclidean) from a centre µj, so that φj(x) = h(∥x −µj∥).
Historically, radial basis functions were introduced for the purpose of exact func-
tion interpolation (Powell, 1987). Given a set of input vectors {x1, . . . , xN} along
with corresponding target values {t1, . . . , tN}, the goal is to ﬁnd a smooth function
f(x) that ﬁts every target value exactly, so that f(xn) = tn for n = 1, . . . , N. This
is achieved by expressing f(x) as a linear combination of radial basis functions, one
centred on every data point
f(x) =
N

n=1
wnh(∥x −xn∥).
(6.38)
The values of the coefﬁcients {wn} are found by least squares, and because there
are the same number of coefﬁcients as there are constraints, the result is a function
that ﬁts every target value exactly. In pattern recognition applications, however, the
target values are generally noisy, and exact interpolation is undesirable because this
corresponds to an over-ﬁtted solution.
Expansions in radial basis functions also arise from regularization theory (Pog-
gio and Girosi, 1990; Bishop, 1995a). For a sum-of-squares error function with a
regularizer deﬁned in terms of a differential operator, the optimal solution is given
by an expansion in the Green’s functions of the operator (which are analogous to the
eigenvectors of a discrete matrix), again with one basis function centred on each data


---
**Page 317**
300
6. KERNEL METHODS
point. If the differential operator is isotropic then the Green’s functions depend only
on the radial distance from the corresponding data point. Due to the presence of the
regularizer, the solution no longer interpolates the training data exactly.
Another motivation for radial basis functions comes from a consideration of
the interpolation problem when the input (rather than the target) variables are noisy
(Webb, 1994; Bishop, 1995a). If the noise on the input variable x is described
by a variable ξ having a distribution ν(ξ), then the sum-of-squares error function
becomes
E = 1
2
N

n=1

{y(xn + ξ) −tn}2 ν(ξ) dξ.
(6.39)
Using the calculus of variations, we can optimize with respect to the function f(x)
Appendix D
to give
Exercise 6.17
y(xn) =
N

n=1
tnh(x −xn)
(6.40)
where the basis functions are given by
h(x −xn) =
ν(x −xn)
N

n=1
ν(x −xn)
.
(6.41)
We see that there is one basis function centred on every data point. This is known as
the Nadaraya-Watson model and will be derived again from a different perspective
in Section 6.3.1. If the noise distribution ν(ξ) is isotropic, so that it is a function
only of ∥ξ∥, then the basis functions will be radial.
Note that the basis functions (6.41) are normalized, so that 
n h(x −xn) = 1
for any value of x. The effect of such normalization is shown in Figure 6.2. Normal-
ization is sometimes used in practice as it avoids having regions of input space where
all of the basis functions take small values, which would necessarily lead to predic-
tions in such regions that are either small or controlled purely by the bias parameter.
Another situation in which expansions in normalized radial basis functions arise
is in the application of kernel density estimation to the problem of regression, as we
shall discuss in Section 6.3.1.
Because there is one basis function associated with every data point, the corre-
sponding model can be computationally costly to evaluate when making predictions
for new data points. Models have therefore been proposed (Broomhead and Lowe,
1988; Moody and Darken, 1989; Poggio and Girosi, 1990), which retain the expan-
sion in radial basis functions but where the number M of basis functions is smaller
than the number N of data points. Typically, the number of basis functions, and the
locations µi of their centres, are determined based on the input data {xn} alone. The
basis functions are then kept ﬁxed and the coefﬁcients {wi} are determined by least
squares by solving the usual set of linear equations, as discussed in Section 3.1.1.


---
**Page 318**
6.3. Radial Basis Function Networks
301
−1
−0.5
0
0.5
1
0
0.2
0.4
0.6
0.8
1
−1
−0.5
0
0.5
1
0
0.2
0.4
0.6
0.8
1
Figure 6.2
Plot of a set of Gaussian basis functions on the left, together with the corresponding normalized
basis functions on the right.
One of the simplest ways of choosing basis function centres is to use a randomly
chosen subset of the data points. A more systematic approach is called orthogonal
least squares (Chen et al., 1991). This is a sequential selection process in which at
each step the next data point to be chosen as a basis function centre corresponds to
the one that gives the greatest reduction in the sum-of-squares error. Values for the
expansion coefﬁcients are determined as part of the algorithm. Clustering algorithms
such as K-means have also been used, which give a set of basis function centres that
Section 9.1
no longer coincide with training data points.
6.3.1
Nadaraya-Watson model
In Section 3.3.3, we saw that the prediction of a linear regression model for a
new input x takes the form of a linear combination of the training set target values
with coefﬁcients given by the ‘equivalent kernel’ (3.62) where the equivalent kernel
satisﬁes the summation constraint (3.64).
We can motivate the kernel regression model (3.61) from a different perspective,
starting with kernel density estimation. Suppose we have a training set {xn, tn} and
we use a Parzen density estimator to model the joint distribution p(x, t), so that
Section 2.5.1
p(x, t) = 1
N
N

n=1
f(x −xn, t −tn)
(6.42)
where f(x, t) is the component density function, and there is one such component
centred on each data point. We now ﬁnd an expression for the regression function
y(x), corresponding to the conditional average of the target variable conditioned on


---
**Page 319**
302
6. KERNEL METHODS
the input variable, which is given by
y(x)
=
E[t|x] =
 ∞
−∞
tp(t|x) dt
=

tp(x, t) dt

p(x, t) dt
=

n

tf(x −xn, t −tn) dt

m

f(x −xm, t −tm) dt
.
(6.43)
We now assume for simplicity that the component density functions have zero mean
so that
 ∞
−∞
f(x, t)t dt = 0
(6.44)
for all values of x. Using a simple change of variable, we then obtain
y(x)
=

n
g(x −xn)tn

m
g(x −xm)
=

n
k(x, xn)tn
(6.45)
where n, m = 1, . . . , N and the kernel function k(x, xn) is given by
k(x, xn) =
g(x −xn)

m
g(x −xm)
(6.46)
and we have deﬁned
g(x) =
 ∞
−∞
f(x, t) dt.
(6.47)
The result (6.45) is known as the Nadaraya-Watson model, or kernel regression
(Nadaraya, 1964; Watson, 1964). For a localized kernel function, it has the prop-
erty of giving more weight to the data points xn that are close to x. Note that the
kernel (6.46) satisﬁes the summation constraint
N

n=1
k(x, xn) = 1.


---
**Page 320**
6.4. Gaussian Processes
303
Figure 6.3
Illustration of the Nadaraya-Watson kernel
regression model using isotropic Gaussian kernels, for the
sinusoidal data set. The original sine function is shown
by the green curve, the data points are shown in blue,
and each is the centre of an isotropic Gaussian kernel.
The resulting regression function, given by the condi-
tional mean, is shown by the red line, along with the two-
standard-deviation region for the conditional distribution
p(t|x) shown by the red shading. The blue ellipse around
each data point shows one standard deviation contour for
the corresponding kernel. These appear noncircular due
to the different scales on the horizontal and vertical axes.
0
0.2
0.4
0.6
0.8
1
−1.5
−1
−0.5
0
0.5
1
1.5
In fact, this model deﬁnes not only a conditional expectation but also a full
conditional distribution given by
p(t|x) =
p(t, x)

p(t, x) dt
=

n
f(x −xn, t −tn)

m

f(x −xm, t −tm) dt
(6.48)
from which other expectations can be evaluated.
As an illustration we consider the case of a single input variable x in which
f(x, t) is given by a zero-mean isotropic Gaussian over the variable z = (x, t) with
variance σ2. The corresponding conditional distribution (6.48) is given by a Gaus-
sian mixture, and is shown, together with the conditional mean, for the sinusoidal
Exercise 6.18
synthetic data set in Figure 6.3.
An obvious extension of this model is to allow for more ﬂexible forms of Gaus-
sian components, for instance having different variance parameters for the input and
target variables. More generally, we could model the joint distribution p(t, x) using
a Gaussian mixture model, trained using techniques discussed in Chapter 9 (Ghahra-
mani and Jordan, 1994), and then ﬁnd the corresponding conditional distribution
p(t|x). In this latter case we no longer have a representation in terms of kernel func-
tions evaluated at the training set data points. However, the number of components
in the mixture model can be smaller than the number of training set points, resulting
in a model that is faster to evaluate for test data points. We have thereby accepted an
increased computational cost during the training phase in order to have a model that
is faster at making predictions.
6.4. Gaussian Processes
In Section 6.1, we introduced kernels by applying the concept of duality to a non-
probabilistic model for regression. Here we extend the role of kernels to probabilis-


---
**Page 321**
304
6. KERNEL METHODS
tic discriminative models, leading to the framework of Gaussian processes. We shall
thereby see how kernels arise naturally in a Bayesian setting.
In Chapter 3, we considered linear regression models of the form y(x, w) =
wTφ(x) in which w is a vector of parameters and φ(x) is a vector of ﬁxed nonlinear
basis functions that depend on the input vector x. We showed that a prior distribution
over w induced a corresponding prior distribution over functions y(x, w). Given a
training data set, we then evaluated the posterior distribution over w and thereby
obtained the corresponding posterior distribution over regression functions, which
in turn (with the addition of noise) implies a predictive distribution p(t|x) for new
input vectors x.
In the Gaussian process viewpoint, we dispense with the parametric model and
instead deﬁne a prior probability distribution over functions directly. At ﬁrst sight, it
might seem difﬁcult to work with a distribution over the uncountably inﬁnite space of
functions. However, as we shall see, for a ﬁnite training set we only need to consider
the values of the function at the discrete set of input values xn corresponding to the
training set and test set data points, and so in practice we can work in a ﬁnite space.
Models equivalent to Gaussian processes have been widely studied in many dif-
ferent ﬁelds. For instance, in the geostatistics literature Gaussian process regression
is known as kriging (Cressie, 1993). Similarly, ARMA (autoregressive moving aver-
age) models, Kalman ﬁlters, and radial basis function networks can all be viewed as
forms of Gaussian process models. Reviews of Gaussian processes from a machine
learning perspective can be found in MacKay (1998), Williams (1999), and MacKay
(2003), and a comparison of Gaussian process models with alternative approaches is
given in Rasmussen (1996). See also Rasmussen and Williams (2006) for a recent
textbook on Gaussian processes.
6.4.1
Linear regression revisited
In order to motivate the Gaussian process viewpoint, let us return to the linear
regression example and re-derive the predictive distribution by working in terms
of distributions over functions y(x, w). This will provide a speciﬁc example of a
Gaussian process.
Consider a model deﬁned in terms of a linear combination of M ﬁxed basis
functions given by the elements of the vector φ(x) so that
y(x) = wTφ(x)
(6.49)
where x is the input vector and w is the M-dimensional weight vector. Now consider
a prior distribution over w given by an isotropic Gaussian of the form
p(w) = N(w|0, α−1I)
(6.50)
governed by the hyperparameter α, which represents the precision (inverse variance)
of the distribution. For any given value of w, the deﬁnition (6.49) deﬁnes a partic-
ular function of x. The probability distribution over w deﬁned by (6.50) therefore
induces a probability distribution over functions y(x). In practice, we wish to eval-
uate this function at speciﬁc values of x, for example at the training data points


---
**Page 322**
6.4. Gaussian Processes
305
x1, . . . , xN. We are therefore interested in the joint distribution of the function val-
ues y(x1), . . . , y(xN), which we denote by the vector y with elements yn = y(xn)
for n = 1, . . . , N. From (6.49), this vector is given by
y = Φw
(6.51)
where Φ is the design matrix with elements Φnk = φk(xn). We can ﬁnd the proba-
bility distribution of y as follows. First of all we note that y is a linear combination of
Gaussian distributed variables given by the elements of w and hence is itself Gaus-
sian. We therefore need only to ﬁnd its mean and covariance, which are given from
Exercise 2.31
(6.50) by
E[y]
=
ΦE[w] = 0
(6.52)
cov[y]
=
E 
yyT	
= ΦE 
wwT	
ΦT = 1
αΦΦT = K
(6.53)
where K is the Gram matrix with elements
Knm = k(xn, xm) = 1
αφ(xn)Tφ(xm)
(6.54)
and k(x, x′) is the kernel function.
This model provides us with a particular example of a Gaussian process. In gen-
eral, a Gaussian process is deﬁned as a probability distribution over functions y(x)
such that the set of values of y(x) evaluated at an arbitrary set of points x1, . . . , xN
jointly have a Gaussian distribution. In cases where the input vector x is two di-
mensional, this may also be known as a Gaussian random ﬁeld. More generally, a
stochastic process y(x) is speciﬁed by giving the joint probability distribution for
any ﬁnite set of values y(x1), . . . , y(xN) in a consistent manner.
A key point about Gaussian stochastic processes is that the joint distribution
over N variables y1, . . . , yN is speciﬁed completely by the second-order statistics,
namely the mean and the covariance. In most applications, we will not have any
prior knowledge about the mean of y(x) and so by symmetry we take it to be zero.
This is equivalent to choosing the mean of the prior over weight values p(w|α) to
be zero in the basis function viewpoint. The speciﬁcation of the Gaussian process is
then completed by giving the covariance of y(x) evaluated at any two values of x,
which is given by the kernel function
E [y(xn)y(xm)] = k(xn, xm).
(6.55)
For the speciﬁc case of a Gaussian process deﬁned by the linear regression model
(6.49) with a weight prior (6.50), the kernel function is given by (6.54).
We can also deﬁne the kernel function directly, rather than indirectly through a
choice of basis function. Figure 6.4 shows samples of functions drawn from Gaus-
sian processes for two different choices of kernel function. The ﬁrst of these is a
‘Gaussian’ kernel of the form (6.23), and the second is the exponential kernel given
by
k(x, x′) = exp (−θ |x −x′|)
(6.56)
which corresponds to the Ornstein-Uhlenbeck process originally introduced by Uh-
lenbeck and Ornstein (1930) to describe Brownian motion.


---
**Page 323**
306
6. KERNEL METHODS
Figure 6.4
Samples
from
Gaus-
sian processes for a ‘Gaussian’ ker-
nel (left) and an exponential kernel
(right).
−1
−0.5
0
0.5
1
−3
−1.5
0
1.5
3
−1
−0.5
0
0.5
1
−3
−1.5
0
1.5
3
6.4.2
Gaussian processes for regression
In order to apply Gaussian process models to the problem of regression, we need
to take account of the noise on the observed target values, which are given by
tn = yn + ϵn
(6.57)
where yn = y(xn), and ϵn is a random noise variable whose value is chosen inde-
pendently for each observation n. Here we shall consider noise processes that have
a Gaussian distribution, so that
p(tn|yn) = N(tn|yn, β−1)
(6.58)
where β is a hyperparameter representing the precision of the noise. Because the
noise is independent for each data point, the joint distribution of the target values
t = (t1, . . . , tN)T conditioned on the values of y = (y1, . . . , yN)T is given by an
isotropic Gaussian of the form
p(t|y) = N(t|y, β−1IN)
(6.59)
where IN denotes the N ×N unit matrix. From the deﬁnition of a Gaussian process,
the marginal distribution p(y) is given by a Gaussian whose mean is zero and whose
covariance is deﬁned by a Gram matrix K so that
p(y) = N(y|0, K).
(6.60)
The kernel function that determines K is typically chosen to express the property
that, for points xn and xm that are similar, the corresponding values y(xn) and
y(xm) will be more strongly correlated than for dissimilar points. Here the notion
of similarity will depend on the application.
In order to ﬁnd the marginal distribution p(t), conditioned on the input values
x1, . . . , xN, we need to integrate over y. This can be done by making use of the
results from Section 2.3.3 for the linear-Gaussian model. Using (2.115), we see that
the marginal distribution of t is given by
p(t) =

p(t|y)p(y) dy = N(t|0, C)
(6.61)


---
**Page 324**
6.4. Gaussian Processes
307
where the covariance matrix C has elements
C(xn, xm) = k(xn, xm) + β−1δnm.
(6.62)
This result reﬂects the fact that the two Gaussian sources of randomness, namely
that associated with y(x) and that associated with ϵ, are independent and so their
covariances simply add.
One widely used kernel function for Gaussian process regression is given by the
exponential of a quadratic form, with the addition of constant and linear terms to
give
k(xn, xm) = θ0 exp

−θ1
2 ∥xn −xm∥2

+ θ2 + θ3xT
nxm.
(6.63)
Note that the term involving θ3 corresponds to a parametric model that is a linear
function of the input variables. Samples from this prior are plotted for various values
of the parameters θ0, . . . , θ3 in Figure 6.5, and Figure 6.6 shows a set of points sam-
pled from the joint distribution (6.60) along with the corresponding values deﬁned
by (6.61).
So far, we have used the Gaussian process viewpoint to build a model of the
joint distribution over sets of data points. Our goal in regression, however, is to
make predictions of the target variables for new inputs, given a set of training data.
Let us suppose that tN = (t1, . . . , tN)T, corresponding to input values x1, . . . , xN,
comprise the observed training set, and our goal is to predict the target variable tN+1
for a new input vector xN+1. This requires that we evaluate the predictive distri-
bution p(tN+1|tN). Note that this distribution is conditioned also on the variables
x1, . . . , xN and xN+1. However, to keep the notation simple we will not show these
conditioning variables explicitly.
To ﬁnd the conditional distribution p(tN+1|t), we begin by writing down the
joint distribution p(tN+1), where tN+1 denotes the vector (t1, . . . , tN, tN+1)T. We
then apply the results from Section 2.3.1 to obtain the required conditional distribu-
tion, as illustrated in Figure 6.7.
From (6.61), the joint distribution over t1, . . . , tN+1 will be given by
p(tN+1) = N(tN+1|0, CN+1)
(6.64)
where CN+1 is an (N + 1) × (N + 1) covariance matrix with elements given by
(6.62). Because this joint distribution is Gaussian, we can apply the results from
Section 2.3.1 to ﬁnd the conditional Gaussian distribution. To do this, we partition
the covariance matrix as follows
CN+1 =

CN
k
kT
c

(6.65)
where CN is the N ×N covariance matrix with elements given by (6.62) for n, m =
1, . . . , N, the vector k has elements k(xn, xN+1) for n = 1, . . . , N, and the scalar

