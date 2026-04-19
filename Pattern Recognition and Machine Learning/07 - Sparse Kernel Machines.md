# 07 - Sparse Kernel Machines
*Pages 325-358 from Pattern Recognition and Machine Learning*

---
**Page 325**
308
6. KERNEL METHODS
(1.00, 4.00, 0.00, 0.00)
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
(9.00, 4.00, 0.00, 0.00)
−1
−0.5
0
0.5
1
−9
−4.5
0
4.5
9
(1.00, 64.00, 0.00, 0.00)
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
(1.00, 0.25, 0.00, 0.00)
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
(1.00, 4.00, 10.00, 0.00)
−1
−0.5
0
0.5
1
−9
−4.5
0
4.5
9
(1.00, 4.00, 0.00, 5.00)
−1
−0.5
0
0.5
1
−4
−2
0
2
4
Figure 6.5
Samples from a Gaussian process prior deﬁned by the covariance function (6.63). The title above
each plot denotes (θ0, θ1, θ2, θ3).
c = k(xN+1, xN+1) + β−1. Using the results (2.81) and (2.82), we see that the con-
ditional distribution p(tN+1|t) is a Gaussian distribution with mean and covariance
given by
m(xN+1)
=
kTC−1
N t
(6.66)
σ2(xN+1)
=
c −kTC−1
N k.
(6.67)
These are the key results that deﬁne Gaussian process regression. Because the vector
k is a function of the test point input value xN+1, we see that the predictive distribu-
tion is a Gaussian whose mean and variance both depend on xN+1. An example of
Gaussian process regression is shown in Figure 6.8.
The only restriction on the kernel function is that the covariance matrix given by
(6.62) must be positive deﬁnite. If λi is an eigenvalue of K, then the corresponding
eigenvalue of C will be λi + β−1. It is therefore sufﬁcient that the kernel matrix
k(xn, xm) be positive semideﬁnite for any pair of points xn and xm, so that λi ⩾0,
because any eigenvalue λi that is zero will still give rise to a positive eigenvalue
for C because β > 0. This is the same restriction on the kernel function discussed
earlier, and so we can again exploit all of the techniques in Section 6.2 to construct


---
**Page 326**
6.4. Gaussian Processes
309
Figure 6.6
Illustration of the sampling of data
points {tn} from a Gaussian process.
The blue curve shows a sample func-
tion from the Gaussian process prior
over functions, and the red points
show the values of yn obtained by
evaluating the function at a set of in-
put values {xn}.
The correspond-
ing values of {tn}, shown in green,
are obtained by adding independent
Gaussian noise to each of the {yn}.
x
t
−1
0
1
−3
0
3
suitable kernels.
Note that the mean (6.66) of the predictive distribution can be written, as a func-
tion of xN+1, in the form
m(xN+1) =
N

n=1
ank(xn, xN+1)
(6.68)
where an is the nth component of C−1
N t. Thus, if the kernel function k(xn, xm)
depends only on the distance ∥xn −xm∥, then we obtain an expansion in radial
basis functions.
The results (6.66) and (6.67) deﬁne the predictive distribution for Gaussian pro-
cess regression with an arbitrary kernel function k(xn, xm). In the particular case in
which the kernel function k(x, x′) is deﬁned in terms of a ﬁnite set of basis functions,
we can derive the results obtained previously in Section 3.3.2 for linear regression
starting from the Gaussian process viewpoint.
Exercise 6.21
For such models, we can therefore obtain the predictive distribution either by
taking a parameter space viewpoint and using the linear regression result or by taking
a function space viewpoint and using the Gaussian process result.
The central computational operation in using Gaussian processes will involve
the inversion of a matrix of size N × N, for which standard methods require O(N 3)
computations. By contrast, in the basis function model we have to invert a matrix
SN of size M × M, which has O(M 3) computational complexity. Note that for
both viewpoints, the matrix inversion must be performed once for the given training
set. For each new test point, both methods require a vector-matrix multiply, which
has cost O(N 2) in the Gaussian process case and O(M 2) for the linear basis func-
tion model. If the number M of basis functions is smaller than the number N of
data points, it will be computationally more efﬁcient to work in the basis function


---
**Page 327**
310
6. KERNEL METHODS
Figure 6.7
Illustration of the mechanism of
Gaussian process regression for
the case of one training point and
one test point, in which the red el-
lipses show contours of the joint dis-
tribution p(t1, t2).
Here t1 is the
training data point, and condition-
ing on the value of t1, correspond-
ing to the vertical blue line, we ob-
tain p(t2|t1) shown as a function of
t2 by the green curve.
t1
t2
m(x2)
−1
0
1
−1
0
1
framework. However, an advantage of a Gaussian processes viewpoint is that we
can consider covariance functions that can only be expressed in terms of an inﬁnite
number of basis functions.
For large training data sets, however, the direct application of Gaussian process
methods can become infeasible, and so a range of approximation schemes have been
developed that have better scaling with training set size than the exact approach
(Gibbs, 1997; Tresp, 2001; Smola and Bartlett, 2001; Williams and Seeger, 2001;
Csat´o and Opper, 2002; Seeger et al., 2003). Practical issues in the application of
Gaussian processes are discussed in Bishop and Nabney (2008).
We have introduced Gaussian process regression for the case of a single tar-
get variable. The extension of this formalism to multiple target variables, known
as co-kriging (Cressie, 1993), is straightforward. Various other extensions of Gaus-
Exercise 6.23
Figure 6.8
Illustration of Gaussian process re-
gression applied to the sinusoidal
data set in Figure A.6 in which the
three right-most data points have
been omitted.
The green curve
shows the sinusoidal function from
which the data points, shown in
blue, are obtained by sampling and
addition of Gaussian noise.
The
red line shows the mean of the
Gaussian process predictive distri-
bution, and the shaded region cor-
responds to plus and minus two
standard deviations.
Notice how
the uncertainty increases in the re-
gion to the right of the data points.
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


---
**Page 328**
6.4. Gaussian Processes
311
sian process regression have also been considered, for purposes such as modelling
the distribution over low-dimensional manifolds for unsupervised learning (Bishop
et al., 1998a) and the solution of stochastic differential equations (Graepel, 2003).
6.4.3
Learning the hyperparameters
The predictions of a Gaussian process model will depend, in part, on the choice
of covariance function. In practice, rather than ﬁxing the covariance function, we
may prefer to use a parametric family of functions and then infer the parameter
values from the data. These parameters govern such things as the length scale of the
correlations and the precision of the noise and correspond to the hyperparameters in
a standard parametric model.
Techniques for learning the hyperparameters are based on the evaluation of the
likelihood function p(t|θ) where θ denotes the hyperparameters of the Gaussian pro-
cess model. The simplest approach is to make a point estimate of θ by maximizing
the log likelihood function. Because θ represents a set of hyperparameters for the
regression problem, this can be viewed as analogous to the type 2 maximum like-
lihood procedure for linear regression models. Maximization of the log likelihood
Section 3.5
can be done using efﬁcient gradient-based optimization algorithms such as conjugate
gradients (Fletcher, 1987; Nocedal and Wright, 1999; Bishop and Nabney, 2008).
The log likelihood function for a Gaussian process regression model is easily
evaluated using the standard form for a multivariate Gaussian distribution, giving
ln p(t|θ) = −1
2 ln |CN| −1
2tTC−1
N t −N
2 ln(2π).
(6.69)
For nonlinear optimization, we also need the gradient of the log likelihood func-
tion with respect to the parameter vector θ. We shall assume that evaluation of the
derivatives of CN is straightforward, as would be the case for the covariance func-
tions considered in this chapter. Making use of the result (C.21) for the derivative of
C−1
N , together with the result (C.22) for the derivative of ln |CN|, we obtain
∂
∂θi
ln p(t|θ) = −1
2Tr

C−1
N
∂CN
∂θi

+ 1
2tTC−1
N
∂CN
∂θi
C−1
N t.
(6.70)
Because ln p(t|θ) will in general be a nonconvex function, it can have multiple max-
ima.
It is straightforward to introduce a prior over θ and to maximize the log poste-
rior using gradient-based methods. In a fully Bayesian treatment, we need to evaluate
marginals over θ weighted by the product of the prior p(θ) and the likelihood func-
tion p(t|θ). In general, however, exact marginalization will be intractable, and we
must resort to approximations.
The Gaussian process regression model gives a predictive distribution whose
mean and variance are functions of the input vector x. However, we have assumed
that the contribution to the predictive variance arising from the additive noise, gov-
erned by the parameter β, is a constant. For some problems, known as heteroscedas-
tic, the noise variance itself will also depend on x. To model this, we can extend the


---
**Page 329**
312
6. KERNEL METHODS
Figure 6.9
Samples from the ARD
prior for Gaussian processes,
in
which the kernel function is given by
(6.71). The left plot corresponds to
η1 = η2 = 1, and the right plot cor-
responds to η1 = 1, η2 = 0.01.
Gaussian process framework by introducing a second Gaussian process to represent
the dependence of β on the input x (Goldberg et al., 1998). Because β is a variance,
and hence nonnegative, we use the Gaussian process to model ln β(x).
6.4.4
Automatic relevance determination
In the previous section, we saw how maximum likelihood could be used to de-
termine a value for the correlation length-scale parameter in a Gaussian process.
This technique can usefully be extended by incorporating a separate parameter for
each input variable (Rasmussen and Williams, 2006). The result, as we shall see, is
that the optimization of these parameters by maximum likelihood allows the relative
importance of different inputs to be inferred from the data. This represents an exam-
ple in the Gaussian process context of automatic relevance determination, or ARD,
which was originally formulated in the framework of neural networks (MacKay,
1994; Neal, 1996). The mechanism by which appropriate inputs are preferred is
discussed in Section 7.2.2.
Consider a Gaussian process with a two-dimensional input space x = (x1, x2),
having a kernel function of the form
k(x, x′) = θ0 exp

−1
2
2

i=1
ηi(xi −x′
i)2

.
(6.71)
Samples from the resulting prior over functions y(x) are shown for two different
settings of the precision parameters ηi in Figure 6.9. We see that, as a particu-
lar parameter ηi becomes small, the function becomes relatively insensitive to the
corresponding input variable xi. By adapting these parameters to a data set using
maximum likelihood, it becomes possible to detect input variables that have little
effect on the predictive distribution, because the corresponding values of ηi will be
small. This can be useful in practice because it allows such inputs to be discarded.
ARD is illustrated using a simple synthetic data set having three inputs x1, x2 and x3
(Nabney, 2002) in Figure 6.10. The target variable t, is generated by sampling 100
values of x1 from a Gaussian, evaluating the function sin(2πx1), and then adding


---
**Page 330**
6.4. Gaussian Processes
313
Figure 6.10
Illustration
of
automatic
rele-
vance determination in a Gaus-
sian process for a synthetic prob-
lem having three inputs x1, x2,
and x3,
for which the curves
show the corresponding values of
the hyperparameters η1 (red), η2
(green), and η3 (blue) as a func-
tion of the number of iterations
when
optimizing
the
marginal
likelihood.
Details are given in
the text.
Note the logarithmic
scale on the vertical axis.
0
20
40
60
80
100
10
−4
10
−2
10
0
10
2
Gaussian noise. Values of x2 are given by copying the corresponding values of x1
and adding noise, and values of x3 are sampled from an independent Gaussian dis-
tribution. Thus x1 is a good predictor of t, x2 is a more noisy predictor of t, and x3
has only chance correlations with t. The marginal likelihood for a Gaussian process
with ARD parameters η1, η2, η3 is optimized using the scaled conjugate gradients
algorithm. We see from Figure 6.10 that η1 converges to a relatively large value, η2
converges to a much smaller value, and η3 becomes very small indicating that x3 is
irrelevant for predicting t.
The ARD framework is easily incorporated into the exponential-quadratic kernel
(6.63) to give the following form of kernel function, which has been found useful for
applications of Gaussian processes to a range of regression problems
k(xn, xm) = θ0 exp

−1
2
D

i=1
ηi(xni −xmi)2

+ θ2 + θ3
D

i=1
xnixmi
(6.72)
where D is the dimensionality of the input space.
6.4.5
Gaussian processes for classiﬁcation
In a probabilistic approach to classiﬁcation, our goal is to model the posterior
probabilities of the target variable for a new input vector, given a set of training
data. These probabilities must lie in the interval (0, 1), whereas a Gaussian process
model makes predictions that lie on the entire real axis. However, we can easily
adapt Gaussian processes to classiﬁcation problems by transforming the output of
the Gaussian process using an appropriate nonlinear activation function.
Consider ﬁrst the two-class problem with a target variable t ∈{0, 1}. If we de-
ﬁne a Gaussian process over a function a(x) and then transform the function using
a logistic sigmoid y = σ(a), given by (4.59), then we will obtain a non-Gaussian
stochastic process over functions y(x) where y ∈(0, 1). This is illustrated for the
case of a one-dimensional input space in Figure 6.11 in which the probability distri-


---
**Page 331**
314
6. KERNEL METHODS
−1
−0.5
0
0.5
1
−10
−5
0
5
10
−1
−0.5
0
0.5
1
0
0.25
0.5
0.75
1
Figure 6.11
The left plot shows a sample from a Gaussian process prior over functions a(x), and the right plot
shows the result of transforming this sample using a logistic sigmoid function.
bution over the target variable t is then given by the Bernoulli distribution
p(t|a) = σ(a)t(1 −σ(a))1−t.
(6.73)
As usual, we denote the training set inputs by x1, . . . , xN with corresponding
observed target variables t = (t1, . . . , tN)T. We also consider a single test point
xN+1 with target value tN+1. Our goal is to determine the predictive distribution
p(tN+1|t), where we have left the conditioning on the input variables implicit. To do
this we introduce a Gaussian process prior over the vector aN+1, which has compo-
nents a(x1), . . . , a(xN+1). This in turn deﬁnes a non-Gaussian process over tN+1,
and by conditioning on the training data tN we obtain the required predictive distri-
bution. The Gaussian process prior for aN+1 takes the form
p(aN+1) = N(aN+1|0, CN+1).
(6.74)
Unlike the regression case, the covariance matrix no longer includes a noise term
because we assume that all of the training data points are correctly labelled. How-
ever, for numerical reasons it is convenient to introduce a noise-like term governed
by a parameter ν that ensures that the covariance matrix is positive deﬁnite. Thus
the covariance matrix CN+1 has elements given by
C(xn, xm) = k(xn, xm) + νδnm
(6.75)
where k(xn, xm) is any positive semideﬁnite kernel function of the kind considered
in Section 6.2, and the value of ν is typically ﬁxed in advance. We shall assume that
the kernel function k(x, x′) is governed by a vector θ of parameters, and we shall
later discuss how θ may be learned from the training data.
For two-class problems, it is sufﬁcient to predict p(tN+1 = 1|tN) because the
value of p(tN+1 = 0|tN) is then given by 1 −p(tN+1 = 1|tN). The required


---
**Page 332**
6.4. Gaussian Processes
315
predictive distribution is given by
p(tN+1 = 1|tN) =

p(tN+1 = 1|aN+1)p(aN+1|tN) daN+1
(6.76)
where p(tN+1 = 1|aN+1) = σ(aN+1).
This integral is analytically intractable, and so may be approximated using sam-
pling methods (Neal, 1997). Alternatively, we can consider techniques based on
an analytical approximation. In Section 4.5.2, we derived the approximate formula
(4.153) for the convolution of a logistic sigmoid with a Gaussian distribution. We
can use this result to evaluate the integral in (6.76) provided we have a Gaussian
approximation to the posterior distribution p(aN+1|tN). The usual justiﬁcation for a
Gaussian approximation to a posterior distribution is that the true posterior will tend
to a Gaussian as the number of data points increases as a consequence of the central
limit theorem. In the case of Gaussian processes, the number of variables grows with
Section 2.3
the number of data points, and so this argument does not apply directly. However, if
we consider increasing the number of data points falling in a ﬁxed region of x space,
then the corresponding uncertainty in the function a(x) will decrease, again leading
asymptotically to a Gaussian (Williams and Barber, 1998).
Three different approaches to obtaining a Gaussian approximation have been
considered. One technique is based on variational inference (Gibbs and MacKay,
Section 10.1
2000) and makes use of the local variational bound (10.144) on the logistic sigmoid.
This allows the product of sigmoid functions to be approximated by a product of
Gaussians thereby allowing the marginalization over aN to be performed analyti-
cally. The approach also yields a lower bound on the likelihood function p(tN|θ).
The variational framework for Gaussian process classiﬁcation can also be extended
to multiclass (K > 2) problems by using a Gaussian approximation to the softmax
function (Gibbs, 1997).
A second approach uses expectation propagation (Opper and Winther, 2000b;
Section 10.7
Minka, 2001b; Seeger, 2003). Because the true posterior distribution is unimodal, as
we shall see shortly, the expectation propagation approach can give good results.
6.4.6
Laplace approximation
The third approach to Gaussian process classiﬁcation is based on the Laplace
approximation, which we now consider in detail. In order to evaluate the predictive
Section 4.4
distribution (6.76), we seek a Gaussian approximation to the posterior distribution
over aN+1, which, using Bayes’ theorem, is given by
p(aN+1|tN)
=

p(aN+1, aN|tN) daN
=
1
p(tN)

p(aN+1, aN)p(tN|aN+1, aN) daN
=
1
p(tN)

p(aN+1|aN)p(aN)p(tN|aN) daN
=

p(aN+1|aN)p(aN|tN) daN
(6.77)


---
**Page 333**
316
6. KERNEL METHODS
where we have used p(tN|aN+1, aN) = p(tN|aN). The conditional distribution
p(aN+1|aN) is obtained by invoking the results (6.66) and (6.67) for Gaussian pro-
cess regression, to give
p(aN+1|aN) = N(aN+1|kTC−1
N aN, c −kTC−1
N k).
(6.78)
We can therefore evaluate the integral in (6.77) by ﬁnding a Laplace approximation
for the posterior distribution p(aN|tN), and then using the standard result for the
convolution of two Gaussian distributions.
The prior p(aN) is given by a zero-mean Gaussian process with covariance ma-
trix CN, and the data term (assuming independence of the data points) is given by
p(tN|aN) =
N

n=1
σ(an)tn(1 −σ(an))1−tn =
N

n=1
eantnσ(−an).
(6.79)
We then obtain the Laplace approximation by Taylor expanding the logarithm of
p(aN|tN), which up to an additive normalization constant is given by the quantity
Ψ(aN)
=
ln p(aN) + ln p(tN|aN)
=
−1
2aT
NC−1
N aN −N
2 ln(2π) −1
2 ln |CN| + tT
NaN
−
N

n=1
ln(1 + ean) + const.
(6.80)
First we need to ﬁnd the mode of the posterior distribution, and this requires that we
evaluate the gradient of Ψ(aN), which is given by
∇Ψ(aN) = tN −σN −C−1
N aN
(6.81)
where σN is a vector with elements σ(an). We cannot simply ﬁnd the mode by
setting this gradient to zero, because σN depends nonlinearly on aN, and so we
resort to an iterative scheme based on the Newton-Raphson method, which gives rise
to an iterative reweighted least squares (IRLS) algorithm. This requires the second
Section 4.3.3
derivatives of Ψ(aN), which we also require for the Laplace approximation anyway,
and which are given by
∇∇Ψ(aN) = −WN −C−1
N
(6.82)
where WN is a diagonal matrix with elements σ(an)(1 −σ(an)), and we have used
the result (4.88) for the derivative of the logistic sigmoid function. Note that these
diagonal elements lie in the range (0, 1/4), and hence WN is a positive deﬁnite
matrix. Because CN (and hence its inverse) is positive deﬁnite by construction, and
because the sum of two positive deﬁnite matrices is also positive deﬁnite, we see
Exercise 6.24
that the Hessian matrix A = −∇∇Ψ(aN) is positive deﬁnite and so the posterior
distribution p(aN|tN) is log convex and therefore has a single mode that is the global


---
**Page 334**
6.4. Gaussian Processes
317
maximum. The posterior distribution is not Gaussian, however, because the Hessian
is a function of aN.
Using the Newton-Raphson formula (4.92), the iterative update equation for aN
is given by
Exercise 6.25
anew
N
= CN(I + WNCN)−1 {tN −σN + WNaN} .
(6.83)
These equations are iterated until they converge to the mode which we denote by
a⋆
N. At the mode, the gradient ∇Ψ(aN) will vanish, and hence a⋆
N will satisfy
a⋆
N = CN(tN −σN).
(6.84)
Once we have found the mode a⋆
N of the posterior, we can evaluate the Hessian
matrix given by
H = −∇∇Ψ(aN) = WN + C−1
N
(6.85)
where the elements of WN are evaluated using a⋆
N. This deﬁnes our Gaussian ap-
proximation to the posterior distribution p(aN|tN) given by
q(aN) = N(aN|a⋆
N, H−1).
(6.86)
We can now combine this with (6.78) and hence evaluate the integral (6.77). Because
this corresponds to a linear-Gaussian model, we can use the general result (2.115) to
give
Exercise 6.26
E[aN+1|tN]
=
kT(tN −σN)
(6.87)
var[aN+1|tN]
=
c −kT(W−1
N + CN)−1k.
(6.88)
Now that we have a Gaussian distribution for p(aN+1|tN), we can approximate
the integral (6.76) using the result (4.153). As with the Bayesian logistic regression
model of Section 4.5, if we are only interested in the decision boundary correspond-
ing to p(tN+1|tN) = 0.5, then we need only consider the mean and we can ignore
the effect of the variance.
We also need to determine the parameters θ of the covariance function. One
approach is to maximize the likelihood function given by p(tN|θ) for which we need
expressions for the log likelihood and its gradient. If desired, suitable regularization
terms can also be added, leading to a penalized maximum likelihood solution. The
likelihood function is deﬁned by
p(tN|θ) =

p(tN|aN)p(aN|θ) daN.
(6.89)
This integral is analytically intractable, so again we make use of the Laplace approx-
imation. Using the result (4.135), we obtain the following approximation for the log
of the likelihood function
ln p(tN|θ) = Ψ(a⋆
N) −1
2 ln |WN + C−1
N | + N
2 ln(2π)
(6.90)


---
**Page 335**
318
6. KERNEL METHODS
where Ψ(a⋆
N) = ln p(a⋆
N|θ) + ln p(tN|a⋆
N). We also need to evaluate the gradient
of ln p(tN|θ) with respect to the parameter vector θ. Note that changes in θ will
cause changes in a⋆
N, leading to additional terms in the gradient. Thus, when we
differentiate (6.90) with respect to θ, we obtain two sets of terms, the ﬁrst arising
from the dependence of the covariance matrix CN on θ, and the rest arising from
dependence of a⋆
N on θ.
The terms arising from the explicit dependence on θ can be found by using
(6.80) together with the results (C.21) and (C.22), and are given by
∂ln p(tN|θ)
∂θj
=
1
2a⋆T
N C−1
N
∂CN
∂θj
C−1
N a⋆
N
−1
2Tr

(I + CNWN)−1WN
∂CN
∂θj

.
(6.91)
To compute the terms arising from the dependence of a⋆
N on θ, we note that
the Laplace approximation has been constructed such that Ψ(aN) has zero gradient
at aN = a⋆
N, and so Ψ(a⋆
N) gives no contribution to the gradient as a result of its
dependence on a⋆
N. This leaves the following contribution to the derivative with
respect to a component θj of θ
−1
2
N

n=1
∂ln |WN + C−1
N |
∂a⋆n
∂a⋆
n
∂θj
= −1
2
N

n=1

(I + CNWN)−1CN
	
nn σ⋆
n(1 −σ⋆
n)(1 −2σ⋆
n)∂a⋆
n
∂θj
(6.92)
where σ⋆
n = σ(a⋆
n), and again we have used the result (C.22) together with the
deﬁnition of WN. We can evaluate the derivative of a⋆
N with respect to θj by differ-
entiating the relation (6.84) with respect to θj to give
∂a⋆
n
∂θj
= ∂CN
∂θj
(tN −σN) −CNWN
∂a⋆
n
∂θj
.
(6.93)
Rearranging then gives
∂a⋆
n
∂θj
= (I + WNCN)−1 ∂CN
∂θj
(tN −σN).
(6.94)
Combining (6.91), (6.92), and (6.94), we can evaluate the gradient of the log
likelihood function, which can be used with standard nonlinear optimization algo-
rithms in order to determine a value for θ.
We can illustrate the application of the Laplace approximation for Gaussian pro-
cesses using the synthetic two-class data set shown in Figure 6.12. Extension of the
Appendix A
Laplace approximation to Gaussian processes involving K > 2 classes, using the
softmax activation function, is straightforward (Williams and Barber, 1998).


---
**Page 336**
6.4. Gaussian Processes
319
−2
0
2
−2
0
2
Figure 6.12
Illustration of the use of a Gaussian process for classiﬁcation, showing the data on the left together
with the optimal decision boundary from the true distribution in green, and the decision boundary from the
Gaussian process classiﬁer in black. On the right is the predicted posterior probability for the blue and red
classes together with the Gaussian process decision boundary.
6.4.7
Connection to neural networks
We have seen that the range of functions which can be represented by a neural
network is governed by the number M of hidden units, and that, for sufﬁciently
large M, a two-layer network can approximate any given function with arbitrary
accuracy. In the framework of maximum likelihood, the number of hidden units
needs to be limited (to a level dependent on the size of the training set) in order
to avoid over-ﬁtting. However, from a Bayesian perspective it makes little sense to
limit the number of parameters in the network according to the size of the training
set.
In a Bayesian neural network, the prior distribution over the parameter vector
w, in conjunction with the network function f(x, w), produces a prior distribution
over functions from y(x) where y is the vector of network outputs. Neal (1996)
has shown that, for a broad class of prior distributions over w, the distribution of
functions generated by a neural network will tend to a Gaussian process in the limit
M →∞. It should be noted, however, that in this limit the output variables of the
neural network become independent. One of the great merits of neural networks is
that the outputs share the hidden units and so they can ‘borrow statistical strength’
from each other, that is, the weights associated with each hidden unit are inﬂuenced
by all of the output variables not just by one of them. This property is therefore lost
in the Gaussian process limit.
We have seen that a Gaussian process is determined by its covariance (kernel)
function. Williams (1998) has given explicit forms for the covariance in the case of
two speciﬁc choices for the hidden unit activation function (probit and Gaussian).
These kernel functions k(x, x′) are nonstationary, i.e. they cannot be expressed as
a function of the difference x −x′, as a consequence of the Gaussian weight prior
being centred on zero which breaks translation invariance in weight space.


---
**Page 337**
320
6. KERNEL METHODS
By working directly with the covariance function we have implicitly marginal-
ized over the distribution of weights. If the weight prior is governed by hyperpa-
rameters, then their values will determine the length scales of the distribution over
functions, as can be understood by studying the examples in Figure 5.11 for the case
of a ﬁnite number of hidden units. Note that we cannot marginalize out the hyperpa-
rameters analytically, and must instead resort to techniques of the kind discussed in
Section 6.4.
Exercises
6.1
(⋆⋆) www
Consider the dual formulation of the least squares linear regression
problem given in Section 6.1. Show that the solution for the components an of
the vector a can be expressed as a linear combination of the elements of the vector
φ(xn). Denoting these coefﬁcients by the vector w, show that the dual of the dual
formulation is given by the original representation in terms of the parameter vector
w.
6.2
(⋆⋆)
In this exercise, we develop a dual formulation of the perceptron learning
algorithm. Using the perceptron learning rule (4.55), show that the learned weight
vector w can be written as a linear combination of the vectors tnφ(xn) where tn ∈
{−1, +1}. Denote the coefﬁcients of this linear combination by αn and derive a
formulation of the perceptron learning algorithm, and the predictive function for the
perceptron, in terms of the αn. Show that the feature vector φ(x) enters only in the
form of the kernel function k(x, x′) = φ(x)Tφ(x′).
6.3
(⋆)
The nearest-neighbour classiﬁer (Section 2.5.2) assigns a new input vector x
to the same class as that of the nearest input vector xn from the training set, where
in the simplest case, the distance is deﬁned by the Euclidean metric ∥x −xn∥2. By
expressing this rule in terms of scalar products and then making use of kernel sub-
stitution, formulate the nearest-neighbour classiﬁer for a general nonlinear kernel.
6.4
(⋆)
In Appendix C, we give an example of a matrix that has positive elements but
that has a negative eigenvalue and hence that is not positive deﬁnite. Find an example
of the converse property, namely a 2 × 2 matrix with positive eigenvalues yet that
has at least one negative element.
6.5
(⋆) www
Verify the results (6.13) and (6.14) for constructing valid kernels.
6.6
(⋆) Verify the results (6.15) and (6.16) for constructing valid kernels.
6.7
(⋆) www
Verify the results (6.17) and (6.18) for constructing valid kernels.
6.8
(⋆) Verify the results (6.19) and (6.20) for constructing valid kernels.
6.9
(⋆) Verify the results (6.21) and (6.22) for constructing valid kernels.
6.10
(⋆)
Show that an excellent choice of kernel for learning a function f(x) is given
by k(x, x′) = f(x)f(x′) by showing that a linear learning machine based on this
kernel will always ﬁnd a solution proportional to f(x).


---
**Page 338**
Exercises
321
6.11
(⋆)
By making use of the expansion (6.25), and then expanding the middle factor
as a power series, show that the Gaussian kernel (6.23) can be expressed as the inner
product of an inﬁnite-dimensional feature vector.
6.12
(⋆⋆) www
Consider the space of all possible subsets A of a given ﬁxed set D.
Show that the kernel function (6.27) corresponds to an inner product in a feature
space of dimensionality 2|D| deﬁned by the mapping φ(A) where A is a subset of D
and the element φU(A), indexed by the subset U, is given by
φU(A) =

1,
if U ⊆A;
0,
otherwise.
(6.95)
Here U ⊆A denotes that U is either a subset of A or is equal to A.
6.13
(⋆)
Show that the Fisher kernel, deﬁned by (6.33), remains invariant if we make
a nonlinear transformation of the parameter vector θ →ψ(θ), where the function
ψ(·) is invertible and differentiable.
6.14
(⋆) www
Write down the form of the Fisher kernel, deﬁned by (6.33), for the
case of a distribution p(x|µ) = N(x|µ, S) that is Gaussian with mean µ and ﬁxed
covariance S.
6.15
(⋆)
By considering the determinant of a 2 × 2 Gram matrix, show that a positive-
deﬁnite kernel function k(x, x′) satisﬁes the Cauchy-Schwartz inequality
k(x1, x2)2 ⩽k(x1, x1)k(x2, x2).
(6.96)
6.16
(⋆⋆)
Consider a parametric model governed by the parameter vector w together
with a data set of input values x1, . . . , xN and a nonlinear feature mapping φ(x).
Suppose that the dependence of the error function on w takes the form
J(w) = f(wTφ(x1), . . . , wTφ(xN)) + g(wTw)
(6.97)
where g(·) is a monotonically increasing function. By writing w in the form
w =
N

n=1
αnφ(xn) + w⊥
(6.98)
show that the value of w that minimizes J(w) takes the form of a linear combination
of the basis functions φ(xn) for n = 1, . . . , N.
6.17
(⋆⋆) www
Consider the sum-of-squares error function (6.39) for data having
noisy inputs, where ν(ξ) is the distribution of the noise. Use the calculus of vari-
ations to minimize this error function with respect to the function y(x), and hence
show that the optimal solution is given by an expansion of the form (6.40) in which
the basis functions are given by (6.41).


---
**Page 339**
322
6. KERNEL METHODS
6.18
(⋆)
Consider a Nadaraya-Watson model with one input variable x and one target
variable t having Gaussian components with isotropic covariances, so that the co-
variance matrix is given by σ2I where I is the unit matrix. Write down expressions
for the conditional density p(t|x) and for the conditional mean E[t|x] and variance
var[t|x], in terms of the kernel function k(x, xn).
6.19
(⋆⋆)
Another viewpoint on kernel regression comes from a consideration of re-
gression problems in which the input variables as well as the target variables are
corrupted with additive noise. Suppose each target value tn is generated as usual
by taking a function y(zn) evaluated at a point zn, and adding Gaussian noise. The
value of zn is not directly observed, however, but only a noise corrupted version
xn = zn + ξn where the random variable ξ is governed by some distribution g(ξ).
Consider a set of observations {xn, tn}, where n = 1, . . . , N, together with a cor-
responding sum-of-squares error function deﬁned by averaging over the distribution
of input noise to give
E = 1
2
N

n=1

{y(xn −ξn) −tn}2 g(ξn) dξn.
(6.99)
By minimizing E with respect to the function y(z) using the calculus of variations
(Appendix D), show that optimal solution for y(x) is given by a Nadaraya-Watson
kernel regression solution of the form (6.45) with a kernel of the form (6.46).
6.20
(⋆⋆) www
Verify the results (6.66) and (6.67).
6.21
(⋆⋆) www
Consider a Gaussian process regression model in which the kernel
function is deﬁned in terms of a ﬁxed set of nonlinear basis functions. Show that the
predictive distribution is identical to the result (3.58) obtained in Section 3.3.2 for the
Bayesian linear regression model. To do this, note that both models have Gaussian
predictive distributions, and so it is only necessary to show that the conditional mean
and variance are the same. For the mean, make use of the matrix identity (C.6), and
for the variance, make use of the matrix identity (C.7).
6.22
(⋆⋆)
Consider a regression problem with N training set input vectors x1, . . . , xN
and L test set input vectors xN+1, . . . , xN+L, and suppose we deﬁne a Gaussian
process prior over functions t(x). Derive an expression for the joint predictive dis-
tribution for t(xN+1), . . . , t(xN+L), given the values of t(x1), . . . , t(xN). Show the
marginal of this distribution for one of the test observations tj where N + 1 ⩽j ⩽
N + L is given by the usual Gaussian process regression result (6.66) and (6.67).
6.23
(⋆⋆) www
Consider a Gaussian process regression model in which the target
variable t has dimensionality D. Write down the conditional distribution of tN+1
for a test input vector xN+1, given a training set of input vectors x1, . . . , xN+1 and
corresponding target observations t1, . . . , tN.
6.24
(⋆) Show that a diagonal matrix W whose elements satisfy 0 < Wii < 1 is positive
deﬁnite. Show that the sum of two positive deﬁnite matrices is itself positive deﬁnite.


---
**Page 340**
Exercises
323
6.25
(⋆) www
Using the Newton-Raphson formula (4.92), derive the iterative update
formula (6.83) for ﬁnding the mode a⋆
N of the posterior distribution in the Gaussian
process classiﬁcation model.
6.26
(⋆)
Using the result (2.115), derive the expressions (6.87) and (6.88) for the mean
and variance of the posterior distribution p(aN+1|tN) in the Gaussian process clas-
siﬁcation model.
6.27
(⋆⋆⋆) Derive the result (6.90) for the log likelihood function in the Laplace approx-
imation framework for Gaussian process classiﬁcation. Similarly, derive the results
(6.91), (6.92), and (6.94) for the terms in the gradient of the log likelihood.


---
**Page 341**
7
Sparse Kernel
Machines
In the previous chapter, we explored a variety of learning algorithms based on non-
linear kernels. One of the signiﬁcant limitations of many such algorithms is that
the kernel function k(xn, xm) must be evaluated for all possible pairs xn and xm
of training points, which can be computationally infeasible during training and can
lead to excessive computation times when making predictions for new data points.
In this chapter we shall look at kernel-based algorithms that have sparse solutions,
so that predictions for new inputs depend only on the kernel function evaluated at a
subset of the training data points.
We begin by looking in some detail at the support vector machine (SVM), which
became popular in some years ago for solving problems in classiﬁcation, regression,
and novelty detection. An important property of support vector machines is that the
determination of the model parameters corresponds to a convex optimization prob-
lem, and so any local solution is also a global optimum. Because the discussion of
support vector machines makes extensive use of Lagrange multipliers, the reader is
325


---
**Page 342**
326
7. SPARSE KERNEL MACHINES
encouraged to review the key concepts covered in Appendix E. Additional infor-
mation on support vector machines can be found in Vapnik (1995), Burges (1998),
Cristianini and Shawe-Taylor (2000), M¨uller et al. (2001), Sch¨olkopf and Smola
(2002), and Herbrich (2002).
The SVM is a decision machine and so does not provide posterior probabilities.
We have already discussed some of the beneﬁts of determining probabilities in Sec-
tion 1.5.4. An alternative sparse kernel technique, known as the relevance vector
machine (RVM), is based on a Bayesian formulation and provides posterior proba-
Section 7.2
bilistic outputs, as well as having typically much sparser solutions than the SVM.
7.1. Maximum Margin Classiﬁers
We begin our discussion of support vector machines by returning to the two-class
classiﬁcation problem using linear models of the form
y(x) = wTφ(x) + b
(7.1)
where φ(x) denotes a ﬁxed feature-space transformation, and we have made the
bias parameter b explicit. Note that we shall shortly introduce a dual representation
expressed in terms of kernel functions, which avoids having to work explicitly in
feature space. The training data set comprises N input vectors x1, . . . , xN, with
corresponding target values t1, . . . , tN where tn ∈{−1, 1}, and new data points x
are classiﬁed according to the sign of y(x).
We shall assume for the moment that the training data set is linearly separable in
feature space, so that by deﬁnition there exists at least one choice of the parameters
w and b such that a function of the form (7.1) satisﬁes y(xn) > 0 for points having
tn = +1 and y(xn) < 0 for points having tn = −1, so that tny(xn) > 0 for all
training data points.
There may of course exist many such solutions that separate the classes exactly.
In Section 4.1.7, we described the perceptron algorithm that is guaranteed to ﬁnd
a solution in a ﬁnite number of steps. The solution that it ﬁnds, however, will be
dependent on the (arbitrary) initial values chosen for w and b as well as on the
order in which the data points are presented. If there are multiple solutions all of
which classify the training data set exactly, then we should try to ﬁnd the one that
will give the smallest generalization error. The support vector machine approaches
this problem through the concept of the margin, which is deﬁned to be the smallest
distance between the decision boundary and any of the samples, as illustrated in
Figure 7.1.
In support vector machines the decision boundary is chosen to be the one for
which the margin is maximized. The maximum margin solution can be motivated us-
ing computational learning theory, also known as statistical learning theory. How-
Section 7.1.5
ever, a simple insight into the origins of maximum margin has been given by Tong
and Koller (2000) who consider a framework for classiﬁcation based on a hybrid of
generative and discriminative approaches. They ﬁrst model the distribution over in-
put vectors x for each class using a Parzen density estimator with Gaussian kernels


---
**Page 343**
7.1. Maximum Margin Classiﬁers
327
y = 1
y = 0
y = −1
margin
y = 1
y = 0
y = −1
Figure 7.1
The margin is deﬁned as the perpendicular distance between the decision boundary and the closest
of the data points, as shown on the left ﬁgure. Maximizing the margin leads to a particular choice of decision
boundary, as shown on the right. The location of this boundary is determined by a subset of the data points,
known as support vectors, which are indicated by the circles.
having a common parameter σ2. Together with the class priors, this deﬁnes an opti-
mal misclassiﬁcation-rate decision boundary. However, instead of using this optimal
boundary, they determine the best hyperplane by minimizing the probability of error
relative to the learned density model. In the limit σ2 →0, the optimal hyperplane
is shown to be the one having maximum margin. The intuition behind this result is
that as σ2 is reduced, the hyperplane is increasingly dominated by nearby data points
relative to more distant ones. In the limit, the hyperplane becomes independent of
data points that are not support vectors.
We shall see in Figure 10.13 that marginalization with respect to the prior distri-
bution of the parameters in a Bayesian approach for a simple linearly separable data
set leads to a decision boundary that lies in the middle of the region separating the
data points. The large margin solution has similar behaviour.
Recall from Figure 4.1 that the perpendicular distance of a point x from a hyper-
plane deﬁned by y(x) = 0 where y(x) takes the form (7.1) is given by |y(x)|/∥w∥.
Furthermore, we are only interested in solutions for which all data points are cor-
rectly classiﬁed, so that tny(xn) > 0 for all n. Thus the distance of a point xn to the
decision surface is given by
tny(xn)
∥w∥
= tn(wTφ(xn) + b)
∥w∥
.
(7.2)
The margin is given by the perpendicular distance to the closest point xn from the
data set, and we wish to optimize the parameters w and b in order to maximize this
distance. Thus the maximum margin solution is found by solving
arg max
w,b
 1
∥w∥min
n

tn

wTφ(xn) + b
	
(7.3)
where we have taken the factor 1/∥w∥outside the optimization over n because w


---
**Page 344**
328
7. SPARSE KERNEL MACHINES
does not depend on n. Direct solution of this optimization problem would be very
complex, and so we shall convert it into an equivalent problem that is much easier
to solve. To do this we note that if we make the rescaling w →κw and b →κb,
then the distance from any point xn to the decision surface, given by tny(xn)/∥w∥,
is unchanged. We can use this freedom to set
tn

wTφ(xn) + b

= 1
(7.4)
for the point that is closest to the surface. In this case, all data points will satisfy the
constraints
tn

wTφ(xn) + b
⩾1,
n = 1, . . . , N.
(7.5)
This is known as the canonical representation of the decision hyperplane. In the
case of data points for which the equality holds, the constraints are said to be active,
whereas for the remainder they are said to be inactive. By deﬁnition, there will
always be at least one active constraint, because there will always be a closest point,
and once the margin has been maximized there will be at least two active constraints.
The optimization problem then simply requires that we maximize ∥w∥−1, which is
equivalent to minimizing ∥w∥2, and so we have to solve the optimization problem
arg min
w,b
1
2∥w∥2
(7.6)
subject to the constraints given by (7.5). The factor of 1/2 in (7.6) is included for
later convenience. This is an example of a quadratic programming problem in which
we are trying to minimize a quadratic function subject to a set of linear inequality
constraints. It appears that the bias parameter b has disappeared from the optimiza-
tion. However, it is determined implicitly via the constraints, because these require
that changes to ∥w∥be compensated by changes to b. We shall see how this works
shortly.
In order to solve this constrained optimization problem, we introduce Lagrange
multipliers an ⩾0, with one multiplier an for each of the constraints in (7.5), giving
Appendix E
the Lagrangian function
L(w, b, a) = 1
2∥w∥2 −
N

n=1
an

tn(wTφ(xn) + b) −1
(7.7)
where a = (a1, . . . , aN)T. Note the minus sign in front of the Lagrange multiplier
term, because we are minimizing with respect to w and b, and maximizing with
respect to a. Setting the derivatives of L(w, b, a) with respect to w and b equal to
zero, we obtain the following two conditions
w
=
N

n=1
antnφ(xn)
(7.8)
0
=
N

n=1
antn.
(7.9)


---
**Page 345**
7.1. Maximum Margin Classiﬁers
329
Eliminating w and b from L(w, b, a) using these conditions then gives the dual
representation of the maximum margin problem in which we maximize
L(a) =
N

n=1
an −1
2
N

n=1
N

m=1
anamtntmk(xn, xm)
(7.10)
with respect to a subject to the constraints
an
⩾
0,
n = 1, . . . , N,
(7.11)
N

n=1
antn
=
0.
(7.12)
Here the kernel function is deﬁned by k(x, x′) = φ(x)Tφ(x′). Again, this takes the
form of a quadratic programming problem in which we optimize a quadratic function
of a subject to a set of inequality constraints. We shall discuss techniques for solving
such quadratic programming problems in Section 7.1.1.
The solution to a quadratic programming problem in M variables in general has
computational complexity that is O(M 3). In going to the dual formulation we have
turned the original optimization problem, which involved minimizing (7.6) over M
variables, into the dual problem (7.10), which has N variables. For a ﬁxed set of
basis functions whose number M is smaller than the number N of data points, the
move to the dual problem appears disadvantageous. However, it allows the model to
be reformulated using kernels, and so the maximum margin classiﬁer can be applied
efﬁciently to feature spaces whose dimensionality exceeds the number of data points,
including inﬁnite feature spaces. The kernel formulation also makes clear the role
of the constraint that the kernel function k(x, x′) be positive deﬁnite, because this
ensures that the Lagrangian function L(a) is bounded below, giving rise to a well-
deﬁned optimization problem.
In order to classify new data points using the trained model, we evaluate the sign
of y(x) deﬁned by (7.1). This can be expressed in terms of the parameters {an} and
the kernel function by substituting for w using (7.8) to give
y(x) =
N

n=1
antnk(x, xn) + b.
(7.13)
Joseph-Louis Lagrange
1736–1813
Although widely considered to be
a French mathematician, Lagrange
was born in Turin in Italy. By the age
of nineteen, he had already made
important contributions mathemat-
ics and had been appointed as Pro-
fessor at the Royal Artillery School in Turin. For many
years, Euler worked hard to persuade Lagrange to
move to Berlin, which he eventually did in 1766 where
he succeeded Euler as Director of Mathematics at
the Berlin Academy.
Later he moved to Paris, nar-
rowly escaping with his life during the French revo-
lution thanks to the personal intervention of Lavoisier
(the French chemist who discovered oxygen) who him-
self was later executed at the guillotine.
Lagrange
made key contributions to the calculus of variations
and the foundations of dynamics.


---
**Page 346**
330
7. SPARSE KERNEL MACHINES
In Appendix E, we show that a constrained optimization of this form satisﬁes the
Karush-Kuhn-Tucker (KKT) conditions, which in this case require that the following
three properties hold
an
⩾
0
(7.14)
tny(xn) −1
⩾
0
(7.15)
an {tny(xn) −1}
=
0.
(7.16)
Thus for every data point, either an = 0 or tny(xn) = 1. Any data point for
which an = 0 will not appear in the sum in (7.13) and hence plays no role in making
predictions for new data points. The remaining data points are called support vectors,
and because they satisfy tny(xn) = 1, they correspond to points that lie on the
maximum margin hyperplanes in feature space, as illustrated in Figure 7.1. This
property is central to the practical applicability of support vector machines. Once
the model is trained, a signiﬁcant proportion of the data points can be discarded and
only the support vectors retained.
Having solved the quadratic programming problem and found a value for a, we
can then determine the value of the threshold parameter b by noting that any support
vector xn satisﬁes tny(xn) = 1. Using (7.13) this gives
tn

m∈S
amtmk(xn, xm) + b

= 1
(7.17)
where S denotes the set of indices of the support vectors. Although we can solve
this equation for b using an arbitrarily chosen support vector xn, a numerically more
stable solution is obtained by ﬁrst multiplying through by tn, making use of t2
n = 1,
and then averaging these equations over all support vectors and solving for b to give
b =
1
NS

n∈S

tn −

m∈S
amtmk(xn, xm)

(7.18)
where NS is the total number of support vectors.
For later comparison with alternative models, we can express the maximum-
margin classiﬁer in terms of the minimization of an error function, with a simple
quadratic regularizer, in the form
N

n=1
E∞(y(xn)tn −1) + λ∥w∥2
(7.19)
where E∞(z) is a function that is zero if z ⩾0 and ∞otherwise and ensures that
the constraints (7.5) are satisﬁed. Note that as long as the regularization parameter
satisﬁes λ > 0, its precise value plays no role.
Figure 7.2 shows an example of the classiﬁcation resulting from training a sup-
port vector machine on a simple synthetic data set using a Gaussian kernel of the


---
**Page 347**
7.1. Maximum Margin Classiﬁers
331
Figure 7.2
Example of synthetic data from
two classes in two dimensions
showing contours of constant
y(x) obtained from a support
vector machine having a Gaus-
sian kernel function. Also shown
are the decision boundary, the
margin boundaries, and the sup-
port vectors.
form (6.23). Although the data set is not linearly separable in the two-dimensional
data space x, it is linearly separable in the nonlinear feature space deﬁned implicitly
by the nonlinear kernel function. Thus the training data points are perfectly separated
in the original data space.
This example also provides a geometrical insight into the origin of sparsity in
the SVM. The maximum margin hyperplane is deﬁned by the location of the support
vectors. Other data points can be moved around freely (so long as they remain out-
side the margin region) without changing the decision boundary, and so the solution
will be independent of such data points.
7.1.1
Overlapping class distributions
So far, we have assumed that the training data points are linearly separable in the
feature space φ(x). The resulting support vector machine will give exact separation
of the training data in the original input space x, although the corresponding decision
boundary will be nonlinear. In practice, however, the class-conditional distributions
may overlap, in which case exact separation of the training data can lead to poor
generalization.
We therefore need a way to modify the support vector machine so as to allow
some of the training points to be misclassiﬁed. From (7.19) we see that in the case
of separable classes, we implicitly used an error function that gave inﬁnite error
if a data point was misclassiﬁed and zero error if it was classiﬁed correctly, and
then optimized the model parameters to maximize the margin. We now modify this
approach so that data points are allowed to be on the ‘wrong side’ of the margin
boundary, but with a penalty that increases with the distance from that boundary. For
the subsequent optimization problem, it is convenient to make this penalty a linear
function of this distance. To do this, we introduce slack variables, ξn ⩾0 where
n = 1, . . . , N, with one slack variable for each training data point (Bennett, 1992;
Cortes and Vapnik, 1995). These are deﬁned by ξn = 0 for data points that are on or
inside the correct margin boundary and ξn = |tn −y(xn)| for other points. Thus a
data point that is on the decision boundary y(xn) = 0 will have ξn = 1, and points


---
**Page 348**
332
7. SPARSE KERNEL MACHINES
Figure 7.3
Illustration of the slack variables ξn ⩾0.
Data points with circles around them are
support vectors.
y = 1
y = 0
y = −1
ξ > 1
ξ < 1
ξ = 0
ξ = 0
with ξn > 1 will be misclassiﬁed. The exact classiﬁcation constraints (7.5) are then
replaced with
tny(xn) ⩾1 −ξn,
n = 1, . . . , N
(7.20)
in which the slack variables are constrained to satisfy ξn ⩾0. Data points for which
ξn = 0 are correctly classiﬁed and are either on the margin or on the correct side
of the margin. Points for which 0 < ξn ⩽1 lie inside the margin, but on the cor-
rect side of the decision boundary, and those data points for which ξn > 1 lie on
the wrong side of the decision boundary and are misclassiﬁed, as illustrated in Fig-
ure 7.3. This is sometimes described as relaxing the hard margin constraint to give a
soft margin and allows some of the training set data points to be misclassiﬁed. Note
that while slack variables allow for overlapping class distributions, this framework is
still sensitive to outliers because the penalty for misclassiﬁcation increases linearly
with ξ.
Our goal is now to maximize the margin while softly penalizing points that lie
on the wrong side of the margin boundary. We therefore minimize
C
N

n=1
ξn + 1
2∥w∥2
(7.21)
where the parameter C > 0 controls the trade-off between the slack variable penalty
and the margin. Because any point that is misclassiﬁed has ξn > 1, it follows that

n ξn is an upper bound on the number of misclassiﬁed points. The parameter C is
therefore analogous to (the inverse of) a regularization coefﬁcient because it controls
the trade-off between minimizing training errors and controlling model complexity.
In the limit C →∞, we will recover the earlier support vector machine for separable
data.
We now wish to minimize (7.21) subject to the constraints (7.20) together with
ξn ⩾0. The corresponding Lagrangian is given by
L(w, b, a) = 1
2∥w∥2 +C
N

n=1
ξn −
N

n=1
an {tny(xn) −1 + ξn}−
N

n=1
µnξn (7.22)


---
**Page 349**
7.1. Maximum Margin Classiﬁers
333
where {an ⩾0} and {µn ⩾0} are Lagrange multipliers. The corresponding set of
KKT conditions are given by
Appendix E
an
⩾
0
(7.23)
tny(xn) −1 + ξn
⩾
0
(7.24)
an (tny(xn) −1 + ξn)
=
0
(7.25)
µn
⩾
0
(7.26)
ξn
⩾
0
(7.27)
µnξn
=
0
(7.28)
where n = 1, . . . , N.
We now optimize out w, b, and {ξn} making use of the deﬁnition (7.1) of y(x)
to give
∂L
∂w = 0
⇒
w =
N

n=1
antnφ(xn)
(7.29)
∂L
∂b = 0
⇒
N

n=1
antn = 0
(7.30)
∂L
∂ξn
= 0
⇒
an = C −µn.
(7.31)
Using these results to eliminate w, b, and {ξn} from the Lagrangian, we obtain the
dual Lagrangian in the form
L(a) =
N

n=1
an −1
2
N

n=1
N

m=1
anamtntmk(xn, xm)
(7.32)
which is identical to the separable case, except that the constraints are somewhat
different. To see what these constraints are, we note that an ⩾0 is required because
these are Lagrange multipliers. Furthermore, (7.31) together with µn ⩾0 implies
an ⩽C. We therefore have to minimize (7.32) with respect to the dual variables
{an} subject to
0 ⩽an ⩽C
(7.33)
N

n=1
antn = 0
(7.34)
for n = 1, . . . , N, where (7.33) are known as box constraints. This again represents
a quadratic programming problem. If we substitute (7.29) into (7.1), we see that
predictions for new data points are again made by using (7.13).
We can now interpret the resulting solution. As before, a subset of the data
points may have an = 0, in which case they do not contribute to the predictive


---
**Page 350**
334
7. SPARSE KERNEL MACHINES
model (7.13). The remaining data points constitute the support vectors. These have
an > 0 and hence from (7.25) must satisfy
tny(xn) = 1 −ξn.
(7.35)
If an < C, then (7.31) implies that µn > 0, which from (7.28) requires ξn = 0 and
hence such points lie on the margin. Points with an = C can lie inside the margin
and can either be correctly classiﬁed if ξn ⩽1 or misclassiﬁed if ξn > 1.
To determine the parameter b in (7.1), we note that those support vectors for
which 0 < an < C have ξn = 0 so that tny(xn) = 1 and hence will satisfy
tn

m∈S
amtmk(xn, xm) + b

= 1.
(7.36)
Again, a numerically stable solution is obtained by averaging to give
b =
1
NM

n∈M

tn −

m∈S
amtmk(xn, xm)

(7.37)
where M denotes the set of indices of data points having 0 < an < C.
An alternative, equivalent formulation of the support vector machine, known as
the ν-SVM, has been proposed by Sch¨olkopf et al. (2000). This involves maximizing
L(a) = −1
2
N

n=1
N

m=1
anamtntmk(xn, xm)
(7.38)
subject to the constraints
0 ⩽an ⩽1/N
(7.39)
N

n=1
antn = 0
(7.40)
N

n=1
an ⩾ν.
(7.41)
This approach has the advantage that the parameter ν, which replaces C, can be
interpreted as both an upper bound on the fraction of margin errors (points for which
ξn > 0 and hence which lie on the wrong side of the margin boundary and which may
or may not be misclassiﬁed) and a lower bound on the fraction of support vectors. An
example of the ν-SVM applied to a synthetic data set is shown in Figure 7.4. Here
Gaussian kernels of the form exp (−γ∥x −x′∥2) have been used, with γ = 0.45.
Although predictions for new inputs are made using only the support vectors,
the training phase (i.e., the determination of the parameters a and b) makes use of
the whole data set, and so it is important to have efﬁcient algorithms for solving


---
**Page 351**
7.1. Maximum Margin Classiﬁers
335
Figure 7.4
Illustration of the ν-SVM applied
to a nonseparable data set in two
dimensions. The support vectors
are indicated by circles.
−2
0
2
−2
0
2
the quadratic programming problem. We ﬁrst note that the objective function L(a)
given by (7.10) or (7.32) is quadratic and so any local optimum will also be a global
optimum provided the constraints deﬁne a convex region (which they do as a conse-
quence of being linear). Direct solution of the quadratic programming problem us-
ing traditional techniques is often infeasible due to the demanding computation and
memory requirements, and so more practical approaches need to be found. The tech-
nique of chunking (Vapnik, 1982) exploits the fact that the value of the Lagrangian
is unchanged if we remove the rows and columns of the kernel matrix corresponding
to Lagrange multipliers that have value zero. This allows the full quadratic pro-
gramming problem to be broken down into a series of smaller ones, whose goal is
eventually to identify all of the nonzero Lagrange multipliers and discard the others.
Chunking can be implemented using protected conjugate gradients (Burges, 1998).
Although chunking reduces the size of the matrix in the quadratic function from the
number of data points squared to approximately the number of nonzero Lagrange
multipliers squared, even this may be too big to ﬁt in memory for large-scale appli-
cations. Decomposition methods (Osuna et al., 1996) also solve a series of smaller
quadratic programming problems but are designed so that each of these is of a ﬁxed
size, and so the technique can be applied to arbitrarily large data sets. However, it
still involves numerical solution of quadratic programming subproblems and these
can be problematic and expensive. One of the most popular approaches to training
support vector machines is called sequential minimal optimization, or SMO (Platt,
1999). It takes the concept of chunking to the extreme limit and considers just two
Lagrange multipliers at a time. In this case, the subproblem can be solved analyti-
cally, thereby avoiding numerical quadratic programming altogether. Heuristics are
given for choosing the pair of Lagrange multipliers to be considered at each step.
In practice, SMO is found to have a scaling with the number of data points that is
somewhere between linear and quadratic depending on the particular application.
We have seen that kernel functions correspond to inner products in feature spaces
that can have high, or even inﬁnite, dimensionality. By working directly in terms of
the kernel function, without introducing the feature space explicitly, it might there-
fore seem that support vector machines somehow manage to avoid the curse of di-


---
**Page 352**
336
7. SPARSE KERNEL MACHINES
mensionality. This is not the case, however, because there are constraints amongst
Section 1.4
the feature values that restrict the effective dimensionality of feature space. To see
this consider a simple second-order polynomial kernel that we can expand in terms
of its components
k(x, z)
=

1 + xTz2 = (1 + x1z1 + x2z2)2
=
1 + 2x1z1 + 2x2z2 + x2
1z2
1 + 2x1z1x2z2 + x2
2z2
2
=
(1,
√
2x1,
√
2x2, x2
1,
√
2x1x2, x2
2)(1,
√
2z1,
√
2z2, z2
1,
√
2z1z2, z2
2)T
=
φ(x)Tφ(z).
(7.42)
This kernel function therefore represents an inner product in a feature space having
six dimensions, in which the mapping from input space to feature space is described
by the vector function φ(x). However, the coefﬁcients weighting these different
features are constrained to have speciﬁc forms. Thus any set of points in the original
two-dimensional space x would be constrained to lie exactly on a two-dimensional
nonlinear manifold embedded in the six-dimensional feature space.
We have already highlighted the fact that the support vector machine does not
provide probabilistic outputs but instead makes classiﬁcation decisions for new in-
put vectors. Veropoulos et al. (1999) discuss modiﬁcations to the SVM to allow
the trade-off between false positive and false negative errors to be controlled. How-
ever, if we wish to use the SVM as a module in a larger probabilistic system, then
probabilistic predictions of the class label t for new inputs x are required.
To address this issue, Platt (2000) has proposed ﬁtting a logistic sigmoid to the
outputs of a previously trained support vector machine. Speciﬁcally, the required
conditional probability is assumed to be of the form
p(t = 1|x) = σ (Ay(x) + B)
(7.43)
where y(x) is deﬁned by (7.1). Values for the parameters A and B are found by
minimizing the cross-entropy error function deﬁned by a training set consisting of
pairs of values y(xn) and tn. The data used to ﬁt the sigmoid needs to be independent
of that used to train the original SVM in order to avoid severe over-ﬁtting. This two-
stage approach is equivalent to assuming that the output y(x) of the support vector
machine represents the log-odds of x belonging to class t = 1. Because the SVM
training procedure is not speciﬁcally intended to encourage this, the SVM can give
a poor approximation to the posterior probabilities (Tipping, 2001).
7.1.2
Relation to logistic regression
As with the separable case, we can re-cast the SVM for nonseparable distri-
butions in terms of the minimization of a regularized error function. This will also
allow us to highlight similarities, and differences, compared to the logistic regression
model.
Section 4.3.2
We have seen that for data points that are on the correct side of the margin
boundary, and which therefore satisfy yntn ⩾1, we have ξn = 0, and for the


---
**Page 353**
7.1. Maximum Margin Classiﬁers
337
Figure 7.5
Plot of the ‘hinge’ error function used
in support vector machines, shown
in blue, along with the error function
for logistic regression, rescaled by a
factor of 1/ ln(2) so that it passes
through the point (0, 1), shown in red.
Also shown are the misclassiﬁcation
error in black and the squared error
in green.
−2
−1
0
1
2
z
E(z)
remaining points we have ξn = 1 −yntn. Thus the objective function (7.21) can be
written (up to an overall multiplicative constant) in the form
N

n=1
ESV(yntn) + λ∥w∥2
(7.44)
where λ = (2C)−1, and ESV(·) is the hinge error function deﬁned by
ESV(yntn) = [1 −yntn]+
(7.45)
where [ · ]+ denotes the positive part. The hinge error function, so-called because
of its shape, is plotted in Figure 7.5. It can be viewed as an approximation to the
misclassiﬁcation error, i.e., the error function that ideally we would like to minimize,
which is also shown in Figure 7.5.
When we considered the logistic regression model in Section 4.3.2, we found it
convenient to work with target variable t ∈{0, 1}. For comparison with the support
vector machine, we ﬁrst reformulate maximum likelihood logistic regression using
the target variable t ∈{−1, 1}. To do this, we note that p(t = 1|y) = σ(y) where
y(x) is given by (7.1), and σ(y) is the logistic sigmoid function deﬁned by (4.59). It
follows that p(t = −1|y) = 1 −σ(y) = σ(−y), where we have used the properties
of the logistic sigmoid function, and so we can write
p(t|y) = σ(yt).
(7.46)
From this we can construct an error function by taking the negative logarithm of the
likelihood function that, with a quadratic regularizer, takes the form
Exercise 7.6
N

n=1
ELR(yntn) + λ∥w∥2.
(7.47)
where
ELR(yt) = ln (1 + exp(−yt)) .
(7.48)


---
**Page 354**
338
7. SPARSE KERNEL MACHINES
For comparison with other error functions, we can divide by ln(2) so that the error
function passes through the point (0, 1). This rescaled error function is also plotted
in Figure 7.5 and we see that it has a similar form to the support vector error function.
The key difference is that the ﬂat region in ESV(yt) leads to sparse solutions.
Both the logistic error and the hinge loss can be viewed as continuous approx-
imations to the misclassiﬁcation error. Another continuous error function that has
sometimes been used to solve classiﬁcation problems is the squared error, which
is again plotted in Figure 7.5. It has the property, however, of placing increasing
emphasis on data points that are correctly classiﬁed but that are a long way from
the decision boundary on the correct side. Such points will be strongly weighted at
the expense of misclassiﬁed points, and so if the objective is to minimize the mis-
classiﬁcation rate, then a monotonically decreasing error function would be a better
choice.
7.1.3
Multiclass SVMs
The support vector machine is fundamentally a two-class classiﬁer. In practice,
however, we often have to tackle problems involving K > 2 classes. Various meth-
ods have therefore been proposed for combining multiple two-class SVMs in order
to build a multiclass classiﬁer.
One commonly used approach (Vapnik, 1998) is to construct K separate SVMs,
in which the kth model yk(x) is trained using the data from class Ck as the positive
examples and the data from the remaining K −1 classes as the negative examples.
This is known as the one-versus-the-rest approach. However, in Figure 4.2 we saw
that using the decisions of the individual classiﬁers can lead to inconsistent results
in which an input is assigned to multiple classes simultaneously. This problem is
sometimes addressed by making predictions for new inputs x using
y(x) = max
k
yk(x).
(7.49)
Unfortunately, this heuristic approach suffers from the problem that the different
classiﬁers were trained on different tasks, and there is no guarantee that the real-
valued quantities yk(x) for different classiﬁers will have appropriate scales.
Another problem with the one-versus-the-rest approach is that the training sets
are imbalanced. For instance, if we have ten classes each with equal numbers of
training data points, then the individual classiﬁers are trained on data sets comprising
90% negative examples and only 10% positive examples, and the symmetry of the
original problem is lost. A variant of the one-versus-the-rest scheme was proposed
by Lee et al. (2001) who modify the target values so that the positive class has target
+1 and the negative class has target −1/(K −1).
Weston and Watkins (1999) deﬁne a single objective function for training all
K SVMs simultaneously, based on maximizing the margin from each to remaining
classes. However, this can result in much slower training because, instead of solving
K separate optimization problems each over N data points with an overall cost of
O(KN 2), a single optimization problem of size (K −1)N must be solved giving an
overall cost of O(K2N 2).


---
**Page 355**
7.1. Maximum Margin Classiﬁers
339
Another approach is to train K(K −1)/2 different 2-class SVMs on all possible
pairs of classes, and then to classify test points according to which class has the high-
est number of ‘votes’, an approach that is sometimes called one-versus-one. Again,
we saw in Figure 4.2 that this can lead to ambiguities in the resulting classiﬁcation.
Also, for large K this approach requires signiﬁcantly more training time than the
one-versus-the-rest approach. Similarly, to evaluate test points, signiﬁcantly more
computation is required.
The latter problem can be alleviated by organizing the pairwise classiﬁers into
a directed acyclic graph (not to be confused with a probabilistic graphical model)
leading to the DAGSVM (Platt et al., 2000). For K classes, the DAGSVM has a total
of K(K −1)/2 classiﬁers, and to classify a new test point only K −1 pairwise
classiﬁers need to be evaluated, with the particular classiﬁers used depending on
which path through the graph is traversed.
A different approach to multiclass classiﬁcation, based on error-correcting out-
put codes, was developed by Dietterich and Bakiri (1995) and applied to support
vector machines by Allwein et al. (2000). This can be viewed as a generalization of
the voting scheme of the one-versus-one approach in which more general partitions
of the classes are used to train the individual classiﬁers. The K classes themselves
are represented as particular sets of responses from the two-class classiﬁers chosen,
and together with a suitable decoding scheme, this gives robustness to errors and to
ambiguity in the outputs of the individual classiﬁers. Although the application of
SVMs to multiclass classiﬁcation problems remains an open issue, in practice the
one-versus-the-rest approach is the most widely used in spite of its ad-hoc formula-
tion and its practical limitations.
There are also single-class support vector machines, which solve an unsuper-
vised learning problem related to probability density estimation. Instead of mod-
elling the density of data, however, these methods aim to ﬁnd a smooth boundary
enclosing a region of high density. The boundary is chosen to represent a quantile of
the density, that is, the probability that a data point drawn from the distribution will
land inside that region is given by a ﬁxed number between 0 and 1 that is speciﬁed in
advance. This is a more restricted problem than estimating the full density but may
be sufﬁcient in speciﬁc applications. Two approaches to this problem using support
vector machines have been proposed. The algorithm of Sch¨olkopf et al. (2001) tries
to ﬁnd a hyperplane that separates all but a ﬁxed fraction ν of the training data from
the origin while at the same time maximizing the distance (margin) of the hyperplane
from the origin, while Tax and Duin (1999) look for the smallest sphere in feature
space that contains all but a fraction ν of the data points. For kernels k(x, x′) that
are functions only of x −x′, the two algorithms are equivalent.
7.1.4
SVMs for regression
We now extend support vector machines to regression problems while at the
same time preserving the property of sparseness. In simple linear regression, we
Section 3.1.4


---
**Page 356**
340
7. SPARSE KERNEL MACHINES
Figure 7.6
Plot of an ϵ-insensitive error function (in
red) in which the error increases lin-
early with distance beyond the insen-
sitive region. Also shown for compar-
ison is the quadratic error function (in
green).
0
z
E(z)
−ϵ
ϵ
minimize a regularized error function given by
1
2
N

n=1
{yn −tn}2 + λ
2 ∥w∥2.
(7.50)
To obtain sparse solutions, the quadratic error function is replaced by an ϵ-insensitive
error function (Vapnik, 1995), which gives zero error if the absolute difference be-
tween the prediction y(x) and the target t is less than ϵ where ϵ > 0. A simple
example of an ϵ-insensitive error function, having a linear cost associated with errors
outside the insensitive region, is given by
Eϵ(y(x) −t) =

0,
if |y(x) −t| < ϵ;
|y(x) −t| −ϵ,
otherwise
(7.51)
and is illustrated in Figure 7.6.
We therefore minimize a regularized error function given by
C
N

n=1
Eϵ(y(xn) −tn) + 1
2∥w∥2
(7.52)
where y(x) is given by (7.1). By convention the (inverse) regularization parameter,
denoted C, appears in front of the error term.
As before, we can re-express the optimization problem by introducing slack
variables. For each data point xn, we now need two slack variables ξn ⩾0 and
ξn ⩾0, where ξn > 0 corresponds to a point for which tn > y(xn) + ϵ, and ξn > 0
corresponds to a point for which tn < y(xn) −ϵ, as illustrated in Figure 7.7.
The condition for a target point to lie inside the ϵ-tube is that yn −ϵ ⩽tn ⩽
yn+ϵ, where yn = y(xn). Introducing the slack variables allows points to lie outside
the tube provided the slack variables are nonzero, and the corresponding conditions
are
tn
⩽
y(xn) + ϵ + ξn
(7.53)
tn
⩾
y(xn) −ϵ −ξn.
(7.54)


---
**Page 357**
7.1. Maximum Margin Classiﬁers
341
Figure 7.7
Illustration of SVM regression, showing
the regression curve together with the ϵ-
insensitive ‘tube’.
Also shown are exam-
ples of the slack variables ξ and bξ. Points
above the ϵ-tube have ξ > 0 and bξ = 0,
points below the ϵ-tube have ξ = 0 and
bξ > 0, and points inside the ϵ-tube have
ξ = bξ = 0.
y
y + ϵ
y −ϵ
y(x)
x
ξ > 0
ξ > 0
The error function for support vector regression can then be written as
C
N

n=1
(ξn + ξn) + 1
2∥w∥2
(7.55)
which must be minimized subject to the constraints ξn ⩾0 and ξn ⩾0 as well as
(7.53) and (7.54). This can be achieved by introducing Lagrange multipliers an ⩾0,
an ⩾0, µn ⩾0, and µn ⩾0 and optimizing the Lagrangian
L
=
C
N

n=1
(ξn + ξn) + 1
2∥w∥2 −
N

n=1
(µnξn + µnξn)
−
N

n=1
an(ϵ + ξn + yn −tn) −
N

n=1
an(ϵ + ξn −yn + tn).
(7.56)
We now substitute for y(x) using (7.1) and then set the derivatives of the La-
grangian with respect to w, b, ξn, and ξn to zero, giving
∂L
∂w = 0
⇒
w =
N

n=1
(an −an)φ(xn)
(7.57)
∂L
∂b = 0
⇒
N

n=1
(an −an) = 0
(7.58)
∂L
∂ξn
= 0
⇒
an + µn = C
(7.59)
∂L
∂ξn
= 0
⇒
an + µn = C.
(7.60)
Using these results to eliminate the corresponding variables from the Lagrangian, we
see that the dual problem involves maximizing
Exercise 7.7


---
**Page 358**
342
7. SPARSE KERNEL MACHINES
L(a, a)
=
−1
2
N

n=1
N

m=1
(an −an)(am −am)k(xn, xm)
−ϵ
N

n=1
(an + an) +
N

n=1
(an −an)tn
(7.61)
with respect to {an} and {an}, where we have introduced the kernel k(x, x′) =
φ(x)Tφ(x′). Again, this is a constrained maximization, and to ﬁnd the constraints
we note that an ⩾0 and an ⩾0 are both required because these are Lagrange
multipliers. Also µn ⩾0 and µn ⩾0 together with (7.59) and (7.60), require
an ⩽C and an ⩽C, and so again we have the box constraints
0 ⩽an ⩽C
(7.62)
0 ⩽an ⩽C
(7.63)
together with the condition (7.58).
Substituting (7.57) into (7.1), we see that predictions for new inputs can be made
using
y(x) =
N

n=1
(an −an)k(x, xn) + b
(7.64)
which is again expressed in terms of the kernel function.
The corresponding Karush-Kuhn-Tucker (KKT) conditions, which state that at
the solution the product of the dual variables and the constraints must vanish, are
given by
an(ϵ + ξn + yn −tn)
=
0
(7.65)
an(ϵ + ξn −yn + tn)
=
0
(7.66)
(C −an)ξn
=
0
(7.67)
(C −an)ξn
=
0.
(7.68)
From these we can obtain several useful results. First of all, we note that a coefﬁcient
an can only be nonzero if ϵ + ξn + yn −tn = 0, which implies that the data point
either lies on the upper boundary of the ϵ-tube (ξn = 0) or lies above the upper
boundary (ξn > 0). Similarly, a nonzero value for an implies ϵ +ξn −yn + tn = 0,
and such points must lie either on or below the lower boundary of the ϵ-tube.
Furthermore, the two constraints ϵ+ξn +yn −tn = 0 and ϵ+ξn −yn +tn = 0
are incompatible, as is easily seen by adding them together and noting that ξn and
ξn are nonnegative while ϵ is strictly positive, and so for every data point xn, either
an or an (or both) must be zero.
The support vectors are those data points that contribute to predictions given by
(7.64), in other words those for which either an ̸= 0 or an ̸= 0. These are points that
lie on the boundary of the ϵ-tube or outside the tube. All points within the tube have

