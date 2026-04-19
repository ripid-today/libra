# 08 - Graphical Models
*Pages 359-422 from Pattern Recognition and Machine Learning*

---
**Page 359**
7.1. Maximum Margin Classiﬁers
343
an = an = 0. We again have a sparse solution, and the only terms that have to be
evaluated in the predictive model (7.64) are those that involve the support vectors.
The parameter b can be found by considering a data point for which 0 < an <
C, which from (7.67) must have ξn = 0, and from (7.65) must therefore satisfy
ϵ + yn −tn = 0. Using (7.1) and solving for b, we obtain
b
=
tn −ϵ −wTφ(xn)
=
tn −ϵ −
N

m=1
(am −am)k(xn, xm)
(7.69)
where we have used (7.57). We can obtain an analogous result by considering a point
for which 0 < an < C. In practice, it is better to average over all such estimates of
b.
As with the classiﬁcation case, there is an alternative formulation of the SVM
for regression in which the parameter governing complexity has a more intuitive
interpretation (Sch¨olkopf et al., 2000). In particular, instead of ﬁxing the width ϵ of
the insensitive region, we ﬁx instead a parameter ν that bounds the fraction of points
lying outside the tube. This involves maximizing
L(a,a)
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
+
N

n=1
(an −an)tn
(7.70)
subject to the constraints
0 ⩽an ⩽C/N
(7.71)
0 ⩽an ⩽C/N
(7.72)
N

n=1
(an −an) = 0
(7.73)
N

n=1
(an + an) ⩽νC.
(7.74)
It can be shown that there are at most νN data points falling outside the insensitive
tube, while at least νN data points are support vectors and so lie either on the tube
or outside it.
The use of a support vector machine to solve a regression problem is illustrated
using the sinusoidal data set in Figure 7.8. Here the parameters ν and C have been
Appendix A
chosen by hand. In practice, their values would typically be determined by cross-
validation.


---
**Page 360**
344
7. SPARSE KERNEL MACHINES
Figure 7.8
Illustration of the ν-SVM for re-
gression applied to the sinusoidal
synthetic data set using Gaussian
kernels. The predicted regression
curve is shown by the red line, and
the ϵ-insensitive tube corresponds
to the shaded region.
Also, the
data points are shown in green,
and those with support vectors
are indicated by blue circles.
x
t
0
1
−1
0
1
7.1.5
Computational learning theory
Historically, support vector machines have largely been motivated and analysed
using a theoretical framework known as computational learning theory, also some-
times called statistical learning theory (Anthony and Biggs, 1992; Kearns and Vazi-
rani, 1994; Vapnik, 1995; Vapnik, 1998). This has its origins with Valiant (1984)
who formulated the probably approximately correct, or PAC, learning framework.
The goal of the PAC framework is to understand how large a data set needs to be in
order to give good generalization. It also gives bounds for the computational cost of
learning, although we do not consider these here.
Suppose that a data set D of size N is drawn from some joint distribution p(x, t)
where x is the input variable and t represents the class label, and that we restrict
attention to ‘noise free’ situations in which the class labels are determined by some
(unknown) deterministic function t = g(x). In PAC learning we say that a function
f(x; D), drawn from a space F of such functions on the basis of the training set
D, has good generalization if its expected error rate is below some pre-speciﬁed
threshold ϵ, so that
Ex,t [I (f(x; D) ̸= t)] < ϵ
(7.75)
where I(·) is the indicator function, and the expectation is with respect to the dis-
tribution p(x, t). The quantity on the left-hand side is a random variable, because
it depends on the training set D, and the PAC framework requires that (7.75) holds,
with probability greater than 1 −δ, for a data set D drawn randomly from p(x, t).
Here δ is another pre-speciﬁed parameter, and the terminology ‘probably approxi-
mately correct’ comes from the requirement that with high probability (greater than
1−δ), the error rate be small (less than ϵ). For a given choice of model space F, and
for given parameters ϵ and δ, PAC learning aims to provide bounds on the minimum
size N of data set needed to meet this criterion. A key quantity in PAC learning is
the Vapnik-Chervonenkis dimension, or VC dimension, which provides a measure of
the complexity of a space of functions, and which allows the PAC framework to be
extended to spaces containing an inﬁnite number of functions.
The bounds derived within the PAC framework are often described as worst-


---
**Page 361**
7.2. Relevance Vector Machines
345
case, because they apply to any choice for the distribution p(x, t), so long as both
the training and the test examples are drawn (independently) from the same distribu-
tion, and for any choice for the function f(x) so long as it belongs to F. In real-world
applications of machine learning, we deal with distributions that have signiﬁcant reg-
ularity, for example in which large regions of input space carry the same class label.
As a consequence of the lack of any assumptions about the form of the distribution,
the PAC bounds are very conservative, in other words they strongly over-estimate
the size of data sets required to achieve a given generalization performance. For this
reason, PAC bounds have found few, if any, practical applications.
One attempt to improve the tightness of the PAC bounds is the PAC-Bayesian
framework (McAllester, 2003), which considers a distribution over the space F of
functions, somewhat analogous to the prior in a Bayesian treatment. This still con-
siders any possible choice for p(x, t), and so although the bounds are tighter, they
are still very conservative.
7.2. Relevance Vector Machines
Support vector machines have been used in a variety of classiﬁcation and regres-
sion applications. Nevertheless, they suffer from a number of limitations, several
of which have been highlighted already in this chapter. In particular, the outputs of
an SVM represent decisions rather than posterior probabilities. Also, the SVM was
originally formulated for two classes, and the extension to K > 2 classes is prob-
lematic. There is a complexity parameter C, or ν (as well as a parameter ϵ in the case
of regression), that must be found using a hold-out method such as cross-validation.
Finally, predictions are expressed as linear combinations of kernel functions that are
centred on training data points and that are required to be positive deﬁnite.
The relevance vector machine or RVM (Tipping, 2001) is a Bayesian sparse ker-
nel technique for regression and classiﬁcation that shares many of the characteristics
of the SVM whilst avoiding its principal limitations. Additionally, it typically leads
to much sparser models resulting in correspondingly faster performance on test data
whilst maintaining comparable generalization error.
In contrast to the SVM we shall ﬁnd it more convenient to introduce the regres-
sion form of the RVM ﬁrst and then consider the extension to classiﬁcation tasks.
7.2.1
RVM for regression
The relevance vector machine for regression is a linear model of the form studied
in Chapter 3 but with a modiﬁed prior that results in sparse solutions. The model
deﬁnes a conditional distribution for a real-valued target variable t, given an input
vector x, which takes the form
p(t|x, w, β) = N(t|y(x), β−1)
(7.76)


---
**Page 362**
346
7. SPARSE KERNEL MACHINES
where β = σ−2 is the noise precision (inverse noise variance), and the mean is given
by a linear model of the form
y(x) =
M

i=1
wiφi(x) = wTφ(x)
(7.77)
with ﬁxed nonlinear basis functions φi(x), which will typically include a constant
term so that the corresponding weight parameter represents a ‘bias’.
The relevance vector machine is a speciﬁc instance of this model, which is in-
tended to mirror the structure of the support vector machine. In particular, the basis
functions are given by kernels, with one kernel associated with each of the data
points from the training set. The general expression (7.77) then takes the SVM-like
form
y(x) =
N

n=1
wnk(x, xn) + b
(7.78)
where b is a bias parameter. The number of parameters in this case is M = N + 1,
and y(x) has the same form as the predictive model (7.64) for the SVM, except that
the coefﬁcients an are here denoted wn. It should be emphasized that the subsequent
analysis is valid for arbitrary choices of basis function, and for generality we shall
work with the form (7.77). In contrast to the SVM, there is no restriction to positive-
deﬁnite kernels, nor are the basis functions tied in either number or location to the
training data points.
Suppose we are given a set of N observations of the input vector x, which we
denote collectively by a data matrix X whose nth row is xT
n with n = 1, . . . , N. The
corresponding target values are given by t = (t1, . . . , tN)T. Thus, the likelihood
function is given by
p(t|X, w, β) =
N

n=1
p(tn|xn, w, β−1).
(7.79)
Next we introduce a prior distribution over the parameter vector w and as in
Chapter 3, we shall consider a zero-mean Gaussian prior. However, the key differ-
ence in the RVM is that we introduce a separate hyperparameter αi for each of the
weight parameters wi instead of a single shared hyperparameter. Thus the weight
prior takes the form
p(w|α) =
M

i=1
N(wi|0, α−1
i )
(7.80)
where αi represents the precision of the corresponding parameter wi, and α denotes
(α1, . . . , αM)T. We shall see that, when we maximize the evidence with respect
to these hyperparameters, a signiﬁcant proportion of them go to inﬁnity, and the
corresponding weight parameters have posterior distributions that are concentrated
at zero. The basis functions associated with these parameters therefore play no role


---
**Page 363**
7.2. Relevance Vector Machines
347
in the predictions made by the model and so are effectively pruned out, resulting in
a sparse model.
Using the result (3.49) for linear regression models, we see that the posterior
distribution for the weights is again Gaussian and takes the form
p(w|t, X, α, β) = N(w|m, Σ)
(7.81)
where the mean and covariance are given by
m
=
βΣΦTt
(7.82)
Σ
=

A + βΦTΦ−1
(7.83)
where Φ is the N × M design matrix with elements Φni = φi(xn), and A =
diag(αi). Note that in the speciﬁc case of the model (7.78), we have Φ = K, where
K is the symmetric (N + 1) × (N + 1) kernel matrix with elements k(xn, xm).
The values of α and β are determined using type-2 maximum likelihood, also
known as the evidence approximation, in which we maximize the marginal likeli-
Section 3.5
hood function obtained by integrating out the weight parameters
p(t|X, α, β) =

p(t|X, w, β)p(w|α) dw.
(7.84)
Because this represents the convolution of two Gaussians, it is readily evaluated to
Exercise 7.10
give the log marginal likelihood in the form
ln p(t|X, α, β)
=
ln N(t|0, C)
=
−1
2

N ln(2π) + ln |C| + tTC−1t
(7.85)
where t = (t1, . . . , tN)T, and we have deﬁned the N × N matrix C given by
C = β−1I + ΦA−1ΦT.
(7.86)
Our goal is now to maximize (7.85) with respect to the hyperparameters α and
β. This requires only a small modiﬁcation to the results obtained in Section 3.5 for
the evidence approximation in the linear regression model. Again, we can identify
two approaches. In the ﬁrst, we simply set the required derivatives of the marginal
likelihood to zero and obtain the following re-estimation equations
Exercise 7.12
αnew
i
=
γi
m2
i
(7.87)
(βnew)−1
=
∥t −Φm∥2
N −
i γi
(7.88)
where mi is the ith component of the posterior mean m deﬁned by (7.82). The
quantity γi measures how well the corresponding parameter wi is determined by the
data and is deﬁned by
Section 3.5.3


---
**Page 364**
348
7. SPARSE KERNEL MACHINES
γi = 1 −αiΣii
(7.89)
in which Σii is the ith diagonal component of the posterior covariance Σ given by
(7.83). Learning therefore proceeds by choosing initial values for α and β, evalu-
ating the mean and covariance of the posterior using (7.82) and (7.83), respectively,
and then alternately re-estimating the hyperparameters, using (7.87) and (7.88), and
re-estimating the posterior mean and covariance, using (7.82) and (7.83), until a suit-
able convergence criterion is satisﬁed.
The second approach is to use the EM algorithm, and is discussed in Sec-
tion 9.3.4. These two approaches to ﬁnding the values of the hyperparameters that
maximize the evidence are formally equivalent. Numerically, however, it is found
Exercise 9.23
that the direct optimization approach corresponding to (7.87) and (7.88) gives some-
what faster convergence (Tipping, 2001).
As a result of the optimization, we ﬁnd that a proportion of the hyperparameters
{αi} are driven to large (in principle inﬁnite) values, and so the weight parameters
Section 7.2.2
wi corresponding to these hyperparameters have posterior distributions with mean
and variance both zero. Thus those parameters, and the corresponding basis func-
tions φi(x), are removed from the model and play no role in making predictions for
new inputs. In the case of models of the form (7.78), the inputs xn corresponding to
the remaining nonzero weights are called relevance vectors, because they are iden-
tiﬁed through the mechanism of automatic relevance determination, and are analo-
gous to the support vectors of an SVM. It is worth emphasizing, however, that this
mechanism for achieving sparsity in probabilistic models through automatic rele-
vance determination is quite general and can be applied to any model expressed as
an adaptive linear combination of basis functions.
Having found values α⋆and β⋆for the hyperparameters that maximize the
marginal likelihood, we can evaluate the predictive distribution over t for a new
input x. Using (7.76) and (7.81), this is given by
Exercise 7.14
p(t|x, X, t, α⋆, β⋆)
=

p(t|x, w, β⋆)p(w|X, t, α⋆, β⋆) dw
=
N

t|mTφ(x), σ2(x)

.
(7.90)
Thus the predictive mean is given by (7.76) with w set equal to the posterior mean
m, and the variance of the predictive distribution is given by
σ2(x) = (β⋆)−1 + φ(x)TΣφ(x)
(7.91)
where Σ is given by (7.83) in which α and β are set to their optimized values α⋆and
β⋆. This is just the familiar result (3.59) obtained in the context of linear regression.
Recall that for localized basis functions, the predictive variance for linear regression
models becomes small in regions of input space where there are no basis functions.
In the case of an RVM with the basis functions centred on data points, the model will
therefore become increasingly certain of its predictions when extrapolating outside
the domain of the data (Rasmussen and Qui˜nonero-Candela, 2005), which of course
is undesirable. The predictive distribution in Gaussian process regression does not
Section 6.4.2


---
**Page 365**
7.2. Relevance Vector Machines
349
Figure 7.9
Illustration of RVM regression us-
ing the same data set, and the
same Gaussian kernel functions,
as used in Figure 7.8 for the
ν-SVM regression model.
The
mean of the predictive distribu-
tion for the RVM is shown by the
red line, and the one standard-
deviation predictive distribution is
shown
by
the
shaded
region.
Also, the data points are shown
in green, and the relevance vec-
tors are indicated by blue circles.
Note that there are only 3 rele-
vance vectors compared to 7 sup-
port vectors for the ν-SVM in Fig-
ure 7.8.
x
t
0
1
−1
0
1
suffer from this problem. However, the computational cost of making predictions
with a Gaussian processes is typically much higher than with an RVM.
Figure 7.9 shows an example of the RVM applied to the sinusoidal regression
data set. Here the noise precision parameter β is also determined through evidence
maximization. We see that the number of relevance vectors in the RVM is signif-
icantly smaller than the number of support vectors used by the SVM. For a wide
range of regression and classiﬁcation tasks, the RVM is found to give models that
are typically an order of magnitude more compact than the corresponding support
vector machine, resulting in a signiﬁcant improvement in the speed of processing on
test data. Remarkably, this greater sparsity is achieved with little or no reduction in
generalization error compared with the corresponding SVM.
The principal disadvantage of the RVM compared to the SVM is that training
involves optimizing a nonconvex function, and training times can be longer than for a
comparable SVM. For a model with M basis functions, the RVM requires inversion
of a matrix of size M × M, which in general requires O(M 3) computation. In the
speciﬁc case of the SVM-like model (7.78), we have M = N +1. As we have noted,
there are techniques for training SVMs whose cost is roughly quadratic in N. Of
course, in the case of the RVM we always have the option of starting with a smaller
number of basis functions than N + 1. More signiﬁcantly, in the relevance vector
machine the parameters governing complexity and noise variance are determined
automatically from a single training run, whereas in the support vector machine the
parameters C and ϵ (or ν) are generally found using cross-validation, which involves
multiple training runs. Furthermore, in the next section we shall derive an alternative
procedure for training the relevance vector machine that improves training speed
signiﬁcantly.
7.2.2
Analysis of sparsity
We have noted earlier that the mechanism of automatic relevance determination
causes a subset of parameters to be driven to zero. We now examine in more detail


---
**Page 366**
350
7. SPARSE KERNEL MACHINES
t1
t2
t
C
t1
t2
t
C
ϕ
Figure 7.10
Illustration of the mechanism for sparsity in a Bayesian linear regression model, showing a training
set vector of target values given by t = (t1, t2)T, indicated by the cross, for a model with one basis vector
ϕ = (φ(x1), φ(x2))T, which is poorly aligned with the target data vector t. On the left we see a model having
only isotropic noise, so that C = β−1I, corresponding to α = ∞, with β set to its most probable value. On
the right we see the same model but with a ﬁnite value of α. In each case the red ellipse corresponds to unit
Mahalanobis distance, with |C| taking the same value for both plots, while the dashed green circle shows the
contrition arising from the noise term β−1. We see that any ﬁnite value of α reduces the probability of the
observed data, and so for the most probable solution the basis vector is removed.
the mechanism of sparsity in the context of the relevance vector machine. In the
process, we will arrive at a signiﬁcantly faster procedure for optimizing the hyper-
parameters compared to the direct techniques given above.
Before proceeding with a mathematical analysis, we ﬁrst give some informal
insight into the origin of sparsity in Bayesian linear models. Consider a data set
comprising N = 2 observations t1 and t2, together with a model having a single
basis function φ(x), with hyperparameter α, along with isotropic noise having pre-
cision β. From (7.85), the marginal likelihood is given by p(t|α, β) = N(t|0, C) in
which the covariance matrix takes the form
C = 1
β I + 1
αϕϕT
(7.92)
where ϕ denotes the N-dimensional vector (φ(x1), φ(x2))T, and similarly t =
(t1, t2)T. Notice that this is just a zero-mean Gaussian process model over t with
covariance C. Given a particular observation for t, our goal is to ﬁnd α⋆and β⋆by
maximizing the marginal likelihood. We see from Figure 7.10 that, if there is a poor
alignment between the direction of ϕ and that of the training data vector t, then the
corresponding hyperparameter α will be driven to ∞, and the basis vector will be
pruned from the model. This arises because any ﬁnite value for α will always assign
a lower probability to the data, thereby decreasing the value of the density at t, pro-
vided that β is set to its optimal value. We see that any ﬁnite value for α would cause
the distribution to be elongated in a direction away from the data, thereby increasing
the probability mass in regions away from the observed data and hence reducing the
value of the density at the target data vector itself. For the more general case of M


---
**Page 367**
7.2. Relevance Vector Machines
351
basis vectors ϕ1, . . . , ϕM a similar intuition holds, namely that if a particular basis
vector is poorly aligned with the data vector t, then it is likely to be pruned from the
model.
We now investigate the mechanism for sparsity from a more mathematical per-
spective, for a general case involving M basis functions. To motivate this analysis
we ﬁrst note that, in the result (7.87) for re-estimating the parameter αi, the terms on
the right-hand side are themselves also functions of αi. These results therefore rep-
resent implicit solutions, and iteration would be required even to determine a single
αi with all other αj for j ̸= i ﬁxed.
This suggests a different approach to solving the optimization problem for the
RVM, in which we make explicit all of the dependence of the marginal likelihood
(7.85) on a particular αi and then determine its stationary points explicitly (Faul and
Tipping, 2002; Tipping and Faul, 2003). To do this, we ﬁrst pull out the contribution
from αi in the matrix C deﬁned by (7.86) to give
C
=
β−1I +

j̸=i
α−1
j ϕjϕT
j + α−1
i ϕiϕT
i
=
C−i + α−1
i ϕiϕT
i
(7.93)
where ϕi denotes the ith column of Φ, in other words the N-dimensional vector with
elements (φi(x1), . . . , φi(xN)), in contrast to φn, which denotes the nth row of Φ.
The matrix C−i represents the matrix C with the contribution from basis function i
removed. Using the matrix identities (C.7) and (C.15), the determinant and inverse
of C can then be written
|C|
=
|C−i||1 + α−1
i ϕT
i C−1
−iϕi|
(7.94)
C−1
=
C−1
−i −C−1
−iϕiϕT
i C−1
−i
αi + ϕT
i C−1
−iϕi
.
(7.95)
Using these results, we can then write the log marginal likelihood function (7.85) in
the form
Exercise 7.15
L(α) = L(α−i) + λ(αi)
(7.96)
where L(α−i) is simply the log marginal likelihood with basis function ϕi omitted,
and the quantity λ(αi) is deﬁned by
λ(αi) = 1
2

ln αi −ln (αi + si) +
q2
i
αi + si

(7.97)
and contains all of the dependence on αi. Here we have introduced the two quantities
si
=
ϕT
i C−1
−iϕi
(7.98)
qi
=
ϕT
i C−1
−it.
(7.99)
Here si is called the sparsity and qi is known as the quality of ϕi, and as we shall
see, a large value of si relative to the value of qi means that the basis function ϕi


---
**Page 368**
352
7. SPARSE KERNEL MACHINES
Figure 7.11
Plots
of
the
log
marginal
likelihood
λ(αi)
versus
ln αi showing on the left, the single
maximum at a ﬁnite αi for q2
i = 4
and si = 1 (so that q2
i > si) and on
the right, the maximum at αi = ∞
for q2
i
= 1 and si = 2 (so that
q2
i < si).
−5
0
5
−4
−2
0
2
−5
0
5
−4
−2
0
2
is more likely to be pruned from the model. The ‘sparsity’ measures the extent to
which basis function ϕi overlaps with the other basis vectors in the model, and the
‘quality’ represents a measure of the alignment of the basis vector ϕn with the error
between the training set values t = (t1, . . . , tN)T and the vector y−i of predictions
that would result from the model with the vector ϕi excluded (Tipping and Faul,
2003).
The stationary points of the marginal likelihood with respect to αi occur when
the derivative
dλ(αi)
dαi
= α−1
i s2
i −(q2
i −si)
2(αi + si)2
(7.100)
is equal to zero. There are two possible forms for the solution. Recalling that αi ⩾0,
we see that if q2
i < si, then αi →∞provides a solution. Conversely, if q2
i > si, we
can solve for αi to obtain
αi =
s2
i
q2
i −si
.
(7.101)
These two solutions are illustrated in Figure 7.11. We see that the relative size of
the quality and sparsity terms determines whether a particular basis vector will be
pruned from the model or not. A more complete analysis (Faul and Tipping, 2002),
based on the second derivatives of the marginal likelihood, conﬁrms these solutions
are indeed the unique maxima of λ(αi).
Exercise 7.16
Note that this approach has yielded a closed-form solution for αi, for given
values of the other hyperparameters. As well as providing insight into the origin of
sparsity in the RVM, this analysis also leads to a practical algorithm for optimizing
the hyperparameters that has signiﬁcant speed advantages. This uses a ﬁxed set
of candidate basis vectors, and then cycles through them in turn to decide whether
each vector should be included in the model or not. The resulting sequential sparse
Bayesian learning algorithm is described below.
Sequential Sparse Bayesian Learning Algorithm
1. If solving a regression problem, initialize β.
2. Initialize using one basis function ϕ1, with hyperparameter α1 set using
(7.101), with the remaining hyperparameters αj for j ̸= i initialized to
inﬁnity, so that only ϕ1 is included in the model.


---
**Page 369**
7.2. Relevance Vector Machines
353
3. Evaluate Σ and m, along with qi and si for all basis functions.
4. Select a candidate basis function ϕi.
5. If q2
i > si, and αi < ∞, so that the basis vector ϕi is already included in
the model, then update αi using (7.101).
6. If q2
i > si, and αi = ∞, then add ϕi to the model, and evaluate hyperpa-
rameter αi using (7.101).
7. If q2
i ⩽si, and αi < ∞then remove basis function ϕi from the model,
and set αi = ∞.
8. If solving a regression problem, update β.
9. If converged terminate, otherwise go to 3.
Note that if q2
i ⩽si and αi = ∞, then the basis function ϕi is already excluded
from the model and no action is required.
In practice, it is convenient to evaluate the quantities
Qi
=
ϕT
i C−1t
(7.102)
Si
=
ϕT
i C−1ϕi.
(7.103)
The quality and sparseness variables can then be expressed in the form
qi
=
αiQi
αi −Si
(7.104)
si
=
αiSi
αi −Si
.
(7.105)
Note that when αi = ∞, we have qi = Qi and si = Si. Using (C.7), we can write
Exercise 7.17
Qi
=
βϕT
i t −β2ϕT
i ΦΣΦTt
(7.106)
Si
=
βϕT
i ϕi −β2ϕT
i ΦΣΦTϕi
(7.107)
where Φ and Σ involve only those basis vectors that correspond to ﬁnite hyperpa-
rameters αi. At each stage the required computations therefore scale like O(M 3),
where M is the number of active basis vectors in the model and is typically much
smaller than the number N of training patterns.
7.2.3
RVM for classiﬁcation
We can extend the relevance vector machine framework to classiﬁcation prob-
lems by applying the ARD prior over weights to a probabilistic linear classiﬁcation
model of the kind studied in Chapter 4. To start with, we consider two-class prob-
lems with a binary target variable t ∈{0, 1}. The model now takes the form of a
linear combination of basis functions transformed by a logistic sigmoid function
y(x, w) = σ

wTφ(x)

(7.108)


---
**Page 370**
354
7. SPARSE KERNEL MACHINES
where σ(·) is the logistic sigmoid function deﬁned by (4.59). If we introduce a
Gaussian prior over the weight vector w, then we obtain the model that has been
considered already in Chapter 4. The difference here is that in the RVM, this model
uses the ARD prior (7.80) in which there is a separate precision hyperparameter
associated with each weight parameter.
In contrast to the regression model, we can no longer integrate analytically over
the parameter vector w. Here we follow Tipping (2001) and use the Laplace ap-
proximation, which was applied to the closely related problem of Bayesian logistic
Section 4.4
regression in Section 4.5.1.
We begin by initializing the hyperparameter vector α. For this given value of
α, we then build a Gaussian approximation to the posterior distribution and thereby
obtain an approximation to the marginal likelihood. Maximization of this approxi-
mate marginal likelihood then leads to a re-estimated value for α, and the process is
repeated until convergence.
Let us consider the Laplace approximation for this model in more detail. For
a ﬁxed value of α, the mode of the posterior distribution over w is obtained by
maximizing
ln p(w|t, α) = ln {p(t|w)p(w|α)} −ln p(t|α)
=
N

n=1
{tn ln yn + (1 −tn) ln(1 −yn)} −1
2wTAw + const (7.109)
where A = diag(αi). This can be done using iterative reweighted least squares
(IRLS) as discussed in Section 4.3.3. For this, we need the gradient vector and
Hessian matrix of the log posterior distribution, which from (7.109) are given by
Exercise 7.18
∇ln p(w|t, α)
=
ΦT(t −y) −Aw
(7.110)
∇∇ln p(w|t, α)
=
−

ΦTBΦ + A

(7.111)
where B is an N × N diagonal matrix with elements bn = yn(1 −yn), the vector
y = (y1, . . . , yN)T, and Φ is the design matrix with elements Φni = φi(xn). Here
we have used the property (4.88) for the derivative of the logistic sigmoid function.
At convergence of the IRLS algorithm, the negative Hessian represents the inverse
covariance matrix for the Gaussian approximation to the posterior distribution.
The mode of the resulting approximation to the posterior distribution, corre-
sponding to the mean of the Gaussian approximation, is obtained setting (7.110) to
zero, giving the mean and covariance of the Laplace approximation in the form
w⋆
=
A−1ΦT(t −y)
(7.112)
Σ
=

ΦTBΦ + A−1 .
(7.113)
We can now use this Laplace approximation to evaluate the marginal likelihood.
Using the general result (4.135) for an integral evaluated using the Laplace approxi-


---
**Page 371**
7.2. Relevance Vector Machines
355
mation, we have
p(t|α)
=

p(t|w)p(w|α) dw
≃
p(t|w⋆)p(w⋆|α)(2π)M/2|Σ|1/2.
(7.114)
If we substitute for p(t|w⋆) and p(w⋆|α) and then set the derivative of the marginal
likelihood with respect to αi equal to zero, we obtain
Exercise 7.19
−1
2(w⋆
i )2 +
1
2αi
−1
2Σii = 0.
(7.115)
Deﬁning γi = 1 −αiΣii and rearranging then gives
αnew
i
=
γi
(w⋆
i )2
(7.116)
which is identical to the re-estimation formula (7.87) obtained for the regression
RVM.
If we deﬁne
t = Φw⋆+ B−1(t −y)
(7.117)
we can write the approximate log marginal likelihood in the form
ln p(t|α, β) = −1
2

N ln(2π) + ln |C| + (t)TC−1t

(7.118)
where
C = B + ΦAΦT.
(7.119)
This takes the same form as (7.85) in the regression case, and so we can apply the
same analysis of sparsity and obtain the same fast learning algorithm in which we
fully optimize a single hyperparameter αi at each step.
Figure 7.12 shows the relevance vector machine applied to a synthetic classiﬁ-
cation data set. We see that the relevance vectors tend not to lie in the region of the
Appendix A
decision boundary, in contrast to the support vector machine. This is consistent with
our earlier discussion of sparsity in the RVM, because a basis function φi(x) centred
on a data point near the boundary will have a vector ϕi that is poorly aligned with
the training data vector t.
One of the potential advantages of the relevance vector machine compared with
the SVM is that it makes probabilistic predictions. For example, this allows the RVM
to be used to help construct an emission density in a nonlinear extension of the linear
dynamical system for tracking faces in video sequences (Williams et al., 2005).
Section 13.3
So far, we have considered the RVM for binary classiﬁcation problems. For
K > 2 classes, we again make use of the probabilistic approach in Section 4.3.4 in
which there are K linear models of the form
ak = wT
k x
(7.120)


---
**Page 372**
356
7. SPARSE KERNEL MACHINES
−2
0
2
−2
0
2
Figure 7.12
Example of the relevance vector machine applied to a synthetic data set, in which the left-hand plot
shows the decision boundary and the data points, with the relevance vectors indicated by circles. Comparison
with the results shown in Figure 7.4 for the corresponding support vector machine shows that the RVM gives a
much sparser model. The right-hand plot shows the posterior probability given by the RVM output in which the
proportion of red (blue) ink indicates the probability of that point belonging to the red (blue) class.
which are combined using a softmax function to give outputs
yk(x) =
exp(ak)

j
exp(aj)
.
(7.121)
The log likelihood function is then given by
ln p(T|w1, . . . , wK) =
N

n=1
K

k=1
ytnk
nk
(7.122)
where the target values tnk have a 1-of-K coding for each data point n, and T is a
matrix with elements tnk. Again, the Laplace approximation can be used to optimize
the hyperparameters (Tipping, 2001), in which the model and its Hessian are found
using IRLS. This gives a more principled approach to multiclass classiﬁcation than
the pairwise method used in the support vector machine and also provides probabilis-
tic predictions for new data points. The principal disadvantage is that the Hessian
matrix has size MK ×MK, where M is the number of active basis functions, which
gives an additional factor of K3 in the computational cost of training compared with
the two-class RVM.
The principal disadvantage of the relevance vector machine is the relatively long
training times compared with the SVM. This is offset, however, by the avoidance of
cross-validation runs to set the model complexity parameters. Furthermore, because
it yields sparser models, the computation time on test points, which is usually the
more important consideration in practice, is typically much less.


---
**Page 373**
Exercises
357
Exercises
7.1
(⋆⋆) www
Suppose we have a data set of input vectors {xn} with corresponding
target values tn ∈{−1, 1}, and suppose that we model the density of input vec-
tors within each class separately using a Parzen kernel density estimator (see Sec-
tion 2.5.1) with a kernel k(x, x′). Write down the minimum misclassiﬁcation-rate
decision rule assuming the two classes have equal prior probability. Show also that,
if the kernel is chosen to be k(x, x′) = xTx′, then the classiﬁcation rule reduces to
simply assigning a new input vector to the class having the closest mean. Finally,
show that, if the kernel takes the form k(x, x′) = φ(x)Tφ(x′), that the classiﬁcation
is based on the closest mean in the feature space φ(x).
7.2
(⋆)
Show that, if the 1 on the right-hand side of the constraint (7.5) is replaced by
some arbitrary constant γ > 0, the solution for the maximum margin hyperplane is
unchanged.
7.3
(⋆⋆)
Show that, irrespective of the dimensionality of the data space, a data set
consisting of just two data points, one from each class, is sufﬁcient to determine the
location of the maximum-margin hyperplane.
7.4
(⋆⋆) www
Show that the value ρ of the margin for the maximum-margin hyper-
plane is given by
1
ρ2 =
N

n=1
an
(7.123)
where {an} are given by maximizing (7.10) subject to the constraints (7.11) and
(7.12).
7.5
(⋆⋆) Show that the values of ρ and {an} in the previous exercise also satisfy
1
ρ2 = 2L(a)
(7.124)
where L(a) is deﬁned by (7.10). Similarly, show that
1
ρ2 = ∥w∥2.
(7.125)
7.6
(⋆)
Consider the logistic regression model with a target variable t ∈{−1, 1}. If
we deﬁne p(t = 1|y) = σ(y) where y(x) is given by (7.1), show that the negative
log likelihood, with the addition of a quadratic regularization term, takes the form
(7.47).
7.7
(⋆)
Consider the Lagrangian (7.56) for the regression support vector machine. By
setting the derivatives of the Lagrangian with respect to w, b, ξn, and ξn to zero and
then back substituting to eliminate the corresponding variables, show that the dual
Lagrangian is given by (7.61).


---
**Page 374**
358
7. SPARSE KERNEL MACHINES
7.8
(⋆) www
For the regression support vector machine considered in Section 7.1.4,
show that all training data points for which ξn > 0 will have an = C, and similarly
all points for which ξn > 0 will have an = C.
7.9
(⋆) Verify the results (7.82) and (7.83) for the mean and covariance of the posterior
distribution over weights in the regression RVM.
7.10
(⋆⋆) www
Derive the result (7.85) for the marginal likelihood function in the
regression RVM, by performing the Gaussian integral over w in (7.84) using the
technique of completing the square in the exponential.
7.11
(⋆⋆) Repeat the above exercise, but this time make use of the general result (2.115).
7.12
(⋆⋆) www
Show that direct maximization of the log marginal likelihood (7.85) for
the regression relevance vector machine leads to the re-estimation equations (7.87)
and (7.88) where γi is deﬁned by (7.89).
7.13
(⋆⋆) In the evidence framework for RVM regression, we obtained the re-estimation
formulae (7.87) and (7.88) by maximizing the marginal likelihood given by (7.85).
Extend this approach by inclusion of hyperpriors given by gamma distributions of
the form (B.26) and obtain the corresponding re-estimation formulae for α and β by
maximizing the corresponding posterior probability p(t, α, β|X) with respect to α
and β.
7.14
(⋆⋆)
Derive the result (7.90) for the predictive distribution in the relevance vector
machine for regression. Show that the predictive variance is given by (7.91).
7.15
(⋆⋆) www
Using the results (7.94) and (7.95), show that the marginal likelihood
(7.85) can be written in the form (7.96), where λ(αn) is deﬁned by (7.97) and the
sparsity and quality factors are deﬁned by (7.98) and (7.99), respectively.
7.16
(⋆)
By taking the second derivative of the log marginal likelihood (7.97) for the
regression RVM with respect to the hyperparameter αi, show that the stationary
point given by (7.101) is a maximum of the marginal likelihood.
7.17
(⋆⋆)
Using (7.83) and (7.86), together with the matrix identity (C.7), show that
the quantities Sn and Qn deﬁned by (7.102) and (7.103) can be written in the form
(7.106) and (7.107).
7.18
(⋆) www
Show that the gradient vector and Hessian matrix of the log poste-
rior distribution (7.109) for the classiﬁcation relevance vector machine are given by
(7.110) and (7.111).
7.19
(⋆⋆) Verify that maximization of the approximate log marginal likelihood function
(7.114) for the classiﬁcation relevance vector machine leads to the result (7.116) for
re-estimation of the hyperparameters.


---
**Page 375**
8
Graphical
Models
Probabilities play a central role in modern pattern recognition. We have seen in
Chapter 1 that probability theory can be expressed in terms of two simple equations
corresponding to the sum rule and the product rule. All of the probabilistic infer-
ence and learning manipulations discussed in this book, no matter how complex,
amount to repeated application of these two equations. We could therefore proceed
to formulate and solve complicated probabilistic models purely by algebraic ma-
nipulation. However, we shall ﬁnd it highly advantageous to augment the analysis
using diagrammatic representations of probability distributions, called probabilistic
graphical models. These offer several useful properties:
1. They provide a simple way to visualize the structure of a probabilistic model
and can be used to design and motivate new models.
2. Insights into the properties of the model, including conditional independence
properties, can be obtained by inspection of the graph.
359


---
**Page 376**
360
8. GRAPHICAL MODELS
3. Complex computations, required to perform inference and learning in sophis-
ticated models, can be expressed in terms of graphical manipulations, in which
underlying mathematical expressions are carried along implicitly.
A graph comprises nodes (also called vertices) connected by links (also known
as edges or arcs). In a probabilistic graphical model, each node represents a random
variable (or group of random variables), and the links express probabilistic relation-
ships between these variables. The graph then captures the way in which the joint
distribution over all of the random variables can be decomposed into a product of
factors each depending only on a subset of the variables. We shall begin by dis-
cussing Bayesian networks, also known as directed graphical models, in which the
links of the graphs have a particular directionality indicated by arrows. The other
major class of graphical models are Markov random ﬁelds, also known as undirected
graphical models, in which the links do not carry arrows and have no directional
signiﬁcance. Directed graphs are useful for expressing causal relationships between
random variables, whereas undirected graphs are better suited to expressing soft con-
straints between random variables. For the purposes of solving inference problems,
it is often convenient to convert both directed and undirected graphs into a different
representation called a factor graph.
In this chapter, we shall focus on the key aspects of graphical models as needed
for applications in pattern recognition and machine learning. More general treat-
ments of graphical models can be found in the books by Whittaker (1990), Lauritzen
(1996), Jensen (1996), Castillo et al. (1997), Jordan (1999), Cowell et al. (1999),
and Jordan (2007).
8.1. Bayesian Networks
In order to motivate the use of directed graphs to describe probability distributions,
consider ﬁrst an arbitrary joint distribution p(a, b, c) over three variables a, b, and c.
Note that at this stage, we do not need to specify anything further about these vari-
ables, such as whether they are discrete or continuous. Indeed, one of the powerful
aspects of graphical models is that a speciﬁc graph can make probabilistic statements
for a broad class of distributions. By application of the product rule of probability
(1.11), we can write the joint distribution in the form
p(a, b, c) = p(c|a, b)p(a, b).
(8.1)
A second application of the product rule, this time to the second term on the right-
hand side of (8.1), gives
p(a, b, c) = p(c|a, b)p(b|a)p(a).
(8.2)
Note that this decomposition holds for any choice of the joint distribution. We now
represent the right-hand side of (8.2) in terms of a simple graphical model as follows.
First we introduce a node for each of the random variables a, b, and c and associate
each node with the corresponding conditional distribution on the right-hand side of


---
**Page 377**
8.1. Bayesian Networks
361
Figure 8.1
A directed graphical model representing the joint probabil-
ity distribution over three variables a, b, and c, correspond-
ing to the decomposition on the right-hand side of (8.2).
a
b
c
(8.2). Then, for each conditional distribution we add directed links (arrows) to the
graph from the nodes corresponding to the variables on which the distribution is
conditioned. Thus for the factor p(c|a, b), there will be links from nodes a and b to
node c, whereas for the factor p(a) there will be no incoming links. The result is the
graph shown in Figure 8.1. If there is a link going from a node a to a node b, then we
say that node a is the parent of node b, and we say that node b is the child of node a.
Note that we shall not make any formal distinction between a node and the variable
to which it corresponds but will simply use the same symbol to refer to both.
An interesting point to note about (8.2) is that the left-hand side is symmetrical
with respect to the three variables a, b, and c, whereas the right-hand side is not.
Indeed, in making the decomposition in (8.2), we have implicitly chosen a particular
ordering, namely a, b, c, and had we chosen a different ordering we would have
obtained a different decomposition and hence a different graphical representation.
We shall return to this point later.
For the moment let us extend the example of Figure 8.1 by considering the joint
distribution over K variables given by p(x1, . . . , xK). By repeated application of
the product rule of probability, this joint distribution can be written as a product of
conditional distributions, one for each of the variables
p(x1, . . . , xK) = p(xK|x1, . . . , xK−1) . . . p(x2|x1)p(x1).
(8.3)
For a given choice of K, we can again represent this as a directed graph having K
nodes, one for each conditional distribution on the right-hand side of (8.3), with each
node having incoming links from all lower numbered nodes. We say that this graph
is fully connected because there is a link between every pair of nodes.
So far, we have worked with completely general joint distributions, so that the
decompositions, and their representations as fully connected graphs, will be applica-
ble to any choice of distribution. As we shall see shortly, it is the absence of links
in the graph that conveys interesting information about the properties of the class of
distributions that the graph represents. Consider the graph shown in Figure 8.2. This
is not a fully connected graph because, for instance, there is no link from x1 to x2 or
from x3 to x7.
We shall now go from this graph to the corresponding representation of the joint
probability distribution written in terms of the product of a set of conditional dis-
tributions, one for each node in the graph. Each such conditional distribution will
be conditioned only on the parents of the corresponding node in the graph. For in-
stance, x5 will be conditioned on x1 and x3. The joint distribution of all 7 variables


---
**Page 378**
362
8. GRAPHICAL MODELS
Figure 8.2
Example of a directed acyclic graph describing the joint
distribution over variables x1, . . . , x7. The corresponding
decomposition of the joint distribution is given by (8.4).
x1
x2
x3
x4
x5
x6
x7
is therefore given by
p(x1)p(x2)p(x3)p(x4|x1, x2, x3)p(x5|x1, x3)p(x6|x4)p(x7|x4, x5).
(8.4)
The reader should take a moment to study carefully the correspondence between
(8.4) and Figure 8.2.
We can now state in general terms the relationship between a given directed
graph and the corresponding distribution over the variables. The joint distribution
deﬁned by a graph is given by the product, over all of the nodes of the graph, of
a conditional distribution for each node conditioned on the variables corresponding
to the parents of that node in the graph. Thus, for a graph with K nodes, the joint
distribution is given by
p(x) =
K

k=1
p(xk|pak)
(8.5)
where pak denotes the set of parents of xk, and x = {x1, . . . , xK}. This key
equation expresses the factorization properties of the joint distribution for a directed
graphical model. Although we have considered each node to correspond to a single
variable, we can equally well associate sets of variables and vector-valued variables
with the nodes of a graph. It is easy to show that the representation on the right-
hand side of (8.5) is always correctly normalized provided the individual conditional
distributions are normalized.
Exercise 8.1
The directed graphs that we are considering are subject to an important restric-
tion namely that there must be no directed cycles, in other words there are no closed
paths within the graph such that we can move from node to node along links follow-
ing the direction of the arrows and end up back at the starting node. Such graphs are
also called directed acyclic graphs, or DAGs. This is equivalent to the statement that
Exercise 8.2
there exists an ordering of the nodes such that there are no links that go from any
node to any lower numbered node.
8.1.1
Example: Polynomial regression
As an illustration of the use of directed graphs to describe probability distri-
butions, we consider the Bayesian polynomial regression model introduced in Sec-


---
**Page 379**
8.1. Bayesian Networks
363
Figure 8.3
Directed graphical model representing the joint
distribution (8.6) corresponding to the Bayesian
polynomial regression model introduced in Sec-
tion 1.2.6.
w
t1
tN
tion 1.2.6. The random variables in this model are the vector of polynomial coefﬁ-
cients w and the observed data t = (t1, . . . , tN)T. In addition, this model contains
the input data x = (x1, . . . , xN)T, the noise variance σ2, and the hyperparameter α
representing the precision of the Gaussian prior over w, all of which are parameters
of the model rather than random variables. Focussing just on the random variables
for the moment, we see that the joint distribution is given by the product of the prior
p(w) and N conditional distributions p(tn|w) for n = 1, . . . , N so that
p(t, w) = p(w)
N

n=1
p(tn|w).
(8.6)
This joint distribution can be represented by a graphical model shown in Figure 8.3.
When we start to deal with more complex models later in the book, we shall ﬁnd
it inconvenient to have to write out multiple nodes of the form t1, . . . , tN explicitly as
in Figure 8.3. We therefore introduce a graphical notation that allows such multiple
nodes to be expressed more compactly, in which we draw a single representative
node tn and then surround this with a box, called a plate, labelled with N indicating
that there are N nodes of this kind. Re-writing the graph of Figure 8.3 in this way,
we obtain the graph shown in Figure 8.4.
We shall sometimes ﬁnd it helpful to make the parameters of a model, as well as
its stochastic variables, explicit. In this case, (8.6) becomes
p(t, w|x, α, σ2) = p(w|α)
N

n=1
p(tn|w, xn, σ2).
Correspondingly, we can make x and α explicit in the graphical representation. To
do this, we shall adopt the convention that random variables will be denoted by open
circles, and deterministic parameters will be denoted by smaller solid circles. If we
take the graph of Figure 8.4 and include the deterministic parameters, we obtain the
graph shown in Figure 8.5.
When we apply a graphical model to a problem in machine learning or pattern
recognition, we will typically set some of the random variables to speciﬁc observed
Figure 8.4
An alternative, more compact, representation of the graph
shown in Figure 8.3 in which we have introduced a plate
(the box labelled N) that represents N nodes of which only
a single example tn is shown explicitly.
tn
N
w


---
**Page 380**
364
8. GRAPHICAL MODELS
Figure 8.5
This shows the same model as in Figure 8.4 but
with the deterministic parameters shown explicitly
by the smaller solid nodes.
tn
xn
N
w
α
σ2
values, for example the variables {tn} from the training set in the case of polynomial
curve ﬁtting. In a graphical model, we will denote such observed variables by shad-
ing the corresponding nodes. Thus the graph corresponding to Figure 8.5 in which
the variables {tn} are observed is shown in Figure 8.6. Note that the value of w is
not observed, and so w is an example of a latent variable, also known as a hidden
variable. Such variables play a crucial role in many probabilistic models and will
form the focus of Chapters 9 and 12.
Having observed the values {tn} we can, if desired, evaluate the posterior dis-
tribution of the polynomial coefﬁcients w as discussed in Section 1.2.5. For the
moment, we note that this involves a straightforward application of Bayes’ theorem
p(w|T) ∝p(w)
N

n=1
p(tn|w)
(8.7)
where again we have omitted the deterministic parameters in order to keep the nota-
tion uncluttered.
In general, model parameters such as w are of little direct interest in themselves,
because our ultimate goal is to make predictions for new input values. Suppose we
are given a new input value x and we wish to ﬁnd the corresponding probability dis-
tribution fort conditioned on the observed data. The graphical model that describes
this problem is shown in Figure 8.7, and the corresponding joint distribution of all
of the random variables in this model, conditioned on the deterministic parameters,
is then given by
p(t, t, w|x, x, α, σ2) =
 N

n=1
p(tn|xn, w, σ2)
 
p(w|α)p(t|x, w, σ2).
(8.8)
Figure 8.6
As in Figure 8.5 but with the nodes {tn} shaded
to indicate that the corresponding random vari-
ables have been set to their observed (training set)
values.
tn
xn
N
w
α
σ2


---
**Page 381**
8.1. Bayesian Networks
365
Figure 8.7
The polynomial regression model, corresponding
to Figure 8.6, showing also a new input value bx
together with the corresponding model prediction
bt.
tn
xn
N
w
α
ˆt
σ2
ˆx
The required predictive distribution for t is then obtained, from the sum rule of
probability, by integrating out the model parameters w so that
p(t|x, x, t, α, σ2) ∝

p(t, t, w|x, x, α, σ2) dw
where we are implicitly setting the random variables in t to the speciﬁc values ob-
served in the data set. The details of this calculation were discussed in Chapter 3.
8.1.2
Generative models
There are many situations in which we wish to draw samples from a given prob-
ability distribution. Although we shall devote the whole of Chapter 11 to a detailed
discussion of sampling methods, it is instructive to outline here one technique, called
ancestral sampling, which is particularly relevant to graphical models. Consider a
joint distribution p(x1, . . . , xK) over K variables that factorizes according to (8.5)
corresponding to a directed acyclic graph. We shall suppose that the variables have
been ordered such that there are no links from any node to any lower numbered node,
in other words each node has a higher number than any of its parents. Our goal is to
draw a sample x1, . . . , xK from the joint distribution.
To do this, we start with the lowest-numbered node and draw a sample from the
distribution p(x1), which we call x1. We then work through each of the nodes in or-
der, so that for node n we draw a sample from the conditional distribution p(xn|pan)
in which the parent variables have been set to their sampled values. Note that at each
stage, these parent values will always be available because they correspond to lower-
numbered nodes that have already been sampled. Techniques for sampling from
speciﬁc distributions will be discussed in detail in Chapter 11. Once we have sam-
pled from the ﬁnal variable xK, we will have achieved our objective of obtaining a
sample from the joint distribution. To obtain a sample from some marginal distribu-
tion corresponding to a subset of the variables, we simply take the sampled values
for the required nodes and ignore the sampled values for the remaining nodes. For
example, to draw a sample from the distribution p(x2, x4), we simply sample from
the full joint distribution and then retain the values x2, x4 and discard the remaining
values {xj̸=2,4}.


---
**Page 382**
366
8. GRAPHICAL MODELS
Figure 8.8
A graphical model representing the process by which
images of objects are created, in which the identity
of an object (a discrete variable) and the position and
orientation of that object (continuous variables) have
independent prior probabilities. The image (a vector
of pixel intensities) has a probability distribution that
is dependent on the identity of the object as well as
on its position and orientation.
Image
Object
Orientation
Position
For practical applications of probabilistic models, it will typically be the higher-
numbered variables corresponding to terminal nodes of the graph that represent the
observations, with lower-numbered nodes corresponding to latent variables. The
primary role of the latent variables is to allow a complicated distribution over the
observed variables to be represented in terms of a model constructed from simpler
(typically exponential family) conditional distributions.
We can interpret such models as expressing the processes by which the observed
data arose. For instance, consider an object recognition task in which each observed
data point corresponds to an image (comprising a vector of pixel intensities) of one
of the objects. In this case, the latent variables might have an interpretation as the
position and orientation of the object. Given a particular observed image, our goal is
to ﬁnd the posterior distribution over objects, in which we integrate over all possible
positions and orientations. We can represent this problem using a graphical model
of the form show in Figure 8.8.
The graphical model captures the causal process (Pearl, 1988) by which the ob-
served data was generated. For this reason, such models are often called generative
models. By contrast, the polynomial regression model described by Figure 8.5 is
not generative because there is no probability distribution associated with the input
variable x, and so it is not possible to generate synthetic data points from this model.
We could make it generative by introducing a suitable prior distribution p(x), at the
expense of a more complex model.
The hidden variables in a probabilistic model need not, however, have any ex-
plicit physical interpretation but may be introduced simply to allow a more complex
joint distribution to be constructed from simpler components. In either case, the
technique of ancestral sampling applied to a generative model mimics the creation
of the observed data and would therefore give rise to ‘fantasy’ data whose probability
distribution (if the model were a perfect representation of reality) would be the same
as that of the observed data. In practice, producing synthetic observations from a
generative model can prove informative in understanding the form of the probability
distribution represented by that model.
8.1.3
Discrete variables
We have discussed the importance of probability distributions that are members
of the exponential family, and we have seen that this family includes many well-
Section 2.4
known distributions as particular cases. Although such distributions are relatively
simple, they form useful building blocks for constructing more complex probability


---
**Page 383**
8.1. Bayesian Networks
367
Figure 8.9
(a) This fully-connected graph describes a general distribu-
tion over two K-state discrete variables having a total of
K2 −1 parameters. (b) By dropping the link between the
nodes, the number of parameters is reduced to 2(K −1).
(a)
x1
x2
(b)
x1
x2
distributions, and the framework of graphical models is very useful in expressing the
way in which these building blocks are linked together.
Such models have particularly nice properties if we choose the relationship be-
tween each parent-child pair in a directed graph to be conjugate, and we shall ex-
plore several examples of this shortly. Two cases are particularly worthy of note,
namely when the parent and child node each correspond to discrete variables and
when they each correspond to Gaussian variables, because in these two cases the
relationship can be extended hierarchically to construct arbitrarily complex directed
acyclic graphs. We begin by examining the discrete case.
The probability distribution p(x|µ) for a single discrete variable x having K
possible states (using the 1-of-K representation) is given by
p(x|µ) =
K

k=1
µxk
k
(8.9)
and is governed by the parameters µ = (µ1, . . . , µK)T.
Due to the constraint

k µk = 1, only K −1 values for µk need to be speciﬁed in order to deﬁne the
distribution.
Now suppose that we have two discrete variables, x1 and x2, each of which has
K states, and we wish to model their joint distribution. We denote the probability of
observing both x1k = 1 and x2l = 1 by the parameter µkl, where x1k denotes the
kth component of x1, and similarly for x2l. The joint distribution can be written
p(x1, x2|µ) =
K

k=1
K

l=1
µx1kx2l
kl
.
Because the parameters µkl are subject to the constraint 
k

l µkl = 1, this distri-
bution is governed by K2 −1 parameters. It is easily seen that the total number of
parameters that must be speciﬁed for an arbitrary joint distribution over M variables
is KM −1 and therefore grows exponentially with the number M of variables.
Using the product rule, we can factor the joint distribution p(x1, x2) in the form
p(x2|x1)p(x1), which corresponds to a two-node graph with a link going from the
x1 node to the x2 node as shown in Figure 8.9(a). The marginal distribution p(x1)
is governed by K −1 parameters, as before, Similarly, the conditional distribution
p(x2|x1) requires the speciﬁcation of K −1 parameters for each of the K possible
values of x1. The total number of parameters that must be speciﬁed in the joint
distribution is therefore (K −1) + K(K −1) = K2 −1 as before.
Now suppose that the variables x1 and x2 were independent, corresponding to
the graphical model shown in Figure 8.9(b). Each variable is then described by


---
**Page 384**
368
8. GRAPHICAL MODELS
Figure 8.10
This chain of M discrete nodes, each
having K states, requires the speciﬁcation of K −1 +
(M −1)K(K −1) parameters, which grows linearly
with the length M of the chain. In contrast, a fully con-
nected graph of M nodes would have KM −1 param-
eters, which grows exponentially with M.
x1
x2
xM
a separate multinomial distribution, and the total number of parameters would be
2(K −1). For a distribution over M independent discrete variables, each having K
states, the total number of parameters would be M(K −1), which therefore grows
linearly with the number of variables. From a graphical perspective, we have reduced
the number of parameters by dropping links in the graph, at the expense of having a
restricted class of distributions.
More generally, if we have M discrete variables x1, . . . , xM, we can model
the joint distribution using a directed graph with one variable corresponding to each
node. The conditional distribution at each node is given by a set of nonnegative pa-
rameters subject to the usual normalization constraint. If the graph is fully connected
then we have a completely general distribution having KM −1 parameters, whereas
if there are no links in the graph the joint distribution factorizes into the product of
the marginals, and the total number of parameters is M(K −1). Graphs having in-
termediate levels of connectivity allow for more general distributions than the fully
factorized one while requiring fewer parameters than the general joint distribution.
As an illustration, consider the chain of nodes shown in Figure 8.10. The marginal
distribution p(x1) requires K −1 parameters, whereas each of the M −1 condi-
tional distributions p(xi|xi−1), for i = 2, . . . , M, requires K(K −1) parameters.
This gives a total parameter count of K −1+(M −1)K(K −1), which is quadratic
in K and which grows linearly (rather than exponentially) with the length M of the
chain.
An alternative way to reduce the number of independent parameters in a model
is by sharing parameters (also known as tying of parameters). For instance, in the
chain example of Figure 8.10, we can arrange that all of the conditional distributions
p(xi|xi−1), for i = 2, . . . , M, are governed by the same set of K(K−1) parameters.
Together with the K−1 parameters governing the distribution of x1, this gives a total
of K2 −1 parameters that must be speciﬁed in order to deﬁne the joint distribution.
We can turn a graph over discrete variables into a Bayesian model by introduc-
ing Dirichlet priors for the parameters. From a graphical point of view, each node
then acquires an additional parent representing the Dirichlet distribution over the pa-
rameters associated with the corresponding discrete node. This is illustrated for the
chain model in Figure 8.11. The corresponding model in which we tie the parame-
ters governing the conditional distributions p(xi|xi−1), for i = 2, . . . , M, is shown
in Figure 8.12.
Another way of controlling the exponential growth in the number of parameters
in models of discrete variables is to use parameterized models for the conditional
distributions instead of complete tables of conditional probability values. To illus-
trate this idea, consider the graph in Figure 8.13 in which all of the nodes represent
binary variables. Each of the parent variables xi is governed by a single parame-


---
**Page 385**
8.1. Bayesian Networks
369
Figure 8.11
An extension of the model of
Figure 8.10 to include Dirich-
let
priors
over
the
param-
eters governing the discrete
distributions.
x1
x2
xM
µ1
µ2
µM
Figure 8.12
As in Figure 8.11 but with a sin-
gle set of parameters µ shared
amongst all of the conditional
distributions p(xi|xi−1).
x1
x2
xM
µ1
µ
ter µi representing the probability p(xi = 1), giving M parameters in total for the
parent nodes. The conditional distribution p(y|x1, . . . , xM), however, would require
2M parameters representing the probability p(y = 1) for each of the 2M possible
settings of the parent variables. Thus in general the number of parameters required
to specify this conditional distribution will grow exponentially with M. We can ob-
tain a more parsimonious form for the conditional distribution by using a logistic
sigmoid function acting on a linear combination of the parent variables, giving
Section 2.4
p(y = 1|x1, . . . , xM) = σ

w0 +
M

i=1
wixi

= σ(wTx)
(8.10)
where σ(a) = (1+exp(−a))−1 is the logistic sigmoid, x = (x0, x1, . . . , xM)T is an
(M + 1)-dimensional vector of parent states augmented with an additional variable
x0 whose value is clamped to 1, and w = (w0, w1, . . . , wM)T is a vector of M + 1
parameters. This is a more restricted form of conditional distribution than the general
case but is now governed by a number of parameters that grows linearly with M. In
this sense, it is analogous to the choice of a restrictive form of covariance matrix (for
example, a diagonal matrix) in a multivariate Gaussian distribution. The motivation
for the logistic sigmoid representation was discussed in Section 4.2.
Figure 8.13
A graph comprising M parents x1, . . . , xM and a sin-
gle child y, used to illustrate the idea of parameterized
conditional distributions for discrete variables.
y
x1
xM


---
**Page 386**
370
8. GRAPHICAL MODELS
8.1.4
Linear-Gaussian models
In the previous section, we saw how to construct joint probability distributions
over a set of discrete variables by expressing the variables as nodes in a directed
acyclic graph. Here we show how a multivariate Gaussian can be expressed as a
directed graph corresponding to a linear-Gaussian model over the component vari-
ables. This allows us to impose interesting structure on the distribution, with the
general Gaussian and the diagonal covariance Gaussian representing opposite ex-
tremes. Several widely used techniques are examples of linear-Gaussian models,
such as probabilistic principal component analysis, factor analysis, and linear dy-
namical systems (Roweis and Ghahramani, 1999). We shall make extensive use of
the results of this section in later chapters when we consider some of these techniques
in detail.
Consider an arbitrary directed acyclic graph over D variables in which node i
represents a single continuous random variable xi having a Gaussian distribution.
The mean of this distribution is taken to be a linear combination of the states of its
parent nodes pai of node i
p(xi|pai) = N
⎛
⎝xi


j∈pai
wijxj + bi, vi
⎞
⎠
(8.11)
where wij and bi are parameters governing the mean, and vi is the variance of the
conditional distribution for xi. The log of the joint distribution is then the log of the
product of these conditionals over all nodes in the graph and hence takes the form
ln p(x)
=
D

i=1
ln p(xi|pai)
(8.12)
=
−
D

i=1
1
2vi
⎛
⎝xi −

j∈pai
wijxj −bi
⎞
⎠
2
+ const
(8.13)
where x = (x1, . . . , xD)T and ‘const’ denotes terms independent of x. We see that
this is a quadratic function of the components of x, and hence the joint distribution
p(x) is a multivariate Gaussian.
We can determine the mean and covariance of the joint distribution recursively
as follows. Each variable xi has (conditional on the states of its parents) a Gaussian
distribution of the form (8.11) and so
xi =

j∈pai
wijxj + bi + √viϵi
(8.14)
where ϵi is a zero mean, unit variance Gaussian random variable satisfying E[ϵi] = 0
and E[ϵiϵj] = Iij, where Iij is the i, j element of the identity matrix. Taking the
expectation of (8.14), we have
E[xi] =

j∈pai
wijE[xj] + bi.
(8.15)


---
**Page 387**
8.1. Bayesian Networks
371
Figure 8.14
A directed graph over three Gaussian variables,
with one missing link.
x1
x2
x3
Thus we can ﬁnd the components of E[x] = (E[x1], . . . , E[xD])T by starting at the
lowest numbered node and working recursively through the graph (here we again
assume that the nodes are numbered such that each node has a higher number than
its parents). Similarly, we can use (8.14) and (8.15) to obtain the i, j element of the
covariance matrix for p(x) in the form of a recursion relation
cov[xi, xj]
=
E [(xi −E[xi])(xj −E[xj])]
=
E
⎡
⎣(xi −E[xi])
⎧
⎨
⎩

k∈paj
wjk(xk −E[xk]) + √vjϵj
⎫
⎬
⎭
⎤
⎦
=

k∈paj
wjkcov[xi, xk] + Iijvj
(8.16)
and so the covariance can similarly be evaluated recursively starting from the lowest
numbered node.
Let us consider two extreme cases. First of all, suppose that there are no links
in the graph, which therefore comprises D isolated nodes. In this case, there are no
parameters wij and so there are just D parameters bi and D parameters vi. From
the recursion relations (8.15) and (8.16), we see that the mean of p(x) is given by
(b1, . . . , bD)T and the covariance matrix is diagonal of the form diag(v1, . . . , vD).
The joint distribution has a total of 2D parameters and represents a set of D inde-
pendent univariate Gaussian distributions.
Now consider a fully connected graph in which each node has all lower num-
bered nodes as parents. The matrix wij then has i −1 entries on the ith row and
hence is a lower triangular matrix (with no entries on the leading diagonal). Then
the total number of parameters wij is obtained by taking the number D2 of elements
in a D ×D matrix, subtracting D to account for the absence of elements on the lead-
ing diagonal, and then dividing by 2 because the matrix has elements only below the
diagonal, giving a total of D(D−1)/2. The total number of independent parameters
{wij} and {vi} in the covariance matrix is therefore D(D + 1)/2 corresponding to
a general symmetric covariance matrix.
Section 2.3
Graphs having some intermediate level of complexity correspond to joint Gaus-
sian distributions with partially constrained covariance matrices. Consider for ex-
ample the graph shown in Figure 8.14, which has a link missing between variables
x1 and x3. Using the recursion relations (8.15) and (8.16), we see that the mean and
covariance of the joint distribution are given by
Exercise 8.7
µ
=
(b1, b2 + w21b1, b3 + w32b2 + w32w21b1)T
(8.17)
Σ
=

v1
w21v1
w32w21v1
w21v1
v2 + w2
21v1
w32(v2 + w2
21v1)
w32w21v1
w32(v2 + w2
21v1)
v3 + w2
32(v2 + w2
21v1)

. (8.18)


---
**Page 388**
372
8. GRAPHICAL MODELS
We can readily extend the linear-Gaussian graphical model to the case in which
the nodes of the graph represent multivariate Gaussian variables. In this case, we can
write the conditional distribution for node i in the form
p(xi|pai) = N
⎛
⎝xi


j∈pai
Wijxj + bi, Σi
⎞
⎠
(8.19)
where now Wij is a matrix (which is nonsquare if xi and xj have different dimen-
sionalities). Again it is easy to verify that the joint distribution over all variables is
Gaussian.
Note that we have already encountered a speciﬁc example of the linear-Gaussian
relationship when we saw that the conjugate prior for the mean µ of a Gaussian
Section 2.3.6
variable x is itself a Gaussian distribution over µ. The joint distribution over x and
µ is therefore Gaussian. This corresponds to a simple two-node graph in which
the node representing µ is the parent of the node representing x. The mean of the
distribution over µ is a parameter controlling a prior, and so it can be viewed as a
hyperparameter. Because the value of this hyperparameter may itself be unknown,
we can again treat it from a Bayesian perspective by introducing a prior over the
hyperparameter, sometimes called a hyperprior, which is again given by a Gaussian
distribution. This type of construction can be extended in principle to any level and is
an illustration of a hierarchical Bayesian model, of which we shall encounter further
examples in later chapters.
8.2. Conditional Independence
An important concept for probability distributions over multiple variables is that of
conditional independence (Dawid, 1980). Consider three variables a, b, and c, and
suppose that the conditional distribution of a, given b and c, is such that it does not
depend on the value of b, so that
p(a|b, c) = p(a|c).
(8.20)
We say that a is conditionally independent of b given c. This can be expressed in a
slightly different way if we consider the joint distribution of a and b conditioned on
c, which we can write in the form
p(a, b|c)
=
p(a|b, c)p(b|c)
=
p(a|c)p(b|c).
(8.21)
where we have used the product rule of probability together with (8.20). Thus we
see that, conditioned on c, the joint distribution of a and b factorizes into the prod-
uct of the marginal distribution of a and the marginal distribution of b (again both
conditioned on c). This says that the variables a and b are statistically independent,
given c. Note that our deﬁnition of conditional independence will require that (8.20),


---
**Page 389**
8.2. Conditional Independence
373
Figure 8.15
The ﬁrst of three examples of graphs over three variables
a, b, and c used to discuss conditional independence
properties of directed graphical models.
c
a
b
or equivalently (8.21), must hold for every possible value of c, and not just for some
values. We shall sometimes use a shorthand notation for conditional independence
(Dawid, 1979) in which
a ⊥⊥b | c
(8.22)
denotes that a is conditionally independent of b given c and is equivalent to (8.20).
Conditional independence properties play an important role in using probabilis-
tic models for pattern recognition by simplifying both the structure of a model and
the computations needed to perform inference and learning under that model. We
shall see examples of this shortly.
If we are given an expression for the joint distribution over a set of variables in
terms of a product of conditional distributions (i.e., the mathematical representation
underlying a directed graph), then we could in principle test whether any poten-
tial conditional independence property holds by repeated application of the sum and
product rules of probability. In practice, such an approach would be very time con-
suming. An important and elegant feature of graphical models is that conditional
independence properties of the joint distribution can be read directly from the graph
without having to perform any analytical manipulations. The general framework
for achieving this is called d-separation, where the ‘d’ stands for ‘directed’ (Pearl,
1988). Here we shall motivate the concept of d-separation and give a general state-
ment of the d-separation criterion. A formal proof can be found in Lauritzen (1996).
8.2.1
Three example graphs
We begin our discussion of the conditional independence properties of directed
graphs by considering three simple examples each involving graphs having just three
nodes. Together, these will motivate and illustrate the key concepts of d-separation.
The ﬁrst of the three examples is shown in Figure 8.15, and the joint distribution
corresponding to this graph is easily written down using the general result (8.5) to
give
p(a, b, c) = p(a|c)p(b|c)p(c).
(8.23)
If none of the variables are observed, then we can investigate whether a and b are
independent by marginalizing both sides of (8.23) with respect to c to give
p(a, b) =

c
p(a|c)p(b|c)p(c).
(8.24)
In general, this does not factorize into the product p(a)p(b), and so
a ̸⊥⊥b | ∅
(8.25)


---
**Page 390**
374
8. GRAPHICAL MODELS
Figure 8.16
As in Figure 8.15 but where we have conditioned on the
value of variable c.
c
a
b
where ∅denotes the empty set, and the symbol ̸⊥⊥means that the conditional inde-
pendence property does not hold in general. Of course, it may hold for a particular
distribution by virtue of the speciﬁc numerical values associated with the various
conditional probabilities, but it does not follow in general from the structure of the
graph.
Now suppose we condition on the variable c, as represented by the graph of
Figure 8.16. From (8.23), we can easily write down the conditional distribution of a
and b, given c, in the form
p(a, b|c)
=
p(a, b, c)
p(c)
=
p(a|c)p(b|c)
and so we obtain the conditional independence property
a ⊥⊥b | c.
We can provide a simple graphical interpretation of this result by considering
the path from node a to node b via c. The node c is said to be tail-to-tail with re-
spect to this path because the node is connected to the tails of the two arrows, and
the presence of such a path connecting nodes a and b causes these nodes to be de-
pendent. However, when we condition on node c, as in Figure 8.16, the conditioned
node ‘blocks’ the path from a to b and causes a and b to become (conditionally)
independent.
We can similarly consider the graph shown in Figure 8.17. The joint distribution
corresponding to this graph is again obtained from our general formula (8.5) to give
p(a, b, c) = p(a)p(c|a)p(b|c).
(8.26)
First of all, suppose that none of the variables are observed. Again, we can test to
see if a and b are independent by marginalizing over c to give
p(a, b) = p(a)

c
p(c|a)p(b|c) = p(a)p(b|a).
Figure 8.17
The second of our three examples of 3-node
graphs used to motivate the conditional indepen-
dence framework for directed graphical models.
a
c
b


---
**Page 391**
8.2. Conditional Independence
375
Figure 8.18
As in Figure 8.17 but now conditioning on node c.
a
c
b
which in general does not factorize into p(a)p(b), and so
a ̸⊥⊥b | ∅
(8.27)
as before.
Now suppose we condition on node c, as shown in Figure 8.18. Using Bayes’
theorem, together with (8.26), we obtain
p(a, b|c)
=
p(a, b, c)
p(c)
=
p(a)p(c|a)p(b|c)
p(c)
=
p(a|c)p(b|c)
and so again we obtain the conditional independence property
a ⊥⊥b | c.
As before, we can interpret these results graphically. The node c is said to be
head-to-tail with respect to the path from node a to node b. Such a path connects
nodes a and b and renders them dependent. If we now observe c, as in Figure 8.18,
then this observation ‘blocks’ the path from a to b and so we obtain the conditional
independence property a ⊥⊥b | c.
Finally, we consider the third of our 3-node examples, shown by the graph in
Figure 8.19. As we shall see, this has a more subtle behaviour than the two previous
graphs.
The joint distribution can again be written down using our general result (8.5) to
give
p(a, b, c) = p(a)p(b)p(c|a, b).
(8.28)
Consider ﬁrst the case where none of the variables are observed. Marginalizing both
sides of (8.28) over c we obtain
p(a, b) = p(a)p(b)
Figure 8.19
The last of our three examples of 3-node graphs used to
explore conditional independence properties in graphi-
cal models. This graph has rather different properties
from the two previous examples.
c
a
b


---
**Page 392**
376
8. GRAPHICAL MODELS
Figure 8.20
As in Figure 8.19 but conditioning on the value of node
c. In this graph, the act of conditioning induces a depen-
dence between a and b.
c
a
b
and so a and b are independent with no variables observed, in contrast to the two
previous examples. We can write this result as
a ⊥⊥b | ∅.
(8.29)
Now suppose we condition on c, as indicated in Figure 8.20. The conditional distri-
bution of a and b is then given by
p(a, b|c)
=
p(a, b, c)
p(c)
=
p(a)p(b)p(c|a, b)
p(c)
which in general does not factorize into the product p(a)p(b), and so
a ̸⊥⊥b | c.
Thus our third example has the opposite behaviour from the ﬁrst two. Graphically,
we say that node c is head-to-head with respect to the path from a to b because it
connects to the heads of the two arrows. When node c is unobserved, it ‘blocks’
the path, and the variables a and b are independent. However, conditioning on c
‘unblocks’ the path and renders a and b dependent.
There is one more subtlety associated with this third example that we need to
consider. First we introduce some more terminology. We say that node y is a de-
scendant of node x if there is a path from x to y in which each step of the path
follows the directions of the arrows. Then it can be shown that a head-to-head path
will become unblocked if either the node, or any of its descendants, is observed.
Exercise 8.10
In summary, a tail-to-tail node or a head-to-tail node leaves a path unblocked
unless it is observed in which case it blocks the path. By contrast, a head-to-head
node blocks a path if it is unobserved, but once the node, and/or at least one of its
descendants, is observed the path becomes unblocked.
It is worth spending a moment to understand further the unusual behaviour of the
graph of Figure 8.20. Consider a particular instance of such a graph corresponding
to a problem with three binary random variables relating to the fuel system on a car,
as shown in Figure 8.21.
The variables are called B, representing the state of a
battery that is either charged (B = 1) or ﬂat (B = 0), F representing the state of
the fuel tank that is either full of fuel (F = 1) or empty (F = 0), and G, which is
the state of an electric fuel gauge and which indicates either full (G = 1) or empty


---
**Page 393**
8.2. Conditional Independence
377
G
B
F
G
B
F
G
B
F
Figure 8.21
An example of a 3-node graph used to illustrate the phenomenon of ‘explaining away’. The three
nodes represent the state of the battery (B), the state of the fuel tank (F) and the reading on the electric fuel
gauge (G). See the text for details.
(G = 0). The battery is either charged or ﬂat, and independently the fuel tank is
either full or empty, with prior probabilities
p(B = 1)
=
0.9
p(F = 1)
=
0.9.
Given the state of the fuel tank and the battery, the fuel gauge reads full with proba-
bilities given by
p(G = 1|B = 1, F = 1)
=
0.8
p(G = 1|B = 1, F = 0)
=
0.2
p(G = 1|B = 0, F = 1)
=
0.2
p(G = 1|B = 0, F = 0)
=
0.1
so this is a rather unreliable fuel gauge! All remaining probabilities are determined
by the requirement that probabilities sum to one, and so we have a complete speciﬁ-
cation of the probabilistic model.
Before we observe any data, the prior probability of the fuel tank being empty
is p(F = 0) = 0.1. Now suppose that we observe the fuel gauge and discover that
it reads empty, i.e., G = 0, corresponding to the middle graph in Figure 8.21. We
can use Bayes’ theorem to evaluate the posterior probability of the fuel tank being
empty. First we evaluate the denominator for Bayes’ theorem given by
p(G = 0) =

B∈{0,1}

F ∈{0,1}
p(G = 0|B, F)p(B)p(F) = 0.315
(8.30)
and similarly we evaluate
p(G = 0|F = 0) =

B∈{0,1}
p(G = 0|B, F = 0)p(B) = 0.81
(8.31)
and using these results we have
p(F = 0|G = 0) = p(G = 0|F = 0)p(F = 0)
p(G = 0)
≃0.257
(8.32)


---
**Page 394**
378
8. GRAPHICAL MODELS
and so p(F = 0|G = 0) > p(F = 0). Thus observing that the gauge reads empty
makes it more likely that the tank is indeed empty, as we would intuitively expect.
Next suppose that we also check the state of the battery and ﬁnd that it is ﬂat, i.e.,
B = 0. We have now observed the states of both the fuel gauge and the battery, as
shown by the right-hand graph in Figure 8.21. The posterior probability that the fuel
tank is empty given the observations of both the fuel gauge and the battery state is
then given by
p(F = 0|G = 0, B = 0) = p(G = 0|B = 0, F = 0)p(F = 0)

F ∈{0,1} p(G = 0|B = 0, F)p(F) ≃0.111
(8.33)
where the prior probability p(B = 0) has cancelled between numerator and denom-
inator. Thus the probability that the tank is empty has decreased (from 0.257 to
0.111) as a result of the observation of the state of the battery. This accords with our
intuition that ﬁnding out that the battery is ﬂat explains away the observation that the
fuel gauge reads empty. We see that the state of the fuel tank and that of the battery
have indeed become dependent on each other as a result of observing the reading
on the fuel gauge. In fact, this would also be the case if, instead of observing the
fuel gauge directly, we observed the state of some descendant of G. Note that the
probability p(F = 0|G = 0, B = 0) ≃0.111 is greater than the prior probability
p(F = 0) = 0.1 because the observation that the fuel gauge reads zero still provides
some evidence in favour of an empty fuel tank.
8.2.2
D-separation
We now give a general statement of the d-separation property (Pearl, 1988) for
directed graphs. Consider a general directed graph in which A, B, and C are arbi-
trary nonintersecting sets of nodes (whose union may be smaller than the complete
set of nodes in the graph). We wish to ascertain whether a particular conditional
independence statement A ⊥⊥B | C is implied by a given directed acyclic graph. To
do so, we consider all possible paths from any node in A to any node in B. Any such
path is said to be blocked if it includes a node such that either
(a) the arrows on the path meet either head-to-tail or tail-to-tail at the node, and the
node is in the set C, or
(b) the arrows meet head-to-head at the node, and neither the node, nor any of its
descendants, is in the set C.
If all paths are blocked, then A is said to be d-separated from B by C, and the joint
distribution over all of the variables in the graph will satisfy A ⊥⊥B | C.
The concept of d-separation is illustrated in Figure 8.22. In graph (a), the path
from a to b is not blocked by node f because it is a tail-to-tail node for this path
and is not observed, nor is it blocked by node e because, although the latter is a
head-to-head node, it has a descendant c because is in the conditioning set. Thus
the conditional independence statement a ⊥⊥b | c does not follow from this graph.
In graph (b), the path from a to b is blocked by node f because this is a tail-to-tail
node that is observed, and so the conditional independence property a ⊥⊥b | f will


---
**Page 395**
8.2. Conditional Independence
379
Figure 8.22
Illustration of the con-
cept of d-separation. See the text for
details.
f
e
b
a
c
(a)
f
e
b
a
c
(b)
be satisﬁed by any distribution that factorizes according to this graph. Note that this
path is also blocked by node e because e is a head-to-head node and neither it nor its
descendant are in the conditioning set.
For the purposes of d-separation, parameters such as α and σ2 in Figure 8.5,
indicated by small ﬁlled circles, behave in the same was as observed nodes. How-
ever, there are no marginal distributions associated with such nodes. Consequently
parameter nodes never themselves have parents and so all paths through these nodes
will always be tail-to-tail and hence blocked. Consequently they play no role in
d-separation.
Another example of conditional independence and d-separation is provided by
the concept of i.i.d. (independent identically distributed) data introduced in Sec-
tion 1.2.4. Consider the problem of ﬁnding the posterior distribution for the mean
of a univariate Gaussian distribution. This can be represented by the directed graph
Section 2.3
shown in Figure 8.23 in which the joint distribution is deﬁned by a prior p(µ) to-
gether with a set of conditional distributions p(xn|µ) for n = 1, . . . , N. In practice,
we observe D = {x1, . . . , xN} and our goal is to infer µ. Suppose, for a moment,
that we condition on µ and consider the joint distribution of the observations. Using
d-separation, we note that there is a unique path from any xi to any other xj̸=i and
that this path is tail-to-tail with respect to the observed node µ. Every such path is
blocked and so the observations D = {x1, . . . , xN} are independent given µ, so that
p(D|µ) =
N

n=1
p(xn|µ).
(8.34)
Figure 8.23
(a)
Directed
graph
corre-
sponding
to
the
problem
of inferring the mean µ of
a univariate Gaussian dis-
tribution from observations
x1, . . . , xN.
(b) The same
graph drawn using the plate
notation.
µ
x1
xN
(a)
xn
N
N
µ
(b)


---
**Page 396**
380
8. GRAPHICAL MODELS
Figure 8.24
A graphical representation of the ‘naive Bayes’
model for classiﬁcation.
Conditioned on the
class label z, the components of the observed
vector x = (x1, . . . , xD)T are assumed to be
independent.
z
x1
xD
However, if we integrate over µ, the observations are in general no longer indepen-
dent
p(D) =
 ∞
0
p(D|µ)p(µ) dµ ̸=
N

n=1
p(xn).
(8.35)
Here µ is a latent variable, because its value is not observed.
Another example of a model representing i.i.d. data is the graph in Figure 8.7
corresponding to Bayesian polynomial regression. Here the stochastic nodes corre-
spond to {tn}, w and t. We see that the node for w is tail-to-tail with respect to
the path fromt to any one of the nodes tn and so we have the following conditional
independence property
t ⊥⊥tn | w.
(8.36)
Thus, conditioned on the polynomial coefﬁcients w, the predictive distribution for
t is independent of the training data {t1, . . . , tN}. We can therefore ﬁrst use the
training data to determine the posterior distribution over the coefﬁcients w and then
we can discard the training data and use the posterior distribution for w to make
predictions oft for new input observations x.
Section 3.3
A related graphical structure arises in an approach to classiﬁcation called the
naive Bayes model, in which we use conditional independence assumptions to sim-
plify the model structure. Suppose our observed variable consists of a D-dimensional
vector x = (x1, . . . , xD)T, and we wish to assign observed values of x to one of K
classes. Using the 1-of-K encoding scheme, we can represent these classes by a K-
dimensional binary vector z. We can then deﬁne a generative model by introducing
a multinomial prior p(z|µ) over the class labels, where the kth component µk of µ
is the prior probability of class Ck, together with a conditional distribution p(x|z)
for the observed vector x. The key assumption of the naive Bayes model is that,
conditioned on the class z, the distributions of the input variables x1, . . . , xD are in-
dependent. The graphical representation of this model is shown in Figure 8.24. We
see that observation of z blocks the path between xi and xj for j ̸= i (because such
paths are tail-to-tail at the node z) and so xi and xj are conditionally independent
given z. If, however, we marginalize out z (so that z is unobserved) the tail-to-tail
path from xi to xj is no longer blocked. This tells us that in general the marginal
density p(x) will not factorize with respect to the components of x. We encountered
a simple application of the naive Bayes model in the context of fusing data from
different sources for medical diagnosis in Section 1.5.
If we are given a labelled training set, comprising inputs {x1, . . . , xN} together
with their class labels, then we can ﬁt the naive Bayes model to the training data


---
**Page 397**
8.2. Conditional Independence
381
using maximum likelihood assuming that the data are drawn independently from
the model. The solution is obtained by ﬁtting the model for each class separately
using the correspondingly labelled data. As an example, suppose that the probability
density within each class is chosen to be Gaussian. In this case, the naive Bayes
assumption then implies that the covariance matrix for each Gaussian is diagonal,
and the contours of constant density within each class will be axis-aligned ellipsoids.
The marginal density, however, is given by a superposition of diagonal Gaussians
(with weighting coefﬁcients given by the class priors) and so will no longer factorize
with respect to its components.
The naive Bayes assumption is helpful when the dimensionality D of the input
space is high, making density estimation in the full D-dimensional space more chal-
lenging. It is also useful if the input vector contains both discrete and continuous
variables, since each can be represented separately using appropriate models (e.g.,
Bernoulli distributions for binary observations or Gaussians for real-valued vari-
ables). The conditional independence assumption of this model is clearly a strong
one that may lead to rather poor representations of the class-conditional densities.
Nevertheless, even if this assumption is not precisely satisﬁed, the model may still
give good classiﬁcation performance in practice because the decision boundaries can
be insensitive to some of the details in the class-conditional densities, as illustrated
in Figure 1.27.
We have seen that a particular directed graph represents a speciﬁc decomposition
of a joint probability distribution into a product of conditional probabilities. The
graph also expresses a set of conditional independence statements obtained through
the d-separation criterion, and the d-separation theorem is really an expression of the
equivalence of these two properties. In order to make this clear, it is helpful to think
of a directed graph as a ﬁlter. Suppose we consider a particular joint probability
distribution p(x) over the variables x corresponding to the (nonobserved) nodes of
the graph. The ﬁlter will allow this distribution to pass through if, and only if, it can
be expressed in terms of the factorization (8.5) implied by the graph. If we present to
the ﬁlter the set of all possible distributions p(x) over the set of variables x, then the
subset of distributions that are passed by the ﬁlter will be denoted DF, for directed
factorization. This is illustrated in Figure 8.25. Alternatively, we can use the graph as
a different kind of ﬁlter by ﬁrst listing all of the conditional independence properties
obtained by applying the d-separation criterion to the graph, and then allowing a
distribution to pass only if it satisﬁes all of these properties. If we present all possible
distributions p(x) to this second kind of ﬁlter, then the d-separation theorem tells us
that the set of distributions that will be allowed through is precisely the set DF.
It should be emphasized that the conditional independence properties obtained
from d-separation apply to any probabilistic model described by that particular di-
rected graph. This will be true, for instance, whether the variables are discrete or
continuous or a combination of these. Again, we see that a particular graph is de-
scribing a whole family of probability distributions.
At one extreme we have a fully connected graph that exhibits no conditional in-
dependence properties at all, and which can represent any possible joint probability
distribution over the given variables. The set DF will contain all possible distribu-


---
**Page 398**
382
8. GRAPHICAL MODELS
p(x)
DF
Figure 8.25
We can view a graphical model (in this case a directed graph) as a ﬁlter in which a prob-
ability distribution p(x) is allowed through the ﬁlter if, and only if, it satisﬁes the directed
factorization property (8.5). The set of all possible probability distributions p(x) that pass
through the ﬁlter is denoted DF. We can alternatively use the graph to ﬁlter distributions
according to whether they respect all of the conditional independencies implied by the
d-separation properties of the graph. The d-separation theorem says that it is the same
set of distributions DF that will be allowed through this second kind of ﬁlter.
tions p(x). At the other extreme, we have the fully disconnected graph, i.e., one
having no links at all. This corresponds to joint distributions which factorize into the
product of the marginal distributions over the variables comprising the nodes of the
graph.
Note that for any given graph, the set of distributions DF will include any dis-
tributions that have additional independence properties beyond those described by
the graph. For instance, a fully factorized distribution will always be passed through
the ﬁlter implied by any graph over the corresponding set of variables.
We end our discussion of conditional independence properties by exploring the
concept of a Markov blanket or Markov boundary. Consider a joint distribution
p(x1, . . . , xD) represented by a directed graph having D nodes, and consider the
conditional distribution of a particular node with variables xi conditioned on all of
the remaining variables xj̸=i. Using the factorization property (8.5), we can express
this conditional distribution in the form
p(xi|x{j̸=i})
=
p(x1, . . . , xD)

p(x1, . . . , xD) dxi
=

k
p(xk|pak)
 
k
p(xk|pak) dxi
in which the integral is replaced by a summation in the case of discrete variables. We
now observe that any factor p(xk|pak) that does not have any functional dependence
on xi can be taken outside the integral over xi, and will therefore cancel between
numerator and denominator. The only factors that remain will be the conditional
distribution p(xi|pai) for node xi itself, together with the conditional distributions
for any nodes xk such that node xi is in the conditioning set of p(xk|pak), in other
words for which xi is a parent of xk. The conditional p(xi|pai) will depend on the
parents of node xi, whereas the conditionals p(xk|pak) will depend on the children


---
**Page 399**
8.3. Markov Random Fields
383
Figure 8.26
The Markov blanket of a node xi comprises the set
of parents, children and co-parents of the node. It
has the property that the conditional distribution of
xi, conditioned on all the remaining variables in the
graph, is dependent only on the variables in the
Markov blanket.
xi
of xi as well as on the co-parents, in other words variables corresponding to parents
of node xk other than node xi. The set of nodes comprising the parents, the children
and the co-parents is called the Markov blanket and is illustrated in Figure 8.26. We
can think of the Markov blanket of a node xi as being the minimal set of nodes that
isolates xi from the rest of the graph. Note that it is not sufﬁcient to include only the
parents and children of node xi because the phenomenon of explaining away means
that observations of the child nodes will not block paths to the co-parents. We must
therefore observe the co-parent nodes also.
8.3. Markov Random Fields
We have seen that directed graphical models specify a factorization of the joint dis-
tribution over a set of variables into a product of local conditional distributions. They
also deﬁne a set of conditional independence properties that must be satisﬁed by any
distribution that factorizes according to the graph. We turn now to the second ma-
jor class of graphical models that are described by undirected graphs and that again
specify both a factorization and a set of conditional independence relations.
A Markov random ﬁeld, also known as a Markov network or an undirected
graphical model (Kindermann and Snell, 1980), has a set of nodes each of which
corresponds to a variable or group of variables, as well as a set of links each of
which connects a pair of nodes. The links are undirected, that is they do not carry
arrows. In the case of undirected graphs, it is convenient to begin with a discussion
of conditional independence properties.
8.3.1
Conditional independence properties
In the case of directed graphs, we saw that it was possible to test whether a par-
Section 8.2
ticular conditional independence property holds by applying a graphical test called
d-separation. This involved testing whether or not the paths connecting two sets of
nodes were ‘blocked’. The deﬁnition of blocked, however, was somewhat subtle
due to the presence of paths having head-to-head nodes. We might ask whether it
is possible to deﬁne an alternative graphical semantics for probability distributions
such that conditional independence is determined by simple graph separation. This
is indeed the case and corresponds to undirected graphical models. By removing the


---
**Page 400**
384
8. GRAPHICAL MODELS
Figure 8.27
An example of an undirected graph in
which every path from any node in set
A to any node in set B passes through
at least one node in set C.
Conse-
quently the conditional independence
property A ⊥⊥B | C holds for any
probability distribution described by this
graph.
A
C
B
directionality from the links of the graph, the asymmetry between parent and child
nodes is removed, and so the subtleties associated with head-to-head nodes no longer
arise.
Suppose that in an undirected graph we identify three sets of nodes, denoted A,
B, and C, and that we consider the conditional independence property
A ⊥⊥B | C.
(8.37)
To test whether this property is satisﬁed by a probability distribution deﬁned by a
graph we consider all possible paths that connect nodes in set A to nodes in set
B. If all such paths pass through one or more nodes in set C, then all such paths are
‘blocked’ and so the conditional independence property holds. However, if there is at
least one such path that is not blocked, then the property does not necessarily hold, or
more precisely there will exist at least some distributions corresponding to the graph
that do not satisfy this conditional independence relation. This is illustrated with an
example in Figure 8.27. Note that this is exactly the same as the d-separation crite-
rion except that there is no ‘explaining away’ phenomenon. Testing for conditional
independence in undirected graphs is therefore simpler than in directed graphs.
An alternative way to view the conditional independence test is to imagine re-
moving all nodes in set C from the graph together with any links that connect to
those nodes. We then ask if there exists a path that connects any node in A to any
node in B. If there are no such paths, then the conditional independence property
must hold.
The Markov blanket for an undirected graph takes a particularly simple form,
because a node will be conditionally independent of all other nodes conditioned only
on the neighbouring nodes, as illustrated in Figure 8.28.
8.3.2
Factorization properties
We now seek a factorization rule for undirected graphs that will correspond to
the above conditional independence test. Again, this will involve expressing the joint
distribution p(x) as a product of functions deﬁned over sets of variables that are local
to the graph. We therefore need to decide what is the appropriate notion of locality
in this case.


---
**Page 401**
8.3. Markov Random Fields
385
Figure 8.28
For an undirected graph, the Markov blanket of a node
xi consists of the set of neighbouring nodes. It has the
property that the conditional distribution of xi, conditioned
on all the remaining variables in the graph, is dependent
only on the variables in the Markov blanket.
If we consider two nodes xi and xj that are not connected by a link, then these
variables must be conditionally independent given all other nodes in the graph. This
follows from the fact that there is no direct path between the two nodes, and all other
paths pass through nodes that are observed, and hence those paths are blocked. This
conditional independence property can be expressed as
p(xi, xj|x\{i,j}) = p(xi|x\{i,j})p(xj|x\{i,j})
(8.38)
where x\{i,j} denotes the set x of all variables with xi and xj removed. The factor-
ization of the joint distribution must therefore be such that xi and xj do not appear
in the same factor in order for the conditional independence property to hold for all
possible distributions belonging to the graph.
This leads us to consider a graphical concept called a clique, which is deﬁned
as a subset of the nodes in a graph such that there exists a link between all pairs of
nodes in the subset. In other words, the set of nodes in a clique is fully connected.
Furthermore, a maximal clique is a clique such that it is not possible to include any
other nodes from the graph in the set without it ceasing to be a clique. These concepts
are illustrated by the undirected graph over four variables shown in Figure 8.29. This
graph has ﬁve cliques of two nodes given by {x1, x2}, {x2, x3}, {x3, x4}, {x4, x2},
and {x1, x3}, as well as two maximal cliques given by {x1, x2, x3} and {x2, x3, x4}.
The set {x1, x2, x3, x4} is not a clique because of the missing link from x1 to x4.
We can therefore deﬁne the factors in the decomposition of the joint distribution
to be functions of the variables in the cliques. In fact, we can consider functions
of the maximal cliques, without loss of generality, because other cliques must be
subsets of maximal cliques. Thus, if {x1, x2, x3} is a maximal clique and we deﬁne
an arbitrary function over this clique, then including another factor deﬁned over a
subset of these variables would be redundant.
Let us denote a clique by C and the set of variables in that clique by xC. Then
Figure 8.29
A four-node undirected graph showing a clique (outlined in
green) and a maximal clique (outlined in blue).
x1
x2
x3
x4


---
**Page 402**
386
8. GRAPHICAL MODELS
the joint distribution is written as a product of potential functions ψC(xC) over the
maximal cliques of the graph
p(x) = 1
Z

C
ψC(xC).
(8.39)
Here the quantity Z, sometimes called the partition function, is a normalization con-
stant and is given by
Z =

x

C
ψC(xC)
(8.40)
which ensures that the distribution p(x) given by (8.39) is correctly normalized.
By considering only potential functions which satisfy ψC(xC) ⩾0 we ensure that
p(x) ⩾0. In (8.40) we have assumed that x comprises discrete variables, but the
framework is equally applicable to continuous variables, or a combination of the two,
in which the summation is replaced by the appropriate combination of summation
and integration.
Note that we do not restrict the choice of potential functions to those that have a
speciﬁc probabilistic interpretation as marginal or conditional distributions. This is
in contrast to directed graphs in which each factor represents the conditional distribu-
tion of the corresponding variable, conditioned on the state of its parents. However,
in special cases, for instance where the undirected graph is constructed by starting
with a directed graph, the potential functions may indeed have such an interpretation,
as we shall see shortly.
One consequence of the generality of the potential functions ψC(xC) is that
their product will in general not be correctly normalized. We therefore have to in-
troduce an explicit normalization factor given by (8.40). Recall that for directed
graphs, the joint distribution was automatically normalized as a consequence of the
normalization of each of the conditional distributions in the factorization.
The presence of this normalization constant is one of the major limitations of
undirected graphs. If we have a model with M discrete nodes each having K states,
then the evaluation of the normalization term involves summing over KM states and
so (in the worst case) is exponential in the size of the model. The partition function
is needed for parameter learning because it will be a function of any parameters that
govern the potential functions ψC(xC). However, for evaluation of local conditional
distributions, the partition function is not needed because a conditional is the ratio
of two marginals, and the partition function cancels between numerator and denom-
inator when evaluating this ratio. Similarly, for evaluating local marginal probabil-
ities we can work with the unnormalized joint distribution and then normalize the
marginals explicitly at the end. Provided the marginals only involves a small number
of variables, the evaluation of their normalization coefﬁcient will be feasible.
So far, we have discussed the notion of conditional independence based on sim-
ple graph separation and we have proposed a factorization of the joint distribution
that is intended to correspond to this conditional independence structure. However,
we have not made any formal connection between conditional independence and
factorization for undirected graphs. To do so we need to restrict attention to poten-
tial functions ψC(xC) that are strictly positive (i.e., never zero or negative for any


---
**Page 403**
8.3. Markov Random Fields
387
choice of xC). Given this restriction, we can make a precise relationship between
factorization and conditional independence.
To do this we again return to the concept of a graphical model as a ﬁlter, corre-
sponding to Figure 8.25. Consider the set of all possible distributions deﬁned over
a ﬁxed set of variables corresponding to the nodes of a particular undirected graph.
We can deﬁne UI to be the set of such distributions that are consistent with the set
of conditional independence statements that can be read from the graph using graph
separation. Similarly, we can deﬁne UF to be the set of such distributions that can
be expressed as a factorization of the form (8.39) with respect to the maximal cliques
of the graph. The Hammersley-Clifford theorem (Clifford, 1990) states that the sets
UI and UF are identical.
Because we are restricted to potential functions which are strictly positive it is
convenient to express them as exponentials, so that
ψC(xC) = exp {−E(xC)}
(8.41)
where E(xC) is called an energy function, and the exponential representation is
called the Boltzmann distribution. The joint distribution is deﬁned as the product of
potentials, and so the total energy is obtained by adding the energies of each of the
maximal cliques.
In contrast to the factors in the joint distribution for a directed graph, the po-
tentials in an undirected graph do not have a speciﬁc probabilistic interpretation.
Although this gives greater ﬂexibility in choosing the potential functions, because
there is no normalization constraint, it does raise the question of how to motivate a
choice of potential function for a particular application. This can be done by view-
ing the potential function as expressing which conﬁgurations of the local variables
are preferred to others. Global conﬁgurations that have a relatively high probability
are those that ﬁnd a good balance in satisfying the (possibly conﬂicting) inﬂuences
of the clique potentials. We turn now to a speciﬁc example to illustrate the use of
undirected graphs.
8.3.3
Illustration: Image de-noising
We can illustrate the application of undirected graphs using an example of noise
removal from a binary image (Besag, 1974; Geman and Geman, 1984; Besag, 1986).
Although a very simple example, this is typical of more sophisticated applications.
Let the observed noisy image be described by an array of binary pixel values yi ∈
{−1, +1}, where the index i = 1, . . . , D runs over all pixels. We shall suppose
that the image is obtained by taking an unknown noise-free image, described by
binary pixel values xi ∈{−1, +1} and randomly ﬂipping the sign of pixels with
some small probability. An example binary image, together with a noise corrupted
image obtained by ﬂipping the sign of the pixels with probability 10%, is shown in
Figure 8.30. Given the noisy image, our goal is to recover the original noise-free
image.
Because the noise level is small, we know that there will be a strong correlation
between xi and yi. We also know that neighbouring pixels xi and xj in an image
are strongly correlated. This prior knowledge can be captured using the Markov


---
**Page 404**
388
8. GRAPHICAL MODELS
Figure 8.30
Illustration of image de-noising using a Markov random ﬁeld. The top row shows the original
binary image on the left and the corrupted image after randomly changing 10% of the pixels on the right. The
bottom row shows the restored images obtained using iterated conditional models (ICM) on the left and using
the graph-cut algorithm on the right. ICM produces an image where 96% of the pixels agree with the original
image, whereas the corresponding number for graph-cut is 99%.
random ﬁeld model whose undirected graph is shown in Figure 8.31. This graph has
two types of cliques, each of which contains two variables. The cliques of the form
{xi, yi} have an associated energy function that expresses the correlation between
these variables. We choose a very simple energy function for these cliques of the
form −ηxiyi where η is a positive constant. This has the desired effect of giving a
lower energy (thus encouraging a higher probability) when xi and yi have the same
sign and a higher energy when they have the opposite sign.
The remaining cliques comprise pairs of variables {xi, xj} where i and j are
indices of neighbouring pixels. Again, we want the energy to be lower when the
pixels have the same sign than when they have the opposite sign, and so we choose
an energy given by −βxixj where β is a positive constant.
Because a potential function is an arbitrary, nonnegative function over a maximal
clique, we can multiply it by any nonnegative functions of subsets of the clique, or


---
**Page 405**
8.3. Markov Random Fields
389
Figure 8.31
An undirected graphical model representing a
Markov random ﬁeld for image de-noising, in
which xi is a binary variable denoting the state
of pixel i in the unknown noise-free image, and yi
denotes the corresponding value of pixel i in the
observed noisy image.
xi
yi
equivalently we can add the corresponding energies. In this example, this allows us
to add an extra term hxi for each pixel i in the noise-free image. Such a term has
the effect of biasing the model towards pixel values that have one particular sign in
preference to the other.
The complete energy function for the model then takes the form
E(x, y) = h

i
xi −β

{i,j}
xixj −η

i
xiyi
(8.42)
which deﬁnes a joint distribution over x and y given by
p(x, y) = 1
Z exp{−E(x, y)}.
(8.43)
We now ﬁx the elements of y to the observed values given by the pixels of the
noisy image, which implicitly deﬁnes a conditional distribution p(x|y) over noise-
free images. This is an example of the Ising model, which has been widely studied in
statistical physics. For the purposes of image restoration, we wish to ﬁnd an image x
having a high probability (ideally the maximum probability). To do this we shall use
a simple iterative technique called iterated conditional modes, or ICM (Kittler and
F¨oglein, 1984), which is simply an application of coordinate-wise gradient ascent.
The idea is ﬁrst to initialize the variables {xi}, which we do by simply setting xi =
yi for all i. Then we take one node xj at a time and we evaluate the total energy
for the two possible states xj = +1 and xj = −1, keeping all other node variables
ﬁxed, and set xj to whichever state has the lower energy. This will either leave
the probability unchanged, if xj is unchanged, or will increase it. Because only
one variable is changed, this is a simple local computation that can be performed
Exercise 8.13
efﬁciently. We then repeat the update for another site, and so on, until some suitable
stopping criterion is satisﬁed. The nodes may be updated in a systematic way, for
instance by repeatedly raster scanning through the image, or by choosing nodes at
random.
If we have a sequence of updates in which every site is visited at least once,
and in which no changes to the variables are made, then by deﬁnition the algorithm


---
**Page 406**
390
8. GRAPHICAL MODELS
Figure 8.32
(a) Example of a directed
graph.
(b) The equivalent undirected
graph.
(a)
x1
x2
xN−1
xN
(b)
x1
x2
xN−1
xN
will have converged to a local maximum of the probability. This need not, however,
correspond to the global maximum.
For the purposes of this simple illustration, we have ﬁxed the parameters to be
β = 1.0, η = 2.1 and h = 0. Note that leaving h = 0 simply means that the prior
probabilities of the two states of xi are equal. Starting with the observed noisy image
as the initial conﬁguration, we run ICM until convergence, leading to the de-noised
image shown in the lower left panel of Figure 8.30. Note that if we set β = 0,
which effectively removes the links between neighbouring pixels, then the global
most probable solution is given by xi = yi for all i, corresponding to the observed
noisy image.
Exercise 8.14
Later we shall discuss a more effective algorithm for ﬁnding high probability so-
lutions called the max-product algorithm, which typically leads to better solutions,
Section 8.4
although this is still not guaranteed to ﬁnd the global maximum of the posterior dis-
tribution. However, for certain classes of model, including the one given by (8.42),
there exist efﬁcient algorithms based on graph cuts that are guaranteed to ﬁnd the
global maximum (Greig et al., 1989; Boykov et al., 2001; Kolmogorov and Zabih,
2004). The lower right panel of Figure 8.30 shows the result of applying a graph-cut
algorithm to the de-noising problem.
8.3.4
Relation to directed graphs
We have introduced two graphical frameworks for representing probability dis-
tributions, corresponding to directed and undirected graphs, and it is instructive to
discuss the relation between these. Consider ﬁrst the problem of taking a model that
is speciﬁed using a directed graph and trying to convert it to an undirected graph. In
some cases this is straightforward, as in the simple example in Figure 8.32. Here the
joint distribution for the directed graph is given as a product of conditionals in the
form
p(x) = p(x1)p(x2|x1)p(x3|x2) · · · p(xN|xN−1).
(8.44)
Now let us convert this to an undirected graph representation, as shown in Fig-
ure 8.32. In the undirected graph, the maximal cliques are simply the pairs of neigh-
bouring nodes, and so from (8.39) we wish to write the joint distribution in the form
p(x) = 1
Z ψ1,2(x1, x2)ψ2,3(x2, x3) · · · ψN−1,N(xN−1, xN).
(8.45)


---
**Page 407**
8.3. Markov Random Fields
391
Figure 8.33
Example of a simple
directed graph (a) and the corre-
sponding moral graph (b).
x1
x3
x4
x2
(a)
x1
x3
x4
x2
(b)
This is easily done by identifying
ψ1,2(x1, x2)
=
p(x1)p(x2|x1)
ψ2,3(x2, x3)
=
p(x3|x2)
...
ψN−1,N(xN−1, xN)
=
p(xN|xN−1)
where we have absorbed the marginal p(x1) for the ﬁrst node into the ﬁrst potential
function. Note that in this case, the partition function Z = 1.
Let us consider how to generalize this construction, so that we can convert any
distribution speciﬁed by a factorization over a directed graph into one speciﬁed by a
factorization over an undirected graph. This can be achieved if the clique potentials
of the undirected graph are given by the conditional distributions of the directed
graph. In order for this to be valid, we must ensure that the set of variables that
appears in each of the conditional distributions is a member of at least one clique of
the undirected graph. For nodes on the directed graph having just one parent, this is
achieved simply by replacing the directed link with an undirected link. However, for
nodes in the directed graph having more than one parent, this is not sufﬁcient. These
are nodes that have ‘head-to-head’ paths encountered in our discussion of conditional
independence. Consider a simple directed graph over 4 nodes shown in Figure 8.33.
The joint distribution for the directed graph takes the form
p(x) = p(x1)p(x2)p(x3)p(x4|x1, x2, x3).
(8.46)
We see that the factor p(x4|x1, x2, x3) involves the four variables x1, x2, x3, and
x4, and so these must all belong to a single clique if this conditional distribution is
to be absorbed into a clique potential. To ensure this, we add extra links between
all pairs of parents of the node x4. Anachronistically, this process of ‘marrying
the parents’ has become known as moralization, and the resulting undirected graph,
after dropping the arrows, is called the moral graph. It is important to observe that
the moral graph in this example is fully connected and so exhibits no conditional
independence properties, in contrast to the original directed graph.
Thus in general to convert a directed graph into an undirected graph, we ﬁrst add
additional undirected links between all pairs of parents for each node in the graph and


---
**Page 408**
392
8. GRAPHICAL MODELS
then drop the arrows on the original links to give the moral graph. Then we initialize
all of the clique potentials of the moral graph to 1. We then take each conditional
distribution factor in the original directed graph and multiply it into one of the clique
potentials. There will always exist at least one maximal clique that contains all of
the variables in the factor as a result of the moralization step. Note that in all cases
the partition function is given by Z = 1.
The process of converting a directed graph into an undirected graph plays an
important role in exact inference techniques such as the junction tree algorithm.
Section 8.4
Converting from an undirected to a directed representation is much less common
and in general presents problems due to the normalization constraints.
We saw that in going from a directed to an undirected representation we had to
discard some conditional independence properties from the graph. Of course, we
could always trivially convert any distribution over a directed graph into one over an
undirected graph by simply using a fully connected undirected graph. This would,
however, discard all conditional independence properties and so would be vacuous.
The process of moralization adds the fewest extra links and so retains the maximum
number of independence properties.
We have seen that the procedure for determining the conditional independence
properties is different between directed and undirected graphs. It turns out that the
two types of graph can express different conditional independence properties, and
it is worth exploring this issue in more detail. To do so, we return to the view of
a speciﬁc (directed or undirected) graph as a ﬁlter, so that the set of all possible
Section 8.2
distributions over the given variables could be reduced to a subset that respects the
conditional independencies implied by the graph. A graph is said to be a D map
(for ‘dependency map’) of a distribution if every conditional independence statement
satisﬁed by the distribution is reﬂected in the graph. Thus a completely disconnected
graph (no links) will be a trivial D map for any distribution.
Alternatively, we can consider a speciﬁc distribution and ask which graphs have
the appropriate conditional independence properties. If every conditional indepen-
dence statement implied by a graph is satisﬁed by a speciﬁc distribution, then the
graph is said to be an I map (for ‘independence map’) of that distribution. Clearly a
fully connected graph will be a trivial I map for any distribution.
If it is the case that every conditional independence property of the distribution
is reﬂected in the graph, and vice versa, then the graph is said to be a perfect map for
Figure 8.34
Venn diagram illustrating the set of all distributions
P over a given set of variables, together with the
set of distributions D that can be represented as a
perfect map using a directed graph, and the set U
that can be represented as a perfect map using an
undirected graph.
P
U
D


---
**Page 409**
8.4. Inference in Graphical Models
393
Figure 8.35
A directed graph whose conditional independence
properties cannot be expressed using an undirected
graph over the same three variables.
C
A
B
that distribution. A perfect map is therefore both an I map and a D map.
Consider the set of distributions such that for each distribution there exists a
directed graph that is a perfect map. This set is distinct from the set of distributions
such that for each distribution there exists an undirected graph that is a perfect map.
In addition there are distributions for which neither directed nor undirected graphs
offer a perfect map. This is illustrated as a Venn diagram in Figure 8.34.
Figure 8.35 shows an example of a directed graph that is a perfect map for
a distribution satisfying the conditional independence properties A ⊥⊥B | ∅and
A ̸⊥⊥B | C. There is no corresponding undirected graph over the same three vari-
ables that is a perfect map.
Conversely, consider the undirected graph over four variables shown in Fig-
ure 8.36. This graph exhibits the properties A ̸⊥⊥B | ∅, C ⊥⊥D | A ∪B and
A ⊥⊥B | C ∪D. There is no directed graph over four variables that implies the same
set of conditional independence properties.
The graphical framework can be extended in a consistent way to graphs that
include both directed and undirected links. These are called chain graphs (Lauritzen
and Wermuth, 1989; Frydenberg, 1990), and contain the directed and undirected
graphs considered so far as special cases. Although such graphs can represent a
broader class of distributions than either directed or undirected alone, there remain
distributions for which even a chain graph cannot provide a perfect map. Chain
graphs are not discussed further in this book.
Figure 8.36
An undirected graph whose conditional independence
properties cannot be expressed in terms of a directed
graph over the same variables.
A
C
B
D
8.4. Inference in Graphical Models
We turn now to the problem of inference in graphical models, in which some of
the nodes in a graph are clamped to observed values, and we wish to compute the
posterior distributions of one or more subsets of other nodes. As we shall see, we
can exploit the graphical structure both to ﬁnd efﬁcient algorithms for inference, and


---
**Page 410**
394
8. GRAPHICAL MODELS
Figure 8.37
A graphical representation of Bayes’ theorem.
See the text for details.
x
y
x
y
x
y
(a)
(b)
(c)
to make the structure of those algorithms transparent. Speciﬁcally, we shall see that
many algorithms can be expressed in terms of the propagation of local messages
around the graph. In this section, we shall focus primarily on techniques for exact
inference, and in Chapter 10 we shall consider a number of approximate inference
algorithms.
To start with, let us consider the graphical interpretation of Bayes’ theorem.
Suppose we decompose the joint distribution p(x, y) over two variables x and y into
a product of factors in the form p(x, y) = p(x)p(y|x). This can be represented by
the directed graph shown in Figure 8.37(a). Now suppose we observe the value of
y, as indicated by the shaded node in Figure 8.37(b). We can view the marginal
distribution p(x) as a prior over the latent variable x, and our goal is to infer the
corresponding posterior distribution over x. Using the sum and product rules of
probability we can evaluate
p(y) =

x′
p(y|x′)p(x′)
(8.47)
which can then be used in Bayes’ theorem to calculate
p(x|y) = p(y|x)p(x)
p(y)
.
(8.48)
Thus the joint distribution is now expressed in terms of p(y) and p(x|y). From a
graphical perspective, the joint distribution p(x, y) is now represented by the graph
shown in Figure 8.37(c), in which the direction of the arrow is reversed. This is the
simplest example of an inference problem for a graphical model.
8.4.1
Inference on a chain
Now consider a more complex problem involving the chain of nodes of the form
shown in Figure 8.32. This example will lay the foundation for a discussion of exact
inference in more general graphs later in this section.
Speciﬁcally, we shall consider the undirected graph in Figure 8.32(b). We have
already seen that the directed chain can be transformed into an equivalent undirected
chain. Because the directed graph does not have any nodes with more than one
parent, this does not require the addition of any extra links, and the directed and
undirected versions of this graph express exactly the same set of conditional inde-
pendence statements.


---
**Page 411**
8.4. Inference in Graphical Models
395
The joint distribution for this graph takes the form
p(x) = 1
Z ψ1,2(x1, x2)ψ2,3(x2, x3) · · · ψN−1,N(xN−1, xN).
(8.49)
We shall consider the speciﬁc case in which the N nodes represent discrete vari-
ables each having K states, in which case each potential function ψn−1,n(xn−1, xn)
comprises an K × K table, and so the joint distribution has (N −1)K2 parameters.
Let us consider the inference problem of ﬁnding the marginal distribution p(xn)
for a speciﬁc node xn that is part way along the chain. Note that, for the moment,
there are no observed nodes. By deﬁnition, the required marginal is obtained by
summing the joint distribution over all variables except xn, so that
p(xn) =

x1
· · ·

xn−1

xn+1
· · ·

xN
p(x).
(8.50)
In a naive implementation, we would ﬁrst evaluate the joint distribution and
then perform the summations explicitly. The joint distribution can be represented as
a set of numbers, one for each possible value for x. Because there are N variables
each with K states, there are KN values for x and so evaluation and storage of the
joint distribution, as well as marginalization to obtain p(xn), all involve storage and
computation that scale exponentially with the length N of the chain.
We can, however, obtain a much more efﬁcient algorithm by exploiting the con-
ditional independence properties of the graphical model. If we substitute the factor-
ized expression (8.49) for the joint distribution into (8.50), then we can rearrange the
order of the summations and the multiplications to allow the required marginal to be
evaluated much more efﬁciently. Consider for instance the summation over xN. The
potential ψN−1,N(xN−1, xN) is the only one that depends on xN, and so we can
perform the summation

xN
ψN−1,N(xN−1, xN)
(8.51)
ﬁrst to give a function of xN−1. We can then use this to perform the summation
over xN−1, which will involve only this new function together with the potential
ψN−2,N−1(xN−2, xN−1), because this is the only other place that xN−1 appears.
Similarly, the summation over x1 involves only the potential ψ1,2(x1, x2) and so
can be performed separately to give a function of x2, and so on. Because each
summation effectively removes a variable from the distribution, this can be viewed
as the removal of a node from the graph.
If we group the potentials and summations together in this way, we can express


---
**Page 412**
396
8. GRAPHICAL MODELS
the desired marginal in the form
p(xn) = 1
Z
⎡
⎣
xn−1
ψn−1,n(xn−1, xn) · · ·

x2
ψ2,3(x2, x3)

x1
ψ1,2(x1, x2)
  
· · ·
⎤
⎦
(
)*
+
µα(xn)
⎡
⎣
xn+1
ψn,n+1(xn, xn+1) · · ·

xN
ψN−1,N(xN−1, xN)
 
· · ·
⎤
⎦
(
)*
+
µβ(xn)
.
(8.52)
The reader is encouraged to study this re-ordering carefully as the underlying idea
forms the basis for the later discussion of the general sum-product algorithm. Here
the key concept that we are exploiting is that multiplication is distributive over addi-
tion, so that
ab + ac = a(b + c)
(8.53)
in which the left-hand side involves three arithmetic operations whereas the right-
hand side reduces this to two operations.
Let us work out the computational cost of evaluating the required marginal using
this re-ordered expression. We have to perform N −1 summations each of which is
over K states and each of which involves a function of two variables. For instance,
the summation over x1 involves only the function ψ1,2(x1, x2), which is a table of
K × K numbers. We have to sum this table over x1 for each value of x2 and so this
has O(K2) cost. The resulting vector of K numbers is multiplied by the matrix of
numbers ψ2,3(x2, x3) and so is again O(K2). Because there are N −1 summations
and multiplications of this kind, the total cost of evaluating the marginal p(xn) is
O(NK2). This is linear in the length of the chain, in contrast to the exponential cost
of a naive approach. We have therefore been able to exploit the many conditional
independence properties of this simple graph in order to obtain an efﬁcient calcula-
tion. If the graph had been fully connected, there would have been no conditional
independence properties, and we would have been forced to work directly with the
full joint distribution.
We now give a powerful interpretation of this calculation in terms of the passing
of local messages around on the graph. From (8.52) we see that the expression for the
marginal p(xn) decomposes into the product of two factors times the normalization
constant
p(xn) = 1
Z µα(xn)µβ(xn).
(8.54)
We shall interpret µα(xn) as a message passed forwards along the chain from node
xn−1 to node xn. Similarly, µβ(xn) can be viewed as a message passed backwards


---
**Page 413**
8.4. Inference in Graphical Models
397
Figure 8.38
The
marginal
distribution
p(xn) for a node xn along the chain is ob-
tained by multiplying the two messages
µα(xn) and µβ(xn), and then normaliz-
ing.
These messages can themselves
be evaluated recursively by passing mes-
sages from both ends of the chain to-
wards node xn.
x1
xn−1
xn
xn+1
xN
µα(xn−1)
µα(xn)
µβ(xn)
µβ(xn+1)
along the chain to node xn from node xn+1. Note that each of the messages com-
prises a set of K values, one for each choice of xn, and so the product of two mes-
sages should be interpreted as the point-wise multiplication of the elements of the
two messages to give another set of K values.
The message µα(xn) can be evaluated recursively because
µα(xn)
=

xn−1
ψn−1,n(xn−1, xn)
⎡
⎣
xn−2
· · ·
⎤
⎦
=

xn−1
ψn−1,n(xn−1, xn)µα(xn−1).
(8.55)
We therefore ﬁrst evaluate
µα(x2) =

x1
ψ1,2(x1, x2)
(8.56)
and then apply (8.55) repeatedly until we reach the desired node. Note carefully the
structure of the message passing equation. The outgoing message µα(xn) in (8.55)
is obtained by multiplying the incoming message µα(xn−1) by the local potential
involving the node variable and the outgoing variable and then summing over the
node variable.
Similarly, the message µβ(xn) can be evaluated recursively by starting with
node xN and using
µβ(xn)
=

xn+1
ψn+1,n(xn+1, xn)
⎡
⎣
xn+2
· · ·
⎤
⎦
=

xn+1
ψn+1,n(xn+1, xn)µβ(xn+1).
(8.57)
This recursive message passing is illustrated in Figure 8.38. The normalization con-
stant Z is easily evaluated by summing the right-hand side of (8.54) over all states
of xn, an operation that requires only O(K) computation.
Graphs of the form shown in Figure 8.38 are called Markov chains, and the
corresponding message passing equations represent an example of the Chapman-
Kolmogorov equations for Markov processes (Papoulis, 1984).


---
**Page 414**
398
8. GRAPHICAL MODELS
Now suppose we wish to evaluate the marginals p(xn) for every node n ∈
{1, . . . , N} in the chain. Simply applying the above procedure separately for each
node will have computational cost that is O(N 2M 2). However, such an approach
would be very wasteful of computation. For instance, to ﬁnd p(x1) we need to prop-
agate a message µβ(·) from node xN back to node x2. Similarly, to evaluate p(x2)
we need to propagate a messages µβ(·) from node xN back to node x3. This will
involve much duplicated computation because most of the messages will be identical
in the two cases.
Suppose instead we ﬁrst launch a message µβ(xN−1) starting from node xN
and propagate corresponding messages all the way back to node x1, and suppose we
similarly launch a message µα(x2) starting from node x1 and propagate the corre-
sponding messages all the way forward to node xN. Provided we store all of the
intermediate messages along the way, then any node can evaluate its marginal sim-
ply by applying (8.54). The computational cost is only twice that for ﬁnding the
marginal of a single node, rather than N times as much. Observe that a message
has passed once in each direction across each link in the graph. Note also that the
normalization constant Z need be evaluated only once, using any convenient node.
If some of the nodes in the graph are observed, then the corresponding variables
are simply clamped to their observed values and there is no summation. To see
this, note that the effect of clamping a variable xn to an observed value xn can
be expressed by multiplying the joint distribution by (one or more copies of) an
additional function I(xn, xn), which takes the value 1 when xn = xn and the value
0 otherwise. One such function can then be absorbed into each of the potentials that
contain xn. Summations over xn then contain only one term in which xn = xn.
Now suppose we wish to calculate the joint distribution p(xn−1, xn) for two
neighbouring nodes on the chain. This is similar to the evaluation of the marginal
for a single node, except that there are now two variables that are not summed out.
A few moments thought will show that the required joint distribution can be written
Exercise 8.15
in the form
p(xn−1, xn) = 1
Z µα(xn−1)ψn−1,n(xn−1, xn)µβ(xn).
(8.58)
Thus we can obtain the joint distributions over all of the sets of variables in each
of the potentials directly once we have completed the message passing required to
obtain the marginals.
This is a useful result because in practice we may wish to use parametric forms
for the clique potentials, or equivalently for the conditional distributions if we started
from a directed graph. In order to learn the parameters of these potentials in situa-
tions where not all of the variables are observed, we can employ the EM algorithm,
Chapter 9
and it turns out that the local joint distributions of the cliques, conditioned on any
observed data, is precisely what is needed in the E step. We shall consider some
examples of this in detail in Chapter 13.
8.4.2
Trees
We have seen that exact inference on a graph comprising a chain of nodes can be
performed efﬁciently in time that is linear in the number of nodes, using an algorithm


---
**Page 415**
8.4. Inference in Graphical Models
399
Figure 8.39
Examples
of
tree-
structured graphs, showing (a) an
undirected tree, (b) a directed tree,
and (c) a directed polytree.
(a)
(b)
(c)
that can be interpreted in terms of messages passed along the chain. More generally,
inference can be performed efﬁciently using local message passing on a broader
class of graphs called trees. In particular, we shall shortly generalize the message
passing formalism derived above for chains to give the sum-product algorithm, which
provides an efﬁcient framework for exact inference in tree-structured graphs.
In the case of an undirected graph, a tree is deﬁned as a graph in which there
is one, and only one, path between any pair of nodes. Such graphs therefore do not
have loops. In the case of directed graphs, a tree is deﬁned such that there is a single
node, called the root, which has no parents, and all other nodes have one parent. If
we convert a directed tree into an undirected graph, we see that the moralization step
will not add any links as all nodes have at most one parent, and as a consequence the
corresponding moralized graph will be an undirected tree. Examples of undirected
and directed trees are shown in Figure 8.39(a) and 8.39(b). Note that a distribution
represented as a directed tree can easily be converted into one represented by an
undirected tree, and vice versa.
Exercise 8.18
If there are nodes in a directed graph that have more than one parent, but there is
still only one path (ignoring the direction of the arrows) between any two nodes, then
the graph is a called a polytree, as illustrated in Figure 8.39(c). Such a graph will
have more than one node with the property of having no parents, and furthermore,
the corresponding moralized undirected graph will have loops.
8.4.3
Factor graphs
The sum-product algorithm that we derive in the next section is applicable to
undirected and directed trees and to polytrees. It can be cast in a particularly simple
and general form if we ﬁrst introduce a new graphical construction called a factor
graph (Frey, 1998; Kschischnang et al., 2001).
Both directed and undirected graphs allow a global function of several vari-
ables to be expressed as a product of factors over subsets of those variables. Factor
graphs make this decomposition explicit by introducing additional nodes for the fac-
tors themselves in addition to the nodes representing the variables. They also allow
us to be more explicit about the details of the factorization, as we shall see.
Let us write the joint distribution over a set of variables in the form of a product
of factors
p(x) =

s
fs(xs)
(8.59)
where xs denotes a subset of the variables. For convenience, we shall denote the


---
**Page 416**
400
8. GRAPHICAL MODELS
Figure 8.40
Example of a factor graph, which corresponds
to the factorization (8.60).
x1
x2
x3
fa
fb
fc
fd
individual variables by xi, however, as in earlier discussions, these can comprise
groups of variables (such as vectors or matrices). Each factor fs is a function of a
corresponding set of variables xs.
Directed graphs, whose factorization is deﬁned by (8.5), represent special cases
of (8.59) in which the factors fs(xs) are local conditional distributions. Similarly,
undirected graphs, given by (8.39), are a special case in which the factors are po-
tential functions over the maximal cliques (the normalizing coefﬁcient 1/Z can be
viewed as a factor deﬁned over the empty set of variables).
In a factor graph, there is a node (depicted as usual by a circle) for every variable
in the distribution, as was the case for directed and undirected graphs. There are also
additional nodes (depicted by small squares) for each factor fs(xs) in the joint dis-
tribution. Finally, there are undirected links connecting each factor node to all of the
variables nodes on which that factor depends. Consider, for example, a distribution
that is expressed in terms of the factorization
p(x) = fa(x1, x2)fb(x1, x2)fc(x2, x3)fd(x3).
(8.60)
This can be expressed by the factor graph shown in Figure 8.40. Note that there are
two factors fa(x1, x2) and fb(x1, x2) that are deﬁned over the same set of variables.
In an undirected graph, the product of two such factors would simply be lumped
together into the same clique potential. Similarly, fc(x2, x3) and fd(x3) could be
combined into a single potential over x2 and x3. The factor graph, however, keeps
such factors explicit and so is able to convey more detailed information about the
underlying factorization.
x1
x2
x3
(a)
x1
x2
x3
f
(b)
x1
x2
x3
fa
fb
(c)
Figure 8.41
(a) An undirected graph with a single clique potential ψ(x1, x2, x3). (b) A factor graph with factor
f(x1, x2, x3) = ψ(x1, x2, x3) representing the same distribution as the undirected graph. (c) A different factor
graph representing the same distribution, whose factors satisfy fa(x1, x2, x3)fb(x1, x2) = ψ(x1, x2, x3).


---
**Page 417**
8.4. Inference in Graphical Models
401
x1
x2
x3
(a)
x1
x2
x3
f
(b)
x1
x2
x3
fc
fa
fb
(c)
Figure 8.42
(a) A directed graph with the factorization p(x1)p(x2)p(x3|x1, x2). (b) A factor graph representing
the same distribution as the directed graph, whose factor satisﬁes f(x1, x2, x3) = p(x1)p(x2)p(x3|x1, x2). (c)
A different factor graph representing the same distribution with factors fa(x1) = p(x1), fb(x2) = p(x2) and
fc(x1, x2, x3) = p(x3|x1, x2).
Factor graphs are said to be bipartite because they consist of two distinct kinds
of nodes, and all links go between nodes of opposite type. In general, factor graphs
can therefore always be drawn as two rows of nodes (variable nodes at the top and
factor nodes at the bottom) with links between the rows, as shown in the example in
Figure 8.40. In some situations, however, other ways of laying out the graph may
be more intuitive, for example when the factor graph is derived from a directed or
undirected graph, as we shall see.
If we are given a distribution that is expressed in terms of an undirected graph,
then we can readily convert it to a factor graph. To do this, we create variable nodes
corresponding to the nodes in the original undirected graph, and then create addi-
tional factor nodes corresponding to the maximal cliques xs. The factors fs(xs) are
then set equal to the clique potentials. Note that there may be several different factor
graphs that correspond to the same undirected graph. These concepts are illustrated
in Figure 8.41.
Similarly, to convert a directed graph to a factor graph, we simply create variable
nodes in the factor graph corresponding to the nodes of the directed graph, and then
create factor nodes corresponding to the conditional distributions, and then ﬁnally
add the appropriate links. Again, there can be multiple factor graphs all of which
correspond to the same directed graph. The conversion of a directed graph to a
factor graph is illustrated in Figure 8.42.
We have already noted the importance of tree-structured graphs for performing
efﬁcient inference. If we take a directed or undirected tree and convert it into a factor
graph, then the result will again be a tree (in other words, the factor graph will have
no loops, and there will be one and only one path connecting any two nodes). In
the case of a directed polytree, conversion to an undirected graph results in loops
due to the moralization step, whereas conversion to a factor graph again results in a
tree, as illustrated in Figure 8.43.
In fact, local cycles in a directed graph due to
links connecting parents of a node can be removed on conversion to a factor graph
by deﬁning the appropriate factor function, as shown in Figure 8.44.
We have seen that multiple different factor graphs can represent the same di-
rected or undirected graph. This allows factor graphs to be more speciﬁc about the


---
**Page 418**
402
8. GRAPHICAL MODELS
(a)
(b)
(c)
Figure 8.43
(a) A directed polytree. (b) The result of converting the polytree into an undirected graph showing
the creation of loops. (c) The result of converting the polytree into a factor graph, which retains the tree structure.
precise form of the factorization. Figure 8.45 shows an example of a fully connected
undirected graph along with two different factor graphs.
In (b), the joint distri-
bution is given by a general form p(x) = f(x1, x2, x3), whereas in (c), it is given
by the more speciﬁc factorization p(x) = fa(x1, x2)fb(x1, x3)fc(x2, x3). It should
be emphasized that the factorization in (c) does not correspond to any conditional
independence properties.
8.4.4
The sum-product algorithm
We shall now make use of the factor graph framework to derive a powerful class
of efﬁcient, exact inference algorithms that are applicable to tree-structured graphs.
Here we shall focus on the problem of evaluating local marginals over nodes or
subsets of nodes, which will lead us to the sum-product algorithm. Later we shall
modify the technique to allow the most probable state to be found, giving rise to the
max-sum algorithm.
Also we shall suppose that all of the variables in the model are discrete, and
so marginalization corresponds to performing sums. The framework, however, is
equally applicable to linear-Gaussian models in which case marginalization involves
integration, and we shall consider an example of this in detail when we discuss linear
dynamical systems.
Section 13.3
Figure 8.44
(a)
A
fragment
of
a
di-
rected graph having a lo-
cal cycle.
(b) Conversion
to a fragment of a factor
graph having a tree struc-
ture, in which f(x1, x2, x3) =
p(x1)p(x2|x1)p(x3|x1, x2).
x1
x2
x3
(a)
x1
x2
x3
f(x1, x2, x3)
(b)


---
**Page 419**
8.4. Inference in Graphical Models
403
x1
x2
x3
(a)
x1
x2
x3
f(x1, x2, x3)
(b)
x1
x2
x3
fa
fc
fb
(c)
Figure 8.45
(a) A fully connected undirected graph. (b) and (c) Two factor graphs each of which corresponds
to the undirected graph in (a).
There is an algorithm for exact inference on directed graphs without loops known
as belief propagation (Pearl, 1988; Lauritzen and Spiegelhalter, 1988), and is equiv-
alent to a special case of the sum-product algorithm. Here we shall consider only the
sum-product algorithm because it is simpler to derive and to apply, as well as being
more general.
We shall assume that the original graph is an undirected tree or a directed tree or
polytree, so that the corresponding factor graph has a tree structure. We ﬁrst convert
the original graph into a factor graph so that we can deal with both directed and
undirected models using the same framework. Our goal is to exploit the structure of
the graph to achieve two things: (i) to obtain an efﬁcient, exact inference algorithm
for ﬁnding marginals; (ii) in situations where several marginals are required to allow
computations to be shared efﬁciently.
We begin by considering the problem of ﬁnding the marginal p(x) for partic-
ular variable node x. For the moment, we shall suppose that all of the variables
are hidden. Later we shall see how to modify the algorithm to incorporate evidence
corresponding to observed variables. By deﬁnition, the marginal is obtained by sum-
ming the joint distribution over all variables except x so that
p(x) =

x\x
p(x)
(8.61)
where x \ x denotes the set of variables in x with variable x omitted. The idea is
to substitute for p(x) using the factor graph expression (8.59) and then interchange
summations and products in order to obtain an efﬁcient algorithm. Consider the
fragment of graph shown in Figure 8.46 in which we see that the tree structure of
the graph allows us to partition the factors in the joint distribution into groups, with
one group associated with each of the factor nodes that is a neighbour of the variable
node x. We see that the joint distribution can be written as a product of the form
p(x) =

s∈ne(x)
Fs(x, Xs)
(8.62)
ne(x) denotes the set of factor nodes that are neighbours of x, and Xs denotes the
set of all variables in the subtree connected to the variable node x via the factor node


---
**Page 420**
404
8. GRAPHICAL MODELS
Figure 8.46
A fragment of a factor graph illustrating the
evaluation of the marginal p(x).
x
fs
µfs→x(x)
Fs(x, Xs)
fs, and Fs(x, Xs) represents the product of all the factors in the group associated
with factor fs.
Substituting (8.62) into (8.61) and interchanging the sums and products, we ob-
tain
p(x)
=

s∈ne(x)

Xs
Fs(x, Xs)
 
=

s∈ne(x)
µfs→x(x).
(8.63)
Here we have introduced a set of functions µfs→x(x), deﬁned by
µfs→x(x) ≡

Xs
Fs(x, Xs)
(8.64)
which can be viewed as messages from the factor nodes fs to the variable node x.
We see that the required marginal p(x) is given by the product of all the incoming
messages arriving at node x.
In order to evaluate these messages, we again turn to Figure 8.46 and note that
each factor Fs(x, Xs) is described by a factor (sub-)graph and so can itself be fac-
torized. In particular, we can write
Fs(x, Xs) = fs(x, x1, . . . , xM)G1 (x1, Xs1) . . . GM (xM, XsM)
(8.65)
where, for convenience, we have denoted the variables associated with factor fx, in
addition to x, by x1, . . . , xM. This factorization is illustrated in Figure 8.47. Note
that the set of variables {x, x1, . . . , xM} is the set of variables on which the factor
fs depends, and so it can also be denoted xs, using the notation of (8.59).
Substituting (8.65) into (8.64) we obtain
µfs→x(x)
=

x1
. . .

xM
fs(x, x1, . . . , xM)

m∈ne(fs)\x

Xxm
Gm(xm, Xsm)
 
=

x1
. . .

xM
fs(x, x1, . . . , xM)

m∈ne(fs)\x
µxm→fs(xm)
(8.66)


---
**Page 421**
8.4. Inference in Graphical Models
405
Figure 8.47
Illustration of the factorization of the subgraph as-
sociated with factor node fs.
xm
xM
x
fs
µxM→fs(xM)
µfs→x(x)
Gm(xm, Xsm)
where ne(fs) denotes the set of variable nodes that are neighbours of the factor node
fs, and ne(fs) \ x denotes the same set but with node x removed. Here we have
deﬁned the following messages from variable nodes to factor nodes
µxm→fs(xm) ≡

Xsm
Gm(xm, Xsm).
(8.67)
We have therefore introduced two distinct kinds of message, those that go from factor
nodes to variable nodes denoted µf→x(x), and those that go from variable nodes to
factor nodes denoted µx→f(x). In each case, we see that messages passed along a
link are always a function of the variable associated with the variable node that link
connects to.
The result (8.66) says that to evaluate the message sent by a factor node to a vari-
able node along the link connecting them, take the product of the incoming messages
along all other links coming into the factor node, multiply by the factor associated
with that node, and then marginalize over all of the variables associated with the
incoming messages. This is illustrated in Figure 8.47. It is important to note that
a factor node can send a message to a variable node once it has received incoming
messages from all other neighbouring variable nodes.
Finally, we derive an expression for evaluating the messages from variable nodes
to factor nodes, again by making use of the (sub-)graph factorization. From Fig-
ure 8.48, we see that term Gm(xm, Xsm) associated with node xm is given by a
product of terms Fl(xm, Xml) each associated with one of the factor nodes fl that is
linked to node xm (excluding node fs), so that
Gm(xm, Xsm) =

l∈ne(xm)\fs
Fl(xm, Xml)
(8.68)
where the product is taken over all neighbours of node xm except for node fs. Note
that each of the factors Fl(xm, Xml) represents a subtree of the original graph of
precisely the same kind as introduced in (8.62). Substituting (8.68) into (8.67), we


---
**Page 422**
406
8. GRAPHICAL MODELS
Figure 8.48
Illustration of the evaluation of the message sent by a
variable node to an adjacent factor node.
xm
fl
fL
fs
Fl(xm, Xml)
then obtain
µxm→fs(xm)
=

l∈ne(xm)\fs

Xml
Fl(xm, Xml)
 
=

l∈ne(xm)\fs
µfl→xm(xm)
(8.69)
where we have used the deﬁnition (8.64) of the messages passed from factor nodes to
variable nodes. Thus to evaluate the message sent by a variable node to an adjacent
factor node along the connecting link, we simply take the product of the incoming
messages along all of the other links. Note that any variable node that has only
two neighbours performs no computation but simply passes messages through un-
changed. Also, we note that a variable node can send a message to a factor node
once it has received incoming messages from all other neighbouring factor nodes.
Recall that our goal is to calculate the marginal for variable node x, and that this
marginal is given by the product of incoming messages along all of the links arriving
at that node. Each of these messages can be computed recursively in terms of other
messages. In order to start this recursion, we can view the node x as the root of the
tree and begin at the leaf nodes. From the deﬁnition (8.69), we see that if a leaf node
is a variable node, then the message that it sends along its one and only link is given
by
µx→f(x) = 1
(8.70)
as illustrated in Figure 8.49(a). Similarly, if the leaf node is a factor node, we see
from (8.66) that the message sent should take the form
µf→x(x) = f(x)
(8.71)
Figure 8.49
The sum-product algorithm
begins with messages sent
by the leaf nodes, which de-
pend on whether the leaf
node is (a) a variable node,
or (b) a factor node.
x
f
µx→f(x) = 1
(a)
x
f
µf→x(x) = f(x)
(b)

