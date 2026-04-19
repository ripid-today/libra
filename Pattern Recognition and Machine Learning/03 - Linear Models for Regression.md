# 03 - Linear Models for Regression
*Pages 137-178 from Pattern Recognition and Machine Learning*

---
**Page 137**
2.4. The Exponential Family
119
an interval A ⩽µ ⩽B as to the shifted interval A −c ⩽µ ⩽B −c. This implies
 B
A
p(µ) dµ =
 B−c
A−c
p(µ) dµ =
 B
A
p(µ −c) dµ
(2.234)
and because this must hold for all choices of A and B, we have
p(µ −c) = p(µ)
(2.235)
which implies that p(µ) is constant. An example of a location parameter would be
the mean µ of a Gaussian distribution. As we have seen, the conjugate prior distri-
bution for µ in this case is a Gaussian p(µ|µ0, σ2
0) = N(µ|µ0, σ2
0), and we obtain a
noninformative prior by taking the limit σ2
0 →∞. Indeed, from (2.141) and (2.142)
we see that this gives a posterior distribution over µ in which the contributions from
the prior vanish.
As a second example, consider a density of the form
p(x|σ) = 1
σ f
x
σ

(2.236)
where σ > 0. Note that this will be a normalized density provided f(x) is correctly
normalized. The parameter σ is known as a scale parameter, and the density exhibits
Exercise 2.59
scale invariance because if we scale x by a constant to give x = cx, then
p(x|σ) = 1
σ f
x
σ

(2.237)
where we have deﬁned σ = cσ. This transformation corresponds to a change of
scale, for example from meters to kilometers if x is a length, and we would like
to choose a prior distribution that reﬂects this scale invariance. If we consider an
interval A ⩽σ ⩽B, and a scaled interval A/c ⩽σ ⩽B/c, then the prior should
assign equal probability mass to these two intervals. Thus we have
 B
A
p(σ) dσ =
 B/c
A/c
p(σ) dσ =
 B
A
p
1
cσ
 1
c dσ
(2.238)
and because this must hold for choices of A and B, we have
p(σ) = p
1
cσ
 1
c
(2.239)
and hence p(σ) ∝1/σ. Note that again this is an improper prior because the integral
of the distribution over 0 ⩽σ ⩽∞is divergent. It is sometimes also convenient
to think of the prior distribution for a scale parameter in terms of the density of the
log of the parameter. Using the transformation rule (1.27) for densities we see that
p(ln σ) = const. Thus, for this prior there is the same probability mass in the range
1 ⩽σ ⩽10 as in the range 10 ⩽σ ⩽100 and in 100 ⩽σ ⩽1000.


---
**Page 138**
120
2. PROBABILITY DISTRIBUTIONS
An example of a scale parameter would be the standard deviation σ of a Gaussian
distribution, after we have taken account of the location parameter µ, because
N(x|µ, σ2) ∝σ−1 exp 
−(x/σ)2
(2.240)
where x = x −µ. As discussed earlier, it is often more convenient to work in terms
of the precision λ = 1/σ2 rather than σ itself. Using the transformation rule for
densities, we see that a distribution p(σ) ∝1/σ corresponds to a distribution over λ
of the form p(λ) ∝1/λ. We have seen that the conjugate prior for λ was the gamma
distribution Gam(λ|a0, b0) given by (2.146). The noninformative prior is obtained
Section 2.3
as the special case a0 = b0 = 0. Again, if we examine the results (2.150) and (2.151)
for the posterior distribution of λ, we see that for a0 = b0 = 0, the posterior depends
only on terms arising from the data and not from the prior.
2.5. Nonparametric Methods
Throughout this chapter, we have focussed on the use of probability distributions
having speciﬁc functional forms governed by a small number of parameters whose
values are to be determined from a data set. This is called the parametric approach
to density modelling. An important limitation of this approach is that the chosen
density might be a poor model of the distribution that generates the data, which can
result in poor predictive performance. For instance, if the process that generates the
data is multimodal, then this aspect of the distribution can never be captured by a
Gaussian, which is necessarily unimodal.
In this ﬁnal section, we consider some nonparametric approaches to density es-
timation that make few assumptions about the form of the distribution. Here we shall
focus mainly on simple frequentist methods. The reader should be aware, however,
that nonparametric Bayesian methods are attracting increasing interest (Walker et al.,
1999; Neal, 2000; M¨uller and Quintana, 2004; Teh et al., 2006).
Let us start with a discussion of histogram methods for density estimation, which
we have already encountered in the context of marginal and conditional distributions
in Figure 1.11 and in the context of the central limit theorem in Figure 2.6. Here we
explore the properties of histogram density models in more detail, focussing on the
case of a single continuous variable x. Standard histograms simply partition x into
distinct bins of width ∆i and then count the number ni of observations of x falling
in bin i. In order to turn this count into a normalized probability density, we simply
divide by the total number N of observations and by the width ∆i of the bins to
obtain probability values for each bin given by
pi =
ni
N∆i
(2.241)
for which it is easily seen that 
p(x) dx = 1. This gives a model for the density
p(x) that is constant over the width of each bin, and often the bins are chosen to have
the same width ∆i = ∆.


---
**Page 139**
2.5. Nonparametric Methods
121
Figure 2.24
An illustration of the histogram approach
to density estimation, in which a data set
of 50 data points is generated from the
distribution shown by the green curve.
Histogram density estimates, based on
(2.241), with a common bin width ∆are
shown for various values of ∆.
∆= 0.04
0
0.5
1
0
5
∆= 0.08
0
0.5
1
0
5
∆= 0.25
0
0.5
1
0
5
In Figure 2.24, we show an example of histogram density estimation. Here
the data is drawn from the distribution, corresponding to the green curve, which is
formed from a mixture of two Gaussians. Also shown are three examples of his-
togram density estimates corresponding to three different choices for the bin width
∆. We see that when ∆is very small (top ﬁgure), the resulting density model is very
spiky, with a lot of structure that is not present in the underlying distribution that
generated the data set. Conversely, if ∆is too large (bottom ﬁgure) then the result is
a model that is too smooth and that consequently fails to capture the bimodal prop-
erty of the green curve. The best results are obtained for some intermediate value
of ∆(middle ﬁgure). In principle, a histogram density model is also dependent on
the choice of edge location for the bins, though this is typically much less signiﬁcant
than the value of ∆.
Note that the histogram method has the property (unlike the methods to be dis-
cussed shortly) that, once the histogram has been computed, the data set itself can
be discarded, which can be advantageous if the data set is large. Also, the histogram
approach is easily applied if the data points are arriving sequentially.
In practice, the histogram technique can be useful for obtaining a quick visual-
ization of data in one or two dimensions but is unsuited to most density estimation
applications. One obvious problem is that the estimated density has discontinuities
that are due to the bin edges rather than any property of the underlying distribution
that generated the data. Another major limitation of the histogram approach is its
scaling with dimensionality. If we divide each variable in a D-dimensional space
into M bins, then the total number of bins will be M D. This exponential scaling
with D is an example of the curse of dimensionality. In a space of high dimensional-
Section 1.4
ity, the quantity of data needed to provide meaningful estimates of local probability
density would be prohibitive.
The histogram approach to density estimation does, however, teach us two im-
portant lessons. First, to estimate the probability density at a particular location,
we should consider the data points that lie within some local neighbourhood of that
point. Note that the concept of locality requires that we assume some form of dis-
tance measure, and here we have been assuming Euclidean distance. For histograms,


---
**Page 140**
122
2. PROBABILITY DISTRIBUTIONS
this neighbourhood property was deﬁned by the bins, and there is a natural ‘smooth-
ing’ parameter describing the spatial extent of the local region, in this case the bin
width. Second, the value of the smoothing parameter should be neither too large nor
too small in order to obtain good results. This is reminiscent of the choice of model
complexity in polynomial curve ﬁtting discussed in Chapter 1 where the degree M
of the polynomial, or alternatively the value α of the regularization parameter, was
optimal for some intermediate value, neither too large nor too small. Armed with
these insights, we turn now to a discussion of two widely used nonparametric tech-
niques for density estimation, kernel estimators and nearest neighbours, which have
better scaling with dimensionality than the simple histogram model.
2.5.1
Kernel density estimators
Let us suppose that observations are being drawn from some unknown probabil-
ity density p(x) in some D-dimensional space, which we shall take to be Euclidean,
and we wish to estimate the value of p(x). From our earlier discussion of locality,
let us consider some small region R containing x. The probability mass associated
with this region is given by
P =

R
p(x) dx.
(2.242)
Now suppose that we have collected a data set comprising N observations drawn
from p(x). Because each data point has a probability P of falling within R, the total
number K of points that lie inside R will be distributed according to the binomial
distribution
Section 2.1
Bin(K|N, P) =
N!
K!(N −K)!P K(1 −P)1−K.
(2.243)
Using (2.11), we see that the mean fraction of points falling inside the region is
E[K/N] = P, and similarly using (2.12) we see that the variance around this mean
is var[K/N] = P(1 −P)/N. For large N, this distribution will be sharply peaked
around the mean and so
K ≃NP.
(2.244)
If, however, we also assume that the region R is sufﬁciently small that the probability
density p(x) is roughly constant over the region, then we have
P ≃p(x)V
(2.245)
where V is the volume of R. Combining (2.244) and (2.245), we obtain our density
estimate in the form
p(x) = K
NV .
(2.246)
Note that the validity of (2.246) depends on two contradictory assumptions, namely
that the region R be sufﬁciently small that the density is approximately constant over
the region and yet sufﬁciently large (in relation to the value of that density) that the
number K of points falling inside the region is sufﬁcient for the binomial distribution
to be sharply peaked.


---
**Page 141**
2.5. Nonparametric Methods
123
We can exploit the result (2.246) in two different ways. Either we can ﬁx K and
determine the value of V from the data, which gives rise to the K-nearest-neighbour
technique discussed shortly, or we can ﬁx V and determine K from the data, giv-
ing rise to the kernel approach. It can be shown that both the K-nearest-neighbour
density estimator and the kernel density estimator converge to the true probability
density in the limit N →∞provided V shrinks suitably with N, and K grows with
N (Duda and Hart, 1973).
We begin by discussing the kernel method in detail, and to start with we take
the region R to be a small hypercube centred on the point x at which we wish to
determine the probability density. In order to count the number K of points falling
within this region, it is convenient to deﬁne the following function
k(u) =

1,
|ui| ⩽1/2,
i = 1, . . . , D,
0,
otherwise
(2.247)
which represents a unit cube centred on the origin. The function k(u) is an example
of a kernel function, and in this context is also called a Parzen window. From (2.247),
the quantity k((x −xn)/h) will be one if the data point xn lies inside a cube of side
h centred on x, and zero otherwise. The total number of data points lying inside this
cube will therefore be
K =
N

n=1
k
x −xn
h

.
(2.248)
Substituting this expression into (2.246) then gives the following result for the esti-
mated density at x
p(x) = 1
N
N

n=1
1
hD k
x −xn
h

(2.249)
where we have used V = hD for the volume of a hypercube of side h in D di-
mensions. Using the symmetry of the function k(u), we can now re-interpret this
equation, not as a single cube centred on x but as the sum over N cubes centred on
the N data points xn.
As it stands, the kernel density estimator (2.249) will suffer from one of the same
problems that the histogram method suffered from, namely the presence of artiﬁcial
discontinuities, in this case at the boundaries of the cubes. We can obtain a smoother
density model if we choose a smoother kernel function, and a common choice is the
Gaussian, which gives rise to the following kernel density model
p(x) = 1
N
N

n=1
1
(2πh2)1/2 exp

−∥x −xn∥2
2h2

(2.250)
where h represents the standard deviation of the Gaussian components. Thus our
density model is obtained by placing a Gaussian over each data point and then adding
up the contributions over the whole data set, and then dividing by N so that the den-
sity is correctly normalized. In Figure 2.25, we apply the model (2.250) to the data


---
**Page 142**
124
2. PROBABILITY DISTRIBUTIONS
Figure 2.25
Illustration of the kernel density model
(2.250) applied to the same data set used
to demonstrate the histogram approach in
Figure 2.24.
We see that h acts as a
smoothing parameter and that if it is set
too small (top panel), the result is a very
noisy density model, whereas if it is set
too large (bottom panel), then the bimodal
nature of the underlying distribution from
which the data is generated (shown by the
green curve) is washed out. The best den-
sity model is obtained for some intermedi-
ate value of h (middle panel).
h = 0.005
0
0.5
1
0
5
h = 0.07
0
0.5
1
0
5
h = 0.2
0
0.5
1
0
5
set used earlier to demonstrate the histogram technique. We see that, as expected,
the parameter h plays the role of a smoothing parameter, and there is a trade-off
between sensitivity to noise at small h and over-smoothing at large h. Again, the
optimization of h is a problem in model complexity, analogous to the choice of bin
width in histogram density estimation, or the degree of the polynomial used in curve
ﬁtting.
We can choose any other kernel function k(u) in (2.249) subject to the condi-
tions
k(u)
⩾
0,
(2.251)

k(u) du
=
1
(2.252)
which ensure that the resulting probability distribution is nonnegative everywhere
and integrates to one. The class of density model given by (2.249) is called a kernel
density estimator, or Parzen estimator. It has a great merit that there is no compu-
tation involved in the ‘training’ phase because this simply requires storage of the
training set. However, this is also one of its great weaknesses because the computa-
tional cost of evaluating the density grows linearly with the size of the data set.
2.5.2
Nearest-neighbour methods
One of the difﬁculties with the kernel approach to density estimation is that the
parameter h governing the kernel width is ﬁxed for all kernels. In regions of high
data density, a large value of h may lead to over-smoothing and a washing out of
structure that might otherwise be extracted from the data. However, reducing h may
lead to noisy estimates elsewhere in data space where the density is smaller. Thus
the optimal choice for h may be dependent on location within the data space. This
issue is addressed by nearest-neighbour methods for density estimation.
We therefore return to our general result (2.246) for local density estimation,
and instead of ﬁxing V and determining the value of K from the data, we consider
a ﬁxed value of K and use the data to ﬁnd an appropriate value for V . To do this,
we consider a small sphere centred on the point x at which we wish to estimate the


---
**Page 143**
2.5. Nonparametric Methods
125
Figure 2.26
Illustration of K-nearest-neighbour den-
sity estimation using the same data set
as in Figures 2.25 and 2.24.
We see
that the parameter K governs the degree
of smoothing, so that a small value of
K leads to a very noisy density model
(top panel), whereas a large value (bot-
tom panel) smoothes out the bimodal na-
ture of the true distribution (shown by the
green curve) from which the data set was
generated.
K = 1
0
0.5
1
0
5
K = 5
0
0.5
1
0
5
K = 30
0
0.5
1
0
5
density p(x), and we allow the radius of the sphere to grow until it contains precisely
K data points. The estimate of the density p(x) is then given by (2.246) with V set to
the volume of the resulting sphere. This technique is known as K nearest neighbours
and is illustrated in Figure 2.26, for various choices of the parameter K, using the
same data set as used in Figure 2.24 and Figure 2.25. We see that the value of K
now governs the degree of smoothing and that again there is an optimum choice for
K that is neither too large nor too small. Note that the model produced by K nearest
neighbours is not a true density model because the integral over all space diverges.
Exercise 2.61
We close this chapter by showing how the K-nearest-neighbour technique for
density estimation can be extended to the problem of classiﬁcation. To do this, we
apply the K-nearest-neighbour density estimation technique to each class separately
and then make use of Bayes’ theorem. Let us suppose that we have a data set com-
prising Nk points in class Ck with N points in total, so that 
k Nk = N. If we
wish to classify a new point x, we draw a sphere centred on x containing precisely
K points irrespective of their class. Suppose this sphere has volume V and contains
Kk points from class Ck. Then (2.246) provides an estimate of the density associated
with each class
p(x|Ck) = Kk
NkV .
(2.253)
Similarly, the unconditional density is given by
p(x) = K
NV
(2.254)
while the class priors are given by
p(Ck) = Nk
N .
(2.255)
We can now combine (2.253), (2.254), and (2.255) using Bayes’ theorem to obtain
the posterior probability of class membership
p(Ck|x) = p(x|Ck)p(Ck)
p(x)
= Kk
K .
(2.256)


---
**Page 144**
126
2. PROBABILITY DISTRIBUTIONS
Figure 2.27
(a) In the K-nearest-
neighbour classiﬁer, a new point,
shown by the black diamond, is clas-
siﬁed according to the majority class
membership of the K closest train-
ing data points, in this case K =
3.
(b) In the nearest-neighbour
(K = 1) approach to classiﬁcation,
the resulting decision boundary is
composed of hyperplanes that form
perpendicular bisectors of pairs of
points from different classes.
x1
x2
(a)
x1
x2
(b)
If we wish to minimize the probability of misclassiﬁcation, this is done by assigning
the test point x to the class having the largest posterior probability, corresponding to
the largest value of Kk/K. Thus to classify a new point, we identify the K nearest
points from the training data set and then assign the new point to the class having the
largest number of representatives amongst this set. Ties can be broken at random.
The particular case of K = 1 is called the nearest-neighbour rule, because a test
point is simply assigned to the same class as the nearest point from the training set.
These concepts are illustrated in Figure 2.27.
In Figure 2.28, we show the results of applying the K-nearest-neighbour algo-
rithm to the oil ﬂow data, introduced in Chapter 1, for various values of K. As
expected, we see that K controls the degree of smoothing, so that small K produces
many small regions of each class, whereas large K leads to fewer larger regions.
x6
x7
K = 1
0
1
2
0
1
2
x6
x7
K = 3
0
1
2
0
1
2
x6
x7
K = 31
0
1
2
0
1
2
Figure 2.28
Plot of 200 data points from the oil data set showing values of x6 plotted against x7, where the
red, green, and blue points correspond to the ‘laminar’, ‘annular’, and ‘homogeneous’ classes, respectively. Also
shown are the classiﬁcations of the input space given by the K-nearest-neighbour algorithm for various values
of K.


---
**Page 145**
Exercises
127
An interesting property of the nearest-neighbour (K = 1) classiﬁer is that, in the
limit N →∞, the error rate is never more than twice the minimum achievable error
rate of an optimal classiﬁer, i.e., one that uses the true class distributions (Cover and
Hart, 1967) .
As discussed so far, both the K-nearest-neighbour method, and the kernel den-
sity estimator, require the entire training data set to be stored, leading to expensive
computation if the data set is large. This effect can be offset, at the expense of some
additional one-off computation, by constructing tree-based search structures to allow
(approximate) near neighbours to be found efﬁciently without doing an exhaustive
search of the data set. Nevertheless, these nonparametric methods are still severely
limited. On the other hand, we have seen that simple parametric models are very
restricted in terms of the forms of distribution that they can represent. We therefore
need to ﬁnd density models that are very ﬂexible and yet for which the complexity
of the models can be controlled independently of the size of the training set, and we
shall see in subsequent chapters how to achieve this.
Exercises
2.1
(⋆) www
Verify that the Bernoulli distribution (2.2) satisﬁes the following prop-
erties
1

x=0
p(x|µ)
=
1
(2.257)
E[x]
=
µ
(2.258)
var[x]
=
µ(1 −µ).
(2.259)
Show that the entropy H[x] of a Bernoulli distributed random binary variable x is
given by
H[x] = −µ ln µ −(1 −µ) ln(1 −µ).
(2.260)
2.2
(⋆⋆)
The form of the Bernoulli distribution given by (2.2) is not symmetric be-
tween the two values of x. In some situations, it will be more convenient to use an
equivalent formulation for which x ∈{−1, 1}, in which case the distribution can be
written
p(x|µ) =
1 −µ
2
(1−x)/2 1 + µ
2
(1+x)/2
(2.261)
where µ ∈[−1, 1]. Show that the distribution (2.261) is normalized, and evaluate its
mean, variance, and entropy.
2.3
(⋆⋆) www
In this exercise, we prove that the binomial distribution (2.9) is nor-
malized. First use the deﬁnition (2.10) of the number of combinations of m identical
objects chosen from a total of N to show that
N
m

+

N
m −1

=
N + 1
m

.
(2.262)


---
**Page 146**
128
2. PROBABILITY DISTRIBUTIONS
Use this result to prove by induction the following result
(1 + x)N =
N

m=0
N
m

xm
(2.263)
which is known as the binomial theorem, and which is valid for all real values of x.
Finally, show that the binomial distribution is normalized, so that
N

m=0
N
m

µm(1 −µ)N−m = 1
(2.264)
which can be done by ﬁrst pulling out a factor (1 −µ)N out of the summation and
then making use of the binomial theorem.
2.4
(⋆⋆) Show that the mean of the binomial distribution is given by (2.11). To do this,
differentiate both sides of the normalization condition (2.264) with respect to µ and
then rearrange to obtain an expression for the mean of n. Similarly, by differentiating
(2.264) twice with respect to µ and making use of the result (2.11) for the mean of
the binomial distribution prove the result (2.12) for the variance of the binomial.
2.5
(⋆⋆) www
In this exercise, we prove that the beta distribution, given by (2.13), is
correctly normalized, so that (2.14) holds. This is equivalent to showing that
 1
0
µa−1(1 −µ)b−1 dµ = Γ(a)Γ(b)
Γ(a + b) .
(2.265)
From the deﬁnition (1.141) of the gamma function, we have
Γ(a)Γ(b) =
 ∞
0
exp(−x)xa−1 dx
 ∞
0
exp(−y)yb−1 dy.
(2.266)
Use this expression to prove (2.265) as follows. First bring the integral over y inside
the integrand of the integral over x, next make the change of variable t = y + x
where x is ﬁxed, then interchange the order of the x and t integrations, and ﬁnally
make the change of variable x = tµ where t is ﬁxed.
2.6
(⋆) Make use of the result (2.265) to show that the mean, variance, and mode of the
beta distribution (2.13) are given respectively by
E[µ]
=
a
a + b
(2.267)
var[µ]
=
ab
(a + b)2(a + b + 1)
(2.268)
mode[µ]
=
a −1
a + b −2.
(2.269)


---
**Page 147**
Exercises
129
2.7
(⋆⋆) Consider a binomial random variable x given by (2.9), with prior distribution
for µ given by the beta distribution (2.13), and suppose we have observed m occur-
rences of x = 1 and l occurrences of x = 0. Show that the posterior mean value of x
lies between the prior mean and the maximum likelihood estimate for µ. To do this,
show that the posterior mean can be written as λ times the prior mean plus (1 −λ)
times the maximum likelihood estimate, where 0 ⩽λ ⩽1. This illustrates the con-
cept of the posterior distribution being a compromise between the prior distribution
and the maximum likelihood solution.
2.8
(⋆) Consider two variables x and y with joint distribution p(x, y). Prove the follow-
ing two results
E[x]
=
Ey [Ex[x|y]]
(2.270)
var[x]
=
Ey [varx[x|y]] + vary [Ex[x|y]] .
(2.271)
Here Ex[x|y] denotes the expectation of x under the conditional distribution p(x|y),
with a similar notation for the conditional variance.
2.9
(⋆⋆⋆) www
. In this exercise, we prove the normalization of the Dirichlet dis-
tribution (2.38) using induction. We have already shown in Exercise 2.5 that the
beta distribution, which is a special case of the Dirichlet for M = 2, is normalized.
We now assume that the Dirichlet distribution is normalized for M −1 variables
and prove that it is normalized for M variables. To do this, consider the Dirichlet
distribution over M variables, and take account of the constraint M
k=1 µk = 1 by
eliminating µM, so that the Dirichlet is written
pM(µ1, . . . , µM−1) = CM
M−1

k=1
µαk−1
k

1 −
M−1

j=1
µj
αM−1
(2.272)
and our goal is to ﬁnd an expression for CM. To do this, integrate over µM−1, taking
care over the limits of integration, and then make a change of variable so that this
integral has limits 0 and 1. By assuming the correct result for CM−1 and making use
of (2.265), derive the expression for CM.
2.10
(⋆⋆)
Using the property Γ(x + 1) = xΓ(x) of the gamma function, derive the
following results for the mean, variance, and covariance of the Dirichlet distribution
given by (2.38)
E[µj]
=
αj
α0
(2.273)
var[µj]
=
αj(α0 −αj)
α2
0(α0 + 1)
(2.274)
cov[µjµl]
=
−
αjαl
α2
0(α0 + 1),
j ̸= l
(2.275)
where α0 is deﬁned by (2.39).


---
**Page 148**
130
2. PROBABILITY DISTRIBUTIONS
2.11
(⋆) www
By expressing the expectation of ln µj under the Dirichlet distribution
(2.38) as a derivative with respect to αj, show that
E[ln µj] = ψ(αj) −ψ(α0)
(2.276)
where α0 is given by (2.39) and
ψ(a) ≡d
da ln Γ(a)
(2.277)
is the digamma function.
2.12
(⋆) The uniform distribution for a continuous variable x is deﬁned by
U(x|a, b) =
1
b −a,
a ⩽x ⩽b.
(2.278)
Verify that this distribution is normalized, and ﬁnd expressions for its mean and
variance.
2.13
(⋆⋆)
Evaluate the Kullback-Leibler divergence (1.113) between two Gaussians
p(x) = N(x|µ, Σ) and q(x) = N(x|m, L).
2.14
(⋆⋆) www
This exercise demonstrates that the multivariate distribution with max-
imum entropy, for a given covariance, is a Gaussian. The entropy of a distribution
p(x) is given by
H[x] = −

p(x) ln p(x) dx.
(2.279)
We wish to maximize H[x] over all distributions p(x) subject to the constraints that
p(x) be normalized and that it have a speciﬁc mean and covariance, so that

p(x) dx = 1
(2.280)

p(x)x dx = µ
(2.281)

p(x)(x −µ)(x −µ)T dx = Σ.
(2.282)
By performing a variational maximization of (2.279) and using Lagrange multipliers
to enforce the constraints (2.280), (2.281), and (2.282), show that the maximum
likelihood distribution is given by the Gaussian (2.43).
2.15
(⋆⋆) Show that the entropy of the multivariate Gaussian N(x|µ, Σ) is given by
H[x] = 1
2 ln |Σ| + D
2 (1 + ln(2π))
(2.283)
where D is the dimensionality of x.


---
**Page 149**
Exercises
131
2.16
(⋆⋆⋆) www
Consider two random variables x1 and x2 having Gaussian distri-
butions with means µ1, µ2 and precisions τ1, τ2 respectively. Derive an expression
for the differential entropy of the variable x = x1 + x2. To do this, ﬁrst ﬁnd the
distribution of x by using the relation
p(x) =
 ∞
−∞
p(x|x2)p(x2) dx2
(2.284)
and completing the square in the exponent. Then observe that this represents the
convolution of two Gaussian distributions, which itself will be Gaussian, and ﬁnally
make use of the result (1.110) for the entropy of the univariate Gaussian.
2.17
(⋆) www
Consider the multivariate Gaussian distribution given by (2.43). By
writing the precision matrix (inverse covariance matrix) Σ−1 as the sum of a sym-
metric and an anti-symmetric matrix, show that the anti-symmetric term does not
appear in the exponent of the Gaussian, and hence that the precision matrix may be
taken to be symmetric without loss of generality. Because the inverse of a symmetric
matrix is also symmetric (see Exercise 2.22), it follows that the covariance matrix
may also be chosen to be symmetric without loss of generality.
2.18
(⋆⋆⋆)
Consider a real, symmetric matrix Σ whose eigenvalue equation is given
by (2.45). By taking the complex conjugate of this equation and subtracting the
original equation, and then forming the inner product with eigenvector ui, show that
the eigenvalues λi are real. Similarly, use the symmetry property of Σ to show that
two eigenvectors ui and uj will be orthogonal provided λj ̸= λi. Finally, show that
without loss of generality, the set of eigenvectors can be chosen to be orthonormal,
so that they satisfy (2.46), even if some of the eigenvalues are zero.
2.19
(⋆⋆) Show that a real, symmetric matrix Σ having the eigenvector equation (2.45)
can be expressed as an expansion in the eigenvectors, with coefﬁcients given by the
eigenvalues, of the form (2.48). Similarly, show that the inverse matrix Σ−1 has a
representation of the form (2.49).
2.20
(⋆⋆) www
A positive deﬁnite matrix Σ can be deﬁned as one for which the
quadratic form
aTΣa
(2.285)
is positive for any real value of the vector a. Show that a necessary and sufﬁcient
condition for Σ to be positive deﬁnite is that all of the eigenvalues λi of Σ, deﬁned
by (2.45), are positive.
2.21
(⋆) Show that a real, symmetric matrix of size D ×D has D(D +1)/2 independent
parameters.
2.22
(⋆) www
Show that the inverse of a symmetric matrix is itself symmetric.
2.23
(⋆⋆) By diagonalizing the coordinate system using the eigenvector expansion (2.45),
show that the volume contained within the hyperellipsoid corresponding to a constant


---
**Page 150**
132
2. PROBABILITY DISTRIBUTIONS
Mahalanobis distance ∆is given by
VD|Σ|1/2∆D
(2.286)
where VD is the volume of the unit sphere in D dimensions, and the Mahalanobis
distance is deﬁned by (2.44).
2.24
(⋆⋆) www
Prove the identity (2.76) by multiplying both sides by the matrix

A
B
C
D

(2.287)
and making use of the deﬁnition (2.77).
2.25
(⋆⋆) In Sections 2.3.1 and 2.3.2, we considered the conditional and marginal distri-
butions for a multivariate Gaussian. More generally, we can consider a partitioning
of the components of x into three groups xa, xb, and xc, with a corresponding par-
titioning of the mean vector µ and of the covariance matrix Σ in the form
µ =
µa
µb
µc

,
Σ =
Σaa
Σab
Σac
Σba
Σbb
Σbc
Σca
Σcb
Σcc

.
(2.288)
By making use of the results of Section 2.3, ﬁnd an expression for the conditional
distribution p(xa|xb) in which xc has been marginalized out.
2.26
(⋆⋆)
A very useful result from linear algebra is the Woodbury matrix inversion
formula given by
(A + BCD)−1 = A−1 −A−1B(C−1 + DA−1B)−1DA−1.
(2.289)
By multiplying both sides by (A + BCD) prove the correctness of this result.
2.27
(⋆)
Let x and z be two independent random vectors, so that p(x, z) = p(x)p(z).
Show that the mean of their sum y = x+z is given by the sum of the means of each
of the variable separately. Similarly, show that the covariance matrix of y is given by
the sum of the covariance matrices of x and z. Conﬁrm that this result agrees with
that of Exercise 1.10.
2.28
(⋆⋆⋆) www
Consider a joint distribution over the variable
z =

x
y

(2.290)
whose mean and covariance are given by (2.108) and (2.105) respectively. By mak-
ing use of the results (2.92) and (2.93) show that the marginal distribution p(x) is
given (2.99). Similarly, by making use of the results (2.81) and (2.82) show that the
conditional distribution p(y|x) is given by (2.100).


---
**Page 151**
Exercises
133
2.29
(⋆⋆) Using the partitioned matrix inversion formula (2.76), show that the inverse of
the precision matrix (2.104) is given by the covariance matrix (2.105).
2.30
(⋆)
By starting from (2.107) and making use of the result (2.105), verify the result
(2.108).
2.31
(⋆⋆)
Consider two multidimensional random vectors x and z having Gaussian
distributions p(x) = N(x|µx, Σx) and p(z) = N(z|µz, Σz) respectively, together
with their sum y = x+z. Use the results (2.109) and (2.110) to ﬁnd an expression for
the marginal distribution p(y) by considering the linear-Gaussian model comprising
the product of the marginal distribution p(x) and the conditional distribution p(y|x).
2.32
(⋆⋆⋆) www
This exercise and the next provide practice at manipulating the
quadratic forms that arise in linear-Gaussian models, as well as giving an indepen-
dent check of results derived in the main text. Consider a joint distribution p(x, y)
deﬁned by the marginal and conditional distributions given by (2.99) and (2.100).
By examining the quadratic form in the exponent of the joint distribution, and using
the technique of ‘completing the square’ discussed in Section 2.3, ﬁnd expressions
for the mean and covariance of the marginal distribution p(y) in which the variable
x has been integrated out. To do this, make use of the Woodbury matrix inversion
formula (2.289). Verify that these results agree with (2.109) and (2.110) obtained
using the results of Chapter 2.
2.33
(⋆⋆⋆)
Consider the same joint distribution as in Exercise 2.32, but now use the
technique of completing the square to ﬁnd expressions for the mean and covariance
of the conditional distribution p(x|y). Again, verify that these agree with the corre-
sponding expressions (2.111) and (2.112).
2.34
(⋆⋆) www
To ﬁnd the maximum likelihood solution for the covariance matrix
of a multivariate Gaussian, we need to maximize the log likelihood function (2.118)
with respect to Σ, noting that the covariance matrix must be symmetric and positive
deﬁnite. Here we proceed by ignoring these constraints and doing a straightforward
maximization. Using the results (C.21), (C.26), and (C.28) from Appendix C, show
that the covariance matrix Σ that maximizes the log likelihood function (2.118) is
given by the sample covariance (2.122). We note that the ﬁnal result is necessarily
symmetric and positive deﬁnite (provided the sample covariance is nonsingular).
2.35
(⋆⋆) Use the result (2.59) to prove (2.62). Now, using the results (2.59), and (2.62),
show that
E[xnxm] = µµT + InmΣ
(2.291)
where xn denotes a data point sampled from a Gaussian distribution with mean µ
and covariance Σ, and Inm denotes the (n, m) element of the identity matrix. Hence
prove the result (2.124).
2.36
(⋆⋆) www
Using an analogous procedure to that used to obtain (2.126), derive
an expression for the sequential estimation of the variance of a univariate Gaussian


---
**Page 152**
134
2. PROBABILITY DISTRIBUTIONS
distribution, by starting with the maximum likelihood expression
σ2
ML = 1
N
N

n=1
(xn −µ)2.
(2.292)
Verify that substituting the expression for a Gaussian distribution into the Robbins-
Monro sequential estimation formula (2.135) gives a result of the same form, and
hence obtain an expression for the corresponding coefﬁcients aN.
2.37
(⋆⋆)
Using an analogous procedure to that used to obtain (2.126), derive an ex-
pression for the sequential estimation of the covariance of a multivariate Gaussian
distribution, by starting with the maximum likelihood expression (2.122). Verify that
substituting the expression for a Gaussian distribution into the Robbins-Monro se-
quential estimation formula (2.135) gives a result of the same form, and hence obtain
an expression for the corresponding coefﬁcients aN.
2.38
(⋆) Use the technique of completing the square for the quadratic form in the expo-
nent to derive the results (2.141) and (2.142).
2.39
(⋆⋆)
Starting from the results (2.141) and (2.142) for the posterior distribution
of the mean of a Gaussian random variable, dissect out the contributions from the
ﬁrst N −1 data points and hence obtain expressions for the sequential update of
µN and σ2
N. Now derive the same results starting from the posterior distribution
p(µ|x1, . . . , xN−1) = N(µ|µN−1, σ2
N−1) and multiplying by the likelihood func-
tion p(xN|µ) = N(xN|µ, σ2) and then completing the square and normalizing to
obtain the posterior distribution after N observations.
2.40
(⋆⋆) www
Consider a D-dimensional Gaussian random variable x with distribu-
tion N(x|µ, Σ) in which the covariance Σ is known and for which we wish to infer
the mean µ from a set of observations X = {x1, . . . , xN}. Given a prior distribution
p(µ) = N(µ|µ0, Σ0), ﬁnd the corresponding posterior distribution p(µ|X).
2.41
(⋆)
Use the deﬁnition of the gamma function (1.141) to show that the gamma dis-
tribution (2.146) is normalized.
2.42
(⋆⋆) Evaluate the mean, variance, and mode of the gamma distribution (2.146).
2.43
(⋆) The following distribution
p(x|σ2, q) =
q
2(2σ2)1/qΓ(1/q) exp

−|x|q
2σ2

(2.293)
is a generalization of the univariate Gaussian distribution. Show that this distribution
is normalized so that
 ∞
−∞
p(x|σ2, q) dx = 1
(2.294)
and that it reduces to the Gaussian when q = 2. Consider a regression model in
which the target variable is given by t = y(x, w) + ϵ and ϵ is a random noise


---
**Page 153**
Exercises
135
variable drawn from the distribution (2.293). Show that the log likelihood function
over w and σ2, for an observed data set of input vectors X = {x1, . . . , xN} and
corresponding target variables t = (t1, . . . , tN)T, is given by
ln p(t|X, w, σ2) = −1
2σ2
N

n=1
|y(xn, w) −tn|q −N
q ln(2σ2) + const
(2.295)
where ‘const’ denotes terms independent of both w and σ2. Note that, as a function
of w, this is the Lq error function considered in Section 1.5.5.
2.44
(⋆⋆)
Consider a univariate Gaussian distribution N(x|µ, τ −1) having conjugate
Gaussian-gamma prior given by (2.154), and a data set x = {x1, . . . , xN} of i.i.d.
observations. Show that the posterior distribution is also a Gaussian-gamma distri-
bution of the same functional form as the prior, and write down expressions for the
parameters of this posterior distribution.
2.45
(⋆)
Verify that the Wishart distribution deﬁned by (2.155) is indeed a conjugate
prior for the precision matrix of a multivariate Gaussian.
2.46
(⋆) www
Verify that evaluating the integral in (2.158) leads to the result (2.159).
2.47
(⋆) www
Show that in the limit ν →∞, the t-distribution (2.159) becomes a
Gaussian. Hint: ignore the normalization coefﬁcient, and simply look at the depen-
dence on x.
2.48
(⋆)
By following analogous steps to those used to derive the univariate Student’s
t-distribution (2.159), verify the result (2.162) for the multivariate form of the Stu-
dent’s t-distribution, by marginalizing over the variable η in (2.161). Using the
deﬁnition (2.161), show by exchanging integration variables that the multivariate
t-distribution is correctly normalized.
2.49
(⋆⋆) By using the deﬁnition (2.161) of the multivariate Student’s t-distribution as a
convolution of a Gaussian with a gamma distribution, verify the properties (2.164),
(2.165), and (2.166) for the multivariate t-distribution deﬁned by (2.162).
2.50
(⋆) Show that in the limit ν →∞, the multivariate Student’s t-distribution (2.162)
reduces to a Gaussian with mean µ and precision Λ.
2.51
(⋆) www
The various trigonometric identities used in the discussion of periodic
variables in this chapter can be proven easily from the relation
exp(iA) = cos A + i sin A
(2.296)
in which i is the square root of minus one. By considering the identity
exp(iA) exp(−iA) = 1
(2.297)
prove the result (2.177). Similarly, using the identity
cos(A −B) = ℜexp{i(A −B)}
(2.298)


---
**Page 154**
136
2. PROBABILITY DISTRIBUTIONS
where ℜdenotes the real part, prove (2.178). Finally, by using sin(A −B) =
ℑexp{i(A −B)}, where ℑdenotes the imaginary part, prove the result (2.183).
2.52
(⋆⋆)
For large m, the von Mises distribution (2.179) becomes sharply peaked
around the mode θ0. By deﬁning ξ = m1/2(θ −θ0) and making the Taylor ex-
pansion of the cosine function given by
cos α = 1 −α2
2 + O(α4)
(2.299)
show that as m →∞, the von Mises distribution tends to a Gaussian.
2.53
(⋆) Using the trigonometric identity (2.183), show that solution of (2.182) for θ0 is
given by (2.184).
2.54
(⋆) By computing ﬁrst and second derivatives of the von Mises distribution (2.179),
and using I0(m) > 0 for m > 0, show that the maximum of the distribution occurs
when θ = θ0 and that the minimum occurs when θ = θ0 + π (mod 2π).
2.55
(⋆) By making use of the result (2.168), together with (2.184) and the trigonometric
identity (2.178), show that the maximum likelihood solution mML for the concentra-
tion of the von Mises distribution satisﬁes A(mML) = r where r is the radius of the
mean of the observations viewed as unit vectors in the two-dimensional Euclidean
plane, as illustrated in Figure 2.17.
2.56
(⋆⋆) www
Express the beta distribution (2.13), the gamma distribution (2.146),
and the von Mises distribution (2.179) as members of the exponential family (2.194)
and thereby identify their natural parameters.
2.57
(⋆)
Verify that the multivariate Gaussian distribution can be cast in exponential
family form (2.194) and derive expressions for η, u(x), h(x) and g(η) analogous to
(2.220)–(2.223).
2.58
(⋆) The result (2.226) showed that the negative gradient of ln g(η) for the exponen-
tial family is given by the expectation of u(x). By taking the second derivatives of
(2.195), show that
−∇∇ln g(η) = E[u(x)u(x)T] −E[u(x)]E[u(x)T] = cov[u(x)].
(2.300)
2.59
(⋆)
By changing variables using y = x/σ, show that the density (2.236) will be
correctly normalized, provided f(x) is correctly normalized.
2.60
(⋆⋆) www
Consider a histogram-like density model in which the space x is di-
vided into ﬁxed regions for which the density p(x) takes the constant value hi over
the ith region, and that the volume of region i is denoted ∆i. Suppose we have a set
of N observations of x such that ni of these observations fall in region i. Using a
Lagrange multiplier to enforce the normalization constraint on the density, derive an
expression for the maximum likelihood estimator for the {hi}.
2.61
(⋆) Show that the K-nearest-neighbour density model deﬁnes an improper distribu-
tion whose integral over all space is divergent.


---
**Page 155**
3
Linear
Models for
Regression
The focus so far in this book has been on unsupervised learning, including topics
such as density estimation and data clustering. We turn now to a discussion of super-
vised learning, starting with regression. The goal of regression is to predict the value
of one or more continuous target variables t given the value of a D-dimensional vec-
tor x of input variables. We have already encountered an example of a regression
problem when we considered polynomial curve ﬁtting in Chapter 1. The polynomial
is a speciﬁc example of a broad class of functions called linear regression models,
which share the property of being linear functions of the adjustable parameters, and
which will form the focus of this chapter. The simplest form of linear regression
models are also linear functions of the input variables. However, we can obtain a
much more useful class of functions by taking linear combinations of a ﬁxed set of
nonlinear functions of the input variables, known as basis functions. Such models
are linear functions of the parameters, which gives them simple analytical properties,
and yet can be nonlinear with respect to the input variables.
137


---
**Page 156**
138
3. LINEAR MODELS FOR REGRESSION
Given a training data set comprising N observations {xn}, where n = 1, . . . , N,
together with corresponding target values {tn}, the goal is to predict the value of t
for a new value of x. In the simplest approach, this can be done by directly con-
structing an appropriate function y(x) whose values for new inputs x constitute the
predictions for the corresponding values of t. More generally, from a probabilistic
perspective, we aim to model the predictive distribution p(t|x) because this expresses
our uncertainty about the value of t for each value of x. From this conditional dis-
tribution we can make predictions of t, for any new value of x, in such a way as to
minimize the expected value of a suitably chosen loss function. As discussed in Sec-
tion 1.5.5, a common choice of loss function for real-valued variables is the squared
loss, for which the optimal solution is given by the conditional expectation of t.
Although linear models have signiﬁcant limitations as practical techniques for
pattern recognition, particularly for problems involving input spaces of high dimen-
sionality, they have nice analytical properties and form the foundation for more so-
phisticated models to be discussed in later chapters.
3.1. Linear Basis Function Models
The simplest linear model for regression is one that involves a linear combination of
the input variables
y(x, w) = w0 + w1x1 + . . . + wDxD
(3.1)
where x = (x1, . . . , xD)T. This is often simply known as linear regression. The key
property of this model is that it is a linear function of the parameters w0, . . . , wD. It is
also, however, a linear function of the input variables xi, and this imposes signiﬁcant
limitations on the model. We therefore extend the class of models by considering
linear combinations of ﬁxed nonlinear functions of the input variables, of the form
y(x, w) = w0 +
M−1

j=1
wjφj(x)
(3.2)
where φj(x) are known as basis functions. By denoting the maximum value of the
index j by M −1, the total number of parameters in this model will be M.
The parameter w0 allows for any ﬁxed offset in the data and is sometimes called
a bias parameter (not to be confused with ‘bias’ in a statistical sense). It is often
convenient to deﬁne an additional dummy ‘basis function’ φ0(x) = 1 so that
y(x, w) =
M−1

j=0
wjφj(x) = wTφ(x)
(3.3)
where w = (w0, . . . , wM−1)T and φ = (φ0, . . . , φM−1)T. In many practical ap-
plications of pattern recognition, we will apply some form of ﬁxed pre-processing,


---
**Page 157**
3.1. Linear Basis Function Models
139
or feature extraction, to the original data variables. If the original variables com-
prise the vector x, then the features can be expressed in terms of the basis functions
{φj(x)}.
By using nonlinear basis functions, we allow the function y(x, w) to be a non-
linear function of the input vector x. Functions of the form (3.2) are called linear
models, however, because this function is linear in w. It is this linearity in the pa-
rameters that will greatly simplify the analysis of this class of models. However, it
also leads to some signiﬁcant limitations, as we discuss in Section 3.6.
The example of polynomial regression considered in Chapter 1 is a particular
example of this model in which there is a single input variable x, and the basis func-
tions take the form of powers of x so that φj(x) = xj. One limitation of polynomial
basis functions is that they are global functions of the input variable, so that changes
in one region of input space affect all other regions. This can be resolved by dividing
the input space up into regions and ﬁt a different polynomial in each region, leading
to spline functions (Hastie et al., 2001).
There are many other possible choices for the basis functions, for example
φj(x) = exp

−(x −µj)2
2s2

(3.4)
where the µj govern the locations of the basis functions in input space, and the pa-
rameter s governs their spatial scale. These are usually referred to as ‘Gaussian’
basis functions, although it should be noted that they are not required to have a prob-
abilistic interpretation, and in particular the normalization coefﬁcient is unimportant
because these basis functions will be multiplied by adaptive parameters wj.
Another possibility is the sigmoidal basis function of the form
φj(x) = σ
x −µj
s

(3.5)
where σ(a) is the logistic sigmoid function deﬁned by
σ(a) =
1
1 + exp(−a).
(3.6)
Equivalently, we can use the ‘tanh’ function because this is related to the logistic
sigmoid by tanh(a) = 2σ(a) −1, and so a general linear combination of logistic
sigmoid functions is equivalent to a general linear combination of ‘tanh’ functions.
These various choices of basis function are illustrated in Figure 3.1.
Yet another possible choice of basis function is the Fourier basis, which leads to
an expansion in sinusoidal functions. Each basis function represents a speciﬁc fre-
quency and has inﬁnite spatial extent. By contrast, basis functions that are localized
to ﬁnite regions of input space necessarily comprise a spectrum of different spatial
frequencies. In many signal processing applications, it is of interest to consider ba-
sis functions that are localized in both space and frequency, leading to a class of
functions known as wavelets. These are also deﬁned to be mutually orthogonal, to
simplify their application. Wavelets are most applicable when the input values live


---
**Page 158**
140
3. LINEAR MODELS FOR REGRESSION
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
Figure 3.1
Examples of basis functions, showing polynomials on the left, Gaussians of the form (3.4) in the
centre, and sigmoidal of the form (3.5) on the right.
on a regular lattice, such as the successive time points in a temporal sequence, or the
pixels in an image. Useful texts on wavelets include Ogden (1997), Mallat (1999),
and Vidakovic (1999).
Most of the discussion in this chapter, however, is independent of the particular
choice of basis function set, and so for most of our discussion we shall not specify
the particular form of the basis functions, except for the purposes of numerical il-
lustration. Indeed, much of our discussion will be equally applicable to the situation
in which the vector φ(x) of basis functions is simply the identity φ(x) = x. Fur-
thermore, in order to keep the notation simple, we shall focus on the case of a single
target variable t. However, in Section 3.1.5, we consider brieﬂy the modiﬁcations
needed to deal with multiple target variables.
3.1.1
Maximum likelihood and least squares
In Chapter 1, we ﬁtted polynomial functions to data sets by minimizing a sum-
of-squares error function. We also showed that this error function could be motivated
as the maximum likelihood solution under an assumed Gaussian noise model. Let
us return to this discussion and consider the least squares approach, and its relation
to maximum likelihood, in more detail.
As before, we assume that the target variable t is given by a deterministic func-
tion y(x, w) with additive Gaussian noise so that
t = y(x, w) + ϵ
(3.7)
where ϵ is a zero mean Gaussian random variable with precision (inverse variance)
β. Thus we can write
p(t|x, w, β) = N(t|y(x, w), β−1).
(3.8)
Recall that, if we assume a squared loss function, then the optimal prediction, for a
new value of x, will be given by the conditional mean of the target variable. In the
Section 1.5.5
case of a Gaussian conditional distribution of the form (3.8), the conditional mean


---
**Page 159**
3.1. Linear Basis Function Models
141
will be simply
E[t|x] =

tp(t|x) dt = y(x, w).
(3.9)
Note that the Gaussian noise assumption implies that the conditional distribution of
t given x is unimodal, which may be inappropriate for some applications. An ex-
tension to mixtures of conditional Gaussian distributions, which permit multimodal
conditional distributions, will be discussed in Section 14.5.1.
Now consider a data set of inputs X = {x1, . . . , xN} with corresponding target
values t1, . . . , tN. We group the target variables {tn} into a column vector that we
denote by t where the typeface is chosen to distinguish it from a single observation
of a multivariate target, which would be denoted t. Making the assumption that
these data points are drawn independently from the distribution (3.8), we obtain the
following expression for the likelihood function, which is a function of the adjustable
parameters w and β, in the form
p(t|X, w, β) =
N

n=1
N(tn|wTφ(xn), β−1)
(3.10)
where we have used (3.3). Note that in supervised learning problems such as regres-
sion (and classiﬁcation), we are not seeking to model the distribution of the input
variables. Thus x will always appear in the set of conditioning variables, and so
from now on we will drop the explicit x from expressions such as p(t|x, w, β) in or-
der to keep the notation uncluttered. Taking the logarithm of the likelihood function,
and making use of the standard form (1.46) for the univariate Gaussian, we have
ln p(t|w, β)
=
N

n=1
ln N(tn|wTφ(xn), β−1)
=
N
2 ln β −N
2 ln(2π) −βED(w)
(3.11)
where the sum-of-squares error function is deﬁned by
ED(w) = 1
2
N

n=1
{tn −wTφ(xn)}2.
(3.12)
Having written down the likelihood function, we can use maximum likelihood to
determine w and β. Consider ﬁrst the maximization with respect to w. As observed
already in Section 1.2.5, we see that maximization of the likelihood function under a
conditional Gaussian noise distribution for a linear model is equivalent to minimizing
a sum-of-squares error function given by ED(w). The gradient of the log likelihood
function (3.11) takes the form
∇ln p(t|w, β) =
N

n=1

tn −wTφ(xn)

φ(xn)T.
(3.13)


---
**Page 160**
142
3. LINEAR MODELS FOR REGRESSION
Setting this gradient to zero gives
0 =
N

n=1
tnφ(xn)T −wT
 N

n=1
φ(xn)φ(xn)T

.
(3.14)
Solving for w we obtain
wML = 
ΦTΦ−1 ΦTt
(3.15)
which are known as the normal equations for the least squares problem. Here Φ is an
N×M matrix, called the design matrix, whose elements are given by Φnj = φj(xn),
so that
Φ =
⎛
⎜
⎜
⎝
φ0(x1)
φ1(x1)
· · ·
φM−1(x1)
φ0(x2)
φ1(x2)
· · ·
φM−1(x2)
...
...
...
...
φ0(xN)
φ1(xN)
· · ·
φM−1(xN)
⎞
⎟
⎟
⎠.
(3.16)
The quantity
Φ† ≡
ΦTΦ−1 ΦT
(3.17)
is known as the Moore-Penrose pseudo-inverse of the matrix Φ (Rao and Mitra,
1971; Golub and Van Loan, 1996). It can be regarded as a generalization of the
notion of matrix inverse to nonsquare matrices. Indeed, if Φ is square and invertible,
then using the property (AB)−1 = B−1A−1 we see that Φ† ≡Φ−1.
At this point, we can gain some insight into the role of the bias parameter w0. If
we make the bias parameter explicit, then the error function (3.12) becomes
ED(w) = 1
2
N

n=1
{tn −w0 −
M−1

j=1
wjφj(xn)}2.
(3.18)
Setting the derivative with respect to w0 equal to zero, and solving for w0, we obtain
w0 = t −
M−1

j=1
wjφj
(3.19)
where we have deﬁned
t = 1
N
N

n=1
tn,
φj = 1
N
N

n=1
φj(xn).
(3.20)
Thus the bias w0 compensates for the difference between the averages (over the
training set) of the target values and the weighted sum of the averages of the basis
function values.
We can also maximize the log likelihood function (3.11) with respect to the noise
precision parameter β, giving
1
βML
= 1
N
N

n=1
{tn −wT
MLφ(xn)}2
(3.21)


---
**Page 161**
3.1. Linear Basis Function Models
143
Figure 3.2
Geometrical interpretation of the least-squares
solution, in an N-dimensional space whose axes
are the values of t1, . . . , tN. The least-squares
regression function is obtained by ﬁnding the or-
thogonal projection of the data vector t onto the
subspace spanned by the basis functions φj(x)
in which each basis function is viewed as a vec-
tor ϕj of length N with elements φj(xn).
S
t
y
ϕ1
ϕ2
and so we see that the inverse of the noise precision is given by the residual variance
of the target values around the regression function.
3.1.2
Geometry of least squares
At this point, it is instructive to consider the geometrical interpretation of the
least-squares solution. To do this we consider an N-dimensional space whose axes
are given by the tn, so that t = (t1, . . . , tN)T is a vector in this space. Each basis
function φj(xn), evaluated at the N data points, can also be represented as a vector in
the same space, denoted by ϕj, as illustrated in Figure 3.2. Note that ϕj corresponds
to the jth column of Φ, whereas φ(xn) corresponds to the nth row of Φ. If the
number M of basis functions is smaller than the number N of data points, then the
M vectors φj(xn) will span a linear subspace S of dimensionality M. We deﬁne
y to be an N-dimensional vector whose nth element is given by y(xn, w), where
n = 1, . . . , N. Because y is an arbitrary linear combination of the vectors ϕj, it can
live anywhere in the M-dimensional subspace. The sum-of-squares error (3.12) is
then equal (up to a factor of 1/2) to the squared Euclidean distance between y and
t. Thus the least-squares solution for w corresponds to that choice of y that lies in
subspace S and that is closest to t. Intuitively, from Figure 3.2, we anticipate that
this solution corresponds to the orthogonal projection of t onto the subspace S. This
is indeed the case, as can easily be veriﬁed by noting that the solution for y is given
by ΦwML, and then conﬁrming that this takes the form of an orthogonal projection.
Exercise 3.2
In practice, a direct solution of the normal equations can lead to numerical difﬁ-
culties when ΦTΦ is close to singular. In particular, when two or more of the basis
vectors ϕj are co-linear, or nearly so, the resulting parameter values can have large
magnitudes. Such near degeneracies will not be uncommon when dealing with real
data sets. The resulting numerical difﬁculties can be addressed using the technique
of singular value decomposition, or SVD (Press et al., 1992; Bishop and Nabney,
2008). Note that the addition of a regularization term ensures that the matrix is non-
singular, even in the presence of degeneracies.
3.1.3
Sequential learning
Batch techniques, such as the maximum likelihood solution (3.15), which in-
volve processing the entire training set in one go, can be computationally costly for
large data sets. As we have discussed in Chapter 1, if the data set is sufﬁciently large,
it may be worthwhile to use sequential algorithms, also known as on-line algorithms,


---
**Page 162**
144
3. LINEAR MODELS FOR REGRESSION
in which the data points are considered one at a time, and the model parameters up-
dated after each such presentation. Sequential learning is also appropriate for real-
time applications in which the data observations are arriving in a continuous stream,
and predictions must be made before all of the data points are seen.
We can obtain a sequential learning algorithm by applying the technique of
stochastic gradient descent, also known as sequential gradient descent, as follows. If
the error function comprises a sum over data points E = 
n En, then after presen-
tation of pattern n, the stochastic gradient descent algorithm updates the parameter
vector w using
w(τ+1) = w(τ) −η∇En
(3.22)
where τ denotes the iteration number, and η is a learning rate parameter. We shall
discuss the choice of value for η shortly. The value of w is initialized to some starting
vector w(0). For the case of the sum-of-squares error function (3.12), this gives
w(τ+1) = w(τ) + η(tn −w(τ)Tφn)φn
(3.23)
where φn = φ(xn). This is known as least-mean-squares or the LMS algorithm.
The value of η needs to be chosen with care to ensure that the algorithm converges
(Bishop and Nabney, 2008).
3.1.4
Regularized least squares
In Section 1.1, we introduced the idea of adding a regularization term to an
error function in order to control over-ﬁtting, so that the total error function to be
minimized takes the form
ED(w) + λEW (w)
(3.24)
where λ is the regularization coefﬁcient that controls the relative importance of the
data-dependent error ED(w) and the regularization term EW (w). One of the sim-
plest forms of regularizer is given by the sum-of-squares of the weight vector ele-
ments
EW (w) = 1
2wTw.
(3.25)
If we also consider the sum-of-squares error function given by
E(w) = 1
2
N

n=1
{tn −wTφ(xn)}2
(3.26)
then the total error function becomes
1
2
N

n=1
{tn −wTφ(xn)}2 + λ
2 wTw.
(3.27)
This particular choice of regularizer is known in the machine learning literature as
weight decay because in sequential learning algorithms, it encourages weight values
to decay towards zero, unless supported by the data. In statistics, it provides an ex-
ample of a parameter shrinkage method because it shrinks parameter values towards


---
**Page 163**
3.1. Linear Basis Function Models
145
q = 0.5
q = 1
q = 2
q = 4
Figure 3.3
Contours of the regularization term in (3.29) for various values of the parameter q.
zero. It has the advantage that the error function remains a quadratic function of
w, and so its exact minimizer can be found in closed form. Speciﬁcally, setting the
gradient of (3.27) with respect to w to zero, and solving for w as before, we obtain
w =

λI + ΦTΦ
−1 ΦTt.
(3.28)
This represents a simple extension of the least-squares solution (3.15).
A more general regularizer is sometimes used, for which the regularized error
takes the form
1
2
N

n=1
{tn −wTφ(xn)}2 + λ
2
M

j=1
|wj|q
(3.29)
where q = 2 corresponds to the quadratic regularizer (3.27). Figure 3.3 shows con-
tours of the regularization function for different values of q.
The case of q = 1 is know as the lasso in the statistics literature (Tibshirani,
1996). It has the property that if λ is sufﬁciently large, some of the coefﬁcients
wj are driven to zero, leading to a sparse model in which the corresponding basis
functions play no role. To see this, we ﬁrst note that minimizing (3.29) is equivalent
to minimizing the unregularized sum-of-squares error (3.12) subject to the constraint
Exercise 3.5
M

j=1
|wj|q ⩽η
(3.30)
for an appropriate value of the parameter η, where the two approaches can be related
using Lagrange multipliers. The origin of the sparsity can be seen from Figure 3.4,
Appendix E
which shows that the minimum of the error function, subject to the constraint (3.30).
As λ is increased, so an increasing number of parameters are driven to zero.
Regularization allows complex models to be trained on data sets of limited size
without severe over-ﬁtting, essentially by limiting the effective model complexity.
However, the problem of determining the optimal model complexity is then shifted
from one of ﬁnding the appropriate number of basis functions to one of determining
a suitable value of the regularization coefﬁcient λ. We shall return to the issue of
model complexity later in this chapter.


---
**Page 164**
146
3. LINEAR MODELS FOR REGRESSION
Figure 3.4
Plot
of
the
contours
of the unregularized error function
(blue) along with the constraint re-
gion (3.30) for the quadratic regular-
izer q = 2 on the left and the lasso
regularizer q = 1 on the right, in
which the optimum value for the pa-
rameter vector w is denoted by w⋆.
The lasso gives a sparse solution in
which w⋆
1 = 0.
w1
w2
w⋆
w1
w2
w⋆
For the remainder of this chapter we shall focus on the quadratic regularizer
(3.27) both for its practical importance and its analytical tractability.
3.1.5
Multiple outputs
So far, we have considered the case of a single target variable t. In some applica-
tions, we may wish to predict K > 1 target variables, which we denote collectively
by the target vector t. This could be done by introducing a different set of basis func-
tions for each component of t, leading to multiple, independent regression problems.
However, a more interesting, and more common, approach is to use the same set of
basis functions to model all of the components of the target vector so that
y(x, w) = WTφ(x)
(3.31)
where y is a K-dimensional column vector, W is an M × K matrix of parameters,
and φ(x) is an M-dimensional column vector with elements φj(x), with φ0(x) = 1
as before. Suppose we take the conditional distribution of the target vector to be an
isotropic Gaussian of the form
p(t|x, W, β) = N(t|WTφ(x), β−1I).
(3.32)
If we have a set of observations t1, . . . , tN, we can combine these into a matrix T
of size N × K such that the nth row is given by tT
n. Similarly, we can combine the
input vectors x1, . . . , xN into a matrix X. The log likelihood function is then given
by
ln p(T|X, W, β)
=
N

n=1
ln N(tn|WTφ(xn), β−1I)
=
NK
2
ln
 β
2π

−β
2
N

n=1
''tn −WTφ(xn)
''2 . (3.33)


---
**Page 165**
3.2. The Bias-Variance Decomposition
147
As before, we can maximize this function with respect to W, giving
WML = 
ΦTΦ−1 ΦTT.
(3.34)
If we examine this result for each target variable tk, we have
wk = 
ΦTΦ−1 ΦTtk = Φ†tk
(3.35)
where tk is an N-dimensional column vector with components tnk for n = 1, . . . N.
Thus the solution to the regression problem decouples between the different target
variables, and we need only compute a single pseudo-inverse matrix Φ†, which is
shared by all of the vectors wk.
The extension to general Gaussian noise distributions having arbitrary covari-
ance matrices is straightforward. Again, this leads to a decoupling into K inde-
Exercise 3.6
pendent regression problems. This result is unsurprising because the parameters W
deﬁne only the mean of the Gaussian noise distribution, and we know from Sec-
tion 2.3.4 that the maximum likelihood solution for the mean of a multivariate Gaus-
sian is independent of the covariance. From now on, we shall therefore consider a
single target variable t for simplicity.
3.2. The Bias-Variance Decomposition
So far in our discussion of linear models for regression, we have assumed that the
form and number of basis functions are both ﬁxed. As we have seen in Chapter 1,
the use of maximum likelihood, or equivalently least squares, can lead to severe
over-ﬁtting if complex models are trained using data sets of limited size. However,
limiting the number of basis functions in order to avoid over-ﬁtting has the side
effect of limiting the ﬂexibility of the model to capture interesting and important
trends in the data. Although the introduction of regularization terms can control
over-ﬁtting for models with many parameters, this raises the question of how to
determine a suitable value for the regularization coefﬁcient λ. Seeking the solution
that minimizes the regularized error function with respect to both the weight vector
w and the regularization coefﬁcient λ is clearly not the right approach since this
leads to the unregularized solution with λ = 0.
As we have seen in earlier chapters, the phenomenon of over-ﬁtting is really an
unfortunate property of maximum likelihood and does not arise when we marginalize
over parameters in a Bayesian setting. In this chapter, we shall consider the Bayesian
view of model complexity in some depth. Before doing so, however, it is instructive
to consider a frequentist viewpoint of the model complexity issue, known as the bias-
variance trade-off. Although we shall introduce this concept in the context of linear
basis function models, where it is easy to illustrate the ideas using simple examples,
the discussion has more general applicability.
In Section 1.5.5, when we discussed decision theory for regression problems,
we considered various loss functions each of which leads to a corresponding optimal
prediction once we are given the conditional distribution p(t|x). A popular choice is


---
**Page 166**
148
3. LINEAR MODELS FOR REGRESSION
the squared loss function, for which the optimal prediction is given by the conditional
expectation, which we denote by h(x) and which is given by
h(x) = E[t|x] =

tp(t|x) dt.
(3.36)
At this point, it is worth distinguishing between the squared loss function arising
from decision theory and the sum-of-squares error function that arose in the maxi-
mum likelihood estimation of model parameters. We might use more sophisticated
techniques than least squares, for example regularization or a fully Bayesian ap-
proach, to determine the conditional distribution p(t|x). These can all be combined
with the squared loss function for the purpose of making predictions.
We showed in Section 1.5.5 that the expected squared loss can be written in the
form
E[L] =

{y(x) −h(x)}2 p(x) dx +

{h(x) −t}2p(x, t) dx dt.
(3.37)
Recall that the second term, which is independent of y(x), arises from the intrinsic
noise on the data and represents the minimum achievable value of the expected loss.
The ﬁrst term depends on our choice for the function y(x), and we will seek a so-
lution for y(x) which makes this term a minimum. Because it is nonnegative, the
smallest that we can hope to make this term is zero. If we had an unlimited supply of
data (and unlimited computational resources), we could in principle ﬁnd the regres-
sion function h(x) to any desired degree of accuracy, and this would represent the
optimal choice for y(x). However, in practice we have a data set D containing only
a ﬁnite number N of data points, and consequently we do not know the regression
function h(x) exactly.
If we model the h(x) using a parametric function y(x, w) governed by a pa-
rameter vector w, then from a Bayesian perspective the uncertainty in our model is
expressed through a posterior distribution over w. A frequentist treatment, however,
involves making a point estimate of w based on the data set D, and tries instead
to interpret the uncertainty of this estimate through the following thought experi-
ment. Suppose we had a large number of data sets each of size N and each drawn
independently from the distribution p(t, x). For any given data set D, we can run
our learning algorithm and obtain a prediction function y(x; D). Different data sets
from the ensemble will give different functions and consequently different values of
the squared loss. The performance of a particular learning algorithm is then assessed
by taking the average over this ensemble of data sets.
Consider the integrand of the ﬁrst term in (3.37), which for a particular data set
D takes the form
{y(x; D) −h(x)}2.
(3.38)
Because this quantity will be dependent on the particular data set D, we take its aver-
age over the ensemble of data sets. If we add and subtract the quantity ED[y(x; D)]


---
**Page 167**
3.2. The Bias-Variance Decomposition
149
inside the braces, and then expand, we obtain
{y(x; D) −ED[y(x; D)] + ED[y(x; D)] −h(x)}2
=
{y(x; D) −ED[y(x; D)]}2 + {ED[y(x; D)] −h(x)}2
+2{y(x; D) −ED[y(x; D)]}{ED[y(x; D)] −h(x)}.
(3.39)
We now take the expectation of this expression with respect to D and note that the
ﬁnal term will vanish, giving
ED

{y(x; D) −h(x)}2	
=
{ED[y(x; D)] −h(x)}2
(
)*
+
(bias)2
+ ED

{y(x; D) −ED[y(x; D)]}2	
(
)*
+
variance
. (3.40)
We see that the expected squared difference between y(x; D) and the regression
function h(x) can be expressed as the sum of two terms. The ﬁrst term, called the
squared bias, represents the extent to which the average prediction over all data sets
differs from the desired regression function. The second term, called the variance,
measures the extent to which the solutions for individual data sets vary around their
average, and hence this measures the extent to which the function y(x; D) is sensitive
to the particular choice of data set. We shall provide some intuition to support these
deﬁnitions shortly when we consider a simple example.
So far, we have considered a single input value x. If we substitute this expansion
back into (3.37), we obtain the following decomposition of the expected squared loss
expected loss = (bias)2 + variance + noise
(3.41)
where
(bias)2
=

{ED[y(x; D)] −h(x)}2p(x) dx
(3.42)
variance
=

ED

{y(x; D) −ED[y(x; D)]}2	
p(x) dx
(3.43)
noise
=

{h(x) −t}2p(x, t) dx dt
(3.44)
and the bias and variance terms now refer to integrated quantities.
Our goal is to minimize the expected loss, which we have decomposed into the
sum of a (squared) bias, a variance, and a constant noise term. As we shall see, there
is a trade-off between bias and variance, with very ﬂexible models having low bias
and high variance, and relatively rigid models having high bias and low variance.
The model with the optimal predictive capability is the one that leads to the best
balance between bias and variance. This is illustrated by considering the sinusoidal
data set from Chapter 1. Here we generate 100 data sets, each containing N = 25
Appendix A
data points, independently from the sinusoidal curve h(x) = sin(2πx). The data
sets are indexed by l = 1, . . . , L, where L = 100, and for each data set D(l) we


---
**Page 168**
150
3. LINEAR MODELS FOR REGRESSION
x
t
ln λ = 2.6
0
1
−1
0
1
x
t
0
1
−1
0
1
x
t
ln λ = −0.31
0
1
−1
0
1
x
t
0
1
−1
0
1
x
t
ln λ = −2.4
0
1
−1
0
1
x
t
0
1
−1
0
1
Figure 3.5
Illustration of the dependence of bias and variance on model complexity, governed by a regulariza-
tion parameter λ, using the sinusoidal data set from Chapter 1. There are L = 100 data sets, each having N = 25
data points, and there are 24 Gaussian basis functions in the model so that the total number of parameters is
M = 25 including the bias parameter. The left column shows the result of ﬁtting the model to the data sets for
various values of ln λ (for clarity, only 20 of the 100 ﬁts are shown). The right column shows the corresponding
average of the 100 ﬁts (red) along with the sinusoidal function from which the data sets were generated (green).


---
**Page 169**
3.2. The Bias-Variance Decomposition
151
Figure 3.6
Plot of squared bias and variance,
together with their sum, correspond-
ing to the results shown in Fig-
ure 3.5. Also shown is the average
test set error for a test data set size
of 1000 points. The minimum value
of (bias)2 + variance occurs around
ln λ = −0.31, which is close to the
value that gives the minimum error
on the test data.
ln λ
−3
−2
−1
0
1
2
0
0.03
0.06
0.09
0.12
0.15
(bias)2
variance
(bias)2 + variance
test error
ﬁt a model with 24 Gaussian basis functions by minimizing the regularized error
function (3.27) to give a prediction function y(l)(x) as shown in Figure 3.5.
The
top row corresponds to a large value of the regularization coefﬁcient λ that gives low
variance (because the red curves in the left plot look similar) but high bias (because
the two curves in the right plot are very different). Conversely on the bottom row, for
which λ is small, there is large variance (shown by the high variability between the
red curves in the left plot) but low bias (shown by the good ﬁt between the average
model ﬁt and the original sinusoidal function). Note that the result of averaging many
solutions for the complex model with M = 25 is a very good ﬁt to the regression
function, which suggests that averaging may be a beneﬁcial procedure. Indeed, a
weighted averaging of multiple solutions lies at the heart of a Bayesian approach,
although the averaging is with respect to the posterior distribution of parameters, not
with respect to multiple data sets.
We can also examine the bias-variance trade-off quantitatively for this example.
The average prediction is estimated from
y(x) = 1
L
L

l=1
y(l)(x)
(3.45)
and the integrated squared bias and integrated variance are then given by
(bias)2
=
1
N
N

n=1
{y(xn) −h(xn)}2
(3.46)
variance
=
1
N
N

n=1
1
L
L

l=1

y(l)(xn) −y(xn)2
(3.47)
where the integral over x weighted by the distribution p(x) is approximated by a
ﬁnite sum over data points drawn from that distribution. These quantities, along
with their sum, are plotted as a function of ln λ in Figure 3.6. We see that small
values of λ allow the model to become ﬁnely tuned to the noise on each individual


---
**Page 170**
152
3. LINEAR MODELS FOR REGRESSION
data set leading to large variance. Conversely, a large value of λ pulls the weight
parameters towards zero leading to large bias.
Although the bias-variance decomposition may provide some interesting in-
sights into the model complexity issue from a frequentist perspective, it is of lim-
ited practical value, because the bias-variance decomposition is based on averages
with respect to ensembles of data sets, whereas in practice we have only the single
observed data set. If we had a large number of independent training sets of a given
size, we would be better off combining them into a single large training set, which
of course would reduce the level of over-ﬁtting for a given model complexity.
Given these limitations, we turn in the next section to a Bayesian treatment of
linear basis function models, which not only provides powerful insights into the
issues of over-ﬁtting but which also leads to practical techniques for addressing the
question model complexity.
3.3. Bayesian Linear Regression
In our discussion of maximum likelihood for setting the parameters of a linear re-
gression model, we have seen that the effective model complexity, governed by the
number of basis functions, needs to be controlled according to the size of the data
set. Adding a regularization term to the log likelihood function means the effective
model complexity can then be controlled by the value of the regularization coefﬁ-
cient, although the choice of the number and form of the basis functions is of course
still important in determining the overall behaviour of the model.
This leaves the issue of deciding the appropriate model complexity for the par-
ticular problem, which cannot be decided simply by maximizing the likelihood func-
tion, because this always leads to excessively complex models and over-ﬁtting. In-
dependent hold-out data can be used to determine model complexity, as discussed
in Section 1.3, but this can be both computationally expensive and wasteful of valu-
able data. We therefore turn to a Bayesian treatment of linear regression, which will
avoid the over-ﬁtting problem of maximum likelihood, and which will also lead to
automatic methods of determining model complexity using the training data alone.
Again, for simplicity we will focus on the case of a single target variable t. Ex-
tension to multiple target variables is straightforward and follows the discussion of
Section 3.1.5.
3.3.1
Parameter distribution
We begin our discussion of the Bayesian treatment of linear regression by in-
troducing a prior probability distribution over the model parameters w. For the mo-
ment, we shall treat the noise precision parameter β as a known constant. First note
that the likelihood function p(t|w) deﬁned by (3.10) is the exponential of a quadratic
function of w. The corresponding conjugate prior is therefore given by a Gaussian
distribution of the form
p(w) = N(w|m0, S0)
(3.48)
having mean m0 and covariance S0.


---
**Page 171**
3.3. Bayesian Linear Regression
153
Next we compute the posterior distribution, which is proportional to the product
of the likelihood function and the prior. Due to the choice of a conjugate Gaus-
sian prior distribution, the posterior will also be Gaussian. We can evaluate this
distribution by the usual procedure of completing the square in the exponential, and
then ﬁnding the normalization coefﬁcient using the standard result for a normalized
Gaussian. However, we have already done the necessary work in deriving the gen-
Exercise 3.7
eral result (2.116), which allows us to write down the posterior distribution directly
in the form
p(w|t) = N(w|mN, SN)
(3.49)
where
mN
=
SN

S−1
0 m0 + βΦTt
(3.50)
S−1
N
=
S−1
0
+ βΦTΦ.
(3.51)
Note that because the posterior distribution is Gaussian, its mode coincides with its
mean. Thus the maximum posterior weight vector is simply given by wMAP = mN.
If we consider an inﬁnitely broad prior S0 = α−1I with α →0, the mean mN
of the posterior distribution reduces to the maximum likelihood value wML given
by (3.15). Similarly, if N = 0, then the posterior distribution reverts to the prior.
Furthermore, if data points arrive sequentially, then the posterior distribution at any
stage acts as the prior distribution for the subsequent data point, such that the new
posterior distribution is again given by (3.49).
Exercise 3.8
For the remainder of this chapter, we shall consider a particular form of Gaus-
sian prior in order to simplify the treatment. Speciﬁcally, we consider a zero-mean
isotropic Gaussian governed by a single precision parameter α so that
p(w|α) = N(w|0, α−1I)
(3.52)
and the corresponding posterior distribution over w is then given by (3.49) with
mN
=
βSNΦTt
(3.53)
S−1
N
=
αI + βΦTΦ.
(3.54)
The log of the posterior distribution is given by the sum of the log likelihood and
the log of the prior and, as a function of w, takes the form
ln p(w|t) = −β
2
N

n=1
{tn −wTφ(xn)}2 −α
2 wTw + const.
(3.55)
Maximization of this posterior distribution with respect to w is therefore equiva-
lent to the minimization of the sum-of-squares error function with the addition of a
quadratic regularization term, corresponding to (3.27) with λ = α/β.
We can illustrate Bayesian learning in a linear basis function model, as well as
the sequential update of a posterior distribution, using a simple example involving
straight-line ﬁtting. Consider a single input variable x, a single target variable t and


---
**Page 172**
154
3. LINEAR MODELS FOR REGRESSION
a linear model of the form y(x, w) = w0 + w1x. Because this has just two adap-
tive parameters, we can plot the prior and posterior distributions directly in parameter
space. We generate synthetic data from the function f(x, a) = a0+a1x with param-
eter values a0 = −0.3 and a1 = 0.5 by ﬁrst choosing values of xn from the uniform
distribution U(x|−1, 1), then evaluating f(xn, a), and ﬁnally adding Gaussian noise
with standard deviation of 0.2 to obtain the target values tn. Our goal is to recover
the values of a0 and a1 from such data, and we will explore the dependence on the
size of the data set. We assume here that the noise variance is known and hence we
set the precision parameter to its true value β = (1/0.2)2 = 25. Similarly, we ﬁx
the parameter α to 2.0. We shall shortly discuss strategies for determining α and
β from the training data. Figure 3.7 shows the results of Bayesian learning in this
model as the size of the data set is increased and demonstrates the sequential nature
of Bayesian learning in which the current posterior distribution forms the prior when
a new data point is observed. It is worth taking time to study this ﬁgure in detail as
it illustrates several important aspects of Bayesian inference. The ﬁrst row of this
ﬁgure corresponds to the situation before any data points are observed and shows a
plot of the prior distribution in w space together with six samples of the function
y(x, w) in which the values of w are drawn from the prior. In the second row, we
see the situation after observing a single data point. The location (x, t) of the data
point is shown by a blue circle in the right-hand column. In the left-hand column is a
plot of the likelihood function p(t|x, w) for this data point as a function of w. Note
that the likelihood function provides a soft constraint that the line must pass close to
the data point, where close is determined by the noise precision β. For comparison,
the true parameter values a0 = −0.3 and a1 = 0.5 used to generate the data set
are shown by a white cross in the plots in the left column of Figure 3.7. When we
multiply this likelihood function by the prior from the top row, and normalize, we
obtain the posterior distribution shown in the middle plot on the second row. Sam-
ples of the regression function y(x, w) obtained by drawing samples of w from this
posterior distribution are shown in the right-hand plot. Note that these sample lines
all pass close to the data point. The third row of this ﬁgure shows the effect of ob-
serving a second data point, again shown by a blue circle in the plot in the right-hand
column. The corresponding likelihood function for this second data point alone is
shown in the left plot. When we multiply this likelihood function by the posterior
distribution from the second row, we obtain the posterior distribution shown in the
middle plot of the third row. Note that this is exactly the same posterior distribution
as would be obtained by combining the original prior with the likelihood function
for the two data points. This posterior has now been inﬂuenced by two data points,
and because two points are sufﬁcient to deﬁne a line this already gives a relatively
compact posterior distribution. Samples from this posterior distribution give rise to
the functions shown in red in the third column, and we see that these functions pass
close to both of the data points. The fourth row shows the effect of observing a total
of 20 data points. The left-hand plot shows the likelihood function for the 20th data
point alone, and the middle plot shows the resulting posterior distribution that has
now absorbed information from all 20 observations. Note how the posterior is much
sharper than in the third row. In the limit of an inﬁnite number of data points, the


---
**Page 173**
3.3. Bayesian Linear Regression
155
Figure 3.7
Illustration of sequential Bayesian learning for a simple linear model of the form y(x, w) =
w0 + w1x. A detailed description of this ﬁgure is given in the text.


---
**Page 174**
156
3. LINEAR MODELS FOR REGRESSION
posterior distribution would become a delta function centred on the true parameter
values, shown by the white cross.
Other forms of prior over the parameters can be considered. For instance, we
can generalize the Gaussian prior to give
p(w|α) =
q
2
α
2
1/q
1
Γ(1/q)
M
exp

−α
2
M

j=1
|wj|q

(3.56)
in which q = 2 corresponds to the Gaussian distribution, and only in this case is the
prior conjugate to the likelihood function (3.10). Finding the maximum of the poste-
rior distribution over w corresponds to minimization of the regularized error function
(3.29). In the case of the Gaussian prior, the mode of the posterior distribution was
equal to the mean, although this will no longer hold if q ̸= 2.
3.3.2
Predictive distribution
In practice, we are not usually interested in the value of w itself but rather in
making predictions of t for new values of x. This requires that we evaluate the
predictive distribution deﬁned by
p(t|t, α, β) =

p(t|w, β)p(w|t, α, β) dw
(3.57)
in which t is the vector of target values from the training set, and we have omitted the
corresponding input vectors from the right-hand side of the conditioning statements
to simplify the notation. The conditional distribution p(t|x, w, β) of the target vari-
able is given by (3.8), and the posterior weight distribution is given by (3.49). We
see that (3.57) involves the convolution of two Gaussian distributions, and so making
use of the result (2.115) from Section 8.1.4, we see that the predictive distribution
takes the form
Exercise 3.10
p(t|x, t, α, β) = N(t|mT
Nφ(x), σ2
N(x))
(3.58)
where the variance σ2
N(x) of the predictive distribution is given by
σ2
N(x) = 1
β + φ(x)TSNφ(x).
(3.59)
The ﬁrst term in (3.59) represents the noise on the data whereas the second term
reﬂects the uncertainty associated with the parameters w. Because the noise process
and the distribution of w are independent Gaussians, their variances are additive.
Note that, as additional data points are observed, the posterior distribution becomes
narrower. As a consequence it can be shown (Qazaz et al., 1997) that σ2
N+1(x) ⩽
σ2
N(x). In the limit N →∞, the second term in (3.59) goes to zero, and the variance
Exercise 3.11
of the predictive distribution arises solely from the additive noise governed by the
parameter β.
As an illustration of the predictive distribution for Bayesian linear regression
models, let us return to the synthetic sinusoidal data set of Section 1.1. In Figure 3.8,


---
**Page 175**
3.3. Bayesian Linear Regression
157
x
t
0
1
−1
0
1
x
t
0
1
−1
0
1
x
t
0
1
−1
0
1
x
t
0
1
−1
0
1
Figure 3.8
Examples of the predictive distribution (3.58) for a model consisting of 9 Gaussian basis functions
of the form (3.4) using the synthetic sinusoidal data set of Section 1.1. See the text for a detailed discussion.
we ﬁt a model comprising a linear combination of Gaussian basis functions to data
sets of various sizes and then look at the corresponding posterior distributions. Here
the green curves correspond to the function sin(2πx) from which the data points
were generated (with the addition of Gaussian noise). Data sets of size N = 1,
N = 2, N = 4, and N = 25 are shown in the four plots by the blue circles. For
each plot, the red curve shows the mean of the corresponding Gaussian predictive
distribution, and the red shaded region spans one standard deviation either side of
the mean. Note that the predictive uncertainty depends on x and is smallest in the
neighbourhood of the data points. Also note that the level of uncertainty decreases
as more data points are observed.
The plots in Figure 3.8 only show the point-wise predictive variance as a func-
tion of x. In order to gain insight into the covariance between the predictions at
different values of x, we can draw samples from the posterior distribution over w,
and then plot the corresponding functions y(x, w), as shown in Figure 3.9.


---
**Page 176**
158
3. LINEAR MODELS FOR REGRESSION
x
t
0
1
−1
0
1
x
t
0
1
−1
0
1
x
t
0
1
−1
0
1
x
t
0
1
−1
0
1
Figure 3.9
Plots of the function y(x, w) using samples from the posterior distributions over w corresponding to
the plots in Figure 3.8.
If we used localized basis functions such as Gaussians, then in regions away
from the basis function centres, the contribution from the second term in the predic-
tive variance (3.59) will go to zero, leaving only the noise contribution β−1. Thus,
the model becomes very conﬁdent in its predictions when extrapolating outside the
region occupied by the basis functions, which is generally an undesirable behaviour.
This problem can be avoided by adopting an alternative Bayesian approach to re-
gression known as a Gaussian process.
Section 6.4
Note that, if both w and β are treated as unknown, then we can introduce a
conjugate prior distribution p(w, β) that, from the discussion in Section 2.3.6, will
be given by a Gaussian-gamma distribution (Denison et al., 2002). In this case, the
Exercise 3.12
predictive distribution is a Student’s t-distribution.
Exercise 3.13


---
**Page 177**
3.3. Bayesian Linear Regression
159
Figure 3.10
The equivalent ker-
nel k(x, x′) for the Gaussian basis
functions in Figure 3.1, shown as
a plot of x versus x′, together with
three slices through this matrix cor-
responding to three different values
of x. The data set used to generate
this kernel comprised 200 values of
x equally spaced over the interval
(−1, 1).
3.3.3
Equivalent kernel
The posterior mean solution (3.53) for the linear basis function model has an in-
teresting interpretation that will set the stage for kernel methods, including Gaussian
processes. If we substitute (3.53) into the expression (3.3), we see that the predictive
Chapter 6
mean can be written in the form
y(x, mN) = mT
Nφ(x) = βφ(x)TSNΦTt =
N

n=1
βφ(x)TSNφ(xn)tn
(3.60)
where SN is deﬁned by (3.51). Thus the mean of the predictive distribution at a point
x is given by a linear combination of the training set target variables tn, so that we
can write
y(x, mN) =
N

n=1
k(x, xn)tn
(3.61)
where the function
k(x, x′) = βφ(x)TSNφ(x′)
(3.62)
is known as the smoother matrix or the equivalent kernel. Regression functions, such
as this, which make predictions by taking linear combinations of the training set
target values are known as linear smoothers. Note that the equivalent kernel depends
on the input values xn from the data set because these appear in the deﬁnition of
SN. The equivalent kernel is illustrated for the case of Gaussian basis functions in
Figure 3.10 in which the kernel functions k(x, x′) have been plotted as a function of
x′ for three different values of x. We see that they are localized around x, and so the
mean of the predictive distribution at x, given by y(x, mN), is obtained by forming
a weighted combination of the target values in which data points close to x are given
higher weight than points further removed from x. Intuitively, it seems reasonable
that we should weight local evidence more strongly than distant evidence. Note that
this localization property holds not only for the localized Gaussian basis functions
but also for the nonlocal polynomial and sigmoidal basis functions, as illustrated in
Figure 3.11.


---
**Page 178**
160
3. LINEAR MODELS FOR REGRESSION
Figure 3.11
Examples of equiva-
lent kernels k(x, x′) for x
=
0
plotted as a function of x′, corre-
sponding (left) to the polynomial ba-
sis functions and (right) to the sig-
moidal basis functions shown in Fig-
ure 3.1. Note that these are local-
ized functions of x′ even though the
corresponding basis functions are
nonlocal.
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
Further insight into the role of the equivalent kernel can be obtained by consid-
ering the covariance between y(x) and y(x′), which is given by
cov[y(x), y(x′)]
=
cov[φ(x)Tw, wTφ(x′)]
=
φ(x)TSNφ(x′) = β−1k(x, x′)
(3.63)
where we have made use of (3.49) and (3.62). From the form of the equivalent
kernel, we see that the predictive mean at nearby points will be highly correlated,
whereas for more distant pairs of points the correlation will be smaller.
The predictive distribution shown in Figure 3.8 allows us to visualize the point-
wise uncertainty in the predictions, governed by (3.59). However, by drawing sam-
ples from the posterior distribution over w, and plotting the corresponding model
functions y(x, w) as in Figure 3.9, we are visualizing the joint uncertainty in the
posterior distribution between the y values at two (or more) x values, as governed by
the equivalent kernel.
The formulation of linear regression in terms of a kernel function suggests an
alternative approach to regression as follows. Instead of introducing a set of basis
functions, which implicitly determines an equivalent kernel, we can instead deﬁne
a localized kernel directly and use this to make predictions for new input vectors x,
given the observed training set. This leads to a practical framework for regression
(and classiﬁcation) called Gaussian processes, which will be discussed in detail in
Section 6.4.
We have seen that the effective kernel deﬁnes the weights by which the training
set target values are combined in order to make a prediction at a new value of x, and
it can be shown that these weights sum to one, in other words
N

n=1
k(x, xn) = 1
(3.64)
for all values of x. This intuitively pleasing result can easily be proven informally
Exercise 3.14
by noting that the summation is equivalent to considering the predictive mean y(x)
for a set of target data in which tn = 1 for all n. Provided the basis functions are
linearly independent, that there are more data points than basis functions, and that
one of the basis functions is constant (corresponding to the bias parameter), then it is
clear that we can ﬁt the training data exactly and hence that the predictive mean will

