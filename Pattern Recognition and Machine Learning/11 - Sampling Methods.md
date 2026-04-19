# 11 - Sampling Methods
*Pages 523-558 from Pattern Recognition and Machine Learning*

---
**Page 523**
508
10. APPROXIMATE INFERENCE
−2
−1
0
1
2
3
4
0
0.2
0.4
0.6
0.8
1
−2
−1
0
1
2
3
4
0
10
20
30
40
Figure 10.14
Illustration of the expectation propagation approximation using a Gaussian distribution for the
example considered earlier in Figures 4.14 and 10.1. The left-hand plot shows the original distribution (yellow)
along with the Laplace (red), global variational (green), and EP (blue) approximations, and the right-hand plot
shows the corresponding negative logarithms of the distributions. Note that the EP distribution is broader than
that variational inference, as a consequence of the different form of KL divergence.
where Zj is the normalization constant given by
Zj =

fj(θ)q\j(θ) dθ.
(10.197)
We now determine a revised factor fj(θ) by minimizing the Kullback-Leibler diver-
gence
KL
 fj(θ)q\j(θ)
Zj
'''' qnew(θ)

.
(10.198)
This is easily solved because the approximating distribution qnew(θ) is from the ex-
ponential family, and so we can appeal to the result (10.187), which tells us that the
parameters of qnew(θ) are obtained by matching its expected sufﬁcient statistics to
the corresponding moments of (10.196). We shall assume that this is a tractable oper-
ation. For example, if we choose q(θ) to be a Gaussian distribution N(θ|µ, Σ), then
µ is set equal to the mean of the (unnormalized) distribution fj(θ)q\j(θ), and Σ is
set to its covariance. More generally, it is straightforward to obtain the required ex-
pectations for any member of the exponential family, provided it can be normalized,
because the expected statistics can be related to the derivatives of the normalization
coefﬁcient, as given by (2.226). The EP approximation is illustrated in Figure 10.14.
From (10.193), we see that the revised factor fj(θ) can be found by taking
qnew(θ) and dividing out the remaining factors so that
fj(θ) = K qnew(θ)
q\j(θ)
(10.199)
where we have used (10.195). The coefﬁcient K is determined by multiplying both


---
**Page 524**
10.7. Expectation Propagation
509
sides of (10.199) by q\i(θ) and integrating to give
K =

fj(θ)q\j(θ) dθ
(10.200)
where we have used the fact that qnew(θ) is normalized. The value of K can therefore
be found by matching zeroth-order moments

fj(θ)q\j(θ) dθ =

fj(θ)q\j(θ) dθ.
(10.201)
Combining this with (10.197), we then see that K = Zj and so can be found by
evaluating the integral in (10.197).
In practice, several passes are made through the set of factors, revising each
factor in turn. The posterior distribution p(θ|D) is then approximated using (10.191),
and the model evidence p(D) can be approximated by using (10.190) with the factors
fi(θ) replaced by their approximations fi(θ).
Expectation Propagation
We are given a joint distribution over observed data D and stochastic variables
θ in the form of a product of factors
p(D, θ) =

i
fi(θ)
(10.202)
and we wish to approximate the posterior distribution p(θ|D) by a distribution
of the form
q(θ) = 1
Z

i
fi(θ).
(10.203)
We also wish to approximate the model evidence p(D).
1. Initialize all of the approximating factors fi(θ).
2. Initialize the posterior approximation by setting
q(θ) ∝

i
fi(θ).
(10.204)
3. Until convergence:
(a) Choose a factor fj(θ) to reﬁne.
(b) Remove fj(θ) from the posterior by division
q\j(θ) = q(θ)
fj(θ)
.
(10.205)


---
**Page 525**
510
10. APPROXIMATE INFERENCE
(c) Evaluate the new posterior by setting the sufﬁcient statistics (moments)
of qnew(θ) equal to those of q\j(θ)fj(θ), including evaluation of the
normalization constant
Zj =

q\j(θ)fj(θ) dθ.
(10.206)
(d) Evaluate and store the new factor
fj(θ) = Zj
qnew(θ)
q\j(θ) .
(10.207)
4. Evaluate the approximation to the model evidence
p(D) ≃
 
i
fi(θ) dθ.
(10.208)
A special case of EP, known as assumed density ﬁltering (ADF) or moment
matching (Maybeck, 1982; Lauritzen, 1992; Boyen and Koller, 1998; Opper and
Winther, 1999), is obtained by initializing all of the approximating factors except
the ﬁrst to unity and then making one pass through the factors updating each of them
once. Assumed density ﬁltering can be appropriate for on-line learning in which data
points are arriving in a sequence and we need to learn from each data point and then
discard it before considering the next point. However, in a batch setting we have the
opportunity to re-use the data points many times in order to achieve improved ac-
curacy, and it is this idea that is exploited in expectation propagation. Furthermore,
if we apply ADF to batch data, the results will have an undesirable dependence on
the (arbitrary) order in which the data points are considered, which again EP can
overcome.
One disadvantage of expectation propagation is that there is no guarantee that
the iterations will converge. However, for approximations q(θ) in the exponential
family, if the iterations do converge, the resulting solution will be a stationary point
of a particular energy function (Minka, 2001a), although each iteration of EP does
not necessarily decrease the value of this energy function. This is in contrast to
variational Bayes, which iteratively maximizes a lower bound on the log marginal
likelihood, in which each iteration is guaranteed not to decrease the bound. It is
possible to optimize the EP cost function directly, in which case it is guaranteed
to converge, although the resulting algorithms can be slower and more complex to
implement.
Another difference between variational Bayes and EP arises from the form of
KL divergence that is minimized by the two algorithms, because the former mini-
mizes KL(q∥p) whereas the latter minimizes KL(p∥q). As we saw in Figure 10.3,
for distributions p(θ) which are multimodal, minimizing KL(p∥q) can lead to poor
approximations. In particular, if EP is applied to mixtures the results are not sen-
sible because the approximation tries to capture all of the modes of the posterior
distribution. Conversely, in logistic-type models, EP often out-performs both local
variational methods and the Laplace approximation (Kuss and Rasmussen, 2006).


---
**Page 526**
10.7. Expectation Propagation
511
Figure 10.15
Illustration of the clutter problem
for a data space dimensionality of
D = 1. Training data points, de-
noted by the crosses, are drawn
from a mixture of two Gaussians
with components shown in red
and green. The goal is to infer the
mean of the green Gaussian from
the observed data.
θ
x
−5
0
5
10
10.7.1
Example: The clutter problem
Following Minka (2001b), we illustrate the EP algorithm using a simple exam-
ple in which the goal is to infer the mean θ of a multivariate Gaussian distribution
over a variable x given a set of observations drawn from that distribution. To make
the problem more interesting, the observations are embedded in background clutter,
which itself is also Gaussian distributed, as illustrated in Figure 10.15. The distribu-
tion of observed values x is therefore a mixture of Gaussians, which we take to be
of the form
p(x|θ) = (1 −w)N(x|θ, I) + wN(x|0, aI)
(10.209)
where w is the proportion of background clutter and is assumed to be known. The
prior over θ is taken to be Gaussian
p(θ) = N(θ|0, bI)
(10.210)
and Minka (2001a) chooses the parameter values a = 10, b = 100 and w = 0.5.
The joint distribution of N observations D = {x1, . . . , xN} and θ is given by
p(D, θ) = p(θ)
N

n=1
p(xn|θ)
(10.211)
and so the posterior distribution comprises a mixture of 2N Gaussians. Thus the
computational cost of solving this problem exactly would grow exponentially with
the size of the data set, and so an exact solution is intractable for moderately large
N.
To apply EP to the clutter problem, we ﬁrst identify the factors f0(θ) = p(θ)
and fn(θ) = p(xn|θ). Next we select an approximating distribution from the expo-
nential family, and for this example it is convenient to choose a spherical Gaussian
q(θ) = N(θ|m, vI).
(10.212)


---
**Page 527**
512
10. APPROXIMATE INFERENCE
The factor approximations will therefore take the form of exponential-quadratic
functions of the form
fn(θ) = snN(θ|mn, vnI)
(10.213)
where n = 1, . . . , N, and we set f0(θ) equal to the prior p(θ). Note that the use of
N(θ|·, ·) does not imply that the right-hand side is a well-deﬁned Gaussian density
(in fact, as we shall see, the variance parameter vn can be negative) but is simply a
convenient shorthand notation. The approximations fn(θ), for n = 1, . . . , N, can
be initialized to unity, corresponding to sn = (2πvn)D/2, vn →∞and mn = 0,
where D is the dimensionality of x and hence of θ. The initial q(θ), deﬁned by
(10.191), is therefore equal to the prior.
We then iteratively reﬁne the factors by taking one factor fn(θ) at a time and
applying (10.205), (10.206), and (10.207). Note that we do not need to revise the
term f0(θ) because an EP update will leave this term unchanged. Here we state the
Exercise 10.37
results and leave the reader to ﬁll in the details.
First we remove the current estimate fn(θ) from q(θ) by division using (10.205)
to give q\n(θ), which has mean and inverse variance given by
Exercise 10.38
m\n
=
m + v\nv−1
n (m −mn)
(10.214)
(v\n)−1
=
v−1 −v−1
n .
(10.215)
Next we evaluate the normalization constant Zn using (10.206) to give
Zn = (1 −w)N(xn|m\n, (v\n + 1)I) + wN(xn|0, aI).
(10.216)
Similarly, we compute the mean and variance of qnew(θ) by ﬁnding the mean and
variance of q\n(θ)fn(θ) to give
Exercise 10.39
m
=
m\n + ρn
v\n
v\n + 1(xn −m\n)
(10.217)
v
=
v\n −ρn
(v\n)2
v\n + 1 + ρn(1 −ρn)(v\n)2∥xn −m\n∥2
D(v\n + 1)2
(10.218)
where the quantity
ρn = 1 −w
Zn
N(xn|0, aI)
(10.219)
has a simple interpretation as the probability of the point xn not being clutter. Then
we use (10.207) to compute the reﬁned factor fn(θ) whose parameters are given by
v−1
n
=
(vnew)−1 −(v\n)−1
(10.220)
mn
=
m\n + (vn + v\n)(v\n)−1(mnew −m\n)
(10.221)
sn
=
Zn
(2πvn)D/2N(mn|m\n, (vn + v\n)I).
(10.222)
This reﬁnement process is repeated until a suitable termination criterion is satisﬁed,
for instance that the maximum change in parameter values resulting from a complete


---
**Page 528**
10.7. Expectation Propagation
513
θ
−5
0
5
10
θ
−5
0
5
10
Figure 10.16
Examples of the approximation of speciﬁc factors for a one-dimensional version of the clutter
problem, showing fn(θ) in blue, efn(θ) in red, and q\n(θ) in green. Notice that the current form for q\n(θ) controls
the range of θ over which efn(θ) will be a good approximation to fn(θ).
pass through all factors is less than some threshold. Finally, we use (10.208) to
evaluate the approximation to the model evidence, given by
p(D) ≃(2πvnew)D/2 exp(B/2)
N

n=1

sn(2πvn)−D/2
(10.223)
where
B = (mnew)Tmnew
v
−
N

n=1
mT
nmn
vn
.
(10.224)
Examples factor approximations for the clutter problem with a one-dimensional pa-
rameter space θ are shown in Figure 10.16. Note that the factor approximations can
have inﬁnite or even negative values for the ‘variance’ parameter vn. This simply
corresponds to approximations that curve upwards instead of downwards and are not
necessarily problematic provided the overall approximate posterior q(θ) has posi-
tive variance. Figure 10.17 compares the performance of EP with variational Bayes
(mean ﬁeld theory) and the Laplace approximation on the clutter problem.
10.7.2
Expectation propagation on graphs
So far in our general discussion of EP, we have allowed the factors fi(θ) in the
distribution p(θ) to be functions of all of the components of θ, and similarly for the
approximating factors f(θ) in the approximating distribution q(θ). We now consider
situations in which the factors depend only on subsets of the variables. Such restric-
tions can be conveniently expressed using the framework of probabilistic graphical
models, as discussed in Chapter 8. Here we use a factor graph representation because
this encompasses both directed and undirected graphs.


---
**Page 529**
514
10. APPROXIMATE INFERENCE
ep
vb
laplace
Posterior mean
FLOPS
Error
10
4
10
6
10
−5
10
0
ep
vb
laplace
Evidence
FLOPS
Error
10
4
10
6
10
−204
10
−202
10
−200
Figure 10.17
Comparison of expectation propagation, variational inference, and the Laplace approximation on
the clutter problem. The left-hand plot shows the error in the predicted posterior mean versus the number of
ﬂoating point operations, and the right-hand plot shows the corresponding results for the model evidence.
We shall focus on the case in which the approximating distribution is fully fac-
torized, and we shall show that in this case expectation propagation reduces to loopy
belief propagation (Minka, 2001a). To start with, we show this in the context of a
simple example, and then we shall explore the general case.
First of all, recall from (10.17) that if we minimize the Kullback-Leibler diver-
gence KL(p∥q) with respect to a factorized distribution q, then the optimal solution
for each factor is simply the corresponding marginal of p.
Now consider the factor graph shown on the left in Figure 10.18, which was
introduced earlier in the context of the sum-product algorithm. The joint distribution
Section 8.4.4
is given by
p(x) = fa(x1, x2)fb(x2, x3)fc(x2, x4).
(10.225)
We seek an approximation q(x) that has the same factorization, so that
q(x) ∝fa(x1, x2)fb(x2, x3)fc(x2, x4).
(10.226)
Note that normalization constants have been omitted, and these can be re-instated at
the end by local normalization, as is generally done in belief propagation. Now sup-
pose we restrict attention to approximations in which the factors themselves factorize
with respect to the individual variables so that
q(x) ∝fa1(x1)fa2(x2)fb2(x2)fb3(x3)fc2(x2)fc4(x4)
(10.227)
which corresponds to the factor graph shown on the right in Figure 10.18. Because
the individual factors are factorized, the overall distribution q(x) is itself fully fac-
torized.
Now we apply the EP algorithm using the fully factorized approximation. Sup-
pose that we have initialized all of the factors and that we choose to reﬁne factor


---
**Page 530**
10.7. Expectation Propagation
515
x1
x2
x3
x4
fa
fb
fc
x1
x2
x3
x4
˜fa1
˜fa2
˜fb2
˜fb3
˜fc2
˜fc4
Figure 10.18
On the left is a simple factor graph from Figure 8.51 and reproduced here for convenience. On
the right is the corresponding factorized approximation.
fb(x2, x3) = fb2(x2)fb3(x3). We ﬁrst remove this factor from the approximating
distribution to give
q\b(x) = fa1(x1)fa2(x2)fc2(x2)fc4(x4)
(10.228)
and we then multiply this by the exact factor fb(x2, x3) to give
p(x) = q\b(x)fb(x2, x3) = fa1(x1)fa2(x2)fc2(x2)fc4(x4)fb(x2, x3).
(10.229)
We now ﬁnd qnew(x) by minimizing the Kullback-Leibler divergence KL(p∥qnew).
The result, as noted above, is that qnew(z) comprises the product of factors, one for
each variable xi, in which each factor is given by the corresponding marginal of
p(x). These four marginals are given by
p(x1)
∝
fa1(x1)
(10.230)
p(x2)
∝
fa2(x2)fc2(x2)

x3
fb(x2, x3)
(10.231)
p(x3)
∝

x2

fb(x2, x3)fa2(x2)fc2(x2)

(10.232)
p(x4)
∝
fc4(x4)
(10.233)
and qnew(x) is obtained by multiplying these marginals together. We see that the
only factors in q(x) that change when we update fb(x2, x3) are those that involve
the variables in fb namely x2 and x3. To obtain the reﬁned factor fb(x2, x3) =
fb2(x2)fb3(x3) we simply divide qnew(x) by q\b(x), which gives
fb2(x2)
∝

x3
fb(x2, x3)
(10.234)
fb3(x3)
∝

x2

fb(x2, x3)fa2(x2)fc2(x2)

.
(10.235)


---
**Page 531**
516
10. APPROXIMATE INFERENCE
These are precisely the messages obtained using belief propagation in which mes-
Section 8.4.4
sages from variable nodes to factor nodes have been folded into the messages from
factor nodes to variable nodes. In particular, fb2(x2) corresponds to the message
µfb→x2(x2) sent by factor node fb to variable node x2 and is given by (8.81). Simi-
larly, if we substitute (8.78) into (8.79), we obtain (10.235) in which fa2(x2) corre-
sponds to µfa→x2(x2) and fc2(x2) corresponds to µfc→x2(x2), giving the message
fb3(x3) which corresponds to µfb→x3(x3).
This result differs slightly from standard belief propagation in that messages are
passed in both directions at the same time. We can easily modify the EP procedure
to give the standard form of the sum-product algorithm by updating just one of the
factors at a time, for instance if we reﬁne only fb3(x3), then fb2(x2) is unchanged
by deﬁnition, while the reﬁned version of fb3(x3) is again given by (10.235). If
we are reﬁning only one term at a time, then we can choose the order in which the
reﬁnements are done as we wish. In particular, for a tree-structured graph we can
follow a two-pass update scheme, corresponding to the standard belief propagation
schedule, which will result in exact inference of the variable and factor marginals.
The initialization of the approximation factors in this case is unimportant.
Now let us consider a general factor graph corresponding to the distribution
p(θ) =

i
fi(θi)
(10.236)
where θi represents the subset of variables associated with factor fi. We approximate
this using a fully factorized distribution of the form
q(θ) ∝

i

k
fik(θk)
(10.237)
where θk corresponds to an individual variable node. Suppose that we wish to reﬁne
the particular term fjl(θl) keeping all other terms ﬁxed. We ﬁrst remove the term
fj(θj) from q(θ) to give
q\j(θ) ∝

i̸=j

k
fik(θk)
(10.238)
and then multiply by the exact factor fj(θj). To determine the reﬁned term fjl(θl),
we need only consider the functional dependence on θl, and so we simply ﬁnd the
corresponding marginal of
q\j(θ)fj(θj).
(10.239)
Up to a multiplicative constant, this involves taking the marginal of fj(θj) multiplied
by any terms from q\j(θ) that are functions of any of the variables in θj. Terms that
correspond to other factors fi(θi) for i ̸= j will cancel between numerator and
denominator when we subsequently divide by q\j(θ). We therefore obtain
fjl(θl) ∝

θm̸=l∈θj
fj(θj)

k

m̸=l
fkm(θm).
(10.240)


---
**Page 532**
Exercises
517
We recognize this as the sum-product rule in the form in which messages from vari-
able nodes to factor nodes have been eliminated, as illustrated by the example shown
in Figure 8.50. The quantity fjm(θm) corresponds to the message µfj→θm(θm),
which factor node j sends to variable node m, and the product over k in (10.240)
is over all factors that depend on the variables θm that have variables (other than
variable θl) in common with factor fj(θj). In other words, to compute the outgoing
message from a factor node, we take the product of all the incoming messages from
other factor nodes, multiply by the local factor, and then marginalize.
Thus, the sum-product algorithm arises as a special case of expectation propa-
gation if we use an approximating distribution that is fully factorized. This suggests
that more ﬂexible approximating distributions, corresponding to partially discon-
nected graphs, could be used to achieve higher accuracy. Another generalization is
to group factors fi(θi) together into sets and to reﬁne all the factors in a set together
at each iteration. Both of these approaches can lead to improvements in accuracy
(Minka, 2001b). In general, the problem of choosing the best combination of group-
ing and disconnection is an open research issue.
We have seen that variational message passing and expectation propagation op-
timize two different forms of the Kullback-Leibler divergence. Minka (2005) has
shown that a broad range of message passing algorithms can be derived from a com-
mon framework involving minimization of members of the alpha family of diver-
gences, given by (10.19). These include variational message passing, loopy belief
propagation, and expectation propagation, as well as a range of other algorithms,
which we do not have space to discuss here, such as tree-reweighted message pass-
ing (Wainwright et al., 2005), fractional belief propagation (Wiegerinck and Heskes,
2003), and power EP (Minka, 2004).
Exercises
10.1
(⋆) www
Verify that the log marginal distribution of the observed data ln p(X)
can be decomposed into two terms in the form (10.2) where L(q) is given by (10.3)
and KL(q∥p) is given by (10.4).
10.2
(⋆) Use the properties E[z1] = m1 and E[z2] = m2 to solve the simultaneous equa-
tions (10.13) and (10.15), and hence show that, provided the original distribution
p(z) is nonsingular, the unique solution for the means of the factors in the approxi-
mation distribution is given by E[z1] = µ1 and E[z2] = µ2.
10.3
(⋆⋆) www
Consider a factorized variational distribution q(Z) of the form (10.5).
By using the technique of Lagrange multipliers, verify that minimization of the
Kullback-Leibler divergence KL(p∥q) with respect to one of the factors qi(Zi),
keeping all other factors ﬁxed, leads to the solution (10.17).
10.4
(⋆⋆) Suppose that p(x) is some ﬁxed distribution and that we wish to approximate
it using a Gaussian distribution q(x) = N(x|µ, Σ). By writing down the form of
the KL divergence KL(p∥q) for a Gaussian q(x) and then differentiating, show that


---
**Page 533**
518
10. APPROXIMATE INFERENCE
minimization of KL(p∥q) with respect to µ and Σ leads to the result that µ is given
by the expectation of x under p(x) and that Σ is given by the covariance.
10.5
(⋆⋆) www
Consider a model in which the set of all hidden stochastic variables, de-
noted collectively by Z, comprises some latent variables z together with some model
parameters θ. Suppose we use a variational distribution that factorizes between la-
tent variables and parameters so that q(z, θ) = qz(z)qθ(θ), in which the distribution
qθ(θ) is approximated by a point estimate of the form qθ(θ) = δ(θ −θ0) where θ0
is a vector of free parameters. Show that variational optimization of this factorized
distribution is equivalent to an EM algorithm, in which the E step optimizes qz(z),
and the M step maximizes the expected complete-data log posterior distribution of θ
with respect to θ0.
10.6
(⋆⋆) The alpha family of divergences is deﬁned by (10.19). Show that the Kullback-
Leibler divergence KL(p∥q) corresponds to α →1. This can be done by writing
pϵ = exp(ϵ ln p) = 1 + ϵ ln p + O(ϵ2) and then taking ϵ →0. Similarly show that
KL(q∥p) corresponds to α →−1.
10.7
(⋆⋆) Consider the problem of inferring the mean and precision of a univariate Gaus-
sian using a factorized variational approximation, as considered in Section 10.1.3.
Show that the factor qµ(µ) is a Gaussian of the form N(µ|µN, λ−1
N ) with mean and
precision given by (10.26) and (10.27), respectively. Similarly show that the factor
qτ(τ) is a gamma distribution of the form Gam(τ|aN, bN) with parameters given by
(10.29) and (10.30).
10.8
(⋆)
Consider the variational posterior distribution for the precision of a univariate
Gaussian whose parameters are given by (10.29) and (10.30). By using the standard
results for the mean and variance of the gamma distribution given by (B.27) and
(B.28), show that if we let N →∞, this variational posterior distribution has a
mean given by the inverse of the maximum likelihood estimator for the variance of
the data, and a variance that goes to zero.
10.9
(⋆⋆) By making use of the standard result E[τ] = aN/bN for the mean of a gamma
distribution, together with (10.26), (10.27), (10.29), and (10.30), derive the result
(10.33) for the reciprocal of the expected precision in the factorized variational treat-
ment of a univariate Gaussian.
10.10
(⋆) www
Derive the decomposition given by (10.34) that is used to ﬁnd approxi-
mate posterior distributions over models using variational inference.
10.11
(⋆⋆) www
By using a Lagrange multiplier to enforce the normalization constraint
on the distribution q(m), show that the maximum of the lower bound (10.35) is given
by (10.36).
10.12
(⋆⋆)
Starting from the joint distribution (10.41), and applying the general result
(10.9), show that the optimal variational distribution q⋆(Z) over the latent variables
for the Bayesian mixture of Gaussians is given by (10.48) by verifying the steps
given in the text.


---
**Page 534**
Exercises
519
10.13
(⋆⋆) www
Starting from (10.54), derive the result (10.59) for the optimum vari-
ational posterior distribution over µk and Λk in the Bayesian mixture of Gaussians,
and hence verify the expressions for the parameters of this distribution given by
(10.60)–(10.63).
10.14
(⋆⋆) Using the distribution (10.59), verify the result (10.64).
10.15
(⋆) Using the result (B.17), show that the expected value of the mixing coefﬁcients
in the variational mixture of Gaussians is given by (10.69).
10.16
(⋆⋆) www
Verify the results (10.71) and (10.72) for the ﬁrst two terms in the
lower bound for the variational Gaussian mixture model given by (10.70).
10.17
(⋆⋆⋆) Verify the results (10.73)–(10.77) for the remaining terms in the lower bound
for the variational Gaussian mixture model given by (10.70).
10.18
(⋆⋆⋆)
In this exercise, we shall derive the variational re-estimation equations for
the Gaussian mixture model by direct differentiation of the lower bound. To do this
we assume that the variational distribution has the factorization deﬁned by (10.42)
and (10.55) with factors given by (10.48), (10.57), and (10.59). Substitute these into
(10.70) and hence obtain the lower bound as a function of the parameters of the varia-
tional distribution. Then, by maximizing the bound with respect to these parameters,
derive the re-estimation equations for the factors in the variational distribution, and
show that these are the same as those obtained in Section 10.2.1.
10.19
(⋆⋆) Derive the result (10.81) for the predictive distribution in the variational treat-
ment of the Bayesian mixture of Gaussians model.
10.20
(⋆⋆) www
This exercise explores the variational Bayes solution for the mixture of
Gaussians model when the size N of the data set is large and shows that it reduces (as
we would expect) to the maximum likelihood solution based on EM derived in Chap-
ter 9. Note that results from Appendix B may be used to help answer this exercise.
First show that the posterior distribution q⋆(Λk) of the precisions becomes sharply
peaked around the maximum likelihood solution. Do the same for the posterior dis-
tribution of the means q⋆(µk|Λk). Next consider the posterior distribution q⋆(π)
for the mixing coefﬁcients and show that this too becomes sharply peaked around
the maximum likelihood solution. Similarly, show that the responsibilities become
equal to the corresponding maximum likelihood values for large N, by making use
of the following asymptotic result for the digamma function for large x
ψ(x) = ln x + O (1/x) .
(10.241)
Finally, by making use of (10.80), show that for large N, the predictive distribution
becomes a mixture of Gaussians.
10.21
(⋆) Show that the number of equivalent parameter settings due to interchange sym-
metries in a mixture model with K components is K!.


---
**Page 535**
520
10. APPROXIMATE INFERENCE
10.22
(⋆⋆) We have seen that each mode of the posterior distribution in a Gaussian mix-
ture model is a member of a family of K! equivalent modes. Suppose that the result
of running the variational inference algorithm is an approximate posterior distribu-
tion q that is localized in the neighbourhood of one of the modes. We can then
approximate the full posterior distribution as a mixture of K! such q distributions,
once centred on each mode and having equal mixing coefﬁcients. Show that if we
assume negligible overlap between the components of the q mixture, the resulting
lower bound differs from that for a single component q distribution through the ad-
dition of an extra term ln K!.
10.23
(⋆⋆) www
Consider a variational Gaussian mixture model in which there is no
prior distribution over mixing coefﬁcients {πk}. Instead, the mixing coefﬁcients are
treated as parameters, whose values are to be found by maximizing the variational
lower bound on the log marginal likelihood. Show that maximizing this lower bound
with respect to the mixing coefﬁcients, using a Lagrange multiplier to enforce the
constraint that the mixing coefﬁcients sum to one, leads to the re-estimation result
(10.83). Note that there is no need to consider all of the terms in the lower bound but
only the dependence of the bound on the {πk}.
10.24
(⋆⋆) www
We have seen in Section 10.2 that the singularities arising in the max-
imum likelihood treatment of Gaussian mixture models do not arise in a Bayesian
treatment. Discuss whether such singularities would arise if the Bayesian model
were solved using maximum posterior (MAP) estimation.
10.25
(⋆⋆)
The variational treatment of the Bayesian mixture of Gaussians, discussed in
Section 10.2, made use of a factorized approximation (10.5) to the posterior distribu-
tion. As we saw in Figure 10.2, the factorized assumption causes the variance of the
posterior distribution to be under-estimated for certain directions in parameter space.
Discuss qualitatively the effect this will have on the variational approximation to the
model evidence, and how this effect will vary with the number of components in
the mixture. Hence explain whether the variational Gaussian mixture will tend to
under-estimate or over-estimate the optimal number of components.
10.26
(⋆⋆⋆)
Extend the variational treatment of Bayesian linear regression to include
a gamma hyperprior Gam(β|c0, d0) over β and solve variationally, by assuming a
factorized variational distribution of the form q(w)q(α)q(β). Derive the variational
update equations for the three factors in the variational distribution and also obtain
an expression for the lower bound and for the predictive distribution.
10.27
(⋆⋆) By making use of the formulae given in Appendix B show that the variational
lower bound for the linear basis function regression model, deﬁned by (10.107), can
be written in the form (10.107) with the various terms deﬁned by (10.108)–(10.112).
10.28
(⋆⋆⋆)
Rewrite the model for the Bayesian mixture of Gaussians, introduced in
Section 10.2, as a conjugate model from the exponential family, as discussed in
Section 10.4. Hence use the general results (10.115) and (10.119) to derive the
speciﬁc results (10.48), (10.57), and (10.59).


---
**Page 536**
Exercises
521
10.29
(⋆) www
Show that the function f(x) = ln(x) is concave for 0 < x < ∞
by computing its second derivative. Determine the form of the dual function g(λ)
deﬁned by (10.133), and verify that minimization of λx −g(λ) with respect to λ
according to (10.132) indeed recovers the function ln(x).
10.30
(⋆) By evaluating the second derivative, show that the log logistic function f(x) =
−ln(1 + e−x) is concave. Derive the variational upper bound (10.137) directly by
making a second order Taylor expansion of the log logistic function around a point
x = ξ.
10.31
(⋆⋆)
By ﬁnding the second derivative with respect to x, show that the function
f(x) = −ln(ex/2 + e−x/2) is a concave function of x. Now consider the second
derivatives with respect to the variable x2 and hence show that it is a convex function
of x2. Plot graphs of f(x) against x and against x2. Derive the lower bound (10.144)
on the logistic sigmoid function directly by making a ﬁrst order Taylor series expan-
sion of the function f(x) in the variable x2 centred on the value ξ2.
10.32
(⋆⋆) www
Consider the variational treatment of logistic regression with sequen-
tial learning in which data points are arriving one at a time and each must be pro-
cessed and discarded before the next data point arrives. Show that a Gaussian ap-
proximation to the posterior distribution can be maintained through the use of the
lower bound (10.151), in which the distribution is initialized using the prior, and as
each data point is absorbed its corresponding variational parameter ξn is optimized.
10.33
(⋆)
By differentiating the quantity Q(ξ, ξold) deﬁned by (10.161) with respect to
the variational parameter ξn show that the update equation for ξn for the Bayesian
logistic regression model is given by (10.163).
10.34
(⋆⋆)
In this exercise we derive re-estimation equations for the variational parame-
ters ξ in the Bayesian logistic regression model of Section 4.5 by direct maximization
of the lower bound given by (10.164). To do this set the derivative of L(ξ) with re-
spect to ξn equal to zero, making use of the result (3.117) for the derivative of the log
of a determinant, together with the expressions (10.157) and (10.158) which deﬁne
the mean and covariance of the variational posterior distribution q(w).
10.35
(⋆⋆) Derive the result (10.164) for the lower bound L(ξ) in the variational logistic
regression model. This is most easily done by substituting the expressions for the
Gaussian prior q(w) = N(w|m0, S0), together with the lower bound h(w, ξ) on
the likelihood function, into the integral (10.159) which deﬁnes L(ξ). Next gather
together the terms which depend on w in the exponential and complete the square
to give a Gaussian integral, which can then be evaluated by invoking the standard
result for the normalization coefﬁcient of a multivariate Gaussian. Finally take the
logarithm to obtain (10.164).
10.36
(⋆⋆) Consider the ADF approximation scheme discussed in Section 10.7, and show
that inclusion of the factor fj(θ) leads to an update of the model evidence of the
form
pj(D) ≃pj−1(D)Zj
(10.242)


---
**Page 537**
522
10. APPROXIMATE INFERENCE
where Zj is the normalization constant deﬁned by (10.197). By applying this result
recursively, and initializing with p0(D) = 1, derive the result
p(D) ≃

j
Zj.
(10.243)
10.37
(⋆) www
Consider the expectation propagation algorithm from Section 10.7, and
suppose that one of the factors f0(θ) in the deﬁnition (10.188) has the same expo-
nential family functional form as the approximating distribution q(θ). Show that if
the factor f0(θ) is initialized to be f0(θ), then an EP update to reﬁne f0(θ) leaves
f0(θ) unchanged. This situation typically arises when one of the factors is the prior
p(θ), and so we see that the prior factor can be incorporated once exactly and does
not need to be reﬁned.
10.38
(⋆⋆⋆)
In this exercise and the next, we shall verify the results (10.214)–(10.224)
for the expectation propagation algorithm applied to the clutter problem. Begin by
using the division formula (10.205) to derive the expressions (10.214) and (10.215)
by completing the square inside the exponential to identify the mean and variance.
Also, show that the normalization constant Zn, deﬁned by (10.206), is given for the
clutter problem by (10.216). This can be done by making use of the general result
(2.115).
10.39
(⋆⋆⋆)
Show that the mean and variance of qnew(θ) for EP applied to the clutter
problem are given by (10.217) and (10.218). To do this, ﬁrst prove the following
results for the expectations of θ and θθT under qnew(θ)
E[θ]
=
m\n + v\n∇m\n ln Zn
(10.244)
E[θTθ]
=
2(v\n)2∇v\n ln Zn + 2E[θ]Tm\n −∥m\n∥2
(10.245)
and then make use of the result (10.216) for Zn. Next, prove the results (10.220)–
(10.222) by using (10.207) and completing the square in the exponential. Finally,
use (10.208) to derive the result (10.223).


---
**Page 538**
11
Sampling
Methods
For most probabilistic models of practical interest, exact inference is intractable, and
so we have to resort to some form of approximation. In Chapter 10, we discussed
inference algorithms based on deterministic approximations, which include methods
such as variational Bayes and expectation propagation. Here we consider approxi-
mate inference methods based on numerical sampling, also known as Monte Carlo
techniques.
Although for some applications the posterior distribution over unobserved vari-
ables will be of direct interest in itself, for most situations the posterior distribution
is required primarily for the purpose of evaluating expectations, for example in order
to make predictions. The fundamental problem that we therefore wish to address in
this chapter involves ﬁnding the expectation of some function f(z) with respect to a
probability distribution p(z). Here, the components of z might comprise discrete or
continuous variables or some combination of the two. Thus in the case of continuous
523


---
**Page 539**
524
11. SAMPLING METHODS
Figure 11.1
Schematic illustration of a function f(z)
whose expectation is to be evaluated with
respect to a distribution p(z).
p(z)
f(z)
z
variables, we wish to evaluate the expectation
E[f] =

f(z)p(z) dz
(11.1)
where the integral is replaced by summation in the case of discrete variables. This
is illustrated schematically for a single continuous variable in Figure 11.1. We shall
suppose that such expectations are too complex to be evaluated exactly using analyt-
ical techniques.
The general idea behind sampling methods is to obtain a set of samples z(l)
(where l = 1, . . . , L) drawn independently from the distribution p(z). This allows
the expectation (11.1) to be approximated by a ﬁnite sum
f = 1
L
L

l=1
f(z(l)).
(11.2)
As long as the samples z(l) are drawn from the distribution p(z), then E[f] = E[f]
and so the estimator f has the correct mean. The variance of the estimator is given
by
Exercise 11.1
var[f] = 1
LE

(f −E[f])2	
(11.3)
is the variance of the function f(z) under the distribution p(z). It is worth emphasiz-
ing that the accuracy of the estimator therefore does not depend on the dimension-
ality of z, and that, in principle, high accuracy may be achievable with a relatively
small number of samples z(l). In practice, ten or twenty independent samples may
sufﬁce to estimate an expectation to sufﬁcient accuracy.
The problem, however, is that the samples {z(l)} might not be independent, and
so the effective sample size might be much smaller than the apparent sample size.
Also, referring back to Figure 11.1, we note that if f(z) is small in regions where
p(z) is large, and vice versa, then the expectation may be dominated by regions
of small probability, implying that relatively large sample sizes will be required to
achieve sufﬁcient accuracy.
For many models, the joint distribution p(z) is conveniently speciﬁed in terms
of a graphical model. In the case of a directed graph with no observed variables, it is


---
**Page 540**
11. SAMPLING METHODS
525
straightforward to sample from the joint distribution (assuming that it is possible to
sample from the conditional distributions at each node) using the following ances-
tral sampling approach, discussed brieﬂy in Section 8.1.2. The joint distribution is
speciﬁed by
p(z) =
M

i=1
p(zi|pai)
(11.4)
where zi are the set of variables associated with node i, and pai denotes the set of
variables associated with the parents of node i. To obtain a sample from the joint
distribution, we make one pass through the set of variables in the order z1, . . . , zM
sampling from the conditional distributions p(zi|pai). This is always possible be-
cause at each step all of the parent values will have been instantiated. After one pass
through the graph, we will have obtained a sample from the joint distribution.
Now consider the case of a directed graph in which some of the nodes are in-
stantiated with observed values. We can in principle extend the above procedure, at
least in the case of nodes representing discrete variables, to give the following logic
sampling approach (Henrion, 1988), which can be seen as a special case of impor-
tance sampling discussed in Section 11.1.4. At each step, when a sampled value is
obtained for a variable zi whose value is observed, the sampled value is compared
to the observed value, and if they agree then the sample value is retained and the al-
gorithm proceeds to the next variable in turn. However, if the sampled value and the
observed value disagree, then the whole sample so far is discarded and the algorithm
starts again with the ﬁrst node in the graph. This algorithm samples correctly from
the posterior distribution because it corresponds simply to drawing samples from the
joint distribution of hidden variables and data variables and then discarding those
samples that disagree with the observed data (with the slight saving of not continu-
ing with the sampling from the joint distribution as soon as one contradictory value is
observed). However, the overall probability of accepting a sample from the posterior
decreases rapidly as the number of observed variables increases and as the number
of states that those variables can take increases, and so this approach is rarely used
in practice.
In the case of probability distributions deﬁned by an undirected graph, there is
no one-pass sampling strategy that will sample even from the prior distribution with
no observed variables. Instead, computationally more expensive techniques must be
employed, such as Gibbs sampling, which is discussed in Section 11.3.
As well as sampling from conditional distributions, we may also require samples
from a marginal distribution. If we already have a strategy for sampling from a joint
distribution p(u, v), then it is straightforward to obtain samples from the marginal
distribution p(u) simply by ignoring the values for v in each sample.
There are numerous texts dealing with Monte Carlo methods. Those of partic-
ular interest from the statistical inference perspective include Chen et al. (2001),
Gamerman (1997), Gilks et al. (1996), Liu (2001), Neal (1996), and Robert and
Casella (1999). Also there are review articles by Besag et al. (1995), Brooks (1998),
Diaconis and Saloff-Coste (1998), Jerrum and Sinclair (1996), Neal (1993), Tierney
(1994), and Andrieu et al. (2003) that provide additional information on sampling


---
**Page 541**
526
11. SAMPLING METHODS
methods for statistical inference.
Diagnostic tests for convergence of Markov chain Monte Carlo algorithms are
summarized in Robert and Casella (1999), and some practical guidance on the use of
sampling methods in the context of machine learning is given in Bishop and Nabney
(2008).
11.1. Basic Sampling Algorithms
In this section, we consider some simple strategies for generating random samples
from a given distribution. Because the samples will be generated by a computer
algorithm they will in fact be pseudo-random numbers, that is, they will be deter-
ministically calculated, but must nevertheless pass appropriate tests for randomness.
Generating such numbers raises several subtleties (Press et al., 1992) that lie outside
the scope of this book. Here we shall assume that an algorithm has been provided
that generates pseudo-random numbers distributed uniformly over (0, 1), and indeed
most software environments have such a facility built in.
11.1.1
Standard distributions
We ﬁrst consider how to generate random numbers from simple nonuniform dis-
tributions, assuming that we already have available a source of uniformly distributed
random numbers. Suppose that z is uniformly distributed over the interval (0, 1),
and that we transform the values of z using some function f(·) so that y = f(z).
The distribution of y will be governed by
p(y) = p(z)

dz
dy

(11.5)
where, in this case, p(z) = 1. Our goal is to choose the function f(z) such that the
resulting values of y have some speciﬁc desired distribution p(y). Integrating (11.5)
we obtain
z = h(y) ≡
 y
−∞
p(y) dy
(11.6)
which is the indeﬁnite integral of p(y). Thus, y = h−1(z), and so we have to
Exercise 11.2
transform the uniformly distributed random numbers using a function which is the
inverse of the indeﬁnite integral of the desired distribution. This is illustrated in
Figure 11.2.
Consider for example the exponential distribution
p(y) = λ exp(−λy)
(11.7)
where 0 ⩽y < ∞. In this case the lower limit of the integral in (11.6) is 0, and so
h(y) = 1 −exp(−λy). Thus, if we transform our uniformly distributed variable z
using y = −λ−1 ln(1 −z), then y will have an exponential distribution.


---
**Page 542**
11.1. Basic Sampling Algorithms
527
Figure 11.2
Geometrical interpretation of the trans-
formation method for generating nonuni-
formly distributed random numbers. h(y)
is the indeﬁnite integral of the desired dis-
tribution p(y).
If a uniformly distributed
random variable z is transformed using
y = h−1(z), then y will be distributed ac-
cording to p(y).
p(y)
h(y)
y
0
1
Another example of a distribution to which the transformation method can be
applied is given by the Cauchy distribution
p(y) = 1
π
1
1 + y2 .
(11.8)
In this case, the inverse of the indeﬁnite integral can be expressed in terms of the
‘tan’ function.
Exercise 11.3
The generalization to multiple variables is straightforward and involves the Ja-
cobian of the change of variables, so that
p(y1, . . . , yM) = p(z1, . . . , zM)

∂(z1, . . . , zM)
∂(y1, . . . , yM)
 .
(11.9)
As a ﬁnal example of the transformation method we consider the Box-Muller
method for generating samples from a Gaussian distribution. First, suppose we gen-
erate pairs of uniformly distributed random numbers z1, z2 ∈(−1, 1), which we can
do by transforming a variable distributed uniformly over (0, 1) using z →2z −1.
Next we discard each pair unless it satisﬁes z2
1 + z2
2 ⩽1. This leads to a uniform
distribution of points inside the unit circle with p(z1, z2) = 1/π, as illustrated in
Figure 11.3. Then, for each pair z1, z2 we evaluate the quantities
Figure 11.3
The Box-Muller method for generating Gaussian dis-
tributed random numbers starts by generating samples
from a uniform distribution inside the unit circle.
−1
−1
1
1
z1
z2


---
**Page 543**
528
11. SAMPLING METHODS
y1
=
z1
−2 ln z1
r2
1/2
(11.10)
y2
=
z2
−2 ln z2
r2
1/2
(11.11)
where r2 = z2
1 + z2
2. Then the joint distribution of y1 and y2 is given by
Exercise 11.4
p(y1, y2)
=
p(z1, z2)

∂(z1, z2)
∂(y1, y2)

=

1
√
2π
exp(−y2
1/2)
 
1
√
2π
exp(−y2
2/2)

(11.12)
and so y1 and y2 are independent and each has a Gaussian distribution with zero
mean and unit variance.
If y has a Gaussian distribution with zero mean and unit variance, then σy + µ
will have a Gaussian distribution with mean µ and variance σ2. To generate vector-
valued variables having a multivariate Gaussian distribution with mean µ and co-
variance Σ, we can make use of the Cholesky decomposition, which takes the form
Σ = LLT (Press et al., 1992). Then, if z is a vector valued random variable whose
components are independent and Gaussian distributed with zero mean and unit vari-
ance, then y = µ + Lz will have mean µ and covariance Σ.
Exercise 11.5
Obviously, the transformation technique depends for its success on the ability
to calculate and then invert the indeﬁnite integral of the required distribution. Such
operations will only be feasible for a limited number of simple distributions, and so
we must turn to alternative approaches in search of a more general strategy. Here
we consider two techniques called rejection sampling and importance sampling. Al-
though mainly limited to univariate distributions and thus not directly applicable to
complex problems in many dimensions, they do form important components in more
general strategies.
11.1.2
Rejection sampling
The rejection sampling framework allows us to sample from relatively complex
distributions, subject to certain constraints. We begin by considering univariate dis-
tributions and discuss the extension to multiple dimensions subsequently.
Suppose we wish to sample from a distribution p(z) that is not one of the simple,
standard distributions considered so far, and that sampling directly from p(z) is dif-
ﬁcult. Furthermore suppose, as is often the case, that we are easily able to evaluate
p(z) for any given value of z, up to some normalizing constant Z, so that
p(z) = 1
Zp
p(z)
(11.13)
where p(z) can readily be evaluated, but Zp is unknown.
In order to apply rejection sampling, we need some simpler distribution q(z),
sometimes called a proposal distribution, from which we can readily draw samples.


---
**Page 544**
11.1. Basic Sampling Algorithms
529
Figure 11.4
In the rejection sampling method,
samples are drawn from a sim-
ple distribution q(z) and rejected
if they fall in the grey area be-
tween the unnormalized distribu-
tion ep(z) and the scaled distribu-
tion kq(z). The resulting samples
are distributed according to p(z),
which is the normalized version of
ep(z).
z0
z
u0
kq(z0)
kq(z)
˜p(z)
We next introduce a constant k whose value is chosen such that kq(z) ⩾p(z) for
all values of z. The function kq(z) is called the comparison function and is illus-
trated for a univariate distribution in Figure 11.4. Each step of the rejection sampler
involves generating two random numbers. First, we generate a number z0 from the
distribution q(z). Next, we generate a number u0 from the uniform distribution over
[0, kq(z0)]. This pair of random numbers has uniform distribution under the curve
of the function kq(z). Finally, if u0 > p(z0) then the sample is rejected, otherwise
u0 is retained. Thus the pair is rejected if it lies in the grey shaded region in Fig-
ure 11.4. The remaining pairs then have uniform distribution under the curve of p(z),
and hence the corresponding z values are distributed according to p(z), as desired.
Exercise 11.6
The original values of z are generated from the distribution q(z), and these sam-
ples are then accepted with probability p(z)/kq(z), and so the probability that a
sample will be accepted is given by
p(accept)
=

{p(z)/kq(z)} q(z) dz
=
1
k

p(z) dz.
(11.14)
Thus the fraction of points that are rejected by this method depends on the ratio of
the area under the unnormalized distribution p(z) to the area under the curve kq(z).
We therefore see that the constant k should be as small as possible subject to the
limitation that kq(z) must be nowhere less than p(z).
As an illustration of the use of rejection sampling, consider the task of sampling
from the gamma distribution
Gam(z|a, b) = baza−1 exp(−bz)
Γ(a)
(11.15)
which, for a > 1, has a bell-shaped form, as shown in Figure 11.5. A suitable
proposal distribution is therefore the Cauchy (11.8) because this too is bell-shaped
and because we can use the transformation method, discussed earlier, to sample from
it. We need to generalize the Cauchy slightly to ensure that it nowhere has a smaller
value than the gamma distribution. This can be achieved by transforming a uniform
random variable y using z = b tan y + c, which gives random numbers distributed
according to.
Exercise 11.7


---
**Page 545**
530
11. SAMPLING METHODS
Figure 11.5
Plot showing the gamma distribu-
tion given by (11.15) as the green
curve, with a scaled Cauchy pro-
posal distribution shown by the red
curve. Samples from the gamma
distribution can be obtained by
sampling from the Cauchy and
then applying the rejection sam-
pling criterion.
z
p(z)
0
10
20
30
0
0.05
0.1
0.15
q(z) =
k
1 + (z −c)2/b2 .
(11.16)
The minimum reject rate is obtained by setting c = a −1, b2 = 2a −1 and choos-
ing the constant k to be as small as possible while still satisfying the requirement
kq(z) ⩾p(z). The resulting comparison function is also illustrated in Figure 11.5.
11.1.3
Adaptive rejection sampling
In many instances where we might wish to apply rejection sampling, it proves
difﬁcult to determine a suitable analytic form for the envelope distribution q(z). An
alternative approach is to construct the envelope function on the ﬂy based on mea-
sured values of the distribution p(z) (Gilks and Wild, 1992). Construction of an
envelope function is particularly straightforward for cases in which p(z) is log con-
cave, in other words when ln p(z) has derivatives that are nonincreasing functions
of z. The construction of a suitable envelope function is illustrated graphically in
Figure 11.6.
The function ln p(z) and its gradient are evaluated at some initial set of grid
points, and the intersections of the resulting tangent lines are used to construct the
envelope function. Next a sample value is drawn from the envelope distribution.
This is straightforward because the log of the envelope distribution is a succession
Exercise 11.9
Figure 11.6
In the case of distributions that are
log concave, an envelope function
for use in rejection sampling can be
constructed using the tangent lines
computed at a set of grid points. If a
sample point is rejected, it is added
to the set of grid points and used to
reﬁne the envelope distribution.
z1
z2
z3
z
ln p(z)


---
**Page 546**
11.1. Basic Sampling Algorithms
531
Figure 11.7
Illustrative
example
of
rejection
sampling involving sampling from a
Gaussian distribution p(z) shown by
the green curve, by using rejection
sampling from a proposal distri-
bution q(z) that is also Gaussian
and whose scaled version kq(z) is
shown by the red curve.
z
p(z)
−5
0
5
0
0.25
0.5
of linear functions, and hence the envelope distribution itself comprises a piecewise
exponential distribution of the form
q(z) = kiλi exp {−λi(z −zi−1)}
zi−1 < z ⩽zi.
(11.17)
Once a sample has been drawn, the usual rejection criterion can be applied. If the
sample is accepted, then it will be a draw from the desired distribution. If, however,
the sample is rejected, then it is incorporated into the set of grid points, a new tangent
line is computed, and the envelope function is thereby reﬁned. As the number of
grid points increases, so the envelope function becomes a better approximation of
the desired distribution p(z) and the probability of rejection decreases.
A variant of the algorithm exists that avoids the evaluation of derivatives (Gilks,
1992). The adaptive rejection sampling framework can also be extended to distri-
butions that are not log concave, simply by following each rejection sampling step
with a Metropolis-Hastings step (to be discussed in Section 11.2.2), giving rise to
adaptive rejection Metropolis sampling (Gilks et al., 1995).
Clearly for rejection sampling to be of practical value, we require that the com-
parison function be close to the required distribution so that the rate of rejection is
kept to a minimum. Now let us examine what happens when we try to use rejection
sampling in spaces of high dimensionality. Consider, for the sake of illustration,
a somewhat artiﬁcial problem in which we wish to sample from a zero-mean mul-
tivariate Gaussian distribution with covariance σ2
pI, where I is the unit matrix, by
rejection sampling from a proposal distribution that is itself a zero-mean Gaussian
distribution having covariance σ2
qI. Obviously, we must have σ2
q ⩾σ2
p in order that
there exists a k such that kq(z) ⩾p(z). In D-dimensions the optimum value of k
is given by k = (σq/σp)D, as illustrated for D = 1 in Figure 11.7. The acceptance
rate will be the ratio of volumes under p(z) and kq(z), which, because both distribu-
tions are normalized, is just 1/k. Thus the acceptance rate diminishes exponentially
with dimensionality. Even if σq exceeds σp by just one percent, for D = 1, 000 the
acceptance ratio will be approximately 1/20, 000. In this illustrative example the
comparison function is close to the required distribution. For more practical exam-
ples, where the desired distribution may be multimodal and sharply peaked, it will
be extremely difﬁcult to ﬁnd a good proposal distribution and comparison function.


---
**Page 547**
532
11. SAMPLING METHODS
Figure 11.8
Importance sampling addresses the prob-
lem of evaluating the expectation of a func-
tion f(z) with respect to a distribution p(z)
from which it is difﬁcult to draw samples di-
rectly. Instead, samples {z(l)} are drawn
from a simpler distribution q(z), and the
corresponding terms in the summation are
weighted by the ratios p(z(l))/q(z(l)).
p(z)
f(z)
z
q(z)
Furthermore, the exponential decrease of acceptance rate with dimensionality is a
generic feature of rejection sampling. Although rejection can be a useful technique
in one or two dimensions it is unsuited to problems of high dimensionality. It can,
however, play a role as a subroutine in more sophisticated algorithms for sampling
in high dimensional spaces.
11.1.4
Importance sampling
One of the principal reasons for wishing to sample from complicated probability
distributions is to be able to evaluate expectations of the form (11.1). The technique
of importance sampling provides a framework for approximating expectations di-
rectly but does not itself provide a mechanism for drawing samples from distribution
p(z).
The ﬁnite sum approximation to the expectation, given by (11.2), depends on
being able to draw samples from the distribution p(z). Suppose, however, that it is
impractical to sample directly from p(z) but that we can evaluate p(z) easily for any
given value of z. One simplistic strategy for evaluating expectations would be to
discretize z-space into a uniform grid and to evaluate the integrand as a sum of the
form
E[f] ≃
L

l=1
p(z(l))f(z(l)).
(11.18)
An obvious problem with this approach is that the number of terms in the summation
grows exponentially with the dimensionality of z. Furthermore, as we have already
noted, the kinds of probability distributions of interest will often have much of their
mass conﬁned to relatively small regions of z space and so uniform sampling will be
very inefﬁcient because in high-dimensional problems, only a very small proportion
of the samples will make a signiﬁcant contribution to the sum. We would really like
to choose the sample points to fall in regions where p(z) is large, or ideally where
the product p(z)f(z) is large.
As in the case of rejection sampling, importance sampling is based on the use
of a proposal distribution q(z) from which it is easy to draw samples, as illustrated
in Figure 11.8. We can then express the expectation in the form of a ﬁnite sum over


---
**Page 548**
11.1. Basic Sampling Algorithms
533
samples {z(l)} drawn from q(z)
E[f]
=

f(z)p(z) dz
=

f(z)p(z)
q(z)q(z) dz
≃
1
L
L

l=1
p(z(l))
q(z(l))f(z(l)).
(11.19)
The quantities rl = p(z(l))/q(z(l)) are known as importance weights, and they cor-
rect the bias introduced by sampling from the wrong distribution. Note that, unlike
rejection sampling, all of the samples generated are retained.
It will often be the case that the distribution p(z) can only be evaluated up to a
normalization constant, so that p(z) = p(z)/Zp where p(z) can be evaluated easily,
whereas Zp is unknown. Similarly, we may wish to use an importance sampling
distribution q(z) = q(z)/Zq, which has the same property. We then have
E[f]
=

f(z)p(z) dz
=
Zq
Zp

f(z)p(z)
q(z)q(z) dz
≃
Zq
Zp
1
L
L

l=1
rlf(z(l)).
(11.20)
where rl = p(z(l))/q(z(l)). We can use the same sample set to evaluate the ratio
Zp/Zq with the result
Zp
Zq
=
1
Zq

p(z) dz =
 p(z)
q(z)q(z) dz
≃
1
L
L

l=1
rl
(11.21)
and hence
E[f] ≃
L

l=1
wlf(z(l))
(11.22)
where we have deﬁned
wl =
rl

m rm
=
p(z(l))/q(z(l))

m p(z(m))/q(z(m)).
(11.23)
As with rejection sampling, the success of the importance sampling approach
depends crucially on how well the sampling distribution q(z) matches the desired


---
**Page 549**
534
11. SAMPLING METHODS
distribution p(z). If, as is often the case, p(z)f(z) is strongly varying and has a sig-
niﬁcant proportion of its mass concentrated over relatively small regions of z space,
then the set of importance weights {rl} may be dominated by a few weights hav-
ing large values, with the remaining weights being relatively insigniﬁcant. Thus the
effective sample size can be much smaller than the apparent sample size L. The prob-
lem is even more severe if none of the samples falls in the regions where p(z)f(z)
is large. In that case, the apparent variances of rl and rlf(z(l)) may be small even
though the estimate of the expectation may be severely wrong. Hence a major draw-
back of the importance sampling method is the potential to produce results that are
arbitrarily in error and with no diagnostic indication. This also highlights a key re-
quirement for the sampling distribution q(z), namely that it should not be small or
zero in regions where p(z) may be signiﬁcant.
For distributions deﬁned in terms of a graphical model, we can apply the impor-
tance sampling technique in various ways. For discrete variables, a simple approach
is called uniform sampling. The joint distribution for a directed graph is deﬁned
by (11.4). Each sample from the joint distribution is obtained by ﬁrst setting those
variables zi that are in the evidence set equal to their observed values. Each of the
remaining variables is then sampled independently from a uniform distribution over
the space of possible instantiations. To determine the corresponding weight associ-
ated with a sample z(l), we note that the sampling distribution q(z) is uniform over
the possible choices for z, and that p(z|x) = p(z), where x denotes the subset of
variables that are observed, and the equality follows from the fact that every sample
z that is generated is necessarily consistent with the evidence. Thus the weights rl
are simply proportional to p(z). Note that the variables can be sampled in any order.
This approach can yield poor results if the posterior distribution is far from uniform,
as is often the case in practice.
An improvement on this approach is called likelihood weighted sampling (Fung
and Chang, 1990; Shachter and Peot, 1990) and is based on ancestral sampling of
the variables. For each variable in turn, if that variable is in the evidence set, then it
is just set to its instantiated value. If it is not in the evidence set, then it is sampled
from the conditional distribution p(zi|pai) in which the conditioning variables are
set to their currently sampled values. The weighting associated with the resulting
sample z is then given by
r(z) =

zi̸∈e
p(zi|pai)
p(zi|pai)

zi∈e
p(zi|pai)
1
=

zi∈e
p(zi|pai).
(11.24)
This method can be further extended using self-importance sampling (Shachter and
Peot, 1990) in which the importance sampling distribution is continually updated to
reﬂect the current estimated posterior distribution.
11.1.5
Sampling-importance-resampling
The rejection sampling method discussed in Section 11.1.2 depends in part for
its success on the determination of a suitable value for the constant k. For many
pairs of distributions p(z) and q(z), it will be impractical to determine a suitable


---
**Page 550**
11.1. Basic Sampling Algorithms
535
value for k in that any value that is sufﬁciently large to guarantee a bound on the
desired distribution will lead to impractically small acceptance rates.
As in the case of rejection sampling, the sampling-importance-resampling (SIR)
approach also makes use of a sampling distribution q(z) but avoids having to de-
termine the constant k. There are two stages to the scheme. In the ﬁrst stage,
L samples z(1), . . . , z(L) are drawn from q(z). Then in the second stage, weights
w1, . . . , wL are constructed using (11.23). Finally, a second set of L samples is
drawn from the discrete distribution (z(1), . . . , z(L)) with probabilities given by the
weights (w1, . . . , wL).
The resulting L samples are only approximately distributed according to p(z),
but the distribution becomes correct in the limit L →∞. To see this, consider the
univariate case, and note that the cumulative distribution of the resampled values is
given by
p(z ⩽a)
=

l:z(l)⩽a
wl
=

l I(z(l) ⩽a)p(z(l))/q(z(l))

l p(z(l))/q(z(l))
(11.25)
where I(.) is the indicator function (which equals 1 if its argument is true and 0
otherwise). Taking the limit L →∞, and assuming suitable regularity of the dis-
tributions, we can replace the sums by integrals weighted according to the original
sampling distribution q(z)
p(z ⩽a)
=

I(z ⩽a) {p(z)/q(z)} q(z) dz

{p(z)/q(z)} q(z) dz
=

I(z ⩽a)p(z) dz

p(z) dz
=

I(z ⩽a)p(z) dz
(11.26)
which is the cumulative distribution function of p(z). Again, we see that the normal-
ization of p(z) is not required.
For a ﬁnite value of L, and a given initial sample set, the resampled values will
only approximately be drawn from the desired distribution. As with rejection sam-
pling, the approximation improves as the sampling distribution q(z) gets closer to
the desired distribution p(z). When q(z) = p(z), the initial samples (z(1), . . . , z(L))
have the desired distribution, and the weights wn = 1/L so that the resampled values
also have the desired distribution.
If moments with respect to the distribution p(z) are required, then they can be


---
**Page 551**
536
11. SAMPLING METHODS
evaluated directly using the original samples together with the weights, because
E[f(z)]
=

f(z)p(z) dz
=

f(z)[p(z)/q(z)]q(z) dz

[p(z)/q(z)]q(z) dz
≃
L

l=1
wlf(zl).
(11.27)
11.1.6
Sampling and the EM algorithm
In addition to providing a mechanism for direct implementation of the Bayesian
framework, Monte Carlo methods can also play a role in the frequentist paradigm,
for example to ﬁnd maximum likelihood solutions. In particular, sampling methods
can be used to approximate the E step of the EM algorithm for models in which the
E step cannot be performed analytically. Consider a model with hidden variables
Z, visible (observed) variables X, and parameters θ. The function that is optimized
with respect to θ in the M step is the expected complete-data log likelihood, given
by
Q(θ, θold) =

p(Z|X, θold) ln p(Z, X|θ) dZ.
(11.28)
We can use sampling methods to approximate this integral by a ﬁnite sum over sam-
ples {Z(l)}, which are drawn from the current estimate for the posterior distribution
p(Z|X, θold), so that
Q(θ, θold) ≃1
L
L

l=1
ln p(Z(l), X|θ).
(11.29)
The Q function is then optimized in the usual way in the M step. This procedure is
called the Monte Carlo EM algorithm.
It is straightforward to extend this to the problem of ﬁnding the mode of the
posterior distribution over θ (the MAP estimate) when a prior distribution p(θ) has
been deﬁned, simply by adding ln p(θ) to the function Q(θ, θold) before performing
the M step.
A particular instance of the Monte Carlo EM algorithm, called stochastic EM,
arises if we consider a ﬁnite mixture model, and draw just one sample at each E step.
Here the latent variable Z characterizes which of the K components of the mixture
is responsible for generating each data point. In the E step, a sample of Z is taken
from the posterior distribution p(Z|X, θold) where X is the data set. This effectively
makes a hard assignment of each data point to one of the components in the mixture.
In the M step, this sampled approximation to the posterior distribution is used to
update the model parameters in the usual way.


---
**Page 552**
11.2. Markov Chain Monte Carlo
537
Now suppose we move from a maximum likelihood approach to a full Bayesian
treatment in which we wish to sample from the posterior distribution over the param-
eter vector θ. In principle, we would like to draw samples from the joint posterior
p(θ, Z|X), but we shall suppose that this is computationally difﬁcult. Suppose fur-
ther that it is relatively straightforward to sample from the complete-data parameter
posterior p(θ|Z, X). This inspires the data augmentation algorithm, which alter-
nates between two steps known as the I-step (imputation step, analogous to an E
step) and the P-step (posterior step, analogous to an M step).
IP Algorithm
I-step. We wish to sample from p(Z|X) but we cannot do this directly. We
therefore note the relation
p(Z|X) =

p(Z|θ, X)p(θ|X) dθ
(11.30)
and hence for l = 1, . . . , L we ﬁrst draw a sample θ(l) from the current esti-
mate for p(θ|X), and then use this to draw a sample Z(l) from p(Z|θ(l), X).
P-step. Given the relation
p(θ|X) =

p(θ|Z, X)p(Z|X) dZ
(11.31)
we use the samples {Z(l)} obtained from the I-step to compute a revised
estimate of the posterior distribution over θ given by
p(θ|X) ≃1
L
L

l=1
p(θ|Z(l), X).
(11.32)
By assumption, it will be feasible to sample from this approximation in the
I-step.
Note that we are making a (somewhat artiﬁcial) distinction between parameters θ
and hidden variables Z. From now on, we blur this distinction and focus simply on
the problem of drawing samples from a given posterior distribution.
11.2. Markov Chain Monte Carlo
In the previous section, we discussed the rejection sampling and importance sam-
pling strategies for evaluating expectations of functions, and we saw that they suffer
from severe limitations particularly in spaces of high dimensionality. We therefore
turn in this section to a very general and powerful framework called Markov chain
Monte Carlo (MCMC), which allows sampling from a large class of distributions,


---
**Page 553**
538
11. SAMPLING METHODS
and which scales well with the dimensionality of the sample space. Markov chain
Monte Carlo methods have their origins in physics (Metropolis and Ulam, 1949),
and it was only towards the end of the 1980s that they started to have a signiﬁcant
impact in the ﬁeld of statistics.
As with rejection and importance sampling, we again sample from a proposal
distribution. This time, however, we maintain a record of the current state z(τ), and
the proposal distribution q(z|z(τ)) depends on this current state, and so the sequence
of samples z(1), z(2), . . . forms a Markov chain. Again, if we write p(z) = p(z)/Zp,
Section 11.2.1
we will assume that p(z) can readily be evaluated for any given value of z, although
the value of Zp may be unknown. The proposal distribution itself is chosen to be
sufﬁciently simple that it is straightforward to draw samples from it directly. At
each cycle of the algorithm, we generate a candidate sample z⋆from the proposal
distribution and then accept the sample according to an appropriate criterion.
In the basic Metropolis algorithm (Metropolis et al., 1953), we assume that the
proposal distribution is symmetric, that is q(zA|zB) = q(zB|zA) for all values of
zA and zB. The candidate sample is then accepted with probability
A(z⋆, z(τ)) = min

1, p(z⋆)
p(z(τ))

.
(11.33)
This can be achieved by choosing a random number u with uniform distribution over
the unit interval (0, 1) and then accepting the sample if A(z⋆, z(τ)) > u. Note that
if the step from z(τ) to z⋆causes an increase in the value of p(z), then the candidate
point is certain to be kept.
If the candidate sample is accepted, then z(τ+1) = z⋆, otherwise the candidate
point z⋆is discarded, z(τ+1) is set to z(τ) and another candidate sample is drawn
from the distribution q(z|z(τ+1)). This is in contrast to rejection sampling, where re-
jected samples are simply discarded. In the Metropolis algorithm when a candidate
point is rejected, the previous sample is included instead in the ﬁnal list of samples,
leading to multiple copies of samples. Of course, in a practical implementation,
only a single copy of each retained sample would be kept, along with an integer
weighting factor recording how many times that state appears. As we shall see, as
long as q(zA|zB) is positive for any values of zA and zB (this is a sufﬁcient but
not necessary condition), the distribution of z(τ) tends to p(z) as τ →∞. It should
be emphasized, however, that the sequence z(1), z(2), . . . is not a set of independent
samples from p(z) because successive samples are highly correlated. If we wish to
obtain independent samples, then we can discard most of the sequence and just re-
tain every M th sample. For M sufﬁciently large, the retained samples will for all
practical purposes be independent. Figure 11.9 shows a simple illustrative exam-
ple of sampling from a two-dimensional Gaussian distribution using the Metropolis
algorithm in which the proposal distribution is an isotropic Gaussian.
Further insight into the nature of Markov chain Monte Carlo algorithms can be
gleaned by looking at the properties of a speciﬁc example, namely a simple random


---
**Page 554**
11.2. Markov Chain Monte Carlo
539
Figure 11.9
A simple illustration using Metropo-
lis
algorithm
to
sample
from
a
Gaussian distribution whose one
standard-deviation contour is shown
by the ellipse. The proposal distribu-
tion is an isotropic Gaussian distri-
bution whose standard deviation is
0.2.
Steps that are accepted are
shown as green lines, and rejected
steps are shown in red. A total of
150 candidate samples are gener-
ated, of which 43 are rejected.
0
0.5
1
1.5
2
2.5
3
0
0.5
1
1.5
2
2.5
3
walk. Consider a state space z consisting of the integers, with probabilities
p(z(τ+1) = z(τ))
=
0.5
(11.34)
p(z(τ+1) = z(τ) + 1)
=
0.25
(11.35)
p(z(τ+1) = z(τ) −1)
=
0.25
(11.36)
where z(τ) denotes the state at step τ. If the initial state is z(1) = 0, then by sym-
metry the expected state at time τ will also be zero E[z(τ)] = 0, and similarly it is
easily seen that E[(z(τ))2] = τ/2. Thus after τ steps, the random walk has only trav-
Exercise 11.10
elled a distance that on average is proportional to the square root of τ. This square
root dependence is typical of random walk behaviour and shows that random walks
are very inefﬁcient in exploring the state space. As we shall see, a central goal in
designing Markov chain Monte Carlo methods is to avoid random walk behaviour.
11.2.1
Markov chains
Before discussing Markov chain Monte Carlo methods in more detail, it is use-
ful to study some general properties of Markov chains in more detail. In particular,
we ask under what circumstances will a Markov chain converge to the desired dis-
tribution. A ﬁrst-order Markov chain is deﬁned to be a series of random variables
z(1), . . . , z(M) such that the following conditional independence property holds for
m ∈{1, . . . , M −1}
p(z(m+1)|z(1), . . . , z(m)) = p(z(m+1)|z(m)).
(11.37)
This of course can be represented as a directed graph in the form of a chain, an ex-
ample of which is shown in Figure 8.38. We can then specify the Markov chain by
giving the probability distribution for the initial variable p(z(0)) together with the


---
**Page 555**
540
11. SAMPLING METHODS
conditional probabilities for subsequent variables in the form of transition probabil-
ities Tm(z(m), z(m+1)) ≡p(z(m+1)|z(m)). A Markov chain is called homogeneous
if the transition probabilities are the same for all m.
The marginal probability for a particular variable can be expressed in terms of
the marginal probability for the previous variable in the chain in the form
p(z(m+1)) =

z(m)
p(z(m+1)|z(m))p(z(m)).
(11.38)
A distribution is said to be invariant, or stationary, with respect to a Markov chain
if each step in the chain leaves that distribution invariant. Thus, for a homogeneous
Markov chain with transition probabilities T(z′, z), the distribution p⋆(z) is invariant
if
p⋆(z) =

z′
T(z′, z)p⋆(z′).
(11.39)
Note that a given Markov chain may have more than one invariant distribution. For
instance, if the transition probabilities are given by the identity transformation, then
any distribution will be invariant.
A sufﬁcient (but not necessary) condition for ensuring that the required distribu-
tion p(z) is invariant is to choose the transition probabilities to satisfy the property
of detailed balance, deﬁned by
p⋆(z)T(z, z′) = p⋆(z′)T(z′, z)
(11.40)
for the particular distribution p⋆(z). It is easily seen that a transition probability
that satisﬁes detailed balance with respect to a particular distribution will leave that
distribution invariant, because

z′
p⋆(z′)T(z′, z) =

z′
p⋆(z)T(z, z′) = p⋆(z)

z′
p(z′|z) = p⋆(z).
(11.41)
A Markov chain that respects detailed balance is said to be reversible.
Our goal is to use Markov chains to sample from a given distribution. We can
achieve this if we set up a Markov chain such that the desired distribution is invariant.
However, we must also require that for m →∞, the distribution p(z(m)) converges
to the required invariant distribution p⋆(z), irrespective of the choice of initial dis-
tribution p(z(0)). This property is called ergodicity, and the invariant distribution
is then called the equilibrium distribution. Clearly, an ergodic Markov chain can
have only one equilibrium distribution. It can be shown that a homogeneous Markov
chain will be ergodic, subject only to weak restrictions on the invariant distribution
and the transition probabilities (Neal, 1993).
In practice we often construct the transition probabilities from a set of ‘base’
transitions B1, . . . , BK. This can be achieved through a mixture distribution of the
form
T(z′, z) =
K

k=1
αkBk(z′, z)
(11.42)


---
**Page 556**
11.2. Markov Chain Monte Carlo
541
for some set of mixing coefﬁcients α1, . . . , αK satisfying αk ⩾0 and 
k αk = 1.
Alternatively, the base transitions may be combined through successive application,
so that
T(z′, z) =

z1
. . .

zn−1
B1(z′, z1) . . . BK−1(zK−2, zK−1)BK(zK−1, z). (11.43)
If a distribution is invariant with respect to each of the base transitions, then obvi-
ously it will also be invariant with respect to either of the T(z′, z) given by (11.42)
or (11.43). For the case of the mixture (11.42), if each of the base transitions sat-
isﬁes detailed balance, then the mixture transition T will also satisfy detailed bal-
ance. This does not hold for the transition probability constructed using (11.43), al-
though by symmetrizing the order of application of the base transitions, in the form
B1, B2, . . . , BK, BK, . . . , B2, B1, detailed balance can be restored. A common ex-
ample of the use of composite transition probabilities is where each base transition
changes only a subset of the variables.
11.2.2
The Metropolis-Hastings algorithm
Earlier we introduced the basic Metropolis algorithm, without actually demon-
strating that it samples from the required distribution. Before giving a proof, we
ﬁrst discuss a generalization, known as the Metropolis-Hastings algorithm (Hast-
ings, 1970), to the case where the proposal distribution is no longer a symmetric
function of its arguments. In particular at step τ of the algorithm, in which the cur-
rent state is z(τ), we draw a sample z⋆from the distribution qk(z|z(τ)) and then
accept it with probability Ak(z⋆, zτ) where
Ak(z⋆, z(τ)) = min

1, p(z⋆)qk(z(τ)|z⋆)
p(z(τ))qk(z⋆|z(τ))

.
(11.44)
Here k labels the members of the set of possible transitions being considered. Again,
the evaluation of the acceptance criterion does not require knowledge of the normal-
izing constant Zp in the probability distribution p(z) = p(z)/Zp. For a symmetric
proposal distribution the Metropolis-Hastings criterion (11.44) reduces to the stan-
dard Metropolis criterion given by (11.33).
We can show that p(z) is an invariant distribution of the Markov chain deﬁned
by the Metropolis-Hastings algorithm by showing that detailed balance, deﬁned by
(11.40), is satisﬁed. Using (11.44) we have
p(z)qk(z|z′)Ak(z′, z)
=
min (p(z)qk(z|z′), p(z′)qk(z′|z))
=
min (p(z′)qk(z′|z), p(z)qk(z|z′))
=
p(z′)qk(z′|z)Ak(z, z′)
(11.45)
as required.
The speciﬁc choice of proposal distribution can have a marked effect on the
performance of the algorithm. For continuous state spaces, a common choice is a
Gaussian centred on the current state, leading to an important trade-off in determin-
ing the variance parameter of this distribution. If the variance is small, then the


---
**Page 557**
542
11. SAMPLING METHODS
Figure 11.10
Schematic illustration of the use of an isotropic
Gaussian proposal distribution (blue circle) to
sample from a correlated multivariate Gaussian
distribution (red ellipse) having very different stan-
dard deviations in different directions, using the
Metropolis-Hastings algorithm. In order to keep
the rejection rate low, the scale ρ of the proposal
distribution should be on the order of the smallest
standard deviation σmin, which leads to random
walk behaviour in which the number of steps sep-
arating states that are approximately independent
is of order (σmax/σmin)2 where σmax is the largest
standard deviation.
σmax
σmin
ρ
proportion of accepted transitions will be high, but progress through the state space
takes the form of a slow random walk leading to long correlation times. However,
if the variance parameter is large, then the rejection rate will be high because, in the
kind of complex problems we are considering, many of the proposed steps will be
to states for which the probability p(z) is low. Consider a multivariate distribution
p(z) having strong correlations between the components of z, as illustrated in Fig-
ure 11.10. The scale ρ of the proposal distribution should be as large as possible
without incurring high rejection rates. This suggests that ρ should be of the same
order as the smallest length scale σmin. The system then explores the distribution
along the more extended direction by means of a random walk, and so the number
of steps to arrive at a state that is more or less independent of the original state is
of order (σmax/σmin)2. In fact in two dimensions, the increase in rejection rate as ρ
increases is offset by the larger steps sizes of those transitions that are accepted, and
more generally for a multivariate Gaussian the number of steps required to obtain
independent samples scales like (σmax/σ2)2 where σ2 is the second-smallest stan-
dard deviation (Neal, 1993). These details aside, it remains the case that if the length
scales over which the distributions vary are very different in different directions, then
the Metropolis Hastings algorithm can have very slow convergence.
11.3. Gibbs Sampling
Gibbs sampling (Geman and Geman, 1984) is a simple and widely applicable Markov
chain Monte Carlo algorithm and can be seen as a special case of the Metropolis-
Hastings algorithm.
Consider the distribution p(z) = p(z1, . . . , zM) from which we wish to sample,
and suppose that we have chosen some initial state for the Markov chain. Each step
of the Gibbs sampling procedure involves replacing the value of one of the variables
by a value drawn from the distribution of that variable conditioned on the values of
the remaining variables. Thus we replace zi by a value drawn from the distribution
p(zi|z\i), where zi denotes the ith component of z, and z\i denotes z1, . . . , zM but
with zi omitted. This procedure is repeated either by cycling through the variables


---
**Page 558**
11.3. Gibbs Sampling
543
in some particular order or by choosing the variable to be updated at each step at
random from some distribution.
For example, suppose we have a distribution p(z1, z2, z3) over three variables,
and at step τ of the algorithm we have selected values z(τ)
1 , z(τ)
2
and z(τ)
3 . We ﬁrst
replace z(τ)
1
by a new value z(τ+1)
1
obtained by sampling from the conditional distri-
bution
p(z1|z(τ)
2 , z(τ)
3 ).
(11.46)
Next we replace z(τ)
2
by a value z(τ+1)
2
obtained by sampling from the conditional
distribution
p(z2|z(τ+1)
1
, z(τ)
3 )
(11.47)
so that the new value for z1 is used straight away in subsequent sampling steps. Then
we update z3 with a sample z(τ+1)
3
drawn from
p(z3|z(τ+1)
1
, z(τ+1)
2
)
(11.48)
and so on, cycling through the three variables in turn.
Gibbs Sampling
1. Initialize {zi : i = 1, . . . , M}
2. For τ = 1, . . . , T:
– Sample z(τ+1)
1
∼p(z1|z(τ)
2 , z(τ)
3 , . . . , z(τ)
M ).
– Sample z(τ+1)
2
∼p(z2|z(τ+1)
1
, z(τ)
3 , . . . , z(τ)
M ).
...
– Sample z(τ+1)
j
∼p(zj|z(τ+1)
1
, . . . , z(τ+1)
j−1 , z(τ)
j+1, . . . , z(τ)
M ).
...
– Sample z(τ+1)
M
∼p(zM|z(τ+1)
1
, z(τ+1)
2
, . . . , z(τ+1)
M−1 ).
Josiah Willard Gibbs
1839–1903
Gibbs spent almost his entire life liv-
ing in a house built by his father in
New Haven, Connecticut. In 1863,
Gibbs was granted the ﬁrst PhD in
engineering in the United States,
and in 1871 he was appointed to
the ﬁrst chair of mathematical physics in the United
States at Yale, a post for which he received no salary
because at the time he had no publications. He de-
veloped the ﬁeld of vector analysis and made contri-
butions to crystallography and planetary orbits.
His
most famous work, entitled On the Equilibrium of Het-
erogeneous Substances, laid the foundations for the
science of physical chemistry.

