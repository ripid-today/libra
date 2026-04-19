# 10 - Approximate Inference
*Pages 461-522 from Pattern Recognition and Machine Learning*

---
**Page 461**
9.3. An Alternative View of EM
445
Consider a set of D binary variables xi, where i = 1, . . . , D, each of which is
governed by a Bernoulli distribution with parameter µi, so that
p(x|µ) =
D

i=1
µxi
i (1 −µi)(1−xi)
(9.44)
where x = (x1, . . . , xD)T and µ = (µ1, . . . , µD)T. We see that the individual
variables xi are independent, given µ. The mean and covariance of this distribution
are easily seen to be
E[x]
=
µ
(9.45)
cov[x]
=
diag{µi(1 −µi)}.
(9.46)
Now let us consider a ﬁnite mixture of these distributions given by
p(x|µ, π) =
K

k=1
πkp(x|µk)
(9.47)
where µ = {µ1, . . . , µK}, π = {π1, . . . , πK}, and
p(x|µk) =
D

i=1
µxi
ki(1 −µki)(1−xi).
(9.48)
The mean and covariance of this mixture distribution are given by
Exercise 9.12
E[x]
=
K

k=1
πkµk
(9.49)
cov[x]
=
K

k=1
πk

Σk + µkµT
k

−E[x]E[x]T
(9.50)
where Σk = diag {µki(1 −µki)}. Because the covariance matrix cov[x] is no
longer diagonal, the mixture distribution can capture correlations between the vari-
ables, unlike a single Bernoulli distribution.
If we are given a data set X = {x1, . . . , xN} then the log likelihood function
for this model is given by
ln p(X|µ, π) =
N

n=1
ln
 K

k=1
πkp(xn|µk)

.
(9.51)
Again we see the appearance of the summation inside the logarithm, so that the
maximum likelihood solution no longer has closed form.
We now derive the EM algorithm for maximizing the likelihood function for
the mixture of Bernoulli distributions. To do this, we ﬁrst introduce an explicit latent


---
**Page 462**
446
9. MIXTURE MODELS AND EM
variable z associated with each instance of x. As in the case of the Gaussian mixture,
z = (z1, . . . , zK)T is a binary K-dimensional variable having a single component
equal to 1, with all other components equal to 0. We can then write the conditional
distribution of x, given the latent variable, as
p(x|z, µ) =
K

k=1
p(x|µk)zk
(9.52)
while the prior distribution for the latent variables is the same as for the mixture of
Gaussians model, so that
p(z|π) =
K

k=1
πzk
k .
(9.53)
If we form the product of p(x|z, µ) and p(z|π) and then marginalize over z, then we
recover (9.47).
Exercise 9.14
In order to derive the EM algorithm, we ﬁrst write down the complete-data log
likelihood function, which is given by
ln p(X, Z|µ, π) =
N

n=1
K

k=1
znk

ln πk
+
D

i=1
[xni ln µki + (1 −xni) ln(1 −µki)]

(9.54)
where X = {xn} and Z = {zn}. Next we take the expectation of the complete-data
log likelihood with respect to the posterior distribution of the latent variables to give
EZ[ln p(X, Z|µ, π)] =
N

n=1
K

k=1
γ(znk)

ln πk
+
D

i=1
[xni ln µki + (1 −xni) ln(1 −µki)]

(9.55)
where γ(znk) = E[znk] is the posterior probability, or responsibility, of component
k given data point xn. In the E step, these responsibilities are evaluated using Bayes’
theorem, which takes the form
γ(znk) = E[znk]
=

znk
znk [πkp(xn|µk)]znk

znj

πjp(xn|µj)
	znj
=
πkp(xn|µk)
K

j=1
πjp(xn|µj)
.
(9.56)


---
**Page 463**
9.3. An Alternative View of EM
447
If we consider the sum over n in (9.55), we see that the responsibilities enter
only through two terms, which can be written as
Nk
=
N

n=1
γ(znk)
(9.57)
xk
=
1
Nk
N

n=1
γ(znk)xn
(9.58)
where Nk is the effective number of data points associated with component k. In the
M step, we maximize the expected complete-data log likelihood with respect to the
parameters µk and π. If we set the derivative of (9.55) with respect to µk equal to
zero and rearrange the terms, we obtain
Exercise 9.15
µk = xk.
(9.59)
We see that this sets the mean of component k equal to a weighted mean of the
data, with weighting coefﬁcients given by the responsibilities that component k takes
for data points. For the maximization with respect to πk, we need to introduce a
Lagrange multiplier to enforce the constraint 
k πk = 1. Following analogous
steps to those used for the mixture of Gaussians, we then obtain
Exercise 9.16
πk = Nk
N
(9.60)
which represents the intuitively reasonable result that the mixing coefﬁcient for com-
ponent k is given by the effective fraction of points in the data set explained by that
component.
Note that in contrast to the mixture of Gaussians, there are no singularities in
which the likelihood function goes to inﬁnity. This can be seen by noting that the
likelihood function is bounded above because 0 ⩽p(xn|µk) ⩽1. There exist
Exercise 9.17
singularities at which the likelihood function goes to zero, but these will not be
found by EM provided it is not initialized to a pathological starting point, because
the EM algorithm always increases the value of the likelihood function, until a local
maximum is found. We illustrate the Bernoulli mixture model in Figure 9.10 by
Section 9.4
using it to model handwritten digits. Here the digit images have been turned into
binary vectors by setting all elements whose values exceed 0.5 to 1 and setting the
remaining elements to 0. We now ﬁt a data set of N = 600 such digits, comprising
the digits ‘2’, ‘3’, and ‘4’, with a mixture of K = 3 Bernoulli distributions by
running 10 iterations of the EM algorithm. The mixing coefﬁcients were initialized
to πk = 1/K, and the parameters µkj were set to random values chosen uniformly in
the range (0.25, 0.75) and then normalized to satisfy the constraint that 
j µkj = 1.
We see that a mixture of 3 Bernoulli distributions is able to ﬁnd the three clusters in
the data set corresponding to the different digits.
The conjugate prior for the parameters of a Bernoulli distribution is given by
the beta distribution, and we have seen that a beta prior is equivalent to introducing


---
**Page 464**
448
9. MIXTURE MODELS AND EM
Figure 9.10
Illustration of the Bernoulli mixture model in which the top row shows examples from the digits data
set after converting the pixel values from grey scale to binary using a threshold of 0.5. On the bottom row the ﬁrst
three images show the parameters µki for each of the three components in the mixture model. As a comparison,
we also ﬁt the same data set using a single multivariate Bernoulli distribution, again using maximum likelihood.
This amounts to simply averaging the counts in each pixel and is shown by the right-most image on the bottom
row.
additional effective observations of x. We can similarly introduce priors into the
Section 2.1.1
Bernoulli mixture model, and use EM to maximize the posterior probability distri-
butions.
Exercise 9.18
It is straightforward to extend the analysis of Bernoulli mixtures to the case of
multinomial binary variables having M > 2 states by making use of the discrete dis-
Exercise 9.19
tribution (2.26). Again, we can introduce Dirichlet priors over the model parameters
if desired.
9.3.4
EM for Bayesian linear regression
As a third example of the application of EM, we return to the evidence ap-
proximation for Bayesian linear regression. In Section 3.5.2, we obtained the re-
estimation equations for the hyperparameters α and β by evaluation of the evidence
and then setting the derivatives of the resulting expression to zero. We now turn to
an alternative approach for ﬁnding α and β based on the EM algorithm. Recall that
our goal is to maximize the evidence function p(t|α, β) given by (3.77) with respect
to α and β. Because the parameter vector w is marginalized out, we can regard it as
a latent variable, and hence we can optimize this marginal likelihood function using
EM. In the E step, we compute the posterior distribution of w given the current set-
ting of the parameters α and β and then use this to ﬁnd the expected complete-data
log likelihood. In the M step, we maximize this quantity with respect to α and β. We
have already derived the posterior distribution of w because this is given by (3.49).
The complete-data log likelihood function is then given by
ln p(t, w|α, β) = ln p(t|w, β) + ln p(w|α)
(9.61)


---
**Page 465**
9.3. An Alternative View of EM
449
where the likelihood p(t|w, β) and the prior p(w|α) are given by (3.10) and (3.52),
respectively, and y(x, w) is given by (3.3). Taking the expectation with respect to
the posterior distribution of w then gives
E [ln p(t, w|α, β)]
=
M
2 ln
 α
2π

−α
2 E 
wTw
	
+ N
2 ln
 β
2π

−β
2
N

n=1
E 
(tn −wTφn)2	
.
(9.62)
Setting the derivatives with respect to α to zero, we obtain the M step re-estimation
equation
Exercise 9.20
α =
M
E [wTw] =
M
mT
NmN + Tr(SN).
(9.63)
An analogous result holds for β.
Exercise 9.21
Note that this re-estimation equation takes a slightly different form from the
corresponding result (3.92) derived by direct evaluation of the evidence function.
However, they each involve computation and inversion (or eigen decomposition) of
an M × M matrix and hence will have comparable computational cost per iteration.
These two approaches to determining α should of course converge to the same
result (assuming they ﬁnd the same local maximum of the evidence function). This
can be veriﬁed by ﬁrst noting that the quantity γ is deﬁned by
γ = M −α
M

i=1
1
λi + α = M −αTr(SN).
(9.64)
At a stationary point of the evidence function, the re-estimation equation (3.92) will
be self-consistently satisﬁed, and hence we can substitute for γ to give
αmT
NmN = γ = M −αTr(SN)
(9.65)
and solving for α we obtain (9.63), which is precisely the EM re-estimation equation.
As a ﬁnal example, we consider a closely related model, namely the relevance
vector machine for regression discussed in Section 7.2.1. There we used direct max-
imization of the marginal likelihood to derive re-estimation equations for the hyper-
parameters α and β. Here we consider an alternative approach in which we view the
weight vector w as a latent variable and apply the EM algorithm. The E step involves
ﬁnding the posterior distribution over the weights, and this is given by (7.81). In the
M step we maximize the expected complete-data log likelihood, which is deﬁned by
Ew [ln p(t|X, w, β)p(w|α)]
(9.66)
where the expectation is taken with respect to the posterior distribution computed
using the ‘old’ parameter values. To compute the new parameter values we maximize
with respect to α and β to give
Exercise 9.22


---
**Page 466**
450
9. MIXTURE MODELS AND EM
αnew
i
=
1
m2
i + Σii
(9.67)
(βnew)−1
=
∥t −ΦmN∥2 + β−1 
i γi
N
(9.68)
These re-estimation equations are formally equivalent to those obtained by direct
maxmization.
Exercise 9.23
9.4. The EM Algorithm in General
The expectation maximization algorithm, or EM algorithm, is a general technique for
ﬁnding maximum likelihood solutions for probabilistic models having latent vari-
ables (Dempster et al., 1977; McLachlan and Krishnan, 1997). Here we give a very
general treatment of the EM algorithm and in the process provide a proof that the
EM algorithm derived heuristically in Sections 9.2 and 9.3 for Gaussian mixtures
does indeed maximize the likelihood function (Csisz`ar and Tusn`ady, 1984; Hath-
away, 1986; Neal and Hinton, 1999). Our discussion will also form the basis for the
derivation of the variational inference framework.
Section 10.1
Consider a probabilistic model in which we collectively denote all of the ob-
served variables by X and all of the hidden variables by Z. The joint distribution
p(X, Z|θ) is governed by a set of parameters denoted θ. Our goal is to maximize
the likelihood function that is given by
p(X|θ) =

Z
p(X, Z|θ).
(9.69)
Here we are assuming Z is discrete, although the discussion is identical if Z com-
prises continuous variables or a combination of discrete and continuous variables,
with summation replaced by integration as appropriate.
We shall suppose that direct optimization of p(X|θ) is difﬁcult, but that opti-
mization of the complete-data likelihood function p(X, Z|θ) is signiﬁcantly easier.
Next we introduce a distribution q(Z) deﬁned over the latent variables, and we ob-
serve that, for any choice of q(Z), the following decomposition holds
ln p(X|θ) = L(q, θ) + KL(q∥p)
(9.70)
where we have deﬁned
L(q, θ)
=

Z
q(Z) ln
p(X, Z|θ)
q(Z)

(9.71)
KL(q∥p)
=
−

Z
q(Z) ln
p(Z|X, θ)
q(Z)

.
(9.72)
Note that L(q, θ) is a functional (see Appendix D for a discussion of functionals)
of the distribution q(Z), and a function of the parameters θ. It is worth studying


---
**Page 467**
9.4. The EM Algorithm in General
451
Figure 9.11
Illustration of the decomposition given
by (9.70), which holds for any choice
of distribution q(Z).
Because the
Kullback-Leibler divergence satisﬁes
KL(q∥p) ⩾0, we see that the quan-
tity L(q, θ) is a lower bound on the log
likelihood function ln p(X|θ).
ln p(X|θ)
L(q, θ)
KL(q||p)
carefully the forms of the expressions (9.71) and (9.72), and in particular noting that
they differ in sign and also that L(q, θ) contains the joint distribution of X and Z
while KL(q∥p) contains the conditional distribution of Z given X. To verify the
decomposition (9.70), we ﬁrst make use of the product rule of probability to give
Exercise 9.24
ln p(X, Z|θ) = ln p(Z|X, θ) + ln p(X|θ)
(9.73)
which we then substitute into the expression for L(q, θ). This gives rise to two terms,
one of which cancels KL(q∥p) while the other gives the required log likelihood
ln p(X|θ) after noting that q(Z) is a normalized distribution that sums to 1.
From (9.72), we see that KL(q∥p) is the Kullback-Leibler divergence between
q(Z) and the posterior distribution p(Z|X, θ). Recall that the Kullback-Leibler di-
vergence satisﬁes KL(q∥p) ⩾0, with equality if, and only if, q(Z) = p(Z|X, θ). It
Section 1.6.1
therefore follows from (9.70) that L(q, θ) ⩽ln p(X|θ), in other words that L(q, θ)
is a lower bound on ln p(X|θ).
The decomposition (9.70) is illustrated in Fig-
ure 9.11.
The EM algorithm is a two-stage iterative optimization technique for ﬁnding
maximum likelihood solutions. We can use the decomposition (9.70) to deﬁne the
EM algorithm and to demonstrate that it does indeed maximize the log likelihood.
Suppose that the current value of the parameter vector is θold. In the E step, the
lower bound L(q, θold) is maximized with respect to q(Z) while holding θold ﬁxed.
The solution to this maximization problem is easily seen by noting that the value
of ln p(X|θold) does not depend on q(Z) and so the largest value of L(q, θold) will
occur when the Kullback-Leibler divergence vanishes, in other words when q(Z) is
equal to the posterior distribution p(Z|X, θold). In this case, the lower bound will
equal the log likelihood, as illustrated in Figure 9.12.
In the subsequent M step, the distribution q(Z) is held ﬁxed and the lower bound
L(q, θ) is maximized with respect to θ to give some new value θnew. This will
cause the lower bound L to increase (unless it is already at a maximum), which will
necessarily cause the corresponding log likelihood function to increase. Because the
distribution q is determined using the old parameter values rather than the new values
and is held ﬁxed during the M step, it will not equal the new posterior distribution
p(Z|X, θnew), and hence there will be a nonzero KL divergence. The increase in the
log likelihood function is therefore greater than the increase in the lower bound, as


---
**Page 468**
452
9. MIXTURE MODELS AND EM
Figure 9.12
Illustration of the E step of
the EM algorithm.
The q
distribution is set equal to
the posterior distribution for
the current parameter val-
ues θold, causing the lower
bound to move up to the
same value as the log like-
lihood function, with the KL
divergence vanishing.
ln p(X|θold)
L(q, θold)
KL(q||p) = 0
shown in Figure 9.13. If we substitute q(Z) = p(Z|X, θold) into (9.71), we see that,
after the E step, the lower bound takes the form
L(q, θ)
=

Z
p(Z|X, θold) ln p(X, Z|θ) −

Z
p(Z|X, θold) ln p(Z|X, θold)
=
Q(θ, θold) + const
(9.74)
where the constant is simply the negative entropy of the q distribution and is there-
fore independent of θ. Thus in the M step, the quantity that is being maximized is the
expectation of the complete-data log likelihood, as we saw earlier in the case of mix-
tures of Gaussians. Note that the variable θ over which we are optimizing appears
only inside the logarithm. If the joint distribution p(Z, X|θ) comprises a member of
the exponential family, or a product of such members, then we see that the logarithm
will cancel the exponential and lead to an M step that will be typically much simpler
than the maximization of the corresponding incomplete-data log likelihood function
p(X|θ).
The operation of the EM algorithm can also be viewed in the space of parame-
ters, as illustrated schematically in Figure 9.14. Here the red curve depicts the (in-
Figure 9.13
Illustration of the M step of the EM
algorithm.
The distribution q(Z)
is held ﬁxed and the lower bound
L(q, θ) is maximized with respect
to the parameter vector θ to give
a revised value θnew. Because the
KL divergence is nonnegative, this
causes the log likelihood ln p(X|θ)
to increase by at least as much as
the lower bound does.
ln p(X|θnew)
L(q, θnew)
KL(q||p)


---
**Page 469**
9.4. The EM Algorithm in General
453
Figure 9.14
The EM algorithm involves alter-
nately computing a lower bound
on the log likelihood for the cur-
rent parameter values and then
maximizing this bound to obtain
the new parameter values.
See
the text for a full discussion.
θold θnew
L (q, θ)
ln p(X|θ)
complete data) log likelihood function whose value we wish to maximize. We start
with some initial parameter value θold, and in the ﬁrst E step we evaluate the poste-
rior distribution over latent variables, which gives rise to a lower bound L(θ, θ(old))
whose value equals the log likelihood at θ(old), as shown by the blue curve. Note that
the bound makes a tangential contact with the log likelihood at θ(old), so that both
curves have the same gradient. This bound is a convex function having a unique
Exercise 9.25
maximum (for mixture components from the exponential family). In the M step, the
bound is maximized giving the value θ(new), which gives a larger value of log likeli-
hood than θ(old). The subsequent E step then constructs a bound that is tangential at
θ(new) as shown by the green curve.
For the particular case of an independent, identically distributed data set, X
will comprise N data points {xn} while Z will comprise N corresponding latent
variables {zn}, where n = 1, . . . , N. From the independence assumption, we have
p(X, Z) = 
n p(xn, zn) and, by marginalizing over the {zn} we have p(X) =

n p(xn). Using the sum and product rules, we see that the posterior probability
that is evaluated in the E step takes the form
p(Z|X, θ) =
p(X, Z|θ)

Z
p(X, Z|θ)
=
N

n=1
p(xn, zn|θ)

Z
N

n=1
p(xn, zn|θ)
=
N

n=1
p(zn|xn, θ)
(9.75)
and so the posterior distribution also factorizes with respect to n. In the case of
the Gaussian mixture model this simply says that the responsibility that each of the
mixture components takes for a particular data point xn depends only on the value
of xn and on the parameters θ of the mixture components, not on the values of the
other data points.
We have seen that both the E and the M steps of the EM algorithm are increas-
ing the value of a well-deﬁned bound on the log likelihood function and that the


---
**Page 470**
454
9. MIXTURE MODELS AND EM
complete EM cycle will change the model parameters in such a way as to cause
the log likelihood to increase (unless it is already at a maximum, in which case the
parameters remain unchanged).
We can also use the EM algorithm to maximize the posterior distribution p(θ|X)
for models in which we have introduced a prior p(θ) over the parameters. To see this,
we note that as a function of θ, we have p(θ|X) = p(θ, X)/p(X) and so
ln p(θ|X) = ln p(θ, X) −ln p(X).
(9.76)
Making use of the decomposition (9.70), we have
ln p(θ|X)
=
L(q, θ) + KL(q∥p) + ln p(θ) −ln p(X)
⩾
L(q, θ) + ln p(θ) −ln p(X).
(9.77)
where ln p(X) is a constant. We can again optimize the right-hand side alternately
with respect to q and θ. The optimization with respect to q gives rise to the same E-
step equations as for the standard EM algorithm, because q only appears in L(q, θ).
The M-step equations are modiﬁed through the introduction of the prior term ln p(θ),
which typically requires only a small modiﬁcation to the standard maximum likeli-
hood M-step equations.
The EM algorithm breaks down the potentially difﬁcult problem of maximizing
the likelihood function into two stages, the E step and the M step, each of which will
often prove simpler to implement. Nevertheless, for complex models it may be the
case that either the E step or the M step, or indeed both, remain intractable. This
leads to two possible extensions of the EM algorithm, as follows.
The generalized EM, or GEM, algorithm addresses the problem of an intractable
M step. Instead of aiming to maximize L(q, θ) with respect to θ, it seeks instead
to change the parameters in such a way as to increase its value. Again, because
L(q, θ) is a lower bound on the log likelihood function, each complete EM cycle of
the GEM algorithm is guaranteed to increase the value of the log likelihood (unless
the parameters already correspond to a local maximum). One way to exploit the
GEM approach would be to use one of the nonlinear optimization strategies, such
as the conjugate gradients algorithm, during the M step. Another form of GEM
algorithm, known as the expectation conditional maximization, or ECM, algorithm,
involves making several constrained optimizations within each M step (Meng and
Rubin, 1993). For instance, the parameters might be partitioned into groups, and the
M step is broken down into multiple steps each of which involves optimizing one of
the subset with the remainder held ﬁxed.
We can similarly generalize the E step of the EM algorithm by performing a
partial, rather than complete, optimization of L(q, θ) with respect to q(Z) (Neal and
Hinton, 1999). As we have seen, for any given value of θ there is a unique maximum
of L(q, θ) with respect to q(Z) that corresponds to the posterior distribution qθ(Z) =
p(Z|X, θ) and that for this choice of q(Z) the bound L(q, θ) is equal to the log
likelihood function ln p(X|θ). It follows that any algorithm that converges to the
global maximum of L(q, θ) will ﬁnd a value of θ that is also a global maximum
of the log likelihood ln p(X|θ). Provided p(X, Z|θ) is a continuous function of θ


---
**Page 471**
Exercises
455
then, by continuity, any local maximum of L(q, θ) will also be a local maximum of
ln p(X|θ).
Consider the case of N independent data points x1, . . . , xN with corresponding
latent variables z1, . . . , zN. The joint distribution p(X, Z|θ) factorizes over the data
points, and this structure can be exploited in an incremental form of EM in which
at each EM cycle only one data point is processed at a time. In the E step, instead
of recomputing the responsibilities for all of the data points, we just re-evaluate the
responsibilities for one data point. It might appear that the subsequent M step would
require computation involving the responsibilities for all of the data points. How-
ever, if the mixture components are members of the exponential family, then the
responsibilities enter only through simple sufﬁcient statistics, and these can be up-
dated efﬁciently. Consider, for instance, the case of a Gaussian mixture, and suppose
we perform an update for data point m in which the corresponding old and new
values of the responsibilities are denoted γold(zmk) and γnew(zmk). In the M step,
the required sufﬁcient statistics can be updated incrementally. For instance, for the
means the sufﬁcient statistics are deﬁned by (9.17) and (9.18) from which we obtain
Exercise 9.26
µnew
k
= µold
k
+
γnew(zmk) −γold(zmk)
N new
k
 
xm −µold
k

(9.78)
together with
N new
k
= N old
k
+ γnew(zmk) −γold(zmk).
(9.79)
The corresponding results for the covariances and the mixing coefﬁcients are analo-
gous.
Thus both the E step and the M step take ﬁxed time that is independent of the
total number of data points. Because the parameters are revised after each data point,
rather than waiting until after the whole data set is processed, this incremental ver-
sion can converge faster than the batch version. Each E or M step in this incremental
algorithm is increasing the value of L(q, θ) and, as we have shown above, if the
algorithm converges to a local (or global) maximum of L(q, θ), this will correspond
to a local (or global) maximum of the log likelihood function ln p(X|θ).
Exercises
9.1
(⋆) www
Consider the K-means algorithm discussed in Section 9.1. Show that as
a consequence of there being a ﬁnite number of possible assignments for the set of
discrete indicator variables rnk, and that for each such assignment there is a unique
optimum for the {µk}, the K-means algorithm must converge after a ﬁnite number
of iterations.
9.2
(⋆)
Apply the Robbins-Monro sequential estimation procedure described in Sec-
tion 2.3.5 to the problem of ﬁnding the roots of the regression function given by
the derivatives of J in (9.1) with respect to µk. Show that this leads to a stochastic
K-means algorithm in which, for each data point xn, the nearest prototype µk is
updated using (9.5).


---
**Page 472**
456
9. MIXTURE MODELS AND EM
9.3
(⋆) www
Consider a Gaussian mixture model in which the marginal distribution
p(z) for the latent variable is given by (9.10), and the conditional distribution p(x|z)
for the observed variable is given by (9.11). Show that the marginal distribution
p(x), obtained by summing p(z)p(x|z) over all possible values of z, is a Gaussian
mixture of the form (9.7).
9.4
(⋆)
Suppose we wish to use the EM algorithm to maximize the posterior distri-
bution over parameters p(θ|X) for a model containing latent variables, where X is
the observed data set. Show that the E step remains the same as in the maximum
likelihood case, whereas in the M step the quantity to be maximized is given by
Q(θ, θold) + ln p(θ) where Q(θ, θold) is deﬁned by (9.30).
9.5
(⋆) Consider the directed graph for a Gaussian mixture model shown in Figure 9.6.
By making use of the d-separation criterion discussed in Section 8.2, show that the
posterior distribution of the latent variables factorizes with respect to the different
data points so that
p(Z|X, µ, Σ, π) =
N

n=1
p(zn|xn, µ, Σ, π).
(9.80)
9.6
(⋆⋆)
Consider a special case of a Gaussian mixture model in which the covari-
ance matrices Σk of the components are all constrained to have a common value
Σ. Derive the EM equations for maximizing the likelihood function under such a
model.
9.7
(⋆) www
Verify that maximization of the complete-data log likelihood (9.36) for
a Gaussian mixture model leads to the result that the means and covariances of each
component are ﬁtted independently to the corresponding group of data points, and
the mixing coefﬁcients are given by the fractions of points in each group.
9.8
(⋆) www
Show that if we maximize (9.40) with respect to µk while keeping the
responsibilities γ(znk) ﬁxed, we obtain the closed form solution given by (9.17).
9.9
(⋆)
Show that if we maximize (9.40) with respect to Σk and πk while keeping the
responsibilities γ(znk) ﬁxed, we obtain the closed form solutions given by (9.19)
and (9.22).
9.10
(⋆⋆) Consider a density model given by a mixture distribution
p(x) =
K

k=1
πkp(x|k)
(9.81)
and suppose that we partition the vector x into two parts so that x = (xa, xb).
Show that the conditional density p(xb|xa) is itself a mixture distribution and ﬁnd
expressions for the mixing coefﬁcients and for the component densities.


---
**Page 473**
Exercises
457
9.11
(⋆)
In Section 9.3.2, we obtained a relationship between K means and EM for
Gaussian mixtures by considering a mixture model in which all components have
covariance ϵI. Show that in the limit ϵ →0, maximizing the expected complete-
data log likelihood for this model, given by (9.40), is equivalent to minimizing the
distortion measure J for the K-means algorithm given by (9.1).
9.12
(⋆) www
Consider a mixture distribution of the form
p(x) =
K

k=1
πkp(x|k)
(9.82)
where the elements of x could be discrete or continuous or a combination of these.
Denote the mean and covariance of p(x|k) by µk and Σk, respectively. Show that
the mean and covariance of the mixture distribution are given by (9.49) and (9.50).
9.13
(⋆⋆)
Using the re-estimation equations for the EM algorithm, show that a mix-
ture of Bernoulli distributions, with its parameters set to values corresponding to a
maximum of the likelihood function, has the property that
E[x] = 1
N
N

n=1
xn ≡x.
(9.83)
Hence show that if the parameters of this model are initialized such that all compo-
nents have the same mean µk = µ for k = 1, . . . , K, then the EM algorithm will
converge after one iteration, for any choice of the initial mixing coefﬁcients, and that
this solution has the property µk = x. Note that this represents a degenerate case of
the mixture model in which all of the components are identical, and in practice we
try to avoid such solutions by using an appropriate initialization.
9.14
(⋆) Consider the joint distribution of latent and observed variables for the Bernoulli
distribution obtained by forming the product of p(x|z, µ) given by (9.52) and p(z|π)
given by (9.53). Show that if we marginalize this joint distribution with respect to z,
then we obtain (9.47).
9.15
(⋆) www
Show that if we maximize the expected complete-data log likelihood
function (9.55) for a mixture of Bernoulli distributions with respect to µk, we obtain
the M step equation (9.59).
9.16
(⋆)
Show that if we maximize the expected complete-data log likelihood function
(9.55) for a mixture of Bernoulli distributions with respect to the mixing coefﬁcients
πk, using a Lagrange multiplier to enforce the summation constraint, we obtain the
M step equation (9.60).
9.17
(⋆) www
Show that as a consequence of the constraint 0 ⩽p(xn|µk) ⩽1 for
the discrete variable xn, the incomplete-data log likelihood function for a mixture
of Bernoulli distributions is bounded above, and hence that there are no singularities
for which the likelihood goes to inﬁnity.


---
**Page 474**
458
9. MIXTURE MODELS AND EM
9.18
(⋆⋆)
Consider a Bernoulli mixture model as discussed in Section 9.3.3, together
with a prior distribution p(µk|ak, bk) over each of the parameter vectors µk given
by the beta distribution (2.13), and a Dirichlet prior p(π|α) given by (2.38). Derive
the EM algorithm for maximizing the posterior probability p(µ, π|X).
9.19
(⋆⋆)
Consider a D-dimensional variable x each of whose components i is itself a
multinomial variable of degree M so that x is a binary vector with components xij
where i = 1, . . . , D and j = 1, . . . , M, subject to the constraint that 
j xij = 1 for
all i. Suppose that the distribution of these variables is described by a mixture of the
discrete multinomial distributions considered in Section 2.2 so that
p(x) =
K

k=1
πkp(x|µk)
(9.84)
where
p(x|µk) =
D

i=1
M

j=1
µxij
kij.
(9.85)
The parameters µkij represent the probabilities p(xij = 1|µk) and must satisfy
0 ⩽µkij ⩽1 together with the constraint 
j µkij = 1 for all values of k and i.
Given an observed data set {xn}, where n = 1, . . . , N, derive the E and M step
equations of the EM algorithm for optimizing the mixing coefﬁcients πk and the
component parameters µkij of this distribution by maximum likelihood.
9.20
(⋆) www
Show that maximization of the expected complete-data log likelihood
function (9.62) for the Bayesian linear regression model leads to the M step re-
estimation result (9.63) for α.
9.21
(⋆⋆) Using the evidence framework of Section 3.5, derive the M-step re-estimation
equations for the parameter β in the Bayesian linear regression model, analogous to
the result (9.63) for α.
9.22
(⋆⋆)
By maximization of the expected complete-data log likelihood deﬁned by
(9.66), derive the M step equations (9.67) and (9.68) for re-estimating the hyperpa-
rameters of the relevance vector machine for regression.
9.23
(⋆⋆) www
In Section 7.2.1 we used direct maximization of the marginal like-
lihood to derive the re-estimation equations (7.87) and (7.88) for ﬁnding values of
the hyperparameters α and β for the regression RVM. Similarly, in Section 9.3.4
we used the EM algorithm to maximize the same marginal likelihood, giving the
re-estimation equations (9.67) and (9.68). Show that these two sets of re-estimation
equations are formally equivalent.
9.24
(⋆)
Verify the relation (9.70) in which L(q, θ) and KL(q∥p) are deﬁned by (9.71)
and (9.72), respectively.


---
**Page 475**
Exercises
459
9.25
(⋆) www
Show that the lower bound L(q, θ) given by (9.71), with q(Z) =
p(Z|X, θ(old)), has the same gradient with respect to θ as the log likelihood function
ln p(X|θ) at the point θ = θ(old).
9.26
(⋆) www
Consider the incremental form of the EM algorithm for a mixture of
Gaussians, in which the responsibilities are recomputed only for a speciﬁc data point
xm. Starting from the M-step formulae (9.17) and (9.18), derive the results (9.78)
and (9.79) for updating the component means.
9.27
(⋆⋆)
Derive M-step formulae for updating the covariance matrices and mixing
coefﬁcients in a Gaussian mixture model when the responsibilities are updated in-
crementally, analogous to the result (9.78) for updating the means.


---
**Page 476**
10
Approximate
Inference
A central task in the application of probabilistic models is the evaluation of the pos-
terior distribution p(Z|X) of the latent variables Z given the observed (visible) data
variables X, and the evaluation of expectations computed with respect to this dis-
tribution. The model might also contain some deterministic parameters, which we
will leave implicit for the moment, or it may be a fully Bayesian model in which any
unknown parameters are given prior distributions and are absorbed into the set of
latent variables denoted by the vector Z. For instance, in the EM algorithm we need
to evaluate the expectation of the complete-data log likelihood with respect to the
posterior distribution of the latent variables. For many models of practical interest, it
will be infeasible to evaluate the posterior distribution or indeed to compute expec-
tations with respect to this distribution. This could be because the dimensionality of
the latent space is too high to work with directly or because the posterior distribution
has a highly complex form for which expectations are not analytically tractable. In
the case of continuous variables, the required integrations may not have closed-form
461


---
**Page 477**
462
10. APPROXIMATE INFERENCE
analytical solutions, while the dimensionality of the space and the complexity of the
integrand may prohibit numerical integration. For discrete variables, the marginal-
izations involve summing over all possible conﬁgurations of the hidden variables,
and though this is always possible in principle, we often ﬁnd in practice that there
may be exponentially many hidden states so that exact calculation is prohibitively
expensive.
In such situations, we need to resort to approximation schemes, and these fall
broadly into two classes, according to whether they rely on stochastic or determin-
istic approximations. Stochastic techniques such as Markov chain Monte Carlo, de-
scribed in Chapter 11, have enabled the widespread use of Bayesian methods across
many domains. They generally have the property that given inﬁnite computational
resource, they can generate exact results, and the approximation arises from the use
of a ﬁnite amount of processor time. In practice, sampling methods can be compu-
tationally demanding, often limiting their use to small-scale problems. Also, it can
be difﬁcult to know whether a sampling scheme is generating independent samples
from the required distribution.
In this chapter, we introduce a range of deterministic approximation schemes,
some of which scale well to large applications. These are based on analytical ap-
proximations to the posterior distribution, for example by assuming that it factorizes
in a particular way or that it has a speciﬁc parametric form such as a Gaussian. As
such, they can never generate exact results, and so their strengths and weaknesses
are complementary to those of sampling methods.
In Section 4.4, we discussed the Laplace approximation, which is based on a
local Gaussian approximation to a mode (i.e., a maximum) of the distribution. Here
we turn to a family of approximation techniques called variational inference or vari-
ational Bayes, which use more global criteria and which have been widely applied.
We conclude with a brief introduction to an alternative variational framework known
as expectation propagation.
10.1. Variational Inference
Variational methods have their origins in the 18th century with the work of Euler,
Lagrange, and others on the calculus of variations. Standard calculus is concerned
with ﬁnding derivatives of functions. We can think of a function as a mapping that
takes the value of a variable as the input and returns the value of the function as the
output. The derivative of the function then describes how the output value varies
as we make inﬁnitesimal changes to the input value. Similarly, we can deﬁne a
functional as a mapping that takes a function as the input and that returns the value
of the functional as the output. An example would be the entropy H[p], which takes
a probability distribution p(x) as the input and returns the quantity
H[p] =

p(x) ln p(x) dx
(10.1)


---
**Page 478**
10.1. Variational Inference
463
as the output. We can the introduce the concept of a functional derivative, which ex-
presses how the value of the functional changes in response to inﬁnitesimal changes
to the input function (Feynman et al., 1964). The rules for the calculus of variations
mirror those of standard calculus and are discussed in Appendix D. Many problems
can be expressed in terms of an optimization problem in which the quantity being
optimized is a functional. The solution is obtained by exploring all possible input
functions to ﬁnd the one that maximizes, or minimizes, the functional. Variational
methods have broad applicability and include such areas as ﬁnite element methods
(Kapur, 1989) and maximum entropy (Schwarz, 1988).
Although there is nothing intrinsically approximate about variational methods,
they do naturally lend themselves to ﬁnding approximate solutions. This is done
by restricting the range of functions over which the optimization is performed, for
instance by considering only quadratic functions or by considering functions com-
posed of a linear combination of ﬁxed basis functions in which only the coefﬁcients
of the linear combination can vary. In the case of applications to probabilistic in-
ference, the restriction may for example take the form of factorization assumptions
(Jordan et al., 1999; Jaakkola, 2001).
Now let us consider in more detail how the concept of variational optimization
can be applied to the inference problem. Suppose we have a fully Bayesian model in
which all parameters are given prior distributions. The model may also have latent
variables as well as parameters, and we shall denote the set of all latent variables
and parameters by Z. Similarly, we denote the set of all observed variables by X.
For example, we might have a set of N independent, identically distributed data,
for which X = {x1, . . . , xN} and Z = {z1, . . . , zN}. Our probabilistic model
speciﬁes the joint distribution p(X, Z), and our goal is to ﬁnd an approximation for
the posterior distribution p(Z|X) as well as for the model evidence p(X). As in our
discussion of EM, we can decompose the log marginal probability using
ln p(X) = L(q) + KL(q∥p)
(10.2)
where we have deﬁned
L(q)
=

q(Z) ln
p(X, Z)
q(Z)

dZ
(10.3)
KL(q∥p)
=
−

q(Z) ln
p(Z|X)
q(Z)

dZ.
(10.4)
This differs from our discussion of EM only in that the parameter vector θ no longer
appears, because the parameters are now stochastic variables and are absorbed into
Z. Since in this chapter we will mainly be interested in continuous variables we have
used integrations rather than summations in formulating this decomposition. How-
ever, the analysis goes through unchanged if some or all of the variables are discrete
simply by replacing the integrations with summations as required. As before, we
can maximize the lower bound L(q) by optimization with respect to the distribution
q(Z), which is equivalent to minimizing the KL divergence. If we allow any possible
choice for q(Z), then the maximum of the lower bound occurs when the KL diver-
gence vanishes, which occurs when q(Z) equals the posterior distribution p(Z|X).


---
**Page 479**
464
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
Figure 10.1
Illustration of the variational approximation for the example considered earlier in Figure 4.14. The
left-hand plot shows the original distribution (yellow) along with the Laplace (red) and variational (green) approx-
imations, and the right-hand plot shows the negative logarithms of the corresponding curves.
However, we shall suppose the model is such that working with the true posterior
distribution is intractable.
We therefore consider instead a restricted family of distributions q(Z) and then
seek the member of this family for which the KL divergence is minimized. Our goal
is to restrict the family sufﬁciently that they comprise only tractable distributions,
while at the same time allowing the family to be sufﬁciently rich and ﬂexible that it
can provide a good approximation to the true posterior distribution. It is important to
emphasize that the restriction is imposed purely to achieve tractability, and that sub-
ject to this requirement we should use as rich a family of approximating distributions
as possible. In particular, there is no ‘over-ﬁtting’ associated with highly ﬂexible dis-
tributions. Using more ﬂexible approximations simply allows us to approach the true
posterior distribution more closely.
One way to restrict the family of approximating distributions is to use a paramet-
ric distribution q(Z|ω) governed by a set of parameters ω. The lower bound L(q)
then becomes a function of ω, and we can exploit standard nonlinear optimization
techniques to determine the optimal values for the parameters. An example of this
approach, in which the variational distribution is a Gaussian and we have optimized
with respect to its mean and variance, is shown in Figure 10.1.
10.1.1
Factorized distributions
Here we consider an alternative way in which to restrict the family of distri-
butions q(Z). Suppose we partition the elements of Z into disjoint groups that we
denote by Zi where i = 1, . . . , M. We then assume that the q distribution factorizes
with respect to these groups, so that
q(Z) =
M

i=1
qi(Zi).
(10.5)


---
**Page 480**
10.1. Variational Inference
465
It should be emphasized that we are making no further assumptions about the distri-
bution. In particular, we place no restriction on the functional forms of the individual
factors qi(Zi). This factorized form of variational inference corresponds to an ap-
proximation framework developed in physics called mean ﬁeld theory (Parisi, 1988).
Amongst all distributions q(Z) having the form (10.5), we now seek that distri-
bution for which the lower bound L(q) is largest. We therefore wish to make a free
form (variational) optimization of L(q) with respect to all of the distributions qi(Zi),
which we do by optimizing with respect to each of the factors in turn. To achieve
this, we ﬁrst substitute (10.5) into (10.3) and then dissect out the dependence on one
of the factors qj(Zj). Denoting qj(Zj) by simply qj to keep the notation uncluttered,
we then obtain
L(q)
=
 
i
qi

ln p(X, Z) −

i
ln qi

dZ
=

qj

ln p(X, Z)

i̸=j
qi dZi

dZj −

qj ln qj dZj + const
=

qj lnp(X, Zj) dZj −

qj ln qj dZj + const
(10.6)
where we have deﬁned a new distribution p(X, Zj) by the relation
lnp(X, Zj) = Ei̸=j[ln p(X, Z)] + const.
(10.7)
Here the notation Ei̸=j[· · · ] denotes an expectation with respect to the q distributions
over all variables zi for i ̸= j, so that
Ei̸=j[ln p(X, Z)] =

ln p(X, Z)

i̸=j
qi dZi.
(10.8)
Now suppose we keep the {qi̸=j} ﬁxed and maximize L(q) in (10.6) with re-
spect to all possible forms for the distribution qj(Zj). This is easily done by rec-
ognizing that (10.6) is a negative Kullback-Leibler divergence between qj(Zj) and
p(X, Zj). Thus maximizing (10.6) is equivalent to minimizing the Kullback-Leibler
Leonhard Euler
1707–1783
Euler was a Swiss mathematician
and physicist who worked in St.
Petersburg and Berlin and who is
widely considered to be one of the
greatest mathematicians of all time.
He is certainly the most proliﬁc, and
his collected works ﬁll 75 volumes. Amongst his many
contributions, he formulated the modern theory of the
function, he developed (together with Lagrange) the
calculus of variations, and he discovered the formula
eiπ = −1, which relates four of the most important
numbers in mathematics. During the last 17 years of
his life, he was almost totally blind, and yet he pro-
duced nearly half of his results during this period.


---
**Page 481**
466
10. APPROXIMATE INFERENCE
divergence, and the minimum occurs when qj(Zj) = p(X, Zj). Thus we obtain a
general expression for the optimal solution q⋆
j (Zj) given by
ln q⋆
j (Zj) = Ei̸=j[ln p(X, Z)] + const.
(10.9)
It is worth taking a few moments to study the form of this solution as it provides the
basis for applications of variational methods. It says that the log of the optimal so-
lution for factor qj is obtained simply by considering the log of the joint distribution
over all hidden and visible variables and then taking the expectation with respect to
all of the other factors {qi} for i ̸= j.
The additive constant in (10.9) is set by normalizing the distribution q⋆
j (Zj).
Thus if we take the exponential of both sides and normalize, we have
q⋆
j (Zj) =
exp (Ei̸=j[ln p(X, Z)])

exp (Ei̸=j[ln p(X, Z)]) dZj
.
In practice, we shall ﬁnd it more convenient to work with the form (10.9) and then re-
instate the normalization constant (where required) by inspection. This will become
clear from subsequent examples.
The set of equations given by (10.9) for j = 1, . . . , M represent a set of con-
sistency conditions for the maximum of the lower bound subject to the factorization
constraint. However, they do not represent an explicit solution because the expres-
sion on the right-hand side of (10.9) for the optimum q⋆
j (Zj) depends on expectations
computed with respect to the other factors qi(Zi) for i ̸= j. We will therefore seek
a consistent solution by ﬁrst initializing all of the factors qi(Zi) appropriately and
then cycling through the factors and replacing each in turn with a revised estimate
given by the right-hand side of (10.9) evaluated using the current estimates for all of
the other factors. Convergence is guaranteed because bound is convex with respect
to each of the factors qi(Zi) (Boyd and Vandenberghe, 2004).
10.1.2
Properties of factorized approximations
Our approach to variational inference is based on a factorized approximation to
the true posterior distribution. Let us consider for a moment the problem of approx-
imating a general distribution by a factorized distribution. To begin with, we discuss
the problem of approximating a Gaussian distribution using a factorized Gaussian,
which will provide useful insight into the types of inaccuracy introduced in using
factorized approximations. Consider a Gaussian distribution p(z) = N(z|µ, Λ−1)
over two correlated variables z = (z1, z2) in which the mean and precision have
elements
µ =

µ1
µ2

,
Λ =

Λ11
Λ12
Λ21
Λ22

(10.10)
and Λ21 = Λ12 due to the symmetry of the precision matrix. Now suppose we
wish to approximate this distribution using a factorized Gaussian of the form q(z) =
q1(z1)q2(z2). We ﬁrst apply the general result (10.9) to ﬁnd an expression for the


---
**Page 482**
10.1. Variational Inference
467
optimal factor q⋆
1(z1). In doing so it is useful to note that on the right-hand side we
only need to retain those terms that have some functional dependence on z1 because
all other terms can be absorbed into the normalization constant. Thus we have
ln q⋆
1(z1)
=
Ez2[ln p(z)] + const
=
Ez2

−1
2(z1 −µ1)2Λ11 −(z1 −µ1)Λ12(z2 −µ2)

+ const
=
−1
2z2
1Λ11 + z1µ1Λ11 −z1Λ12 (E[z2] −µ2) + const.
(10.11)
Next we observe that the right-hand side of this expression is a quadratic function of
z1, and so we can identify q⋆(z1) as a Gaussian distribution. It is worth emphasizing
that we did not assume that q(zi) is Gaussian, but rather we derived this result by
variational optimization of the KL divergence over all possible distributions q(zi).
Note also that we do not need to consider the additive constant in (10.9) explicitly
because it represents the normalization constant that can be found at the end by
inspection if required. Using the technique of completing the square, we can identify
Section 2.3.1
the mean and precision of this Gaussian, giving
q⋆(z1) = N(z1|m1, Λ−1
11 )
(10.12)
where
m1 = µ1 −Λ−1
11 Λ12 (E[z2] −µ2) .
(10.13)
By symmetry, q⋆
2(z2) is also Gaussian and can be written as
q⋆
2(z2) = N(z2|m2, Λ−1
22 )
(10.14)
in which
m2 = µ2 −Λ−1
22 Λ21 (E[z1] −µ1) .
(10.15)
Note that these solutions are coupled, so that q⋆(z1) depends on expectations com-
puted with respect to q⋆(z2) and vice versa. In general, we address this by treating
the variational solutions as re-estimation equations and cycling through the variables
in turn updating them until some convergence criterion is satisﬁed. We shall see
an example of this shortly. Here, however, we note that the problem is sufﬁciently
simple that a closed form solution can be found. In particular, because E[z1] = m1
and E[z2] = m2, we see that the two equations are satisﬁed if we take E[z1] = µ1
and E[z2] = µ2, and it is easily shown that this is the only solution provided the dis-
tribution is nonsingular. This result is illustrated in Figure 10.2(a). We see that the
Exercise 10.2
mean is correctly captured but that the variance of q(z) is controlled by the direction
of smallest variance of p(z), and that the variance along the orthogonal direction is
signiﬁcantly under-estimated. It is a general result that a factorized variational ap-
proximation tends to give approximations to the posterior distribution that are too
compact.
By way of comparison, suppose instead that we had been minimizing the reverse
Kullback-Leibler divergence KL(p∥q). As we shall see, this form of KL divergence


---
**Page 483**
468
10. APPROXIMATE INFERENCE
Figure 10.2
Comparison
of
the two alternative forms for the
Kullback-Leibler divergence.
The
green contours corresponding to
1, 2, and 3 standard deviations for
a correlated Gaussian distribution
p(z) over two variables z1 and z2,
and
the
red
contours
represent
the
corresponding
levels
for
an
approximating
distribution
q(z)
over the same variables given by
the
product
of
two
independent
univariate
Gaussian
distributions
whose parameters are obtained by
minimization of (a) the Kullback-
Leibler divergence KL(q∥p),
and
(b)
the
reverse
Kullback-Leibler
divergence KL(p∥q).
z1
z2
(a)
0
0.5
1
0
0.5
1
z1
z2
(b)
0
0.5
1
0
0.5
1
is used in an alternative approximate inference framework called expectation prop-
agation. We therefore consider the general problem of minimizing KL(p∥q) when
Section 10.7
q(Z) is a factorized approximation of the form (10.5). The KL divergence can then
be written in the form
KL(p∥q) = −

p(Z)
 M

i=1
ln qi(Zi)
 
dZ + const
(10.16)
where the constant term is simply the entropy of p(Z) and so does not depend on
q(Z). We can now optimize with respect to each of the factors qj(Zj), which is
easily done using a Lagrange multiplier to give
Exercise 10.3
q⋆
j (Zj) =

p(Z)

i̸=j
dZi = p(Zj).
(10.17)
In this case, we ﬁnd that the optimal solution for qj(Zj) is just given by the corre-
sponding marginal distribution of p(Z). Note that this is a closed-form solution and
so does not require iteration.
To apply this result to the illustrative example of a Gaussian distribution p(z)
over a vector z we can use (2.98), which gives the result shown in Figure 10.2(b).
We see that once again the mean of the approximation is correct, but that it places
signiﬁcant probability mass in regions of variable space that have very low probabil-
ity.
The difference between these two results can be understood by noting that there
is a large positive contribution to the Kullback-Leibler divergence
KL(q∥p) = −

q(Z) ln
p(Z)
q(Z)

dZ
(10.18)


---
**Page 484**
10.1. Variational Inference
469
(a)
(b)
(c)
Figure 10.3
Another comparison of the two alternative forms for the Kullback-Leibler divergence. (a) The blue
contours show a bimodal distribution p(Z) given by a mixture of two Gaussians, and the red contours correspond
to the single Gaussian distribution q(Z) that best approximates p(Z) in the sense of minimizing the Kullback-
Leibler divergence KL(p∥q). (b) As in (a) but now the red contours correspond to a Gaussian distribution q(Z)
found by numerical minimization of the Kullback-Leibler divergence KL(q∥p). (c) As in (b) but showing a different
local minimum of the Kullback-Leibler divergence.
from regions of Z space in which p(Z) is near zero unless q(Z) is also close to
zero. Thus minimizing this form of KL divergence leads to distributions q(Z) that
avoid regions in which p(Z) is small. Conversely, the Kullback-Leibler divergence
KL(p∥q) is minimized by distributions q(Z) that are nonzero in regions where p(Z)
is nonzero.
We can gain further insight into the different behaviour of the two KL diver-
gences if we consider approximating a multimodal distribution by a unimodal one,
as illustrated in Figure 10.3.
In practical applications, the true posterior distri-
bution will often be multimodal, with most of the posterior mass concentrated in
some number of relatively small regions of parameter space. These multiple modes
may arise through nonidentiﬁability in the latent space or through complex nonlin-
ear dependence on the parameters. Both types of multimodality were encountered in
Chapter 9 in the context of Gaussian mixtures, where they manifested themselves as
multiple maxima in the likelihood function, and a variational treatment based on the
minimization of KL(q∥p) will tend to ﬁnd one of these modes. By contrast, if we
were to minimize KL(p∥q), the resulting approximations would average across all
of the modes and, in the context of the mixture model, would lead to poor predictive
distributions (because the average of two good parameter values is typically itself
not a good parameter value). It is possible to make use of KL(p∥q) to deﬁne a useful
inference procedure, but this requires a rather different approach to the one discussed
here, and will be considered in detail when we discuss expectation propagation.
Section 10.7
The two forms of Kullback-Leibler divergence are members of the alpha family


---
**Page 485**
470
10. APPROXIMATE INFERENCE
of divergences (Ali and Silvey, 1966; Amari, 1985; Minka, 2005) deﬁned by
Dα(p∥q) =
4
1 −α2

1 −

p(x)(1+α)/2q(x)(1−α)/2 dx

(10.19)
where −∞< α < ∞is a continuous parameter. The Kullback-Leibler divergence
KL(p∥q) corresponds to the limit α →1, whereas KL(q∥p) corresponds to the limit
α →−1. For all values of α we have Dα(p∥q) ⩾0, with equality if, and only if,
Exercise 10.6
p(x) = q(x). Suppose p(x) is a ﬁxed distribution, and we minimize Dα(p∥q) with
respect to some set of distributions q(x). Then for α ⩽−1 the divergence is zero
forcing, so that any values of x for which p(x) = 0 will have q(x) = 0, and typically
q(x) will under-estimate the support of p(x) and will tend to seek the mode with the
largest mass. Conversely for α ⩾1 the divergence is zero-avoiding, so that values
of x for which p(x) > 0 will have q(x) > 0, and typically q(x) will stretch to cover
all of p(x), and will over-estimate the support of p(x). When α = 0 we obtain a
symmetric divergence that is linearly related to the Hellinger distance given by
DH(p∥q) =
 
p(x)1/2 −q(x)1/2
dx.
(10.20)
The square root of the Hellinger distance is a valid distance metric.
10.1.3
Example: The univariate Gaussian
We now illustrate the factorized variational approximation using a Gaussian dis-
tribution over a single variable x (MacKay, 2003). Our goal is to infer the posterior
distribution for the mean µ and precision τ, given a data set D = {x1, . . . , xN} of
observed values of x which are assumed to be drawn independently from the Gaus-
sian. The likelihood function is given by
p(D|µ, τ) =
 τ
2π
N/2
exp

−τ
2
N

n=1
(xn −µ)2

.
(10.21)
We now introduce conjugate prior distributions for µ and τ given by
p(µ|τ)
=
N

µ|µ0, (λ0τ)−1
(10.22)
p(τ)
=
Gam(τ|a0, b0)
(10.23)
where Gam(τ|a0, b0) is the gamma distribution deﬁned by (2.146). Together these
distributions constitute a Gaussian-Gamma conjugate prior distribution.
Section 2.3.6
For this simple problem the posterior distribution can be found exactly, and again
takes the form of a Gaussian-gamma distribution. However, for tutorial purposes
Exercise 2.44
we will consider a factorized variational approximation to the posterior distribution
given by
q(µ, τ) = qµ(µ)qτ(τ).
(10.24)


---
**Page 486**
10.1. Variational Inference
471
Note that the true posterior distribution does not factorize in this way. The optimum
factors qµ(µ) and qτ(τ) can be obtained from the general result (10.9) as follows.
For qµ(µ) we have
ln q⋆
µ(µ)
=
Eτ [ln p(D|µ, τ) + ln p(µ|τ)] + const
=
−E[τ]
2

λ0(µ −µ0)2 +
N

n=1
(xn −µ)2

+ const. (10.25)
Completing the square over µ we see that qµ(µ) is a Gaussian N 
µ|µN, λ−1
N

with
mean and precision given by
Exercise 10.7
µN
=
λ0µ0 + Nx
λ0 + N
(10.26)
λN
=
(λ0 + N)E[τ].
(10.27)
Note that for N →∞this gives the maximum likelihood result in which µN = x
and the precision is inﬁnite.
Similarly, the optimal solution for the factor qτ(τ) is given by
ln q⋆
τ(τ)
=
Eµ [ln p(D|µ, τ) + ln p(µ|τ)] + ln p(τ) + const
=
(a0 −1) ln τ −b0τ + N
2 ln τ
−τ
2Eµ
 N

n=1
(xn −µ)2 + λ0(µ −µ0)2
 
+ const
(10.28)
and hence qτ(τ) is a gamma distribution Gam(τ|aN, bN) with parameters
aN
=
a0 + N
2
(10.29)
bN
=
b0 + 1
2Eµ
 N

n=1
(xn −µ)2 + λ0(µ −µ0)2
 
.
(10.30)
Again this exhibits the expected behaviour when N →∞.
Exercise 10.8
It should be emphasized that we did not assume these speciﬁc functional forms
for the optimal distributions qµ(µ) and qτ(τ). They arose naturally from the structure
of the likelihood function and the corresponding conjugate priors.
Section 10.4.1
Thus we have expressions for the optimal distributions qµ(µ) and qτ(τ) each of
which depends on moments evaluated with respect to the other distribution. One ap-
proach to ﬁnding a solution is therefore to make an initial guess for, say, the moment
E[τ] and use this to re-compute the distribution qµ(µ). Given this revised distri-
bution we can then extract the required moments E[µ] and E[µ2], and use these to
recompute the distribution qτ(τ), and so on. Since the space of hidden variables for
this example is only two dimensional, we can illustrate the variational approxima-
tion to the posterior distribution by plotting contours of both the true posterior and
the factorized approximation, as illustrated in Figure 10.4.


---
**Page 487**
472
10. APPROXIMATE INFERENCE
µ
τ
(a)
−1
0
1
0
1
2
µ
τ
(b)
−1
0
1
0
1
2
µ
τ
(c)
−1
0
1
0
1
2
µ
τ
(d)
−1
0
1
0
1
2
Figure 10.4
Illustration of variational inference for the mean µ and precision τ of a univariate Gaussian distribu-
tion. Contours of the true posterior distribution p(µ, τ|D) are shown in green. (a) Contours of the initial factorized
approximation qµ(µ)qτ(τ) are shown in blue. (b) After re-estimating the factor qµ(µ). (c) After re-estimating the
factor qτ(τ). (d) Contours of the optimal factorized approximation, to which the iterative scheme converges, are
shown in red.
In general, we will need to use an iterative approach such as this in order to
solve for the optimal factorized posterior distribution. For the very simple example
we are considering here, however, we can ﬁnd an explicit solution by solving the
simultaneous equations for the optimal factors qµ(µ) and qτ(τ). Before doing this,
we can simplify these expressions by considering broad, noninformative priors in
which µ0 = a0 = b0 = λ0 = 0. Although these parameter settings correspond to
improper priors, we see that the posterior distribution is still well deﬁned. Using the
standard result E[τ] = aN/bN for the mean of a gamma distribution, together with
Appendix B
(10.29) and (10.30), we have
1
E[τ] = E

1
N
N

n=1
(xn −µ)2
 
= x2 −2xE[µ] + E[µ2].
(10.31)
Then, using (10.26) and (10.27), we obtain the ﬁrst and second order moments of


---
**Page 488**
10.1. Variational Inference
473
qµ(µ) in the form
E[µ] = x,
E[µ2] = x2 +
1
NE[τ].
(10.32)
We can now substitute these moments into (10.31) and then solve for E[τ] to give
Exercise 10.9
1
E[τ]
=
1
N −1(x2 −x2)
=
1
N −1
N

n=1
(xn −x)2.
(10.33)
We recognize the right-hand side as the familiar unbiased estimator for the variance
of a univariate Gaussian distribution, and so we see that the use of a Bayesian ap-
proach has avoided the bias of the maximum likelihood solution.
Section 1.2.4
10.1.4
Model comparison
As well as performing inference over the hidden variables Z, we may also
wish to compare a set of candidate models, labelled by the index m, and having
prior probabilities p(m). Our goal is then to approximate the posterior probabilities
p(m|X), where X is the observed data. This is a slightly more complex situation
than that considered so far because different models may have different structure
and indeed different dimensionality for the hidden variables Z. We cannot there-
fore simply consider a factorized approximation q(Z)q(m), but must instead recog-
nize that the posterior over Z must be conditioned on m, and so we must consider
q(Z, m) = q(Z|m)q(m). We can readily verify the following decomposition based
on this variational distribution
Exercise 10.10
ln p(X) = Lm −

m

Z
q(Z|m)q(m) ln
 p(Z, m|X)
q(Z|m)q(m)

(10.34)
where the Lm is a lower bound on ln p(X) and is given by
Lm =

m

Z
q(Z|m)q(m) ln
 p(Z, X, m)
q(Z|m)q(m)

.
(10.35)
Here we are assuming discrete Z, but the same analysis applies to continuous latent
variables provided the summations are replaced with integrations. We can maximize
Lm with respect to the distribution q(m) using a Lagrange multiplier, with the result
Exercise 10.11
q(m) ∝p(m) exp{Lm}.
(10.36)
However, if we maximize Lm with respect to the q(Z|m), we ﬁnd that the solutions
for different m are coupled, as we expect because they are conditioned on m. We
proceed instead by ﬁrst optimizing each of the q(Z|m) individually by optimization


---
**Page 489**
474
10. APPROXIMATE INFERENCE
of (10.35), and then subsequently determining the q(m) using (10.36). After nor-
malization the resulting values for q(m) can be used for model selection or model
averaging in the usual way.
10.2. Illustration: Variational Mixture of Gaussians
We now return to our discussion of the Gaussian mixture model and apply the vari-
ational inference machinery developed in the previous section. This will provide a
good illustration of the application of variational methods and will also demonstrate
how a Bayesian treatment elegantly resolves many of the difﬁculties associated with
the maximum likelihood approach (Attias, 1999b). The reader is encouraged to work
through this example in detail as it provides many insights into the practical appli-
cation of variational methods. Many Bayesian models, corresponding to much more
sophisticated distributions, can be solved by straightforward extensions and general-
izations of this analysis.
Our starting point is the likelihood function for the Gaussian mixture model, il-
lustrated by the graphical model in Figure 9.6. For each observation xn we have
a corresponding latent variable zn comprising a 1-of-K binary vector with ele-
ments znk for k = 1, . . . , K. As before we denote the observed data set by X =
{x1, . . . , xN}, and similarly we denote the latent variables by Z = {z1, . . . , zN}.
From (9.10) we can write down the conditional distribution of Z, given the mixing
coefﬁcients π, in the form
p(Z|π) =
N

n=1
K

k=1
πznk
k
.
(10.37)
Similarly, from (9.11), we can write down the conditional distribution of the ob-
served data vectors, given the latent variables and the component parameters
p(X|Z, µ, Λ) =
N

n=1
K

k=1
N

xn|µk, Λ−1
k
znk
(10.38)
where µ = {µk} and Λ = {Λk}. Note that we are working in terms of precision
matrices rather than covariance matrices as this somewhat simpliﬁes the mathemat-
ics.
Next we introduce priors over the parameters µ, Λ and π. The analysis is con-
siderably simpliﬁed if we use conjugate prior distributions. We therefore choose a
Section 10.4.1
Dirichlet distribution over the mixing coefﬁcients π
p(π) = Dir(π|α0) = C(α0)
K

k=1
πα0−1
k
(10.39)
where by symmetry we have chosen the same parameter α0 for each of the compo-
nents, and C(α0) is the normalization constant for the Dirichlet distribution deﬁned


---
**Page 490**
10.2. Illustration: Variational Mixture of Gaussians
475
Figure 10.5
Directed acyclic graph representing the Bayesian mix-
ture of Gaussians model, in which the box (plate) de-
notes a set of N i.i.d. observations. Here µ denotes
{µk} and Λ denotes {Λk}.
xn
zn
N
π
µ
Λ
by (B.23). As we have seen, the parameter α0 can be interpreted as the effective
Section 2.2.1
prior number of observations associated with each component of the mixture. If the
value of α0 is small, then the posterior distribution will be inﬂuenced primarily by
the data rather than by the prior.
Similarly, we introduce an independent Gaussian-Wishart prior governing the
mean and precision of each Gaussian component, given by
p(µ, Λ)
=
p(µ|Λ)p(Λ)
=
K

k=1
N

µk|m0, (β0Λk)−1
W(Λk|W0, ν0)
(10.40)
because this represents the conjugate prior distribution when both the mean and pre-
cision are unknown. Typically we would choose m0 = 0 by symmetry.
Section 2.3.6
The resulting model can be represented as a directed graph as shown in Fig-
ure 10.5. Note that there is a link from Λ to µ since the variance of the distribution
over µ in (10.40) is a function of Λ.
This example provides a nice illustration of the distinction between latent vari-
ables and parameters. Variables such as zn that appear inside the plate are regarded
as latent variables because the number of such variables grows with the size of the
data set. By contrast, variables such as µ that are outside the plate are ﬁxed in
number independently of the size of the data set, and so are regarded as parameters.
From the perspective of graphical models, however, there is really no fundamental
difference between them.
10.2.1
Variational distribution
In order to formulate a variational treatment of this model, we next write down
the joint distribution of all of the random variables, which is given by
p(X, Z, π, µ, Λ) = p(X|Z, µ, Λ)p(Z|π)p(π)p(µ|Λ)p(Λ)
(10.41)
in which the various factors are deﬁned above. The reader should take a moment to
verify that this decomposition does indeed correspond to the probabilistic graphical
model shown in Figure 10.5. Note that only the variables X = {x1, . . . , xN} are
observed.


---
**Page 491**
476
10. APPROXIMATE INFERENCE
We now consider a variational distribution which factorizes between the latent
variables and the parameters so that
q(Z, π, µ, Λ) = q(Z)q(π, µ, Λ).
(10.42)
It is remarkable that this is the only assumption that we need to make in order to
obtain a tractable practical solution to our Bayesian mixture model. In particular, the
functional form of the factors q(Z) and q(π, µ, Λ) will be determined automatically
by optimization of the variational distribution. Note that we are omitting the sub-
scripts on the q distributions, much as we do with the p distributions in (10.41), and
are relying on the arguments to distinguish the different distributions.
The corresponding sequential update equations for these factors can be easily
derived by making use of the general result (10.9). Let us consider the derivation of
the update equation for the factor q(Z). The log of the optimized factor is given by
ln q⋆(Z) = Eπ,µ,Λ[ln p(X, Z, π, µ, Λ)] + const.
(10.43)
We now make use of the decomposition (10.41). Note that we are only interested in
the functional dependence of the right-hand side on the variable Z. Thus any terms
that do not depend on Z can be absorbed into the additive normalization constant,
giving
ln q⋆(Z) = Eπ[ln p(Z|π)] + Eµ,Λ[ln p(X|Z, µ, Λ)] + const.
(10.44)
Substituting for the two conditional distributions on the right-hand side, and again
absorbing any terms that are independent of Z into the additive constant, we have
ln q⋆(Z) =
N

n=1
K

k=1
znk ln ρnk + const
(10.45)
where we have deﬁned
ln ρnk
=
E[ln πk] + 1
2E [ln |Λk| ] −D
2 ln(2π)
−1
2Eµk,Λk

(xn −µk)TΛk(xn −µk)	
(10.46)
where D is the dimensionality of the data variable x. Taking the exponential of both
sides of (10.45) we obtain
q⋆(Z) ∝
N

n=1
K

k=1
ρznk
nk .
(10.47)
Requiring that this distribution be normalized, and noting that for each value of n
the quantities znk are binary and sum to 1 over all values of k, we obtain
Exercise 10.12
q⋆(Z) =
N

n=1
K

k=1
rznk
nk
(10.48)


---
**Page 492**
10.2. Illustration: Variational Mixture of Gaussians
477
where
rnk =
ρnk
K

j=1
ρnj
.
(10.49)
We see that the optimal solution for the factor q(Z) takes the same functional form
as the prior p(Z|π). Note that because ρnk is given by the exponential of a real
quantity, the quantities rnk will be nonnegative and will sum to one, as required.
For the discrete distribution q⋆(Z) we have the standard result
E[znk] = rnk
(10.50)
from which we see that the quantities rnk are playing the role of responsibilities.
Note that the optimal solution for q⋆(Z) depends on moments evaluated with respect
to the distributions of other variables, and so again the variational update equations
are coupled and must be solved iteratively.
At this point, we shall ﬁnd it convenient to deﬁne three statistics of the observed
data set evaluated with respect to the responsibilities, given by
Nk
=
N

n=1
rnk
(10.51)
xk
=
1
Nk
N

n=1
rnkxn
(10.52)
Sk
=
1
Nk
N

n=1
rnk(xn −xk)(xn −xk)T.
(10.53)
Note that these are analogous to quantities evaluated in the maximum likelihood EM
algorithm for the Gaussian mixture model.
Now let us consider the factor q(π, µ, Λ) in the variational posterior distribu-
tion. Again using the general result (10.9) we have
ln q⋆(π, µ, Λ) = ln p(π) +
K

k=1
ln p(µk, Λk) + EZ [ln p(Z|π)]
+
K

k=1
N

n=1
E[znk] ln N 
xn|µk, Λ−1
k

+ const.
(10.54)
We observe that the right-hand side of this expression decomposes into a sum of
terms involving only π together with terms only involving µ and Λ, which implies
that the variational posterior q(π, µ, Λ) factorizes to give q(π)q(µ, Λ). Further-
more, the terms involving µ and Λ themselves comprise a sum over k of terms
involving µk and Λk leading to the further factorization
q(π, µ, Λ) = q(π)
K

k=1
q(µk, Λk).
(10.55)


---
**Page 493**
478
10. APPROXIMATE INFERENCE
Identifying the terms on the right-hand side of (10.54) that depend on π, we have
ln q⋆(π) = (α0 −1)
K

k=1
ln πk +
K

k=1
N

n=1
rnk ln πk + const
(10.56)
where we have used (10.50). Taking the exponential of both sides, we recognize
q⋆(π) as a Dirichlet distribution
q⋆(π) = Dir(π|α)
(10.57)
where α has components αk given by
αk = α0 + Nk.
(10.58)
Finally, the variational posterior distribution q⋆(µk, Λk) does not factorize into
the product of the marginals, but we can always use the product rule to write it in the
form q⋆(µk, Λk) = q⋆(µk|Λk)q⋆(Λk). The two factors can be found by inspecting
(10.54) and reading off those terms that involve µk and Λk. The result, as expected,
is a Gaussian-Wishart distribution and is given by
Exercise 10.13
q⋆(µk, Λk) = N

µk|mk, (βkΛk)−1
W(Λk|Wk, νk)
(10.59)
where we have deﬁned
βk
=
β0 + Nk
(10.60)
mk
=
1
βk
(β0m0 + Nkxk)
(10.61)
W−1
k
=
W−1
0
+ NkSk +
β0Nk
β0 + Nk
(xk −m0)(xk −m0)T
(10.62)
νk
=
ν0 + Nk.
(10.63)
These update equations are analogous to the M-step equations of the EM algorithm
for the maximum likelihood solution of the mixture of Gaussians. We see that the
computations that must be performed in order to update the variational posterior
distribution over the model parameters involve evaluation of the same sums over the
data set, as arose in the maximum likelihood treatment.
In order to perform this variational M step, we need the expectations E[znk] =
rnk representing the responsibilities. These are obtained by normalizing the ρnk that
are given by (10.46). We see that this expression involves expectations with respect
to the variational distributions of the parameters, and these are easily evaluated to
give
Exercise 10.14
Eµk,Λk

(xn −µk)TΛk(xn −µk)
	
=
Dβ−1
k
+ νk(xn −mk)TWk(xn −mk)
(10.64)
ln Λk ≡E [ln |Λk| ]
=
D

i=1
ψ
νk + 1 −i
2

+ D ln 2 + ln |Wk| (10.65)
ln πk ≡E [ln πk]
=
ψ(αk) −ψ(α)
(10.66)


---
**Page 494**
10.2. Illustration: Variational Mixture of Gaussians
479
where we have introduced deﬁnitions of Λk and πk, and ψ(·) is the digamma function
deﬁned by (B.25), with α = 
k αk. The results (10.65) and (10.66) follow from
the standard properties of the Wishart and Dirichlet distributions.
Appendix B
If we substitute (10.64), (10.65), and (10.66) into (10.46) and make use of
(10.49), we obtain the following result for the responsibilities
rnk ∝πkΛ1/2
k
exp

−D
2βk
−νk
2 (xn −mk)TWk(xn −mk)

.
(10.67)
Notice the similarity to the corresponding result for the responsibilities in maximum
likelihood EM, which from (9.13) can be written in the form
rnk ∝πk|Λk|1/2 exp

−1
2(xn −µk)TΛk(xn −µk)

(10.68)
where we have used the precision in place of the covariance to highlight the similarity
to (10.67).
Thus the optimization of the variational posterior distribution involves cycling
between two stages analogous to the E and M steps of the maximum likelihood EM
algorithm. In the variational equivalent of the E step, we use the current distributions
over the model parameters to evaluate the moments in (10.64), (10.65), and (10.66)
and hence evaluate E[znk] = rnk. Then in the subsequent variational equivalent
of the M step, we keep these responsibilities ﬁxed and use them to re-compute the
variational distribution over the parameters using (10.57) and (10.59). In each case,
we see that the variational posterior distribution has the same functional form as the
corresponding factor in the joint distribution (10.41). This is a general result and is
a consequence of the choice of conjugate distributions.
Section 10.4.1
Figure 10.6 shows the results of applying this approach to the rescaled Old Faith-
ful data set for a Gaussian mixture model having K = 6 components. We see that
after convergence, there are only two components for which the expected values
of the mixing coefﬁcients are numerically distinguishable from their prior values.
This effect can be understood qualitatively in terms of the automatic trade-off in a
Bayesian model between ﬁtting the data and the complexity of the model, in which
Section 3.4
the complexity penalty arises from components whose parameters are pushed away
from their prior values. Components that take essentially no responsibility for ex-
plaining the data points have rnk ≃0 and hence Nk ≃0. From (10.58), we see
that αk ≃α0 and from (10.60)–(10.63) we see that the other parameters revert to
their prior values. In principle such components are ﬁtted slightly to the data points,
but for broad priors this effect is too small to be seen numerically. For the varia-
tional Gaussian mixture model the expected values of the mixing coefﬁcients in the
posterior distribution are given by
Exercise 10.15
E[πk] = αk + Nk
Kα0 + N .
(10.69)
Consider a component for which Nk ≃0 and αk ≃α0. If the prior is broad so that
α0 →0, then E[πk] →0 and the component plays no role in the model, whereas if


---
**Page 495**
480
10. APPROXIMATE INFERENCE
Figure 10.6
Variational
Bayesian
mixture of K = 6 Gaussians ap-
plied to the Old Faithful data set, in
which the ellipses denote the one
standard-deviation density contours
for each of the components, and the
density of red ink inside each ellipse
corresponds to the mean value of
the mixing coefﬁcient for each com-
ponent. The number in the top left
of each diagram shows the num-
ber of iterations of variational infer-
ence. Components whose expected
mixing coefﬁcient are numerically in-
distinguishable from zero are not
plotted.
0
15
60
120
the prior tightly constrains the mixing coefﬁcients so that α0 →∞, then E[πk] →
1/K.
In Figure 10.6, the prior over the mixing coefﬁcients is a Dirichlet of the form
(10.39). Recall from Figure 2.5 that for α0 < 1 the prior favours solutions in which
some of the mixing coefﬁcients are zero. Figure 10.6 was obtained using α0 = 10−3,
and resulted in two components having nonzero mixing coefﬁcients. If instead we
choose α0 = 1 we obtain three components with nonzero mixing coefﬁcients, and
for α = 10 all six components have nonzero mixing coefﬁcients.
As we have seen there is a close similarity between the variational solution for
the Bayesian mixture of Gaussians and the EM algorithm for maximum likelihood.
In fact if we consider the limit N →∞then the Bayesian treatment converges to the
maximum likelihood EM algorithm. For anything other than very small data sets,
the dominant computational cost of the variational algorithm for Gaussian mixtures
arises from the evaluation of the responsibilities, together with the evaluation and
inversion of the weighted data covariance matrices. These computations mirror pre-
cisely those that arise in the maximum likelihood EM algorithm, and so there is little
computational overhead in using this Bayesian approach as compared to the tradi-
tional maximum likelihood one. There are, however, some substantial advantages.
First of all, the singularities that arise in maximum likelihood when a Gaussian com-
ponent ‘collapses’ onto a speciﬁc data point are absent in the Bayesian treatment.


---
**Page 496**
10.2. Illustration: Variational Mixture of Gaussians
481
Indeed, these singularities are removed if we simply introduce a prior and then use a
MAP estimate instead of maximum likelihood. Furthermore, there is no over-ﬁtting
if we choose a large number K of components in the mixture, as we saw in Fig-
ure 10.6. Finally, the variational treatment opens up the possibility of determining
the optimal number of components in the mixture without resorting to techniques
such as cross validation.
Section 10.2.4
10.2.2
Variational lower bound
We can also straightforwardly evaluate the lower bound (10.3) for this model.
In practice, it is useful to be able to monitor the bound during the re-estimation in
order to test for convergence. It can also provide a valuable check on both the math-
ematical expressions for the solutions and their software implementation, because at
each step of the iterative re-estimation procedure the value of this bound should not
decrease. We can take this a stage further to provide a deeper test of the correctness
of both the mathematical derivation of the update equations and of their software im-
plementation by using ﬁnite differences to check that each update does indeed give
a (constrained) maximum of the bound (Svens´en and Bishop, 2004).
For the variational mixture of Gaussians, the lower bound (10.3) is given by
L
=

Z

q(Z, π, µ, Λ) ln
p(X, Z, π, µ, Λ)
q(Z, π, µ, Λ)

dπ dµ dΛ
=
E[ln p(X, Z, π, µ, Λ)] −E[ln q(Z, π, µ, Λ)]
=
E[ln p(X|Z, µ, Λ)] + E[ln p(Z|π)] + E[ln p(π)] + E[ln p(µ, Λ)]
−E[ln q(Z)] −E[ln q(π)] −E[ln q(µ, Λ)]
(10.70)
where, to keep the notation uncluttered, we have omitted the ⋆superscript on the
q distributions, along with the subscripts on the expectation operators because each
expectation is taken with respect to all of the random variables in its argument. The
various terms in the bound are easily evaluated to give the following results
Exercise 10.16
E[ln p(X|Z, µ, Λ)] = 1
2
K

k=1
Nk

ln Λk −Dβ−1
k
−νkTr(SkWk)
−νk(xk −mk)TWk(xk −mk) −D ln(2π)

(10.71)
E[ln p(Z|π)]
=
N

n=1
K

k=1
rnk ln πk
(10.72)
E[ln p(π)]
=
ln C(α0) + (α0 −1)
K

k=1
ln πk
(10.73)


---
**Page 497**
482
10. APPROXIMATE INFERENCE
E[ln p(µ, Λ)] = 1
2
K

k=1

D ln(β0/2π) + ln Λk −Dβ0
βk
−β0νk(mk −m0)TWk(mk −m0)

+ K ln B(W0, ν0)
+(ν0 −D −1)
2
K

k=1
ln Λk −1
2
K

k=1
νkTr(W−1
0 Wk)
(10.74)
E[ln q(Z)] =
N

n=1
K

k=1
rnk ln rnk
(10.75)
E[ln q(π)] =
K

k=1
(αk −1) ln πk + ln C(α)
(10.76)
E[ln q(µ, Λ)] =
K

k=1
1
2 ln Λk + D
2 ln
βk
2π

−D
2 −H [q(Λk)]

(10.77)
where D is the dimensionality of x, H[q(Λk)] is the entropy of the Wishart distribu-
tion given by (B.82), and the coefﬁcients C(α) and B(W, ν) are deﬁned by (B.23)
and (B.79), respectively. Note that the terms involving expectations of the logs of the
q distributions simply represent the negative entropies of those distributions. Some
simpliﬁcations and combination of terms can be performed when these expressions
are summed to give the lower bound. However, we have kept the expressions sepa-
rate for ease of understanding.
Finally, it is worth noting that the lower bound provides an alternative approach
for deriving the variational re-estimation equations obtained in Section 10.2.1. To do
this we use the fact that, since the model has conjugate priors, the functional form of
the factors in the variational posterior distribution is known, namely discrete for Z,
Dirichlet for π, and Gaussian-Wishart for (µk, Λk). By taking general parametric
forms for these distributions we can derive the form of the lower bound as a function
of the parameters of the distributions. Maximizing the bound with respect to these
parameters then gives the required re-estimation equations.
Exercise 10.18
10.2.3
Predictive density
In applications of the Bayesian mixture of Gaussians model we will often be
interested in the predictive density for a new value x of the observed variable. As-
sociated with this observation will be a corresponding latent variable z, and the pre-
dictive density is then given by
p(x|X) =

bz

p(x|z, µ, Λ)p(z|π)p(π, µ, Λ|X) dπ dµ dΛ
(10.78)


---
**Page 498**
10.2. Illustration: Variational Mixture of Gaussians
483
where p(π, µ, Λ|X) is the (unknown) true posterior distribution of the parameters.
Using (10.37) and (10.38) we can ﬁrst perform the summation over z to give
p(x|X) =
K

k=1

πkN

x|µk, Λ−1
k

p(π, µ, Λ|X) dπ dµ dΛ.
(10.79)
Because the remaining integrations are intractable, we approximate the predictive
density by replacing the true posterior distribution p(π, µ, Λ|X) with its variational
approximation q(π)q(µ, Λ) to give
p(x|X) =
K

k=1

πkN 
x|µk, Λ−1
k

q(π)q(µk, Λk) dπ dµk dΛk
(10.80)
where we have made use of the factorization (10.55) and in each term we have im-
plicitly integrated out all variables {µj, Λj} for j ̸= k The remaining integrations
can now be evaluated analytically giving a mixture of Student’s t-distributions
Exercise 10.19
p(x|X) = 1
α
K

k=1
αkSt(x|mk, Lk, νk + 1 −D)
(10.81)
in which the kth component has mean mk, and the precision is given by
Lk = (νk + 1 −D)βk
(1 + βk)
Wk
(10.82)
in which νk is given by (10.63). When the size N of the data set is large the predictive
distribution (10.81) reduces to a mixture of Gaussians.
Exercise 10.20
10.2.4
Determining the number of components
We have seen that the variational lower bound can be used to determine a pos-
terior distribution over the number K of components in the mixture model. There
Section 10.1.4
is, however, one subtlety that needs to be addressed. For any given setting of the
parameters in a Gaussian mixture model (except for speciﬁc degenerate settings),
there will exist other parameter settings for which the density over the observed vari-
ables will be identical. These parameter values differ only through a re-labelling of
the components. For instance, consider a mixture of two Gaussians and a single ob-
served variable x, in which the parameters have the values π1 = a, π2 = b, µ1 = c,
µ2 = d, σ1 = e, σ2 = f. Then the parameter values π1 = b, π2 = a, µ1 = d,
µ2 = c, σ1 = f, σ2 = e, in which the two components have been exchanged, will
by symmetry give rise to the same value of p(x). If we have a mixture model com-
prising K components, then each parameter setting will be a member of a family of
K! equivalent settings.
Exercise 10.21
In the context of maximum likelihood, this redundancy is irrelevant because the
parameter optimization algorithm (for example EM) will, depending on the initial-
ization of the parameters, ﬁnd one speciﬁc solution, and the other equivalent solu-
tions play no role. In a Bayesian setting, however, we marginalize over all possible


---
**Page 499**
484
10. APPROXIMATE INFERENCE
Figure 10.7
Plot of the variational lower bound
L versus the number K of com-
ponents in the Gaussian mixture
model, for the Old Faithful data,
showing a distinct peak at K =
2 components.
For each value
of K, the model is trained from
100 different random starts, and
the results shown as ‘+’ symbols
plotted with small random hori-
zontal perturbations so that they
can be distinguished.
Note that
some solutions ﬁnd suboptimal
local maxima, but that this hap-
pens infrequently.
K
p(D|K)
1
2
3
4
5
6
parameter values. We have seen in Figure 10.2 that if the true posterior distribution
is multimodal, variational inference based on the minimization of KL(q∥p) will tend
to approximate the distribution in the neighbourhood of one of the modes and ignore
the others. Again, because equivalent modes have equivalent predictive densities,
this is of no concern provided we are considering a model having a speciﬁc number
K of components. If, however, we wish to compare different values of K, then we
need to take account of this multimodality. A simple approximate solution is to add
a term ln K! onto the lower bound when used for model comparison and averaging.
Exercise 10.22
Figure 10.7 shows a plot of the lower bound, including the multimodality fac-
tor, versus the number K of components for the Old Faithful data set. It is worth
emphasizing once again that maximum likelihood would lead to values of the likeli-
hood function that increase monotonically with K (assuming the singular solutions
have been avoided, and discounting the effects of local maxima) and so cannot be
used to determine an appropriate model complexity. By contrast, Bayesian inference
automatically makes the trade-off between model complexity and ﬁtting the data.
Section 3.4
This approach to the determination of K requires that a range of models having
different K values be trained and compared. An alternative approach to determining
a suitable value for K is to treat the mixing coefﬁcients π as parameters and make
point estimates of their values by maximizing the lower bound (Corduneanu and
Bishop, 2001) with respect to π instead of maintaining a probability distribution
over them as in the fully Bayesian approach. This leads to the re-estimation equation
Exercise 10.23
πk = 1
N
N

n=1
rnk
(10.83)
and this maximization is interleaved with the variational updates for the q distribution
over the remaining parameters. Components that provide insufﬁcient contribution


---
**Page 500**
10.2. Illustration: Variational Mixture of Gaussians
485
to explaining the data will have their mixing coefﬁcients driven to zero during the
optimization, and so they are effectively removed from the model through automatic
relevance determination. This allows us to make a single training run in which we
start with a relatively large initial value of K, and allow surplus components to be
pruned out of the model. The origins of the sparsity when optimizing with respect to
hyperparameters is discussed in detail in the context of the relevance vector machine.
Section 7.2.2
10.2.5
Induced factorizations
In deriving these variational update equations for the Gaussian mixture model,
we assumed a particular factorization of the variational posterior distribution given
by (10.42). However, the optimal solutions for the various factors exhibit additional
factorizations. In particular, the solution for q⋆(µ, Λ) is given by the product of an
independent distribution q⋆(µk, Λk) over each of the components k of the mixture,
whereas the variational posterior distribution q⋆(Z) over the latent variables, given
by (10.48), factorizes into an independent distribution q⋆(zn) for each observation n
(note that it does not further factorize with respect to k because, for each value of n,
the znk are constrained to sum to one over k). These additional factorizations are a
consequence of the interaction between the assumed factorization and the conditional
independence properties of the true distribution, as characterized by the directed
graph in Figure 10.5.
We shall refer to these additional factorizations as induced factorizations be-
cause they arise from an interaction between the factorization assumed in the varia-
tional posterior distribution and the conditional independence properties of the true
joint distribution. In a numerical implementation of the variational approach it is
important to take account of such additional factorizations. For instance, it would
be very inefﬁcient to maintain a full precision matrix for the Gaussian distribution
over a set of variables if the optimal form for that distribution always had a diago-
nal precision matrix (corresponding to a factorization with respect to the individual
variables described by that Gaussian).
Such induced factorizations can easily be detected using a simple graphical test
based on d-separation as follows. We partition the latent variables into three disjoint
groups A, B, C and then let us suppose that we are assuming a factorization between
C and the remaining latent variables, so that
q(A, B, C) = q(A, B)q(C).
(10.84)
Using the general result (10.9), together with the product rule for probabilities, we
see that the optimal solution for q(A, B) is given by
ln q⋆(A, B)
=
EC[ln p(X, A, B, C)] + const
=
EC[ln p(A, B|X, C)] + const.
(10.85)
We now ask whether this resulting solution will factorize between A and B, in
other words whether q⋆(A, B) = q⋆(A)q⋆(B). This will happen if, and only if,
ln p(A, B|X, C) = ln p(A|X, C) + ln p(B|X, C), that is, if the conditional inde-
pendence relation
A ⊥⊥B | X, C
(10.86)


---
**Page 501**
486
10. APPROXIMATE INFERENCE
is satisﬁed. We can test to see if this relation does hold, for any choice of A and B
by making use of the d-separation criterion.
To illustrate this, consider again the Bayesian mixture of Gaussians represented
by the directed graph in Figure 10.5, in which we are assuming a variational fac-
torization given by (10.42). We can see immediately that the variational posterior
distribution over the parameters must factorize between π and the remaining param-
eters µ and Λ because all paths connecting π to either µ or Λ must pass through
one of the nodes zn all of which are in the conditioning set for our conditional inde-
pendence test and all of which are head-to-tail with respect to such paths.
10.3. Variational Linear Regression
As a second illustration of variational inference, we return to the Bayesian linear
regression model of Section 3.3. In the evidence framework, we approximated the
integration over α and β by making point estimates obtained by maximizing the log
marginal likelihood. A fully Bayesian approach would integrate over the hyperpa-
rameters as well as over the parameters. Although exact integration is intractable,
we can use variational methods to ﬁnd a tractable approximation. In order to sim-
plify the discussion, we shall suppose that the noise precision parameter β is known,
and is ﬁxed to its true value, although the framework is easily extended to include
the distribution over β. For the linear regression model, the variational treatment
Exercise 10.26
will turn out to be equivalent to the evidence framework. Nevertheless, it provides a
good exercise in the use of variational methods and will also lay the foundation for
variational treatment of Bayesian logistic regression in Section 10.6.
Recall that the likelihood function for w, and the prior over w, are given by
p(t|w)
=
N

n=1
N(tn|wTφn, β−1)
(10.87)
p(w|α)
=
N(w|0, α−1I)
(10.88)
where φn = φ(xn). We now introduce a prior distribution over α. From our dis-
cussion in Section 2.3.6, we know that the conjugate prior for the precision of a
Gaussian is given by a gamma distribution, and so we choose
p(α) = Gam(α|a0, b0)
(10.89)
where Gam(·|·, ·) is deﬁned by (B.26). Thus the joint distribution of all the variables
is given by
p(t, w, α) = p(t|w)p(w|α)p(α).
(10.90)
This can be represented as a directed graphical model as shown in Figure 10.8.
10.3.1
Variational distribution
Our ﬁrst goal is to ﬁnd an approximation to the posterior distribution p(w, α|t).
To do this, we employ the variational framework of Section 10.1, with a variational


---
**Page 502**
10.3. Variational Linear Regression
487
Figure 10.8
Probabilistic graphical model representing the joint dis-
tribution (10.90) for the Bayesian linear regression
model.
tn
φn
N
w
α
β
posterior distribution given by the factorized expression
q(w, α) = q(w)q(α).
(10.91)
We can ﬁnd re-estimation equations for the factors in this distribution by making use
of the general result (10.9). Recall that for each factor, we take the log of the joint
distribution over all variables and then average with respect to those variables not in
that factor. Consider ﬁrst the distribution over α. Keeping only terms that have a
functional dependence on α, we have
ln q⋆(α) = ln p(α) + Ew [ln p(w|α)] + const
=
(a0 −1) ln α −b0α + M
2 ln α −α
2 E[wTw] + const.
(10.92)
We recognize this as the log of a gamma distribution, and so identifying the coefﬁ-
cients of α and ln α we obtain
q⋆(α) = Gam(α|aN, bN)
(10.93)
where
aN
=
a0 + M
2
(10.94)
bN
=
b0 + 1
2E[wTw].
(10.95)
Similarly, we can ﬁnd the variational re-estimation equation for the posterior
distribution over w. Again, using the general result (10.9), and keeping only those
terms that have a functional dependence on w, we have
ln q⋆(w)
=
ln p(t|w) + Eα [ln p(w|α)] + const
(10.96)
=
−β
2
N

n=1
{wTφn −tn}2 −1
2E[α]wTw + const
(10.97)
=
−1
2wT 
E[α]I + βΦTΦ

w + βwTΦTt + const.
(10.98)
Because this is a quadratic form, the distribution q⋆(w) is Gaussian, and so we can
complete the square in the usual way to identify the mean and covariance, giving
q⋆(w) = N(w|mN, SN)
(10.99)


---
**Page 503**
488
10. APPROXIMATE INFERENCE
where
mN
=
βSNΦTt
(10.100)
SN
=

E[α]I + βΦTΦ−1 .
(10.101)
Note the close similarity to the posterior distribution (3.52) obtained when α was
treated as a ﬁxed parameter. The difference is that here α is replaced by its expecta-
tion E[α] under the variational distribution. Indeed, we have chosen to use the same
notation for the covariance matrix SN in both cases.
Using the standard results (B.27), (B.38), and (B.39), we can obtain the required
moments as follows
E[α]
=
aN/bN
(10.102)
E[wwT]
=
mNmT
N + SN.
(10.103)
The evaluation of the variational posterior distribution begins by initializing the pa-
rameters of one of the distributions q(w) or q(α), and then alternately re-estimates
these factors in turn until a suitable convergence criterion is satisﬁed (usually speci-
ﬁed in terms of the lower bound to be discussed shortly).
It is instructive to relate the variational solution to that found using the evidence
framework in Section 3.5. To do this consider the case a0 = b0 = 0, corresponding
to the limit of an inﬁnitely broad prior over α. The mean of the variational posterior
distribution q(α) is then given by
E[α] = aN
bN
=
M/2
E[wTw]/2 =
M
mT
NmN + Tr(SN).
(10.104)
Comparison with (9.63) shows that in the case of this particularly simple model,
the variational approach gives precisely the same expression as that obtained by
maximizing the evidence function using EM except that the point estimate for α
is replaced by its expected value. Because the distribution q(w) depends on q(α)
only through the expectation E[α], we see that the two approaches will give identical
results for the case of an inﬁnitely broad prior.
10.3.2
Predictive distribution
The predictive distribution over t, given a new input x, is easily evaluated for
this model using the Gaussian variational posterior for the parameters
p(t|x, t)
=

p(t|x, w)p(w|t) dw
≃

p(t|x, w)q(w) dw
=

N(t|wTφ(x), β−1)N(w|mN, SN) dw
=
N(t|mT
Nφ(x), σ2(x))
(10.105)


---
**Page 504**
10.3. Variational Linear Regression
489
where we have evaluated the integral by making use of the result (2.115) for the
linear-Gaussian model. Here the input-dependent variance is given by
σ2(x) = 1
β + φ(x)TSNφ(x).
(10.106)
Note that this takes the same form as the result (3.59) obtained with ﬁxed α except
that now the expected value E[α] appears in the deﬁnition of SN.
10.3.3
Lower bound
Another quantity of importance is the lower bound L deﬁned by
L(q)
=
E[ln p(w, α, t)] −E[ln q(w, α)]
=
Ew[ln p(t|w)] + Ew,α[ln p(w|α)] + Eα[ln p(α)]
−Eα[ln q(w)]w −E[ln q(α)].
(10.107)
Evaluation of the various terms is straightforward, making use of results obtained in
Exercise 10.27
previous chapters, and gives
E[ln p(t|w)]w
=
N
2 ln
 β
2π

−β
2 tTt + βmT
NΦTt
−β
2 Tr

ΦTΦ(mNmT
N + SN)
	
(10.108)
E[ln p(w|α)]w,α
=
−M
2 ln(2π) + M
2 (ψ(aN) −ln bN)
−aN
2bN

mT
NmN + Tr(SN)
	
(10.109)
E[ln p(α)]α
=
a0 ln b0 + (a0 −1) [ψ(aN) −ln bN]
−b0
aN
bN
−ln Γ(aN)
(10.110)
−E[ln q(w)]w
=
1
2 ln |SN| + M
2 [1 + ln(2π)]
(10.111)
−E[ln q(α)]α
=
ln Γ(aN) −(aN −1)ψ(aN) −ln bN + aN. (10.112)
Figure 10.9 shows a plot of the lower bound L(q) versus the degree of a polynomial
model for a synthetic data set generated from a degree three polynomial. Here the
prior parameters have been set to a0 = b0 = 0, corresponding to the noninformative
prior p(α) ∝1/α, which is uniform over ln α as discussed in Section 2.3.6. As
we saw in Section 10.1, the quantity L represents lower bound on the log marginal
likelihood p(t|M) for the model. If we assign equal prior probabilities p(M) to the
different values of M, then we can interpret L as an approximation to the poste-
rior model probability p(M|t). Thus the variational framework assigns the highest
probability to the model with M = 3. This should be contrasted with the maximum
likelihood result, which assigns ever smaller residual error to models of increasing
complexity until the residual error is driven to zero, causing maximum likelihood to
favour severely over-ﬁtted models.


---
**Page 505**
490
10. APPROXIMATE INFERENCE
Figure 10.9
Plot of the lower bound L ver-
sus the order M of the polyno-
mial, for a polynomial model, in
which a set of 10 data points is
generated from a polynomial with
M = 3 sampled over the inter-
val (−5, 5) with additive Gaussian
noise of variance 0.09. The value
of the bound gives the log prob-
ability of the model, and we see
that the value of the bound peaks
at M = 3, corresponding to the
true model from which the data
set was generated.
1
3
5
7
9
10.4. Exponential Family Distributions
In Chapter 2, we discussed the important role played by the exponential family of
distributions and their conjugate priors. For many of the models discussed in this
book, the complete-data likelihood is drawn from the exponential family. However,
in general this will not be the case for the marginal likelihood function for the ob-
served data. For example, in a mixture of Gaussians, the joint distribution of obser-
vations xn and corresponding hidden variables zn is a member of the exponential
family, whereas the marginal distribution of xn is a mixture of Gaussians and hence
is not.
Up to now we have grouped the variables in the model into observed variables
and hidden variables. We now make a further distinction between latent variables,
denoted Z, and parameters, denoted θ, where parameters are intensive (ﬁxed in num-
ber independent of the size of the data set), whereas latent variables are extensive
(scale in number with the size of the data set). For example, in a Gaussian mixture
model, the indicator variables zkn (which specify which component k is responsible
for generating data point xn) represent the latent variables, whereas the means µk,
precisions Λk and mixing proportions πk represent the parameters.
Consider the case of independent identically distributed data. We denote the
data values by X = {xn}, where n = 1, . . . N, with corresponding latent variables
Z = {zn}. Now suppose that the joint distribution of observed and latent variables
is a member of the exponential family, parameterized by natural parameters η so that
p(X, Z|η) =
N

n=1
h(xn, zn)g(η) exp

ηTu(xn, zn)

.
(10.113)
We shall also use a conjugate prior for η, which can be written as
p(η|ν0, v0) = f(ν0, χ0)g(η)ν0 exp 
νoηTχ0

.
(10.114)
Recall that the conjugate prior distribution can be interpreted as a prior number ν0
of observations all having the value χ0 for the u vector. Now consider a variational


---
**Page 506**
10.4. Exponential Family Distributions
491
distribution that factorizes between the latent variables and the parameters, so that
q(Z, η) = q(Z)q(η). Using the general result (10.9), we can solve for the two
factors as follows
ln q⋆(Z)
=
Eη[ln p(X, Z|η)] + const
=
N

n=1

ln h(xn, zn) + E[ηT]u(xn, zn)

+ const. (10.115)
Thus we see that this decomposes into a sum of independent terms, one for each
value of n, and hence the solution for q⋆(Z) will factorize over n so that q⋆(Z) =

n q⋆(zn). This is an example of an induced factorization. Taking the exponential
Section 10.2.5
of both sides, we have
q⋆(zn) = h(xn, zn)g (E[η]) exp

E[ηT]u(xn, zn)

(10.116)
where the normalization coefﬁcient has been re-instated by comparison with the
standard form for the exponential family.
Similarly, for the variational distribution over the parameters, we have
ln q⋆(η) = ln p(η|ν0, χ0) + EZ[ln p(X, Z|η)] + const
(10.117)
=
ν0 ln g(η) + ηTχ0 +
N

n=1

ln g(η) + ηTEzn[u(xn, zn)]
+ const. (10.118)
Again, taking the exponential of both sides, and re-instating the normalization coef-
ﬁcient by inspection, we have
q⋆(η) = f(νN, χN)g(η)νN exp 
ηTχN

(10.119)
where we have deﬁned
νN
=
ν0 + N
(10.120)
χN
=
χ0 +
N

n=1
Ezn[u(xn, zn)].
(10.121)
Note that the solutions for q⋆(zn) and q⋆(η) are coupled, and so we solve them iter-
atively in a two-stage procedure. In the variational E step, we evaluate the expected
sufﬁcient statistics E[u(xn, zn)] using the current posterior distribution q(zn) over
the latent variables and use this to compute a revised posterior distribution q(η) over
the parameters. Then in the subsequent variational M step, we use this revised pa-
rameter posterior distribution to ﬁnd the expected natural parameters E[ηT], which
gives rise to a revised variational distribution over the latent variables.
10.4.1
Variational message passing
We have illustrated the application of variational methods by considering a spe-
ciﬁc model, the Bayesian mixture of Gaussians, in some detail. This model can be


---
**Page 507**
492
10. APPROXIMATE INFERENCE
described by the directed graph shown in Figure 10.5. Here we consider more gen-
erally the use of variational methods for models described by directed graphs and
derive a number of widely applicable results.
The joint distribution corresponding to a directed graph can be written using the
decomposition
p(x) =

i
p(xi|pai)
(10.122)
where xi denotes the variable(s) associated with node i, and pai denotes the parent
set corresponding to node i. Note that xi may be a latent variable or it may belong
to the set of observed variables. Now consider a variational approximation in which
the distribution q(x) is assumed to factorize with respect to the xi so that
q(x) =

i
qi(xi).
(10.123)
Note that for observed nodes, there is no factor q(xi) in the variational distribution.
We now substitute (10.122) into our general result (10.9) to give
ln q⋆
j (xj) = Ei̸=j

i
ln p(xi|pai)
 
+ const.
(10.124)
Any terms on the right-hand side that do not depend on xj can be absorbed into
the additive constant. In fact, the only terms that do depend on xj are the con-
ditional distribution for xj given by p(xj|paj) together with any other conditional
distributions that have xj in the conditioning set. By deﬁnition, these conditional
distributions correspond to the children of node j, and they therefore also depend on
the co-parents of the child nodes, i.e., the other parents of the child nodes besides
node xj itself. We see that the set of all nodes on which q⋆(xj) depends corresponds
to the Markov blanket of node xj, as illustrated in Figure 8.26. Thus the update
of the factors in the variational posterior distribution represents a local calculation
on the graph. This makes possible the construction of general purpose software for
variational inference in which the form of the model does not need to be speciﬁed in
advance (Bishop et al., 2003).
If we now specialize to the case of a model in which all of the conditional dis-
tributions have a conjugate-exponential structure, then the variational update proce-
dure can be cast in terms of a local message passing algorithm (Winn and Bishop,
2005). In particular, the distribution associated with a particular node can be updated
once that node has received messages from all of its parents and all of its children.
This in turn requires that the children have already received messages from their co-
parents. The evaluation of the lower bound can also be simpliﬁed because many of
the required quantities are already evaluated as part of the message passing scheme.
This distributed message passing formulation has good scaling properties and is well
suited to large networks.


---
**Page 508**
10.5. Local Variational Methods
493
10.5. Local Variational Methods
The variational framework discussed in Sections 10.1 and 10.2 can be considered a
‘global’ method in the sense that it directly seeks an approximation to the full poste-
rior distribution over all random variables. An alternative ‘local’ approach involves
ﬁnding bounds on functions over individual variables or groups of variables within
a model. For instance, we might seek a bound on a conditional distribution p(y|x),
which is itself just one factor in a much larger probabilistic model speciﬁed by a
directed graph. The purpose of introducing the bound of course is to simplify the
resulting distribution. This local approximation can be applied to multiple variables
in turn until a tractable approximation is obtained, and in Section 10.6.1 we shall
give a practical example of this approach in the context of logistic regression. Here
we focus on developing the bounds themselves.
We have already seen in our discussion of the Kullback-Leibler divergence that
the convexity of the logarithm function played a key role in developing the lower
bound in the global variational approach. We have deﬁned a (strictly) convex func-
tion as one for which every chord lies above the function. Convexity also plays a
Section 1.6.1
central role in the local variational framework. Note that our discussion will ap-
ply equally to concave functions with ‘min’ and ‘max’ interchanged and with lower
bounds replaced by upper bounds.
Let us begin by considering a simple example, namely the function f(x) =
exp(−x), which is a convex function of x, and which is shown in the left-hand plot
of Figure 10.10. Our goal is to approximate f(x) by a simpler function, in particular
a linear function of x. From Figure 10.10, we see that this linear function will be a
lower bound on f(x) if it corresponds to a tangent. We can obtain the tangent line
y(x) at a speciﬁc value of x, say x = ξ, by making a ﬁrst order Taylor expansion
y(x) = f(ξ) + f ′(ξ)(x −ξ)
(10.125)
so that y(x) ⩽f(x) with equality when x = ξ. For our example function f(x) =
Figure 10.10
In the left-hand ﬁg-
ure the red curve shows the function
exp(−x), and the blue line shows
the tangent at x = ξ deﬁned by
(10.125) with ξ = 1. This line has
slope λ = f ′(ξ) = −exp(−ξ). Note
that any other tangent line, for ex-
ample the ones shown in green, will
have a smaller value of y at x =
ξ. The right-hand ﬁgure shows the
corresponding plot of the function
λξ −g(λ), where g(λ) is given by
(10.131), versus λ for ξ
= 1, in
which the maximum corresponds to
λ = −exp(−ξ) = −1/e.
x
ξ
0
1.5
3
0
0.5
1
λ
λξ −g(λ)
−1
−0.5
0
0
0.2
0.4


---
**Page 509**
494
10. APPROXIMATE INFERENCE
x
y
f(x)
λx
x
y
f(x)
λx −g(λ)
−g(λ)
Figure 10.11
In the left-hand plot the red curve shows a convex function f(x), and the blue line represents the
linear function λx, which is a lower bound on f(x) because f(x) > λx for all x. For the given value of slope λ the
contact point of the tangent line having the same slope is found by minimizing with respect to x the discrepancy
(shown by the green dashed lines) given by f(x) −λx. This deﬁnes the dual function g(λ), which corresponds
to the (negative of the) intercept of the tangent line having slope λ.
exp(−x), we therefore obtain the tangent line in the form
y(x) = exp(−ξ) −exp(−ξ)(x −ξ)
(10.126)
which is a linear function parameterized by ξ. For consistency with subsequent
discussion, let us deﬁne λ = −exp(−ξ) so that
y(x, λ) = λx −λ + λ ln(−λ).
(10.127)
Different values of λ correspond to different tangent lines, and because all such lines
are lower bounds on the function, we have f(x) ⩾y(x, λ). Thus we can write the
function in the form
f(x) = max
λ
{λx −λ + λ ln(−λ)} .
(10.128)
We have succeeded in approximating the convex function f(x) by a simpler, lin-
ear function y(x, λ). The price we have paid is that we have introduced a variational
parameter λ, and to obtain the tightest bound we must optimize with respect to λ.
We can formulate this approach more generally using the framework of convex
duality (Rockafellar, 1972; Jordan et al., 1999). Consider the illustration of a convex
function f(x) shown in the left-hand plot in Figure 10.11. In this example, the
function λx is a lower bound on f(x) but it is not the best lower bound that can
be achieved by a linear function having slope λ, because the tightest bound is given
by the tangent line. Let us write the equation of the tangent line, having slope λ as
λx −g(λ) where the (negative) intercept g(λ) clearly depends on the slope λ of the
tangent. To determine the intercept, we note that the line must be moved vertically by
an amount equal to the smallest vertical distance between the line and the function,
as shown in Figure 10.11. Thus
g(λ)
=
−min
x {f(x) −λx}
=
max
x
{λx −f(x)} .
(10.129)


---
**Page 510**
10.5. Local Variational Methods
495
Now, instead of ﬁxing λ and varying x, we can consider a particular x and then
adjust λ until the tangent plane is tangent at that particular x. Because the y value
of the tangent line at a particular x is maximized when that value coincides with its
contact point, we have
f(x) = max
λ
{λx −g(λ)} .
(10.130)
We see that the functions f(x) and g(λ) play a dual role, and are related through
(10.129) and (10.130).
Let us apply these duality relations to our simple example f(x) = exp(−x).
From (10.129) we see that the maximizing value of x is given by ξ = −ln(−λ), and
back-substituting we obtain the conjugate function g(λ) in the form
g(λ) = λ −λ ln(−λ)
(10.131)
as obtained previously. The function λξ −g(λ) is shown, for ξ = 1 in the right-hand
plot in Figure 10.10. As a check, we can substitute (10.131) into (10.130), which
gives the maximizing value of λ = −exp(−x), and back-substituting then recovers
the original function f(x) = exp(−x).
For concave functions, we can follow a similar argument to obtain upper bounds,
in which max’ is replaced with ‘min’, so that
f(x)
=
min
λ {λx −g(λ)}
(10.132)
g(λ)
=
min
x {λx −f(x)} .
(10.133)
If the function of interest is not convex (or concave), then we cannot directly
apply the method above to obtain a bound. However, we can ﬁrst seek invertible
transformations either of the function or of its argument which change it into a con-
vex form. We then calculate the conjugate function and then transform back to the
original variables.
An important example, which arises frequently in pattern recognition, is the
logistic sigmoid function deﬁned by
σ(x) =
1
1 + e−x .
(10.134)
As it stands this function is neither convex nor concave. However, if we take the
logarithm we obtain a function which is concave, as is easily veriﬁed by ﬁnding the
second derivative. From (10.133) the corresponding conjugate function then takes
Exercise 10.30
the form
g(λ) = min
x {λx −f(x)} = −λ ln λ −(1 −λ) ln(1 −λ)
(10.135)
which we recognize as the binary entropy function for a variable whose probability
of having the value 1 is λ. Using (10.132), we then obtain an upper bound on the log
Appendix B
sigmoid
ln σ(x) ⩽λx −g(λ)
(10.136)


---
**Page 511**
496
10. APPROXIMATE INFERENCE
λ = 0.2
λ = 0.7
−6
0
6
0
0.5
1
ξ = 2.5
−ξ
ξ
−6
0
6
0
0.5
1
Figure 10.12
The left-hand plot shows the logistic sigmoid function σ(x) deﬁned by (10.134) in red, together
with two examples of the exponential upper bound (10.137) shown in blue. The right-hand plot shows the logistic
sigmoid again in red together with the Gaussian lower bound (10.144) shown in blue.
Here the parameter
ξ = 2.5, and the bound is exact at x = ξ and x = −ξ, denoted by the dashed green lines.
and taking the exponential, we obtain an upper bound on the logistic sigmoid itself
of the form
σ(x) ⩽exp(λx −g(λ))
(10.137)
which is plotted for two values of λ on the left-hand plot in Figure 10.12.
We can also obtain a lower bound on the sigmoid having the functional form of
a Gaussian. To do this, we follow Jaakkola and Jordan (2000) and make transforma-
tions both of the input variable and of the function itself. First we take the log of the
logistic function and then decompose it so that
ln σ(x)
=
−ln(1 + e−x) = −ln 
e−x/2(ex/2 + e−x/2)
=
x/2 −ln(ex/2 + e−x/2).
(10.138)
We now note that the function f(x) = −ln(ex/2 + e−x/2) is a convex function of
the variable x2, as can again be veriﬁed by ﬁnding the second derivative. This leads
Exercise 10.31
to a lower bound on f(x), which is a linear function of x2 whose conjugate function
is given by
g(λ) = max
x2

λx2 −f
√
x2

.
(10.139)
The stationarity condition leads to
0 = λ −dx
dx2
d
dxf(x) = λ + 1
4x tanh
x
2

.
(10.140)
If we denote this value of x, corresponding to the contact point of the tangent line
for this particular value of λ, by ξ, then we have
λ(ξ) = −1
4ξ tanh
ξ
2

= −1
2ξ

σ(ξ) −1
2

.
(10.141)


---
**Page 512**
10.5. Local Variational Methods
497
Instead of thinking of λ as the variational parameter, we can let ξ play this role as
this leads to simpler expressions for the conjugate function, which is then given by
g(λ) = λ(ξ)ξ2 −f(ξ) = λ(ξ)ξ2 + ln(eξ/2 + e−ξ/2).
(10.142)
Hence the bound on f(x) can be written as
f(x) ⩾λx2 −g(λ) = λx2 −λξ2 −ln(eξ/2 + e−ξ/2).
(10.143)
The bound on the sigmoid then becomes
σ(x) ⩾σ(ξ) exp

(x −ξ)/2 −λ(ξ)(x2 −ξ2)

(10.144)
where λ(ξ) is deﬁned by (10.141). This bound is illustrated in the right-hand plot of
Figure 10.12. We see that the bound has the form of the exponential of a quadratic
function of x, which will prove useful when we seek Gaussian representations of
posterior distributions deﬁned through logistic sigmoid functions.
Section 4.5
The logistic sigmoid arises frequently in probabilistic models over binary vari-
ables because it is the function that transforms a log odds ratio into a posterior prob-
ability. The corresponding transformation for a multiclass distribution is given by
the softmax function. Unfortunately, the lower bound derived here for the logistic
Section 4.3
sigmoid does not directly extend to the softmax. Gibbs (1997) proposes a method
for constructing a Gaussian distribution that is conjectured to be a bound (although
no rigorous proof is given), which may be used to apply local variational methods to
multiclass problems.
We shall see an example of the use of local variational bounds in Sections 10.6.1.
For the moment, however, it is instructive to consider in general terms how these
bounds can be used. Suppose we wish to evaluate an integral of the form
I =

σ(a)p(a) da
(10.145)
where σ(a) is the logistic sigmoid, and p(a) is a Gaussian probability density. Such
integrals arise in Bayesian models when, for instance, we wish to evaluate the pre-
dictive distribution, in which case p(a) represents a posterior parameter distribution.
Because the integral is intractable, we employ the variational bound (10.144), which
we write in the form σ(a) ⩾f(a, ξ) where ξ is a variational parameter. The inte-
gral now becomes the product of two exponential-quadratic functions and so can be
integrated analytically to give a bound on I
I ⩾

f(a, ξ)p(a) da = F(ξ).
(10.146)
We now have the freedom to choose the variational parameter ξ, which we do by
ﬁnding the value ξ⋆that maximizes the function F(ξ). The resulting value F(ξ⋆)
represents the tightest bound within this family of bounds and can be used as an
approximation to I. This optimized bound, however, will in general not be exact.


---
**Page 513**
498
10. APPROXIMATE INFERENCE
Although the bound σ(a) ⩾f(a, ξ) on the logistic sigmoid can be optimized exactly,
the required choice for ξ depends on the value of a, so that the bound is exact for one
value of a only. Because the quantity F(ξ) is obtained by integrating over all values
of a, the value of ξ⋆represents a compromise, weighted by the distribution p(a).
10.6. Variational Logistic Regression
We now illustrate the use of local variational methods by returning to the Bayesian
logistic regression model studied in Section 4.5. There we focussed on the use of
the Laplace approximation, while here we consider a variational treatment based on
the approach of Jaakkola and Jordan (2000). Like the Laplace method, this also
leads to a Gaussian approximation to the posterior distribution. However, the greater
ﬂexibility of the variational approximation leads to improved accuracy compared
to the Laplace method. Furthermore (unlike the Laplace method), the variational
approach is optimizing a well deﬁned objective function given by a rigourous bound
on the model evidence. Logistic regression has also been treated by Dybowski and
Roberts (2005) from a Bayesian perspective using Monte Carlo sampling techniques.
10.6.1
Variational posterior distribution
Here we shall make use of a variational approximation based on the local bounds
introduced in Section 10.5. This allows the likelihood function for logistic regres-
sion, which is governed by the logistic sigmoid, to be approximated by the expo-
nential of a quadratic form. It is therefore again convenient to choose a conjugate
Gaussian prior of the form (4.140). For the moment, we shall treat the hyperparam-
eters m0 and S0 as ﬁxed constants. In Section 10.6.3, we shall demonstrate how the
variational formalism can be extended to the case where there are unknown hyper-
parameters whose values are to be inferred from the data.
In the variational framework, we seek to maximize a lower bound on the marginal
likelihood. For the Bayesian logistic regression model, the marginal likelihood takes
the form
p(t) =

p(t|w)p(w) dw =
  N

n=1
p(tn|w)
 
p(w) dw.
(10.147)
We ﬁrst note that the conditional distribution for t can be written as
p(t|w)
=
σ(a)t {1 −σ(a)}1−t
=

1
1 + e−a
t 
1 −
1
1 + e−a
1−t
=
eat
e−a
1 + e−a = eatσ(−a)
(10.148)
where a = wTφ. In order to obtain a lower bound on p(t), we make use of the
variational lower bound on the logistic sigmoid function given by (10.144), which


---
**Page 514**
10.6. Variational Logistic Regression
499
we reproduce here for convenience
σ(z) ⩾σ(ξ) exp 
(z −ξ)/2 −λ(ξ)(z2 −ξ2)
(10.149)
where
λ(ξ) = 1
2ξ

σ(ξ) −1
2

.
(10.150)
We can therefore write
p(t|w) = eatσ(−a) ⩾eatσ(ξ) exp 
−(a + ξ)/2 −λ(ξ)(a2 −ξ2)
.
(10.151)
Note that because this bound is applied to each of the terms in the likelihood function
separately, there is a variational parameter ξn corresponding to each training set
observation (φn, tn). Using a = wTφ, and multiplying by the prior distribution, we
obtain the following bound on the joint distribution of t and w
p(t, w) = p(t|w)p(w) ⩾h(w, ξ)p(w)
(10.152)
where ξ denotes the set {ξn} of variational parameters, and
h(w, ξ)
=
N

n=1
σ(ξn) exp 
wTφntn −(wTφn + ξn)/2
−λ(ξn)([wTφn]2 −ξ2
n)

.
(10.153)
Evaluation of the exact posterior distribution would require normalization of the left-
hand side of this inequality. Because this is intractable, we work instead with the
right-hand side. Note that the function on the right-hand side cannot be interpreted
as a probability density because it is not normalized. Once it is normalized to give a
variational posterior distribution q(w), however, it no longer represents a bound.
Because the logarithm function is monotonically increasing, the inequality A ⩾
B implies ln A ⩾ln B. This gives a lower bound on the log of the joint distribution
of t and w of the form
ln {p(t|w)p(w)} ⩾ln p(w) +
N

n=1

ln σ(ξn) + wTφntn
−(wTφn + ξn)/2 −λ(ξn)([wTφn]2 −ξ2
n)

.
(10.154)
Substituting for the prior p(w), the right-hand side of this inequality becomes, as a
function of w
−1
2(w −m0)TS−1
0 (w −m0)
+
N

n=1

wTφn(tn −1/2) −λ(ξn)wT(φnφT
n)w
+ const.
(10.155)


---
**Page 515**
500
10. APPROXIMATE INFERENCE
This is a quadratic function of w, and so we can obtain the corresponding variational
approximation to the posterior distribution by identifying the linear and quadratic
terms in w, giving a Gaussian variational posterior of the form
q(w) = N(w|mN, SN)
(10.156)
where
mN
=
SN

S−1
0 m0 +
N

n=1
(tn −1/2)φn

(10.157)
S−1
N
=
S−1
0
+ 2
N

n=1
λ(ξn)φnφT
n.
(10.158)
As with the Laplace framework, we have again obtained a Gaussian approximation
to the posterior distribution. However, the additional ﬂexibility provided by the vari-
ational parameters {ξn} leads to improved accuracy in the approximation (Jaakkola
and Jordan, 2000).
Here we have considered a batch learning context in which all of the training
data is available at once. However, Bayesian methods are intrinsically well suited
to sequential learning in which the data points are processed one at a time and then
discarded. The formulation of this variational approach for the sequential case is
straightforward.
Exercise 10.32
Note that the bound given by (10.149) applies only to the two-class problem and
so this approach does not directly generalize to classiﬁcation problems with K > 2
classes. An alternative bound for the multiclass case has been explored by Gibbs
(1997).
10.6.2
Optimizing the variational parameters
We now have a normalized Gaussian approximation to the posterior distribution,
which we shall use shortly to evaluate the predictive distribution for new data points.
First, however, we need to determine the variational parameters {ξn} by maximizing
the lower bound on the marginal likelihood.
To do this, we substitute the inequality (10.152) back into the marginal likeli-
hood to give
ln p(t) = ln

p(t|w)p(w) dw ⩾ln

h(w, ξ)p(w) dw = L(ξ).
(10.159)
As with the optimization of the hyperparameter α in the linear regression model of
Section 3.5, there are two approaches to determining the ξn. In the ﬁrst approach, we
recognize that the function L(ξ) is deﬁned by an integration over w and so we can
view w as a latent variable and invoke the EM algorithm. In the second approach,
we integrate over w analytically and then perform a direct maximization over ξ. Let
us begin by considering the EM approach.
The EM algorithm starts by choosing some initial values for the parameters
{ξn}, which we denote collectively by ξold. In the E step of the EM algorithm,


---
**Page 516**
10.6. Variational Logistic Regression
501
we then use these parameter values to ﬁnd the posterior distribution over w, which
is given by (10.156). In the M step, we then maximize the expected complete-data
log likelihood which is given by
Q(ξ, ξold) = E [ln h(w, ξ)p(w)]
(10.160)
where the expectation is taken with respect to the posterior distribution q(w) evalu-
ated using ξold. Noting that p(w) does not depend on ξ, and substituting for h(w, ξ)
we obtain
Q(ξ, ξold) =
N

n=1

ln σ(ξn) −ξn/2 −λ(ξn)(φT
nE[wwT]φn −ξ2
n)
+ const
(10.161)
where ‘const’ denotes terms that are independent of ξ. We now set the derivative with
respect to ξn equal to zero. A few lines of algebra, making use of the deﬁnitions of
σ(ξ) and λ(ξ), then gives
0 = λ′(ξn)(φT
nE[wwT]φn −ξ2
n).
(10.162)
We now note that λ′(ξ) is a monotonic function of ξ for ξ ⩾0, and that we can
restrict attention to nonnegative values of ξ without loss of generality due to the
symmetry of the bound around ξ = 0. Thus λ′(ξ) ̸= 0, and hence we obtain the
following re-estimation equations
Exercise 10.33
(ξnew
n
)2 = φT
nE[wwT]φn = φT
n

SN + mNmT
N

φn
(10.163)
where we have used (10.156).
Let us summarize the EM algorithm for ﬁnding the variational posterior distri-
bution. We ﬁrst initialize the variational parameters ξold. In the E step, we evaluate
the posterior distribution over w given by (10.156), in which the mean and covari-
ance are deﬁned by (10.157) and (10.158). In the M step, we then use this variational
posterior to compute a new value for ξ given by (10.163). The E and M steps are
repeated until a suitable convergence criterion is satisﬁed, which in practice typically
requires only a few iterations.
An alternative approach to obtaining re-estimation equations for ξ is to note
that in the integral over w in the deﬁnition (10.159) of the lower bound L(ξ), the
integrand has a Gaussian-like form and so the integral can be evaluated analytically.
Having evaluated the integral, we can then differentiate with respect to ξn. It turns
out that this gives rise to exactly the same re-estimation equations as does the EM
approach given by (10.163).
Exercise 10.34
As we have emphasized already, in the application of variational methods it is
useful to be able to evaluate the lower bound L(ξ) given by (10.159). The integration
over w can be performed analytically by noting that p(w) is Gaussian and h(w, ξ)
is the exponential of a quadratic function of w. Thus, by completing the square
and making use of the standard result for the normalization coefﬁcient of a Gaussian
distribution, we can obtain a closed form solution which takes the form
Exercise 10.35


---
**Page 517**
502
10. APPROXIMATE INFERENCE
0.01
0.25
0.75
0.99
−4
−2
0
2
4
−6
−4
−2
0
2
4
6
−4
−2
0
2
4
−6
−4
−2
0
2
4
6
Figure 10.13
Illustration of the Bayesian approach to logistic regression for a simple linearly separable data
set. The plot on the left shows the predictive distribution obtained using variational inference. We see that
the decision boundary lies roughly mid way between the clusters of data points, and that the contours of the
predictive distribution splay out away from the data reﬂecting the greater uncertainty in the classiﬁcation of such
regions. The plot on the right shows the decision boundaries corresponding to ﬁve samples of the parameter
vector w drawn from the posterior distribution p(w|t).
L(ξ)
=
1
2 ln |SN|
|S0| −1
2mT
NS−1
N mN + 1
2mT
0 S−1
0 m0
+
N

n=1

ln σ(ξn) −1
2ξn −λ(ξn)ξ2
n

.
(10.164)
This variational framework can also be applied to situations in which the data
is arriving sequentially (Jaakkola and Jordan, 2000). In this case we maintain a
Gaussian posterior distribution over w, which is initialized using the prior p(w). As
each data point arrives, the posterior is updated by making use of the bound (10.151)
and then normalized to give an updated posterior distribution.
The predictive distribution is obtained by marginalizing over the posterior dis-
tribution, and takes the same form as for the Laplace approximation discussed in
Section 4.5.2. Figure 10.13 shows the variational predictive distributions for a syn-
thetic data set. This example provides interesting insights into the concept of ‘large
margin’, which was discussed in Section 7.1 and which has qualitatively similar be-
haviour to the Bayesian solution.
10.6.3
Inference of hyperparameters
So far, we have treated the hyperparameter α in the prior distribution as a known
constant. We now extend the Bayesian logistic regression model to allow the value of
this parameter to be inferred from the data set. This can be achieved by combining
the global and local variational approximations into a single framework, so as to
maintain a lower bound on the marginal likelihood at each stage. Such a combined
approach was adopted by Bishop and Svens´en (2003) in the context of a Bayesian
treatment of the hierarchical mixture of experts model.


---
**Page 518**
10.6. Variational Logistic Regression
503
Speciﬁcally, we consider once again a simple isotropic Gaussian prior distribu-
tion of the form
p(w|α) = N(w|0, α−1I).
(10.165)
Our analysis is readily extended to more general Gaussian priors, for instance if we
wish to associate a different hyperparameter with different subsets of the parame-
ters wj. As usual, we consider a conjugate hyperprior over α given by a gamma
distribution
p(α) = Gam(α|a0, b0)
(10.166)
governed by the constants a0 and b0.
The marginal likelihood for this model now takes the form
p(t) =

p(w, α, t) dw dα
(10.167)
where the joint distribution is given by
p(w, α, t) = p(t|w)p(w|α)p(α).
(10.168)
We are now faced with an analytically intractable integration over w and α, which
we shall tackle by using both the local and global variational approaches in the same
model
To begin with, we introduce a variational distribution q(w, α), and then apply
the decomposition (10.2), which in this instance takes the form
ln p(t) = L(q) + KL(q∥p)
(10.169)
where the lower bound L(q) and the Kullback-Leibler divergence KL(q∥p) are de-
ﬁned by
L(q)
=

q(w, α) ln
p(w, α, t)
q(w, α)

dw dα
(10.170)
KL(q∥p)
=
−

q(w, α) ln
p(w, α|t))
q(w, α)

dw dα.
(10.171)
At this point, the lower bound L(q) is still intractable due to the form of the
likelihood factor p(t|w). We therefore apply the local variational bound to each of
the logistic sigmoid factors as before. This allows us to use the inequality (10.152)
and place a lower bound on L(q), which will therefore also be a lower bound on the
log marginal likelihood
ln p(t)
⩾
L(q) ⩾L(q, ξ)
=

q(w, α) ln
h(w, ξ)p(w|α)p(α)
q(w, α)

dw dα.
(10.172)
Next we assume that the variational distribution factorizes between parameters and
hyperparameters so that
q(w, α) = q(w)q(α).
(10.173)


---
**Page 519**
504
10. APPROXIMATE INFERENCE
With this factorization we can appeal to the general result (10.9) to ﬁnd expressions
for the optimal factors. Consider ﬁrst the distribution q(w). Discarding terms that
are independent of w, we have
ln q(w)
=
Eα [ln {h(w, ξ)p(w|α)p(α)}] + const
=
ln h(w, ξ) + Eα [ln p(w|α)] + const.
We now substitute for ln h(w, ξ) using (10.153), and for ln p(w|α) using (10.165),
giving
ln q(w) = −E[α]
2 wTw +
N

n=1

(tn −1/2)wTφn −λ(ξn)wTφnφT
nw
+ const.
We see that this is a quadratic function of w and so the solution for q(w) will be
Gaussian. Completing the square in the usual way, we obtain
q(w) = N(w|µN, ΣN)
(10.174)
where we have deﬁned
Σ−1
N µN
=
N

n=1
(tn −1/2)φn
(10.175)
Σ−1
N
=
E[α]I + 2
N

n=1
λ(ξn)φnφT
n.
(10.176)
Similarly, the optimal solution for the factor q(α) is obtained from
ln q(α) = Ew [ln p(w|α)] + ln p(α) + const.
Substituting for ln p(w|α) using (10.165), and for ln p(α) using (10.166), we obtain
ln q(α) = M
2 ln α −α
2 E

wTw
	
+ (a0 −1) ln α −b0α + const.
We recognize this as the log of a gamma distribution, and so we obtain
q(α) = Gam(α|aN, bN) =
1
Γ(a0)ab0
0 αa0−1e−b0α
(10.177)
where
aN
=
a0 + M
2
(10.178)
bN
=
b0 + 1
2Ew

wTw
	
.
(10.179)


---
**Page 520**
10.7. Expectation Propagation
505
We also need to optimize the variational parameters ξn, and this is also done by
maximizing the lower bound L(q, ξ). Omitting terms that are independent of ξ, and
integrating over α, we have
L(q, ξ) =

q(w) ln h(w, ξ) dw + const.
(10.180)
Note that this has precisely the same form as (10.159), and so we can again appeal
to our earlier result (10.163), which can be obtained by direct optimization of the
marginal likelihood function, leading to re-estimation equations of the form
(ξnew
n
)2 = φT
n

ΣN + µNµT
N

φn.
(10.181)
We have obtained re-estimation equations for the three quantities q(w), q(α),
and ξ, and so after making suitable initializations, we can cycle through these quan-
tities, updating each in turn. The required moments are given by
Appendix B
E [α]
=
aN
bN
(10.182)
E 
wTw	
=
ΣN + µT
NµN.
(10.183)
10.7. Expectation Propagation
We conclude this chapter by discussing an alternative form of deterministic approx-
imate inference, known as expectation propagation or EP (Minka, 2001a; Minka,
2001b). As with the variational Bayes methods discussed so far, this too is based
on the minimization of a Kullback-Leibler divergence but now of the reverse form,
which gives the approximation rather different properties.
Consider for a moment the problem of minimizing KL(p∥q) with respect to q(z)
when p(z) is a ﬁxed distribution and q(z) is a member of the exponential family and
so, from (2.194), can be written in the form
q(z) = h(z)g(η) exp 
ηTu(z)
.
(10.184)
As a function of η, the Kullback-Leibler divergence then becomes
KL(p∥q) = −ln g(η) −ηTEp(z)[u(z)] + const
(10.185)
where the constant terms are independent of the natural parameters η. We can mini-
mize KL(p∥q) within this family of distributions by setting the gradient with respect
to η to zero, giving
−∇ln g(η) = Ep(z)[u(z)].
(10.186)
However, we have already seen in (2.226) that the negative gradient of ln g(η) is
given by the expectation of u(z) under the distribution q(z). Equating these two
results, we obtain
Eq(z)[u(z)] = Ep(z)[u(z)].
(10.187)


---
**Page 521**
506
10. APPROXIMATE INFERENCE
We see that the optimum solution simply corresponds to matching the expected suf-
ﬁcient statistics. So, for instance, if q(z) is a Gaussian N(z|µ, Σ) then we minimize
the Kullback-Leibler divergence by setting the mean µ of q(z) equal to the mean of
the distribution p(z) and the covariance Σ equal to the covariance of p(z). This is
sometimes called moment matching. An example of this was seen in Figure 10.3(a).
Now let us exploit this result to obtain a practical algorithm for approximate
inference. For many probabilistic models, the joint distribution of data D and hidden
variables (including parameters) θ comprises a product of factors in the form
p(D, θ) =

i
fi(θ).
(10.188)
This would arise, for example, in a model for independent, identically distributed
data in which there is one factor fn(θ) = p(xn|θ) for each data point xn, along
with a factor f0(θ) = p(θ) corresponding to the prior. More generally, it would also
apply to any model deﬁned by a directed probabilistic graph in which each factor is a
conditional distribution corresponding to one of the nodes, or an undirected graph in
which each factor is a clique potential. We are interested in evaluating the posterior
distribution p(θ|D) for the purpose of making predictions, as well as the model
evidence p(D) for the purpose of model comparison. From (10.188) the posterior is
given by
p(θ|D) =
1
p(D)

i
fi(θ)
(10.189)
and the model evidence is given by
p(D) =
 
i
fi(θ) dθ.
(10.190)
Here we are considering continuous variables, but the following discussion applies
equally to discrete variables with integrals replaced by summations. We shall sup-
pose that the marginalization over θ, along with the marginalizations with respect to
the posterior distribution required to make predictions, are intractable so that some
form of approximation is required.
Expectation propagation is based on an approximation to the posterior distribu-
tion which is also given by a product of factors
q(θ) = 1
Z

i
fi(θ)
(10.191)
in which each factor fi(θ) in the approximation corresponds to one of the factors
fi(θ) in the true posterior (10.189), and the factor 1/Z is the normalizing constant
needed to ensure that the left-hand side of (10.191) integrates to unity. In order to
obtain a practical algorithm, we need to constrain the factors fi(θ) in some way,
and in particular we shall assume that they come from the exponential family. The
product of the factors will therefore also be from the exponential family and so can


---
**Page 522**
10.7. Expectation Propagation
507
be described by a ﬁnite set of sufﬁcient statistics. For example, if each of the fi(θ)
is a Gaussian, then the overall approximation q(θ) will also be Gaussian.
Ideally we would like to determine the fi(θ) by minimizing the Kullback-Leibler
divergence between the true posterior and the approximation given by
KL (p∥q) = KL

1
p(D)

i
fi(θ)
'''''
1
Z

i
fi(θ)

.
(10.192)
Note that this is the reverse form of KL divergence compared with that used in varia-
tional inference. In general, this minimization will be intractable because the KL di-
vergence involves averaging with respect to the true distribution. As a rough approx-
imation, we could instead minimize the KL divergences between the corresponding
pairs fi(θ) and fi(θ) of factors. This represents a much simpler problem to solve,
and has the advantage that the algorithm is noniterative. However, because each fac-
tor is individually approximated, the product of the factors could well give a poor
approximation.
Expectation propagation makes a much better approximation by optimizing each
factor in turn in the context of all of the remaining factors. It starts by initializing
the factors fi(θ), and then cycles through the factors reﬁning them one at a time.
This is similar in spirit to the update of factors in the variational Bayes framework
considered earlier. Suppose we wish to reﬁne factor fj(θ). We ﬁrst remove this
factor from the product to give 
i̸=j fi(θ). Conceptually, we will now determine a
revised form of the factor fj(θ) by ensuring that the product
qnew(θ) ∝fj(θ)

i̸=j
fi(θ)
(10.193)
is as close as possible to
fj(θ)

i̸=j
fi(θ)
(10.194)
in which we keep ﬁxed all of the factors fi(θ) for i ̸= j. This ensures that the
approximation is most accurate in the regions of high posterior probability as deﬁned
by the remaining factors. We shall see an example of this effect when we apply EP
to the ‘clutter problem’. To achieve this, we ﬁrst remove the factor fj(θ) from the
Section 10.7.1
current approximation to the posterior by deﬁning the unnormalized distribution
q\j(θ) = q(θ)
fj(θ)
.
(10.195)
Note that we could instead ﬁnd q\j(θ) from the product of factors i ̸= j, although
in practice division is usually easier. This is now combined with the factor fj(θ) to
give a distribution
1
Zj
fj(θ)q\j(θ)
(10.196)

