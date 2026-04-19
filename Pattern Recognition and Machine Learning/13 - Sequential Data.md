# 13 - Sequential Data
*Pages 605-652 from Pattern Recognition and Machine Learning*

---
**Page 605**
636
13. SEQUENTIAL DATA
make a greater contribution than less recent ones.
Although this sort of intuitive argument seems plausible, it does not tell us how
to form a weighted average, and any sort of hand-crafted weighing is hardly likely
to be optimal. Fortunately, we can address problems such as this much more sys-
tematically by deﬁning a probabilistic model that captures the time evolution and
measurement processes and then applying the inference and learning methods devel-
oped in earlier chapters. Here we shall focus on a widely used model known as a
linear dynamical system.
As we have seen, the HMM corresponds to the state space model shown in
Figure 13.5 in which the latent variables are discrete but with arbitrary emission
probability distributions. This graph of course describes a much broader class of
probability distributions, all of which factorize according to (13.6). We now consider
extensions to other distributions for the latent variables. In particular, we consider
continuous latent variables in which the summations of the sum-product algorithm
become integrals. The general form of the inference algorithms will, however, be
the same as for the hidden Markov model. It is interesting to note that, historically,
hidden Markov models and linear dynamical systems were developed independently.
Once they are both expressed as graphical models, however, the deep relationship
between them immediately becomes apparent.
One key requirement is that we retain an efﬁcient algorithm for inference which
is linear in the length of the chain. This requires that, for instance, when we take
a quantity α(zn−1), representing the posterior probability of zn given observations
x1, . . . , xn, and multiply by the transition probability p(zn|zn−1) and the emission
probability p(xn|zn) and then marginalize over zn−1, we obtain a distribution over
zn that is of the same functional form as that over α(zn−1). That is to say, the
distribution must not become more complex at each stage, but must only change in
its parameter values. Not surprisingly, the only distributions that have this property
of being closed under multiplication are those belonging to the exponential family.
Here we consider the most important example from a practical perspective,
which is the Gaussian. In particular, we consider a linear-Gaussian state space model
so that the latent variables {zn}, as well as the observed variables {xn}, are multi-
variate Gaussian distributions whose means are linear functions of the states of their
parents in the graph. We have seen that a directed graph of linear-Gaussian units
is equivalent to a joint Gaussian distribution over all of the variables. Furthermore,
marginals such as α(zn) are also Gaussian, so that the functional form of the mes-
sages is preserved and we will obtain an efﬁcient inference algorithm. By contrast,
suppose that the emission densities p(xn|zn) comprise a mixture of K Gaussians
each of which has a mean that is linear in zn. Then even if α(z1) is Gaussian, the
quantity α(z2) will be a mixture of K Gaussians, α(z3) will be a mixture of K2
Gaussians, and so on, and exact inference will not be of practical value.
We have seen that the hidden Markov model can be viewed as an extension of
the mixture models of Chapter 9 to allow for sequential correlations in the data.
In a similar way, we can view the linear dynamical system as a generalization of the
continuous latent variable models of Chapter 12 such as probabilistic PCA and factor
analysis. Each pair of nodes {zn, xn} represents a linear-Gaussian latent variable


---
**Page 606**
13.3. Linear Dynamical Systems
637
model for that particular observation. However, the latent variables {zn} are no
longer treated as independent but now form a Markov chain.
Because the model is represented by a tree-structured directed graph, inference
problems can be solved efﬁciently using the sum-product algorithm. The forward re-
cursions, analogous to the α messages of the hidden Markov model, are known as the
Kalman ﬁlter equations (Kalman, 1960; Zarchan and Musoff, 2005), and the back-
ward recursions, analogous to the β messages, are known as the Kalman smoother
equations, or the Rauch-Tung-Striebel (RTS) equations (Rauch et al., 1965). The
Kalman ﬁlter is widely used in many real-time tracking applications.
Because the linear dynamical system is a linear-Gaussian model, the joint distri-
bution over all variables, as well as all marginals and conditionals, will be Gaussian.
It follows that the sequence of individually most probable latent variable values is
the same as the most probable latent sequence. There is thus no need to consider the
Exercise 13.19
analogue of the Viterbi algorithm for the linear dynamical system.
Because the model has linear-Gaussian conditional distributions, we can write
the transition and emission distributions in the general form
p(zn|zn−1)
=
N(zn|Azn−1, Γ)
(13.75)
p(xn|zn)
=
N(xn|Czn, Σ).
(13.76)
The initial latent variable also has a Gaussian distribution which we write as
p(z1) = N(z1|µ0, V0).
(13.77)
Note that in order to simplify the notation, we have omitted additive constant terms
from the means of the Gaussians. In fact, it is straightforward to include them if
desired. Traditionally, these distributions are more commonly expressed in an equiv-
Exercise 13.24
alent form in terms of noisy linear equations given by
zn
=
Azn−1 + wn
(13.78)
xn
=
Czn + vn
(13.79)
z1
=
µ0 + u
(13.80)
where the noise terms have the distributions
w
∼
N(w|0, Γ)
(13.81)
v
∼
N(v|0, Σ)
(13.82)
u
∼
N(u|0, V0).
(13.83)
The parameters of the model, denoted by θ = {A, Γ, C, Σ, µ0, V0}, can be
determined using maximum likelihood through the EM algorithm. In the E step, we
need to solve the inference problem of determining the local posterior marginals for
the latent variables, which can be solved efﬁciently using the sum-product algorithm,
as we discuss in the next section.


---
**Page 607**
638
13. SEQUENTIAL DATA
13.3.1
Inference in LDS
We now turn to the problem of ﬁnding the marginal distributions for the latent
variables conditional on the observation sequence. For given parameter settings, we
also wish to make predictions of the next latent state zn and of the next observation
xn conditioned on the observed data x1, . . . , xn−1 for use in real-time applications.
These inference problems can be solved efﬁciently using the sum-product algorithm,
which in the context of the linear dynamical system gives rise to the Kalman ﬁlter
and Kalman smoother equations.
It is worth emphasizing that because the linear dynamical system is a linear-
Gaussian model, the joint distribution over all latent and observed variables is simply
a Gaussian, and so in principle we could solve inference problems by using the
standard results derived in previous chapters for the marginals and conditionals of a
multivariate Gaussian. The role of the sum-product algorithm is to provide a more
efﬁcient way to perform such computations.
Linear dynamical systems have the identical factorization, given by (13.6), to
hidden Markov models, and are again described by the factor graphs in Figures 13.14
and 13.15. Inference algorithms therefore take precisely the same form except that
summations over latent variables are replaced by integrations. We begin by consid-
ering the forward equations in which we treat zN as the root node, and propagate
messages from the leaf node h(z1) to the root. From (13.77), the initial message will
be Gaussian, and because each of the factors is Gaussian, all subsequent messages
will also be Gaussian. By convention, we shall propagate messages that are nor-
malized marginal distributions corresponding to p(zn|x1, . . . , xn), which we denote
by
α(zn) = N(zn|µn, Vn).
(13.84)
This is precisely analogous to the propagation of scaled variables α(zn) given by
(13.59) in the discrete case of the hidden Markov model, and so the recursion equa-
tion now takes the form
cnα(zn) = p(xn|zn)

α(zn−1)p(zn|zn−1) dzn−1.
(13.85)
Substituting for the conditionals p(zn|zn−1) and p(xn|zn), using (13.75) and (13.76),
respectively, and making use of (13.84), we see that (13.85) becomes
cnN(zn|µn, Vn) = N(xn|Czn, Σ)

N(zn|Azn−1, Γ)N(zn−1|µn−1, Vn−1) dzn−1.
(13.86)
Here we are supposing that µn−1 and Vn−1 are known, and by evaluating the inte-
gral in (13.86), we wish to determine values for µn and Vn. The integral is easily
evaluated by making use of the result (2.115), from which it follows that

N(zn|Azn−1, Γ)N(zn−1|µn−1, Vn−1) dzn−1
= N(zn|Aµn−1, Pn−1)
(13.87)


---
**Page 608**
13.3. Linear Dynamical Systems
639
where we have deﬁned
Pn−1 = AVn−1AT + Γ.
(13.88)
We can now combine this result with the ﬁrst factor on the right-hand side of (13.86)
by making use of (2.115) and (2.116) to give
µn
=
Aµn−1 + Kn(xn −CAµn−1)
(13.89)
Vn
=
(I −KnC)Pn−1
(13.90)
cn
=
N(xn|CAµn−1, CPn−1CT + Σ).
(13.91)
Here we have made use of the matrix inverse identities (C.5) and (C.7) and also
deﬁned the Kalman gain matrix
Kn = Pn−1CT 
CPn−1CT + Σ
−1 .
(13.92)
Thus, given the values of µn−1 and Vn−1, together with the new observation xn,
we can evaluate the Gaussian marginal for zn having mean µn and covariance Vn,
as well as the normalization coefﬁcient cn.
The initial conditions for these recursion equations are obtained from
c1α(z1) = p(z1)p(x1|z1).
(13.93)
Because p(z1) is given by (13.77), and p(x1|z1) is given by (13.76), we can again
make use of (2.115) to calculate c1 and (2.116) to calculate µ1 and V1 giving
µ1
=
µ0 + K1(x1 −Cµ0)
(13.94)
V1
=
(I −K1C)V0
(13.95)
c1
=
N(x1|Cµ0, CV0CT + Σ)
(13.96)
where
K1 = V0CT 
CV0CT + Σ−1 .
(13.97)
Similarly, the likelihood function for the linear dynamical system is given by (13.63)
in which the factors cn are found using the Kalman ﬁltering equations.
We can interpret the steps involved in going from the posterior marginal over
zn−1 to the posterior marginal over zn as follows. In (13.89), we can view the
quantity Aµn−1 as the prediction of the mean over zn obtained by simply taking the
mean over zn−1 and projecting it forward one step using the transition probability
matrix A. This predicted mean would give a predicted observation for xn given by
CAzn−1 obtained by applying the emission probability matrix C to the predicted
hidden state mean. We can view the update equation (13.89) for the mean of the
hidden variable distribution as taking the predicted mean Aµn−1 and then adding
a correction that is proportional to the error xn −CAzn−1 between the predicted
observation and the actual observation. The coefﬁcient of this correction is given by
the Kalman gain matrix. Thus we can view the Kalman ﬁlter as a process of making
successive predictions and then correcting these predictions in the light of the new
observations. This is illustrated graphically in Figure 13.21.


---
**Page 609**
640
13. SEQUENTIAL DATA
zn−1
zn
zn
Figure 13.21
The linear dynamical system can be viewed as a sequence of steps in which increasing un-
certainty in the state variable due to diffusion is compensated by the arrival of new data. In the left-hand plot,
the blue curve shows the distribution p(zn−1|x1, . . . , xn−1), which incorporates all the data up to step n −1.
The diffusion arising from the nonzero variance of the transition probability p(zn|zn−1) gives the distribution
p(zn|x1, . . . , xn−1), shown in red in the centre plot. Note that this is broader and shifted relative to the blue curve
(which is shown dashed in the centre plot for comparison). The next data observation xn contributes through the
emission density p(xn|zn), which is shown as a function of zn in green on the right-hand plot. Note that this is not
a density with respect to zn and so is not normalized to one. Inclusion of this new data point leads to a revised
distribution p(zn|x1, . . . , xn) for the state density shown in blue. We see that observation of the data has shifted
and narrowed the distribution compared to p(zn|x1, . . . , xn−1) (which is shown in dashed in the right-hand plot
for comparison).
If we consider a situation in which the measurement noise is small compared
to the rate at which the latent variable is evolving, then we ﬁnd that the posterior
distribution for zn depends only on the current measurement xn, in accordance with
Exercise 13.27
the intuition from our simple example at the start of the section. Similarly, if the
latent variable is evolving slowly relative to the observation noise level, we ﬁnd that
the posterior mean for zn is obtained by averaging all of the measurements obtained
up to that time.
Exercise 13.28
One of the most important applications of the Kalman ﬁlter is to tracking, and
this is illustrated using a simple example of an object moving in two dimensions in
Figure 13.22.
So far, we have solved the inference problem of ﬁnding the posterior marginal
for a node zn given observations from x1 up to xn. Next we turn to the problem of
ﬁnding the marginal for a node zn given all observations x1 to xN. For temporal
data, this corresponds to the inclusion of future as well as past observations. Al-
though this cannot be used for real-time prediction, it plays a key role in learning the
parameters of the model. By analogy with the hidden Markov model, this problem
can be solved by propagating messages from node xN back to node x1 and com-
bining this information with that obtained during the forward message passing stage
used to compute the α(zn).
In the LDS literature, it is usual to formulate this backward recursion in terms
of γ(zn) = α(zn)β(zn) rather than in terms of β(zn). Because γ(zn) must also be
Gaussian, we write it in the form
γ(zn) = α(zn)β(zn) = N(zn|µn, Vn).
(13.98)
To derive the required recursion, we start from the backward recursion (13.62) for


---
**Page 610**
13.3. Linear Dynamical Systems
641
Figure 13.22
An illustration of a linear dy-
namical system being used to
track a moving object. The blue
points indicate the true positions
of the object in a two-dimensional
space at successive time steps,
the green points denote noisy
measurements of the positions,
and the red crosses indicate the
means of the inferred posterior
distributions of the positions ob-
tained by running the Kalman ﬁl-
tering equations.
The covari-
ances of the inferred positions
are indicated by the red ellipses,
which correspond to contours
having one standard deviation.
β(zn), which, for continuous latent variables, can be written in the form
cn+1β(zn) =

β(zn+1)p(xn+1|zn+1)p(zn+1|zn) dzn+1.
(13.99)
We now multiply both sides of (13.99) by α(zn) and substitute for p(xn+1|zn+1)
and p(zn+1|zn) using (13.75) and (13.76). Then we make use of (13.89), (13.90)
and (13.91), together with (13.98), and after some manipulation we obtain
Exercise 13.29
µn
=
µn + Jn

µn+1 −AµN

(13.100)
Vn
=
Vn + Jn

Vn+1 −Pn

JT
n
(13.101)
where we have deﬁned
Jn = VnAT (Pn)−1
(13.102)
and we have made use of AVn = PnJT
n. Note that these recursions require that the
forward pass be completed ﬁrst so that the quantities µn and Vn will be available
for the backward pass.
For the EM algorithm, we also require the pairwise posterior marginals, which
can be obtained from (13.65) in the form
ξ(zn−1, zn) = (cn)−1 α(zn−1)p(xn|zn)p(zn|z−1)β(zn)
=
N(zn−1|µn−1, Vn−1)N(zn|Azn−1, Γ)N(xn|Czn, Σ)N(zn|µn, Vn)
cnα(zn)
.
(13.103)
Substituting for α(zn) using (13.84) and rearranging, we see that ξ(zn−1, zn) is a
Gaussian with mean given with components γ(zn−1) and γ(zn), and a covariance
between zn and zn−1 given by
Exercise 13.31
cov[zn, zn−1] = Jn−1 Vn.
(13.104)


---
**Page 611**
642
13. SEQUENTIAL DATA
13.3.2
Learning in LDS
So far, we have considered the inference problem for linear dynamical systems,
assuming that the model parameters θ = {A, Γ, C, Σ, µ0, V0} are known. Next, we
consider the determination of these parameters using maximum likelihood (Ghahra-
mani and Hinton, 1996b). Because the model has latent variables, this can be ad-
dressed using the EM algorithm, which was discussed in general terms in Chapter 9.
We can derive the EM algorithm for the linear dynamical system as follows. Let
us denote the estimated parameter values at some particular cycle of the algorithm
by θold. For these parameter values, we can run the inference algorithm to determine
the posterior distribution of the latent variables p(Z|X, θold), or more precisely those
local posterior marginals that are required in the M step. In particular, we shall
require the following expectations
E [zn]
=
µn
(13.105)
E 
znzT
n−1
	
=
Jn−1 Vn + µnµT
n−1
(13.106)
E

znzT
n
	
=
Vn + µnµT
n
(13.107)
where we have used (13.104).
Now we consider the complete-data log likelihood function, which is obtained
by taking the logarithm of (13.6) and is therefore given by
ln p(X, Z|θ)
=
ln p(z1|µ0, V0) +
N

n=2
ln p(zn|zn−1, A, Γ)
+
N

n=1
ln p(xn|zn, C, Σ)
(13.108)
in which we have made the dependence on the parameters explicit. We now take the
expectation of the complete-data log likelihood with respect to the posterior distri-
bution p(Z|X, θold) which deﬁnes the function
Q(θ, θold) = EZ|θold [ln p(X, Z|θ)] .
(13.109)
In the M step, this function is maximized with respect to the components of θ.
Consider ﬁrst the parameters µ0 and V0. If we substitute for p(z1|µ0, V0) in
(13.108) using (13.77), and then take the expectation with respect to Z, we obtain
Q(θ, θold) = −1
2 ln |V0| −EZ|θold
1
2(z1 −µ0)TV−1
0 (z1 −µ0)

+ const
where all terms not dependent on µ0 or V0 have been absorbed into the additive
constant. Maximization with respect to µ0 and V0 is easily performed by making
use of the maximum likelihood solution for a Gaussian distribution discussed in
Section 2.3.4, giving
Exercise 13.32


---
**Page 612**
13.3. Linear Dynamical Systems
643
µnew
0
=
E[z1]
(13.110)
Vnew
0
=
E[z1zT
1 ] −E[z1]E[zT
1 ].
(13.111)
Similarly, to optimize A and Γ, we substitute for p(zn|zn−1, A, Γ) in (13.108)
using (13.75) giving
Q(θ, θold) = −N −1
2
ln |Γ|
−EZ|θold

1
2
N

n=2
(zn −Azn−1)TΓ−1(zn −Azn−1)
 
+ const
(13.112)
in which the constant comprises terms that are independent of A and Γ. Maximizing
with respect to these parameters then gives
Exercise 13.33
Anew
=
 N

n=2
E 
znzT
n−1
	
  N

n=2
E 
zn−1zT
n−1
	
−1
(13.113)
Γnew
=
1
N −1
N

n=2

E 
znzT
n
	
−AnewE 
zn−1zT
n
	
−E

znzT
n−1
	
Anew + AnewE

zn−1zT
n−1
	
(Anew)T
.
(13.114)
Note that Anew must be evaluated ﬁrst, and the result can then be used to determine
Γnew.
Finally, in order to determine the new values of C and Σ, we substitute for
p(xn|zn, C, Σ) in (13.108) using (13.76) giving
Q(θ, θold)
=
−N
2 ln |Σ|
−EZ|θold

1
2
N

n=1
(xn −Czn)TΣ−1(xn −Czn)
 
+ const.
Maximizing with respect to C and Σ then gives
Exercise 13.34
Cnew
=
 N

n=1
xnE

zT
n
	
  N

n=1
E

znzT
n
	
−1
(13.115)
Σnew
=
1
N
N

n=1

xnxT
n −CnewE [zn] xT
n
−xnE 
zT
n
	
Cnew + CnewE 
znzT
n
	
Cnew
.
(13.116)


---
**Page 613**
644
13. SEQUENTIAL DATA
We have approached parameter learning in the linear dynamical system using
maximum likelihood. Inclusion of priors to give a MAP estimate is straightforward,
and a fully Bayesian treatment can be found by applying the analytical approxima-
tion techniques discussed in Chapter 10, though a detailed treatment is precluded
here due to lack of space.
13.3.3
Extensions of LDS
As with the hidden Markov model, there is considerable interest in extending
the basic linear dynamical system in order to increase its capabilities. Although the
assumption of a linear-Gaussian model leads to efﬁcient algorithms for inference
and learning, it also implies that the marginal distribution of the observed variables
is simply a Gaussian, which represents a signiﬁcant limitation. One simple extension
of the linear dynamical system is to use a Gaussian mixture as the initial distribution
for z1. If this mixture has K components, then the forward recursion equations
(13.85) will lead to a mixture of K Gaussians over each hidden variable zn, and so
the model is again tractable.
For many applications, the Gaussian emission density is a poor approximation.
If instead we try to use a mixture of K Gaussians as the emission density, then the
posterior α(z1) will also be a mixture of K Gaussians. However, from (13.85) the
posterior α(z2) will comprise a mixture of K2 Gaussians, and so on, with α(zn)
being given by a mixture of Kn Gaussians. Thus the number of components grows
exponentially with the length of the chain, and so this model is impractical.
More generally, introducing transition or emission models that depart from the
linear-Gaussian (or other exponential family) model leads to an intractable infer-
ence problem. We can make deterministic approximations such as assumed den-
sity ﬁltering or expectation propagation, or we can make use of sampling methods,
Chapter 10
as discussed in Section 13.3.4. One widely used approach is to make a Gaussian
approximation by linearizing around the mean of the predicted distribution, which
gives rise to the extended Kalman ﬁlter (Zarchan and Musoff, 2005).
As with hidden Markov models, we can develop interesting extensions of the ba-
sic linear dynamical system by expanding its graphical representation. For example,
the switching state space model (Ghahramani and Hinton, 1998) can be viewed as
a combination of the hidden Markov model with a set of linear dynamical systems.
The model has multiple Markov chains of continuous linear-Gaussian latent vari-
ables, each of which is analogous to the latent chain of the linear dynamical system
discussed earlier, together with a Markov chain of discrete variables of the form used
in a hidden Markov model. The output at each time step is determined by stochas-
tically choosing one of the continuous latent chains, using the state of the discrete
latent variable as a switch, and then emitting an observation from the corresponding
conditional output distribution. Exact inference in this model is intractable, but vari-
ational methods lead to an efﬁcient inference scheme involving forward-backward
recursions along each of the continuous and discrete Markov chains independently.
Note that, if we consider multiple chains of discrete latent variables, and use one as
the switch to select from the remainder, we obtain an analogous model having only
discrete latent variables known as the switching hidden Markov model.


---
**Page 614**
13.3. Linear Dynamical Systems
645
13.3.4
Particle ﬁlters
For dynamical systems which do not have a linear-Gaussian, for example, if
they use a non-Gaussian emission density, we can turn to sampling methods in order
Chapter 11
to ﬁnd a tractable inference algorithm. In particular, we can apply the sampling-
importance-resampling formalism of Section 11.1.5 to obtain a sequential Monte
Carlo algorithm known as the particle ﬁlter.
Consider the class of distributions represented by the graphical model in Fig-
ure 13.5, and suppose we are given the observed values Xn = (x1, . . . , xn) and
we wish to draw L samples from the posterior distribution p(zn|Xn). Using Bayes’
theorem, we have
E[f(zn)]
=

f(zn)p(zn|Xn) dzn
=

f(zn)p(zn|xn, Xn−1) dzn
=

f(zn)p(xn|zn)p(zn|Xn−1) dzn

p(xn|zn)p(zn|Xn−1) dzn
≃
L

l=1
w(l)
n f(z(l)
n )
(13.117)
where {z(l)
n } is a set of samples drawn from p(zn|Xn−1) and we have made use of
the conditional independence property p(xn|zn, Xn−1) = p(xn|zn), which follows
from the graph in Figure 13.5. The sampling weights {w(l)
n } are deﬁned by
w(l)
n =
p(xn|z(l)
n )
L
m=1 p(xn|z(m)
n
)
(13.118)
where the same samples are used in the numerator as in the denominator. Thus the
posterior distribution p(zn|xn) is represented by the set of samples {z(l)
n } together
with the corresponding weights {w(l)
n }. Note that these weights satisfy 0 ⩽w(l)
n 1
and 
l w(l)
n = 1.
Because we wish to ﬁnd a sequential sampling scheme, we shall suppose that
a set of samples and weights have been obtained at time step n, and that we have
subsequently observed the value of xn+1, and we wish to ﬁnd the weights and sam-
ples at time step n + 1. We ﬁrst sample from the distribution p(zn+1|Xn). This is


---
**Page 615**
646
13. SEQUENTIAL DATA
straightforward since, again using Bayes’ theorem
p(zn+1|Xn)
=

p(zn+1|zn, Xn)p(zn|Xn) dzn
=

p(zn+1|zn)p(zn|Xn) dzn
=

p(zn+1|zn)p(zn|xn, Xn−1) dzn
=

p(zn+1|zn)p(xn|zn)p(zn|Xn−1) dzn

p(xn|zn)p(zn|Xn−1) dzn
=

l
w(l)
n p(zn+1|z(l)
n )
(13.119)
where we have made use of the conditional independence properties
p(zn+1|zn, Xn)
=
p(zn+1|zn)
(13.120)
p(xn|zn, Xn−1)
=
p(xn|zn)
(13.121)
which follow from the application of the d-separation criterion to the graph in Fig-
ure 13.5. The distribution given by (13.119) is a mixture distribution, and samples
can be drawn by choosing a component l with probability given by the mixing coef-
ﬁcients w(l) and then drawing a sample from the corresponding component.
In summary, we can view each step of the particle ﬁlter algorithm as comprising
two stages. At time step n, we have a sample representation of the posterior dis-
tribution p(zn|Xn) expressed as samples {z(l)
n } with corresponding weights {w(l)
n }.
This can be viewed as a mixture representation of the form (13.119). To obtain the
corresponding representation for the next time step, we ﬁrst draw L samples from
the mixture distribution (13.119), and then for each sample we use the new obser-
vation xn+1 to evaluate the corresponding weights w(l)
n+1 ∝p(xn+1|z(l)
n+1). This is
illustrated, for the case of a single variable z, in Figure 13.23.
The particle ﬁltering, or sequential Monte Carlo, approach has appeared in the
literature under various names including the bootstrap ﬁlter (Gordon et al., 1993),
survival of the ﬁttest (Kanazawa et al., 1995), and the condensation algorithm (Isard
and Blake, 1998).
Exercises
13.1
(⋆) www
Use the technique of d-separation, discussed in Section 8.2, to verify
that the Markov model shown in Figure 13.3 having N nodes in total satisﬁes the
conditional independence properties (13.3) for n = 2, . . . , N. Similarly, show that
a model described by the graph in Figure 13.4 in which there are N nodes in total


---
**Page 616**
Exercises
647
p(zn|Xn)
p(zn+1|Xn)
p(xn+1|zn+1)
p(zn+1|Xn+1)
z
Figure 13.23
Schematic illustration of the operation of the particle ﬁlter for a one-dimensional latent
space. At time step n, the posterior p(zn|xn) is represented as a mixture distribution,
shown schematically as circles whose sizes are proportional to the weights w(l)
n . A set of
L samples is then drawn from this distribution and the new weights w(l)
n+1 evaluated using
p(xn+1|z(l)
n+1).
satisﬁes the conditional independence properties
p(xn|x1, . . . , xn−1) = p(xn|xn−1, xn−2)
(13.122)
for n = 3, . . . , N.
13.2
(⋆⋆) Consider the joint probability distribution (13.2) corresponding to the directed
graph of Figure 13.3. Using the sum and product rules of probability, verify that
this joint distribution satisﬁes the conditional independence property (13.3) for n =
2, . . . , N. Similarly, show that the second-order Markov model described by the
joint distribution (13.4) satisﬁes the conditional independence property
p(xn|x1, . . . , xn−1) = p(xn|xn−1, xn−2)
(13.123)
for n = 3, . . . , N.
13.3
(⋆) By using d-separation, show that the distribution p(x1, . . . , xN) of the observed
data for the state space model represented by the directed graph in Figure 13.5 does
not satisfy any conditional independence properties and hence does not exhibit the
Markov property at any ﬁnite order.
13.4
(⋆⋆) www
Consider a hidden Markov model in which the emission densities are
represented by a parametric model p(x|z, w), such as a linear regression model or
a neural network, in which w is a vector of adaptive parameters. Describe how the
parameters w can be learned from data using maximum likelihood.


---
**Page 617**
648
13. SEQUENTIAL DATA
13.5
(⋆⋆) Verify the M-step equations (13.18) and (13.19) for the initial state probabili-
ties and transition probability parameters of the hidden Markov model by maximiza-
tion of the expected complete-data log likelihood function (13.17), using appropriate
Lagrange multipliers to enforce the summation constraints on the components of π
and A.
13.6
(⋆)
Show that if any elements of the parameters π or A for a hidden Markov
model are initially set to zero, then those elements will remain zero in all subsequent
updates of the EM algorithm.
13.7
(⋆) Consider a hidden Markov model with Gaussian emission densities. Show that
maximization of the function Q(θ, θold) with respect to the mean and covariance
parameters of the Gaussians gives rise to the M-step equations (13.20) and (13.21).
13.8
(⋆⋆) www
For a hidden Markov model having discrete observations governed by
a multinomial distribution, show that the conditional distribution of the observations
given the hidden variables is given by (13.22) and the corresponding M step equa-
tions are given by (13.23). Write down the analogous equations for the conditional
distribution and the M step equations for the case of a hidden Markov with multiple
binary output variables each of which is governed by a Bernoulli conditional dis-
tribution. Hint: refer to Sections 2.1 and 2.2 for a discussion of the corresponding
maximum likelihood solutions for i.i.d. data if required.
13.9
(⋆⋆) www
Use the d-separation criterion to verify that the conditional indepen-
dence properties (13.24)–(13.31) are satisﬁed by the joint distribution for the hidden
Markov model deﬁned by (13.6).
13.10
(⋆⋆⋆) By applying the sum and product rules of probability, verify that the condi-
tional independence properties (13.24)–(13.31) are satisﬁed by the joint distribution
for the hidden Markov model deﬁned by (13.6).
13.11
(⋆⋆) Starting from the expression (8.72) for the marginal distribution over the vari-
ables of a factor in a factor graph, together with the results for the messages in the
sum-product algorithm obtained in Section 13.2.3, derive the result (13.43) for the
joint posterior distribution over two successive latent variables in a hidden Markov
model.
13.12
(⋆⋆)
Suppose we wish to train a hidden Markov model by maximum likelihood
using data that comprises R independent sequences of observations, which we de-
note by X(r) where r = 1, . . . , R. Show that in the E step of the EM algorithm,
we simply evaluate posterior probabilities for the latent variables by running the α
and β recursions independently for each of the sequences. Also show that in the
M step, the initial probability and transition probability parameters are re-estimated


---
**Page 618**
Exercises
649
using modiﬁed forms of (13.18 ) and (13.19) given by
πk
=
R

r=1
γ(z(r)
1k )
R

r=1
K

j=1
γ(z(r)
1j )
(13.124)
Ajk
=
R

r=1
N

n=2
ξ(z(r)
n−1,j, z(r)
n,k)
R

r=1
K

l=1
N

n=2
ξ(z(r)
n−1,j, z(r)
n,l)
(13.125)
where, for notational convenience, we have assumed that the sequences are of the
same length (the generalization to sequences of different lengths is straightforward).
Similarly, show that the M-step equation for re-estimation of the means of Gaussian
emission models is given by
µk =
R

r=1
N

n=1
γ(z(r)
nk )x(r)
n
R

r=1
N

n=1
γ(z(r)
nk )
.
(13.126)
Note that the M-step equations for other emission model parameters and distributions
take an analogous form.
13.13
(⋆⋆) www
Use the deﬁnition (8.64) of the messages passed from a factor node
to a variable node in a factor graph, together with the expression (13.6) for the joint
distribution in a hidden Markov model, to show that the deﬁnition (13.50) of the
alpha message is the same as the deﬁnition (13.34).
13.14
(⋆⋆)
Use the deﬁnition (8.67) of the messages passed from a factor node to a
variable node in a factor graph, together with the expression (13.6) for the joint
distribution in a hidden Markov model, to show that the deﬁnition (13.52) of the
beta message is the same as the deﬁnition (13.35).
13.15
(⋆⋆) Use the expressions (13.33) and (13.43) for the marginals in a hidden Markov
model to derive the corresponding results (13.64) and (13.65) expressed in terms of
re-scaled variables.
13.16
(⋆⋆⋆)
In this exercise, we derive the forward message passing equation for the
Viterbi algorithm directly from the expression (13.6) for the joint distribution. This
involves maximizing over all of the hidden variables z1, . . . , zN. By taking the log-
arithm and then exchanging maximizations and summations, derive the recursion


---
**Page 619**
650
13. SEQUENTIAL DATA
(13.68) where the quantities ω(zn) are deﬁned by (13.70). Show that the initial
condition for this recursion is given by (13.69).
13.17
(⋆) www
Show that the directed graph for the input-output hidden Markov model,
given in Figure 13.18, can be expressed as a tree-structured factor graph of the form
shown in Figure 13.15 and write down expressions for the initial factor h(z1) and
for the general factor fn(zn−1, zn) where 2 ⩽n ⩽N.
13.18
(⋆⋆⋆)
Using the result of Exercise 13.17, derive the recursion equations, includ-
ing the initial conditions, for the forward-backward algorithm for the input-output
hidden Markov model shown in Figure 13.18.
13.19
(⋆) www
The Kalman ﬁlter and smoother equations allow the posterior distribu-
tions over individual latent variables, conditioned on all of the observed variables,
to be found efﬁciently for linear dynamical systems. Show that the sequence of
latent variable values obtained by maximizing each of these posterior distributions
individually is the same as the most probable sequence of latent values. To do this,
simply note that the joint distribution of all latent and observed variables in a linear
dynamical system is Gaussian, and hence all conditionals and marginals will also be
Gaussian, and then make use of the result (2.98).
13.20
(⋆⋆) www
Use the result (2.115) to prove (13.87).
13.21
(⋆⋆)
Use the results (2.115) and (2.116), together with the matrix identities (C.5)
and (C.7), to derive the results (13.89), (13.90), and (13.91), where the Kalman gain
matrix Kn is deﬁned by (13.92).
13.22
(⋆⋆) www
Using (13.93), together with the deﬁnitions (13.76) and (13.77) and
the result (2.115), derive (13.96).
13.23
(⋆⋆) Using (13.93), together with the deﬁnitions (13.76) and (13.77) and the result
(2.116), derive (13.94), (13.95) and (13.97).
13.24
(⋆⋆) www
Consider a generalization of (13.75) and (13.76) in which we include
constant terms a and c in the Gaussian means, so that
p(zn|zn−1) = N(zn|Azn−1 + a, Γ)
(13.127)
p(xn|zn) = N(xn|Czn + c, Σ).
(13.128)
Show that this extension can be re-case in the framework discussed in this chapter by
deﬁning a state vector z with an additional component ﬁxed at unity, and then aug-
menting the matrices A and C using extra columns corresponding to the parameters
a and c.
13.25
(⋆⋆)
In this exercise, we show that when the Kalman ﬁlter equations are applied
to independent observations, they reduce to the results given in Section 2.3 for the
maximum likelihood solution for a single Gaussian distribution. Consider the prob-
lem of ﬁnding the mean µ of a single Gaussian random variable x, in which we are
given a set of independent observations {x1, . . . , xN}. To model this we can use


---
**Page 620**
Exercises
651
a linear dynamical system governed by (13.75) and (13.76), with latent variables
{z1, . . . , zN} in which C becomes the identity matrix and where the transition prob-
ability A = 0 because the observations are independent. Let the parameters m0
and V0 of the initial state be denoted by µ0 and σ2
0, respectively, and suppose that
Σ becomes σ2. Write down the corresponding Kalman ﬁlter equations starting from
the general results (13.89) and (13.90), together with (13.94) and (13.95). Show that
these are equivalent to the results (2.141) and (2.142) obtained directly by consider-
ing independent data.
13.26
(⋆⋆⋆) Consider a special case of the linear dynamical system of Section 13.3 that is
equivalent to probabilistic PCA, so that the transition matrix A = 0, the covariance
Γ = I, and the noise covariance Σ = σ2I. By making use of the matrix inversion
identity (C.7) show that, if the emission density matrix C is denoted W, then the
posterior distribution over the hidden states deﬁned by (13.89) and (13.90) reduces
to the result (12.42) for probabilistic PCA.
13.27
(⋆) www
Consider a linear dynamical system of the form discussed in Sec-
tion 13.3 in which the amplitude of the observation noise goes to zero, so that Σ = 0.
Show that the posterior distribution for zn has mean xn and zero variance. This
accords with our intuition that if there is no noise, we should just use the current
observation xn to estimate the state variable zn and ignore all previous observations.
13.28
(⋆⋆⋆)
Consider a special case of the linear dynamical system of Section 13.3 in
which the state variable zn is constrained to be equal to the previous state variable,
which corresponds to A = I and Γ = 0. For simplicity, assume also that V0 →∞
so that the initial conditions for z are unimportant, and the predictions are determined
purely by the data. Use proof by induction to show that the posterior mean for state
zn is determined by the average of x1, . . . , xn. This corresponds to the intuitive
result that if the state variable is constant, our best estimate is obtained by averaging
the observations.
13.29
(⋆⋆⋆)
Starting from the backwards recursion equation (13.99), derive the RTS
smoothing equations (13.100) and (13.101) for the Gaussian linear dynamical sys-
tem.
13.30
(⋆⋆)
Starting from the result (13.65) for the pairwise posterior marginal in a state
space model, derive the speciﬁc form (13.103) for the case of the Gaussian linear
dynamical system.
13.31
(⋆⋆) Starting from the result (13.103) and by substituting for α(zn) using (13.84),
verify the result (13.104) for the covariance between zn and zn−1.
13.32
(⋆⋆) www
Verify the results (13.110) and (13.111) for the M-step equations for
µ0 and V0 in the linear dynamical system.
13.33
(⋆⋆) Verify the results (13.113) and (13.114) for the M-step equations for A and Γ
in the linear dynamical system.


---
**Page 621**
652
13. SEQUENTIAL DATA
13.34
(⋆⋆) Verify the results (13.115) and (13.116) for the M-step equations for C and Σ
in the linear dynamical system.


---
**Page 622**
14
Combining
Models
In earlier chapters, we have explored a range of different models for solving classiﬁ-
cation and regression problems. It is often found that improved performance can be
obtained by combining multiple models together in some way, instead of just using
a single model in isolation. For instance, we might train L different models and then
make predictions using the average of the predictions made by each model. Such
combinations of models are sometimes called committees. In Section 14.2, we dis-
cuss ways to apply the committee concept in practice, and we also give some insight
into why it can sometimes be an effective procedure.
One important variant of the committee method, known as boosting, involves
training multiple models in sequence in which the error function used to train a par-
ticular model depends on the performance of the previous models. This can produce
substantial improvements in performance compared to the use of a single model and
is discussed in Section 14.3.
Instead of averaging the predictions of a set of models, an alternative form of
653


---
**Page 623**
654
14. COMBINING MODELS
model combination is to select one of the models to make the prediction, in which
the choice of model is a function of the input variables. Thus different models be-
come responsible for making predictions in different regions of input space. One
widely used framework of this kind is known as a decision tree in which the selec-
tion process can be described as a sequence of binary selections corresponding to
the traversal of a tree structure and is discussed in Section 14.4. In this case, the
individual models are generally chosen to be very simple, and the overall ﬂexibility
of the model arises from the input-dependent selection process. Decision trees can
be applied to both classiﬁcation and regression problems.
One limitation of decision trees is that the division of input space is based on
hard splits in which only one model is responsible for making predictions for any
given value of the input variables. The decision process can be softened by moving
to a probabilistic framework for combining models, as discussed in Section 14.5. For
example, if we have a set of K models for a conditional distribution p(t|x, k) where
x is the input variable, t is the target variable, and k = 1, . . . , K indexes the model,
then we can form a probabilistic mixture of the form
p(t|x) =
K

k=1
πk(x)p(t|x, k)
(14.1)
in which πk(x) = p(k|x) represent the input-dependent mixing coefﬁcients. Such
models can be viewed as mixture distributions in which the component densities, as
well as the mixing coefﬁcients, are conditioned on the input variables and are known
as mixtures of experts. They are closely related to the mixture density network model
discussed in Section 5.6.
14.1. Bayesian Model Averaging
It is important to distinguish between model combination methods and Bayesian
model averaging, as the two are often confused. To understand the difference, con-
sider the example of density estimation using a mixture of Gaussians in which several
Section 9.2
Gaussian components are combined probabilistically. The model contains a binary
latent variable z that indicates which component of the mixture is responsible for
generating the corresponding data point. Thus the model is speciﬁed in terms of a
joint distribution
p(x, z)
(14.2)
and the corresponding density over the observed variable x is obtained by marginal-
izing over the latent variable
p(x) =

z
p(x, z).
(14.3)


---
**Page 624**
14.2. Committees
655
In the case of our Gaussian mixture example, this leads to a distribution of the form
p(x) =
K

k=1
πkN(x|µk, Σk)
(14.4)
with the usual interpretation of the symbols. This is an example of model combi-
nation. For independent, identically distributed data, we can use (14.3) to write the
marginal probability of a data set X = {x1, . . . , xN} in the form
p(X) =
N

n=1
p(xn) =
N

n=1

zn
p(xn, zn)
 
.
(14.5)
Thus we see that each observed data point xn has a corresponding latent variable zn.
Now suppose we have several different models indexed by h = 1, . . . , H with
prior probabilities p(h). For instance one model might be a mixture of Gaussians and
another model might be a mixture of Cauchy distributions. The marginal distribution
over the data set is given by
p(X) =
H

h=1
p(X|h)p(h).
(14.6)
This is an example of Bayesian model averaging. The interpretation of this summa-
tion over h is that just one model is responsible for generating the whole data set,
and the probability distribution over h simply reﬂects our uncertainty as to which
model that is. As the size of the data set increases, this uncertainty reduces, and
the posterior probabilities p(h|X) become increasingly focussed on just one of the
models.
This highlights the key difference between Bayesian model averaging and model
combination, because in Bayesian model averaging the whole data set is generated
by a single model. By contrast, when we combine multiple models, as in (14.5), we
see that different data points within the data set can potentially be generated from
different values of the latent variable z and hence by different components.
Although we have considered the marginal probability p(X), the same consid-
erations apply for the predictive density p(x|X) or for conditional distributions such
as p(t|x, X, T).
Exercise 14.1
14.2. Committees
The simplest way to construct a committee is to average the predictions of a set of
individual models. Such a procedure can be motivated from a frequentist perspective
by considering the trade-off between bias and variance, which decomposes the er-
Section 3.2
ror due to a model into the bias component that arises from differences between the
model and the true function to be predicted, and the variance component that repre-
sents the sensitivity of the model to the individual data points. Recall from Figure 3.5


---
**Page 625**
656
14. COMBINING MODELS
that when we trained multiple polynomials using the sinusoidal data, and then aver-
aged the resulting functions, the contribution arising from the variance term tended to
cancel, leading to improved predictions. When we averaged a set of low-bias mod-
els (corresponding to higher order polynomials), we obtained accurate predictions
for the underlying sinusoidal function from which the data were generated.
In practice, of course, we have only a single data set, and so we have to ﬁnd
a way to introduce variability between the different models within the committee.
One approach is to use bootstrap data sets, discussed in Section 1.2.3. Consider a
regression problem in which we are trying to predict the value of a single continuous
variable, and suppose we generate M bootstrap data sets and then use each to train
a separate copy ym(x) of a predictive model where m = 1, . . . , M. The committee
prediction is given by
yCOM(x) = 1
M
M

m=1
ym(x).
(14.7)
This procedure is known as bootstrap aggregation or bagging (Breiman, 1996).
Suppose the true regression function that we are trying to predict is given by
h(x), so that the output of each of the models can be written as the true value plus
an error in the form
ym(x) = h(x) + ϵm(x).
(14.8)
The average sum-of-squares error then takes the form
Ex

{ym(x) −h(x)}2	
= Ex

ϵm(x)2	
(14.9)
where Ex[·] denotes a frequentist expectation with respect to the distribution of the
input vector x. The average error made by the models acting individually is therefore
EAV = 1
M
M

m=1
Ex

ϵm(x)2	
.
(14.10)
Similarly, the expected error from the committee (14.7) is given by
ECOM
=
Ex
⎡
⎣

1
M
M

m=1
ym(x) −h(x)
2⎤
⎦
=
Ex
⎡
⎣

1
M
M

m=1
ϵm(x)
2⎤
⎦
(14.11)
If we assume that the errors have zero mean and are uncorrelated, so that
Ex [ϵm(x)]
=
0
(14.12)
Ex [ϵm(x)ϵl(x)]
=
0,
m ̸= l
(14.13)


---
**Page 626**
14.3. Boosting
657
then we obtain
Exercise 14.2
ECOM = 1
M EAV.
(14.14)
This apparently dramatic result suggests that the average error of a model can be
reduced by a factor of M simply by averaging M versions of the model. Unfortu-
nately, it depends on the key assumption that the errors due to the individual models
are uncorrelated. In practice, the errors are typically highly correlated, and the reduc-
tion in overall error is generally small. It can, however, be shown that the expected
committee error will not exceed the expected error of the constituent models, so that
ECOM ⩽EAV. In order to achieve more signiﬁcant improvements, we turn to a more
Exercise 14.3
sophisticated technique for building committees, known as boosting.
14.3. Boosting
Boosting is a powerful technique for combining multiple ‘base’ classiﬁers to produce
a form of committee whose performance can be signiﬁcantly better than that of any
of the base classiﬁers. Here we describe the most widely used form of boosting
algorithm called AdaBoost, short for ‘adaptive boosting’, developed by Freund and
Schapire (1996). Boosting can give good results even if the base classiﬁers have a
performance that is only slightly better than random, and hence sometimes the base
classiﬁers are known as weak learners. Originally designed for solving classiﬁcation
problems, boosting can also be extended to regression (Friedman, 2001).
The principal difference between boosting and the committee methods such as
bagging discussed above, is that the base classiﬁers are trained in sequence, and
each base classiﬁer is trained using a weighted form of the data set in which the
weighting coefﬁcient associated with each data point depends on the performance
of the previous classiﬁers. In particular, points that are misclassiﬁed by one of the
base classiﬁers are given greater weight when used to train the next classiﬁer in
the sequence. Once all the classiﬁers have been trained, their predictions are then
combined through a weighted majority voting scheme, as illustrated schematically
in Figure 14.1.
Consider a two-class classiﬁcation problem, in which the training data comprises
input vectors x1, . . . , xN along with corresponding binary target variables t1, . . . , tN
where tn ∈{−1, 1}. Each data point is given an associated weighting parameter
wn, which is initially set 1/N for all data points. We shall suppose that we have
a procedure available for training a base classiﬁer using weighted data to give a
function y(x) ∈{−1, 1}. At each stage of the algorithm, AdaBoost trains a new
classiﬁer using a data set in which the weighting coefﬁcients are adjusted according
to the performance of the previously trained classiﬁer so as to give greater weight
to the misclassiﬁed data points. Finally, when the desired number of base classiﬁers
have been trained, they are combined to form a committee using coefﬁcients that
give different weight to different base classiﬁers. The precise form of the AdaBoost
algorithm is given below.


---
**Page 627**
658
14. COMBINING MODELS
Figure 14.1
Schematic illustration of the
boosting framework.
Each
base classiﬁer ym(x) is trained
on a weighted form of the train-
ing set (blue arrows) in which
the weights w(m)
n
depend on
the performance of the pre-
vious base classiﬁer ym−1(x)
(green arrows). Once all base
classiﬁers have been trained,
they
are
combined
to
give
the ﬁnal classiﬁer YM(x) (red
arrows).
{w(1)
n }
{w(2)
n }
{w(M)
n
}
y1(x)
y2(x)
yM(x)
YM(x) = sign
 M

m
αmym(x)

AdaBoost
1. Initialize the data weighting coefﬁcients {wn} by setting w(1)
n
= 1/N for
n = 1, . . . , N.
2. For m = 1, . . . , M:
(a) Fit a classiﬁer ym(x) to the training data by minimizing the weighted
error function
Jm =
N

n=1
w(m)
n
I(ym(xn) ̸= tn)
(14.15)
where I(ym(xn) ̸= tn) is the indicator function and equals 1 when
ym(xn) ̸= tn and 0 otherwise.
(b) Evaluate the quantities
ϵm =
N

n=1
w(m)
n
I(ym(xn) ̸= tn)
N

n=1
w(m)
n
(14.16)
and then use these to evaluate
αm = ln
1 −ϵm
ϵm

.
(14.17)
(c) Update the data weighting coefﬁcients
w(m+1)
n
= w(m)
n
exp {αmI(ym(xn) ̸= tn)}
(14.18)


---
**Page 628**
14.3. Boosting
659
3. Make predictions using the ﬁnal model, which is given by
YM(x) = sign
 M

m=1
αmym(x)

.
(14.19)
We see that the ﬁrst base classiﬁer y1(x) is trained using weighting coefﬁ-
cients w(1)
n
that are all equal, which therefore corresponds to the usual procedure
for training a single classiﬁer. From (14.18), we see that in subsequent iterations
the weighting coefﬁcients w(m)
n
are increased for data points that are misclassiﬁed
and decreased for data points that are correctly classiﬁed. Successive classiﬁers are
therefore forced to place greater emphasis on points that have been misclassiﬁed by
previous classiﬁers, and data points that continue to be misclassiﬁed by successive
classiﬁers receive ever greater weight. The quantities ϵm represent weighted mea-
sures of the error rates of each of the base classiﬁers on the data set. We therefore
see that the weighting coefﬁcients αm deﬁned by (14.17) give greater weight to the
more accurate classiﬁers when computing the overall output given by (14.19).
The AdaBoost algorithm is illustrated in Figure 14.2, using a subset of 30 data
points taken from the toy classiﬁcation data set shown in Figure A.7. Here each base
learners consists of a threshold on one of the input variables. This simple classiﬁer
corresponds to a form of decision tree known as a ‘decision stumps’, i.e., a deci-
Section 14.4
sion tree with a single node. Thus each base learner classiﬁes an input according to
whether one of the input features exceeds some threshold and therefore simply parti-
tions the space into two regions separated by a linear decision surface that is parallel
to one of the axes.
14.3.1
Minimizing exponential error
Boosting was originally motivated using statistical learning theory, leading to
upper bounds on the generalization error. However, these bounds turn out to be too
loose to have practical value, and the actual performance of boosting is much better
than the bounds alone would suggest. Friedman et al. (2000) gave a different and
very simple interpretation of boosting in terms of the sequential minimization of an
exponential error function.
Consider the exponential error function deﬁned by
E =
N

n=1
exp {−tnfm(xn)}
(14.20)
where fm(x) is a classiﬁer deﬁned in terms of a linear combination of base classiﬁers
yl(x) of the form
fm(x) = 1
2
m

l=1
αlyl(x)
(14.21)
and tn ∈{−1, 1} are the training set target values. Our goal is to minimize E with
respect to both the weighting coefﬁcients αl and the parameters of the base classiﬁers
yl(x).


---
**Page 629**
660
14. COMBINING MODELS
m = 1
−1
0
1
2
−2
0
2
m = 2
−1
0
1
2
−2
0
2
m = 3
−1
0
1
2
−2
0
2
m = 6
−1
0
1
2
−2
0
2
m = 10
−1
0
1
2
−2
0
2
m = 150
−1
0
1
2
−2
0
2
Figure 14.2
Illustration of boosting in which the base learners consist of simple thresholds applied to one or
other of the axes. Each ﬁgure shows the number m of base learners trained so far, along with the decision
boundary of the most recent base learner (dashed black line) and the combined decision boundary of the en-
semble (solid green line). Each data point is depicted by a circle whose radius indicates the weight assigned to
that data point when training the most recently added base learner. Thus, for instance, we see that points that
are misclassiﬁed by the m = 1 base learner are given greater weight when training the m = 2 base learner.
Instead of doing a global error function minimization, however, we shall sup-
pose that the base classiﬁers y1(x), . . . , ym−1(x) are ﬁxed, as are their coefﬁcients
α1, . . . , αm−1, and so we are minimizing only with respect to αm and ym(x). Sep-
arating off the contribution from base classiﬁer ym(x), we can then write the error
function in the form
E
=
N

n=1
exp

−tnfm−1(xn) −1
2tnαmym(xn)

=
N

n=1
w(m)
n
exp

−1
2tnαmym(xn)

(14.22)
where the coefﬁcients w(m)
n
= exp{−tnfm−1(xn)} can be viewed as constants
because we are optimizing only αm and ym(x). If we denote by Tm the set of
data points that are correctly classiﬁed by ym(x), and if we denote the remaining
misclassiﬁed points by Mm, then we can in turn rewrite the error function in the


---
**Page 630**
14.3. Boosting
661
form
E
=
e−αm/2 
n∈Tm
w(m)
n
+ eαm/2 
n∈Mm
w(m)
n
=
(eαm/2 −e−αm/2)
N

n=1
w(m)
n
I(ym(xn) ̸= tn) + e−αm/2
N

n=1
w(m)
n
.
(14.23)
When we minimize this with respect to ym(x), we see that the second term is con-
stant, and so this is equivalent to minimizing (14.15) because the overall multiplica-
tive factor in front of the summation does not affect the location of the minimum.
Similarly, minimizing with respect to αm, we obtain (14.17) in which ϵm is deﬁned
by (14.16).
Exercise 14.6
From (14.22) we see that, having found αm and ym(x), the weights on the data
points are updated using
w(m+1)
n
= w(m)
n
exp

−1
2tnαmym(xn)

.
(14.24)
Making use of the fact that
tnym(xn) = 1 −2I(ym(xn) ̸= tn)
(14.25)
we see that the weights w(m)
n
are updated at the next iteration using
w(m+1)
n
= w(m)
n
exp(−αm/2) exp {αmI(ym(xn) ̸= tn)} .
(14.26)
Because the term exp(−αm/2) is independent of n, we see that it weights all data
points by the same factor and so can be discarded. Thus we obtain (14.18).
Finally, once all the base classiﬁers are trained, new data points are classiﬁed by
evaluating the sign of the combined function deﬁned according to (14.21). Because
the factor of 1/2 does not affect the sign it can be omitted, giving (14.19).
14.3.2
Error functions for boosting
The exponential error function that is minimized by the AdaBoost algorithm
differs from those considered in previous chapters. To gain some insight into the
nature of the exponential error function, we ﬁrst consider the expected error given
by
Ex,t [exp{−ty(x)}] =

t

exp{−ty(x)}p(t|x)p(x) dx.
(14.27)
If we perform a variational minimization with respect to all possible functions y(x),
we obtain
Exercise 14.7
y(x) = 1
2 ln
 p(t = 1|x)
p(t = −1|x)

(14.28)


---
**Page 631**
662
14. COMBINING MODELS
Figure 14.3
Plot of the exponential (green) and
rescaled cross-entropy (red) error
functions along with the hinge er-
ror (blue) used in support vector
machines, and the misclassiﬁcation
error (black).
Note that for large
negative values of z = ty(x), the
cross-entropy
gives
a
linearly
in-
creasing penalty, whereas the expo-
nential loss gives an exponentially in-
creasing penalty.
−2
−1
0
1
2
z
E(z)
which is half the log-odds. Thus the AdaBoost algorithm is seeking the best approx-
imation to the log odds ratio, within the space of functions represented by the linear
combination of base classiﬁers, subject to the constrained minimization resulting
from the sequential optimization strategy. This result motivates the use of the sign
function in (14.19) to arrive at the ﬁnal classiﬁcation decision.
We have already seen that the minimizer y(x) of the cross-entropy error (4.90)
for two-class classiﬁcation is given by the posterior class probability. In the case
of a target variable t ∈{−1, 1}, we have seen that the error function is given by
Section 7.1.2
ln(1 + exp(−yt)). This is compared with the exponential error function in Fig-
ure 14.3, where we have divided the cross-entropy error by a constant factor ln(2)
so that it passes through the point (0, 1) for ease of comparison. We see that both
can be seen as continuous approximations to the ideal misclassiﬁcation error func-
tion. An advantage of the exponential error is that its sequential minimization leads
to the simple AdaBoost scheme. One drawback, however, is that it penalizes large
negative values of ty(x) much more strongly than cross-entropy. In particular, we
see that for large negative values of ty, the cross-entropy grows linearly with |ty|,
whereas the exponential error function grows exponentially with |ty|. Thus the ex-
ponential error function will be much less robust to outliers or misclassiﬁed data
points. Another important difference between cross-entropy and the exponential er-
ror function is that the latter cannot be interpreted as the log likelihood function of
any well-deﬁned probabilistic model. Furthermore, the exponential error does not
Exercise 14.8
generalize to classiﬁcation problems having K > 2 classes, again in contrast to the
cross-entropy for a probabilistic model, which is easily generalized to give (4.108).
Section 4.3.4
The interpretation of boosting as the sequential optimization of an additive model
under an exponential error (Friedman et al., 2000) opens the door to a wide range
of boosting-like algorithms, including multiclass extensions, by altering the choice
of error function. It also motivates the extension to regression problems (Friedman,
2001). If we consider a sum-of-squares error function for regression, then sequential
minimization of an additive model of the form (14.21) simply involves ﬁtting each
new base classiﬁer to the residual errors tn−fm−1(xn) from the previous model. As
Exercise 14.9
we have noted, however, the sum-of-squares error is not robust to outliers, and this


---
**Page 632**
14.4. Tree-based Models
663
Figure 14.4
Comparison
of
the
squared
error
(green) with the absolute error (red)
showing how the latter places much
less emphasis on large errors and
hence is more robust to outliers and
mislabelled data points.
0
z
E(z)
−1
1
can be addressed by basing the boosting algorithm on the absolute deviation |y −t|
instead. These two error functions are compared in Figure 14.4.
14.4. Tree-based Models
There are various simple, but widely used, models that work by partitioning the
input space into cuboid regions, whose edges are aligned with the axes, and then
assigning a simple model (for example, a constant) to each region. They can be
viewed as a model combination method in which only one model is responsible
for making predictions at any given point in input space. The process of selecting
a speciﬁc model, given a new input x, can be described by a sequential decision
making process corresponding to the traversal of a binary tree (one that splits into
two branches at each node). Here we focus on a particular tree-based framework
called classiﬁcation and regression trees, or CART (Breiman et al., 1984), although
there are many other variants going by such names as ID3 and C4.5 (Quinlan, 1986;
Quinlan, 1993).
Figure 14.5 shows an illustration of a recursive binary partitioning of the input
space, along with the corresponding tree structure. In this example, the ﬁrst step
Figure 14.5
Illustration of a two-dimensional in-
put space that has been partitioned
into ﬁve regions using axis-aligned
boundaries.
A
B
C
D
E
θ1
θ4
θ2
θ3
x1
x2


---
**Page 633**
664
14. COMBINING MODELS
Figure 14.6
Binary tree corresponding to the par-
titioning of input space shown in Fig-
ure 14.5.
x1 > θ1
x2 > θ3
x1 ⩽θ4
x2 ⩽θ2
A
B
C
D
E
divides the whole of the input space into two regions according to whether x1 ⩽θ1
or x1 > θ1 where θ1 is a parameter of the model. This creates two subregions, each
of which can then be subdivided independently. For instance, the region x1 ⩽θ1
is further subdivided according to whether x2 ⩽θ2 or x2 > θ2, giving rise to the
regions denoted A and B. The recursive subdivision can be described by the traversal
of the binary tree shown in Figure 14.6. For any new input x, we determine which
region it falls into by starting at the top of the tree at the root node and following
a path down to a speciﬁc leaf node according to the decision criteria at each node.
Note that such decision trees are not probabilistic graphical models.
Within each region, there is a separate model to predict the target variable. For
instance, in regression we might simply predict a constant over each region, or in
classiﬁcation we might assign each region to a speciﬁc class. A key property of tree-
based models, which makes them popular in ﬁelds such as medical diagnosis, for
example, is that they are readily interpretable by humans because they correspond
to a sequence of binary decisions applied to the individual input variables. For in-
stance, to predict a patient’s disease, we might ﬁrst ask “is their temperature greater
than some threshold?”. If the answer is yes, then we might next ask “is their blood
pressure less than some threshold?”. Each leaf of the tree is then associated with a
speciﬁc diagnosis.
In order to learn such a model from a training set, we have to determine the
structure of the tree, including which input variable is chosen at each node to form
the split criterion as well as the value of the threshold parameter θi for the split. We
also have to determine the values of the predictive variable within each region.
Consider ﬁrst a regression problem in which the goal is to predict a single target
variable t from a D-dimensional vector x = (x1, . . . , xD)T of input variables. The
training data consists of input vectors {x1, . . . , xN} along with the corresponding
continuous labels {t1, . . . , tN}. If the partitioning of the input space is given, and we
minimize the sum-of-squares error function, then the optimal value of the predictive
variable within any given region is just given by the average of the values of tn for
those data points that fall in that region.
Exercise 14.10
Now consider how to determine the structure of the decision tree. Even for a
ﬁxed number of nodes in the tree, the problem of determining the optimal structure
(including choice of input variable for each split as well as the corresponding thresh-


---
**Page 634**
14.4. Tree-based Models
665
olds) to minimize the sum-of-squares error is usually computationally infeasible due
to the combinatorially large number of possible solutions. Instead, a greedy opti-
mization is generally done by starting with a single root node, corresponding to the
whole input space, and then growing the tree by adding nodes one at a time. At each
step there will be some number of candidate regions in input space that can be split,
corresponding to the addition of a pair of leaf nodes to the existing tree. For each
of these, there is a choice of which of the D input variables to split, as well as the
value of the threshold. The joint optimization of the choice of region to split, and the
choice of input variable and threshold, can be done efﬁciently by exhaustive search
noting that, for a given choice of split variable and threshold, the optimal choice of
predictive variable is given by the local average of the data, as noted earlier. This
is repeated for all possible choices of variable to be split, and the one that gives the
smallest residual sum-of-squares error is retained.
Given a greedy strategy for growing the tree, there remains the issue of when
to stop adding nodes. A simple approach would be to stop when the reduction in
residual error falls below some threshold. However, it is found empirically that often
none of the available splits produces a signiﬁcant reduction in error, and yet after
several more splits a substantial error reduction is found. For this reason, it is com-
mon practice to grow a large tree, using a stopping criterion based on the number
of data points associated with the leaf nodes, and then prune back the resulting tree.
The pruning is based on a criterion that balances residual error against a measure of
model complexity. If we denote the starting tree for pruning by T0, then we deﬁne
T ⊂T0 to be a subtree of T0 if it can be obtained by pruning nodes from T0 (in
other words, by collapsing internal nodes by combining the corresponding regions).
Suppose the leaf nodes are indexed by τ = 1, . . . , |T|, with leaf node τ representing
a region Rτ of input space having Nτ data points, and |T| denoting the total number
of leaf nodes. The optimal prediction for region Rτ is then given by
yτ = 1
Nτ

xn∈Rτ
tn
(14.29)
and the corresponding contribution to the residual sum-of-squares is then
Qτ(T) =

xn∈Rτ
{tn −yτ}2 .
(14.30)
The pruning criterion is then given by
C(T) =
|T |

τ=1
Qτ(T) + λ|T|
(14.31)
The regularization parameter λ determines the trade-off between the overall residual
sum-of-squares error and the complexity of the model as measured by the number
|T| of leaf nodes, and its value is chosen by cross-validation.
For classiﬁcation problems, the process of growing and pruning the tree is sim-
ilar, except that the sum-of-squares error is replaced by a more appropriate measure


---
**Page 635**
666
14. COMBINING MODELS
of performance. If we deﬁne pτk to be the proportion of data points in region Rτ
assigned to class k, where k = 1, . . . , K, then two commonly used choices are the
cross-entropy
Qτ(T) =
K

k=1
pτk ln pτk
(14.32)
and the Gini index
Qτ(T) =
K

k=1
pτk (1 −pτk) .
(14.33)
These both vanish for pτk = 0 and pτk = 1 and have a maximum at pτk = 0.5. They
encourage the formation of regions in which a high proportion of the data points are
assigned to one class. The cross entropy and the Gini index are better measures than
the misclassiﬁcation rate for growing the tree because they are more sensitive to the
node probabilities. Also, unlike misclassiﬁcation rate, they are differentiable and
Exercise 14.11
hence better suited to gradient based optimization methods. For subsequent pruning
of the tree, the misclassiﬁcation rate is generally used.
The human interpretability of a tree model such as CART is often seen as its
major strength. However, in practice it is found that the particular tree structure that
is learned is very sensitive to the details of the data set, so that a small change to the
training data can result in a very different set of splits (Hastie et al., 2001).
There are other problems with tree-based methods of the kind considered in
this section. One is that the splits are aligned with the axes of the feature space,
which may be very suboptimal. For instance, to separate two classes whose optimal
decision boundary runs at 45 degrees to the axes would need a large number of
axis-parallel splits of the input space as compared to a single non-axis-aligned split.
Furthermore, the splits in a decision tree are hard, so that each region of input space
is associated with one, and only one, leaf node model. The last issue is particularly
problematic in regression where we are typically aiming to model smooth functions,
and yet the tree model produces piecewise-constant predictions with discontinuities
at the split boundaries.
14.5. Conditional Mixture Models
We have seen that standard decision trees are restricted by hard, axis-aligned splits of
the input space. These constraints can be relaxed, at the expense of interpretability,
by allowing soft, probabilistic splits that can be functions of all of the input variables,
not just one of them at a time. If we also give the leaf models a probabilistic inter-
pretation, we arrive at a fully probabilistic tree-based model called the hierarchical
mixture of experts, which we consider in Section 14.5.3.
An alternative way to motivate the hierarchical mixture of experts model is to
start with a standard probabilistic mixtures of unconditional density models such as
Gaussians and replace the component densities with conditional distributions. Here
Chapter 9
we consider mixtures of linear regression models (Section 14.5.1) and mixtures of


---
**Page 636**
14.5. Conditional Mixture Models
667
logistic regression models (Section 14.5.2). In the simplest case, the mixing coefﬁ-
cients are independent of the input variables. If we make a further generalization to
allow the mixing coefﬁcients also to depend on the inputs then we obtain a mixture
of experts model. Finally, if we allow each component in the mixture model to be
itself a mixture of experts model, then we obtain a hierarchical mixture of experts.
14.5.1
Mixtures of linear regression models
One of the many advantages of giving a probabilistic interpretation to the lin-
ear regression model is that it can then be used as a component in more complex
probabilistic models. This can be done, for instance, by viewing the conditional
distribution representing the linear regression model as a node in a directed prob-
abilistic graph. Here we consider a simple example corresponding to a mixture of
linear regression models, which represents a straightforward extension of the Gaus-
sian mixture model discussed in Section 9.2 to the case of conditional Gaussian
distributions.
We therefore consider K linear regression models, each governed by its own
weight parameter wk. In many applications, it will be appropriate to use a common
noise variance, governed by a precision parameter β, for all K components, and this
is the case we consider here. We will once again restrict attention to a single target
variable t, though the extension to multiple outputs is straightforward. If we denote
Exercise 14.12
the mixing coefﬁcients by πk, then the mixture distribution can be written
p(t|θ) =
K

k=1
πkN(t|wT
k φ, β−1)
(14.34)
where θ denotes the set of all adaptive parameters in the model, namely W = {wk},
π = {πk}, and β. The log likelihood function for this model, given a data set of
observations {φn, tn}, then takes the form
ln p(t|θ) =
N

n=1
ln
 K

k=1
πkN(tn|wT
k φn, β−1)

(14.35)
where t = (t1, . . . , tN)T denotes the vector of target variables.
In order to maximize this likelihood function, we can once again appeal to the
EM algorithm, which will turn out to be a simple extension of the EM algorithm for
unconditional Gaussian mixtures of Section 9.2. We can therefore build on our expe-
rience with the unconditional mixture and introduce a set Z = {zn} of binary latent
variables where znk ∈{0, 1} in which, for each data point n, all of the elements
k = 1, . . . , K are zero except for a single value of 1 indicating which component
of the mixture was responsible for generating that data point. The joint distribution
over latent and observed variables can be represented by the graphical model shown
in Figure 14.7.
The complete-data log likelihood function then takes the form
Exercise 14.13
ln p(t, Z|θ) =
N

n=1
K

k=1
znk ln

πkN(tn|wT
k φn, β−1)

.
(14.36)


---
**Page 637**
668
14. COMBINING MODELS
Figure 14.7
Probabilistic directed graph representing a mixture of
linear regression models, deﬁned by (14.35).
zn
tn
φn
N
W
β
π
The EM algorithm begins by ﬁrst choosing an initial value θold for the model param-
eters. In the E step, these parameter values are then used to evaluate the posterior
probabilities, or responsibilities, of each component k for every data point n given
by
γnk = E[znk] = p(k|φn, θold) =
πkN(tn|wT
k φn, β−1)

j πjN(tn|wT
j φn, β−1).
(14.37)
The responsibilities are then used to determine the expectation, with respect to the
posterior distribution p(Z|t, θold), of the complete-data log likelihood, which takes
the form
Q(θ, θold) = EZ [ln p(t, Z|θ)] =
N

n=1
K

k=1
γnk

ln πk + ln N(tn|wT
k φn, β−1)
.
In the M step, we maximize the function Q(θ, θold) with respect to θ, keeping the
γnk ﬁxed. For the optimization with respect to the mixing coefﬁcients πk we need
to take account of the constraint 
k πk = 1, which can be done with the aid of a
Lagrange multiplier, leading to an M-step re-estimation equation for πk in the form
Exercise 14.14
πk = 1
N
N

n=1
γnk.
(14.38)
Note that this has exactly the same form as the corresponding result for a simple
mixture of unconditional Gaussians given by (9.22).
Next consider the maximization with respect to the parameter vector wk of the
kth linear regression model. Substituting for the Gaussian distribution, we see that
the function Q(θ, θold), as a function of the parameter vector wk, takes the form
Q(θ, θold) =
N

n=1
γnk

−β
2

tn −wT
k φn
2
+ const
(14.39)
where the constant term includes the contributions from other weight vectors wj for
j ̸= k. Note that the quantity we are maximizing is similar to the (negative of the)
standard sum-of-squares error (3.12) for a single linear regression model, but with
the inclusion of the responsibilities γnk. This represents a weighted least squares


---
**Page 638**
14.5. Conditional Mixture Models
669
problem, in which the term corresponding to the nth data point carries a weighting
coefﬁcient given by βγnk, which could be interpreted as an effective precision for
each data point. We see that each component linear regression model in the mixture,
governed by its own parameter vector wk, is ﬁtted separately to the whole data set in
the M step, but with each data point n weighted by the responsibility γnk that model
k takes for that data point. Setting the derivative of (14.39) with respect to wk equal
to zero gives
0 =
N

n=1
γnk

tn −wT
k φn

φn
(14.40)
which we can write in matrix notation as
0 = ΦTRk(t −Φwk)
(14.41)
where Rk = diag(γnk) is a diagonal matrix of size N × N. Solving for wk, we
obtain
wk = 
ΦTRkΦ−1 ΦTRkt.
(14.42)
This represents a set of modiﬁed normal equations corresponding to the weighted
least squares problem, of the same form as (4.99) found in the context of logistic
regression. Note that after each E step, the matrix Rk will change and so we will
have to solve the normal equations afresh in the subsequent M step.
Finally, we maximize Q(θ, θold) with respect to β. Keeping only terms that
depend on β, the function Q(θ, θold) can be written
Q(θ, θold) =
N

n=1
K

k=1
γnk
1
2 ln β −β
2

tn −wT
k φn
2
.
(14.43)
Setting the derivative with respect to β equal to zero, and rearranging, we obtain the
M-step equation for β in the form
1
β = 1
N
N

n=1
K

k=1
γnk

tn −wT
k φn
2 .
(14.44)
In Figure 14.8, we illustrate this EM algorithm using the simple example of
ﬁtting a mixture of two straight lines to a data set having one input variable x and
one target variable t. The predictive density (14.34) is plotted in Figure 14.9 using
the converged parameter values obtained from the EM algorithm, corresponding to
the right-hand plot in Figure 14.8.
Also shown in this ﬁgure is the result of ﬁtting
a single linear regression model, which gives a unimodal predictive density. We see
that the mixture model gives a much better representation of the data distribution,
and this is reﬂected in the higher likelihood value. However, the mixture model
also assigns signiﬁcant probability mass to regions where there is no data because its
predictive distribution is bimodal for all values of x. This problem can be resolved by
extending the model to allow the mixture coefﬁcients themselves to be functions of
x, leading to models such as the mixture density networks discussed in Section 5.6,
and hierarchical mixture of experts discussed in Section 14.5.3.


---
**Page 639**
670
14. COMBINING MODELS
−1
−0.5
0
0.5
1
−1.5
−1
−0.5
0
0.5
1
1.5
−1
−0.5
0
0.5
1
−1.5
−1
−0.5
0
0.5
1
1.5
−1
−0.5
0
0.5
1
−1.5
−1
−0.5
0
0.5
1
1.5
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
Figure 14.8
Example of a synthetic data set, shown by the green points, having one input variable x and one
target variable t, together with a mixture of two linear regression models whose mean functions y(x, wk), where
k ∈{1, 2}, are shown by the blue and red lines. The upper three plots show the initial conﬁguration (left), the
result of running 30 iterations of EM (centre), and the result after 50 iterations of EM (right). Here β was initialized
to the reciprocal of the true variance of the set of target values. The lower three plots show the corresponding
responsibilities plotted as a vertical line for each data point in which the length of the blue segment gives the
posterior probability of the blue line for that data point (and similarly for the red segment).
14.5.2
Mixtures of logistic models
Because the logistic regression model deﬁnes a conditional distribution for the
target variable, given the input vector, it is straightforward to use it as the component
distribution in a mixture model, thereby giving rise to a richer family of conditional
distributions compared to a single logistic regression model. This example involves
a straightforward combination of ideas encountered in earlier sections of the book
and will help consolidate these for the reader.
The conditional distribution of the target variable, for a probabilistic mixture of
K logistic regression models, is given by
p(t|φ, θ) =
K

k=1
πkyt
k [1 −yk]1−t
(14.45)
where φ is the feature vector, yk = σ

wT
k φ

is the output of component k, and θ
denotes the adjustable parameters namely {πk} and {wk}.
Now suppose we are given a data set {φn, tn}. The corresponding likelihood


---
**Page 640**
14.5. Conditional Mixture Models
671
Figure 14.9
The left plot shows the predictive conditional density corresponding to the converged solution in
Figure 14.8. This gives a log likelihood value of −3.0. A vertical slice through one of these plots at a particular
value of x represents the corresponding conditional distribution p(t|x), which we see is bimodal. The plot on the
right shows the predictive density for a single linear regression model ﬁtted to the same data set using maximum
likelihood. This model has a smaller log likelihood of −27.6.
function is then given by
p(t|θ) =
N

n=1
 K

k=1
πkytn
nk [1 −ynk]1−tn

(14.46)
where ynk = σ(wT
k φn) and t = (t1, . . . , tN)T. We can maximize this likelihood
function iteratively by making use of the EM algorithm. This involves introducing
latent variables znk that correspond to a 1-of-K coded binary indicator variable for
each data point n. The complete-data likelihood function is then given by
p(t, Z|θ) =
N

n=1
K

k=1

πkytn
nk [1 −ynk]1−tnznk
(14.47)
where Z is the matrix of latent variables with elements znk. We initialize the EM
algorithm by choosing an initial value θold for the model parameters. In the E step,
we then use these parameter values to evaluate the posterior probabilities of the com-
ponents k for each data point n, which are given by
γnk = E[znk] = p(k|φn, θold) =
πkytn
nk [1 −ynk]1−tn

j πjytn
nj [1 −ynj]1−tn .
(14.48)
These responsibilities are then used to ﬁnd the expected complete-data log likelihood
as a function of θ, given by
Q(θ, θold) = EZ [ln p(t, Z|θ)]
=
N

n=1
K

k=1
γnk {ln πk + tn ln ynk + (1 −tn) ln (1 −ynk)} .
(14.49)


---
**Page 641**
672
14. COMBINING MODELS
The M step involves maximization of this function with respect to θ, keeping θold,
and hence γnk, ﬁxed. Maximization with respect to πk can be done in the usual way,
with a Lagrange multiplier to enforce the summation constraint 
k πk = 1, giving
the familiar result
πk = 1
N
N

n=1
γnk.
(14.50)
To determine the {wk}, we note that the Q(θ, θold) function comprises a sum
over terms indexed by k each of which depends only on one of the vectors wk, so
that the different vectors are decoupled in the M step of the EM algorithm. In other
words, the different components interact only via the responsibilities, which are ﬁxed
during the M step. Note that the M step does not have a closed-form solution and
must be solved iteratively using, for instance, the iterative reweighted least squares
(IRLS) algorithm. The gradient and the Hessian for the vector wk are given by
Section 4.3.3
∇kQ
=
N

n=1
γnk(tn −ynk)φn
(14.51)
Hk
=
−∇k∇kQ =
N

n=1
γnkynk(1 −ynk)φnφT
n
(14.52)
where ∇k denotes the gradient with respect to wk. For ﬁxed γnk, these are indepen-
dent of {wj} for j ̸= k and so we can solve for each wk separately using the IRLS
algorithm. Thus the M-step equations for component k correspond simply to ﬁtting
Section 4.3.3
a single logistic regression model to a weighted data set in which data point n carries
a weight γnk. Figure 14.10 shows an example of the mixture of logistic regression
models applied to a simple classiﬁcation problem. The extension of this model to a
mixture of softmax models for more than two classes is straightforward.
Exercise 14.16
14.5.3
Mixtures of experts
In Section 14.5.1, we considered a mixture of linear regression models, and in
Section 14.5.2 we discussed the analogous mixture of linear classiﬁers. Although
these simple mixtures extend the ﬂexibility of linear models to include more com-
plex (e.g., multimodal) predictive distributions, they are still very limited. We can
further increase the capability of such models by allowing the mixing coefﬁcients
themselves to be functions of the input variable, so that
p(t|x) =
K

k=1
πk(x)pk(t|x).
(14.53)
This is known as a mixture of experts model (Jacobs et al., 1991) in which the mix-
ing coefﬁcients πk(x) are known as gating functions and the individual component
densities pk(t|x) are called experts. The notion behind the terminology is that differ-
ent components can model the distribution in different regions of input space (they


---
**Page 642**
14.5. Conditional Mixture Models
673
Figure 14.10
Illustration of a mixture of logistic regression models. The left plot shows data points drawn
from two classes denoted red and blue, in which the background colour (which varies from pure red to pure blue)
denotes the true probability of the class label. The centre plot shows the result of ﬁtting a single logistic regression
model using maximum likelihood, in which the background colour denotes the corresponding probability of the
class label. Because the colour is a near-uniform purple, we see that the model assigns a probability of around
0.5 to each of the classes over most of input space. The right plot shows the result of ﬁtting a mixture of two
logistic regression models, which now gives much higher probability to the correct labels for many of the points
in the blue class.
are ‘experts’ at making predictions in their own regions), and the gating functions
determine which components are dominant in which region.
The gating functions πk(x) must satisfy the usual constraints for mixing co-
efﬁcients, namely 0 ⩽πk(x) ⩽1 and 
k πk(x) = 1. They can therefore be
represented, for example, by linear softmax models of the form (4.104) and (4.105).
If the experts are also linear (regression or classiﬁcation) models, then the whole
model can be ﬁtted efﬁciently using the EM algorithm, with iterative reweighted
least squares being employed in the M step (Jordan and Jacobs, 1994).
Such a model still has signiﬁcant limitations due to the use of linear models
for the gating and expert functions. A much more ﬂexible model is obtained by
using a multilevel gating function to give the hierarchical mixture of experts, or
HME model (Jordan and Jacobs, 1994). To understand the structure of this model,
imagine a mixture distribution in which each component in the mixture is itself a
mixture distribution. For simple unconditional mixtures, this hierarchical mixture is
trivially equivalent to a single ﬂat mixture distribution. However, when the mixing
Exercise 14.17
coefﬁcients are input dependent, this hierarchical model becomes nontrivial. The
HME model can also be viewed as a probabilistic version of decision trees discussed
in Section 14.4 and can again be trained efﬁciently by maximum likelihood using an
EM algorithm with IRLS in the M step. A Bayesian treatment of the HME has been
Section 4.3.3
given by Bishop and Svens´en (2003) based on variational inference.
We shall not discuss the HME in detail here. However, it is worth pointing out
the close connection with the mixture density network discussed in Section 5.6. The
principal advantage of the mixtures of experts model is that it can be optimized by
EM in which the M step for each mixture component and gating model involves
a convex optimization (although the overall optimization is nonconvex). By con-
trast, the advantage of the mixture density network approach is that the component


---
**Page 643**
674
14. COMBINING MODELS
densities and the mixing coefﬁcients share the hidden units of the neural network.
Furthermore, in the mixture density network, the splits of the input space are further
relaxed compared to the hierarchical mixture of experts in that they are not only soft,
and not constrained to be axis aligned, but they can also be nonlinear.
Exercises
14.1
(⋆⋆) www
Consider a set models of the form p(t|x, zh, θh, h) in which x is the
input vector, t is the target vector, h indexes the different models, zh is a latent vari-
able for model h, and θh is the set of parameters for model h. Suppose the models
have prior probabilities p(h) and that we are given a training set X = {x1, . . . , xN}
and T = {t1, . . . , tN}. Write down the formulae needed to evaluate the predic-
tive distribution p(t|x, X, T) in which the latent variables and the model index are
marginalized out. Use these formulae to highlight the difference between Bayesian
averaging of different models and the use of latent variables within a single model.
14.2
(⋆)
The expected sum-of-squares error EAV for a simple committee model can
be deﬁned by (14.10), and the expected error of the committee itself is given by
(14.11). Assuming that the individual errors satisfy (14.12) and (14.13), derive the
result (14.14).
14.3
(⋆) www
By making use of Jensen’s inequality (1.115), for the special case of
the convex function f(x) = x2, show that the average expected sum-of-squares
error EAV of the members of a simple committee model, given by (14.10), and the
expected error ECOM of the committee itself, given by (14.11), satisfy
ECOM ⩽EAV.
(14.54)
14.4
(⋆⋆)
By making use of Jensen’s in equality (1.115), show that the result (14.54)
derived in the previous exercise hods for any error function E(y), not just sum-of-
squares, provided it is a convex function of y.
14.5
(⋆⋆) www
Consider a committee in which we allow unequal weighting of the
constituent models, so that
yCOM(x) =
M

m=1
αmym(x).
(14.55)
In order to ensure that the predictions yCOM(x) remain within sensible limits, sup-
pose that we require that they be bounded at each value of x by the minimum and
maximum values given by any of the members of the committee, so that
ymin(x) ⩽yCOM(x) ⩽ymax(x).
(14.56)
Show that a necessary and sufﬁcient condition for this constraint is that the coefﬁ-
cients αm satisfy
αm ⩾0,
M

m=1
αm = 1.
(14.57)


---
**Page 644**
Exercises
675
14.6
(⋆) www
By differentiating the error function (14.23) with respect to αm, show
that the parameters αm in the AdaBoost algorithm are updated using (14.17) in
which ϵm is deﬁned by (14.16).
14.7
(⋆) By making a variational minimization of the expected exponential error function
given by (14.27) with respect to all possible functions y(x), show that the minimizing
function is given by (14.28).
14.8
(⋆)
Show that the exponential error function (14.20), which is minimized by the
AdaBoost algorithm, does not correspond to the log likelihood of any well-behaved
probabilistic model. This can be done by showing that the corresponding conditional
distribution p(t|x) cannot be correctly normalized.
14.9
(⋆) www
Show that the sequential minimization of the sum-of-squares error func-
tion for an additive model of the form (14.21) in the style of boosting simply involves
ﬁtting each new base classiﬁer to the residual errors tn−fm−1(xn) from the previous
model.
14.10
(⋆)
Verify that if we minimize the sum-of-squares error between a set of training
values {tn} and a single predictive value t, then the optimal solution for t is given
by the mean of the {tn}.
14.11
(⋆⋆)
Consider a data set comprising 400 data points from class C1 and 400 data
points from class C2. Suppose that a tree model A splits these into (300, 100) at
the ﬁrst leaf node and (100, 300) at the second leaf node, where (n, m) denotes that
n points are assigned to C1 and m points are assigned to C2. Similarly, suppose
that a second tree model B splits them into (200, 400) and (200, 0). Evaluate the
misclassiﬁcation rates for the two trees and hence show that they are equal. Similarly,
evaluate the cross-entropy (14.32) and Gini index (14.33) for the two trees and show
that they are both lower for tree B than for tree A.
14.12
(⋆⋆) Extend the results of Section 14.5.1 for a mixture of linear regression models
to the case of multiple target values described by a vector t. To do this, make use of
the results of Section 3.1.5.
14.13
(⋆) www
Verify that the complete-data log likelihood function for the mixture of
linear regression models is given by (14.36).
14.14
(⋆) Use the technique of Lagrange multipliers (Appendix E) to show that the M-step
re-estimation equation for the mixing coefﬁcients in the mixture of linear regression
models trained by maximum likelihood EM is given by (14.38).
14.15
(⋆) www
We have already noted that if we use a squared loss function in a regres-
sion problem, the corresponding optimal prediction of the target variable for a new
input vector is given by the conditional mean of the predictive distribution. Show
that the conditional mean for the mixture of linear regression models discussed in
Section 14.5.1 is given by a linear combination of the means of each component dis-
tribution. Note that if the conditional distribution of the target data is multimodal,
the conditional mean can give poor predictions.


---
**Page 645**
676
14. COMBINING MODELS
14.16
(⋆⋆⋆) Extend the logistic regression mixture model of Section 14.5.2 to a mixture
of softmax classiﬁers representing C ⩾2 classes. Write down the EM algorithm for
determining the parameters of this model through maximum likelihood.
14.17
(⋆⋆) www
Consider a mixture model for a conditional distribution p(t|x) of the
form
p(t|x) =
K

k=1
πkψk(t|x)
(14.58)
in which each mixture component ψk(t|x) is itself a mixture model. Show that this
two-level hierarchical mixture is equivalent to a conventional single-level mixture
model. Now suppose that the mixing coefﬁcients in both levels of such a hierar-
chical model are arbitrary functions of x. Again, show that this hierarchical model
is again equivalent to a single-level model with x-dependent mixing coefﬁcients.
Finally, consider the case in which the mixing coefﬁcients at both levels of the hi-
erarchical mixture are constrained to be linear classiﬁcation (logistic or softmax)
models. Show that the hierarchical mixture cannot in general be represented by a
single-level mixture having linear classiﬁcation models for the mixing coefﬁcients.
Hint: to do this it is sufﬁcient to construct a single counter-example, so consider a
mixture of two components in which one of those components is itself a mixture of
two components, with mixing coefﬁcients given by linear-logistic models. Show that
this cannot be represented by a single-level mixture of 3 components having mixing
coefﬁcients determined by a linear-softmax model.


---
**Page 646**
Appendix A. Data Sets
In this appendix, we give a brief introduction to the data sets used to illustrate some
of the algorithms described in this book. Detailed information on ﬁle formats for
these data sets, as well as the data ﬁles themselves, can be obtained from the book
web site:
http://research.microsoft.com/∼cmbishop/PRML
Handwritten Digits
The digits data used in this book is taken from the MNIST data set (LeCun et al.,
1998), which itself was constructed by modifying a subset of the much larger data
set produced by NIST (the National Institute of Standards and Technology). It com-
prises a training set of 60, 000 examples and a test set of 10, 000 examples. Some
of the data was collected from Census Bureau employees and the rest was collected
from high-school children, and care was taken to ensure that the test examples were
written by different individuals to the training examples.
The original NIST data had binary (black or white) pixels. To create MNIST,
these images were size normalized to ﬁt in a 20×20 pixel box while preserving their
aspect ratio. As a consequence of the anti-aliasing used to change the resolution of
the images, the resulting MNIST digits are grey scale. These images were then
centred in a 28 × 28 box. Examples of the MNIST digits are shown in Figure A.1.
Error rates for classifying the digits range from 12% for a simple linear classi-
ﬁer, through 0.56% for a carefully designed support vector machine, to 0.4% for a
convolutional neural network (LeCun et al., 1998).
677


---
**Page 647**
678
A. DATA SETS
Figure A.1
One hundred examples of the
MNIST digits chosen at ran-
dom from the training set.
Oil Flow
This is a synthetic data set that arose out of a project aimed at measuring nonin-
vasively the proportions of oil, water, and gas in North Sea oil transfer pipelines
(Bishop and James, 1993). It is based on the principle of dual-energy gamma densit-
ometry. The ideas is that if a narrow beam of gamma rays is passed through the pipe,
the attenuation in the intensity of the beam provides information about the density of
material along its path. Thus, for instance, the beam will be attenuated more strongly
by oil than by gas.
A single attenuation measurement alone is not sufﬁcient because there are two
degrees of freedom corresponding to the fraction of oil and the fraction of water (the
fraction of gas is redundant because the three fractions must add to one). To address
this, two gamma beams of different energies (in other words different frequencies or
wavelengths) are passed through the pipe along the same path, and the attenuation of
each is measured. Because the absorbtion properties of different materials vary dif-
ferently as a function of energy, measurement of the attenuations at the two energies
provides two independent pieces of information. Given the known absorbtion prop-
erties of oil, water, and gas at the two energies, it is then a simple matter to calculate
the average fractions of oil and water (and hence of gas) measured along the path of
the gamma beams.
There is a further complication, however, associated with the motion of the ma-
terials along the pipe. If the ﬂow velocity is small, then the oil ﬂoats on top of the
water with the gas sitting above the oil. This is known as a laminar or stratiﬁed


---
**Page 648**
A. DATA SETS
679
Figure A.2
The three geometrical conﬁgurations of the oil,
water, and gas phases used to generate the oil-
ﬂow data set. For each conﬁguration, the pro-
portions of the three phases can vary.
Mix
Gas
Water
Oil
Homogeneous
Stratiﬁed
Annular
ﬂow conﬁguration and is illustrated in Figure A.2. As the ﬂow velocity is increased,
more complex geometrical conﬁgurations of the oil, water, and gas can arise. For the
purposes of this data set, two speciﬁc idealizations are considered. In the annular
conﬁguration the oil, water, and gas form concentric cylinders with the water around
the outside and the gas in the centre, whereas in the homogeneous conﬁguration the
oil, water and gas are assumed to be intimately mixed as might occur at high ﬂow
velocities under turbulent conditions. These conﬁgurations are also illustrated in
Figure A.2.
We have seen that a single dual-energy beam gives the oil and water fractions
measured along the path length, whereas we are interested in the volume fractions of
oil and water. This can be addressed by using multiple dual-energy gamma densit-
ometers whose beams pass through different regions of the pipe. For this particular
data set, there are six such beams, and their spatial arrangement is shown in Fig-
ure A.3. A single observation is therefore represented by a 12-dimensional vector
comprising the fractions of oil and water measured along the paths of each of the
beams. We are, however, interested in obtaining the overall volume fractions of the
three phases in the pipe. This is much like the classical problem of tomographic re-
construction, used in medical imaging for example, in which a two-dimensional dis-
Figure A.3
Cross section of the pipe showing the arrangement of the
six beam lines, each of which comprises a single dual-
energy gamma densitometer. Note that the vertical beams
are asymmetrically arranged relative to the central axis
(shown by the dotted line).


---
**Page 649**
680
A. DATA SETS
tribution is to be reconstructed from an number of one-dimensional averages. Here
there are far fewer line measurements than in a typical tomography application. On
the other hand the range of geometrical conﬁgurations is much more limited, and so
the conﬁguration, as well as the phase fractions, can be predicted with reasonable
accuracy from the densitometer data.
For safety reasons, the intensity of the gamma beams is kept relatively weak and
so to obtain an accurate measurement of the attenuation, the measured beam intensity
is integrated over a speciﬁc time interval. For a ﬁnite integration time, there are
random ﬂuctuations in the measured intensity due to the fact that the gamma beams
comprise discrete packets of energy called photons. In practice, the integration time
is chosen as a compromise between reducing the noise level (which requires a long
integration time) and detecting temporal variations in the ﬂow (which requires a short
integration time). The oil ﬂow data set is generated using realistic known values for
the absorption properties of oil, water, and gas at the two gamma energies used, and
with a speciﬁc choice of integration time (10 seconds) chosen as characteristic of a
typical practical setup.
Each point in the data set is generated independently using the following steps:
1. Choose one of the three phase conﬁgurations at random with equal probability.
2. Choose three random numbers f1, f2 and f3 from the uniform distribution over
(0, 1) and deﬁne
foil =
f1
f1 + f2 + f3
,
fwater =
f2
f1 + f2 + f3
.
(A.1)
This treats the three phases on an equal footing and ensures that the volume
fractions add to one.
3. For each of the six beam lines, calculate the effective path lengths through oil
and water for the given phase conﬁguration.
4. Perturb the path lengths using the Poisson distribution based on the known
beam intensities and integration time to allow for the effect of photon statistics.
Each point in the data set comprises the 12 path length measurements, together
with the fractions of oil and water and a binary label describing the phase conﬁgu-
ration. The data set is divided into training, validation, and test sets, each of which
comprises 1, 000 independent data points. Details of the data format are available
from the book web site.
In Bishop and James (1993), statistical machine learning techniques were used
to predict the volume fractions and also the geometrical conﬁguration of the phases
shown in Figure A.2, from the 12-dimensional vector of measurements. The 12-
dimensional observation vectors can also be used to test data visualization algo-
rithms.
This data set has a rich and interesting structure, as follows. For any given
conﬁguration there are two degrees of freedom corresponding to the fractions of


---
**Page 650**
A. DATA SETS
681
oil and water, and so for inﬁnite integration time the data will locally live on a two-
dimensional manifold. For a ﬁnite integration time, the individual data points will be
perturbed away from the manifold by the photon noise. In the homogeneous phase
conﬁguration, the path lengths in oil and water are linearly related to the fractions of
oil and water, and so the data points lie close to a linear manifold. For the annular
conﬁguration, the relationship between phase fraction and path length is nonlinear
and so the manifold will be nonlinear. In the case of the laminar conﬁguration the
situation is even more complex because small variations in the phase fractions can
cause one of the horizontal phase boundaries to move across one of the horizontal
beam lines leading to a discontinuous jump in the 12-dimensional observation space.
In this way, the two-dimensional nonlinear manifold for the laminar conﬁguration is
broken into six distinct segments. Note also that some of the manifolds for different
phase conﬁgurations meet at speciﬁc points, for example if the pipe is ﬁlled entirely
with oil, it corresponds to speciﬁc instances of the laminar, annular, and homoge-
neous conﬁgurations.
Old Faithful
Old Faithful, shown in Figure A.4, is a hydrothermal geyser in Yellowstone National
Park in the state of Wyoming, U.S.A., and is a popular tourist attraction. Its name
stems from the supposed regularity of its eruptions.
The data set comprises 272 observations, each of which represents a single erup-
tion and contains two variables corresponding to the duration in minutes of the erup-
tion, and the time until the next eruption, also in minutes. Figure A.5 shows a plot of
the time to the next eruption versus the duration of the eruptions. It can be seen that
the time to the next eruption varies considerably, although knowledge of the duration
of the current eruption allows it to be predicted more accurately. Note that there exist
several other data sets relating to the eruptions of Old Faithful.
Figure A.4
The
Old
Faithful
geyser
in
Yellowstone
National
Park.
c⃝Bruce T. Gourley
www.brucegourley.com.


---
**Page 651**
682
A. DATA SETS
Figure A.5
Plot of the time to the next eruption
in minutes (vertical axis) versus the
duration of the eruption in minutes
(horizontal axis) for the Old Faithful
data set.
1
2
3
4
5
6
40
50
60
70
80
90
100
Synthetic Data
Throughout the book, we use two simple synthetic data sets to illustrate many of the
algorithms. The ﬁrst of these is a regression problem, based on the sinusoidal func-
tion, shown in Figure A.6. The input values {xn} are generated uniformly in range
(0, 1), and the corresponding target values {tn} are obtained by ﬁrst computing the
corresponding values of the function sin(2πx), and then adding random noise with
a Gaussian distribution having standard deviation 0.3. Various forms of this data set,
having different numbers of data points, are used in the book.
The second data set is a classiﬁcation problem having two classes, with equal
prior probabilities, and is shown in Figure A.7. The blue class is generated from a
single Gaussian while the red class comes from a mixture of two Gaussians. Be-
cause we know the class priors and the class-conditional densities, it is straightfor-
ward to evaluate and plot the true posterior probabilities as well as the minimum
misclassiﬁcation-rate decision boundary, as shown in Figure A.7.


---
**Page 652**
A. DATA SETS
683
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
Figure A.6
The left-hand plot shows the synthetic regression data set along with the underlying sinusoidal
function from which the data points were generated. The right-hand plot shows the true conditional distribution
p(t|x) from which the labels are generated, in which the green curve denotes the mean, and the shaded region
spans one standard deviation on each side of the mean.
−2
0
2
−2
0
2
Figure A.7
The left plot shows the synthetic classiﬁcation data set with data from the two classes shown in
red and blue. On the right is a plot of the true posterior probabilities, shown on a colour scale going from pure
red denoting probability of the red class is 1 to pure blue denoting probability of the red class is 0. Because
these probabilities are known, the optimal decision boundary for minimizing the misclassiﬁcation rate (which
corresponds to the contour along which the posterior probabilities for each class equal 0.5) can be evaluated
and is shown by the green curve. This decision boundary is also plotted on the left-hand ﬁgure.

