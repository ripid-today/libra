# 12 - Continuous Latent Variables
*Pages 559-604 from Pattern Recognition and Machine Learning*

---
**Page 559**
544
11. SAMPLING METHODS
To show that this procedure samples from the required distribution, we ﬁrst of
all note that the distribution p(z) is an invariant of each of the Gibbs sampling steps
individually and hence of the whole Markov chain. This follows from the fact that
when we sample from p(zi|{z\i), the marginal distribution p(z\i) is clearly invariant
because the value of z\i is unchanged. Also, each step by deﬁnition samples from the
correct conditional distribution p(zi|z\i). Because these conditional and marginal
distributions together specify the joint distribution, we see that the joint distribution
is itself invariant.
The second requirement to be satisﬁed in order that the Gibbs sampling proce-
dure samples from the correct distribution is that it be ergodic. A sufﬁcient condition
for ergodicity is that none of the conditional distributions be anywhere zero. If this
is the case, then any point in z space can be reached from any other point in a ﬁnite
number of steps involving one update of each of the component variables. If this
requirement is not satisﬁed, so that some of the conditional distributions have zeros,
then ergodicity, if it applies, must be proven explicitly.
The distribution of initial states must also be speciﬁed in order to complete the
algorithm, although samples drawn after many iterations will effectively become
independent of this distribution. Of course, successive samples from the Markov
chain will be highly correlated, and so to obtain samples that are nearly independent
it will be necessary to subsample the sequence.
We can obtain the Gibbs sampling procedure as a particular instance of the
Metropolis-Hastings algorithm as follows. Consider a Metropolis-Hastings sampling
step involving the variable zk in which the remaining variables z\k remain ﬁxed, and
for which the transition probability from z to z⋆is given by qk(z⋆|z) = p(z⋆
k|z\k).
We note that z⋆
\k = z\k because these components are unchanged by the sampling
step. Also, p(z) = p(zk|z\k)p(z\k). Thus the factor that determines the acceptance
probability in the Metropolis-Hastings (11.44) is given by
A(z⋆, z) = p(z⋆)qk(z|z⋆)
p(z)qk(z⋆|z) =
p(z⋆
k|z⋆
\k)p(z⋆
\k)p(zk|z⋆
\k)
p(zk|z\k)p(z\k)p(z⋆
k|z\k) = 1
(11.49)
where we have used z⋆
\k = z\k. Thus the Metropolis-Hastings steps are always
accepted.
As with the Metropolis algorithm, we can gain some insight into the behaviour of
Gibbs sampling by investigating its application to a Gaussian distribution. Consider
a correlated Gaussian in two variables, as illustrated in Figure 11.11, having con-
ditional distributions of width l and marginal distributions of width L. The typical
step size is governed by the conditional distributions and will be of order l. Because
the state evolves according to a random walk, the number of steps needed to obtain
independent samples from the distribution will be of order (L/l)2. Of course if the
Gaussian distribution were uncorrelated, then the Gibbs sampling procedure would
be optimally efﬁcient. For this simple problem, we could rotate the coordinate sys-
tem in order to decorrelate the variables. However, in practical applications it will
generally be infeasible to ﬁnd such transformations.
One approach to reducing random walk behaviour in Gibbs sampling is called
over-relaxation (Adler, 1981). In its original form, this applies to problems for which


---
**Page 560**
11.3. Gibbs Sampling
545
Figure 11.11
Illustration of Gibbs sampling by alter-
nate updates of two variables whose
distribution is a correlated Gaussian.
The step size is governed by the stan-
dard deviation of the conditional distri-
bution (green curve), and is O(l), lead-
ing to slow progress in the direction of
elongation of the joint distribution (red
ellipse). The number of steps needed
to obtain an independent sample from
the distribution is O((L/l)2).
z1
z2
L
l
the conditional distributions are Gaussian, which represents a more general class of
distributions than the multivariate Gaussian because, for example, the non-Gaussian
distribution p(z, y) ∝exp(−z2y2) has Gaussian conditional distributions. At each
step of the Gibbs sampling algorithm, the conditional distribution for a particular
component zi has some mean µi and some variance σ2
i . In the over-relaxation frame-
work, the value of zi is replaced with
z′
i = µi + α(zi −µi) + σi(1 −α2
i)1/2ν
(11.50)
where ν is a Gaussian random variable with zero mean and unit variance, and α
is a parameter such that −1 < α < 1. For α = 0, the method is equivalent to
standard Gibbs sampling, and for α < 0 the step is biased to the opposite side of the
mean. This step leaves the desired distribution invariant because if zi has mean µi
and variance σ2
i , then so too does z′
i. The effect of over-relaxation is to encourage
directed motion through state space when the variables are highly correlated. The
framework of ordered over-relaxation (Neal, 1999) generalizes this approach to non-
Gaussian distributions.
The practical applicability of Gibbs sampling depends on the ease with which
samples can be drawn from the conditional distributions p(zk|z\k). In the case of
probability distributions speciﬁed using graphical models, the conditional distribu-
tions for individual nodes depend only on the variables in the corresponding Markov
blankets, as illustrated in Figure 11.12. For directed graphs, a wide choice of condi-
tional distributions for the individual nodes conditioned on their parents will lead to
conditional distributions for Gibbs sampling that are log concave. The adaptive re-
jection sampling methods discussed in Section 11.1.3 therefore provide a framework
for Monte Carlo sampling from directed graphs with broad applicability.
If the graph is constructed using distributions from the exponential family, and
if the parent-child relationships preserve conjugacy, then the full conditional distri-
butions arising in Gibbs sampling will have the same functional form as the orig-


---
**Page 561**
546
11. SAMPLING METHODS
Figure 11.12
The Gibbs sampling method requires samples
to be drawn from the conditional distribution of a variable condi-
tioned on the remaining variables. For graphical models, this
conditional distribution is a function only of the states of the
nodes in the Markov blanket. For an undirected graph this com-
prises the set of neighbours, as shown on the left, while for a
directed graph the Markov blanket comprises the parents, the
children, and the co-parents, as shown on the right.
inal conditional distributions (conditioned on the parents) deﬁning each node, and
so standard sampling techniques can be employed. In general, the full conditional
distributions will be of a complex form that does not permit the use of standard sam-
pling algorithms. However, if these conditionals are log concave, then sampling can
be done efﬁciently using adaptive rejection sampling (assuming the corresponding
variable is a scalar).
If, at each stage of the Gibbs sampling algorithm, instead of drawing a sample
from the corresponding conditional distribution, we make a point estimate of the
variable given by the maximum of the conditional distribution, then we obtain the
iterated conditional modes (ICM) algorithm discussed in Section 8.3.3. Thus ICM
can be seen as a greedy approximation to Gibbs sampling.
Because the basic Gibbs sampling technique considers one variable at a time,
there are strong dependencies between successive samples. At the opposite extreme,
if we could draw samples directly from the joint distribution (an operation that we
are supposing is intractable), then successive samples would be independent. We can
hope to improve on the simple Gibbs sampler by adopting an intermediate strategy in
which we sample successively from groups of variables rather than individual vari-
ables. This is achieved in the blocking Gibbs sampling algorithm by choosing blocks
of variables, not necessarily disjoint, and then sampling jointly from the variables in
each block in turn, conditioned on the remaining variables (Jensen et al., 1995).
11.4. Slice Sampling
We have seen that one of the difﬁculties with the Metropolis algorithm is the sensi-
tivity to step size. If this is too small, the result is slow decorrelation due to random
walk behaviour, whereas if it is too large the result is inefﬁciency due to a high rejec-
tion rate. The technique of slice sampling (Neal, 2003) provides an adaptive step size
that is automatically adjusted to match the characteristics of the distribution. Again
it requires that we are able to evaluate the unnormalized distribution p(z).
Consider ﬁrst the univariate case. Slice sampling involves augmenting z with
an additional variable u and then drawing samples from the joint (z, u) space. We
shall see another example of this approach when we discuss hybrid Monte Carlo in
Section 11.5. The goal is to sample uniformly from the area under the distribution


---
**Page 562**
11.4. Slice Sampling
547
˜p(z)
z(τ)
z
u
(a)
˜p(z)
z(τ)
z
u
zmin
zmax
(b)
Figure 11.13
Illustration of slice sampling. (a) For a given value z(τ), a value of u is chosen uniformly in
the region 0 ⩽u ⩽ep(z(τ)), which then deﬁnes a ‘slice’ through the distribution, shown by the solid horizontal
lines. (b) Because it is infeasible to sample directly from a slice, a new sample of z is drawn from a region
zmin ⩽z ⩽zmax, which contains the previous value z(τ).
given by
p(z, u) =
1/Zp
if 0 ⩽u ⩽p(z)
0
otherwise
(11.51)
where Zp =  p(z) dz. The marginal distribution over z is given by

p(z, u) du =
 ep(z)
0
1
Zp
du = p(z)
Zp
= p(z)
(11.52)
and so we can sample from p(z) by sampling from p(z, u) and then ignoring the u
values. This can be achieved by alternately sampling z and u. Given the value of z
we evaluate p(z) and then sample u uniformly in the range 0 ⩽u ⩽p(z), which is
straightforward. Then we ﬁx u and sample z uniformly from the ‘slice’ through the
distribution deﬁned by {z : p(z) > u}. This is illustrated in Figure 11.13(a).
In practice, it can be difﬁcult to sample directly from a slice through the distribu-
tion and so instead we deﬁne a sampling scheme that leaves the uniform distribution
under p(z, u) invariant, which can be achieved by ensuring that detailed balance is
satisﬁed. Suppose the current value of z is denoted z(τ) and that we have obtained
a corresponding sample u. The next value of z is obtained by considering a region
zmin ⩽z ⩽zmax that contains z(τ). It is in the choice of this region that the adap-
tation to the characteristic length scales of the distribution takes place. We want the
region to encompass as much of the slice as possible so as to allow large moves in z
space while having as little as possible of this region lying outside the slice, because
this makes the sampling less efﬁcient.
One approach to the choice of region involves starting with a region containing
z(τ) having some width w and then testing each of the end points to see if they lie
within the slice. If either end point does not, then the region is extended in that
direction by increments of value w until the end point lies outside the region. A
candidate value z′ is then chosen uniformly from this region, and if it lies within the
slice, then it forms z(τ+1). If it lies outside the slice, then the region is shrunk such
that z′ forms an end point and such that the region still contains z(τ). Then another


---
**Page 563**
548
11. SAMPLING METHODS
candidate point is drawn uniformly from this reduced region and so on, until a value
of z is found that lies within the slice.
Slice sampling can be applied to multivariate distributions by repeatedly sam-
pling each variable in turn, in the manner of Gibbs sampling. This requires that
we are able to compute, for each component zi, a function that is proportional to
p(zi|z\i).
11.5. The Hybrid Monte Carlo Algorithm
As we have already noted, one of the major limitations of the Metropolis algorithm
is that it can exhibit random walk behaviour whereby the distance traversed through
the state space grows only as the square root of the number of steps. The problem
cannot be resolved simply by taking bigger steps as this leads to a high rejection rate.
In this section, we introduce a more sophisticated class of transitions based on an
analogy with physical systems and that has the property of being able to make large
changes to the system state while keeping the rejection probability small. It is ap-
plicable to distributions over continuous variables for which we can readily evaluate
the gradient of the log probability with respect to the state variables. We will discuss
the dynamical systems framework in Section 11.5.1, and then in Section 11.5.2 we
explain how this may be combined with the Metropolis algorithm to yield the pow-
erful hybrid Monte Carlo algorithm. A background in physics is not required as this
section is self-contained and the key results are all derived from ﬁrst principles.
11.5.1
Dynamical systems
The dynamical approach to stochastic sampling has its origins in algorithms for
simulating the behaviour of physical systems evolving under Hamiltonian dynam-
ics. In a Markov chain Monte Carlo simulation, the goal is to sample from a given
probability distribution p(z). The framework of Hamiltonian dynamics is exploited
by casting the probabilistic simulation in the form of a Hamiltonian system. In order
to remain in keeping with the literature in this area, we make use of the relevant
dynamical systems terminology where appropriate, which will be deﬁned as we go
along.
The dynamics that we consider corresponds to the evolution of the state variable
z = {zi} under continuous time, which we denote by τ. Classical dynamics is de-
scribed by Newton’s second law of motion in which the acceleration of an object is
proportional to the applied force, corresponding to a second-order differential equa-
tion over time. We can decompose a second-order equation into two coupled ﬁrst-
order equations by introducing intermediate momentum variables r, corresponding
to the rate of change of the state variables z, having components
ri = dzi
dτ
(11.53)
where the zi can be regarded as position variables in this dynamics perspective. Thus


---
**Page 564**
11.5. The Hybrid Monte Carlo Algorithm
549
for each position variable there is a corresponding momentum variable, and the joint
space of position and momentum variables is called phase space.
Without loss of generality, we can write the probability distribution p(z) in the
form
p(z) = 1
Zp
exp (−E(z))
(11.54)
where E(z) is interpreted as the potential energy of the system when in state z. The
system acceleration is the rate of change of momentum and is given by the applied
force, which itself is the negative gradient of the potential energy
dri
dτ = −∂E(z)
∂zi
.
(11.55)
It is convenient to reformulate this dynamical system using the Hamiltonian
framework. To do this, we ﬁrst deﬁne the kinetic energy by
K(r) = 1
2∥r∥2 = 1
2

i
r2
i .
(11.56)
The total energy of the system is then the sum of its potential and kinetic energies
H(z, r) = E(z) + K(r)
(11.57)
where H is the Hamiltonian function. Using (11.53), (11.55), (11.56), and (11.57),
we can now express the dynamics of the system in terms of the Hamiltonian equa-
tions given by
Exercise 11.15
dzi
dτ
=
∂H
∂ri
(11.58)
dri
dτ
=
−∂H
∂zi
.
(11.59)
William Hamilton
1805–1865
William Rowan Hamilton was an
Irish mathematician and physicist,
and child prodigy, who was ap-
pointed Professor of Astronomy at
Trinity College, Dublin, in 1827, be-
fore he had even graduated. One
of Hamilton’s most important contributions was a new
formulation of dynamics, which played a signiﬁcant
role in the later development of quantum mechanics.
His other great achievement was the development of
quaternions, which generalize the concept of complex
numbers by introducing three distinct square roots of
minus one, which satisfy i2 = j2 = k2 = ijk = −1.
It is said that these equations occurred to him while
walking along the Royal Canal in Dublin with his wife,
on 16 October 1843, and he promptly carved the
equations into the side of Broome bridge. Although
there is no longer any evidence of the carving, there is
now a stone plaque on the bridge commemorating the
discovery and displaying the quaternion equations.


---
**Page 565**
550
11. SAMPLING METHODS
During the evolution of this dynamical system, the value of the Hamiltonian H is
constant, as is easily seen by differentiation
dH
dτ
=

i
∂H
∂zi
dzi
dτ + ∂H
∂ri
dri
dτ

=

i
∂H
∂zi
∂H
∂ri
−∂H
∂ri
∂H
∂zi

= 0.
(11.60)
A second important property of Hamiltonian dynamical systems, known as Li-
ouville’s Theorem, is that they preserve volume in phase space. In other words, if
we consider a region within the space of variables (z, r), then as this region evolves
under the equations of Hamiltonian dynamics, its shape may change but its volume
will not. This can be seen by noting that the ﬂow ﬁeld (rate of change of location in
phase space) is given by
V =
 dz
dτ , dr
dτ

(11.61)
and that the divergence of this ﬁeld vanishes
div V
=

i
 ∂
∂zi
dzi
dτ + ∂
∂ri
dri
dτ

=

i

−∂
∂zi
∂H
∂ri
+ ∂
∂ri
∂H
∂zi

= 0.
(11.62)
Now consider the joint distribution over phase space whose total energy is the
Hamiltonian, i.e., the distribution given by
p(z, r) =
1
ZH
exp(−H(z, r)).
(11.63)
Using the two results of conservation of volume and conservation of H, it follows
that the Hamiltonian dynamics will leave p(z, r) invariant. This can be seen by
considering a small region of phase space over which H is approximately constant.
If we follow the evolution of the Hamiltonian equations for a ﬁnite time, then the
volume of this region will remain unchanged as will the value of H in this region, and
hence the probability density, which is a function only of H, will also be unchanged.
Although H is invariant, the values of z and r will vary, and so by integrating
the Hamiltonian dynamics over a ﬁnite time duration it becomes possible to make
large changes to z in a systematic way that avoids random walk behaviour.
Evolution under the Hamiltonian dynamics will not, however, sample ergodi-
cally from p(z, r) because the value of H is constant. In order to arrive at an ergodic
sampling scheme, we can introduce additional moves in phase space that change
the value of H while also leaving the distribution p(z, r) invariant. The simplest
way to achieve this is to replace the value of r with one drawn from its distribution
conditioned on z. This can be regarded as a Gibbs sampling step, and hence from


---
**Page 566**
11.5. The Hybrid Monte Carlo Algorithm
551
Section 11.3 we see that this also leaves the desired distribution invariant. Noting
that z and r are independent in the distribution p(z, r), we see that the conditional
distribution p(r|z) is a Gaussian from which it is straightforward to sample.
Exercise 11.16
In a practical application of this approach, we have to address the problem of
performing a numerical integration of the Hamiltonian equations. This will neces-
sarily introduce numerical errors and so we should devise a scheme that minimizes
the impact of such errors. In fact, it turns out that integration schemes can be devised
for which Liouville’s theorem still holds exactly. This property will be important in
the hybrid Monte Carlo algorithm, which is discussed in Section 11.5.2. One scheme
for achieving this is called the leapfrog discretization and involves alternately updat-
ing discrete-time approximations z and r to the position and momentum variables
using
ri(τ + ϵ/2)
=
ri(τ) −ϵ
2
∂E
∂zi
(z(τ))
(11.64)
zi(τ + ϵ)
=
zi(τ) + ϵri(τ + ϵ/2)
(11.65)
ri(τ + ϵ)
=
ri(τ + ϵ/2) −ϵ
2
∂E
∂zi
(z(τ + ϵ)).
(11.66)
We see that this takes the form of a half-step update of the momentum variables with
step size ϵ/2, followed by a full-step update of the position variables with step size ϵ,
followed by a second half-step update of the momentum variables. If several leapfrog
steps are applied in succession, it can be seen that half-step updates to the momentum
variables can be combined into full-step updates with step size ϵ. The successive
updates to position and momentum variables then leapfrog over each other. In order
to advance the dynamics by a time interval τ, we need to take τ/ϵ steps. The error
involved in the discretized approximation to the continuous time dynamics will go to
zero, assuming a smooth function E(z), in the limit ϵ →0. However, for a nonzero
ϵ as used in practice, some residual error will remain. We shall see in Section 11.5.2
how the effects of such errors can be eliminated in the hybrid Monte Carlo algorithm.
In summary then, the Hamiltonian dynamical approach involves alternating be-
tween a series of leapfrog updates and a resampling of the momentum variables from
their marginal distribution.
Note that the Hamiltonian dynamics method, unlike the basic Metropolis algo-
rithm, is able to make use of information about the gradient of the log probability
distribution as well as about the distribution itself. An analogous situation is familiar
from the domain of function optimization. In most cases where gradient informa-
tion is available, it is highly advantageous to make use of it. Informally, this follows
from the fact that in a space of dimension D, the additional computational cost of
evaluating a gradient compared with evaluating the function itself will typically be a
ﬁxed factor independent of D, whereas the D-dimensional gradient vector conveys
D pieces of information compared with the one piece of information given by the
function itself.


---
**Page 567**
552
11. SAMPLING METHODS
11.5.2
Hybrid Monte Carlo
As we discussed in the previous section, for a nonzero step size ϵ, the discretiza-
tion of the leapfrog algorithm will introduce errors into the integration of the Hamil-
tonian dynamical equations. Hybrid Monte Carlo (Duane et al., 1987; Neal, 1996)
combines Hamiltonian dynamics with the Metropolis algorithm and thereby removes
any bias associated with the discretization.
Speciﬁcally, the algorithm uses a Markov chain consisting of alternate stochastic
updates of the momentum variable r and Hamiltonian dynamical updates using the
leapfrog algorithm. After each application of the leapfrog algorithm, the resulting
candidate state is accepted or rejected according to the Metropolis criterion based
on the value of the Hamiltonian H. Thus if (z, r) is the initial state and (z⋆, r⋆)
is the state after the leapfrog integration, then this candidate state is accepted with
probability
min (1, exp{H(z, r) −H(z⋆, r⋆)}) .
(11.67)
If the leapfrog integration were to simulate the Hamiltonian dynamics perfectly,
then every such candidate step would automatically be accepted because the value
of H would be unchanged. Due to numerical errors, the value of H may sometimes
decrease, and we would like the Metropolis criterion to remove any bias due to this
effect and ensure that the resulting samples are indeed drawn from the required dis-
tribution. In order for this to be the case, we need to ensure that the update equations
corresponding to the leapfrog integration satisfy detailed balance (11.40). This is
easily achieved by modifying the leapfrog scheme as follows.
Before the start of each leapfrog integration sequence, we choose at random,
with equal probability, whether to integrate forwards in time (using step size ϵ) or
backwards in time (using step size −ϵ). We ﬁrst note that the leapfrog integration
scheme (11.64), (11.65), and (11.66) is time-reversible, so that integration for L steps
using step size −ϵ will exactly undo the effect of integration for L steps using step
size ϵ. Next we show that the leapfrog integration preserves phase-space volume
exactly. This follows from the fact that each step in the leapfrog scheme updates
either a zi variable or an ri variable by an amount that is a function only of the other
variable. As shown in Figure 11.14, this has the effect of shearing a region of phase
space while not altering its volume.
Finally, we use these results to show that detailed balance holds. Consider a
small region R of phase space that, under a sequence of L leapfrog iterations of
step size ϵ, maps to a region R′. Using conservation of volume under the leapfrog
iteration, we see that if R has volume δV then so too will R′. If we choose an initial
point from the distribution (11.63) and then update it using L leapfrog interactions,
the probability of the transition going from R to R′ is given by
1
ZH
exp(−H(R))δV 1
2 min {1, exp(−H(R) + H(R′))} .
(11.68)
where the factor of 1/2 arises from the probability of choosing to integrate with a
positive step size rather than a negative one. Similarly, the probability of starting in


---
**Page 568**
11.5. The Hybrid Monte Carlo Algorithm
553
ri
zi
r′
i
z′
i
Figure 11.14
Each step of the leapfrog algorithm (11.64)–(11.66) modiﬁes either a position variable zi or a
momentum variable ri. Because the change to one variable is a function only of the other, any region in phase
space will be sheared without change of volume.
region R′ and integrating backwards in time to end up in region R is given by
1
ZH
exp(−H(R′))δV 1
2 min {1, exp(−H(R′) + H(R))} .
(11.69)
It is easily seen that the two probabilities (11.68) and (11.69) are equal, and hence
detailed balance holds. Note that this proof ignores any overlap between the regions
Exercise 11.17
R and R′ but is easily generalized to allow for such overlap.
It is not difﬁcult to construct examples for which the leapfrog algorithm returns
to its starting position after a ﬁnite number of iterations. In such cases, the random
replacement of the momentum values before each leapfrog integration will not be
sufﬁcient to ensure ergodicity because the position variables will never be updated.
Such phenomena are easily avoided by choosing the magnitude of the step size at
random from some small interval, before each leapfrog integration.
We can gain some insight into the behaviour of the hybrid Monte Carlo algo-
rithm by considering its application to a multivariate Gaussian. For convenience,
consider a Gaussian distribution p(z) with independent components, for which the
Hamiltonian is given by
H(z, r) = 1
2

i
1
σ2
i
z2
i + 1
2

i
r2
i .
(11.70)
Our conclusions will be equally valid for a Gaussian distribution having correlated
components because the hybrid Monte Carlo algorithm exhibits rotational isotropy.
During the leapfrog integration, each pair of phase-space variables zi, ri evolves in-
dependently. However, the acceptance or rejection of the candidate point is based
on the value of H, which depends on the values of all of the variables. Thus, a
signiﬁcant integration error in any one of the variables could lead to a high prob-
ability of rejection. In order that the discrete leapfrog integration be a reasonably


---
**Page 569**
554
11. SAMPLING METHODS
good approximation to the true continuous-time dynamics, it is necessary for the
leapfrog integration scale ϵ to be smaller than the shortest length-scale over which
the potential is varying signiﬁcantly. This is governed by the smallest value of σi,
which we denote by σmin. Recall that the goal of the leapfrog integration in hybrid
Monte Carlo is to move a substantial distance through phase space to a new state
that is relatively independent of the initial state and still achieve a high probability of
acceptance. In order to achieve this, the leapfrog integration must be continued for a
number of iterations of order σmax/σmin.
By contrast, consider the behaviour of a simple Metropolis algorithm with an
isotropic Gaussian proposal distribution of variance s2, considered earlier. In order
to avoid high rejection rates, the value of s must be of order σmin. The exploration of
state space then proceeds by a random walk and takes of order (σmax/σmin)2 steps
to arrive at a roughly independent state.
11.6. Estimating the Partition Function
As we have seen, most of the sampling algorithms considered in this chapter re-
quire only the functional form of the probability distribution up to a multiplicative
constant. Thus if we write
pE(z) =
1
ZE
exp(−E(z))
(11.71)
then the value of the normalization constant ZE, also known as the partition func-
tion, is not needed in order to draw samples from p(z). However, knowledge of the
value of ZE can be useful for Bayesian model comparison since it represents the
model evidence (i.e., the probability of the observed data given the model), and so
it is of interest to consider how its value might be obtained. We assume that direct
evaluation by summing, or integrating, the function exp(−E(z)) over the state space
of z is intractable.
For model comparison, it is actually the ratio of the partition functions for two
models that is required. Multiplication of this ratio by the ratio of prior probabilities
gives the ratio of posterior probabilities, which can then be used for model selection
or model averaging.
One way to estimate a ratio of partition functions is to use importance sampling
from a distribution with energy function G(z)
ZE
ZG
=

z exp(−E(z))

z exp(−G(z))
=

z exp(−E(z) + G(z)) exp(−G(z))

z exp(−G(z))
=
EG(z)[exp(−E + G)]
≃

l
exp(−E(z(l)) + G(z(l)))
(11.72)


---
**Page 570**
11.6. Estimating the Partition Function
555
where {z(l)} are samples drawn from the distribution deﬁned by pG(z). If the dis-
tribution pG is one for which the partition function can be evaluated analytically, for
example a Gaussian, then the absolute value of ZE can be obtained.
This approach will only yield accurate results if the importance sampling distri-
bution pG is closely matched to the distribution pE, so that the ratio pE/pG does not
have wide variations. In practice, suitable analytically speciﬁed importance sampling
distributions cannot readily be found for the kinds of complex models considered in
this book.
An alternative approach is therefore to use the samples obtained from a Markov
chain to deﬁne the importance-sampling distribution. If the transition probability for
the Markov chain is given by T(z, z′), and the sample set is given by z(1), . . . , z(L),
then the sampling distribution can be written as
1
ZG
exp (−G(z)) =
L

l=1
T(z(l), z)
(11.73)
which can be used directly in (11.72).
Methods for estimating the ratio of two partition functions require for their suc-
cess that the two corresponding distributions be reasonably closely matched. This is
especially problematic if we wish to ﬁnd the absolute value of the partition function
for a complex distribution because it is only for relatively simple distributions that
the partition function can be evaluated directly, and so attempting to estimate the
ratio of partition functions directly is unlikely to be successful. This problem can be
tackled using a technique known as chaining (Neal, 1993; Barber and Bishop, 1997),
which involves introducing a succession of intermediate distributions p2, . . . , pM−1
that interpolate between a simple distribution p1(z) for which we can evaluate the
normalization coefﬁcient Z1 and the desired complex distribution pM(z). We then
have
ZM
Z1
= Z2
Z1
Z3
Z2
· · · ZM
ZM−1
(11.74)
in which the intermediate ratios can be determined using Monte Carlo methods as
discussed above. One way to construct such a sequence of intermediate systems
is to use an energy function containing a continuous parameter 0 ⩽α ⩽1 that
interpolates between the two distributions
Eα(z) = (1 −α)E1(z) + αEM(z).
(11.75)
If the intermediate ratios in (11.74) are to be found using Monte Carlo, it may be
more efﬁcient to use a single Markov chain run than to restart the Markov chain for
each ratio. In this case, the Markov chain is run initially for the system p1 and then
after some suitable number of steps moves on to the next distribution in the sequence.
Note, however, that the system must remain close to the equilibrium distribution at
each stage.


---
**Page 571**
556
11. SAMPLING METHODS
Exercises
11.1
(⋆) www
Show that the ﬁnite sample estimator f deﬁned by (11.2) has mean
equal to E[f] and variance given by (11.3).
11.2
(⋆)
Suppose that z is a random variable with uniform distribution over (0, 1) and
that we transform z using y = h−1(z) where h(y) is given by (11.6). Show that y
has the distribution p(y).
11.3
(⋆) Given a random variable z that is uniformly distributed over (0, 1), ﬁnd a trans-
formation y = f(z) such that y has a Cauchy distribution given by (11.8).
11.4
(⋆⋆)
Suppose that z1 and z2 are uniformly distributed over the unit circle, as
shown in Figure 11.3, and that we make the change of variables given by (11.10)
and (11.11). Show that (y1, y2) will be distributed according to (11.12).
11.5
(⋆) www
Let z be a D-dimensional random variable having a Gaussian distribu-
tion with zero mean and unit covariance matrix, and suppose that the positive deﬁnite
symmetric matrix Σ has the Cholesky decomposition Σ = LLT where L is a lower-
triangular matrix (i.e., one with zeros above the leading diagonal). Show that the
variable y = µ + Lz has a Gaussian distribution with mean µ and covariance Σ.
This provides a technique for generating samples from a general multivariate Gaus-
sian using samples from a univariate Gaussian having zero mean and unit variance.
11.6
(⋆⋆) www
In this exercise, we show more carefully that rejection sampling does
indeed draw samples from the desired distribution p(z). Suppose the proposal dis-
tribution is q(z) and show that the probability of a sample value z being accepted is
given by p(z)/kq(z) where p is any unnormalized distribution that is proportional to
p(z), and the constant k is set to the smallest value that ensures kq(z) ⩾p(z) for all
values of z. Note that the probability of drawing a value z is given by the probability
of drawing that value from q(z) times the probability of accepting that value given
that it has been drawn. Make use of this, along with the sum and product rules of
probability, to write down the normalized form for the distribution over z, and show
that it equals p(z).
11.7
(⋆) Suppose that z has a uniform distribution over the interval [0, 1]. Show that the
variable y = b tan z + c has a Cauchy distribution given by (11.16).
11.8
(⋆⋆)
Determine expressions for the coefﬁcients ki in the envelope distribution
(11.17) for adaptive rejection sampling using the requirements of continuity and nor-
malization.
11.9
(⋆⋆)
By making use of the technique discussed in Section 11.1.1 for sampling
from a single exponential distribution, devise an algorithm for sampling from the
piecewise exponential distribution deﬁned by (11.17).
11.10
(⋆) Show that the simple random walk over the integers deﬁned by (11.34), (11.35),
and (11.36) has the property that E[(z(τ))2] = E[(z(τ−1))2] + 1/2 and hence by
induction that E[(z(τ))2] = τ/2.


---
**Page 572**
Exercises
557
Figure 11.15
A probability distribution over two variables z1
and z2 that is uniform over the shaded regions
and that is zero everywhere else.
z1
z2
11.11
(⋆⋆) www
Show that the Gibbs sampling algorithm, discussed in Section 11.3,
satisﬁes detailed balance as deﬁned by (11.40).
11.12
(⋆) Consider the distribution shown in Figure 11.15. Discuss whether the standard
Gibbs sampling procedure for this distribution is ergodic, and therefore whether it
would sample correctly from this distribution
11.13
(⋆⋆) Consider the simple 3-node graph shown in Figure 11.16 in which the observed
node x is given by a Gaussian distribution N(x|µ, τ −1) with mean µ and precision
τ. Suppose that the marginal distributions over the mean and precision are given
by N(µ|µ0, s0) and Gam(τ|a, b), where Gam(·|·, ·) denotes a gamma distribution.
Write down expressions for the conditional distributions p(µ|x, τ) and p(τ|x, µ) that
would be required in order to apply Gibbs sampling to the posterior distribution
p(µ, τ|x).
11.14
(⋆)
Verify that the over-relaxation update (11.50), in which zi has mean µi and
variance σi, and where ν has zero mean and unit variance, gives a value z′
i with
mean µi and variance σ2
i .
11.15
(⋆) www
Using (11.56) and (11.57), show that the Hamiltonian equation (11.58)
is equivalent to (11.53). Similarly, using (11.57) show that (11.59) is equivalent to
(11.55).
11.16
(⋆)
By making use of (11.56), (11.57), and (11.63), show that the conditional dis-
tribution p(r|z) is a Gaussian.
Figure 11.16
A graph involving an observed Gaussian variable x with
prior distributions over its mean µ and precision τ.
µ
τ
x


---
**Page 573**
558
11. SAMPLING METHODS
11.17
(⋆) www
Verify that the two probabilities (11.68) and (11.69) are equal, and hence
that detailed balance holds for the hybrid Monte Carlo algorithm.


---
**Page 574**
13
Sequential
Data
So far in this book, we have focussed primarily on sets of data points that were as-
sumed to be independent and identically distributed (i.i.d.). This assumption allowed
us to express the likelihood function as the product over all data points of the prob-
ability distribution evaluated at each data point. For many applications, however,
the i.i.d. assumption will be a poor one. Here we consider a particularly important
class of such data sets, namely those that describe sequential data. These often arise
through measurement of time series, for example the rainfall measurements on suc-
cessive days at a particular location, or the daily values of a currency exchange rate,
or the acoustic features at successive time frames used for speech recognition. An
example involving speech data is shown in Figure 13.1. Sequential data can also
arise in contexts other than time series, for example the sequence of nucleotide base
pairs along a strand of DNA or the sequence of characters in an English sentence.
For convenience, we shall sometimes refer to ‘past’ and ‘future’ observations in a
sequence. However, the models explored in this chapter are equally applicable to all
605


---
**Page 575**
606
13. SEQUENTIAL DATA
Figure 13.1
Example
of
a
spectro-
gram of the spoken words “Bayes’ theo-
rem” showing a plot of the intensity of the
spectral coefﬁcients versus time index.
forms of sequential data, not just temporal sequences.
It is useful to distinguish between stationary and nonstationary sequential dis-
tributions. In the stationary case, the data evolves in time, but the distribution from
which it is generated remains the same. For the more complex nonstationary situa-
tion, the generative distribution itself is evolving with time. Here we shall focus on
the stationary case.
For many applications, such as ﬁnancial forecasting, we wish to be able to pre-
dict the next value in a time series given observations of the previous values. In-
tuitively, we expect that recent observations are likely to be more informative than
more historical observations in predicting future values. The example in Figure 13.1
shows that successive observations of the speech spectrum are indeed highly cor-
related. Furthermore, it would be impractical to consider a general dependence of
future observations on all previous observations because the complexity of such a
model would grow without limit as the number of observations increases. This leads
us to consider Markov models in which we assume that future predictions are inde-


---
**Page 576**
13.1. Markov Models
607
Figure 13.2
The
simplest
approach
to
modelling a sequence of ob-
servations is to treat them
as independent, correspond-
ing to a graph without links.
x1
x2
x3
x4
pendent of all but the most recent observations.
Although such models are tractable, they are also severely limited. We can ob-
tain a more general framework, while still retaining tractability, by the introduction
of latent variables, leading to state space models. As in Chapters 9 and 12, we shall
see that complex models can thereby be constructed from simpler components (in
particular, from distributions belonging to the exponential family) and can be read-
ily characterized using the framework of probabilistic graphical models. Here we
focus on the two most important examples of state space models, namely the hid-
den Markov model, in which the latent variables are discrete, and linear dynamical
systems, in which the latent variables are Gaussian. Both models are described by di-
rected graphs having a tree structure (no loops) for which inference can be performed
efﬁciently using the sum-product algorithm.
13.1. Markov Models
The easiest way to treat sequential data would be simply to ignore the sequential
aspects and treat the observations as i.i.d., corresponding to the graph in Figure 13.2.
Such an approach, however, would fail to exploit the sequential patterns in the data,
such as correlations between observations that are close in the sequence. Suppose,
for instance, that we observe a binary variable denoting whether on a particular day
it rained or not. Given a time series of recent observations of this variable, we wish
to predict whether it will rain on the next day. If we treat the data as i.i.d., then the
only information we can glean from the data is the relative frequency of rainy days.
However, we know in practice that the weather often exhibits trends that may last for
several days. Observing whether or not it rains today is therefore of signiﬁcant help
in predicting if it will rain tomorrow.
To express such effects in a probabilistic model, we need to relax the i.i.d. as-
sumption, and one of the simplest ways to do this is to consider a Markov model.
First of all we note that, without loss of generality, we can use the product rule to
express the joint distribution for a sequence of observations in the form
p(x1, . . . , xN) =
N

n=1
p(xn|x1, . . . , xn−1).
(13.1)
If we now assume that each of the conditional distributions on the right-hand side
is independent of all previous observations except the most recent, we obtain the
ﬁrst-order Markov chain, which is depicted as a graphical model in Figure 13.3. The


---
**Page 577**
608
13. SEQUENTIAL DATA
Figure 13.3
A ﬁrst-order Markov chain of ob-
servations {xn} in which the dis-
tribution p(xn|xn−1) of a particu-
lar observation xn is conditioned
on the value of the previous ob-
servation xn−1.
x1
x2
x3
x4
joint distribution for a sequence of N observations under this model is given by
p(x1, . . . , xN) = p(x1)
N

n=2
p(xn|xn−1).
(13.2)
From the d-separation property, we see that the conditional distribution for observa-
Section 8.2
tion xn, given all of the observations up to time n, is given by
p(xn|x1, . . . , xn−1) = p(xn|xn−1)
(13.3)
which is easily veriﬁed by direct evaluation starting from (13.2) and using the prod-
uct rule of probability. Thus if we use such a model to predict the next observation
Exercise 13.1
in a sequence, the distribution of predictions will depend only on the value of the im-
mediately preceding observation and will be independent of all earlier observations.
In most applications of such models, the conditional distributions p(xn|xn−1)
that deﬁne the model will be constrained to be equal, corresponding to the assump-
tion of a stationary time series. The model is then known as a homogeneous Markov
chain. For instance, if the conditional distributions depend on adjustable parameters
(whose values might be inferred from a set of training data), then all of the condi-
tional distributions in the chain will share the same values of those parameters.
Although this is more general than the independence model, it is still very re-
strictive. For many sequential observations, we anticipate that the trends in the data
over several successive observations will provide important information in predict-
ing the next value. One way to allow earlier observations to have an inﬂuence is to
move to higher-order Markov chains. If we allow the predictions to depend also on
the previous-but-one value, we obtain a second-order Markov chain, represented by
the graph in Figure 13.4. The joint distribution is now given by
p(x1, . . . , xN) = p(x1)p(x2|x1)
N

n=3
p(xn|xn−1, xn−2).
(13.4)
Again, using d-separation or by direct evaluation, we see that the conditional distri-
bution of xn given xn−1 and xn−2 is independent of all observations x1, . . . xn−3.
Figure 13.4
A second-order Markov chain, in
which the conditional distribution
of a particular observation xn
depends on the values of the two
previous observations xn−1 and
xn−2.
x1
x2
x3
x4


---
**Page 578**
13.1. Markov Models
609
Figure 13.5
We can represent sequen-
tial data using a Markov chain of latent
variables, with each observation condi-
tioned on the state of the corresponding
latent variable. This important graphical
structure forms the foundation both for the
hidden Markov model and for linear dy-
namical systems.
zn−1
zn
zn+1
xn−1
xn
xn+1
z1
z2
x1
x2
Each observation is now inﬂuenced by two previous observations. We can similarly
consider extensions to an M th order Markov chain in which the conditional distri-
bution for a particular variable depends on the previous M variables. However, we
have paid a price for this increased ﬂexibility because the number of parameters in
the model is now much larger. Suppose the observations are discrete variables hav-
ing K states. Then the conditional distribution p(xn|xn−1) in a ﬁrst-order Markov
chain will be speciﬁed by a set of K −1 parameters for each of the K states of xn−1
giving a total of K(K −1) parameters. Now suppose we extend the model to an
M th order Markov chain, so that the joint distribution is built up from conditionals
p(xn|xn−M, . . . , xn−1). If the variables are discrete, and if the conditional distri-
butions are represented by general conditional probability tables, then the number
of parameters in such a model will have KM−1(K −1) parameters. Because this
grows exponentially with M, it will often render this approach impractical for larger
values of M.
For continuous variables, we can use linear-Gaussian conditional distributions
in which each node has a Gaussian distribution whose mean is a linear function
of its parents. This is known as an autoregressive or AR model (Box et al., 1994;
Thiesson et al., 2004). An alternative approach is to use a parametric model for
p(xn|xn−M, . . . , xn−1) such as a neural network.
This technique is sometimes
called a tapped delay line because it corresponds to storing (delaying) the previous
M values of the observed variable in order to predict the next value. The number
of parameters can then be much smaller than in a completely general model (for ex-
ample it may grow linearly with M), although this is achieved at the expense of a
restricted family of conditional distributions.
Suppose we wish to build a model for sequences that is not limited by the
Markov assumption to any order and yet that can be speciﬁed using a limited number
of free parameters. We can achieve this by introducing additional latent variables to
permit a rich class of models to be constructed out of simple components, as we did
with mixture distributions in Chapter 9 and with continuous latent variable models in
Chapter 12. For each observation xn, we introduce a corresponding latent variable
zn (which may be of different type or dimensionality to the observed variable). We
now assume that it is the latent variables that form a Markov chain, giving rise to the
graphical structure known as a state space model, which is shown in Figure 13.5. It
satisﬁes the key conditional independence property that zn−1 and zn+1 are indepen-
dent given zn, so that
zn+1 ⊥⊥zn−1 | zn.
(13.5)


---
**Page 579**
610
13. SEQUENTIAL DATA
The joint distribution for this model is given by
p(x1, . . . , xN, z1, . . . , zN) = p(z1)
 N

n=2
p(zn|zn−1)
 N

n=1
p(xn|zn).
(13.6)
Using the d-separation criterion, we see that there is always a path connecting any
two observed variables xn and xm via the latent variables, and that this path is never
blocked. Thus the predictive distribution p(xn+1|x1, . . . , xn) for observation xn+1
given all previous observations does not exhibit any conditional independence prop-
erties, and so our predictions for xn+1 depends on all previous observations. The
observed variables, however, do not satisfy the Markov property at any order. We
shall discuss how to evaluate the predictive distribution in later sections of this chap-
ter.
There are two important models for sequential data that are described by this
graph. If the latent variables are discrete, then we obtain the hidden Markov model,
or HMM (Elliott et al., 1995). Note that the observed variables in an HMM may
Section 13.2
be discrete or continuous, and a variety of different conditional distributions can be
used to model them. If both the latent and the observed variables are Gaussian (with
a linear-Gaussian dependence of the conditional distributions on their parents), then
we obtain the linear dynamical system.
Section 13.3
13.2. Hidden Markov Models
The hidden Markov model can be viewed as a speciﬁc instance of the state space
model of Figure 13.5 in which the latent variables are discrete. However, if we
examine a single time slice of the model, we see that it corresponds to a mixture
distribution, with component densities given by p(x|z). It can therefore also be
interpreted as an extension of a mixture model in which the choice of mixture com-
ponent for each observation is not selected independently but depends on the choice
of component for the previous observation. The HMM is widely used in speech
recognition (Jelinek, 1997; Rabiner and Juang, 1993), natural language modelling
(Manning and Sch¨utze, 1999), on-line handwriting recognition (Nag et al., 1986),
and for the analysis of biological sequences such as proteins and DNA (Krogh et al.,
1994; Durbin et al., 1998; Baldi and Brunak, 2001).
As in the case of a standard mixture model, the latent variables are the discrete
multinomial variables zn describing which component of the mixture is responsible
for generating the corresponding observation xn. Again, it is convenient to use a
1-of-K coding scheme, as used for mixture models in Chapter 9. We now allow the
probability distribution of zn to depend on the state of the previous latent variable
zn−1 through a conditional distribution p(zn|zn−1). Because the latent variables are
K-dimensional binary variables, this conditional distribution corresponds to a table
of numbers that we denote by A, the elements of which are known as transition
probabilities. They are given by Ajk ≡p(znk = 1|zn−1,j = 1), and because they
are probabilities, they satisfy 0 ⩽Ajk ⩽1 with 
k Ajk = 1, so that the matrix A


---
**Page 580**
13.2. Hidden Markov Models
611
Figure 13.6
Transition diagram showing a model whose la-
tent variables have three possible states corre-
sponding to the three boxes. The black lines
denote the elements of the transition matrix
Ajk.
A12
A23
A31
A21
A32
A13
A11
A22
A33
k = 1
k = 2
k = 3
has K(K−1) independent parameters. We can then write the conditional distribution
explicitly in the form
p(zn|zn−1,A) =
K

k=1
K

j=1
Azn−1,jznk
jk
.
(13.7)
The initial latent node z1 is special in that it does not have a parent node, and so
it has a marginal distribution p(z1) represented by a vector of probabilities π with
elements πk ≡p(z1k = 1), so that
p(z1|π) =
K

k=1
πz1k
k
(13.8)
where 
k πk = 1.
The transition matrix is sometimes illustrated diagrammatically by drawing the
states as nodes in a state transition diagram as shown in Figure 13.6 for the case of
K = 3. Note that this does not represent a probabilistic graphical model, because
the nodes are not separate variables but rather states of a single variable, and so we
have shown the states as boxes rather than circles.
It is sometimes useful to take a state transition diagram, of the kind shown in
Figure 13.6, and unfold it over time. This gives an alternative representation of the
transitions between latent states, known as a lattice or trellis diagram, and which is
Section 8.4.5
shown for the case of the hidden Markov model in Figure 13.7.
The speciﬁcation of the probabilistic model is completed by deﬁning the con-
ditional distributions of the observed variables p(xn|zn, φ), where φ is a set of pa-
rameters governing the distribution. These are known as emission probabilities, and
might for example be given by Gaussians of the form (9.11) if the elements of x are
continuous variables, or by conditional probability tables if x is discrete. Because
xn is observed, the distribution p(xn|zn, φ) consists, for a given value of φ, of a
vector of K numbers corresponding to the K possible states of the binary vector zn.


---
**Page 581**
612
13. SEQUENTIAL DATA
Figure 13.7
If we unfold the state transition dia-
gram of Figure 13.6 over time, we obtain a lattice,
or trellis, representation of the latent states. Each
column of this diagram corresponds to one of the
latent variables zn.
k = 1
k = 2
k = 3
n −2
n −1
n
n + 1
A11
A11
A11
A33
A33
A33
We can represent the emission probabilities in the form
p(xn|zn, φ) =
K

k=1
p(xn|φk)znk.
(13.9)
We shall focuss attention on homogeneous models for which all of the condi-
tional distributions governing the latent variables share the same parameters A, and
similarly all of the emission distributions share the same parameters φ (the extension
to more general cases is straightforward). Note that a mixture model for an i.i.d. data
set corresponds to the special case in which the parameters Ajk are the same for all
values of j, so that the conditional distribution p(zn|zn−1) is independent of zn−1.
This corresponds to deleting the horizontal links in the graphical model shown in
Figure 13.5.
The joint probability distribution over both latent and observed variables is then
given by
p(X, Z|θ) = p(z1|π)
 N

n=2
p(zn|zn−1, A)
 
N

m=1
p(xm|zm, φ)
(13.10)
where X = {x1, . . . , xN}, Z = {z1, . . . , zN}, and θ = {π, A, φ} denotes the set
of parameters governing the model. Most of our discussion of the hidden Markov
model will be independent of the particular choice of the emission probabilities.
Indeed, the model is tractable for a wide range of emission distributions including
discrete tables, Gaussians, and mixtures of Gaussians. It is also possible to exploit
discriminative models such as neural networks. These can be used to model the
Exercise 13.4
emission density p(x|z) directly, or to provide a representation for p(z|x) that can
be converted into the required emission density p(x|z) using Bayes’ theorem (Bishop
et al., 2004).
We can gain a better understanding of the hidden Markov model by considering
it from a generative point of view. Recall that to generate samples from a mixture of


---
**Page 582**
13.2. Hidden Markov Models
613
k = 1
k = 2
k = 3
0
0.5
1
0
0.5
1
0
0.5
1
0
0.5
1
Figure 13.8
Illustration of sampling from a hidden Markov model having a 3-state latent variable z and a
Gaussian emission model p(x|z) where x is 2-dimensional. (a) Contours of constant probability density for the
emission distributions corresponding to each of the three states of the latent variable. (b) A sample of 50 points
drawn from the hidden Markov model, colour coded according to the component that generated them and with
lines connecting the successive observations. Here the transition matrix was ﬁxed so that in any state there is a
5% probability of making a transition to each of the other states, and consequently a 90% probability of remaining
in the same state.
Gaussians, we ﬁrst chose one of the components at random with probability given by
the mixing coefﬁcients πk and then generate a sample vector x from the correspond-
ing Gaussian component. This process is repeated N times to generate a data set of
N independent samples. In the case of the hidden Markov model, this procedure is
modiﬁed as follows. We ﬁrst choose the initial latent variable z1 with probabilities
governed by the parameters πk and then sample the corresponding observation x1.
Now we choose the state of the variable z2 according to the transition probabilities
p(z2|z1) using the already instantiated value of z1. Thus suppose that the sample for
z1 corresponds to state j. Then we choose the state k of z2 with probabilities Ajk
for k = 1, . . . , K. Once we know z2 we can draw a sample for x2 and also sample
the next latent variable z3 and so on. This is an example of ancestral sampling for
a directed graphical model. If, for instance, we have a model in which the diago-
Section 8.1.2
nal transition elements Akk are much larger than the off-diagonal elements, then a
typical data sequence will have long runs of points generated from a single compo-
nent, with infrequent transitions from one component to another. The generation of
samples from a hidden Markov model is illustrated in Figure 13.8.
There are many variants of the standard HMM model, obtained for instance by
imposing constraints on the form of the transition matrix A (Rabiner, 1989). Here we
mention one of particular practical importance called the left-to-right HMM, which
is obtained by setting the elements Ajk of A to zero if k < j, as illustrated in the


---
**Page 583**
614
13. SEQUENTIAL DATA
Figure 13.9
Example of the state transition diagram for a 3-state
left-to-right hidden Markov model. Note that once a
state has been vacated, it cannot later be re-entered.
k = 1
k = 2
k = 3
A11
A22
A33
A12
A23
A13
state transition diagram for a 3-state HMM in Figure 13.9. Typically for such models
the initial state probabilities for p(z1) are modiﬁed so that p(z11) = 1 and p(z1j) = 0
for j ̸= 1, in other words every sequence is constrained to start in state j = 1. The
transition matrix may be further constrained to ensure that large changes in the state
index do not occur, so that Ajk = 0 if k > j + ∆. This type of model is illustrated
using a lattice diagram in Figure 13.10.
Many applications of hidden Markov models, for example speech recognition,
or on-line character recognition, make use of left-to-right architectures. As an illus-
tration of the left-to-right hidden Markov model, we consider an example involving
handwritten digits. This uses on-line data, meaning that each digit is represented
by the trajectory of the pen as a function of time in the form of a sequence of pen
coordinates, in contrast to the off-line digits data, discussed in Appendix A, which
comprises static two-dimensional pixellated images of the ink. Examples of the on-
line digits are shown in Figure 13.11. Here we train a hidden Markov model on a
subset of data comprising 45 examples of the digit ‘2’. There are K = 16 states,
each of which can generate a line segment of ﬁxed length having one of 16 possible
angles, and so the emission distribution is simply a 16 × 16 table of probabilities
associated with the allowed angle values for each state index value. Transition prob-
abilities are all set to zero except for those that keep the state index k the same or
that increment it by 1, and the model parameters are optimized using 25 iterations of
EM. We can gain some insight into the resulting model by running it generatively, as
shown in Figure 13.11.
Figure 13.10
Lattice diagram for a 3-state left-
to-right HMM in which the state index k is allowed
to increase by at most 1 at each transition.
k = 1
k = 2
k = 3
n −2
n −1
n
n + 1
A11
A11
A11
A33
A33
A33


---
**Page 584**
13.2. Hidden Markov Models
615
Figure 13.11
Top row: examples of on-line handwritten
digits. Bottom row: synthetic digits sam-
pled generatively from a left-to-right hid-
den Markov model that has been trained
on a data set of 45 handwritten digits.
One of the most powerful properties of hidden Markov models is their ability to
exhibit some degree of invariance to local warping (compression and stretching) of
the time axis. To understand this, consider the way in which the digit ‘2’ is written
in the on-line handwritten digits example. A typical digit comprises two distinct
sections joined at a cusp. The ﬁrst part of the digit, which starts at the top left, has a
sweeping arc down to the cusp or loop at the bottom left, followed by a second more-
or-less straight sweep ending at the bottom right. Natural variations in writing style
will cause the relative sizes of the two sections to vary, and hence the location of the
cusp or loop within the temporal sequence will vary. From a generative perspective
such variations can be accommodated by the hidden Markov model through changes
in the number of transitions to the same state versus the number of transitions to the
successive state. Note, however, that if a digit ‘2’ is written in the reverse order, that
is, starting at the bottom right and ending at the top left, then even though the pen tip
coordinates may be identical to an example from the training set, the probability of
the observations under the model will be extremely small. In the speech recognition
context, warping of the time axis is associated with natural variations in the speed of
speech, and again the hidden Markov model can accommodate such a distortion and
not penalize it too heavily.
13.2.1
Maximum likelihood for the HMM
If we have observed a data set X = {x1, . . . , xN}, we can determine the param-
eters of an HMM using maximum likelihood. The likelihood function is obtained
from the joint distribution (13.10) by marginalizing over the latent variables
p(X|θ) =

Z
p(X, Z|θ).
(13.11)
Because the joint distribution p(X, Z|θ) does not factorize over n (in contrast to the
mixture distribution considered in Chapter 9), we cannot simply treat each of the
summations over zn independently. Nor can we perform the summations explicitly
because there are N variables to be summed over, each of which has K states, re-
sulting in a total of KN terms. Thus the number of terms in the summation grows


---
**Page 585**
616
13. SEQUENTIAL DATA
exponentially with the length of the chain. In fact, the summation in (13.11) cor-
responds to summing over exponentially many paths through the lattice diagram in
Figure 13.7.
We have already encountered a similar difﬁculty when we considered the infer-
ence problem for the simple chain of variables in Figure 8.32. There we were able
to make use of the conditional independence properties of the graph to re-order the
summations in order to obtain an algorithm whose cost scales linearly, instead of
exponentially, with the length of the chain. We shall apply a similar technique to the
hidden Markov model.
A further difﬁculty with the expression (13.11) for the likelihood function is that,
because it corresponds to a generalization of a mixture distribution, it represents a
summation over the emission models for different settings of the latent variables.
Direct maximization of the likelihood function will therefore lead to complex ex-
pressions with no closed-form solutions, as was the case for simple mixture models
Section 9.2
(recall that a mixture model for i.i.d. data is a special case of the HMM).
We therefore turn to the expectation maximization algorithm to ﬁnd an efﬁcient
framework for maximizing the likelihood function in hidden Markov models. The
EM algorithm starts with some initial selection for the model parameters, which we
denote by θold. In the E step, we take these parameter values and ﬁnd the posterior
distribution of the latent variables p(Z|X, θold). We then use this posterior distri-
bution to evaluate the expectation of the logarithm of the complete-data likelihood
function, as a function of the parameters θ, to give the function Q(θ, θold) deﬁned
by
Q(θ, θold) =

Z
p(Z|X, θold) ln p(X, Z|θ).
(13.12)
At this point, it is convenient to introduce some notation. We shall use γ(zn) to
denote the marginal posterior distribution of a latent variable zn, and ξ(zn−1, zn) to
denote the joint posterior distribution of two successive latent variables, so that
γ(zn)
=
p(zn|X, θold)
(13.13)
ξ(zn−1, zn)
=
p(zn−1, zn|X, θold).
(13.14)
For each value of n, we can store γ(zn) using a set of K nonnegative numbers
that sum to unity, and similarly we can store ξ(zn−1, zn) using a K × K matrix of
nonnegative numbers that again sum to unity. We shall also use γ(znk) to denote the
conditional probability of znk = 1, with a similar use of notation for ξ(zn−1,j, znk)
and for other probabilistic variables introduced later. Because the expectation of a
binary random variable is just the probability that it takes the value 1, we have
γ(znk)
=
E[znk] =

z
γ(z)znk
(13.15)
ξ(zn−1,j, znk)
=
E[zn−1,jznk] =

z
γ(z)zn−1,jznk.
(13.16)
If we substitute the joint distribution p(X, Z|θ) given by (13.10) into (13.12),


---
**Page 586**
13.2. Hidden Markov Models
617
and make use of the deﬁnitions of γ and ξ , we obtain
Q(θ, θold)
=
K

k=1
γ(z1k) ln πk +
N

n=2
K

j=1
K

k=1
ξ(zn−1,j, znk) ln Ajk
+
N

n=1
K

k=1
γ(znk) ln p(xn|φk).
(13.17)
The goal of the E step will be to evaluate the quantities γ(zn) and ξ(zn−1, zn) efﬁ-
ciently, and we shall discuss this in detail shortly.
In the M step, we maximize Q(θ, θold) with respect to the parameters θ =
{π, A, φ} in which we treat γ(zn) and ξ(zn−1, zn) as constant. Maximization with
respect to π and A is easily achieved using appropriate Lagrange multipliers with
the results
Exercise 13.5
πk
=
γ(z1k)
K

j=1
γ(z1j)
(13.18)
Ajk
=
N

n=2
ξ(zn−1,j, znk)
K

l=1
N

n=2
ξ(zn−1,j, znl)
.
(13.19)
The EM algorithm must be initialized by choosing starting values for π and A, which
should of course respect the summation constraints associated with their probabilis-
tic interpretation. Note that any elements of π or A that are set to zero initially will
remain zero in subsequent EM updates. A typical initialization procedure would
Exercise 13.6
involve selecting random starting values for these parameters subject to the summa-
tion and non-negativity constraints. Note that no particular modiﬁcation to the EM
results are required for the case of left-to-right models beyond choosing initial values
for the elements Ajk in which the appropriate elements are set to zero, because these
will remain zero throughout.
To maximize Q(θ, θold) with respect to φk, we notice that only the ﬁnal term
in (13.17) depends on φk, and furthermore this term has exactly the same form as
the data-dependent term in the corresponding function for a standard mixture dis-
tribution for i.i.d. data, as can be seen by comparison with (9.40) for the case of a
Gaussian mixture. Here the quantities γ(znk) are playing the role of the responsibil-
ities. If the parameters φk are independent for the different components, then this
term decouples into a sum of terms one for each value of k, each of which can be
maximized independently. We are then simply maximizing the weighted log likeli-
hood function for the emission density p(x|φk) with weights γ(znk). Here we shall
suppose that this maximization can be done efﬁciently. For instance, in the case of


---
**Page 587**
618
13. SEQUENTIAL DATA
Gaussian emission densities we have p(x|φk) = N(x|µk, Σk), and maximization
of the function Q(θ, θold) then gives
µk
=
N

n=1
γ(znk)xn
N

n=1
γ(znk)
(13.20)
Σk
=
N

n=1
γ(znk)(xn −µk)(xn −µk)T
N

n=1
γ(znk)
.
(13.21)
For the case of discrete multinomial observed variables, the conditional distribution
of the observations takes the form
p(x|z) =
D

i=1
K

k=1
µxizk
ik
(13.22)
and the corresponding M-step equations are given by
Exercise 13.8
µik =
N

n=1
γ(znk)xni
N

n=1
γ(znk)
.
(13.23)
An analogous result holds for Bernoulli observed variables.
The EM algorithm requires initial values for the parameters of the emission dis-
tribution. One way to set these is ﬁrst to treat the data initially as i.i.d. and ﬁt the
emission density by maximum likelihood, and then use the resulting values to ini-
tialize the parameters for EM.
13.2.2
The forward-backward algorithm
Next we seek an efﬁcient procedure for evaluating the quantities γ(znk) and
ξ(zn−1,j, znk), corresponding to the E step of the EM algorithm. The graph for the
hidden Markov model, shown in Figure 13.5, is a tree, and so we know that the
posterior distribution of the latent variables can be obtained efﬁciently using a two-
stage message passing algorithm. In the particular context of the hidden Markov
Section 8.4
model, this is known as the forward-backward algorithm (Rabiner, 1989), or the
Baum-Welch algorithm (Baum, 1972). There are in fact several variants of the basic
algorithm, all of which lead to the exact marginals, according to the precise form of


---
**Page 588**
13.2. Hidden Markov Models
619
the messages that are propagated along the chain (Jordan, 2007). We shall focus on
the most widely used of these, known as the alpha-beta algorithm.
As well as being of great practical importance in its own right, the forward-
backward algorithm provides us with a nice illustration of many of the concepts
introduced in earlier chapters. We shall therefore begin in this section with a ‘con-
ventional’ derivation of the forward-backward equations, making use of the sum
and product rules of probability, and exploiting conditional independence properties
which we shall obtain from the corresponding graphical model using d-separation.
Then in Section 13.2.3, we shall see how the forward-backward algorithm can be
obtained very simply as a speciﬁc example of the sum-product algorithm introduced
in Section 8.4.4.
It is worth emphasizing that evaluation of the posterior distributions of the latent
variables is independent of the form of the emission density p(x|z) or indeed of
whether the observed variables are continuous or discrete. All we require is the
values of the quantities p(xn|zn) for each value of zn for every n. Also, in this
section and the next we shall omit the explicit dependence on the model parameters
θold because these ﬁxed throughout.
We therefore begin by writing down the following conditional independence
properties (Jordan, 2007)
p(X|zn)
=
p(x1, . . . , xn|zn)
p(xn+1, . . . , xN|zn)
(13.24)
p(x1, . . . , xn−1|xn, zn)
=
p(x1, . . . , xn−1|zn)
(13.25)
p(x1, . . . , xn−1|zn−1, zn)
=
p(x1, . . . , xn−1|zn−1)
(13.26)
p(xn+1, . . . , xN|zn, zn+1)
=
p(xn+1, . . . , xN|zn+1)
(13.27)
p(xn+2, . . . , xN|zn+1, xn+1)
=
p(xn+2, . . . , xN|zn+1)
(13.28)
p(X|zn−1, zn)
=
p(x1, . . . , xn−1|zn−1)
p(xn|zn)p(xn+1, . . . , xN|zn) (13.29)
p(xN+1|X, zN+1)
=
p(xN+1|zN+1)
(13.30)
p(zN+1|zN, X)
=
p(zN+1|zN)
(13.31)
where X = {x1, . . . , xN}. These relations are most easily proved using d-separation.
For instance in the ﬁrst of these results, we note that every path from any one of the
nodes x1, . . . , xn−1 to the node xn passes through the node zn, which is observed.
Because all such paths are head-to-tail, it follows that the conditional independence
property must hold. The reader should take a few moments to verify each of these
properties in turn, as an exercise in the application of d-separation. These relations
can also be proved directly, though with signiﬁcantly greater effort, from the joint
distribution for the hidden Markov model using the sum and product rules of proba-
bility.
Exercise 13.10
Let us begin by evaluating γ(znk). Recall that for a discrete multinomial ran-
dom variable the expected value of one of its components is just the probability of
that component having the value 1. Thus we are interested in ﬁnding the posterior
distribution p(zn|x1, . . . , xN) of zn given the observed data set x1, . . . , xN. This


---
**Page 589**
620
13. SEQUENTIAL DATA
represents a vector of length K whose entries correspond to the expected values of
znk. Using Bayes’ theorem, we have
γ(zn) = p(zn|X) = p(X|zn)p(zn)
p(X)
.
(13.32)
Note that the denominator p(X) is implicitly conditioned on the parameters θold
of the HMM and hence represents the likelihood function. Using the conditional
independence property (13.24), together with the product rule of probability, we
obtain
γ(zn) = p(x1, . . . , xn, zn)p(xn+1, . . . , xN|zn)
p(X)
= α(zn)β(zn)
p(X)
(13.33)
where we have deﬁned
α(zn)
≡
p(x1, . . . , xn, zn)
(13.34)
β(zn)
≡
p(xn+1, . . . , xN|zn).
(13.35)
The quantity α(zn) represents the joint probability of observing all of the given
data up to time n and the value of zn, whereas β(zn) represents the conditional
probability of all future data from time n + 1 up to N given the value of zn. Again,
α(zn) and β(zn) each represent set of K numbers, one for each of the possible
settings of the 1-of-K coded binary vector zn. We shall use the notation α(znk) to
denote the value of α(zn) when znk = 1, with an analogous interpretation of β(znk).
We now derive recursion relations that allow α(zn) and β(zn) to be evaluated
efﬁciently. Again, we shall make use of conditional independence properties, in
particular (13.25) and (13.26), together with the sum and product rules, allowing us
to express α(zn) in terms of α(zn−1) as follows
α(zn)
=
p(x1, . . . , xn, zn)
=
p(x1, . . . , xn|zn)p(zn)
=
p(xn|zn)p(x1, . . . , xn−1|zn)p(zn)
=
p(xn|zn)p(x1, . . . , xn−1, zn)
=
p(xn|zn)

zn−1
p(x1, . . . , xn−1, zn−1, zn)
=
p(xn|zn)

zn−1
p(x1, . . . , xn−1, zn|zn−1)p(zn−1)
=
p(xn|zn)

zn−1
p(x1, . . . , xn−1|zn−1)p(zn|zn−1)p(zn−1)
=
p(xn|zn)

zn−1
p(x1, . . . , xn−1, zn−1)p(zn|zn−1)
Making use of the deﬁnition (13.34) for α(zn), we then obtain
α(zn) = p(xn|zn)

zn−1
α(zn−1)p(zn|zn−1).
(13.36)


---
**Page 590**
13.2. Hidden Markov Models
621
Figure 13.12
Illustration of the forward recursion (13.36) for
evaluation of the α variables. In this fragment
of the lattice, we see that the quantity α(zn1)
is obtained by taking the elements α(zn−1,j) of
α(zn−1) at step n−1 and summing them up with
weights given by Aj1, corresponding to the val-
ues of p(zn|zn−1), and then multiplying by the
data contribution p(xn|zn1).
k = 1
k = 2
k = 3
n −1
n
α(zn−1,1)
α(zn−1,2)
α(zn−1,3)
α(zn,1)
A11
A21
A31
p(xn|zn,1)
It is worth taking a moment to study this recursion relation in some detail. Note
that there are K terms in the summation, and the right-hand side has to be evaluated
for each of the K values of zn so each step of the α recursion has computational
cost that scaled like O(K2). The forward recursion equation for α(zn) is illustrated
using a lattice diagram in Figure 13.12.
In order to start this recursion, we need an initial condition that is given by
α(z1) = p(x1, z1) = p(z1)p(x1|z1) =
K

k=1
{πkp(x1|φk)}z1k
(13.37)
which tells us that α(z1k), for k = 1, . . . , K, takes the value πkp(x1|φk). Starting
at the ﬁrst node of the chain, we can then work along the chain and evaluate α(zn)
for every latent node. Because each step of the recursion involves multiplying by a
K × K matrix, the overall cost of evaluating these quantities for the whole chain is
of O(K2N).
We can similarly ﬁnd a recursion relation for the quantities β(zn) by making
use of the conditional independence properties (13.27) and (13.28) giving
β(zn)
=
p(xn+1, . . . , xN|zn)
=

zn+1
p(xn+1, . . . , xN, zn+1|zn)
=

zn+1
p(xn+1, . . . , xN|zn, zn+1)p(zn+1|zn)
=

zn+1
p(xn+1, . . . , xN|zn+1)p(zn+1|zn)
=

zn+1
p(xn+2, . . . , xN|zn+1)p(xn+1|zn+1)p(zn+1|zn).


---
**Page 591**
622
13. SEQUENTIAL DATA
Figure 13.13
Illustration
of
the
backward
recursion
(13.38) for evaluation of the β variables. In
this fragment of the lattice, we see that the
quantity β(zn1) is obtained by taking the
components β(zn+1,k) of β(zn+1) at step
n + 1 and summing them up with weights
given by the products of A1k, correspond-
ing to the values of p(zn+1|zn) and the cor-
responding values of the emission density
p(xn|zn+1,k).
k = 1
k = 2
k = 3
n
n + 1
β(zn,1)
β(zn+1,1)
β(zn+1,2)
β(zn+1,3)
A11
A12
A13
p(xn|zn+1,1)
p(xn|zn+1,2)
p(xn|zn+1,3)
Making use of the deﬁnition (13.35) for β(zn), we then obtain
β(zn) =

zn+1
β(zn+1)p(xn+1|zn+1)p(zn+1|zn).
(13.38)
Note that in this case we have a backward message passing algorithm that evaluates
β(zn) in terms of β(zn+1). At each step, we absorb the effect of observation xn+1
through the emission probability p(xn+1|zn+1), multiply by the transition matrix
p(zn+1|zn), and then marginalize out zn+1. This is illustrated in Figure 13.13.
Again we need a starting condition for the recursion, namely a value for β(zN).
This can be obtained by setting n = N in (13.33) and replacing α(zN) with its
deﬁnition (13.34) to give
p(zN|X) = p(X, zN)β(zN)
p(X)
(13.39)
which we see will be correct provided we take β(zN) = 1 for all settings of zN.
In the M step equations, the quantity p(X) will cancel out, as can be seen, for
instance, in the M-step equation for µk given by (13.20), which takes the form
µk =
n

n=1
γ(znk)xn
n

n=1
γ(znk)
=
n

n=1
α(znk)β(znk)xn
n

n=1
α(znk)β(znk)
.
(13.40)
However, the quantity p(X) represents the likelihood function whose value we typ-
ically wish to monitor during the EM optimization, and so it is useful to be able to
evaluate it. If we sum both sides of (13.33) over zn, and use the fact that the left-hand
side is a normalized distribution, we obtain
p(X) =

zn
α(zn)β(zn).
(13.41)


---
**Page 592**
13.2. Hidden Markov Models
623
Thus we can evaluate the likelihood function by computing this sum, for any conve-
nient choice of n. For instance, if we only want to evaluate the likelihood function,
then we can do this by running the α recursion from the start to the end of the chain,
and then use this result for n = N, making use of the fact that β(zN) is a vector of
1s. In this case no β recursion is required, and we simply have
p(X) =

zN
α(zN).
(13.42)
Let us take a moment to interpret this result for p(X). Recall that to compute the
likelihood we should take the joint distribution p(X, Z) and sum over all possible
values of Z. Each such value represents a particular choice of hidden state for every
time step, in other words every term in the summation is a path through the lattice
diagram, and recall that there are exponentially many such paths. By expressing
the likelihood function in the form (13.42), we have reduced the computational cost
from being exponential in the length of the chain to being linear by swapping the
order of the summation and multiplications, so that at each time step n we sum
the contributions from all paths passing through each of the states znk to give the
intermediate quantities α(zn).
Next we consider the evaluation of the quantities ξ(zn−1, zn), which correspond
to the values of the conditional probabilities p(zn−1, zn|X) for each of the K × K
settings for (zn−1, zn). Using the deﬁnition of ξ(zn−1, zn), and applying Bayes’
theorem, we have
ξ(zn−1, zn) = p(zn−1, zn|X)
=
p(X|zn−1, zn)p(zn−1, zn)
p(X)
=
p(x1, . . . , xn−1|zn−1)p(xn|zn)p(xn+1, . . . , xN|zn)p(zn|zn−1)p(zn−1)
p(X)
=
α(zn−1)p(xn|zn)p(zn|zn−1)β(zn)
p(X)
(13.43)
where we have made use of the conditional independence property (13.29) together
with the deﬁnitions of α(zn) and β(zn) given by (13.34) and (13.35). Thus we can
calculate the ξ(zn−1, zn) directly by using the results of the α and β recursions.
Let us summarize the steps required to train a hidden Markov model using
the EM algorithm. We ﬁrst make an initial selection of the parameters θold where
θ ≡(π, A, φ). The A and π parameters are often initialized either uniformly or
randomly from a uniform distribution (respecting their non-negativity and summa-
tion constraints). Initialization of the parameters φ will depend on the form of the
distribution. For instance in the case of Gaussians, the parameters µk might be ini-
tialized by applying the K-means algorithm to the data, and Σk might be initialized
to the covariance matrix of the corresponding K means cluster. Then we run both
the forward α recursion and the backward β recursion and use the results to evaluate
γ(zn) and ξ(zn−1, zn). At this stage, we can also evaluate the likelihood function.


---
**Page 593**
624
13. SEQUENTIAL DATA
This completes the E step, and we use the results to ﬁnd a revised set of parameters
θnew using the M-step equations from Section 13.2.1. We then continue to alternate
between E and M steps until some convergence criterion is satisﬁed, for instance
when the change in the likelihood function is below some threshold.
Note that in these recursion relations the observations enter through conditional
distributions of the form p(xn|zn). The recursions are therefore independent of
the type or dimensionality of the observed variables or the form of this conditional
distribution, so long as its value can be computed for each of the K possible states
of zn. Since the observed variables {xn} are ﬁxed, the quantities p(xn|zn) can be
pre-computed as functions of zn at the start of the EM algorithm, and remain ﬁxed
throughout.
We have seen in earlier chapters that the maximum likelihood approach is most
effective when the number of data points is large in relation to the number of parame-
ters. Here we note that a hidden Markov model can be trained effectively, using max-
imum likelihood, provided the training sequence is sufﬁciently long. Alternatively,
we can make use of multiple shorter sequences, which requires a straightforward
modiﬁcation of the hidden Markov model EM algorithm. In the case of left-to-right
Exercise 13.12
models, this is particularly important because, in a given observation sequence, a
given state transition corresponding to a nondiagonal element of A will seen at most
once.
Another quantity of interest is the predictive distribution, in which the observed
data is X = {x1, . . . , xN} and we wish to predict xN+1, which would be important
for real-time applications such as ﬁnancial forecasting. Again we make use of the
sum and product rules together with the conditional independence properties (13.29)
and (13.31) giving
p(xN+1|X)
=

zN+1
p(xN+1, zN+1|X)
=

zN+1
p(xN+1|zN+1)p(zN+1|X)
=

zN+1
p(xN+1|zN+1)

zN
p(zN+1, zN|X)
=

zN+1
p(xN+1|zN+1)

zN
p(zN+1|zN)p(zN|X)
=

zN+1
p(xN+1|zN+1)

zN
p(zN+1|zN)p(zN, X)
p(X)
=
1
p(X)

zN+1
p(xN+1|zN+1)

zN
p(zN+1|zN)α(zN) (13.44)
which can be evaluated by ﬁrst running a forward α recursion and then computing
the ﬁnal summations over zN and zN+1. The result of the ﬁrst summation over zN
can be stored and used once the value of xN+1 is observed in order to run the α
recursion forward to the next step in order to predict the subsequent value xN+2.


---
**Page 594**
13.2. Hidden Markov Models
625
Figure 13.14
A fragment of the fac-
tor graph representation for the hidden
Markov model.
χ
ψn
g1
gn−1
gn
z1
zn−1
zn
x1
xn−1
xn
Note that in (13.44), the inﬂuence of all data from x1 to xN is summarized in the K
values of α(zN). Thus the predictive distribution can be carried forward indeﬁnitely
using a ﬁxed amount of storage, as may be required for real-time applications.
Here we have discussed the estimation of the parameters of an HMM using max-
imum likelihood. This framework is easily extended to regularized maximum likeli-
hood by introducing priors over the model parameters π, A and φ whose values are
then estimated by maximizing their posterior probability. This can again be done us-
ing the EM algorithm in which the E step is the same as discussed above, and the M
step involves adding the log of the prior distribution p(θ) to the function Q(θ, θold)
before maximization and represents a straightforward application of the techniques
developed at various points in this book. Furthermore, we can use variational meth-
ods to give a fully Bayesian treatment of the HMM in which we marginalize over the
Section 10.1
parameter distributions (MacKay, 1997). As with maximum likelihood, this leads to
a two-pass forward-backward recursion to compute posterior probabilities.
13.2.3
The sum-product algorithm for the HMM
The directed graph that represents the hidden Markov model, shown in Fig-
ure 13.5, is a tree and so we can solve the problem of ﬁnding local marginals for the
hidden variables using the sum-product algorithm. Not surprisingly, this turns out to
Section 8.4.4
be equivalent to the forward-backward algorithm considered in the previous section,
and so the sum-product algorithm therefore provides us with a simple way to derive
the alpha-beta recursion formulae.
We begin by transforming the directed graph of Figure 13.5 into a factor graph,
of which a representative fragment is shown in Figure 13.14. This form of the fac-
tor graph shows all variables, both latent and observed, explicitly. However, for
the purpose of solving the inference problem, we shall always be conditioning on
the variables x1, . . . , xN, and so we can simplify the factor graph by absorbing the
emission probabilities into the transition probability factors. This leads to the sim-
pliﬁed factor graph representation in Figure 13.15, in which the factors are given
by
h(z1)
=
p(z1)p(x1|z1)
(13.45)
fn(zn−1, zn)
=
p(zn|zn−1)p(xn|zn).
(13.46)


---
**Page 595**
626
13. SEQUENTIAL DATA
Figure 13.15
A simpliﬁed form of fac-
tor graph to describe the hidden Markov
model.
h
fn
z1
zn−1
zn
To derive the alpha-beta algorithm, we denote the ﬁnal hidden variable zN as
the root node, and ﬁrst pass messages from the leaf node h to the root. From the
general results (8.66) and (8.69) for message propagation, we see that the messages
which are propagated in the hidden Markov model take the form
µzn−1→fn(zn−1)
=
µfn−1→zn−1(zn−1)
(13.47)
µfn→zn(zn)
=

zn−1
fn(zn−1, zn)µzn−1→fn(zn−1)
(13.48)
These equations represent the propagation of messages forward along the chain and
are equivalent to the alpha recursions derived in the previous section, as we shall
now show. Note that because the variable nodes zn have only two neighbours, they
perform no computation.
We can eliminate µzn−1→fn(zn−1) from (13.48) using (13.47) to give a recur-
sion for the f →z messages of the form
µfn→zn(zn) =

zn−1
fn(zn−1, zn)µfn−1→zn−1(zn−1).
(13.49)
If we now recall the deﬁnition (13.46), and if we deﬁne
α(zn) = µfn→zn(zn)
(13.50)
then we obtain the alpha recursion given by (13.36). We also need to verify that
the quantities α(zn) are themselves equivalent to those deﬁned previously. This
is easily done by using the initial condition (8.71) and noting that α(z1) is given
by h(z1) = p(z1)p(x1|z1) which is identical to (13.37). Because the initial α is
the same, and because they are iteratively computed using the same equation, all
subsequent α quantities must be the same.
Next we consider the messages that are propagated from the root node back to
the leaf node. These take the form
µfn+1→fn(zn) =

zn+1
fn+1(zn, zn+1)µfn+2→fn+1(zn+1)
(13.51)
where, as before, we have eliminated the messages of the type z →f since the
variable nodes perform no computation. Using the deﬁnition (13.46) to substitute
for fn+1(zn, zn+1), and deﬁning
β(zn) = µfn+1→zn(zn)
(13.52)


---
**Page 596**
13.2. Hidden Markov Models
627
we obtain the beta recursion given by (13.38). Again, we can verify that the beta
variables themselves are equivalent by noting that (8.70) implies that the initial mes-
sage send by the root variable node is µzN→fN (zN) = 1, which is identical to the
initialization of β(zN) given in Section 13.2.2.
The sum-product algorithm also speciﬁes how to evaluate the marginals once all
the messages have been evaluated. In particular, the result (8.63) shows that the local
marginal at the node zn is given by the product of the incoming messages. Because
we have conditioned on the variables X = {x1, . . . , xN}, we are computing the
joint distribution
p(zn, X) = µfn→zn(zn)µfn+1→zn(zn) = α(zn)β(zn).
(13.53)
Dividing both sides by p(X), we then obtain
γ(zn) = p(zn, X)
p(X)
= α(zn)β(zn)
p(X)
(13.54)
in agreement with (13.33). The result (13.43) can similarly be derived from (8.72).
Exercise 13.11
13.2.4
Scaling factors
There is an important issue that must be addressed before we can make use of the
forward backward algorithm in practice. From the recursion relation (13.36), we note
that at each step the new value α(zn) is obtained from the previous value α(zn−1)
by multiplying by quantities p(zn|zn−1) and p(xn|zn). Because these probabilities
are often signiﬁcantly less than unity, as we work our way forward along the chain,
the values of α(zn) can go to zero exponentially quickly. For moderate lengths of
chain (say 100 or so), the calculation of the α(zn) will soon exceed the dynamic
range of the computer, even if double precision ﬂoating point is used.
In the case of i.i.d. data, we implicitly circumvented this problem with the eval-
uation of likelihood functions by taking logarithms. Unfortunately, this will not help
here because we are forming sums of products of small numbers (we are in fact im-
plicitly summing over all possible paths through the lattice diagram of Figure 13.7).
We therefore work with re-scaled versions of α(zn) and β(zn) whose values remain
of order unity. As we shall see, the corresponding scaling factors cancel out when
we use these re-scaled quantities in the EM algorithm.
In (13.34), we deﬁned α(zn) = p(x1, . . . , xn, zn) representing the joint distri-
bution of all the observations up to xn and the latent variable zn. Now we deﬁne a
normalized version of α given by
α(zn) = p(zn|x1, . . . , xn) =
α(zn)
p(x1, . . . , xn)
(13.55)
which we expect to be well behaved numerically because it is a probability distribu-
tion over K variables for any value of n. In order to relate the scaled and original al-
pha variables, we introduce scaling factors deﬁned by conditional distributions over
the observed variables
cn = p(xn|x1, . . . , xn−1).
(13.56)


---
**Page 597**
628
13. SEQUENTIAL DATA
From the product rule, we then have
p(x1, . . . , xn) =
n

m=1
cm
(13.57)
and so
α(zn) = p(zn|x1, . . . , xn)p(x1, . . . , xn) =

n

m=1
cm

α(zn).
(13.58)
We can then turn the recursion equation (13.36) for α into one for α given by
cnα(zn) = p(xn|zn)

zn−1
α(zn−1)p(zn|zn−1).
(13.59)
Note that at each stage of the forward message passing phase, used to evaluate α(zn),
we have to evaluate and store cn, which is easily done because it is the coefﬁcient
that normalizes the right-hand side of (13.59) to give α(zn).
We can similarly deﬁne re-scaled variables β(zn) using
β(zn) =

N

m=n+1
cm

β(zn)
(13.60)
which will again remain within machine precision because, from (13.35), the quan-
tities β(zn) are simply the ratio of two conditional probabilities
β(zn) =
p(xn+1, . . . , xN|zn)
p(xn+1, . . . , xN|x1, . . . , xn).
(13.61)
The recursion result (13.38) for β then gives the following recursion for the re-scaled
variables
cn+1β(zn) =

zn+1
β(zn+1)p(xn+1|zn+1)p(zn+1|zn).
(13.62)
In applying this recursion relation, we make use of the scaling factors cn that were
previously computed in the α phase.
From (13.57), we see that the likelihood function can be found using
p(X) =
N

n=1
cn.
(13.63)
Similarly, using (13.33) and (13.43), together with (13.63), we see that the required
marginals are given by
Exercise 13.15
γ(zn)
=
α(zn)β(zn)
(13.64)
ξ(zn−1, zn)
=
cnα(zn−1)p(xn|zn)p(zn|z−1)β(zn).
(13.65)


---
**Page 598**
13.2. Hidden Markov Models
629
Finally, we note that there is an alternative formulation of the forward-backward
algorithm (Jordan, 2007) in which the backward pass is deﬁned by a recursion based
the quantities γ(zn) = α(zn)β(zn) instead of using β(zn). This α–γ recursion
requires that the forward pass be completed ﬁrst so that all the quantities α(zn)
are available for the backward pass, whereas the forward and backward passes of
the α–β algorithm can be done independently. Although these two algorithms have
comparable computational cost, the α–β version is the most commonly encountered
one in the case of hidden Markov models, whereas for linear dynamical systems a
Section 13.3
recursion analogous to the α–γ form is more usual.
13.2.5
The Viterbi algorithm
In many applications of hidden Markov models, the latent variables have some
meaningful interpretation, and so it is often of interest to ﬁnd the most probable
sequence of hidden states for a given observation sequence. For instance in speech
recognition, we might wish to ﬁnd the most probable phoneme sequence for a given
series of acoustic observations. Because the graph for the hidden Markov model is
a directed tree, this problem can be solved exactly using the max-sum algorithm.
We recall from our discussion in Section 8.4.5 that the problem of ﬁnding the most
probable sequence of latent states is not the same as that of ﬁnding the set of states
that are individually the most probable. The latter problem can be solved by ﬁrst
running the forward-backward (sum-product) algorithm to ﬁnd the latent variable
marginals γ(zn) and then maximizing each of these individually (Duda et al., 2001).
However, the set of such states will not, in general, correspond to the most probable
sequence of states. In fact, this set of states might even represent a sequence having
zero probability, if it so happens that two successive states, which in isolation are
individually the most probable, are such that the transition matrix element connecting
them is zero.
In practice, we are usually interested in ﬁnding the most probable sequence of
states, and this can be solved efﬁciently using the max-sum algorithm, which in the
context of hidden Markov models is known as the Viterbi algorithm (Viterbi, 1967).
Note that the max-sum algorithm works with log probabilities and so there is no
need to use re-scaled variables as was done with the forward-backward algorithm.
Figure 13.16 shows a fragment of the hidden Markov model expanded as lattice
diagram. As we have already noted, the number of possible paths through the lattice
grows exponentially with the length of the chain. The Viterbi algorithm searches this
space of paths efﬁciently to ﬁnd the most probable path with a computational cost
that grows only linearly with the length of the chain.
As with the sum-product algorithm, we ﬁrst represent the hidden Markov model
as a factor graph, as shown in Figure 13.15. Again, we treat the variable node zN
as the root, and pass messages to the root starting with the leaf nodes. Using the
results (8.93) and (8.94), we see that the messages passed in the max-sum algorithm
are given by
µzn→fn+1(zn)
=
µfn→zn(zn)
(13.66)
µfn+1→zn+1(zn+1)
=
max
zn

ln fn+1(zn, zn+1) + µzn→fn+1(zn)
. (13.67)


---
**Page 599**
630
13. SEQUENTIAL DATA
Figure 13.16
A fragment of the HMM lattice
showing two possible paths. The Viterbi algorithm
efﬁciently determines the most probable path from
amongst the exponentially many possibilities. For
any given path, the corresponding probability is
given by the product of the elements of the tran-
sition matrix Ajk, corresponding to the probabil-
ities p(zn+1|zn) for each segment of the path,
along with the emission densities p(xn|k) asso-
ciated with each node on the path.
k = 1
k = 2
k = 3
n −2
n −1
n
n + 1
If we eliminate µzn→fn+1(zn) between these two equations, and make use of (13.46),
we obtain a recursion for the f →z messages of the form
ω(zn+1) = ln p(xn+1|zn+1) + max
zn {ln p(x+1|zn) + ω(zn)}
(13.68)
where we have introduced the notation ω(zn) ≡µfn→zn(zn).
From (8.95) and (8.96), these messages are initialized using
ω(z1) = ln p(z1) + ln p(x1|z1).
(13.69)
where we have used (13.45). Note that to keep the notation uncluttered, we omit
the dependence on the model parameters θ that are held ﬁxed when ﬁnding the most
probable sequence.
The Viterbi algorithm can also be derived directly from the deﬁnition (13.6) of
the joint distribution by taking the logarithm and then exchanging maximizations
and summations. It is easily seen that the quantities ω(zn) have the probabilistic
Exercise 13.16
interpretation
ω(zn) =
max
z1,...,zn−1 p(x1, . . . , xn, z1, . . . , zn).
(13.70)
Once we have completed the ﬁnal maximization over zN, we will obtain the
value of the joint distribution p(X, Z) corresponding to the most probable path. We
also wish to ﬁnd the sequence of latent variable values that corresponds to this path.
To do this, we simply make use of the back-tracking procedure discussed in Sec-
tion 8.4.5. Speciﬁcally, we note that the maximization over zn must be performed
for each of the K possible values of zn+1. Suppose we keep a record of the values
of zn that correspond to the maxima for each value of the K values of zn+1. Let us
denote this function by ψ(kn) where k ∈{1, . . . , K}. Once we have passed mes-
sages to the end of the chain and found the most probable state of zN, we can then
use this function to backtrack along the chain by applying it recursively
kmax
n
= ψ(kmax
n+1).
(13.71)


---
**Page 600**
13.2. Hidden Markov Models
631
Intuitively, we can understand the Viterbi algorithm as follows. Naively, we
could consider explicitly all of the exponentially many paths through the lattice,
evaluate the probability for each, and then select the path having the highest proba-
bility. However, we notice that we can make a dramatic saving in computational cost
as follows. Suppose that for each path we evaluate its probability by summing up
products of transition and emission probabilities as we work our way forward along
each path through the lattice. Consider a particular time step n and a particular state
k at that time step. There will be many possible paths converging on the correspond-
ing node in the lattice diagram. However, we need only retain that particular path
that so far has the highest probability. Because there are K states at time step n, we
need to keep track of K such paths. At time step n + 1, there will be K2 possible
paths to consider, comprising K possible paths leading out of each of the K current
states, but again we need only retain K of these corresponding to the best path for
each state at time n+1. When we reach the ﬁnal time step N we will discover which
state corresponds to the overall most probable path. Because there is a unique path
coming into that state we can trace the path back to step N −1 to see what state it
occupied at that time, and so on back through the lattice to the state n = 1.
13.2.6
Extensions of the hidden Markov model
The basic hidden Markov model, along with the standard training algorithm
based on maximum likelihood, has been extended in numerous ways to meet the
requirements of particular applications. Here we discuss a few of the more important
examples.
We see from the digits example in Figure 13.11 that hidden Markov models can
be quite poor generative models for the data, because many of the synthetic digits
look quite unrepresentative of the training data. If the goal is sequence classiﬁca-
tion, there can be signiﬁcant beneﬁt in determining the parameters of hidden Markov
models using discriminative rather than maximum likelihood techniques. Suppose
we have a training set of R observation sequences Xr, where r = 1, . . . , R, each of
which is labelled according to its class m, where m = 1, . . . , M. For each class, we
have a separate hidden Markov model with its own parameters θm, and we treat the
problem of determining the parameter values as a standard classiﬁcation problem in
which we optimize the cross-entropy
R

r=1
ln p(mr|Xr).
(13.72)
Using Bayes’ theorem this can be expressed in terms of the sequence probabilities
associated with the hidden Markov models
R

r=1
ln

p(Xr|θr)p(mr)
M
l=1 p(Xr|θl)p(lr)

(13.73)
where p(m) is the prior probability of class m. Optimization of this cost function
is more complex than for maximum likelihood (Kapadia, 1998), and in particular


---
**Page 601**
632
13. SEQUENTIAL DATA
Figure 13.17
Section of an autoregressive hidden
Markov model, in which the distribution
of the observation xn depends on a
subset of the previous observations as
well as on the hidden state zn. In this
example, the distribution of xn depends
on the two previous observations xn−1
and xn−2.
zn−1
zn
zn+1
xn−1
xn
xn+1
requires that every training sequence be evaluated under each of the models in or-
der to compute the denominator in (13.73). Hidden Markov models, coupled with
discriminative training methods, are widely used in speech recognition (Kapadia,
1998).
A signiﬁcant weakness of the hidden Markov model is the way in which it rep-
resents the distribution of times for which the system remains in a given state. To see
the problem, note that the probability that a sequence sampled from a given hidden
Markov model will spend precisely T steps in state k and then make a transition to a
different state is given by
p(T) = (Akk)T (1 −Akk) ∝exp (−T ln Akk)
(13.74)
and so is an exponentially decaying function of T. For many applications, this will
be a very unrealistic model of state duration. The problem can be resolved by mod-
elling state duration directly in which the diagonal coefﬁcients Akk are all set to zero,
and each state k is explicitly associated with a probability distribution p(T|k) of pos-
sible duration times. From a generative point of view, when a state k is entered, a
value T representing the number of time steps that the system will remain in state k
is then drawn from p(T|k). The model then emits T values of the observed variable
xt, which are generally assumed to be independent so that the corresponding emis-
sion density is simply T
t=1 p(xt|k). This approach requires some straightforward
modiﬁcations to the EM optimization procedure (Rabiner, 1989).
Another limitation of the standard HMM is that it is poor at capturing long-
range correlations between the observed variables (i.e., between variables that are
separated by many time steps) because these must be mediated via the ﬁrst-order
Markov chain of hidden states. Longer-range effects could in principle be included
by adding extra links to the graphical model of Figure 13.5. One way to address this
is to generalize the HMM to give the autoregressive hidden Markov model (Ephraim
et al., 1989), an example of which is shown in Figure 13.17. For discrete observa-
tions, this corresponds to expanded tables of conditional probabilities for the emis-
sion distributions. In the case of a Gaussian emission density, we can use the linear-
Gaussian framework in which the conditional distribution for xn given the values
of the previous observations, and the value of zn, is a Gaussian whose mean is a
linear combination of the values of the conditioning variables. Clearly the number
of additional links in the graph must be limited to avoid an excessive the number of
free parameters. In the example shown in Figure 13.17, each observation depends on


---
**Page 602**
13.2. Hidden Markov Models
633
Figure 13.18
Example
of
an
input-output
hidden
Markov model.
In this case, both the
emission probabilities and the transition
probabilities depend on the values of a
sequence of observations u1, . . . , uN.
zn−1
zn
zn+1
xn−1
xn
xn+1
un−1
un
un+1
the two preceding observed variables as well as on the hidden state. Although this
graph looks messy, we can again appeal to d-separation to see that in fact it still has
a simple probabilistic structure. In particular, if we imagine conditioning on zn we
see that, as with the standard HMM, the values of zn−1 and zn+1 are independent,
corresponding to the conditional independence property (13.5). This is easily veri-
ﬁed by noting that every path from node zn−1 to node zn+1 passes through at least
one observed node that is head-to-tail with respect to that path. As a consequence,
we can again use a forward-backward recursion in the E step of the EM algorithm to
determine the posterior distributions of the latent variables in a computational time
that is linear in the length of the chain. Similarly, the M step involves only a minor
modiﬁcation of the standard M-step equations. In the case of Gaussian emission
densities this involves estimating the parameters using the standard linear regression
equations, discussed in Chapter 3.
We have seen that the autoregressive HMM appears as a natural extension of the
standard HMM when viewed as a graphical model. In fact the probabilistic graphical
modelling viewpoint motivates a plethora of different graphical structures based on
the HMM. Another example is the input-output hidden Markov model (Bengio and
Frasconi, 1995), in which we have a sequence of observed variables u1, . . . , uN, in
addition to the output variables x1, . . . , xN, whose values inﬂuence either the dis-
tribution of latent variables or output variables, or both. An example is shown in
Figure 13.18. This extends the HMM framework to the domain of supervised learn-
ing for sequential data. It is again easy to show, through the use of the d-separation
criterion, that the Markov property (13.5) for the chain of latent variables still holds.
To verify this, simply note that there is only one path from node zn−1 to node zn+1
and this is head-to-tail with respect to the observed node zn. This conditional inde-
pendence property again allows the formulation of a computationally efﬁcient learn-
ing algorithm. In particular, we can determine the parameters θ of the model by
maximizing the likelihood function L(θ) = p(X|U, θ) where U is a matrix whose
rows are given by uT
n. As a consequence of the conditional independence property
(13.5) this likelihood function can be maximized efﬁciently using an EM algorithm
in which the E step involves forward and backward recursions.
Exercise 13.18
Another variant of the HMM worthy of mention is the factorial hidden Markov
model (Ghahramani and Jordan, 1997), in which there are multiple independent


---
**Page 603**
634
13. SEQUENTIAL DATA
Figure 13.19
A factorial hidden Markov model com-
prising two Markov chains of latent vari-
ables. For continuous observed variables
x, one possible choice of emission model
is a linear-Gaussian density in which the
mean of the Gaussian is a linear combi-
nation of the states of the corresponding
latent variables.
z(1)
n−1
z(1)
n
z(1)
n+1
z(2)
n−1
z(2)
n
z(2)
n+1
xn−1
xn
xn+1
Markov chains of latent variables, and the distribution of the observed variable at
a given time step is conditional on the states of all of the corresponding latent vari-
ables at that same time step. Figure 13.19 shows the corresponding graphical model.
The motivation for considering factorial HMM can be seen by noting that in order to
represent, say, 10 bits of information at a given time step, a standard HMM would
need K = 210 = 1024 latent states, whereas a factorial HMM could make use of 10
binary latent chains. The primary disadvantage of factorial HMMs, however, lies in
the additional complexity of training them. The M step for the factorial HMM model
is straightforward. However, observation of the x variables introduces dependencies
between the latent chains, leading to difﬁculties with the E step. This can be seen
by noting that in Figure 13.19, the variables z(1)
n
and z(2)
n
are connected by a path
which is head-to-head at node xn and hence they are not d-separated. The exact E
step for this model does not correspond to running forward and backward recursions
along the M Markov chains independently. This is conﬁrmed by noting that the key
conditional independence property (13.5) is not satisﬁed for the individual Markov
chains in the factorial HMM model, as is shown using d-separation in Figure 13.20.
Now suppose that there are M chains of hidden nodes and for simplicity suppose
that all latent variables have the same number K of states. Then one approach would
be to note that there are KM combinations of latent variables at a given time step
Figure 13.20
Example of a path, highlighted in green,
which is head-to-head at the observed
nodes xn−1 and xn+1, and head-to-tail
at the unobserved nodes z(2)
n−1, z(2)
n
and
z(2)
n+1. Thus the path is not blocked and
so the conditional independence property
(13.5) does not hold for the individual la-
tent chains of the factorial HMM model.
As a consequence, there is no efﬁcient
exact E step for this model.
z(1)
n−1
z(1)
n
z(1)
n+1
z(2)
n−1
z(2)
n
z(2)
n+1
xn−1
xn
xn+1


---
**Page 604**
13.3. Linear Dynamical Systems
635
and so we can transform the model into an equivalent standard HMM having a single
chain of latent variables each of which has KM latent states. We can then run the
standard forward-backward recursions in the E step. This has computational com-
plexity O(NK2M) that is exponential in the number M of latent chains and so will
be intractable for anything other than small values of M. One solution would be
to use sampling methods (discussed in Chapter 11). As an elegant deterministic al-
ternative, Ghahramani and Jordan (1997) exploited variational inference techniques
Section 10.1
to obtain a tractable algorithm for approximate inference. This can be done using
a simple variational posterior distribution that is fully factorized with respect to the
latent variables, or alternatively by using a more powerful approach in which the
variational distribution is described by independent Markov chains corresponding to
the chains of latent variables in the original model. In the latter case, the variational
inference algorithms involves running independent forward and backward recursions
along each chain, which is computationally efﬁcient and yet is also able to capture
correlations between variables within the same chain.
Clearly, there are many possible probabilistic structures that can be constructed
according to the needs of particular applications. Graphical models provide a general
technique for motivating, describing, and analysing such structures, and variational
methods provide a powerful framework for performing inference in those models for
which exact solution is intractable.
13.3. Linear Dynamical Systems
In order to motivate the concept of linear dynamical systems, let us consider the
following simple problem, which often arises in practical settings. Suppose we wish
to measure the value of an unknown quantity z using a noisy sensor that returns a
observation x representing the value of z plus zero-mean Gaussian noise. Given a
single measurement, our best guess for z is to assume that z = x. However, we
can improve our estimate for z by taking lots of measurements and averaging them,
because the random noise terms will tend to cancel each other. Now let’s make the
situation more complicated by assuming that we wish to measure a quantity z that
is changing over time. We can take regular measurements of x so that at some point
in time we have obtained x1, . . . , xN and we wish to ﬁnd the corresponding values
z1, . . . , xN. If we simply average the measurements, the error due to random noise
will be reduced, but unfortunately we will just obtain a single averaged estimate, in
which we have averaged over the changing value of z, thereby introducing a new
source of error.
Intuitively, we could imagine doing a bit better as follows. To estimate the value
of zN, we take only the most recent few measurements, say xN−L, . . . , xN and just
average these. If z is changing slowly, and the random noise level in the sensor is
high, it would make sense to choose a relatively long window of observations to
average. Conversely, if the signal is changing quickly, and the noise levels are small,
we might be better just to use xN directly as our estimate of zN. Perhaps we could
do even better if we take a weighted average, in which more recent measurements

