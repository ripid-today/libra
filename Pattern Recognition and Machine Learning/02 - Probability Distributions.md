# 02 - Probability Distributions
*Pages 67-136 from Pattern Recognition and Machine Learning*

---
**Page 67**
1.6. Information Theory
49
y −t
|y −t|q
q = 0.3
−2
−1
0
1
2
0
1
2
y −t
|y −t|q
q = 1
−2
−1
0
1
2
0
1
2
y −t
|y −t|q
q = 2
−2
−1
0
1
2
0
1
2
y −t
|y −t|q
q = 10
−2
−1
0
1
2
0
1
2
Figure 1.29
Plots of the quantity Lq = |y −t|q for various values of q.
h(x) = −log2 p(x)
(1.92)
where the negative sign ensures that information is positive or zero. Note that low
probability events x correspond to high information content. The choice of basis
for the logarithm is arbitrary, and for the moment we shall adopt the convention
prevalent in information theory of using logarithms to the base of 2. In this case, as
we shall see shortly, the units of h(x) are bits (‘binary digits’).
Now suppose that a sender wishes to transmit the value of a random variable to
a receiver. The average amount of information that they transmit in the process is
obtained by taking the expectation of (1.92) with respect to the distribution p(x) and
is given by
H[x] = −

x
p(x) log2 p(x).
(1.93)
This important quantity is called the entropy of the random variable x. Note that
limp→0 p ln p = 0 and so we shall take p(x) ln p(x) = 0 whenever we encounter a
value for x such that p(x) = 0.
So far we have given a rather heuristic motivation for the deﬁnition of informa-


---
**Page 68**
50
1. INTRODUCTION
tion (1.92) and the corresponding entropy (1.93). We now show that these deﬁnitions
indeed possess useful properties. Consider a random variable x having 8 possible
states, each of which is equally likely. In order to communicate the value of x to
a receiver, we would need to transmit a message of length 3 bits. Notice that the
entropy of this variable is given by
H[x] = −8 × 1
8 log2
1
8 = 3 bits.
Now consider an example (Cover and Thomas, 1991) of a variable having 8 pos-
sible states {a, b, c, d, e, f, g, h} for which the respective probabilities are given by
( 1
2, 1
4, 1
8, 1
16, 1
64, 1
64, 1
64, 1
64). The entropy in this case is given by
H[x] = −1
2 log2
1
2 −1
4 log2
1
4 −1
8 log2
1
8 −1
16 log2
1
16 −4
64 log2
1
64 = 2 bits.
We see that the nonuniform distribution has a smaller entropy than the uniform one,
and we shall gain some insight into this shortly when we discuss the interpretation of
entropy in terms of disorder. For the moment, let us consider how we would transmit
the identity of the variable’s state to a receiver. We could do this, as before, using
a 3-bit number. However, we can take advantage of the nonuniform distribution by
using shorter codes for the more probable events, at the expense of longer codes for
the less probable events, in the hope of getting a shorter average code length. This
can be done by representing the states {a, b, c, d, e, f, g, h} using, for instance, the
following set of code strings: 0, 10, 110, 1110, 111100, 111101, 111110, 111111.
The average length of the code that has to be transmitted is then
average code length = 1
2 × 1 + 1
4 × 2 + 1
8 × 3 + 1
16 × 4 + 4 × 1
64 × 6 = 2 bits
which again is the same as the entropy of the random variable. Note that shorter code
strings cannot be used because it must be possible to disambiguate a concatenation
of such strings into its component parts. For instance, 11001110 decodes uniquely
into the state sequence c, a, d.
This relation between entropy and shortest coding length is a general one. The
noiseless coding theorem (Shannon, 1948) states that the entropy is a lower bound
on the number of bits needed to transmit the state of a random variable.
From now on, we shall switch to the use of natural logarithms in deﬁning en-
tropy, as this will provide a more convenient link with ideas elsewhere in this book.
In this case, the entropy is measured in units of ‘nats’ instead of bits, which differ
simply by a factor of ln 2.
We have introduced the concept of entropy in terms of the average amount of
information needed to specify the state of a random variable. In fact, the concept of
entropy has much earlier origins in physics where it was introduced in the context
of equilibrium thermodynamics and later given a deeper interpretation as a measure
of disorder through developments in statistical mechanics. We can understand this
alternative view of entropy by considering a set of N identical objects that are to be
divided amongst a set of bins, such that there are ni objects in the ith bin. Consider


---
**Page 69**
1.6. Information Theory
51
the number of different ways of allocating the objects to the bins. There are N
ways to choose the ﬁrst object, (N −1) ways to choose the second object, and
so on, leading to a total of N! ways to allocate all N objects to the bins, where N!
(pronounced ‘factorial N’) denotes the product N ×(N −1)×· · ·×2×1. However,
we don’t wish to distinguish between rearrangements of objects within each bin. In
the ith bin there are ni! ways of reordering the objects, and so the total number of
ways of allocating the N objects to the bins is given by
W =
N!

i ni!
(1.94)
which is called the multiplicity. The entropy is then deﬁned as the logarithm of the
multiplicity scaled by an appropriate constant
H = 1
N ln W = 1
N ln N! −1
N

i
ln ni!.
(1.95)
We now consider the limit N →∞, in which the fractions ni/N are held ﬁxed, and
apply Stirling’s approximation
ln N! ≃N ln N −N
(1.96)
which gives
H = −lim
N→∞

i
ni
N

ln
ni
N

= −

i
pi ln pi
(1.97)
where we have used 
i ni = N. Here pi = limN→∞(ni/N) is the probability
of an object being assigned to the ith bin. In physics terminology, the speciﬁc ar-
rangements of objects in the bins is called a microstate, and the overall distribution
of occupation numbers, expressed through the ratios ni/N, is called a macrostate.
The multiplicity W is also known as the weight of the macrostate.
We can interpret the bins as the states xi of a discrete random variable X, where
p(X = xi) = pi. The entropy of the random variable X is then
H[p] = −

i
p(xi) ln p(xi).
(1.98)
Distributions p(xi) that are sharply peaked around a few values will have a relatively
low entropy, whereas those that are spread more evenly across many values will
have higher entropy, as illustrated in Figure 1.30. Because 0 ⩽pi ⩽1, the entropy
is nonnegative, and it will equal its minimum value of 0 when one of the pi =
1 and all other pj̸=i = 0. The maximum entropy conﬁguration can be found by
maximizing H using a Lagrange multiplier to enforce the normalization constraint
Appendix E
on the probabilities. Thus we maximize
H = −

i
p(xi) ln p(xi) + λ

i
p(xi) −1

(1.99)


---
**Page 70**
52
1. INTRODUCTION
probabilities
H = 1.77
0
0.25
0.5
probabilities
H = 3.09
0
0.25
0.5
Figure 1.30
Histograms of two probability distributions over 30 bins illustrating the higher value of the entropy
H for the broader distribution. The largest entropy would arise from a uniform distribution that would give H =
−ln(1/30) = 3.40.
from which we ﬁnd that all of the p(xi) are equal and are given by p(xi) = 1/M
where M is the total number of states xi. The corresponding value of the entropy
is then H = ln M. This result can also be derived from Jensen’s inequality (to be
discussed shortly). To verify that the stationary point is indeed a maximum, we can
Exercise 1.29
evaluate the second derivative of the entropy, which gives
∂H
∂p(xi)∂p(xj) = −Iij
1
pi
(1.100)
where Iij are the elements of the identity matrix.
We can extend the deﬁnition of entropy to include distributions p(x) over con-
tinuous variables x as follows. First divide x into bins of width ∆. Then, assuming
p(x) is continuous, the mean value theorem (Weisstein, 1999) tells us that, for each
such bin, there must exist a value xi such that
 (i+1)∆
i∆
p(x) dx = p(xi)∆.
(1.101)
We can now quantize the continuous variable x by assigning any value x to the value
xi whenever x falls in the ith bin. The probability of observing the value xi is then
p(xi)∆. This gives a discrete distribution for which the entropy takes the form
H∆= −

i
p(xi)∆ln (p(xi)∆) = −

i
p(xi)∆ln p(xi) −ln ∆
(1.102)
where we have used 
i p(xi)∆= 1, which follows from (1.101). We now omit
the second term −ln ∆on the right-hand side of (1.102) and then consider the limit


---
**Page 71**
1.6. Information Theory
53
∆→0. The ﬁrst term on the right-hand side of (1.102) will approach the integral of
p(x) ln p(x) in this limit so that
lim
∆→0

i
p(xi)∆ln p(xi)

= −

p(x) ln p(x) dx
(1.103)
where the quantity on the right-hand side is called the differential entropy. We see
that the discrete and continuous forms of the entropy differ by a quantity ln ∆, which
diverges in the limit ∆→0. This reﬂects the fact that to specify a continuous
variable very precisely requires a large number of bits. For a density deﬁned over
multiple continuous variables, denoted collectively by the vector x, the differential
entropy is given by
H[x] = −

p(x) ln p(x) dx.
(1.104)
In the case of discrete distributions, we saw that the maximum entropy con-
ﬁguration corresponded to an equal distribution of probabilities across the possible
states of the variable. Let us now consider the maximum entropy conﬁguration for
a continuous variable. In order for this maximum to be well deﬁned, it will be nec-
essary to constrain the ﬁrst and second moments of p(x) as well as preserving the
normalization constraint. We therefore maximize the differential entropy with the
Ludwig Boltzmann
1844–1906
Ludwig Eduard Boltzmann was an
Austrian physicist who created the
ﬁeld of statistical mechanics. Prior
to Boltzmann, the concept of en-
tropy
was
already
known
from
classical thermodynamics where it
quantiﬁes the fact that when we take energy from a
system, not all of that energy is typically available
to do useful work. Boltzmann showed that the ther-
modynamic entropy S, a macroscopic quantity, could
be related to the statistical properties at the micro-
scopic level. This is expressed through the famous
equation S
= k ln W in which W represents the
number of possible microstates in a macrostate, and
k ≃1.38 × 10−23 (in units of Joules per Kelvin) is
known as Boltzmann’s constant.
Boltzmann’s ideas
were disputed by many scientists of they day. One dif-
ﬁculty they saw arose from the second law of thermo-
dynamics, which states that the entropy of a closed
system tends to increase with time. By contrast, at
the microscopic level the classical Newtonian equa-
tions of physics are reversible, and so they found it
difﬁcult to see how the latter could explain the for-
mer.
They didn’t fully appreciate Boltzmann’s argu-
ments, which were statistical in nature and which con-
cluded not that entropy could never decrease over
time but simply that with overwhelming probability it
would generally increase. Boltzmann even had a long-
running dispute with the editor of the leading German
physics journal who refused to let him refer to atoms
and molecules as anything other than convenient the-
oretical constructs. The continued attacks on his work
lead to bouts of depression, and eventually he com-
mitted suicide. Shortly after Boltzmann’s death, new
experiments by Perrin on colloidal suspensions veri-
ﬁed his theories and conﬁrmed the value of the Boltz-
mann constant. The equation S = k ln W is carved on
Boltzmann’s tombstone.


---
**Page 72**
54
1. INTRODUCTION
three constraints
 ∞
−∞
p(x) dx
=
1
(1.105)
 ∞
−∞
xp(x) dx
=
µ
(1.106)
 ∞
−∞
(x −µ)2p(x) dx
=
σ2.
(1.107)
The constrained maximization can be performed using Lagrange multipliers so that
Appendix E
we maximize the following functional with respect to p(x)
−
 ∞
−∞
p(x) ln p(x) dx + λ1
 ∞
−∞
p(x) dx −1

+λ2
 ∞
−∞
xp(x) dx −µ

+ λ3
 ∞
−∞
(x −µ)2p(x) dx −σ2

.
Using the calculus of variations, we set the derivative of this functional to zero giving
Appendix D
p(x) = exp 
−1 + λ1 + λ2x + λ3(x −µ)2
.
(1.108)
The Lagrange multipliers can be found by back substitution of this result into the
three constraint equations, leading ﬁnally to the result
Exercise 1.34
p(x) =
1
(2πσ2)1/2 exp

−(x −µ)2
2σ2

(1.109)
and so the distribution that maximizes the differential entropy is the Gaussian. Note
that we did not constrain the distribution to be nonnegative when we maximized the
entropy. However, because the resulting distribution is indeed nonnegative, we see
with hindsight that such a constraint is not necessary.
If we evaluate the differential entropy of the Gaussian, we obtain
Exercise 1.35
H[x] = 1
2

1 + ln(2πσ2)
.
(1.110)
Thus we see again that the entropy increases as the distribution becomes broader,
i.e., as σ2 increases. This result also shows that the differential entropy, unlike the
discrete entropy, can be negative, because H(x) < 0 in (1.110) for σ2 < 1/(2πe).
Suppose we have a joint distribution p(x, y) from which we draw pairs of values
of x and y. If a value of x is already known, then the additional information needed
to specify the corresponding value of y is given by −ln p(y|x). Thus the average
additional information needed to specify y can be written as
H[y|x] = −

p(y, x) ln p(y|x) dy dx
(1.111)


---
**Page 73**
1.6. Information Theory
55
which is called the conditional entropy of y given x. It is easily seen, using the
product rule, that the conditional entropy satisﬁes the relation
Exercise 1.37
H[x, y] = H[y|x] + H[x]
(1.112)
where H[x, y] is the differential entropy of p(x, y) and H[x] is the differential en-
tropy of the marginal distribution p(x). Thus the information needed to describe x
and y is given by the sum of the information needed to describe x alone plus the
additional information required to specify y given x.
1.6.1
Relative entropy and mutual information
So far in this section, we have introduced a number of concepts from information
theory, including the key notion of entropy. We now start to relate these ideas to
pattern recognition. Consider some unknown distribution p(x), and suppose that
we have modelled this using an approximating distribution q(x). If we use q(x) to
construct a coding scheme for the purpose of transmitting values of x to a receiver,
then the average additional amount of information (in nats) required to specify the
value of x (assuming we choose an efﬁcient coding scheme) as a result of using q(x)
instead of the true distribution p(x) is given by
KL(p∥q)
=
−

p(x) ln q(x) dx −

−

p(x) ln p(x) dx

=
−

p(x) ln
q(x)
p(x)

dx.
(1.113)
This is known as the relative entropy or Kullback-Leibler divergence, or KL diver-
gence (Kullback and Leibler, 1951), between the distributions p(x) and q(x). Note
that it is not a symmetrical quantity, that is to say KL(p∥q) ̸≡KL(q∥p).
We now show that the Kullback-Leibler divergence satisﬁes KL(p∥q) ⩾0 with
equality if, and only if, p(x) = q(x). To do this we ﬁrst introduce the concept of
convex functions. A function f(x) is said to be convex if it has the property that
every chord lies on or above the function, as shown in Figure 1.31. Any value of x
in the interval from x = a to x = b can be written in the form λa + (1 −λ)b where
0 ⩽λ ⩽1. The corresponding point on the chord is given by λf(a) + (1 −λ)f(b),
Claude Shannon
1916–2001
After graduating from Michigan and
MIT, Shannon joined the AT&T Bell
Telephone laboratories in 1941. His
paper ‘A Mathematical Theory of
Communication’ published in the
Bell System Technical Journal in
1948 laid the foundations for modern information the-
ory. This paper introduced the word ‘bit’, and his con-
cept that information could be sent as a stream of 1s
and 0s paved the way for the communications revo-
lution. It is said that von Neumann recommended to
Shannon that he use the term entropy, not only be-
cause of its similarity to the quantity used in physics,
but also because “nobody knows what entropy really
is, so in any discussion you will always have an advan-
tage”.


---
**Page 74**
56
1. INTRODUCTION
Figure 1.31
A convex function f(x) is one for which ev-
ery chord (shown in blue) lies on or above
the function (shown in red).
x
a
b
xλ
chord
xλ
f(x)
and the corresponding value of the function is f (λa + (1 −λ)b). Convexity then
implies
f(λa + (1 −λ)b) ⩽λf(a) + (1 −λ)f(b).
(1.114)
This is equivalent to the requirement that the second derivative of the function be
everywhere positive. Examples of convex functions are x ln x (for x > 0) and x2. A
Exercise 1.36
function is called strictly convex if the equality is satisﬁed only for λ = 0 and λ = 1.
If a function has the opposite property, namely that every chord lies on or below the
function, it is called concave, with a corresponding deﬁnition for strictly concave. If
a function f(x) is convex, then −f(x) will be concave.
Using the technique of proof by induction, we can show from (1.114) that a
Exercise 1.38
convex function f(x) satisﬁes
f
 M

i=1
λixi

⩽
M

i=1
λif(xi)
(1.115)
where λi ⩾0 and 
i λi = 1, for any set of points {xi}. The result (1.115) is
known as Jensen’s inequality. If we interpret the λi as the probability distribution
over a discrete variable x taking the values {xi}, then (1.115) can be written
f (E[x]) ⩽E[f(x)]
(1.116)
where E[·] denotes the expectation. For continuous variables, Jensen’s inequality
takes the form
f

xp(x) dx

⩽

f(x)p(x) dx.
(1.117)
We can apply Jensen’s inequality in the form (1.117) to the Kullback-Leibler
divergence (1.113) to give
KL(p∥q) = −

p(x) ln
q(x)
p(x)

dx ⩾−ln

q(x) dx = 0
(1.118)


---
**Page 75**
1.6. Information Theory
57
where we have used the fact that −ln x is a convex function, together with the nor-
malization condition 
q(x) dx = 1. In fact, −ln x is a strictly convex function,
so the equality will hold if, and only if, q(x) = p(x) for all x. Thus we can in-
terpret the Kullback-Leibler divergence as a measure of the dissimilarity of the two
distributions p(x) and q(x).
We see that there is an intimate relationship between data compression and den-
sity estimation (i.e., the problem of modelling an unknown probability distribution)
because the most efﬁcient compression is achieved when we know the true distri-
bution. If we use a distribution that is different from the true one, then we must
necessarily have a less efﬁcient coding, and on average the additional information
that must be transmitted is (at least) equal to the Kullback-Leibler divergence be-
tween the two distributions.
Suppose that data is being generated from an unknown distribution p(x) that we
wish to model. We can try to approximate this distribution using some parametric
distribution q(x|θ), governed by a set of adjustable parameters θ, for example a
multivariate Gaussian. One way to determine θ is to minimize the Kullback-Leibler
divergence between p(x) and q(x|θ) with respect to θ. We cannot do this directly
because we don’t know p(x). Suppose, however, that we have observed a ﬁnite set
of training points xn, for n = 1, . . . , N, drawn from p(x). Then the expectation
with respect to p(x) can be approximated by a ﬁnite sum over these points, using
(1.35), so that
KL(p∥q) ≃
N

n=1
{−ln q(xn|θ) + ln p(xn)} .
(1.119)
The second term on the right-hand side of (1.119) is independent of θ, and the ﬁrst
term is the negative log likelihood function for θ under the distribution q(x|θ) eval-
uated using the training set. Thus we see that minimizing this Kullback-Leibler
divergence is equivalent to maximizing the likelihood function.
Now consider the joint distribution between two sets of variables x and y given
by p(x, y). If the sets of variables are independent, then their joint distribution will
factorize into the product of their marginals p(x, y) = p(x)p(y). If the variables are
not independent, we can gain some idea of whether they are ‘close’ to being indepen-
dent by considering the Kullback-Leibler divergence between the joint distribution
and the product of the marginals, given by
I[x, y]
≡
KL(p(x, y)∥p(x)p(y))
=
−

p(x, y) ln
p(x)p(y)
p(x, y)

dx dy
(1.120)
which is called the mutual information between the variables x and y. From the
properties of the Kullback-Leibler divergence, we see that I(x, y) ⩾0 with equal-
ity if, and only if, x and y are independent. Using the sum and product rules of
probability, we see that the mutual information is related to the conditional entropy
through
Exercise 1.41
I[x, y] = H[x] −H[x|y] = H[y] −H[y|x].
(1.121)


---
**Page 76**
58
1. INTRODUCTION
Thus we can view the mutual information as the reduction in the uncertainty about x
by virtue of being told the value of y (or vice versa). From a Bayesian perspective,
we can view p(x) as the prior distribution for x and p(x|y) as the posterior distribu-
tion after we have observed new data y. The mutual information therefore represents
the reduction in uncertainty about x as a consequence of the new observation y.
Exercises
1.1
(⋆) www
Consider the sum-of-squares error function given by (1.2) in which
the function y(x, w) is given by the polynomial (1.1). Show that the coefﬁcients
w = {wi} that minimize this error function are given by the solution to the following
set of linear equations
M

j=0
Aijwj = Ti
(1.122)
where
Aij =
N

n=1
(xn)i+j,
Ti =
N

n=1
(xn)itn.
(1.123)
Here a sufﬁx i or j denotes the index of a component, whereas (x)i denotes x raised
to the power of i.
1.2
(⋆)
Write down the set of coupled linear equations, analogous to (1.122), satisﬁed
by the coefﬁcients wi which minimize the regularized sum-of-squares error function
given by (1.4).
1.3
(⋆⋆)
Suppose that we have three coloured boxes r (red), b (blue), and g (green).
Box r contains 3 apples, 4 oranges, and 3 limes, box b contains 1 apple, 1 orange,
and 0 limes, and box g contains 3 apples, 3 oranges, and 4 limes. If a box is chosen
at random with probabilities p(r) = 0.2, p(b) = 0.2, p(g) = 0.6, and a piece of
fruit is removed from the box (with equal probability of selecting any of the items in
the box), then what is the probability of selecting an apple? If we observe that the
selected fruit is in fact an orange, what is the probability that it came from the green
box?
1.4
(⋆⋆) www
Consider a probability density px(x) deﬁned over a continuous vari-
able x, and suppose that we make a nonlinear change of variable using x = g(y),
so that the density transforms according to (1.27). By differentiating (1.27), show
that the location y of the maximum of the density in y is not in general related to the
location x of the maximum of the density over x by the simple functional relation
x = g(y) as a consequence of the Jacobian factor. This shows that the maximum
of a probability density (in contrast to a simple function) is dependent on the choice
of variable. Verify that, in the case of a linear transformation, the location of the
maximum transforms in the same way as the variable itself.
1.5
(⋆) Using the deﬁnition (1.38) show that var[f(x)] satisﬁes (1.39).


---
**Page 77**
Exercises
59
1.6
(⋆)
Show that if two variables x and y are independent, then their covariance is
zero.
1.7
(⋆⋆) www
In this exercise, we prove the normalization condition (1.48) for the
univariate Gaussian. To do this consider, the integral
I =
 ∞
−∞
exp

−1
2σ2 x2

dx
(1.124)
which we can evaluate by ﬁrst writing its square in the form
I2 =
 ∞
−∞
 ∞
−∞
exp

−1
2σ2 x2 −
1
2σ2 y2

dx dy.
(1.125)
Now make the transformation from Cartesian coordinates (x, y) to polar coordinates
(r, θ) and then substitute u = r2. Show that, by performing the integrals over θ and
u, and then taking the square root of both sides, we obtain
I = 
2πσ21/2 .
(1.126)
Finally, use this result to show that the Gaussian distribution N(x|µ, σ2) is normal-
ized.
1.8
(⋆⋆) www
By using a change of variables, verify that the univariate Gaussian
distribution given by (1.46) satisﬁes (1.49). Next, by differentiating both sides of the
normalization condition
 ∞
−∞
N

x|µ, σ2
dx = 1
(1.127)
with respect to σ2, verify that the Gaussian satisﬁes (1.50). Finally, show that (1.51)
holds.
1.9
(⋆) www
Show that the mode (i.e. the maximum) of the Gaussian distribution
(1.46) is given by µ. Similarly, show that the mode of the multivariate Gaussian
(1.52) is given by µ.
1.10
(⋆) www
Suppose that the two variables x and z are statistically independent.
Show that the mean and variance of their sum satisﬁes
E[x + z]
=
E[x] + E[z]
(1.128)
var[x + z]
=
var[x] + var[z].
(1.129)
1.11
(⋆) By setting the derivatives of the log likelihood function (1.54) with respect to µ
and σ2 equal to zero, verify the results (1.55) and (1.56).


---
**Page 78**
60
1. INTRODUCTION
1.12
(⋆⋆) www
Using the results (1.49) and (1.50), show that
E[xnxm] = µ2 + Inmσ2
(1.130)
where xn and xm denote data points sampled from a Gaussian distribution with mean
µ and variance σ2, and Inm satisﬁes Inm = 1 if n = m and Inm = 0 otherwise.
Hence prove the results (1.57) and (1.58).
1.13
(⋆) Suppose that the variance of a Gaussian is estimated using the result (1.56) but
with the maximum likelihood estimate µML replaced with the true value µ of the
mean. Show that this estimator has the property that its expectation is given by the
true variance σ2.
1.14
(⋆⋆)
Show that an arbitrary square matrix with elements wij can be written in
the form wij = wS
ij + wA
ij where wS
ij and wA
ij are symmetric and anti-symmetric
matrices, respectively, satisfying wS
ij = wS
ji and wA
ij = −wA
ji for all i and j. Now
consider the second order term in a higher order polynomial in D dimensions, given
by
D

i=1
D

j=1
wijxixj.
(1.131)
Show that
D

i=1
D

j=1
wijxixj =
D

i=1
D

j=1
wS
ijxixj
(1.132)
so that the contribution from the anti-symmetric matrix vanishes. We therefore see
that, without loss of generality, the matrix of coefﬁcients wij can be chosen to be
symmetric, and so not all of the D2 elements of this matrix can be chosen indepen-
dently. Show that the number of independent parameters in the matrix wS
ij is given
by D(D + 1)/2.
1.15
(⋆⋆⋆) www
In this exercise and the next, we explore how the number of indepen-
dent parameters in a polynomial grows with the order M of the polynomial and with
the dimensionality D of the input space. We start by writing down the M th order
term for a polynomial in D dimensions in the form
D

i1=1
D

i2=1
· · ·
D

iM =1
wi1i2···iMxi1xi2 · · · xiM.
(1.133)
The coefﬁcients wi1i2···iM comprise DM elements, but the number of independent
parameters is signiﬁcantly fewer due to the many interchange symmetries of the
factor xi1xi2 · · · xiM . Begin by showing that the redundancy in the coefﬁcients can
be removed by rewriting this M th order term in the form
D

i1=1
i1

i2=1
· · ·
iM−1

iM =1
wi1i2···iMxi1xi2 · · · xiM.
(1.134)


---
**Page 79**
Exercises
61
Note that the precise relationship between the w coefﬁcients and w coefﬁcients need
not be made explicit. Use this result to show that the number of independent param-
eters n(D, M), which appear at order M, satisﬁes the following recursion relation
n(D, M) =
D

i=1
n(i, M −1).
(1.135)
Next use proof by induction to show that the following result holds
D

i=1
(i + M −2)!
(i −1)! (M −1)! = (D + M −1)!
(D −1)! M!
(1.136)
which can be done by ﬁrst proving the result for D = 1 and arbitrary M by making
use of the result 0! = 1, then assuming it is correct for dimension D and verifying
that it is correct for dimension D + 1. Finally, use the two previous results, together
with proof by induction, to show
n(D, M) = (D + M −1)!
(D −1)! M! .
(1.137)
To do this, ﬁrst show that the result is true for M = 2, and any value of D ⩾1,
by comparison with the result of Exercise 1.14. Then make use of (1.135), together
with (1.136), to show that, if the result holds at order M −1, then it will also hold at
order M
1.16
(⋆⋆⋆) In Exercise 1.15, we proved the result (1.135) for the number of independent
parameters in the M th order term of a D-dimensional polynomial. We now ﬁnd an
expression for the total number N(D, M) of independent parameters in all of the
terms up to and including the M6th order. First show that N(D, M) satisﬁes
N(D, M) =
M

m=0
n(D, m)
(1.138)
where n(D, m) is the number of independent parameters in the term of order m.
Now make use of the result (1.137), together with proof by induction, to show that
N(d, M) = (D + M)!
D! M!
.
(1.139)
This can be done by ﬁrst proving that the result holds for M = 0 and arbitrary
D ⩾1, then assuming that it holds at order M, and hence showing that it holds at
order M + 1. Finally, make use of Stirling’s approximation in the form
n! ≃nne−n
(1.140)
for large n to show that, for D ≫M, the quantity N(D, M) grows like DM,
and for M ≫D it grows like M D. Consider a cubic (M = 3) polynomial in D
dimensions, and evaluate numerically the total number of independent parameters
for (i) D = 10 and (ii) D = 100, which correspond to typical small-scale and
medium-scale machine learning applications.


---
**Page 80**
62
1. INTRODUCTION
1.17
(⋆⋆) www
The gamma function is deﬁned by
Γ(x) ≡
 ∞
0
ux−1e−u du.
(1.141)
Using integration by parts, prove the relation Γ(x + 1) = xΓ(x). Show also that
Γ(1) = 1 and hence that Γ(x + 1) = x! when x is an integer.
1.18
(⋆⋆) www
We can use the result (1.126) to derive an expression for the surface
area SD, and the volume VD, of a sphere of unit radius in D dimensions. To do this,
consider the following result, which is obtained by transforming from Cartesian to
polar coordinates
D

i=1
 ∞
−∞
e−x2
i dxi = SD
 ∞
0
e−r2rD−1 dr.
(1.142)
Using the deﬁnition (1.141) of the Gamma function, together with (1.126), evaluate
both sides of this equation, and hence show that
SD = 2πD/2
Γ(D/2).
(1.143)
Next, by integrating with respect to radius from 0 to 1, show that the volume of the
unit sphere in D dimensions is given by
VD = SD
D .
(1.144)
Finally, use the results Γ(1) = 1 and Γ(3/2) = √π/2 to show that (1.143) and
(1.144) reduce to the usual expressions for D = 2 and D = 3.
1.19
(⋆⋆)
Consider a sphere of radius a in D-dimensions together with the concentric
hypercube of side 2a, so that the sphere touches the hypercube at the centres of each
of its sides. By using the results of Exercise 1.18, show that the ratio of the volume
of the sphere to the volume of the cube is given by
volume of sphere
volume of cube =
πD/2
D2D−1Γ(D/2).
(1.145)
Now make use of Stirling’s formula in the form
Γ(x + 1) ≃(2π)1/2e−xxx+1/2
(1.146)
which is valid for x ≫1, to show that, as D →∞, the ratio (1.145) goes to zero.
Show also that the ratio of the distance from the centre of the hypercube to one of
the corners, divided by the perpendicular distance to one of the sides, is
√
D, which
therefore goes to ∞as D →∞. From these results we see that, in a space of high
dimensionality, most of the volume of a cube is concentrated in the large number of
corners, which themselves become very long ‘spikes’!


---
**Page 81**
Exercises
63
1.20
(⋆⋆) www
In this exercise, we explore the behaviour of the Gaussian distribution
in high-dimensional spaces. Consider a Gaussian distribution in D dimensions given
by
p(x) =
1
(2πσ2)D/2 exp

−∥x∥2
2σ2

.
(1.147)
We wish to ﬁnd the density with respect to radius in polar coordinates in which the
direction variables have been integrated out. To do this, show that the integral of
the probability density over a thin shell of radius r and thickness ϵ, where ϵ ≪1, is
given by p(r)ϵ where
p(r) =
SDrD−1
(2πσ2)D/2 exp

−r2
2σ2

(1.148)
where SD is the surface area of a unit sphere in D dimensions. Show that the function
p(r) has a single stationary point located, for large D, at r ≃
√
Dσ. By considering
p(r + ϵ) where ϵ ≪r, show that for large D,
p(r + ϵ) = p(r) exp

−3ϵ2
2σ2

(1.149)
which shows that r is a maximum of the radial probability density and also that p(r)
decays exponentially away from its maximum at r with length scale σ. We have
already seen that σ ≪r for large D, and so we see that most of the probability
mass is concentrated in a thin shell at large radius. Finally, show that the probability
density p(x) is larger at the origin than at the radius r by a factor of exp(D/2).
We therefore see that most of the probability mass in a high-dimensional Gaussian
distribution is located at a different radius from the region of high probability density.
This property of distributions in spaces of high dimensionality will have important
consequences when we consider Bayesian inference of model parameters in later
chapters.
1.21
(⋆⋆)
Consider two nonnegative numbers a and b, and show that, if a ⩽b, then
a ⩽(ab)1/2. Use this result to show that, if the decision regions of a two-class
classiﬁcation problem are chosen to minimize the probability of misclassiﬁcation,
this probability will satisfy
p(mistake) ⩽

{p(x, C1)p(x, C2)}1/2 dx.
(1.150)
1.22
(⋆) www
Given a loss matrix with elements Lkj, the expected risk is minimized
if, for each x, we choose the class that minimizes (1.81). Verify that, when the
loss matrix is given by Lkj = 1 −Ikj, where Ikj are the elements of the identity
matrix, this reduces to the criterion of choosing the class having the largest posterior
probability. What is the interpretation of this form of loss matrix?
1.23
(⋆)
Derive the criterion for minimizing the expected loss when there is a general
loss matrix and general prior probabilities for the classes.


---
**Page 82**
64
1. INTRODUCTION
1.24
(⋆⋆) www
Consider a classiﬁcation problem in which the loss incurred when
an input vector from class Ck is classiﬁed as belonging to class Cj is given by the
loss matrix Lkj, and for which the loss incurred in selecting the reject option is λ.
Find the decision criterion that will give the minimum expected loss. Verify that this
reduces to the reject criterion discussed in Section 1.5.3 when the loss matrix is given
by Lkj = 1 −Ikj. What is the relationship between λ and the rejection threshold θ?
1.25
(⋆) www
Consider the generalization of the squared loss function (1.87) for a
single target variable t to the case of multiple target variables described by the vector
t given by
E[L(t, y(x))] =

∥y(x) −t∥2p(x, t) dx dt.
(1.151)
Using the calculus of variations, show that the function y(x) for which this expected
loss is minimized is given by y(x) = Et[t|x]. Show that this result reduces to (1.89)
for the case of a single target variable t.
1.26
(⋆)
By expansion of the square in (1.151), derive a result analogous to (1.90) and
hence show that the function y(x) that minimizes the expected squared loss for the
case of a vector t of target variables is again given by the conditional expectation of
t.
1.27
(⋆⋆) www
Consider the expected loss for regression problems under the Lq loss
function given by (1.91). Write down the condition that y(x) must satisfy in order
to minimize E[Lq]. Show that, for q = 1, this solution represents the conditional
median, i.e., the function y(x) such that the probability mass for t < y(x) is the
same as for t ⩾y(x). Also show that the minimum expected Lq loss for q →0 is
given by the conditional mode, i.e., by the function y(x) equal to the value of t that
maximizes p(t|x) for each x.
1.28
(⋆) In Section 1.6, we introduced the idea of entropy h(x) as the information gained
on observing the value of a random variable x having distribution p(x). We saw
that, for independent variables x and y for which p(x, y) = p(x)p(y), the entropy
functions are additive, so that h(x, y) = h(x) + h(y). In this exercise, we derive the
relation between h and p in the form of a function h(p). First show that h(p2) =
2h(p), and hence by induction that h(pn) = nh(p) where n is a positive integer.
Hence show that h(pn/m) = (n/m)h(p) where m is also a positive integer. This
implies that h(px) = xh(p) where x is a positive rational number, and hence by
continuity when it is a positive real number. Finally, show that this implies h(p)
must take the form h(p) ∝ln p.
1.29
(⋆) www
Consider an M-state discrete random variable x, and use Jensen’s in-
equality in the form (1.115) to show that the entropy of its distribution p(x) satisﬁes
H[x] ⩽ln M.
1.30
(⋆⋆)
Evaluate the Kullback-Leibler divergence (1.113) between two Gaussians
p(x) = N(x|µ, σ2) and q(x) = N(x|m, s2).


---
**Page 83**
Exercises
65
Table 1.3
The joint distribution p(x, y) for two binary variables
x and y used in Exercise 1.39.
y
0
1
x
0
1/3
1/3
1
0
1/3
1.31
(⋆⋆) www
Consider two variables x and y having joint distribution p(x, y). Show
that the differential entropy of this pair of variables satisﬁes
H[x, y] ⩽H[x] + H[y]
(1.152)
with equality if, and only if, x and y are statistically independent.
1.32
(⋆)
Consider a vector x of continuous variables with distribution p(x) and corre-
sponding entropy H[x]. Suppose that we make a nonsingular linear transformation
of x to obtain a new variable y = Ax. Show that the corresponding entropy is given
by H[y] = H[x] + ln |A| where |A| denotes the determinant of A.
1.33
(⋆⋆)
Suppose that the conditional entropy H[y|x] between two discrete random
variables x and y is zero. Show that, for all values of x such that p(x) > 0, the
variable y must be a function of x, in other words for each x there is only one value
of y such that p(y|x) ̸= 0.
1.34
(⋆⋆) www
Use the calculus of variations to show that the stationary point of the
functional (1.108) is given by (1.108). Then use the constraints (1.105), (1.106),
and (1.107) to eliminate the Lagrange multipliers and hence show that the maximum
entropy solution is given by the Gaussian (1.109).
1.35
(⋆) www
Use the results (1.106) and (1.107) to show that the entropy of the
univariate Gaussian (1.109) is given by (1.110).
1.36
(⋆)
A strictly convex function is deﬁned as one for which every chord lies above
the function. Show that this is equivalent to the condition that the second derivative
of the function be positive.
1.37
(⋆) Using the deﬁnition (1.111) together with the product rule of probability, prove
the result (1.112).
1.38
(⋆⋆) www
Using proof by induction, show that the inequality (1.114) for convex
functions implies the result (1.115).
1.39
(⋆⋆⋆) Consider two binary variables x and y having the joint distribution given in
Table 1.3.
Evaluate the following quantities
(a) H[x]
(c) H[y|x]
(e) H[x, y]
(b) H[y]
(d) H[x|y]
(f) I[x, y].
Draw a diagram to show the relationship between these various quantities.


---
**Page 84**
66
1. INTRODUCTION
1.40
(⋆)
By applying Jensen’s inequality (1.115) with f(x) = ln x, show that the arith-
metic mean of a set of real numbers is never less than their geometrical mean.
1.41
(⋆) www
Using the sum and product rules of probability, show that the mutual
information I(x, y) satisﬁes the relation (1.121).


---
**Page 85**
2
Probability
Distributions
In Chapter 1, we emphasized the central role played by probability theory in the
solution of pattern recognition problems. We turn now to an exploration of some
particular examples of probability distributions and their properties. As well as be-
ing of great interest in their own right, these distributions can form building blocks
for more complex models and will be used extensively throughout the book. The
distributions introduced in this chapter will also serve another important purpose,
namely to provide us with the opportunity to discuss some key statistical concepts,
such as Bayesian inference, in the context of simple models before we encounter
them in more complex situations in later chapters.
One role for the distributions discussed in this chapter is to model the prob-
ability distribution p(x) of a random variable x, given a ﬁnite set x1, . . . , xN of
observations. This problem is known as density estimation. For the purposes of
this chapter, we shall assume that the data points are independent and identically
distributed. It should be emphasized that the problem of density estimation is fun-
67


---
**Page 86**
68
2. PROBABILITY DISTRIBUTIONS
damentally ill-posed, because there are inﬁnitely many probability distributions that
could have given rise to the observed ﬁnite data set. Indeed, any distribution p(x)
that is nonzero at each of the data points x1, . . . , xN is a potential candidate. The
issue of choosing an appropriate distribution relates to the problem of model selec-
tion that has already been encountered in the context of polynomial curve ﬁtting in
Chapter 1 and that is a central issue in pattern recognition.
We begin by considering the binomial and multinomial distributions for discrete
random variables and the Gaussian distribution for continuous random variables.
These are speciﬁc examples of parametric distributions, so-called because they are
governed by a small number of adaptive parameters, such as the mean and variance in
the case of a Gaussian for example. To apply such models to the problem of density
estimation, we need a procedure for determining suitable values for the parameters,
given an observed data set. In a frequentist treatment, we choose speciﬁc values
for the parameters by optimizing some criterion, such as the likelihood function. By
contrast, in a Bayesian treatment we introduce prior distributions over the parameters
and then use Bayes’ theorem to compute the corresponding posterior distribution
given the observed data.
We shall see that an important role is played by conjugate priors, that lead to
posterior distributions having the same functional form as the prior, and that there-
fore lead to a greatly simpliﬁed Bayesian analysis. For example, the conjugate prior
for the parameters of the multinomial distribution is called the Dirichlet distribution,
while the conjugate prior for the mean of a Gaussian is another Gaussian. All of these
distributions are examples of the exponential family of distributions, which possess
a number of important properties, and which will be discussed in some detail.
One limitation of the parametric approach is that it assumes a speciﬁc functional
form for the distribution, which may turn out to be inappropriate for a particular
application. An alternative approach is given by nonparametric density estimation
methods in which the form of the distribution typically depends on the size of the data
set. Such models still contain parameters, but these control the model complexity
rather than the form of the distribution. We end this chapter by considering three
nonparametric methods based respectively on histograms, nearest-neighbours, and
kernels.
2.1. Binary Variables
We begin by considering a single binary random variable x ∈{0, 1}. For example,
x might describe the outcome of ﬂipping a coin, with x = 1 representing ‘heads’,
and x = 0 representing ‘tails’. We can imagine that this is a damaged coin so that
the probability of landing heads is not necessarily the same as that of landing tails.
The probability of x = 1 will be denoted by the parameter µ so that
p(x = 1|µ) = µ
(2.1)


---
**Page 87**
2.1. Binary Variables
69
where 0 ⩽µ ⩽1, from which it follows that p(x = 0|µ) = 1 −µ. The probability
distribution over x can therefore be written in the form
Bern(x|µ) = µx(1 −µ)1−x
(2.2)
which is known as the Bernoulli distribution. It is easily veriﬁed that this distribution
Exercise 2.1
is normalized and that it has mean and variance given by
E[x]
=
µ
(2.3)
var[x]
=
µ(1 −µ).
(2.4)
Now suppose we have a data set D = {x1, . . . , xN} of observed values of x.
We can construct the likelihood function, which is a function of µ, on the assumption
that the observations are drawn independently from p(x|µ), so that
p(D|µ) =
N

n=1
p(xn|µ) =
N

n=1
µxn(1 −µ)1−xn.
(2.5)
In a frequentist setting, we can estimate a value for µ by maximizing the likelihood
function, or equivalently by maximizing the logarithm of the likelihood. In the case
of the Bernoulli distribution, the log likelihood function is given by
ln p(D|µ) =
N

n=1
ln p(xn|µ) =
N

n=1
{xn ln µ + (1 −xn) ln(1 −µ)} .
(2.6)
At this point, it is worth noting that the log likelihood function depends on the N
observations xn only through their sum 
n xn. This sum provides an example of a
sufﬁcient statistic for the data under this distribution, and we shall study the impor-
tant role of sufﬁcient statistics in some detail. If we set the derivative of ln p(D|µ)
Section 2.4
with respect to µ equal to zero, we obtain the maximum likelihood estimator
µML = 1
N
N

n=1
xn
(2.7)
Jacob Bernoulli
1654–1705
Jacob Bernoulli,
also known as
Jacques or James Bernoulli, was a
Swiss mathematician and was the
ﬁrst of many in the Bernoulli family
to pursue a career in science and
mathematics.
Although compelled
to study philosophy and theology against his will by
his parents, he travelled extensively after graduating
in order to meet with many of the leading scientists of
his time, including Boyle and Hooke in England. When
he returned to Switzerland, he taught mechanics and
became Professor of Mathematics at Basel in 1687.
Unfortunately, rivalry between Jacob and his younger
brother Johann turned an initially productive collabora-
tion into a bitter and public dispute. Jacob’s most sig-
niﬁcant contributions to mathematics appeared in The
Art of Conjecture published in 1713, eight years after
his death, which deals with topics in probability the-
ory including what has become known as the Bernoulli
distribution.


---
**Page 88**
70
2. PROBABILITY DISTRIBUTIONS
Figure 2.1
Histogram plot of the binomial dis-
tribution (2.9) as a function of m for
N = 10 and µ = 0.25.
m
0
1
2
3
4
5
6
7
8
9
10
0
0.1
0.2
0.3
which is also known as the sample mean. If we denote the number of observations
of x = 1 (heads) within this data set by m, then we can write (2.7) in the form
µML = m
N
(2.8)
so that the probability of landing heads is given, in this maximum likelihood frame-
work, by the fraction of observations of heads in the data set.
Now suppose we ﬂip a coin, say, 3 times and happen to observe 3 heads. Then
N = m = 3 and µML = 1. In this case, the maximum likelihood result would
predict that all future observations should give heads. Common sense tells us that
this is unreasonable, and in fact this is an extreme example of the over-ﬁtting associ-
ated with maximum likelihood. We shall see shortly how to arrive at more sensible
conclusions through the introduction of a prior distribution over µ.
We can also work out the distribution of the number m of observations of x = 1,
given that the data set has size N. This is called the binomial distribution, and
from (2.5) we see that it is proportional to µm(1 −µ)N−m. In order to obtain the
normalization coefﬁcient we note that out of N coin ﬂips, we have to add up all
of the possible ways of obtaining m heads, so that the binomial distribution can be
written
Bin(m|N, µ) =
N
m

µm(1 −µ)N−m
(2.9)
where
N
m

≡
N!
(N −m)!m!
(2.10)
is the number of ways of choosing m objects out of a total of N identical objects.
Exercise 2.3
Figure 2.1 shows a plot of the binomial distribution for N = 10 and µ = 0.25.
The mean and variance of the binomial distribution can be found by using the
result of Exercise 1.10, which shows that for independent events the mean of the
sum is the sum of the means, and the variance of the sum is the sum of the variances.
Because m = x1 + . . . + xN, and for each observation the mean and variance are


---
**Page 89**
2.1. Binary Variables
71
given by (2.3) and (2.4), respectively, we have
E[m] ≡
N

m=0
mBin(m|N, µ)
=
Nµ
(2.11)
var[m] ≡
N

m=0
(m −E[m])2 Bin(m|N, µ)
=
Nµ(1 −µ).
(2.12)
These results can also be proved directly using calculus.
Exercise 2.4
2.1.1
The beta distribution
We have seen in (2.8) that the maximum likelihood setting for the parameter µ
in the Bernoulli distribution, and hence in the binomial distribution, is given by the
fraction of the observations in the data set having x = 1. As we have already noted,
this can give severely over-ﬁtted results for small data sets. In order to develop a
Bayesian treatment for this problem, we need to introduce a prior distribution p(µ)
over the parameter µ. Here we consider a form of prior distribution that has a simple
interpretation as well as some useful analytical properties. To motivate this prior,
we note that the likelihood function takes the form of the product of factors of the
form µx(1 −µ)1−x. If we choose a prior to be proportional to powers of µ and
(1 −µ), then the posterior distribution, which is proportional to the product of the
prior and the likelihood function, will have the same functional form as the prior.
This property is called conjugacy and we will see several examples of it later in this
chapter. We therefore choose a prior, called the beta distribution, given by
Beta(µ|a, b) = Γ(a + b)
Γ(a)Γ(b)µa−1(1 −µ)b−1
(2.13)
where Γ(x) is the gamma function deﬁned by (1.141), and the coefﬁcient in (2.13)
ensures that the beta distribution is normalized, so that
Exercise 2.5
 1
0
Beta(µ|a, b) dµ = 1.
(2.14)
The mean and variance of the beta distribution are given by
Exercise 2.6
E[µ]
=
a
a + b
(2.15)
var[µ]
=
ab
(a + b)2(a + b + 1).
(2.16)
The parameters a and b are often called hyperparameters because they control the
distribution of the parameter µ. Figure 2.2 shows plots of the beta distribution for
various values of the hyperparameters.
The posterior distribution of µ is now obtained by multiplying the beta prior
(2.13) by the binomial likelihood function (2.9) and normalizing. Keeping only the
factors that depend on µ, we see that this posterior distribution has the form
p(µ|m, l, a, b) ∝µm+a−1(1 −µ)l+b−1
(2.17)


---
**Page 90**
72
2. PROBABILITY DISTRIBUTIONS
µ
a = 0.1
b = 0.1
0
0.5
1
0
1
2
3
µ
a = 1
b = 1
0
0.5
1
0
1
2
3
µ
a = 2
b = 3
0
0.5
1
0
1
2
3
µ
a = 8
b = 4
0
0.5
1
0
1
2
3
Figure 2.2
Plots of the beta distribution Beta(µ|a, b) given by (2.13) as a function of µ for various values of the
hyperparameters a and b.
where l = N −m, and therefore corresponds to the number of ‘tails’ in the coin
example. We see that (2.17) has the same functional dependence on µ as the prior
distribution, reﬂecting the conjugacy properties of the prior with respect to the like-
lihood function. Indeed, it is simply another beta distribution, and its normalization
coefﬁcient can therefore be obtained by comparison with (2.13) to give
p(µ|m, l, a, b) = Γ(m + a + l + b)
Γ(m + a)Γ(l + b)µm+a−1(1 −µ)l+b−1.
(2.18)
We see that the effect of observing a data set of m observations of x = 1 and
l observations of x = 0 has been to increase the value of a by m, and the value of
b by l, in going from the prior distribution to the posterior distribution. This allows
us to provide a simple interpretation of the hyperparameters a and b in the prior as
an effective number of observations of x = 1 and x = 0, respectively. Note that
a and b need not be integers. Furthermore, the posterior distribution can act as the
prior if we subsequently observe additional data. To see this, we can imagine taking
observations one at a time and after each observation updating the current posterior


---
**Page 91**
2.1. Binary Variables
73
µ
prior
0
0.5
1
0
1
2
µ
likelihood function
0
0.5
1
0
1
2
µ
posterior
0
0.5
1
0
1
2
Figure 2.3
Illustration of one step of sequential Bayesian inference. The prior is given by a beta distribution
with parameters a = 2, b = 2, and the likelihood function, given by (2.9) with N = m = 1, corresponds to a
single observation of x = 1, so that the posterior is given by a beta distribution with parameters a = 3, b = 2.
distribution by multiplying by the likelihood function for the new observation and
then normalizing to obtain the new, revised posterior distribution. At each stage, the
posterior is a beta distribution with some total number of (prior and actual) observed
values for x = 1 and x = 0 given by the parameters a and b. Incorporation of an
additional observation of x = 1 simply corresponds to incrementing the value of a
by 1, whereas for an observation of x = 0 we increment b by 1. Figure 2.3 illustrates
one step in this process.
We see that this sequential approach to learning arises naturally when we adopt
a Bayesian viewpoint. It is independent of the choice of prior and of the likelihood
function and depends only on the assumption of i.i.d. data. Sequential methods make
use of observations one at a time, or in small batches, and then discard them before
the next observations are used. They can be used, for example, in real-time learning
scenarios where a steady stream of data is arriving, and predictions must be made
before all of the data is seen. Because they do not require the whole data set to be
stored or loaded into memory, sequential methods are also useful for large data sets.
Maximum likelihood methods can also be cast into a sequential framework.
Section 2.3.5
If our goal is to predict, as best we can, the outcome of the next trial, then we
must evaluate the predictive distribution of x, given the observed data set D. From
the sum and product rules of probability, this takes the form
p(x = 1|D) =
 1
0
p(x = 1|µ)p(µ|D) dµ =
 1
0
µp(µ|D) dµ = E[µ|D].
(2.19)
Using the result (2.18) for the posterior distribution p(µ|D), together with the result
(2.15) for the mean of the beta distribution, we obtain
p(x = 1|D) =
m + a
m + a + l + b
(2.20)
which has a simple interpretation as the total fraction of observations (both real ob-
servations and ﬁctitious prior observations) that correspond to x = 1. Note that in
the limit of an inﬁnitely large data set m, l →∞the result (2.20) reduces to the
maximum likelihood result (2.8). As we shall see, it is a very general property that
the Bayesian and maximum likelihood results will agree in the limit of an inﬁnitely


---
**Page 92**
74
2. PROBABILITY DISTRIBUTIONS
large data set. For a ﬁnite data set, the posterior mean for µ always lies between the
prior mean and the maximum likelihood estimate for µ corresponding to the relative
frequencies of events given by (2.7).
Exercise 2.7
From Figure 2.2, we see that as the number of observations increases, so the
posterior distribution becomes more sharply peaked. This can also be seen from
the result (2.16) for the variance of the beta distribution, in which we see that the
variance goes to zero for a →∞or b →∞. In fact, we might wonder whether it is
a general property of Bayesian learning that, as we observe more and more data, the
uncertainty represented by the posterior distribution will steadily decrease.
To address this, we can take a frequentist view of Bayesian learning and show
that, on average, such a property does indeed hold. Consider a general Bayesian
inference problem for a parameter θ for which we have observed a data set D, de-
scribed by the joint distribution p(θ, D). The following result
Exercise 2.8
Eθ[θ] = ED [Eθ[θ|D]]
(2.21)
where
Eθ[θ]
≡

p(θ)θ dθ
(2.22)
ED[Eθ[θ|D]]
≡
 
θp(θ|D) dθ

p(D) dD
(2.23)
says that the posterior mean of θ, averaged over the distribution generating the data,
is equal to the prior mean of θ. Similarly, we can show that
varθ[θ] = ED [varθ[θ|D]] + varD [Eθ[θ|D]] .
(2.24)
The term on the left-hand side of (2.24) is the prior variance of θ. On the right-
hand side, the ﬁrst term is the average posterior variance of θ, and the second term
measures the variance in the posterior mean of θ. Because this variance is a positive
quantity, this result shows that, on average, the posterior variance of θ is smaller than
the prior variance. The reduction in variance is greater if the variance in the posterior
mean is greater. Note, however, that this result only holds on average, and that for a
particular observed data set it is possible for the posterior variance to be larger than
the prior variance.
2.2. Multinomial Variables
Binary variables can be used to describe quantities that can take one of two possible
values. Often, however, we encounter discrete variables that can take on one of K
possible mutually exclusive states. Although there are various alternative ways to
express such variables, we shall see shortly that a particularly convenient represen-
tation is the 1-of-K scheme in which the variable is represented by a K-dimensional
vector x in which one of the elements xk equals 1, and all remaining elements equal


---
**Page 93**
2.2. Multinomial Variables
75
0. So, for instance if we have a variable that can take K = 6 states and a particular
observation of the variable happens to correspond to the state where x3 = 1, then x
will be represented by
x = (0, 0, 1, 0, 0, 0)T.
(2.25)
Note that such vectors satisfy K
k=1 xk = 1. If we denote the probability of xk = 1
by the parameter µk, then the distribution of x is given
p(x|µ) =
K

k=1
µxk
k
(2.26)
where µ = (µ1, . . . , µK)T, and the parameters µk are constrained to satisfy µk ⩾0
and 
k µk = 1, because they represent probabilities. The distribution (2.26) can be
regarded as a generalization of the Bernoulli distribution to more than two outcomes.
It is easily seen that the distribution is normalized

x
p(x|µ) =
K

k=1
µk = 1
(2.27)
and that
E[x|µ] =

x
p(x|µ)x = (µ1, . . . , µM)T = µ.
(2.28)
Now consider a data set D of N independent observations x1, . . . , xN. The
corresponding likelihood function takes the form
p(D|µ) =
N

n=1
K

k=1
µxnk
k
=
K

k=1
µ(
P
n xnk)
k
=
K

k=1
µmk
k .
(2.29)
We see that the likelihood function depends on the N data points only through the
K quantities
mk =

n
xnk
(2.30)
which represent the number of observations of xk = 1. These are called the sufﬁcient
statistics for this distribution.
Section 2.4
In order to ﬁnd the maximum likelihood solution for µ, we need to maximize
ln p(D|µ) with respect to µk taking account of the constraint that the µk must sum
to one. This can be achieved using a Lagrange multiplier λ and maximizing
Appendix E
K

k=1
mk ln µk + λ
 K

k=1
µk −1

.
(2.31)
Setting the derivative of (2.31) with respect to µk to zero, we obtain
µk = −mk/λ.
(2.32)


---
**Page 94**
76
2. PROBABILITY DISTRIBUTIONS
We can solve for the Lagrange multiplier λ by substituting (2.32) into the constraint

k µk = 1 to give λ = −N. Thus we obtain the maximum likelihood solution in
the form
µML
k
= mk
N
(2.33)
which is the fraction of the N observations for which xk = 1.
We can consider the joint distribution of the quantities m1, . . . , mK, conditioned
on the parameters µ and on the total number N of observations. From (2.29) this
takes the form
Mult(m1, m2, . . . , mK|µ, N) =

N
m1m2 . . . mK
 K

k=1
µmk
k
(2.34)
which is known as the multinomial distribution. The normalization coefﬁcient is the
number of ways of partitioning N objects into K groups of size m1, . . . , mK and is
given by

N
m1m2 . . . mK

=
N!
m1!m2! . . . mK!.
(2.35)
Note that the variables mk are subject to the constraint
K

k=1
mk = N.
(2.36)
2.2.1
The Dirichlet distribution
We now introduce a family of prior distributions for the parameters {µk} of
the multinomial distribution (2.34). By inspection of the form of the multinomial
distribution, we see that the conjugate prior is given by
p(µ|α) ∝
K

k=1
µαk−1
k
(2.37)
where 0 ⩽µk ⩽1 and 
k µk = 1. Here α1, . . . , αK are the parameters of the
distribution, and α denotes (α1, . . . , αK)T. Note that, because of the summation
constraint, the distribution over the space of the {µk} is conﬁned to a simplex of
dimensionality K −1, as illustrated for K = 3 in Figure 2.4.
The normalized form for this distribution is by
Exercise 2.9
Dir(µ|α) =
Γ(α0)
Γ(α1) · · · Γ(αK)
K

k=1
µαk−1
k
(2.38)
which is called the Dirichlet distribution. Here Γ(x) is the gamma function deﬁned
by (1.141) while
α0 =
K

k=1
αk.
(2.39)


---
**Page 95**
2.2. Multinomial Variables
77
Figure 2.4
The Dirichlet distribution over three variables µ1, µ2, µ3
is conﬁned to a simplex (a bounded linear manifold) of
the form shown, as a consequence of the constraints
0 ⩽µk ⩽1 and P
k µk = 1.
µ1
µ2
µ3
Plots of the Dirichlet distribution over the simplex, for various settings of the param-
eters αk, are shown in Figure 2.5.
Multiplying the prior (2.38) by the likelihood function (2.34), we obtain the
posterior distribution for the parameters {µk} in the form
p(µ|D, α) ∝p(D|µ)p(µ|α) ∝
K

k=1
µαk+mk−1
k
.
(2.40)
We see that the posterior distribution again takes the form of a Dirichlet distribution,
conﬁrming that the Dirichlet is indeed a conjugate prior for the multinomial. This
allows us to determine the normalization coefﬁcient by comparison with (2.38) so
that
p(µ|D, α)
=
Dir(µ|α + m)
=
Γ(α0 + N)
Γ(α1 + m1) · · · Γ(αK + mK)
K

k=1
µαk+mk−1
k
(2.41)
where we have denoted m = (m1, . . . , mK)T. As for the case of the binomial
distribution with its beta prior, we can interpret the parameters αk of the Dirichlet
prior as an effective number of observations of xk = 1.
Note that two-state quantities can either be represented as binary variables and
Lejeune Dirichlet
1805–1859
Johann
Peter
Gustav
Lejeune
Dirichlet was a modest and re-
served mathematician who made
contributions in number theory, me-
chanics, and astronomy, and who
gave the ﬁrst rigorous analysis of
Fourier series.
His family originated from Richelet
in Belgium, and the name Lejeune Dirichlet comes
from ‘le jeune de Richelet’ (the young person from
Richelet). Dirichlet’s ﬁrst paper, which was published
in 1825, brought him instant fame. It concerned Fer-
mat’s last theorem, which claims that there are no
positive integer solutions to xn + yn = zn for n > 2.
Dirichlet gave a partial proof for the case n = 5, which
was sent to Legendre for review and who in turn com-
pleted the proof. Later, Dirichlet gave a complete proof
for n = 14, although a full proof of Fermat’s last theo-
rem for arbitrary n had to wait until the work of Andrew
Wiles in the closing years of the 20th century.


---
**Page 96**
78
2. PROBABILITY DISTRIBUTIONS
Figure 2.5
Plots of the Dirichlet distribution over three variables, where the two horizontal axes are coordinates
in the plane of the simplex and the vertical axis corresponds to the value of the density. Here {αk} = 0.1 on the
left plot, {αk} = 1 in the centre plot, and {αk} = 10 in the right plot.
modelled using the binomial distribution (2.9) or as 1-of-2 variables and modelled
using the multinomial distribution (2.34) with K = 2.
2.3. The Gaussian Distribution
The Gaussian, also known as the normal distribution, is a widely used model for the
distribution of continuous variables. In the case of a single variable x, the Gaussian
distribution can be written in the form
N (x|µ, σ2) =
1
(2πσ2)1/2 exp

−1
2σ2(x −µ)2

(2.42)
where µ is the mean and σ2 is the variance. For a D-dimensional vector x, the
multivariate Gaussian distribution takes the form
N (x|µ, Σ) =
1
(2π)D/2
1
|Σ|1/2 exp

−1
2(x −µ)TΣ−1(x −µ)

(2.43)
where µ is a D-dimensional mean vector, Σ is a D × D covariance matrix, and |Σ|
denotes the determinant of Σ.
The Gaussian distribution arises in many different contexts and can be motivated
from a variety of different perspectives. For example, we have already seen that for
Section 1.6
a single real variable, the distribution that maximizes the entropy is the Gaussian.
This property applies also to the multivariate Gaussian.
Exercise 2.14
Another situation in which the Gaussian distribution arises is when we consider
the sum of multiple random variables. The central limit theorem (due to Laplace)
tells us that, subject to certain mild conditions, the sum of a set of random variables,
which is of course itself a random variable, has a distribution that becomes increas-
ingly Gaussian as the number of terms in the sum increases (Walker, 1969). We can


---
**Page 97**
2.3. The Gaussian Distribution
79
N = 1
0
0.5
1
0
1
2
3
N = 2
0
0.5
1
0
1
2
3
N = 10
0
0.5
1
0
1
2
3
Figure 2.6
Histogram plots of the mean of N uniformly distributed numbers for various values of N.
We
observe that as N increases, the distribution tends towards a Gaussian.
illustrate this by considering N variables x1, . . . , xN each of which has a uniform
distribution over the interval [0, 1] and then considering the distribution of the mean
(x1 + · · · + xN)/N. For large N, this distribution tends to a Gaussian, as illustrated
in Figure 2.6.
In practice, the convergence to a Gaussian as N increases can be
very rapid. One consequence of this result is that the binomial distribution (2.9),
which is a distribution over m deﬁned by the sum of N observations of the random
binary variable x, will tend to a Gaussian as N →∞(see Figure 2.1 for the case of
N = 10).
The Gaussian distribution has many important analytical properties, and we shall
consider several of these in detail. As a result, this section will be rather more tech-
nically involved than some of the earlier sections, and will require familiarity with
various matrix identities. However, we strongly encourage the reader to become pro-
Appendix C
ﬁcient in manipulating Gaussian distributions using the techniques presented here as
this will prove invaluable in understanding the more complex models presented in
later chapters.
We begin by considering the geometrical form of the Gaussian distribution. The
Carl Friedrich Gauss
1777–1855
It is said that when Gauss went
to elementary school at age 7, his
teacher B¨uttner, trying to keep the
class occupied, asked the pupils to
sum the integers from 1 to 100. To
the teacher’s amazement, Gauss
arrived at the answer in a matter of moments by noting
that the sum can be represented as 50 pairs (1 + 100,
2+99, etc.) each of which added to 101, giving the an-
swer 5,050. It is now believed that the problem which
was actually set was of the same form but somewhat
harder in that the sequence had a larger starting value
and a larger increment. Gauss was a German math-
ematician and scientist with a reputation for being a
hard-working perfectionist. One of his many contribu-
tions was to show that least squares can be derived
under the assumption of normally distributed errors.
He also created an early formulation of non-Euclidean
geometry (a self-consistent geometrical theory that vi-
olates the axioms of Euclid) but was reluctant to dis-
cuss it openly for fear that his reputation might suffer
if it were seen that he believed in such a geometry.
At one point, Gauss was asked to conduct a geodetic
survey of the state of Hanover, which led to his for-
mulation of the normal distribution, now also known
as the Gaussian. After his death, a study of his di-
aries revealed that he had discovered several impor-
tant mathematical results years or even decades be-
fore they were published by others.


---
**Page 98**
80
2. PROBABILITY DISTRIBUTIONS
functional dependence of the Gaussian on x is through the quadratic form
∆2 = (x −µ)TΣ−1(x −µ)
(2.44)
which appears in the exponent. The quantity ∆is called the Mahalanobis distance
from µ to x and reduces to the Euclidean distance when Σ is the identity matrix. The
Gaussian distribution will be constant on surfaces in x-space for which this quadratic
form is constant.
First of all, we note that the matrix Σ can be taken to be symmetric, without
loss of generality, because any antisymmetric component would disappear from the
exponent. Now consider the eigenvector equation for the covariance matrix
Exercise 2.17
Σui = λiui
(2.45)
where i = 1, . . . , D. Because Σ is a real, symmetric matrix its eigenvalues will be
real, and its eigenvectors can be chosen to form an orthonormal set, so that
Exercise 2.18
uT
i uj = Iij
(2.46)
where Iij is the i, j element of the identity matrix and satisﬁes
Iij =

1,
if i = j
0,
otherwise.
(2.47)
The covariance matrix Σ can be expressed as an expansion in terms of its eigenvec-
tors in the form
Exercise 2.19
Σ =
D

i=1
λiuiuT
i
(2.48)
and similarly the inverse covariance matrix Σ−1 can be expressed as
Σ−1 =
D

i=1
1
λi
uiuT
i .
(2.49)
Substituting (2.49) into (2.44), the quadratic form becomes
∆2 =
D

i=1
y2
i
λi
(2.50)
where we have deﬁned
yi = uT
i (x −µ).
(2.51)
We can interpret {yi} as a new coordinate system deﬁned by the orthonormal vectors
ui that are shifted and rotated with respect to the original xi coordinates. Forming
the vector y = (y1, . . . , yD)T, we have
y = U(x −µ)
(2.52)


---
**Page 99**
2.3. The Gaussian Distribution
81
Figure 2.7
The red curve shows the ellip-
tical surface of constant proba-
bility density for a Gaussian in
a two-dimensional space x =
(x1, x2) on which the density
is exp(−1/2) of its value at
x = µ.
The major axes of
the ellipse are deﬁned by the
eigenvectors ui of the covari-
ance matrix, with correspond-
ing eigenvalues λi.
x1
x2
λ1/2
1
λ1/2
2
y1
y2
u1
u2
µ
where U is a matrix whose rows are given by uT
i . From (2.46) it follows that U is
an orthogonal matrix, i.e., it satisﬁes UUT = I, and hence also UTU = I, where I
Appendix C
is the identity matrix.
The quadratic form, and hence the Gaussian density, will be constant on surfaces
for which (2.51) is constant. If all of the eigenvalues λi are positive, then these
surfaces represent ellipsoids, with their centres at µ and their axes oriented along ui,
and with scaling factors in the directions of the axes given by λ1/2
i
, as illustrated in
Figure 2.7.
For the Gaussian distribution to be well deﬁned, it is necessary for all of the
eigenvalues λi of the covariance matrix to be strictly positive, otherwise the dis-
tribution cannot be properly normalized. A matrix whose eigenvalues are strictly
positive is said to be positive deﬁnite. In Chapter 12, we will encounter Gaussian
distributions for which one or more of the eigenvalues are zero, in which case the
distribution is singular and is conﬁned to a subspace of lower dimensionality. If all
of the eigenvalues are nonnegative, then the covariance matrix is said to be positive
semideﬁnite.
Now consider the form of the Gaussian distribution in the new coordinate system
deﬁned by the yi. In going from the x to the y coordinate system, we have a Jacobian
matrix J with elements given by
Jij = ∂xi
∂yj
= Uji
(2.53)
where Uji are the elements of the matrix UT. Using the orthonormality property of
the matrix U, we see that the square of the determinant of the Jacobian matrix is
|J|2 =
UT2 =
UT |U| =
UTU
 = |I| = 1
(2.54)
and hence |J| = 1. Also, the determinant |Σ| of the covariance matrix can be written


---
**Page 100**
82
2. PROBABILITY DISTRIBUTIONS
as the product of its eigenvalues, and hence
|Σ|1/2 =
D

j=1
λ1/2
j
.
(2.55)
Thus in the yj coordinate system, the Gaussian distribution takes the form
p(y) = p(x)|J| =
D

j=1
1
(2πλj)1/2 exp

−y2
j
2λj

(2.56)
which is the product of D independent univariate Gaussian distributions. The eigen-
vectors therefore deﬁne a new set of shifted and rotated coordinates with respect
to which the joint probability distribution factorizes into a product of independent
distributions. The integral of the distribution in the y coordinate system is then

p(y) dy =
D

j=1
 ∞
−∞
1
(2πλj)1/2 exp

−y2
j
2λj

dyj = 1
(2.57)
where we have used the result (1.48) for the normalization of the univariate Gaussian.
This conﬁrms that the multivariate Gaussian (2.43) is indeed normalized.
We now look at the moments of the Gaussian distribution and thereby provide an
interpretation of the parameters µ and Σ. The expectation of x under the Gaussian
distribution is given by
E[x]
=
1
(2π)D/2
1
|Σ|1/2

exp

−1
2(x −µ)TΣ−1(x −µ)

x dx
=
1
(2π)D/2
1
|Σ|1/2

exp

−1
2zTΣ−1z

(z + µ) dz
(2.58)
where we have changed variables using z = x −µ. We now note that the exponent
is an even function of the components of z and, because the integrals over these are
taken over the range (−∞, ∞), the term in z in the factor (z + µ) will vanish by
symmetry. Thus
E[x] = µ
(2.59)
and so we refer to µ as the mean of the Gaussian distribution.
We now consider second order moments of the Gaussian. In the univariate case,
we considered the second order moment given by E[x2]. For the multivariate Gaus-
sian, there are D2 second order moments given by E[xixj], which we can group
together to form the matrix E[xxT]. This matrix can be written as
E[xxT] =
1
(2π)D/2
1
|Σ|1/2

exp

−1
2(x −µ)TΣ−1(x −µ)

xxT dx
=
1
(2π)D/2
1
|Σ|1/2

exp

−1
2zTΣ−1z

(z + µ)(z + µ)T dz


---
**Page 101**
2.3. The Gaussian Distribution
83
where again we have changed variables using z = x −µ. Note that the cross-terms
involving µzT and µTz will again vanish by symmetry. The term µµT is constant
and can be taken outside the integral, which itself is unity because the Gaussian
distribution is normalized. Consider the term involving zzT. Again, we can make
use of the eigenvector expansion of the covariance matrix given by (2.45), together
with the completeness of the set of eigenvectors, to write
z =
D

j=1
yjuj
(2.60)
where yj = uT
j z, which gives
1
(2π)D/2
1
|Σ|1/2

exp

−1
2zTΣ−1z

zzT dz
=
1
(2π)D/2
1
|Σ|1/2
D

i=1
D

j=1
uiuT
j

exp

−
D

k=1
y2
k
2λk

yiyj dy
=
D

i=1
uiuT
i λi = Σ
(2.61)
where we have made use of the eigenvector equation (2.45), together with the fact
that the integral on the right-hand side of the middle line vanishes by symmetry
unless i = j, and in the ﬁnal line we have made use of the results (1.50) and (2.55),
together with (2.48). Thus we have
E[xxT] = µµT + Σ.
(2.62)
For single random variables, we subtracted the mean before taking second mo-
ments in order to deﬁne a variance. Similarly, in the multivariate case it is again
convenient to subtract off the mean, giving rise to the covariance of a random vector
x deﬁned by
cov[x] = E 
(x −E[x])(x −E[x])T	
.
(2.63)
For the speciﬁc case of a Gaussian distribution, we can make use of E[x] = µ,
together with the result (2.62), to give
cov[x] = Σ.
(2.64)
Because the parameter matrix Σ governs the covariance of x under the Gaussian
distribution, it is called the covariance matrix.
Although the Gaussian distribution (2.43) is widely used as a density model, it
suffers from some signiﬁcant limitations. Consider the number of free parameters in
the distribution. A general symmetric covariance matrix Σ will have D(D + 1)/2
independent parameters, and there are another D independent parameters in µ, giv-
Exercise 2.21
ing D(D + 3)/2 parameters in total. For large D, the total number of parameters


---
**Page 102**
84
2. PROBABILITY DISTRIBUTIONS
Figure 2.8
Contours of constant
probability density for a Gaussian
distribution in two dimensions in
which the covariance matrix is (a) of
general form, (b) diagonal, in which
the elliptical contours are aligned
with the coordinate axes, and (c)
proportional to the identity matrix, in
which the contours are concentric
circles.
x1
x2
(a)
x1
x2
(b)
x1
x2
(c)
therefore grows quadratically with D, and the computational task of manipulating
and inverting large matrices can become prohibitive. One way to address this prob-
lem is to use restricted forms of the covariance matrix. If we consider covariance
matrices that are diagonal, so that Σ = diag(σ2
i ), we then have a total of 2D inde-
pendent parameters in the density model. The corresponding contours of constant
density are given by axis-aligned ellipsoids. We could further restrict the covariance
matrix to be proportional to the identity matrix, Σ = σ2I, known as an isotropic co-
variance, giving D + 1 independent parameters in the model and spherical surfaces
of constant density. The three possibilities of general, diagonal, and isotropic covari-
ance matrices are illustrated in Figure 2.8. Unfortunately, whereas such approaches
limit the number of degrees of freedom in the distribution and make inversion of the
covariance matrix a much faster operation, they also greatly restrict the form of the
probability density and limit its ability to capture interesting correlations in the data.
A further limitation of the Gaussian distribution is that it is intrinsically uni-
modal (i.e., has a single maximum) and so is unable to provide a good approximation
to multimodal distributions. Thus the Gaussian distribution can be both too ﬂexible,
in the sense of having too many parameters, while also being too limited in the range
of distributions that it can adequately represent. We will see later that the introduc-
tion of latent variables, also called hidden variables or unobserved variables, allows
both of these problems to be addressed. In particular, a rich family of multimodal
distributions is obtained by introducing discrete latent variables leading to mixtures
of Gaussians, as discussed in Section 2.3.9. Similarly, the introduction of continuous
latent variables, as described in Chapter 12, leads to models in which the number of
free parameters can be controlled independently of the dimensionality D of the data
space while still allowing the model to capture the dominant correlations in the data
set. Indeed, these two approaches can be combined and further extended to derive
a very rich set of hierarchical models that can be adapted to a broad range of prac-
tical applications. For instance, the Gaussian version of the Markov random ﬁeld,
Section 8.3
which is widely used as a probabilistic model of images, is a Gaussian distribution
over the joint space of pixel intensities but rendered tractable through the imposition
of considerable structure reﬂecting the spatial organization of the pixels. Similarly,
the linear dynamical system, used to model time series data for applications such
Section 13.3
as tracking, is also a joint Gaussian distribution over a potentially large number of
observed and latent variables and again is tractable due to the structure imposed on
the distribution. A powerful framework for expressing the form and properties of


---
**Page 103**
2.3. The Gaussian Distribution
85
such complex distributions is that of probabilistic graphical models, which will form
the subject of Chapter 8.
2.3.1
Conditional Gaussian distributions
An important property of the multivariate Gaussian distribution is that if two
sets of variables are jointly Gaussian, then the conditional distribution of one set
conditioned on the other is again Gaussian. Similarly, the marginal distribution of
either set is also Gaussian.
Consider ﬁrst the case of conditional distributions. Suppose x is a D-dimensional
vector with Gaussian distribution N(x|µ, Σ) and that we partition x into two dis-
joint subsets xa and xb. Without loss of generality, we can take xa to form the ﬁrst
M components of x, with xb comprising the remaining D −M components, so that
x =

xa
xb

.
(2.65)
We also deﬁne corresponding partitions of the mean vector µ given by
µ =

µa
µb

(2.66)
and of the covariance matrix Σ given by
Σ =

Σaa
Σab
Σba
Σbb

.
(2.67)
Note that the symmetry ΣT = Σ of the covariance matrix implies that Σaa and Σbb
are symmetric, while Σba = ΣT
ab.
In many situations, it will be convenient to work with the inverse of the covari-
ance matrix
Λ ≡Σ−1
(2.68)
which is known as the precision matrix. In fact, we shall see that some properties
of Gaussian distributions are most naturally expressed in terms of the covariance,
whereas others take a simpler form when viewed in terms of the precision. We
therefore also introduce the partitioned form of the precision matrix
Λ =

Λaa
Λab
Λba
Λbb

(2.69)
corresponding to the partitioning (2.65) of the vector x. Because the inverse of a
symmetric matrix is also symmetric, we see that Λaa and Λbb are symmetric, while
Exercise 2.22
ΛT
ab = Λba. It should be stressed at this point that, for instance, Λaa is not simply
given by the inverse of Σaa. In fact, we shall shortly examine the relation between
the inverse of a partitioned matrix and the inverses of its partitions.
Let us begin by ﬁnding an expression for the conditional distribution p(xa|xb).
From the product rule of probability, we see that this conditional distribution can be


---
**Page 104**
86
2. PROBABILITY DISTRIBUTIONS
evaluated from the joint distribution p(x) = p(xa, xb) simply by ﬁxing xb to the
observed value and normalizing the resulting expression to obtain a valid probability
distribution over xa. Instead of performing this normalization explicitly, we can
obtain the solution more efﬁciently by considering the quadratic form in the exponent
of the Gaussian distribution given by (2.44) and then reinstating the normalization
coefﬁcient at the end of the calculation. If we make use of the partitioning (2.65),
(2.66), and (2.69), we obtain
−1
2(x −µ)TΣ−1(x −µ) =
−1
2(xa −µa)TΛaa(xa −µa) −1
2(xa −µa)TΛab(xb −µb)
−1
2(xb −µb)TΛba(xa −µa) −1
2(xb −µb)TΛbb(xb −µb).
(2.70)
We see that as a function of xa, this is again a quadratic form, and hence the cor-
responding conditional distribution p(xa|xb) will be Gaussian. Because this distri-
bution is completely characterized by its mean and its covariance, our goal will be
to identify expressions for the mean and covariance of p(xa|xb) by inspection of
(2.70).
This is an example of a rather common operation associated with Gaussian
distributions, sometimes called ‘completing the square’, in which we are given a
quadratic form deﬁning the exponent terms in a Gaussian distribution, and we need
to determine the corresponding mean and covariance. Such problems can be solved
straightforwardly by noting that the exponent in a general Gaussian distribution
N(x|µ, Σ) can be written
−1
2(x −µ)TΣ−1(x −µ) = −1
2xTΣ−1x + xTΣ−1µ + const
(2.71)
where ‘const’ denotes terms which are independent of x, and we have made use of
the symmetry of Σ. Thus if we take our general quadratic form and express it in
the form given by the right-hand side of (2.71), then we can immediately equate the
matrix of coefﬁcients entering the second order term in x to the inverse covariance
matrix Σ−1 and the coefﬁcient of the linear term in x to Σ−1µ, from which we can
obtain µ.
Now let us apply this procedure to the conditional Gaussian distribution p(xa|xb)
for which the quadratic form in the exponent is given by (2.70). We will denote the
mean and covariance of this distribution by µa|b and Σa|b, respectively. Consider
the functional dependence of (2.70) on xa in which xb is regarded as a constant. If
we pick out all terms that are second order in xa, we have
−1
2xT
a Λaaxa
(2.72)
from which we can immediately conclude that the covariance (inverse precision) of
p(xa|xb) is given by
Σa|b = Λ−1
aa .
(2.73)


---
**Page 105**
2.3. The Gaussian Distribution
87
Now consider all of the terms in (2.70) that are linear in xa
xT
a {Λaaµa −Λab(xb −µb)}
(2.74)
where we have used ΛT
ba = Λab. From our discussion of the general form (2.71),
the coefﬁcient of xa in this expression must equal Σ−1
a|bµa|b and hence
µa|b
=
Σa|b {Λaaµa −Λab(xb −µb)}
=
µa −Λ−1
aa Λab(xb −µb)
(2.75)
where we have made use of (2.73).
The results (2.73) and (2.75) are expressed in terms of the partitioned precision
matrix of the original joint distribution p(xa, xb). We can also express these results
in terms of the corresponding partitioned covariance matrix. To do this, we make use
of the following identity for the inverse of a partitioned matrix
Exercise 2.24

A
B
C
D
−1
=

M
−MBD−1
−D−1CM
D−1 + D−1CMBD−1

(2.76)
where we have deﬁned
M = (A −BD−1C)−1.
(2.77)
The quantity M−1 is known as the Schur complement of the matrix on the left-hand
side of (2.76) with respect to the submatrix D. Using the deﬁnition

Σaa
Σab
Σba
Σbb
−1
=

Λaa
Λab
Λba
Λbb

(2.78)
and making use of (2.76), we have
Λaa
=
(Σaa −ΣabΣ−1
bb Σba)−1
(2.79)
Λab
=
−(Σaa −ΣabΣ−1
bb Σba)−1ΣabΣ−1
bb .
(2.80)
From these we obtain the following expressions for the mean and covariance of the
conditional distribution p(xa|xb)
µa|b
=
µa + ΣabΣ−1
bb (xb −µb)
(2.81)
Σa|b
=
Σaa −ΣabΣ−1
bb Σba.
(2.82)
Comparing (2.73) and (2.82), we see that the conditional distribution p(xa|xb) takes
a simpler form when expressed in terms of the partitioned precision matrix than
when it is expressed in terms of the partitioned covariance matrix. Note that the
mean of the conditional distribution p(xa|xb), given by (2.81), is a linear function of
xb and that the covariance, given by (2.82), is independent of xa. This represents an
example of a linear-Gaussian model.
Section 8.1.4


---
**Page 106**
88
2. PROBABILITY DISTRIBUTIONS
2.3.2
Marginal Gaussian distributions
We have seen that if a joint distribution p(xa, xb) is Gaussian, then the condi-
tional distribution p(xa|xb) will again be Gaussian. Now we turn to a discussion of
the marginal distribution given by
p(xa) =

p(xa, xb) dxb
(2.83)
which, as we shall see, is also Gaussian. Once again, our strategy for evaluating this
distribution efﬁciently will be to focus on the quadratic form in the exponent of the
joint distribution and thereby to identify the mean and covariance of the marginal
distribution p(xa).
The quadratic form for the joint distribution can be expressed, using the par-
titioned precision matrix, in the form (2.70). Because our goal is to integrate out
xb, this is most easily achieved by ﬁrst considering the terms involving xb and then
completing the square in order to facilitate integration. Picking out just those terms
that involve xb, we have
−1
2xT
b Λbbxb+xT
b m = −1
2(xb−Λ−1
bb m)TΛbb(xb−Λ−1
bb m)+1
2mTΛ−1
bb m (2.84)
where we have deﬁned
m = Λbbµb −Λba(xa −µa).
(2.85)
We see that the dependence on xb has been cast into the standard quadratic form of a
Gaussian distribution corresponding to the ﬁrst term on the right-hand side of (2.84),
plus a term that does not depend on xb (but that does depend on xa). Thus, when
we take the exponential of this quadratic form, we see that the integration over xb
required by (2.83) will take the form

exp

−1
2(xb −Λ−1
bb m)TΛbb(xb −Λ−1
bb m)

dxb.
(2.86)
This integration is easily performed by noting that it is the integral over an unnor-
malized Gaussian, and so the result will be the reciprocal of the normalization co-
efﬁcient. We know from the form of the normalized Gaussian given by (2.43), that
this coefﬁcient is independent of the mean and depends only on the determinant of
the covariance matrix. Thus, by completing the square with respect to xb, we can
integrate out xb and the only term remaining from the contributions on the left-hand
side of (2.84) that depends on xa is the last term on the right-hand side of (2.84) in
which m is given by (2.85). Combining this term with the remaining terms from


---
**Page 107**
2.3. The Gaussian Distribution
89
(2.70) that depend on xa, we obtain
1
2 [Λbbµb −Λba(xa −µa)]T Λ−1
bb [Λbbµb −Λba(xa −µa)]
−1
2xT
a Λaaxa + xT
a (Λaaµa + Λabµb) + const
=
−1
2xT
a (Λaa −ΛabΛ−1
bb Λba)xa
+xT
a (Λaa −ΛabΛ−1
bb Λba)−1µa + const
(2.87)
where ‘const’ denotes quantities independent of xa. Again, by comparison with
(2.71), we see that the covariance of the marginal distribution of p(xa) is given by
Σa = (Λaa −ΛabΛ−1
bb Λba)−1.
(2.88)
Similarly, the mean is given by
Σa(Λaa −ΛabΛ−1
bb Λba)µa = µa
(2.89)
where we have used (2.88). The covariance in (2.88) is expressed in terms of the
partitioned precision matrix given by (2.69). We can rewrite this in terms of the
corresponding partitioning of the covariance matrix given by (2.67), as we did for
the conditional distribution. These partitioned matrices are related by

Λaa
Λab
Λba
Λbb
−1
=

Σaa
Σab
Σba
Σbb

(2.90)
Making use of (2.76), we then have

Λaa −ΛabΛ−1
bb Λba
−1 = Σaa.
(2.91)
Thus we obtain the intuitively satisfying result that the marginal distribution p(xa)
has mean and covariance given by
E[xa]
=
µa
(2.92)
cov[xa]
=
Σaa.
(2.93)
We see that for a marginal distribution, the mean and covariance are most simply ex-
pressed in terms of the partitioned covariance matrix, in contrast to the conditional
distribution for which the partitioned precision matrix gives rise to simpler expres-
sions.
Our results for the marginal and conditional distributions of a partitioned Gaus-
sian are summarized below.
Partitioned Gaussians
Given a joint Gaussian distribution N(x|µ, Σ) with Λ ≡Σ−1 and
x =

xa
xb

,
µ =

µa
µb

(2.94)


---
**Page 108**
90
2. PROBABILITY DISTRIBUTIONS
xa
xb = 0.7
xb
p(xa,xb)
0
0.5
1
0
0.5
1
xa
p(xa)
p(xa|xb = 0.7)
0
0.5
1
0
5
10
Figure 2.9
The plot on the left shows the contours of a Gaussian distribution p(xa, xb) over two variables, and
the plot on the right shows the marginal distribution p(xa) (blue curve) and the conditional distribution p(xa|xb)
for xb = 0.7 (red curve).
Σ =

Σaa
Σab
Σba
Σbb

,
Λ =

Λaa
Λab
Λba
Λbb

.
(2.95)
Conditional distribution:
p(xa|xb)
=
N(x|µa|b, Λ−1
aa )
(2.96)
µa|b
=
µa −Λ−1
aa Λab(xb −µb).
(2.97)
Marginal distribution:
p(xa) = N(xa|µa, Σaa).
(2.98)
We illustrate the idea of conditional and marginal distributions associated with
a multivariate Gaussian using an example involving two variables in Figure 2.9.
2.3.3
Bayes’ theorem for Gaussian variables
In Sections 2.3.1 and 2.3.2, we considered a Gaussian p(x) in which we parti-
tioned the vector x into two subvectors x = (xa, xb) and then found expressions for
the conditional distribution p(xa|xb) and the marginal distribution p(xa). We noted
that the mean of the conditional distribution p(xa|xb) was a linear function of xb.
Here we shall suppose that we are given a Gaussian marginal distribution p(x) and a
Gaussian conditional distribution p(y|x) in which p(y|x) has a mean that is a linear
function of x, and a covariance which is independent of x. This is an example of


---
**Page 109**
2.3. The Gaussian Distribution
91
a linear Gaussian model (Roweis and Ghahramani, 1999), which we shall study in
greater generality in Section 8.1.4. We wish to ﬁnd the marginal distribution p(y)
and the conditional distribution p(x|y). This is a problem that will arise frequently
in subsequent chapters, and it will prove convenient to derive the general results here.
We shall take the marginal and conditional distributions to be
p(x)
=
N 
x|µ, Λ−1
(2.99)
p(y|x)
=
N

y|Ax + b, L−1
(2.100)
where µ, A, and b are parameters governing the means, and Λ and L are precision
matrices. If x has dimensionality M and y has dimensionality D, then the matrix A
has size D × M.
First we ﬁnd an expression for the joint distribution over x and y. To do this, we
deﬁne
z =

x
y

(2.101)
and then consider the log of the joint distribution
ln p(z)
=
ln p(x) + ln p(y|x)
=
−1
2(x −µ)TΛ(x −µ)
−1
2(y −Ax −b)TL(y −Ax −b) + const
(2.102)
where ‘const’ denotes terms independent of x and y. As before, we see that this is a
quadratic function of the components of z, and hence p(z) is Gaussian distribution.
To ﬁnd the precision of this Gaussian, we consider the second order terms in (2.102),
which can be written as
−1
2xT(Λ + ATLA)x −1
2yTLy + 1
2yTLAx + 1
2xTATLy
=
−1
2

x
y
T 
Λ + ATLA
−ATL
−LA
L
 
x
y

= −1
2zTRz
(2.103)
and so the Gaussian distribution over z has precision (inverse covariance) matrix
given by
R =

Λ + ATLA
−ATL
−LA
L

.
(2.104)
The covariance matrix is found by taking the inverse of the precision, which can be
done using the matrix inversion formula (2.76) to give
Exercise 2.29
cov[z] = R−1 =

Λ−1
Λ−1AT
AΛ−1
L−1 + AΛ−1AT

.
(2.105)


---
**Page 110**
92
2. PROBABILITY DISTRIBUTIONS
Similarly, we can ﬁnd the mean of the Gaussian distribution over z by identify-
ing the linear terms in (2.102), which are given by
xTΛµ −xTATLb + yTLb =

x
y
T 
Λµ −ATLb
Lb

.
(2.106)
Using our earlier result (2.71) obtained by completing the square over the quadratic
form of a multivariate Gaussian, we ﬁnd that the mean of z is given by
E[z] = R−1

Λµ −ATLb
Lb

.
(2.107)
Making use of (2.105), we then obtain
Exercise 2.30
E[z] =

µ
Aµ + b

.
(2.108)
Next we ﬁnd an expression for the marginal distribution p(y) in which we have
marginalized over x. Recall that the marginal distribution over a subset of the com-
ponents of a Gaussian random vector takes a particularly simple form when ex-
pressed in terms of the partitioned covariance matrix. Speciﬁcally, its mean and
Section 2.3
covariance are given by (2.92) and (2.93), respectively. Making use of (2.105) and
(2.108) we see that the mean and covariance of the marginal distribution p(y) are
given by
E[y]
=
Aµ + b
(2.109)
cov[y]
=
L−1 + AΛ−1AT.
(2.110)
A special case of this result is when A = I, in which case it reduces to the convolu-
tion of two Gaussians, for which we see that the mean of the convolution is the sum
of the mean of the two Gaussians, and the covariance of the convolution is the sum
of their covariances.
Finally, we seek an expression for the conditional p(x|y). Recall that the results
for the conditional distribution are most easily expressed in terms of the partitioned
precision matrix, using (2.73) and (2.75). Applying these results to (2.105) and
Section 2.3
(2.108) we see that the conditional distribution p(x|y) has mean and covariance
given by
E[x|y]
=
(Λ + ATLA)−1 
ATL(y −b) + Λµ

(2.111)
cov[x|y]
=
(Λ + ATLA)−1.
(2.112)
The evaluation of this conditional can be seen as an example of Bayes’ theorem.
We can interpret the distribution p(x) as a prior distribution over x. If the variable
y is observed, then the conditional distribution p(x|y) represents the corresponding
posterior distribution over x. Having found the marginal and conditional distribu-
tions, we effectively expressed the joint distribution p(z) = p(x)p(y|x) in the form
p(x|y)p(y). These results are summarized below.


---
**Page 111**
2.3. The Gaussian Distribution
93
Marginal and Conditional Gaussians
Given a marginal Gaussian distribution for x and a conditional Gaussian distri-
bution for y given x in the form
p(x)
=
N(x|µ, Λ−1)
(2.113)
p(y|x)
=
N(y|Ax + b, L−1)
(2.114)
the marginal distribution of y and the conditional distribution of x given y are
given by
p(y)
=
N(y|Aµ + b, L−1 + AΛ−1AT)
(2.115)
p(x|y)
=
N(x|Σ{ATL(y −b) + Λµ}, Σ)
(2.116)
where
Σ = (Λ + ATLA)−1.
(2.117)
2.3.4
Maximum likelihood for the Gaussian
Given a data set X = (x1, . . . , xN)T in which the observations {xn} are as-
sumed to be drawn independently from a multivariate Gaussian distribution, we can
estimate the parameters of the distribution by maximum likelihood. The log likeli-
hood function is given by
ln p(X|µ, Σ) = −ND
2
ln(2π)−N
2 ln |Σ|−1
2
N

n=1
(xn−µ)TΣ−1(xn−µ). (2.118)
By simple rearrangement, we see that the likelihood function depends on the data set
only through the two quantities
N

n=1
xn,
N

n=1
xnxT
n.
(2.119)
These are known as the sufﬁcient statistics for the Gaussian distribution. Using
(C.19), the derivative of the log likelihood with respect to µ is given by
Appendix C
∂
∂µ ln p(X|µ, Σ) =
N

n=1
Σ−1(xn −µ)
(2.120)
and setting this derivative to zero, we obtain the solution for the maximum likelihood
estimate of the mean given by
µML = 1
N
N

n=1
xn
(2.121)


---
**Page 112**
94
2. PROBABILITY DISTRIBUTIONS
which is the mean of the observed set of data points. The maximization of (2.118)
with respect to Σ is rather more involved. The simplest approach is to ignore the
symmetry constraint and show that the resulting solution is symmetric as required.
Exercise 2.34
Alternative derivations of this result, which impose the symmetry and positive deﬁ-
niteness constraints explicitly, can be found in Magnus and Neudecker (1999). The
result is as expected and takes the form
ΣML = 1
N
N

n=1
(xn −µML)(xn −µML)T
(2.122)
which involves µML because this is the result of a joint maximization with respect
to µ and Σ. Note that the solution (2.121) for µML does not depend on ΣML, and so
we can ﬁrst evaluate µML and then use this to evaluate ΣML.
If we evaluate the expectations of the maximum likelihood solutions under the
true distribution, we obtain the following results
Exercise 2.35
E[µML]
=
µ
(2.123)
E[ΣML]
=
N −1
N
Σ.
(2.124)
We see that the expectation of the maximum likelihood estimate for the mean is equal
to the true mean. However, the maximum likelihood estimate for the covariance has
an expectation that is less than the true value, and hence it is biased. We can correct
this bias by deﬁning a different estimator Σ given by
Σ =
1
N −1
N

n=1
(xn −µML)(xn −µML)T.
(2.125)
Clearly from (2.122) and (2.124), the expectation of Σ is equal to Σ.
2.3.5
Sequential estimation
Our discussion of the maximum likelihood solution for the parameters of a Gaus-
sian distribution provides a convenient opportunity to give a more general discussion
of the topic of sequential estimation for maximum likelihood. Sequential methods
allow data points to be processed one at a time and then discarded and are important
for on-line applications, and also where large data sets are involved so that batch
processing of all data points at once is infeasible.
Consider the result (2.121) for the maximum likelihood estimator of the mean
µML, which we will denote by µ(N)
ML when it is based on N observations. If we


---
**Page 113**
2.3. The Gaussian Distribution
95
Figure 2.10
A schematic illustration of two correlated ran-
dom variables z and θ, together with the
regression function f(θ) given by the con-
ditional expectation E[z|θ].
The Robbins-
Monro algorithm provides a general sequen-
tial procedure for ﬁnding the root θ⋆of such
functions.
θ
z
θ⋆
f(θ)
dissect out the contribution from the ﬁnal data point xN, we obtain
µ(N)
ML
=
1
N
N

n=1
xn
=
1
N xN + 1
N
N−1

n=1
xn
=
1
N xN + N −1
N
µ(N−1)
ML
=
µ(N−1)
ML
+ 1
N (xN −µ(N−1)
ML
).
(2.126)
This result has a nice interpretation, as follows. After observing N −1 data points
we have estimated µ by µ(N−1)
ML
. We now observe data point xN, and we obtain our
revised estimate µ(N)
ML by moving the old estimate a small amount, proportional to
1/N, in the direction of the ‘error signal’ (xN −µ(N−1)
ML
). Note that, as N increases,
so the contribution from successive data points gets smaller.
The result (2.126) will clearly give the same answer as the batch result (2.121)
because the two formulae are equivalent. However, we will not always be able to de-
rive a sequential algorithm by this route, and so we seek a more general formulation
of sequential learning, which leads us to the Robbins-Monro algorithm. Consider a
pair of random variables θ and z governed by a joint distribution p(z, θ). The con-
ditional expectation of z given θ deﬁnes a deterministic function f(θ) that is given
by
f(θ) ≡E[z|θ] =

zp(z|θ) dz
(2.127)
and is illustrated schematically in Figure 2.10. Functions deﬁned in this way are
called regression functions.
Our goal is to ﬁnd the root θ⋆at which f(θ⋆) = 0. If we had a large data set
of observations of z and θ, then we could model the regression function directly and
then obtain an estimate of its root. Suppose, however, that we observe values of
z one at a time and we wish to ﬁnd a corresponding sequential estimation scheme
for θ⋆. The following general procedure for solving such problems was given by


---
**Page 114**
96
2. PROBABILITY DISTRIBUTIONS
Robbins and Monro (1951). We shall assume that the conditional variance of z is
ﬁnite so that
E 
(z −f)2 | θ	
< ∞
(2.128)
and we shall also, without loss of generality, consider the case where f(θ) > 0 for
θ > θ⋆and f(θ) < 0 for θ < θ⋆, as is the case in Figure 2.10. The Robbins-Monro
procedure then deﬁnes a sequence of successive estimates of the root θ⋆given by
θ(N) = θ(N−1) + aN−1z(θ(N−1))
(2.129)
where z(θ(N)) is an observed value of z when θ takes the value θ(N). The coefﬁcients
{aN} represent a sequence of positive numbers that satisfy the conditions
lim
N→∞aN
=
0
(2.130)
∞

N=1
aN
=
∞
(2.131)
∞

N=1
a2
N
<
∞.
(2.132)
It can then be shown (Robbins and Monro, 1951; Fukunaga, 1990) that the sequence
of estimates given by (2.129) does indeed converge to the root with probability one.
Note that the ﬁrst condition (2.130) ensures that the successive corrections decrease
in magnitude so that the process can converge to a limiting value. The second con-
dition (2.131) is required to ensure that the algorithm does not converge short of the
root, and the third condition (2.132) is needed to ensure that the accumulated noise
has ﬁnite variance and hence does not spoil convergence.
Now let us consider how a general maximum likelihood problem can be solved
sequentially using the Robbins-Monro algorithm. By deﬁnition, the maximum like-
lihood solution θML is a stationary point of the log likelihood function and hence
satisﬁes
∂
∂θ

1
N
N

n=1
ln p(xn|θ)

θML
= 0.
(2.133)
Exchanging the derivative and the summation, and taking the limit N →∞we have
lim
N→∞
1
N
N

n=1
∂
∂θ ln p(xn|θ) = Ex
 ∂
∂θ ln p(x|θ)

(2.134)
and so we see that ﬁnding the maximum likelihood solution corresponds to ﬁnd-
ing the root of a regression function. We can therefore apply the Robbins-Monro
procedure, which now takes the form
θ(N) = θ(N−1) + aN−1
∂
∂θ(N−1) ln p(xN|θ(N−1)).
(2.135)


---
**Page 115**
2.3. The Gaussian Distribution
97
Figure 2.11
In the case of a Gaussian distribution, with θ
corresponding to the mean µ, the regression
function illustrated in Figure 2.10 takes the form
of a straight line, as shown in red.
In this
case, the random variable z corresponds to the
derivative of the log likelihood function and is
given by (x −µML)/σ2, and its expectation that
deﬁnes the regression function is a straight line
given by (µ −µML)/σ2. The root of the regres-
sion function corresponds to the maximum like-
lihood estimator µML.
µ
z
p(z|µ)
µML
As a speciﬁc example, we consider once again the sequential estimation of the
mean of a Gaussian distribution, in which case the parameter θ(N) is the estimate
µ(N)
ML of the mean of the Gaussian, and the random variable z is given by
z =
∂
∂µML
ln p(x|µML, σ2) = 1
σ2 (x −µML).
(2.136)
Thus the distribution of z is Gaussian with mean µ −µML, as illustrated in Fig-
ure 2.11. Substituting (2.136) into (2.135), we obtain the univariate form of (2.126),
provided we choose the coefﬁcients aN to have the form aN = σ2/N. Note that
although we have focussed on the case of a single variable, the same technique,
together with the same restrictions (2.130)–(2.132) on the coefﬁcients aN, apply
equally to the multivariate case (Blum, 1965).
2.3.6
Bayesian inference for the Gaussian
The maximum likelihood framework gave point estimates for the parameters µ
and Σ. Now we develop a Bayesian treatment by introducing prior distributions
over these parameters. Let us begin with a simple example in which we consider a
single Gaussian random variable x. We shall suppose that the variance σ2 is known,
and we consider the task of inferring the mean µ given a set of N observations
X = {x1, . . . , xN}. The likelihood function, that is the probability of the observed
data given µ, viewed as a function of µ, is given by
p(X|µ) =
N

n=1
p(xn|µ) =
1
(2πσ2)N/2 exp

−1
2σ2
N

n=1
(xn −µ)2

.
(2.137)
Again we emphasize that the likelihood function p(X|µ) is not a probability distri-
bution over µ and is not normalized.
We see that the likelihood function takes the form of the exponential of a quad-
ratic form in µ. Thus if we choose a prior p(µ) given by a Gaussian, it will be a


---
**Page 116**
98
2. PROBABILITY DISTRIBUTIONS
conjugate distribution for this likelihood function because the corresponding poste-
rior will be a product of two exponentials of quadratic functions of µ and hence will
also be Gaussian. We therefore take our prior distribution to be
p(µ) = N

µ|µ0, σ2
0

(2.138)
and the posterior distribution is given by
p(µ|X) ∝p(X|µ)p(µ).
(2.139)
Simple manipulation involving completing the square in the exponent shows that the
Exercise 2.38
posterior distribution is given by
p(µ|X) = N 
µ|µN, σ2
N

(2.140)
where
µN
=
σ2
Nσ2
0 + σ2 µ0 +
Nσ2
0
Nσ2
0 + σ2 µML
(2.141)
1
σ2
N
=
1
σ2
0
+ N
σ2
(2.142)
in which µML is the maximum likelihood solution for µ given by the sample mean
µML = 1
N
N

n=1
xn.
(2.143)
It is worth spending a moment studying the form of the posterior mean and
variance. First of all, we note that the mean of the posterior distribution given by
(2.141) is a compromise between the prior mean µ0 and the maximum likelihood
solution µML. If the number of observed data points N = 0, then (2.141) reduces
to the prior mean as expected. For N →∞, the posterior mean is given by the
maximum likelihood solution. Similarly, consider the result (2.142) for the variance
of the posterior distribution. We see that this is most naturally expressed in terms
of the inverse variance, which is called the precision. Furthermore, the precisions
are additive, so that the precision of the posterior is given by the precision of the
prior plus one contribution of the data precision from each of the observed data
points. As we increase the number of observed data points, the precision steadily
increases, corresponding to a posterior distribution with steadily decreasing variance.
With no observed data points, we have the prior variance, whereas if the number of
data points N →∞, the variance σ2
N goes to zero and the posterior distribution
becomes inﬁnitely peaked around the maximum likelihood solution. We therefore
see that the maximum likelihood result of a point estimate for µ given by (2.143) is
recovered precisely from the Bayesian formalism in the limit of an inﬁnite number
of observations. Note also that for ﬁnite N, if we take the limit σ2
0 →∞in which the
prior has inﬁnite variance then the posterior mean (2.141) reduces to the maximum
likelihood result, while from (2.142) the posterior variance is given by σ2
N = σ2/N.


---
**Page 117**
2.3. The Gaussian Distribution
99
Figure 2.12
Illustration of Bayesian inference for
the mean µ of a Gaussian distri-
bution, in which the variance is as-
sumed to be known.
The curves
show the prior distribution over µ
(the curve labelled N = 0), which
in this case is itself Gaussian, along
with the posterior distribution given
by (2.140) for increasing numbers N
of data points. The data points are
generated from a Gaussian of mean
0.8 and variance 0.1, and the prior is
chosen to have mean 0. In both the
prior and the likelihood function, the
variance is set to the true value.
N = 0
N = 1
N = 2
N = 10
−1
0
1
0
5
We illustrate our analysis of Bayesian inference for the mean of a Gaussian
distribution in Figure 2.12. The generalization of this result to the case of a D-
dimensional Gaussian random variable x with known covariance and unknown mean
is straightforward.
Exercise 2.40
We have already seen how the maximum likelihood expression for the mean of
a Gaussian can be re-cast as a sequential update formula in which the mean after
Section 2.3.5
observing N data points was expressed in terms of the mean after observing N −1
data points together with the contribution from data point xN. In fact, the Bayesian
paradigm leads very naturally to a sequential view of the inference problem. To see
this in the context of the inference of the mean of a Gaussian, we write the posterior
distribution with the contribution from the ﬁnal data point xN separated out so that
p(µ|D) ∝

p(µ)
N−1

n=1
p(xn|µ)
 
p(xN|µ).
(2.144)
The term in square brackets is (up to a normalization coefﬁcient) just the posterior
distribution after observing N −1 data points. We see that this can be viewed as
a prior distribution, which is combined using Bayes’ theorem with the likelihood
function associated with data point xN to arrive at the posterior distribution after
observing N data points. This sequential view of Bayesian inference is very general
and applies to any problem in which the observed data are assumed to be independent
and identically distributed.
So far, we have assumed that the variance of the Gaussian distribution over the
data is known and our goal is to infer the mean. Now let us suppose that the mean
is known and we wish to infer the variance. Again, our calculations will be greatly
simpliﬁed if we choose a conjugate form for the prior distribution. It turns out to be
most convenient to work with the precision λ ≡1/σ2. The likelihood function for λ
takes the form
p(X|λ) =
N

n=1
N(xn|µ, λ−1) ∝λN/2 exp

−λ
2
N

n=1
(xn −µ)2

.
(2.145)


---
**Page 118**
100
2. PROBABILITY DISTRIBUTIONS
λ
a = 0.1
b = 0.1
0
1
2
0
1
2
λ
a = 1
b = 1
0
1
2
0
1
2
λ
a = 4
b = 6
0
1
2
0
1
2
Figure 2.13
Plot of the gamma distribution Gam(λ|a, b) deﬁned by (2.146) for various values of the parameters
a and b.
The corresponding conjugate prior should therefore be proportional to the product
of a power of λ and the exponential of a linear function of λ. This corresponds to
the gamma distribution which is deﬁned by
Gam(λ|a, b) =
1
Γ(a)baλa−1 exp(−bλ).
(2.146)
Here Γ(a) is the gamma function that is deﬁned by (1.141) and that ensures that
(2.146) is correctly normalized. The gamma distribution has a ﬁnite integral if a > 0,
Exercise 2.41
and the distribution itself is ﬁnite if a ⩾1. It is plotted, for various values of a and
b, in Figure 2.13. The mean and variance of the gamma distribution are given by
Exercise 2.42
E[λ]
=
a
b
(2.147)
var[λ]
=
a
b2 .
(2.148)
Consider a prior distribution Gam(λ|a0, b0). If we multiply by the likelihood
function (2.145), then we obtain a posterior distribution
p(λ|X) ∝λa0−1λN/2 exp

−b0λ −λ
2
N

n=1
(xn −µ)2

(2.149)
which we recognize as a gamma distribution of the form Gam(λ|aN, bN) where
aN
=
a0 + N
2
(2.150)
bN
=
b0 + 1
2
N

n=1
(xn −µ)2 = b0 + N
2 σ2
ML
(2.151)
where σ2
ML is the maximum likelihood estimator of the variance. Note that in (2.149)
there is no need to keep track of the normalization constants in the prior and the
likelihood function because, if required, the correct coefﬁcient can be found at the
end using the normalized form (2.146) for the gamma distribution.


---
**Page 119**
2.3. The Gaussian Distribution
101
From (2.150), we see that the effect of observing N data points is to increase
the value of the coefﬁcient a by N/2. Thus we can interpret the parameter a0 in
the prior in terms of 2a0 ‘effective’ prior observations. Similarly, from (2.151) we
see that the N data points contribute Nσ2
ML/2 to the parameter b, where σ2
ML is
the variance, and so we can interpret the parameter b0 in the prior as arising from
the 2a0 ‘effective’ prior observations having variance 2b0/(2a0) = b0/a0. Recall
that we made an analogous interpretation for the Dirichlet prior. These distributions
Section 2.2
are examples of the exponential family, and we shall see that the interpretation of
a conjugate prior in terms of effective ﬁctitious data points is a general one for the
exponential family of distributions.
Instead of working with the precision, we can consider the variance itself. The
conjugate prior in this case is called the inverse gamma distribution, although we
shall not discuss this further because we will ﬁnd it more convenient to work with
the precision.
Now suppose that both the mean and the precision are unknown. To ﬁnd a
conjugate prior, we consider the dependence of the likelihood function on µ and λ
p(X|µ, λ) =
N

n=1
 λ
2π
1/2
exp

−λ
2 (xn −µ)2

∝

λ1/2 exp

−λµ2
2
N
exp

λµ
N

n=1
xn −λ
2
N

n=1
x2
n

.
(2.152)
We now wish to identify a prior distribution p(µ, λ) that has the same functional
dependence on µ and λ as the likelihood function and that should therefore take the
form
p(µ, λ) ∝

λ1/2 exp

−λµ2
2
β
exp {cλµ −dλ}
=
exp

−βλ
2 (µ −c/β)2

λβ/2 exp

−

d −c2
2β

λ

(2.153)
where c, d, and β are constants. Since we can always write p(µ, λ) = p(µ|λ)p(λ),
we can ﬁnd p(µ|λ) and p(λ) by inspection. In particular, we see that p(µ|λ) is a
Gaussian whose precision is a linear function of λ and that p(λ) is a gamma distri-
bution, so that the normalized prior takes the form
p(µ, λ) = N(µ|µ0, (βλ)−1)Gam(λ|a, b)
(2.154)
where we have deﬁned new constants given by µ0 = c/β, a = 1 + β/2, b =
d−c2/2β. The distribution (2.154) is called the normal-gamma or Gaussian-gamma
distribution and is plotted in Figure 2.14. Note that this is not simply the product
of an independent Gaussian prior over µ and a gamma prior over λ, because the
precision of µ is a linear function of λ. Even if we chose a prior in which µ and λ
were independent, the posterior distribution would exhibit a coupling between the
precision of µ and the value of λ.


---
**Page 120**
102
2. PROBABILITY DISTRIBUTIONS
Figure 2.14
Contour plot of the normal-gamma
distribution (2.154) for parameter
values µ0 = 0, β = 2, a = 5 and
b = 6.
µ
λ
−2
0
2
0
1
2
In the case of the multivariate Gaussian distribution N 
x|µ, Λ−1
for a D-
dimensional variable x, the conjugate prior distribution for the mean µ, assuming
the precision is known, is again a Gaussian. For known mean and unknown precision
matrix Λ, the conjugate prior is the Wishart distribution given by
Exercise 2.45
W(Λ|W, ν) = B|Λ|(ν−D−1)/2 exp

−1
2Tr(W−1Λ)

(2.155)
where ν is called the number of degrees of freedom of the distribution, W is a D×D
scale matrix, and Tr(·) denotes the trace. The normalization constant B is given by
B(W, ν) = |W|−ν/2

2νD/2 πD(D−1)/4
D

i=1
Γ
ν + 1 −i
2
−1
.
(2.156)
Again, it is also possible to deﬁne a conjugate prior over the covariance matrix itself,
rather than over the precision matrix, which leads to the inverse Wishart distribu-
tion, although we shall not discuss this further. If both the mean and the precision
are unknown, then, following a similar line of reasoning to the univariate case, the
conjugate prior is given by
p(µ, Λ|µ0, β, W, ν) = N(µ|µ0, (βΛ)−1) W(Λ|W, ν)
(2.157)
which is known as the normal-Wishart or Gaussian-Wishart distribution.
2.3.7
Student’s t-distribution
We have seen that the conjugate prior for the precision of a Gaussian is given
by a gamma distribution. If we have a univariate Gaussian N(x|µ, τ −1) together
Section 2.3.6
with a Gamma prior Gam(τ|a, b) and we integrate out the precision, we obtain the
marginal distribution of x in the form
Exercise 2.46


---
**Page 121**
2.3. The Gaussian Distribution
103
Figure 2.15
Plot of Student’s t-distribution (2.159)
for µ = 0 and λ = 1 for various values
of ν. The limit ν →∞corresponds
to a Gaussian distribution with mean
µ and precision λ.
ν →∞
ν = 1.0
ν = 0.1
−5
0
5
0
0.1
0.2
0.3
0.4
0.5
p(x|µ, a, b)
=
 ∞
0
N(x|µ, τ −1)Gam(τ|a, b) dτ
(2.158)
=
 ∞
0
bae(−bτ)τ a−1
Γ(a)
 τ
2π
1/2
exp

−τ
2(x −µ)2
dτ
=
ba
Γ(a)
 1
2π
1/2 
b + (x −µ)2
2
−a−1/2
Γ(a + 1/2)
where we have made the change of variable z = τ[b + (x −µ)2/2]. By convention
we deﬁne new parameters given by ν = 2a and λ = a/b, in terms of which the
distribution p(x|µ, a, b) takes the form
St(x|µ, λ, ν) = Γ(ν/2 + 1/2)
Γ(ν/2)
 λ
πν
1/2 
1 + λ(x −µ)2
ν
−ν/2−1/2
(2.159)
which is known as Student’s t-distribution. The parameter λ is sometimes called the
precision of the t-distribution, even though it is not in general equal to the inverse
of the variance. The parameter ν is called the degrees of freedom, and its effect is
illustrated in Figure 2.15. For the particular case of ν = 1, the t-distribution reduces
to the Cauchy distribution, while in the limit ν →∞the t-distribution St(x|µ, λ, ν)
becomes a Gaussian N(x|µ, λ−1) with mean µ and precision λ.
Exercise 2.47
From (2.158), we see that Student’s t-distribution is obtained by adding up an
inﬁnite number of Gaussian distributions having the same mean but different preci-
sions. This can be interpreted as an inﬁnite mixture of Gaussians (Gaussian mixtures
will be discussed in detail in Section 2.3.9. The result is a distribution that in gen-
eral has longer ‘tails’ than a Gaussian, as was seen in Figure 2.15. This gives the t-
distribution an important property called robustness, which means that it is much less
sensitive than the Gaussian to the presence of a few data points which are outliers.
The robustness of the t-distribution is illustrated in Figure 2.16, which compares the
maximum likelihood solutions for a Gaussian and a t-distribution. Note that the max-
imum likelihood solution for the t-distribution can be found using the expectation-
maximization (EM) algorithm. Here we see that the effect of a small number of
Exercise 12.24


---
**Page 122**
104
2. PROBABILITY DISTRIBUTIONS
(a)
−5
0
5
10
0
0.1
0.2
0.3
0.4
0.5
(b)
−5
0
5
10
0
0.1
0.2
0.3
0.4
0.5
Figure 2.16
Illustration of the robustness of Student’s t-distribution compared to a Gaussian. (a) Histogram
distribution of 30 data points drawn from a Gaussian distribution, together with the maximum likelihood ﬁt ob-
tained from a t-distribution (red curve) and a Gaussian (green curve, largely hidden by the red curve). Because
the t-distribution contains the Gaussian as a special case it gives almost the same solution as the Gaussian.
(b) The same data set but with three additional outlying data points showing how the Gaussian (green curve) is
strongly distorted by the outliers, whereas the t-distribution (red curve) is relatively unaffected.
outliers is much less signiﬁcant for the t-distribution than for the Gaussian. Outliers
can arise in practical applications either because the process that generates the data
corresponds to a distribution having a heavy tail or simply through mislabelled data.
Robustness is also an important property for regression problems. Unsurprisingly,
the least squares approach to regression does not exhibit robustness, because it cor-
responds to maximum likelihood under a (conditional) Gaussian distribution. By
basing a regression model on a heavy-tailed distribution such as a t-distribution, we
obtain a more robust model.
If we go back to (2.158) and substitute the alternative parameters ν = 2a, λ =
a/b, and η = τb/a, we see that the t-distribution can be written in the form
St(x|µ, λ, ν) =
 ∞
0
N

x|µ, (ηλ)−1
Gam(η|ν/2, ν/2) dη.
(2.160)
We can then generalize this to a multivariate Gaussian N(x|µ, Λ) to obtain the cor-
responding multivariate Student’s t-distribution in the form
St(x|µ, Λ, ν) =
 ∞
0
N(x|µ, (ηΛ)−1)Gam(η|ν/2, ν/2) dη.
(2.161)
Using the same technique as for the univariate case, we can evaluate this integral to
give
Exercise 2.48


---
**Page 123**
2.3. The Gaussian Distribution
105
St(x|µ, Λ, ν) = Γ(D/2 + ν/2)
Γ(ν/2)
|Λ|1/2
(πν)D/2

1 + ∆2
ν
−D/2−ν/2
(2.162)
where D is the dimensionality of x, and ∆2 is the squared Mahalanobis distance
deﬁned by
∆2 = (x −µ)TΛ(x −µ).
(2.163)
This is the multivariate form of Student’s t-distribution and satisﬁes the following
properties
Exercise 2.49
E[x]
=
µ,
if
ν > 1
(2.164)
cov[x]
=
ν
(ν −2)Λ−1,
if
ν > 2
(2.165)
mode[x]
=
µ
(2.166)
with corresponding results for the univariate case.
2.3.8
Periodic variables
Although Gaussian distributions are of great practical signiﬁcance, both in their
own right and as building blocks for more complex probabilistic models, there are
situations in which they are inappropriate as density models for continuous vari-
ables. One important case, which arises in practical applications, is that of periodic
variables.
An example of a periodic variable would be the wind direction at a particular
geographical location. We might, for instance, measure values of wind direction on a
number of days and wish to summarize this using a parametric distribution. Another
example is calendar time, where we may be interested in modelling quantities that
are believed to be periodic over 24 hours or over an annual cycle. Such quantities
can conveniently be represented using an angular (polar) coordinate 0 ⩽θ < 2π.
We might be tempted to treat periodic variables by choosing some direction
as the origin and then applying a conventional distribution such as the Gaussian.
Such an approach, however, would give results that were strongly dependent on the
arbitrary choice of origin. Suppose, for instance, that we have two observations at
θ1 = 1◦and θ2 = 359◦, and we model them using a standard univariate Gaussian
distribution. If we choose the origin at 0◦, then the sample mean of this data set
will be 180◦with standard deviation 179◦, whereas if we choose the origin at 180◦,
then the mean will be 0◦and the standard deviation will be 1◦. We clearly need to
develop a special approach for the treatment of periodic variables.
Let us consider the problem of evaluating the mean of a set of observations
D = {θ1, . . . , θN} of a periodic variable. From now on, we shall assume that θ is
measured in radians. We have already seen that the simple average (θ1+· · ·+θN)/N
will be strongly coordinate dependent. To ﬁnd an invariant measure of the mean, we
note that the observations can be viewed as points on the unit circle and can therefore
be described instead by two-dimensional unit vectors x1, . . . , xN where ∥xn∥= 1
for n = 1, . . . , N, as illustrated in Figure 2.17. We can average the vectors {xn}


---
**Page 124**
106
2. PROBABILITY DISTRIBUTIONS
Figure 2.17
Illustration of the representation of val-
ues θn of a periodic variable as two-
dimensional vectors xn living on the unit
circle. Also shown is the average x of
those vectors.
x1
x2
x1
x2
x3
x4
¯x
¯r
¯θ
instead to give
x = 1
N
N

n=1
xn
(2.167)
and then ﬁnd the corresponding angle θ of this average. Clearly, this deﬁnition will
ensure that the location of the mean is independent of the origin of the angular coor-
dinate. Note that x will typically lie inside the unit circle. The Cartesian coordinates
of the observations are given by xn = (cos θn, sin θn), and we can write the Carte-
sian coordinates of the sample mean in the form x = (r cos θ, r sin θ). Substituting
into (2.167) and equating the x1 and x2 components then gives
r cos θ = 1
N
N

n=1
cos θn,
r sin θ = 1
N
N

n=1
sin θn.
(2.168)
Taking the ratio, and using the identity tan θ = sin θ/ cos θ, we can solve for θ to
give
θ = tan−1
 
n sin θn

n cos θn

.
(2.169)
Shortly, we shall see how this result arises naturally as the maximum likelihood
estimator for an appropriately deﬁned distribution over a periodic variable.
We now consider a periodic generalization of the Gaussian called the von Mises
distribution. Here we shall limit our attention to univariate distributions, although
periodic distributions can also be found over hyperspheres of arbitrary dimension.
For an extensive discussion of periodic distributions, see Mardia and Jupp (2000).
By convention, we will consider distributions p(θ) that have period 2π. Any
probability density p(θ) deﬁned over θ must not only be nonnegative and integrate


---
**Page 125**
2.3. The Gaussian Distribution
107
Figure 2.18
The von Mises distribution can be derived by considering
a two-dimensional Gaussian of the form (2.173), whose
density contours are shown in blue and conditioning on
the unit circle shown in red.
x1
x2
p(x)
r = 1
to one, but it must also be periodic. Thus p(θ) must satisfy the three conditions
p(θ)
⩾
0
(2.170)
 2π
0
p(θ) dθ
=
1
(2.171)
p(θ + 2π)
=
p(θ).
(2.172)
From (2.172), it follows that p(θ + M2π) = p(θ) for any integer M.
We can easily obtain a Gaussian-like distribution that satisﬁes these three prop-
erties as follows. Consider a Gaussian distribution over two variables x = (x1, x2)
having mean µ = (µ1, µ2) and a covariance matrix Σ = σ2I where I is the 2 × 2
identity matrix, so that
p(x1, x2) =
1
2πσ2 exp

−(x1 −µ1)2 + (x2 −µ2)2
2σ2

.
(2.173)
The contours of constant p(x) are circles, as illustrated in Figure 2.18. Now suppose
we consider the value of this distribution along a circle of ﬁxed radius. Then by con-
struction this distribution will be periodic, although it will not be normalized. We can
determine the form of this distribution by transforming from Cartesian coordinates
(x1, x2) to polar coordinates (r, θ) so that
x1 = r cos θ,
x2 = r sin θ.
(2.174)
We also map the mean µ into polar coordinates by writing
µ1 = r0 cos θ0,
µ2 = r0 sin θ0.
(2.175)
Next we substitute these transformations into the two-dimensional Gaussian distribu-
tion (2.173), and then condition on the unit circle r = 1, noting that we are interested
only in the dependence on θ. Focussing on the exponent in the Gaussian distribution
we have
−1
2σ2

(r cos θ −r0 cos θ0)2 + (r sin θ −r0 sin θ0)2
=
−1
2σ2

1 + r2
0 −2r0 cos θ cos θ0 −2r0 sin θ sin θ0

=
r0
σ2 cos(θ −θ0) + const
(2.176)


---
**Page 126**
108
2. PROBABILITY DISTRIBUTIONS
m = 5, θ0 = π/4
m = 1, θ0 = 3π/4
2π
0
π/4
3π/4
m = 5, θ0 = π/4
m = 1, θ0 = 3π/4
Figure 2.19
The von Mises distribution plotted for two different parameter values, shown as a Cartesian plot
on the left and as the corresponding polar plot on the right.
where ‘const’ denotes terms independent of θ, and we have made use of the following
trigonometrical identities
Exercise 2.51
cos2 A + sin2 A
=
1
(2.177)
cos A cos B + sin A sin B
=
cos(A −B).
(2.178)
If we now deﬁne m = r0/σ2, we obtain our ﬁnal expression for the distribution of
p(θ) along the unit circle r = 1 in the form
p(θ|θ0, m) =
1
2πI0(m) exp {m cos(θ −θ0)}
(2.179)
which is called the von Mises distribution, or the circular normal. Here the param-
eter θ0 corresponds to the mean of the distribution, while m, which is known as
the concentration parameter, is analogous to the inverse variance (precision) for the
Gaussian. The normalization coefﬁcient in (2.179) is expressed in terms of I0(m),
which is the zeroth-order Bessel function of the ﬁrst kind (Abramowitz and Stegun,
1965) and is deﬁned by
I0(m) = 1
2π
 2π
0
exp {m cos θ} dθ.
(2.180)
For large m, the distribution becomes approximately Gaussian. The von Mises dis-
Exercise 2.52
tribution is plotted in Figure 2.19, and the function I0(m) is plotted in Figure 2.20.
Now consider the maximum likelihood estimators for the parameters θ0 and m
for the von Mises distribution. The log likelihood function is given by
ln p(D|θ0, m) = −N ln(2π) −N ln I0(m) + m
N

n=1
cos(θn −θ0).
(2.181)


---
**Page 127**
2.3. The Gaussian Distribution
109
I0(m)
m
0
5
10
0
1000
2000
3000
A(m)
m
0
5
10
0
0.5
1
Figure 2.20
Plot of the Bessel function I0(m) deﬁned by (2.180), together with the function A(m) deﬁned by
(2.186).
Setting the derivative with respect to θ0 equal to zero gives
N

n=1
sin(θn −θ0) = 0.
(2.182)
To solve for θ0, we make use of the trigonometric identity
sin(A −B) = cos B sin A −cos A sin B
(2.183)
from which we obtain
Exercise 2.53
θML
0
= tan−1
 
n sin θn

n cos θn

(2.184)
which we recognize as the result (2.169) obtained earlier for the mean of the obser-
vations viewed in a two-dimensional Cartesian space.
Similarly, maximizing (2.181) with respect to m, and making use of I′
0(m) =
I1(m) (Abramowitz and Stegun, 1965), we have
A(m) = 1
N
N

n=1
cos(θn −θML
0
)
(2.185)
where we have substituted for the maximum likelihood solution for θML
0
(recalling
that we are performing a joint optimization over θ and m), and we have deﬁned
A(m) = I1(m)
I0(m).
(2.186)
The function A(m) is plotted in Figure 2.20. Making use of the trigonometric iden-
tity (2.178), we can write (2.185) in the form
A(mML) =

1
N
N

n=1
cos θn

cos θML
0
−

1
N
N

n=1
sin θn

sin θML
0
.
(2.187)


---
**Page 128**
110
2. PROBABILITY DISTRIBUTIONS
Figure 2.21
Plots of the ‘old faith-
ful’ data in which the blue curves
show contours of constant proba-
bility density.
On the left is a
single Gaussian distribution which
has been ﬁtted to the data us-
ing maximum likelihood. Note that
this distribution fails to capture the
two clumps in the data and indeed
places much of its probability mass
in the central region between the
clumps where the data are relatively
sparse. On the right the distribution
is given by a linear combination of
two Gaussians which has been ﬁtted
to the data by maximum likelihood
using techniques discussed Chap-
ter 9, and which gives a better rep-
resentation of the data.
1
2
3
4
5
6
40
60
80
100
1
2
3
4
5
6
40
60
80
100
The right-hand side of (2.187) is easily evaluated, and the function A(m) can be
inverted numerically.
For completeness, we mention brieﬂy some alternative techniques for the con-
struction of periodic distributions. The simplest approach is to use a histogram of
observations in which the angular coordinate is divided into ﬁxed bins. This has the
virtue of simplicity and ﬂexibility but also suffers from signiﬁcant limitations, as we
shall see when we discuss histogram methods in more detail in Section 2.5. Another
approach starts, like the von Mises distribution, from a Gaussian distribution over a
Euclidean space but now marginalizes onto the unit circle rather than conditioning
(Mardia and Jupp, 2000). However, this leads to more complex forms of distribution
and will not be discussed further. Finally, any valid distribution over the real axis
(such as a Gaussian) can be turned into a periodic distribution by mapping succes-
sive intervals of width 2π onto the periodic variable (0, 2π), which corresponds to
‘wrapping’ the real axis around unit circle. Again, the resulting distribution is more
complex to handle than the von Mises distribution.
One limitation of the von Mises distribution is that it is unimodal. By forming
mixtures of von Mises distributions, we obtain a ﬂexible framework for modelling
periodic variables that can handle multimodality. For an example of a machine learn-
ing application that makes use of von Mises distributions, see Lawrence et al. (2002),
and for extensions to modelling conditional densities for regression problems, see
Bishop and Nabney (1996).
2.3.9
Mixtures of Gaussians
While the Gaussian distribution has some important analytical properties, it suf-
fers from signiﬁcant limitations when it comes to modelling real data sets. Consider
the example shown in Figure 2.21. This is known as the ‘Old Faithful’ data set,
and comprises 272 measurements of the eruption of the Old Faithful geyser at Yel-
lowstone National Park in the USA. Each measurement comprises the duration of
Appendix A


---
**Page 129**
2.3. The Gaussian Distribution
111
Figure 2.22
Example of a Gaussian mixture distribution
in one dimension showing three Gaussians
(each scaled by a coefﬁcient) in blue and
their sum in red.
x
p(x)
the eruption in minutes (horizontal axis) and the time in minutes to the next erup-
tion (vertical axis). We see that the data set forms two dominant clumps, and that
a simple Gaussian distribution is unable to capture this structure, whereas a linear
superposition of two Gaussians gives a better characterization of the data set.
Such superpositions, formed by taking linear combinations of more basic dis-
tributions such as Gaussians, can be formulated as probabilistic models known as
mixture distributions (McLachlan and Basford, 1988; McLachlan and Peel, 2000).
In Figure 2.22 we see that a linear combination of Gaussians can give rise to very
complex densities. By using a sufﬁcient number of Gaussians, and by adjusting their
means and covariances as well as the coefﬁcients in the linear combination, almost
any continuous density can be approximated to arbitrary accuracy.
We therefore consider a superposition of K Gaussian densities of the form
p(x) =
K

k=1
πkN(x|µk, Σk)
(2.188)
which is called a mixture of Gaussians. Each Gaussian density N(x|µk, Σk) is
called a component of the mixture and has its own mean µk and covariance Σk.
Contour and surface plots for a Gaussian mixture having 3 components are shown in
Figure 2.23.
In this section we shall consider Gaussian components to illustrate the frame-
work of mixture models. More generally, mixture models can comprise linear com-
binations of other distributions. For instance, in Section 9.3.3 we shall consider
mixtures of Bernoulli distributions as an example of a mixture model for discrete
variables.
Section 9.3.3
The parameters πk in (2.188) are called mixing coefﬁcients. If we integrate both
sides of (2.188) with respect to x, and note that both p(x) and the individual Gaussian
components are normalized, we obtain
K

k=1
πk = 1.
(2.189)
Also, the requirement that p(x) ⩾0, together with N(x|µk, Σk) ⩾0, implies
πk ⩾0 for all k. Combining this with the condition (2.189) we obtain
0 ⩽πk ⩽1.
(2.190)


---
**Page 130**
112
2. PROBABILITY DISTRIBUTIONS
0.5
0.3
0.2
(a)
0
0.5
1
0
0.5
1
(b)
0
0.5
1
0
0.5
1
Figure 2.23
Illustration of a mixture of 3 Gaussians in a two-dimensional space. (a) Contours of constant
density for each of the mixture components, in which the 3 components are denoted red, blue and green, and
the values of the mixing coefﬁcients are shown below each component. (b) Contours of the marginal probability
density p(x) of the mixture distribution. (c) A surface plot of the distribution p(x).
We therefore see that the mixing coefﬁcients satisfy the requirements to be probabil-
ities.
From the sum and product rules, the marginal density is given by
p(x) =
K

k=1
p(k)p(x|k)
(2.191)
which is equivalent to (2.188) in which we can view πk = p(k) as the prior prob-
ability of picking the kth component, and the density N(x|µk, Σk) = p(x|k) as
the probability of x conditioned on k. As we shall see in later chapters, an impor-
tant role is played by the posterior probabilities p(k|x), which are also known as
responsibilities. From Bayes’ theorem these are given by
γk(x)
≡
p(k|x)
=
p(k)p(x|k)

l p(l)p(x|l)
=
πkN(x|µk, Σk)

l πlN(x|µl, Σl).
(2.192)
We shall discuss the probabilistic interpretation of the mixture distribution in greater
detail in Chapter 9.
The form of the Gaussian mixture distribution is governed by the parameters π,
µ and Σ, where we have used the notation π ≡{π1, . . . , πK}, µ ≡{µ1, . . . , µK}
and Σ ≡{Σ1, . . . ΣK}. One way to set the values of these parameters is to use
maximum likelihood. From (2.188) the log of the likelihood function is given by
ln p(X|π, µ, Σ) =
N

n=1
ln
 K

k=1
πkN(xn|µk, Σk)

(2.193)


---
**Page 131**
2.4. The Exponential Family
113
where X = {x1, . . . , xN}. We immediately see that the situation is now much
more complex than with a single Gaussian, due to the presence of the summation
over k inside the logarithm. As a result, the maximum likelihood solution for the
parameters no longer has a closed-form analytical solution. One approach to maxi-
mizing the likelihood function is to use iterative numerical optimization techniques
(Fletcher, 1987; Nocedal and Wright, 1999; Bishop and Nabney, 2008). Alterna-
tively we can employ a powerful framework called expectation maximization, which
will be discussed at length in Chapter 9.
2.4. The Exponential Family
The probability distributions that we have studied so far in this chapter (with the
exception of the Gaussian mixture) are speciﬁc examples of a broad class of distri-
butions called the exponential family (Duda and Hart, 1973; Bernardo and Smith,
1994). Members of the exponential family have many important properties in com-
mon, and it is illuminating to discuss these properties in some generality.
The exponential family of distributions over x, given parameters η, is deﬁned to
be the set of distributions of the form
p(x|η) = h(x)g(η) exp 
ηTu(x)
(2.194)
where x may be scalar or vector, and may be discrete or continuous. Here η are
called the natural parameters of the distribution, and u(x) is some function of x.
The function g(η) can be interpreted as the coefﬁcient that ensures that the distribu-
tion is normalized and therefore satisﬁes
g(η)

h(x) exp 
ηTu(x)
dx = 1
(2.195)
where the integration is replaced by summation if x is a discrete variable.
We begin by taking some examples of the distributions introduced earlier in
the chapter and showing that they are indeed members of the exponential family.
Consider ﬁrst the Bernoulli distribution
p(x|µ) = Bern(x|µ) = µx(1 −µ)1−x.
(2.196)
Expressing the right-hand side as the exponential of the logarithm, we have
p(x|µ)
=
exp {x ln µ + (1 −x) ln(1 −µ)}
=
(1 −µ) exp

ln

µ
1 −µ

x

.
(2.197)
Comparison with (2.194) allows us to identify
η = ln

µ
1 −µ

(2.198)


---
**Page 132**
114
2. PROBABILITY DISTRIBUTIONS
which we can solve for µ to give µ = σ(η), where
σ(η) =
1
1 + exp(−η)
(2.199)
is called the logistic sigmoid function. Thus we can write the Bernoulli distribution
using the standard representation (2.194) in the form
p(x|η) = σ(−η) exp(ηx)
(2.200)
where we have used 1 −σ(η) = σ(−η), which is easily proved from (2.199). Com-
parison with (2.194) shows that
u(x)
=
x
(2.201)
h(x)
=
1
(2.202)
g(η)
=
σ(−η).
(2.203)
Next consider the multinomial distribution that, for a single observation x, takes
the form
p(x|µ) =
M

k=1
µxk
k = exp
 M

k=1
xk ln µk

(2.204)
where x = (x1, . . . , xN)T. Again, we can write this in the standard representation
(2.194) so that
p(x|η) = exp(ηTx)
(2.205)
where ηk = ln µk, and we have deﬁned η = (η1, . . . , ηM)T. Again, comparing with
(2.194) we have
u(x)
=
x
(2.206)
h(x)
=
1
(2.207)
g(η)
=
1.
(2.208)
Note that the parameters ηk are not independent because the parameters µk are sub-
ject to the constraint
M

k=1
µk = 1
(2.209)
so that, given any M −1 of the parameters µk, the value of the remaining parameter
is ﬁxed. In some circumstances, it will be convenient to remove this constraint by
expressing the distribution in terms of only M −1 parameters. This can be achieved
by using the relationship (2.209) to eliminate µM by expressing it in terms of the
remaining {µk} where k = 1, . . . , M −1, thereby leaving M −1 parameters. Note
that these remaining parameters are still subject to the constraints
0 ⩽µk ⩽1,
M−1

k=1
µk ⩽1.
(2.210)


---
**Page 133**
2.4. The Exponential Family
115
Making use of the constraint (2.209), the multinomial distribution in this representa-
tion then becomes
exp
 M

k=1
xk ln µk

=
exp
M−1

k=1
xk ln µk +

1 −
M−1

k=1
xk

ln

1 −
M−1

k=1
µk

=
exp
M−1

k=1
xk ln

µk
1 −M−1
j=1 µj

+ ln

1 −
M−1

k=1
µk

.
(2.211)
We now identify
ln

µk
1 −
j µj

= ηk
(2.212)
which we can solve for µk by ﬁrst summing both sides over k and then rearranging
and back-substituting to give
µk =
exp(ηk)
1 + 
j exp(ηj).
(2.213)
This is called the softmax function, or the normalized exponential. In this represen-
tation, the multinomial distribution therefore takes the form
p(x|η) =

1 +
M−1

k=1
exp(ηk)
−1
exp(ηTx).
(2.214)
This is the standard form of the exponential family, with parameter vector η =
(η1, . . . , ηM−1)T in which
u(x)
=
x
(2.215)
h(x)
=
1
(2.216)
g(η)
=

1 +
M−1

k=1
exp(ηk)
−1
.
(2.217)
Finally, let us consider the Gaussian distribution. For the univariate Gaussian,
we have
p(x|µ, σ2)
=
1
(2πσ2)1/2 exp

−1
2σ2 (x −µ)2

(2.218)
=
1
(2πσ2)1/2 exp

−1
2σ2 x2 + µ
σ2 x −
1
2σ2 µ2

(2.219)


---
**Page 134**
116
2. PROBABILITY DISTRIBUTIONS
which, after some simple rearrangement, can be cast in the standard exponential
family form (2.194) with
Exercise 2.57
η
=

µ/σ2
−1/2σ2

(2.220)
u(x)
=

x
x2

(2.221)
h(x)
=
(2π)−1/2
(2.222)
g(η)
=
(−2η2)1/2 exp
 η2
1
4η2

.
(2.223)
2.4.1
Maximum likelihood and sufﬁcient statistics
Let us now consider the problem of estimating the parameter vector η in the gen-
eral exponential family distribution (2.194) using the technique of maximum likeli-
hood. Taking the gradient of both sides of (2.195) with respect to η, we have
∇g(η)

h(x) exp 
ηTu(x)
dx
+
g(η)

h(x) exp

ηTu(x)

u(x) dx = 0.
(2.224)
Rearranging, and making use again of (2.195) then gives
−
1
g(η)∇g(η) = g(η)

h(x) exp

ηTu(x)

u(x) dx = E[u(x)]
(2.225)
where we have used (2.194). We therefore obtain the result
−∇ln g(η) = E[u(x)].
(2.226)
Note that the covariance of u(x) can be expressed in terms of the second derivatives
of g(η), and similarly for higher order moments. Thus, provided we can normalize a
Exercise 2.58
distribution from the exponential family, we can always ﬁnd its moments by simple
differentiation.
Now consider a set of independent identically distributed data denoted by X =
{x1, . . . , xn}, for which the likelihood function is given by
p(X|η) =
 N

n=1
h(xn)

g(η)N exp

ηT
N

n=1
u(xn)

.
(2.227)
Setting the gradient of ln p(X|η) with respect to η to zero, we get the following
condition to be satisﬁed by the maximum likelihood estimator ηML
−∇ln g(ηML) = 1
N
N

n=1
u(xn)
(2.228)


---
**Page 135**
2.4. The Exponential Family
117
which can in principle be solved to obtain ηML. We see that the solution for the
maximum likelihood estimator depends on the data only through 
n u(xn), which
is therefore called the sufﬁcient statistic of the distribution (2.194). We do not need
to store the entire data set itself but only the value of the sufﬁcient statistic. For
the Bernoulli distribution, for example, the function u(x) is given just by x and
so we need only keep the sum of the data points {xn}, whereas for the Gaussian
u(x) = (x, x2)T, and so we should keep both the sum of {xn} and the sum of {x2
n}.
If we consider the limit N →∞, then the right-hand side of (2.228) becomes
E[u(x)], and so by comparing with (2.226) we see that in this limit ηML will equal
the true value η.
In fact, this sufﬁciency property holds also for Bayesian inference, although
we shall defer discussion of this until Chapter 8 when we have equipped ourselves
with the tools of graphical models and can thereby gain a deeper insight into these
important concepts.
2.4.2
Conjugate priors
We have already encountered the concept of a conjugate prior several times, for
example in the context of the Bernoulli distribution (for which the conjugate prior
is the beta distribution) or the Gaussian (where the conjugate prior for the mean is
a Gaussian, and the conjugate prior for the precision is the Wishart distribution). In
general, for a given probability distribution p(x|η), we can seek a prior p(η) that is
conjugate to the likelihood function, so that the posterior distribution has the same
functional form as the prior. For any member of the exponential family (2.194), there
exists a conjugate prior that can be written in the form
p(η|χ, ν) = f(χ, ν)g(η)ν exp

νηTχ

(2.229)
where f(χ, ν) is a normalization coefﬁcient, and g(η) is the same function as ap-
pears in (2.194). To see that this is indeed conjugate, let us multiply the prior (2.229)
by the likelihood function (2.227) to obtain the posterior distribution, up to a nor-
malization coefﬁcient, in the form
p(η|X, χ, ν) ∝g(η)ν+N exp

ηT
 N

n=1
u(xn) + νχ

.
(2.230)
This again takes the same functional form as the prior (2.229), conﬁrming conjugacy.
Furthermore, we see that the parameter ν can be interpreted as a effective number of
pseudo-observations in the prior, each of which has a value for the sufﬁcient statistic
u(x) given by χ.
2.4.3
Noninformative priors
In some applications of probabilistic inference, we may have prior knowledge
that can be conveniently expressed through the prior distribution. For example, if
the prior assigns zero probability to some value of variable, then the posterior dis-
tribution will necessarily also assign zero probability to that value, irrespective of


---
**Page 136**
118
2. PROBABILITY DISTRIBUTIONS
any subsequent observations of data. In many cases, however, we may have little
idea of what form the distribution should take. We may then seek a form of prior
distribution, called a noninformative prior, which is intended to have as little inﬂu-
ence on the posterior distribution as possible (Jeffries, 1946; Box and Tao, 1973;
Bernardo and Smith, 1994). This is sometimes referred to as ‘letting the data speak
for themselves’.
If we have a distribution p(x|λ) governed by a parameter λ, we might be tempted
to propose a prior distribution p(λ) = const as a suitable prior. If λ is a discrete
variable with K states, this simply amounts to setting the prior probability of each
state to 1/K. In the case of continuous parameters, however, there are two potential
difﬁculties with this approach. The ﬁrst is that, if the domain of λ is unbounded,
this prior distribution cannot be correctly normalized because the integral over λ
diverges. Such priors are called improper. In practice, improper priors can often
be used provided the corresponding posterior distribution is proper, i.e., that it can
be correctly normalized. For instance, if we put a uniform prior distribution over
the mean of a Gaussian, then the posterior distribution for the mean, once we have
observed at least one data point, will be proper.
A second difﬁculty arises from the transformation behaviour of a probability
density under a nonlinear change of variables, given by (1.27). If a function h(λ)
is constant, and we change variables to λ = η2, then h(η) = h(η2) will also be
constant. However, if we choose the density pλ(λ) to be constant, then the density
of η will be given, from (1.27), by
pη(η) = pλ(λ)

dλ
dη
 = pλ(η2)2η ∝η
(2.231)
and so the density over η will not be constant. This issue does not arise when we use
maximum likelihood, because the likelihood function p(x|λ) is a simple function of
λ and so we are free to use any convenient parameterization. If, however, we are to
choose a prior distribution that is constant, we must take care to use an appropriate
representation for the parameters.
Here we consider two simple examples of noninformative priors (Berger, 1985).
First of all, if a density takes the form
p(x|µ) = f(x −µ)
(2.232)
then the parameter µ is known as a location parameter. This family of densities
exhibits translation invariance because if we shift x by a constant to give x = x + c,
then
p(x|µ) = f(x −µ)
(2.233)
where we have deﬁned µ = µ + c. Thus the density takes the same form in the
new variable as in the original one, and so the density is independent of the choice
of origin. We would like to choose a prior distribution that reﬂects this translation
invariance property, and so we choose a prior that assigns equal probability mass to

