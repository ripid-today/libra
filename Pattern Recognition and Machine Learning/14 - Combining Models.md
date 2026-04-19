# 14 - Combining Models
*Pages 653-676 from Pattern Recognition and Machine Learning*

---
**Page 653**
Appendix B. Probability Distributions
In this appendix, we summarize the main properties of some of the most widely used
probability distributions, and for each distribution we list some key statistics such as
the expectation E[x], the variance (or covariance), the mode, and the entropy H[x].
All of these distributions are members of the exponential family and are widely used
as building blocks for more sophisticated probabilistic models.
Bernoulli
This is the distribution for a single binary variable x ∈{0, 1} representing, for
example, the result of ﬂipping a coin. It is governed by a single continuous parameter
µ ∈[0, 1] that represents the probability of x = 1.
Bern(x|µ)
=
µx(1 −µ)1−x
(B.1)
E[x]
=
µ
(B.2)
var[x]
=
µ(1 −µ)
(B.3)
mode[x]
=

1
if µ ⩾0.5,
0
otherwise
(B.4)
H[x]
=
−µ ln µ −(1 −µ) ln(1 −µ).
(B.5)
The Bernoulli is a special case of the binomial distribution for the case of a single
observation. Its conjugate prior for µ is the beta distribution.
685


---
**Page 654**
686
B. PROBABILITY DISTRIBUTIONS
Beta
This is a distribution over a continuous variable µ ∈[0, 1], which is often used to
represent the probability for some binary event. It is governed by two parameters a
and b that are constrained by a > 0 and b > 0 to ensure that the distribution can be
normalized.
Beta(µ|a, b)
=
Γ(a + b)
Γ(a)Γ(b)µa−1(1 −µ)b−1
(B.6)
E[µ]
=
a
a + b
(B.7)
var[µ]
=
ab
(a + b)2(a + b + 1)
(B.8)
mode[µ]
=
a −1
a + b −2.
(B.9)
The beta is the conjugate prior for the Bernoulli distribution, for which a and b can
be interpreted as the effective prior number of observations of x = 1 and x = 0,
respectively. Its density is ﬁnite if a ⩾1 and b ⩾1, otherwise there is a singularity
at µ = 0 and/or µ = 1. For a = b = 1, it reduces to a uniform distribution. The beta
distribution is a special case of the K-state Dirichlet distribution for K = 2.
Binomial
The binomial distribution gives the probability of observing m occurrences of x = 1
in a set of N samples from a Bernoulli distribution, where the probability of observ-
ing x = 1 is µ ∈[0, 1].
Bin(m|N, µ)
=
N
m

µm(1 −µ)N−m
(B.10)
E[m]
=
Nµ
(B.11)
var[m]
=
Nµ(1 −µ)
(B.12)
mode[m]
=
⌊(N + 1)µ⌋
(B.13)
where ⌊(N + 1)µ⌋denotes the largest integer that is less than or equal to (N + 1)µ,
and the quantity
N
m

=
N!
m!(N −m)!
(B.14)
denotes the number of ways of choosing m objects out of a total of N identical
objects. Here m!, pronounced ‘factorial m’, denotes the product m × (m −1) ×
. . . , ×2 × 1. The particular case of the binomial distribution for N = 1 is known as
the Bernoulli distribution, and for large N the binomial distribution is approximately
Gaussian. The conjugate prior for µ is the beta distribution.


---
**Page 655**
B. PROBABILITY DISTRIBUTIONS
687
Dirichlet
The Dirichlet is a multivariate distribution over K random variables 0 ⩽µk ⩽1,
where k = 1, . . ., K, subject to the constraints
0 ⩽µk ⩽1,
K

k=1
µk = 1.
(B.15)
Denoting µ = (µ1, . . . , µK)T and α = (α1, . . . , αK)T, we have
Dir(µ|α)
=
C(α)
K

k=1
µαk−1
k
(B.16)
E[µk]
=
αk
α
(B.17)
var[µk]
=
αk(α −αk)
α2(α + 1)
(B.18)
cov[µjµk]
=
−
αjαk
α2(α + 1)
(B.19)
mode[µk]
=
αk −1
α −K
(B.20)
E[ln µk]
=
ψ(αk) −ψ(α)
(B.21)
H[µ]
=
−
K

k=1
(αk −1) {ψ(αk) −ψ(α)} −ln C(α)
(B.22)
where
C(α) =
Γ(α)
Γ(α1) · · ·Γ(αK)
(B.23)
and
α =
K

k=1
αk.
(B.24)
Here
ψ(a) ≡d
da ln Γ(a)
(B.25)
is known as the digamma function (Abramowitz and Stegun, 1965). The parameters
αk are subject to the constraint αk > 0 in order to ensure that the distribution can be
normalized.
The Dirichlet forms the conjugate prior for the multinomial distribution and rep-
resents a generalization of the beta distribution. In this case, the parameters αk can
be interpreted as effective numbers of observations of the corresponding values of
the K-dimensional binary observation vector x. As with the beta distribution, the
Dirichlet has ﬁnite density everywhere provided αk ⩾1 for all k.


---
**Page 656**
688
B. PROBABILITY DISTRIBUTIONS
Gamma
The Gamma is a probability distribution over a positive random variable τ > 0
governed by parameters a and b that are subject to the constraints a > 0 and b > 0
to ensure that the distribution can be normalized.
Gam(τ|a, b)
=
1
Γ(a)baτ a−1e−bτ
(B.26)
E[τ]
=
a
b
(B.27)
var[τ]
=
a
b2
(B.28)
mode[τ]
=
a −1
b
for α ⩾1
(B.29)
E[ln τ]
=
ψ(a) −ln b
(B.30)
H[τ]
=
ln Γ(a) −(a −1)ψ(a) −ln b + a
(B.31)
where ψ(·) is the digamma function deﬁned by (B.25). The gamma distribution is
the conjugate prior for the precision (inverse variance) of a univariate Gaussian. For
a ⩾1 the density is everywhere ﬁnite, and the special case of a = 1 is known as the
exponential distribution.
Gaussian
The Gaussian is the most widely used distribution for continuous variables. It is also
known as the normal distribution. In the case of a single variable x ∈(−∞, ∞) it is
governed by two parameters, the mean µ ∈(−∞, ∞) and the variance σ2 > 0.
N(x|µ, σ2)
=
1
(2πσ2)1/2 exp

−1
2σ2 (x −µ)2

(B.32)
E[x]
=
µ
(B.33)
var[x]
=
σ2
(B.34)
mode[x]
=
µ
(B.35)
H[x]
=
1
2 ln σ2 + 1
2 (1 + ln(2π)) .
(B.36)
The inverse of the variance τ = 1/σ2 is called the precision, and the square root
of the variance σ is called the standard deviation. The conjugate prior for µ is the
Gaussian, and the conjugate prior for τ is the gamma distribution. If both µ and τ
are unknown, their joint conjugate prior is the Gaussian-gamma distribution.
For a D-dimensional vector x, the Gaussian is governed by a D-dimensional
mean vector µ and a D × D covariance matrix Σ that must be symmetric and


---
**Page 657**
B. PROBABILITY DISTRIBUTIONS
689
positive-deﬁnite.
N(x|µ, Σ)
=
1
(2π)D/2
1
|Σ|1/2 exp

−1
2(x −µ)TΣ−1(x −µ)

(B.37)
E[x]
=
µ
(B.38)
cov[x]
=
Σ
(B.39)
mode[x]
=
µ
(B.40)
H[x]
=
1
2 ln |Σ| + D
2 (1 + ln(2π)) .
(B.41)
The inverse of the covariance matrix Λ = Σ−1 is the precision matrix, which is also
symmetric and positive deﬁnite. Averages of random variables tend to a Gaussian, by
the central limit theorem, and the sum of two Gaussian variables is again Gaussian.
The Gaussian is the distribution that maximizes the entropy for a given variance
(or covariance). Any linear transformation of a Gaussian random variable is again
Gaussian. The marginal distribution of a multivariate Gaussian with respect to a
subset of the variables is itself Gaussian, and similarly the conditional distribution is
also Gaussian. The conjugate prior for µ is the Gaussian, the conjugate prior for Λ
is the Wishart, and the conjugate prior for (µ, Λ) is the Gaussian-Wishart.
If we have a marginal Gaussian distribution for x and a conditional Gaussian
distribution for y given x in the form
p(x)
=
N(x|µ, Λ−1)
(B.42)
p(y|x)
=
N(y|Ax + b, L−1)
(B.43)
then the marginal distribution of y, and the conditional distribution of x given y, are
given by
p(y)
=
N(y|Aµ + b, L−1 + AΛ−1AT)
(B.44)
p(x|y)
=
N(x|Σ{ATL(y −b) + Λµ}, Σ)
(B.45)
where
Σ = (Λ + ATLA)−1.
(B.46)
If we have a joint Gaussian distribution N(x|µ, Σ) with Λ ≡Σ−1 and we
deﬁne the following partitions
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
(B.47)
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
(B.48)
then the conditional distribution p(xa|xb) is given by
p(xa|xb)
=
N(x|µa|b, Λ−1
aa )
(B.49)
µa|b
=
µa −Λ−1
aa Λab(xb −µb)
(B.50)


---
**Page 658**
690
B. PROBABILITY DISTRIBUTIONS
and the marginal distribution p(xa) is given by
p(xa) = N(xa|µa, Σaa).
(B.51)
Gaussian-Gamma
This is the conjugate prior distribution for a univariate Gaussian N(x|µ, λ−1) in
which the mean µ and the precision λ are both unknown and is also called the
normal-gamma distribution. It comprises the product of a Gaussian distribution for
µ, whose precision is proportional to λ, and a gamma distribution over λ.
p(µ, λ|µ0, β, a, b) = N 
µ|µo, (βλ)−1
Gam(λ|a, b).
(B.52)
Gaussian-Wishart
This is the conjugate prior distribution for a multivariate Gaussian N(x|µ, Λ) in
which both the mean µ and the precision Λ are unknown, and is also called the
normal-Wishart distribution. It comprises the product of a Gaussian distribution for
µ, whose precision is proportional to Λ, and a Wishart distribution over Λ.
p(µ, Λ|µ0, β, W, ν) = N 
µ|µ0, (βΛ)−1
W(Λ|W, ν).
(B.53)
For the particular case of a scalar x, this is equivalent to the Gaussian-gamma distri-
bution.
Multinomial
If we generalize the Bernoulli distribution to an K-dimensional binary variable x
with components xk ∈{0, 1} such that 
k xk = 1, then we obtain the following
discrete distribution
p(x)
=
K

k=1
µxk
k
(B.54)
E[xk]
=
µk
(B.55)
var[xk]
=
µk(1 −µk)
(B.56)
cov[xjxk]
=
Ijkµk
(B.57)
H[x]
=
−
M

k=1
µk ln µk
(B.58)


---
**Page 659**
B. PROBABILITY DISTRIBUTIONS
691
where Ijk is the j, k element of the identity matrix. Because p(xk = 1) = µk, the
parameters must satisfy 0 ⩽µk ⩽1 and 
k µk = 1.
The multinomial distribution is a multivariate generalization of the binomial and
gives the distribution over counts mk for a K-state discrete variable to be in state k
given a total number of observations N.
Mult(m1, m2, . . . , mK|µ, N)
=

N
m1m2 . . . mM
 M

k=1
µmk
k
(B.59)
E[mk]
=
Nµk
(B.60)
var[mk]
=
Nµk(1 −µk)
(B.61)
cov[mjmk]
=
−Nµjµk
(B.62)
where µ = (µ1, . . . , µK)T, and the quantity

N
m1m2 . . . mK

=
N!
m1! . . . mK!
(B.63)
gives the number of ways of taking N identical objects and assigning mk of them to
bin k for k = 1, . . . , K. The value of µk gives the probability of the random variable
taking state k, and so these parameters are subject to the constraints 0 ⩽µk ⩽1
and 
k µk = 1. The conjugate prior distribution for the parameters {µk} is the
Dirichlet.
Normal
The normal distribution is simply another name for the Gaussian. In this book, we
use the term Gaussian throughout, although we retain the conventional use of the
symbol N to denote this distribution. For consistency, we shall refer to the normal-
gamma distribution as the Gaussian-gamma distribution, and similarly the normal-
Wishart is called the Gaussian-Wishart.
Student’s t
This distribution was published by William Gosset in 1908, but his employer, Gui-
ness Breweries, required him to publish under a pseudonym, so he chose ‘Student’.
In the univariate form, Student’s t-distribution is obtained by placing a conjugate
gamma prior over the precision of a univariate Gaussian distribution and then inte-
grating out the precision variable. It can therefore be viewed as an inﬁnite mixture


---
**Page 660**
692
B. PROBABILITY DISTRIBUTIONS
of Gaussians having the same mean but different variances.
St(x|µ, λ, ν)
=
Γ(ν/2 + 1/2)
Γ(ν/2)
 λ
πν
1/2 
1 + λ(x −µ)2
ν
−ν/2−1/2
(B.64)
E[x]
=
µ
for ν > 1
(B.65)
var[x]
=
1
λ
ν
ν −2
for ν > 2
(B.66)
mode[x]
=
µ.
(B.67)
Here ν > 0 is called the number of degrees of freedom of the distribution. The
particular case of ν = 1 is called the Cauchy distribution.
For a D-dimensional variable x, Student’s t-distribution corresponds to marginal-
izing the precision matrix of a multivariate Gaussian with respect to a conjugate
Wishart prior and takes the form
St(x|µ, Λ, ν)
=
Γ(ν/2 + D/2)
Γ(ν/2)
|Λ|1/2
(νπ)D/2

1 + ∆2
ν
−ν/2−D/2
(B.68)
E[x]
=
µ
for ν > 1
(B.69)
cov[x]
=
ν
ν −2Λ−1
for ν > 2
(B.70)
mode[x]
=
µ
(B.71)
where ∆2 is the squared Mahalanobis distance deﬁned by
∆2 = (x −µ)TΛ(x −µ).
(B.72)
In the limit ν →∞, the t-distribution reduces to a Gaussian with mean µ and pre-
cision Λ. Student’s t-distribution provides a generalization of the Gaussian whose
maximum likelihood parameter values are robust to outliers.
Uniform
This is a simple distribution for a continuous variable x deﬁned over a ﬁnite interval
x ∈[a, b] where b > a.
U(x|a, b)
=
1
b −a
(B.73)
E[x]
=
(b + a)
2
(B.74)
var[x]
=
(b −a)2
12
(B.75)
H[x]
=
ln(b −a).
(B.76)
If x has distribution U(x|0, 1), then a + (b −a)x will have distribution U(x|a, b).


---
**Page 661**
B. PROBABILITY DISTRIBUTIONS
693
Von Mises
The von Mises distribution, also known as the circular normal or the circular Gaus-
sian, is a univariate Gaussian-like periodic distribution for a variable θ ∈[0, 2π).
p(θ|θ0, m)
=
1
2πI0(m) exp {m cos(θ −θ0)}
(B.77)
where I0(m) is the zeroth-order Bessel function of the ﬁrst kind. The distribution
has period 2π so that p(θ + 2π) = p(θ) for all θ. Care must be taken in interpret-
ing this distribution because simple expectations will be dependent on the (arbitrary)
choice of origin for the variable θ. The parameter θ0 is analogous to the mean of a
univariate Gaussian, and the parameter m > 0, known as the concentration param-
eter, is analogous to the precision (inverse variance). For large m, the von Mises
distribution is approximately a Gaussian centred on θ0.
Wishart
The Wishart distribution is the conjugate prior for the precision matrix of a multi-
variate Gaussian.
W(Λ|W, ν) = B(W, ν)|Λ|(ν−D−1)/2 exp

−1
2Tr(W−1Λ)

(B.78)
where
B(W, ν)
≡
|W|−ν/2

2νD/2 πD(D−1)/4
D

i=1
Γ
ν + 1 −i
2
−1
(B.79)
E[Λ]
=
νW
(B.80)
E [ln |Λ|]
=
D

i=1
ψ
ν + 1 −i
2

+ D ln 2 + ln |W|
(B.81)
H[Λ]
=
−ln B(W, ν) −(ν −D −1)
2
E [ln |Λ|] + νD
2
(B.82)
where W is a D × D symmetric, positive deﬁnite matrix, and ψ(·) is the digamma
function deﬁned by (B.25). The parameter ν is called the number of degrees of
freedom of the distribution and is restricted to ν > D −1 to ensure that the Gamma
function in the normalization factor is well-deﬁned. In one dimension, the Wishart
reduces to the gamma distribution Gam(λ|a, b) given by (B.26) with parameters
a = ν/2 and b = 1/2W.


---
**Page 662**
Appendix C. Properties of Matrices
In this appendix, we gather together some useful properties and identities involving
matrices and determinants. This is not intended to be an introductory tutorial, and
it is assumed that the reader is already familiar with basic linear algebra. For some
results, we indicate how to prove them, whereas in more complex cases we leave
the interested reader to refer to standard textbooks on the subject. In all cases, we
assume that inverses exist and that matrix dimensions are such that the formulae
are correctly deﬁned. A comprehensive discussion of linear algebra can be found in
Golub and Van Loan (1996), and an extensive collection of matrix properties is given
by L¨utkepohl (1996). Matrix derivatives are discussed in Magnus and Neudecker
(1999).
Basic Matrix Identities
A matrix A has elements Aij where i indexes the rows, and j indexes the columns.
We use IN to denote the N × N identity matrix (also called the unit matrix), and
where there is no ambiguity over dimensionality we simply use I. The transpose
matrix AT has elements (AT)ij = Aji. From the deﬁnition of transpose, we have
(AB)T = BTAT
(C.1)
which can be veriﬁed by writing out the indices. The inverse of A, denoted A−1,
satisﬁes
AA−1 = A−1A = I.
(C.2)
Because ABB−1A−1 = I, we have
(AB)−1 = B−1A−1.
(C.3)
Also we have

AT−1 =

A−1T
(C.4)
695


---
**Page 663**
696
C. PROPERTIES OF MATRICES
which is easily proven by taking the transpose of (C.2) and applying (C.1).
A useful identity involving matrix inverses is the following
(P−1 + BTR−1B)−1BTR−1 = PBT(BPBT + R)−1.
(C.5)
which is easily veriﬁed by right multiplying both sides by (BPBT + R). Suppose
that P has dimensionality N × N while R has dimensionality M × M, so that B is
M × N. Then if M ≪N, it will be much cheaper to evaluate the right-hand side of
(C.5) than the left-hand side. A special case that sometimes arises is
(I + AB)−1A = A(I + BA)−1.
(C.6)
Another useful identity involving inverses is the following:
(A + BD−1C)−1 = A−1 −A−1B(D + CA−1B)−1CA−1
(C.7)
which is known as the Woodbury identity and which can be veriﬁed by multiplying
both sides by (A + BD−1C). This is useful, for instance, when A is large and
diagonal, and hence easy to invert, while B has many rows but few columns (and
conversely for C) so that the right-hand side is much cheaper to evaluate than the
left-hand side.
A set of vectors {a1, . . . , aN} is said to be linearly independent if the relation

n αnan = 0 holds only if all αn = 0. This implies that none of the vectors
can be expressed as a linear combination of the remainder. The rank of a matrix is
the maximum number of linearly independent rows (or equivalently the maximum
number of linearly independent columns).
Traces and Determinants
Trace and determinant apply to square matrices. The trace Tr(A) of a matrix A
is deﬁned as the sum of the elements on the leading diagonal. By writing out the
indices, we see that
Tr(AB) = Tr(BA).
(C.8)
By applying this formula multiple times to the product of three matrices, we see that
Tr(ABC) = Tr(CAB) = Tr(BCA)
(C.9)
which is known as the cyclic property of the trace operator and which clearly extends
to the product of any number of matrices. The determinant |A| of an N × N matrix
A is deﬁned by
|A| =

(±1)A1i1A2i2 · · · ANiN
(C.10)
in which the sum is taken over all products consisting of precisely one element from
each row and one element from each column, with a coefﬁcient +1 or −1 according


---
**Page 664**
C. PROPERTIES OF MATRICES
697
to whether the permutation i1i2 . . . iN is even or odd, respectively. Note that |I| = 1.
Thus, for a 2 × 2 matrix, the determinant takes the form
|A| =

a11
a12
a21
a22
 = a11a22 −a12a21.
(C.11)
The determinant of a product of two matrices is given by
|AB| = |A||B|
(C.12)
as can be shown from (C.10). Also, the determinant of an inverse matrix is given by
A−1 =
1
|A|
(C.13)
which can be shown by taking the determinant of (C.2) and applying (C.12).
If A and B are matrices of size N × M, then
IN + ABT =
IM + ATB
 .
(C.14)
A useful special case is
IN + abT = 1 + aTb
(C.15)
where a and b are N-dimensional column vectors.
Matrix Derivatives
Sometimes we need to consider derivatives of vectors and matrices with respect to
scalars. The derivative of a vector a with respect to a scalar x is itself a vector whose
components are given by
∂a
∂x

i
= ∂ai
∂x
(C.16)
with an analogous deﬁnition for the derivative of a matrix. Derivatives with respect
to vectors and matrices can also be deﬁned, for instance
∂x
∂a

i
= ∂x
∂ai
(C.17)
and similarly
 ∂a
∂b

ij
= ∂ai
∂bj
.
(C.18)
The following is easily proven by writing out the components
∂
∂x

xTa
= ∂
∂x

aTx
= a.
(C.19)


---
**Page 665**
698
C. PROPERTIES OF MATRICES
Similarly
∂
∂x (AB) = ∂A
∂x B + A∂B
∂x .
(C.20)
The derivative of the inverse of a matrix can be expressed as
∂
∂x

A−1
= −A−1 ∂A
∂x A−1
(C.21)
as can be shown by differentiating the equation A−1A = I using (C.20) and then
right multiplying by A−1. Also
∂
∂x ln |A| = Tr

A−1 ∂A
∂x

(C.22)
which we shall prove later. If we choose x to be one of the elements of A, we have
∂
∂Aij
Tr (AB) = Bji
(C.23)
as can be seen by writing out the matrices using index notation. We can write this
result more compactly in the form
∂
∂ATr (AB) = BT.
(C.24)
With this notation, we have the following properties
∂
∂ATr 
ATB

=
B
(C.25)
∂
∂ATr(A)
=
I
(C.26)
∂
∂ATr(ABAT)
=
A(B + BT)
(C.27)
which can again be proven by writing out the matrix indices. We also have
∂
∂A ln |A| = 
A−1T
(C.28)
which follows from (C.22) and (C.26).
Eigenvector Equation
For a square matrix A of size M × M, the eigenvector equation is deﬁned by
Aui = λiui
(C.29)


---
**Page 666**
C. PROPERTIES OF MATRICES
699
for i = 1, . . . , M, where ui is an eigenvector and λi is the corresponding eigenvalue.
This can be viewed as a set of M simultaneous homogeneous linear equations, and
the condition for a solution is that
|A −λiI| = 0
(C.30)
which is known as the characteristic equation. Because this is a polynomial of order
M in λi, it must have M solutions (though these need not all be distinct). The rank
of A is equal to the number of nonzero eigenvalues.
Of particular interest are symmetric matrices, which arise as covariance ma-
trices, kernel matrices, and Hessians. Symmetric matrices have the property that
Aij = Aji, or equivalently AT = A. The inverse of a symmetric matrix is also sym-
metric, as can be seen by taking the transpose of A−1A = I and using AA−1 = I
together with the symmetry of I.
In general, the eigenvalues of a matrix are complex numbers, but for symmetric
matrices the eigenvalues λi are real. This can be seen by ﬁrst left multiplying (C.29)
by (u⋆
i )T, where ⋆denotes the complex conjugate, to give
(u⋆
i )T Aui = λi (u⋆
i )T ui.
(C.31)
Next we take the complex conjugate of (C.29) and left multiply by uT
i to give
uT
i Au⋆
i = λ⋆
i uT
i u⋆
i .
(C.32)
where we have used A⋆= A because we consider only real matrices A. Taking
the transpose of the second of these equations, and using AT = A, we see that the
left-hand sides of the two equations are equal, and hence that λ⋆
i = λi and so λi
must be real.
The eigenvectors ui of a real symmetric matrix can be chosen to be orthonormal
(i.e., orthogonal and of unit length) so that
uT
i uj = Iij
(C.33)
where Iij are the elements of the identity matrix I. To show this, we ﬁrst left multiply
(C.29) by uT
j to give
uT
j Aui = λiuT
j ui
(C.34)
and hence, by exchange of indices, we have
uT
i Auj = λjuT
i uj.
(C.35)
We now take the transpose of the second equation and make use of the symmetry
property AT = A, and then subtract the two equations to give
(λi −λj) uT
i uj = 0.
(C.36)
Hence, for λi ̸= λj, we have uT
i uj = 0, and hence ui and uj are orthogonal. If the
two eigenvalues are equal, then any linear combination αui + βuj is also an eigen-
vector with the same eigenvalue, so we can select one linear combination arbitrarily,


---
**Page 667**
700
C. PROPERTIES OF MATRICES
and then choose the second to be orthogonal to the ﬁrst (it can be shown that the de-
generate eigenvectors are never linearly dependent). Hence the eigenvectors can be
chosen to be orthogonal, and by normalizing can be set to unit length. Because there
are M eigenvalues, the corresponding M orthogonal eigenvectors form a complete
set and so any M-dimensional vector can be expressed as a linear combination of
the eigenvectors.
We can take the eigenvectors ui to be the columns of an M × M matrix U,
which from orthonormality satisﬁes
UTU = I.
(C.37)
Such a matrix is said to be orthogonal. Interestingly, the rows of this matrix are also
orthogonal, so that UUT = I. To show this, note that (C.37) implies UTUU−1 =
U−1 = UT and so UU−1 = UUT = I. Using (C.12), it also follows that |U| = 1.
The eigenvector equation (C.29) can be expressed in terms of U in the form
AU = UΛ
(C.38)
where Λ is an M × M diagonal matrix whose diagonal elements are given by the
eigenvalues λi.
If we consider a column vector x that is transformed by an orthogonal matrix U
to give a new vector
x = Ux
(C.39)
then the length of the vector is preserved because
xTx = xTUTUx = xTx
(C.40)
and similarly the angle between any two such vectors is preserved because
xTy = xTUTUy = xTy.
(C.41)
Thus, multiplication by U can be interpreted as a rigid rotation of the coordinate
system.
From (C.38), it follows that
UTAU = Λ
(C.42)
and because Λ is a diagonal matrix, we say that the matrix A is diagonalized by the
matrix U. If we left multiply by U and right multiply by UT, we obtain
A = UΛUT
(C.43)
Taking the inverse of this equation, and using (C.3) together with U−1 = UT, we
have
A−1 = UΛ−1UT.
(C.44)


---
**Page 668**
C. PROPERTIES OF MATRICES
701
These last two equations can also be written in the form
A
=
M

i=1
λiuiuT
i
(C.45)
A−1
=
M

i=1
1
λi
uiuT
i .
(C.46)
If we take the determinant of (C.43), and use (C.12), we obtain
|A| =
M

i=1
λi.
(C.47)
Similarly, taking the trace of (C.43), and using the cyclic property (C.8) of the trace
operator together with UTU = I, we have
Tr(A) =
M

i=1
λi.
(C.48)
We leave it as an exercise for the reader to verify (C.22) by making use of the results
(C.33), (C.45), (C.46), and (C.47).
A matrix A is said to be positive deﬁnite, denoted by A ≻0, if wTAw > 0 for
all values of the vector w. Equivalently, a positive deﬁnite matrix has λi > 0 for all
of its eigenvalues (as can be seen by setting w to each of the eigenvectors in turn,
and by noting that an arbitrary vector can be expanded as a linear combination of the
eigenvectors). Note that positive deﬁnite is not the same as all the elements being
positive. For example, the matrix

1
2
3
4

(C.49)
has eigenvalues λ1 ≃5.37 and λ2 ≃−0.37. A matrix is said to be positive semidef-
inite if wTAw ⩾0 holds for all values of w, which is denoted A ⪰0, and is
equivalent to λi ⩾0.


---
**Page 669**
Appendix D. Calculus of Variations
We can think of a function y(x) as being an operator that, for any input value x,
returns an output value y. In the same way, we can deﬁne a functional F[y] to be
an operator that takes a function y(x) and returns an output value F. An example of
a functional is the length of a curve drawn in a two-dimensional plane in which the
path of the curve is deﬁned in terms of a function. In the context of machine learning,
a widely used functional is the entropy H[x] for a continuous variable x because, for
any choice of probability density function p(x), it returns a scalar value representing
the entropy of x under that density. Thus the entropy of p(x) could equally well have
been written as H[p].
A common problem in conventional calculus is to ﬁnd a value of x that max-
imizes (or minimizes) a function y(x). Similarly, in the calculus of variations we
seek a function y(x) that maximizes (or minimizes) a functional F[y]. That is, of all
possible functions y(x), we wish to ﬁnd the particular function for which the func-
tional F[y] is a maximum (or minimum). The calculus of variations can be used, for
instance, to show that the shortest path between two points is a straight line or that
the maximum entropy distribution is a Gaussian.
If we weren’t familiar with the rules of ordinary calculus, we could evaluate a
conventional derivative dy/ dx by making a small change ϵ to the variable x and
then expanding in powers of ϵ, so that
y(x + ϵ) = y(x) + dy
dxϵ + O(ϵ2)
(D.1)
and ﬁnally taking the limit ϵ →0. Similarly, for a function of several variables
y(x1, . . . , xD), the corresponding partial derivatives are deﬁned by
y(x1 + ϵ1, . . . , xD + ϵD) = y(x1, . . . , xD) +
D

i=1
∂y
∂xi
ϵi + O(ϵ2).
(D.2)
The analogous deﬁnition of a functional derivative arises when we consider how
much a functional F[y] changes when we make a small change ϵη(x) to the function
703


---
**Page 670**
704
D. CALCULUS OF VARIATIONS
Figure D.1
A functional derivative can be deﬁned by
considering how the value of a functional
F[y] changes when the function y(x) is
changed to y(x) + ϵη(x) where η(x) is an
arbitrary function of x.
y(x)
y(x) + ϵη(x)
x
y(x), where η(x) is an arbitrary function of x, as illustrated in Figure D.1. We denote
the functional derivative of E[f] with respect to f(x) by δF/δf(x), and deﬁne it by
the following relation:
F[y(x) + ϵη(x)] = F[y(x)] + ϵ

δF
δy(x)η(x) dx + O(ϵ2).
(D.3)
This can be seen as a natural extension of (D.2) in which F[y] now depends on a
continuous set of variables, namely the values of y at all points x. Requiring that the
functional be stationary with respect to small variations in the function y(x) gives

δE
δy(x)η(x) dx = 0.
(D.4)
Because this must hold for an arbitrary choice of η(x), it follows that the functional
derivative must vanish. To see this, imagine choosing a perturbation η(x) that is zero
everywhere except in the neighbourhood of a point x, in which case the functional
derivative must be zero at x = x. However, because this must be true for every
choice of x, the functional derivative must vanish for all values of x.
Consider a functional that is deﬁned by an integral over a function G(y, y′, x)
that depends on both y(x) and its derivative y′(x) as well as having a direct depen-
dence on x
F[y] =

G (y(x), y′(x), x) dx
(D.5)
where the value of y(x) is assumed to be ﬁxed at the boundary of the region of
integration (which might be at inﬁnity). If we now consider variations in the function
y(x), we obtain
F[y(x) + ϵη(x)] = F[y(x)] + ϵ
 ∂G
∂y η(x) + ∂G
∂y′ η′(x)

dx + O(ϵ2).
(D.6)
We now have to cast this in the form (D.3). To do so, we integrate the second term by
parts and make use of the fact that η(x) must vanish at the boundary of the integral
(because y(x) is ﬁxed at the boundary). This gives
F[y(x) + ϵη(x)] = F[y(x)] + ϵ
 ∂G
∂y −d
dx
∂G
∂y′

η(x) dx + O(ϵ2) (D.7)


---
**Page 671**
D. CALCULUS OF VARIATIONS
705
from which we can read off the functional derivative by comparison with (D.3).
Requiring that the functional derivative vanishes then gives
∂G
∂y −d
dx
∂G
∂y′

= 0
(D.8)
which are known as the Euler-Lagrange equations. For example, if
G = y(x)2 + (y′(x))
2
(D.9)
then the Euler-Lagrange equations take the form
y(x) −d2y
dx2 = 0.
(D.10)
This second order differential equation can be solved for y(x) by making use of the
boundary conditions on y(x).
Often, we consider functionals deﬁned by integrals whose integrands take the
form G(y, x) and that do not depend on the derivatives of y(x). In this case, station-
arity simply requires that ∂G/∂y(x) = 0 for all values of x.
If we are optimizing a functional with respect to a probability distribution, then
we need to maintain the normalization constraint on the probabilities. This is often
most conveniently done using a Lagrange multiplier, which then allows an uncon-
Appendix E
strained optimization to be performed.
The extension of the above results to a multidimensional variable x is straight-
forward. For a more comprehensive discussion of the calculus of variations, see
Sagan (1969).


---
**Page 672**
Appendix E. Lagrange Multipliers
Lagrange multipliers, also sometimes called undetermined multipliers, are used to
ﬁnd the stationary points of a function of several variables subject to one or more
constraints.
Consider the problem of ﬁnding the maximum of a function f(x1, x2) subject to
a constraint relating x1 and x2, which we write in the form
g(x1, x2) = 0.
(E.1)
One approach would be to solve the constraint equation (E.1) and thus express x2 as
a function of x1 in the form x2 = h(x1). This can then be substituted into f(x1, x2)
to give a function of x1 alone of the form f(x1, h(x1)). The maximum with respect
to x1 could then be found by differentiation in the usual way, to give the stationary
value x⋆
1, with the corresponding value of x2 given by x⋆
2 = h(x⋆
1).
One problem with this approach is that it may be difﬁcult to ﬁnd an analytic
solution of the constraint equation that allows x2 to be expressed as an explicit func-
tion of x1. Also, this approach treats x1 and x2 differently and so spoils the natural
symmetry between these variables.
A more elegant, and often simpler, approach is based on the introduction of a
parameter λ called a Lagrange multiplier. We shall motivate this technique from
a geometrical perspective. Consider a D-dimensional variable x with components
x1, . . . , xD. The constraint equation g(x) = 0 then represents a (D−1)-dimensional
surface in x-space as indicated in Figure E.1.
We ﬁrst note that at any point on the constraint surface the gradient ∇g(x) of
the constraint function will be orthogonal to the surface. To see this, consider a point
x that lies on the constraint surface, and consider a nearby point x + ϵ that also lies
on the surface. If we make a Taylor expansion around x, we have
g(x + ϵ) ≃g(x) + ϵT∇g(x).
(E.2)
Because both x and x+ϵ lie on the constraint surface, we have g(x) = g(x+ϵ) and
hence ϵT∇g(x) ≃0. In the limit ∥ϵ∥→0 we have ϵT∇g(x) = 0, and because ϵ is
707


---
**Page 673**
708
E. LAGRANGE MULTIPLIERS
Figure E.1
A geometrical picture of the technique of La-
grange multipliers in which we seek to maximize a
function f(x), subject to the constraint g(x) = 0.
If x is D dimensional, the constraint g(x) = 0 cor-
responds to a subspace of dimensionality D −1,
indicated by the red curve.
The problem can
be solved by optimizing the Lagrangian function
L(x, λ) = f(x) + λg(x).
∇f(x)
∇g(x)
xA
g(x) = 0
then parallel to the constraint surface g(x) = 0, we see that the vector ∇g is normal
to the surface.
Next we seek a point x⋆on the constraint surface such that f(x) is maximized.
Such a point must have the property that the vector ∇f(x) is also orthogonal to the
constraint surface, as illustrated in Figure E.1, because otherwise we could increase
the value of f(x) by moving a short distance along the constraint surface. Thus ∇f
and ∇g are parallel (or anti-parallel) vectors, and so there must exist a parameter λ
such that
∇f + λ∇g = 0
(E.3)
where λ ̸= 0 is known as a Lagrange multiplier. Note that λ can have either sign.
At this point, it is convenient to introduce the Lagrangian function deﬁned by
L(x, λ) ≡f(x) + λg(x).
(E.4)
The constrained stationarity condition (E.3) is obtained by setting ∇xL = 0. Fur-
thermore, the condition ∂L/∂λ = 0 leads to the constraint equation g(x) = 0.
Thus to ﬁnd the maximum of a function f(x) subject to the constraint g(x) = 0,
we deﬁne the Lagrangian function given by (E.4) and we then ﬁnd the stationary
point of L(x, λ) with respect to both x and λ. For a D-dimensional vector x, this
gives D +1 equations that determine both the stationary point x⋆and the value of λ.
If we are only interested in x⋆, then we can eliminate λ from the stationarity equa-
tions without needing to ﬁnd its value (hence the term ‘undetermined multiplier’).
As a simple example, suppose we wish to ﬁnd the stationary point of the function
f(x1, x2) = 1 −x2
1 −x2
2 subject to the constraint g(x1, x2) = x1 + x2 −1 = 0, as
illustrated in Figure E.2. The corresponding Lagrangian function is given by
L(x, λ) = 1 −x2
1 −x2
2 + λ(x1 + x2 −1).
(E.5)
The conditions for this Lagrangian to be stationary with respect to x1, x2, and λ give
the following coupled equations:
−2x1 + λ
=
0
(E.6)
−2x2 + λ
=
0
(E.7)
x1 + x2 −1
=
0.
(E.8)


---
**Page 674**
E. LAGRANGE MULTIPLIERS
709
Figure E.2
A simple example of the use of Lagrange multipli-
ers in which the aim is to maximize f(x1, x2) =
1 −x2
1 −x2
2 subject to the constraint g(x1, x2) = 0
where g(x1, x2) = x1 + x2 −1. The circles show
contours of the function f(x1, x2), and the diagonal
line shows the constraint surface g(x1, x2) = 0.
g(x1, x2) = 0
x1
x2
(x⋆
1, x⋆
2)
Solution of these equations then gives the stationary point as (x⋆
1, x⋆
2) = ( 1
2, 1
2), and
the corresponding value for the Lagrange multiplier is λ = 1.
So far, we have considered the problem of maximizing a function subject to an
equality constraint of the form g(x) = 0. We now consider the problem of maxi-
mizing f(x) subject to an inequality constraint of the form g(x) ⩾0, as illustrated
in Figure E.3.
There are now two kinds of solution possible, according to whether the con-
strained stationary point lies in the region where g(x) > 0, in which case the con-
straint is inactive, or whether it lies on the boundary g(x) = 0, in which case the
constraint is said to be active. In the former case, the function g(x) plays no role
and so the stationary condition is simply ∇f(x) = 0. This again corresponds to
a stationary point of the Lagrange function (E.4) but this time with λ = 0. The
latter case, where the solution lies on the boundary, is analogous to the equality con-
straint discussed previously and corresponds to a stationary point of the Lagrange
function (E.4) with λ ̸= 0. Now, however, the sign of the Lagrange multiplier is
crucial, because the function f(x) will only be at a maximum if its gradient is ori-
ented away from the region g(x) > 0, as illustrated in Figure E.3. We therefore have
∇f(x) = −λ∇g(x) for some value of λ > 0.
For either of these two cases, the product λg(x) = 0. Thus the solution to the
Figure E.3
Illustration of the problem of maximizing
f(x) subject to the inequality constraint
g(x) ⩾0.
∇f(x)
∇g(x)
xA
xB
g(x) = 0
g(x) > 0


---
**Page 675**
710
E. LAGRANGE MULTIPLIERS
problem of maximizing f(x) subject to g(x) ⩾0 is obtained by optimizing the
Lagrange function (E.4) with respect to x and λ subject to the conditions
g(x)
⩾
0
(E.9)
λ
⩾
0
(E.10)
λg(x)
=
0
(E.11)
These are known as the Karush-Kuhn-Tucker (KKT) conditions (Karush, 1939; Kuhn
and Tucker, 1951).
Note that if we wish to minimize (rather than maximize) the function f(x) sub-
ject to an inequality constraint g(x) ⩾0, then we minimize the Lagrangian function
L(x, λ) = f(x) −λg(x) with respect to x, again subject to λ ⩾0.
Finally, it is straightforward to extend the technique of Lagrange multipliers to
the case of multiple equality and inequality constraints. Suppose we wish to maxi-
mize f(x) subject to gj(x) = 0 for j = 1, . . . , J, and hk(x) ⩾0 for k = 1, . . . , K.
We then introduce Lagrange multipliers {λj} and {µk}, and then optimize the La-
grangian function given by
L(x, {λj}, {µk}) = f(x) +
J

j=1
λjgj(x) +
K

k=1
µkhk(x)
(E.12)
subject to µk ⩾0 and µkhk(x) = 0 for k = 1, . . . , K. Extensions to constrained
functional derivatives are similarly straightforward. For a more detailed discussion
Appendix D
of the technique of Lagrange multipliers, see Nocedal and Wright (1999).


---
**Page 676**
REFERENCES
711
References
Abramowitz, M. and I. A. Stegun (1965). Handbook
of Mathematical Functions. Dover.
Adler, S. L. (1981). Over-relaxation method for the
Monte Carlo evaluation of the partition func-
tion for multiquadratic actions. Physical Review
D 23, 2901–2904.
Ahn, J. H. and J. H. Oh (2003). A constrained EM
algorithm for principal component analysis. Neu-
ral Computation 15(1), 57–65.
Aizerman, M. A., E. M. Braverman, and L. I. Rozo-
noer (1964). The probability problem of pattern
recognition learning and the method of potential
functions. Automation and Remote Control 25,
1175–1190.
Akaike, H. (1974). A new look at statistical model
identiﬁcation. IEEE Transactions on Automatic
Control 19, 716–723.
Ali, S. M. and S. D. Silvey (1966). A general class
of coefﬁcients of divergence of one distribution
from another. Journal of the Royal Statistical So-
ciety, B 28(1), 131–142.
Allwein, E. L., R. E. Schapire, and Y. Singer (2000).
Reducing multiclass to binary: a unifying ap-
proach for margin classiﬁers. Journal of Machine
Learning Research 1, 113–141.
Amari, S. (1985). Differential-Geometrical Methods
in Statistics. Springer.
Amari, S., A. Cichocki, and H. H. Yang (1996). A
new learning algorithm for blind signal separa-
tion. In D. S. Touretzky, M. C. Mozer, and M. E.
Hasselmo (Eds.), Advances in Neural Informa-
tion Processing Systems, Volume 8, pp. 757–763.
MIT Press.
Amari, S. I. (1998). Natural gradient works efﬁ-
ciently in learning. Neural Computation
10,
251–276.
Anderson, J. A. and E. Rosenfeld (Eds.) (1988).
Neurocomputing: Foundations of Research. MIT
Press.
Anderson, T. W. (1963). Asymptotic theory for prin-
cipal component analysis. Annals of Mathemati-
cal Statistics 34, 122–148.
Andrieu, C., N. de Freitas, A. Doucet, and M. I. Jor-
dan (2003). An introduction to MCMC for ma-
chine learning. Machine Learning 50, 5–43.
Anthony, M. and N. Biggs (1992). An Introduction
to Computational Learning Theory. Cambridge
University Press.
Attias, H. (1999a). Independent factor analysis. Neu-
ral Computation 11(4), 803–851.
Attias, H. (1999b). Inferring parameters and struc-
ture of latent variable models by variational
Bayes. In K. B. Laskey and H. Prade (Eds.),

