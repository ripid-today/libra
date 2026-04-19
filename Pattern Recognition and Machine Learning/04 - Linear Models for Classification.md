# 04 - Linear Models for Classification
*Pages 179-224 from Pattern Recognition and Machine Learning*

---
**Page 179**
3.4. Bayesian Model Comparison
161
be simply y(x) = 1, from which we obtain (3.64). Note that the kernel function can
be negative as well as positive, so although it satisﬁes a summation constraint, the
corresponding predictions are not necessarily convex combinations of the training
set target variables.
Finally, we note that the equivalent kernel (3.62) satisﬁes an important property
shared by kernel functions in general, namely that it can be expressed in the form an
Chapter 6
inner product with respect to a vector ψ(x) of nonlinear functions, so that
k(x, z) = ψ(x)Tψ(z)
(3.65)
where ψ(x) = β1/2S1/2
N φ(x).
3.4. Bayesian Model Comparison
In Chapter 1, we highlighted the problem of over-ﬁtting as well as the use of cross-
validation as a technique for setting the values of regularization parameters or for
choosing between alternative models. Here we consider the problem of model se-
lection from a Bayesian perspective. In this section, our discussion will be very
general, and then in Section 3.5 we shall see how these ideas can be applied to the
determination of regularization parameters in linear regression.
As we shall see, the over-ﬁtting associated with maximum likelihood can be
avoided by marginalizing (summing or integrating) over the model parameters in-
stead of making point estimates of their values. Models can then be compared di-
rectly on the training data, without the need for a validation set. This allows all
available data to be used for training and avoids the multiple training runs for each
model associated with cross-validation. It also allows multiple complexity parame-
ters to be determined simultaneously as part of the training process. For example,
in Chapter 7 we shall introduce the relevance vector machine, which is a Bayesian
model having one complexity parameter for every training data point.
The Bayesian view of model comparison simply involves the use of probabilities
to represent uncertainty in the choice of model, along with a consistent application
of the sum and product rules of probability. Suppose we wish to compare a set of L
models {Mi} where i = 1, . . . , L. Here a model refers to a probability distribution
over the observed data D. In the case of the polynomial curve-ﬁtting problem, the
distribution is deﬁned over the set of target values t, while the set of input values X
is assumed to be known. Other types of model deﬁne a joint distributions over X
and t. We shall suppose that the data is generated from one of these models but we
Section 1.5.4
are uncertain which one. Our uncertainty is expressed through a prior probability
distribution p(Mi). Given a training set D, we then wish to evaluate the posterior
distribution
p(Mi|D) ∝p(Mi)p(D|Mi).
(3.66)
The prior allows us to express a preference for different models. Let us simply
assume that all models are given equal prior probability. The interesting term is
the model evidence p(D|Mi) which expresses the preference shown by the data for


---
**Page 180**
162
3. LINEAR MODELS FOR REGRESSION
different models, and we shall examine this term in more detail shortly. The model
evidence is sometimes also called the marginal likelihood because it can be viewed
as a likelihood function over the space of models, in which the parameters have been
marginalized out. The ratio of model evidences p(D|Mi)/p(D|Mj) for two models
is known as a Bayes factor (Kass and Raftery, 1995).
Once we know the posterior distribution over models, the predictive distribution
is given, from the sum and product rules, by
p(t|x, D) =
L

i=1
p(t|x, Mi, D)p(Mi|D).
(3.67)
This is an example of a mixture distribution in which the overall predictive distribu-
tion is obtained by averaging the predictive distributions p(t|x, Mi, D) of individual
models, weighted by the posterior probabilities p(Mi|D) of those models. For in-
stance, if we have two models that are a-posteriori equally likely and one predicts
a narrow distribution around t = a while the other predicts a narrow distribution
around t = b, the overall predictive distribution will be a bimodal distribution with
modes at t = a and t = b, not a single model at t = (a + b)/2.
A simple approximation to model averaging is to use the single most probable
model alone to make predictions. This is known as model selection.
For a model governed by a set of parameters w, the model evidence is given,
from the sum and product rules of probability, by
p(D|Mi) =

p(D|w, Mi)p(w|Mi) dw.
(3.68)
From a sampling perspective, the marginal likelihood can be viewed as the proba-
Chapter 11
bility of generating the data set D from a model whose parameters are sampled at
random from the prior. It is also interesting to note that the evidence is precisely the
normalizing term that appears in the denominator in Bayes’ theorem when evaluating
the posterior distribution over parameters because
p(w|D, Mi) = p(D|w, Mi)p(w|Mi)
p(D|Mi)
.
(3.69)
We can obtain some insight into the model evidence by making a simple approx-
imation to the integral over parameters. Consider ﬁrst the case of a model having a
single parameter w. The posterior distribution over parameters is proportional to
p(D|w)p(w), where we omit the dependence on the model Mi to keep the notation
uncluttered. If we assume that the posterior distribution is sharply peaked around the
most probable value wMAP, with width ∆wposterior, then we can approximate the in-
tegral by the value of the integrand at its maximum times the width of the peak. If we
further assume that the prior is ﬂat with width ∆wprior so that p(w) = 1/∆wprior,
then we have
p(D) =

p(D|w)p(w) dw ≃p(D|wMAP)∆wposterior
∆wprior
(3.70)


---
**Page 181**
3.4. Bayesian Model Comparison
163
Figure 3.12
We can obtain a rough approximation to
the model evidence if we assume that
the posterior distribution over parame-
ters is sharply peaked around its mode
wMAP.
∆wposterior
∆wprior
wMAP
w
and so taking logs we obtain
ln p(D) ≃ln p(D|wMAP) + ln
∆wposterior
∆wprior

.
(3.71)
This approximation is illustrated in Figure 3.12. The ﬁrst term represents the ﬁt to
the data given by the most probable parameter values, and for a ﬂat prior this would
correspond to the log likelihood. The second term penalizes the model according to
its complexity. Because ∆wposterior < ∆wprior this term is negative, and it increases
in magnitude as the ratio ∆wposterior/∆wprior gets smaller. Thus, if parameters are
ﬁnely tuned to the data in the posterior distribution, then the penalty term is large.
For a model having a set of M parameters, we can make a similar approximation
for each parameter in turn. Assuming that all parameters have the same ratio of
∆wposterior/∆wprior, we obtain
ln p(D) ≃ln p(D|wMAP) + M ln
∆wposterior
∆wprior

.
(3.72)
Thus, in this very simple approximation, the size of the complexity penalty increases
linearly with the number M of adaptive parameters in the model. As we increase
the complexity of the model, the ﬁrst term will typically decrease, because a more
complex model is better able to ﬁt the data, whereas the second term will increase
due to the dependence on M. The optimal model complexity, as determined by
the maximum evidence, will be given by a trade-off between these two competing
terms. We shall later develop a more reﬁned version of this approximation, based on
a Gaussian approximation to the posterior distribution.
Section 4.4.1
We can gain further insight into Bayesian model comparison and understand
how the marginal likelihood can favour models of intermediate complexity by con-
sidering Figure 3.13. Here the horizontal axis is a one-dimensional representation
of the space of possible data sets, so that each point on this axis corresponds to a
speciﬁc data set. We now consider three models M1, M2 and M3 of successively
increasing complexity. Imagine running these models generatively to produce exam-
ple data sets, and then looking at the distribution of data sets that result. Any given


---
**Page 182**
164
3. LINEAR MODELS FOR REGRESSION
Figure 3.13
Schematic illustration of the
distribution of data sets for
three models of different com-
plexity, in which M1 is the
simplest and M3 is the most
complex.
Note that the dis-
tributions are normalized.
In
this example, for the partic-
ular observed data set D0,
the model M2 with intermedi-
ate complexity has the largest
evidence.
p(D)
D
D0
M1
M2
M3
model can generate a variety of different data sets since the parameters are governed
by a prior probability distribution, and for any choice of the parameters there may
be random noise on the target variables. To generate a particular data set from a spe-
ciﬁc model, we ﬁrst choose the values of the parameters from their prior distribution
p(w), and then for these parameter values we sample the data from p(D|w). A sim-
ple model (for example, based on a ﬁrst order polynomial) has little variability and
so will generate data sets that are fairly similar to each other. Its distribution p(D)
is therefore conﬁned to a relatively small region of the horizontal axis. By contrast,
a complex model (such as a ninth order polynomial) can generate a great variety of
different data sets, and so its distribution p(D) is spread over a large region of the
space of data sets. Because the distributions p(D|Mi) are normalized, we see that
the particular data set D0 can have the highest value of the evidence for the model
of intermediate complexity. Essentially, the simpler model cannot ﬁt the data well,
whereas the more complex model spreads its predictive probability over too broad a
range of data sets and so assigns relatively small probability to any one of them.
Implicit in the Bayesian model comparison framework is the assumption that
the true distribution from which the data are generated is contained within the set of
models under consideration. Provided this is so, we can show that Bayesian model
comparison will on average favour the correct model. To see this, consider two
models M1 and M2 in which the truth corresponds to M1. For a given ﬁnite data
set, it is possible for the Bayes factor to be larger for the incorrect model. However, if
we average the Bayes factor over the distribution of data sets, we obtain the expected
Bayes factor in the form

p(D|M1) ln p(D|M1)
p(D|M2) dD
(3.73)
where the average has been taken with respect to the true distribution of the data.
This quantity is an example of the Kullback-Leibler divergence and satisﬁes the prop-
Section 1.6.1
erty of always being positive unless the two distributions are equal in which case it
is zero. Thus on average the Bayes factor will always favour the correct model.
We have seen that the Bayesian framework avoids the problem of over-ﬁtting
and allows models to be compared on the basis of the training data alone. However,


---
**Page 183**
3.5. The Evidence Approximation
165
a Bayesian approach, like any approach to pattern recognition, needs to make as-
sumptions about the form of the model, and if these are invalid then the results can
be misleading. In particular, we see from Figure 3.12 that the model evidence can
be sensitive to many aspects of the prior, such as the behaviour in the tails. Indeed,
the evidence is not deﬁned if the prior is improper, as can be seen by noting that
an improper prior has an arbitrary scaling factor (in other words, the normalization
coefﬁcient is not deﬁned because the distribution cannot be normalized). If we con-
sider a proper prior and then take a suitable limit in order to obtain an improper prior
(for example, a Gaussian prior in which we take the limit of inﬁnite variance) then
the evidence will go to zero, as can be seen from (3.70) and Figure 3.12. It may,
however, be possible to consider the evidence ratio between two models ﬁrst and
then take a limit to obtain a meaningful answer.
In a practical application, therefore, it will be wise to keep aside an independent
test set of data on which to evaluate the overall performance of the ﬁnal system.
3.5. The Evidence Approximation
In a fully Bayesian treatment of the linear basis function model, we would intro-
duce prior distributions over the hyperparameters α and β and make predictions by
marginalizing with respect to these hyperparameters as well as with respect to the
parameters w. However, although we can integrate analytically over either w or
over the hyperparameters, the complete marginalization over all of these variables
is analytically intractable. Here we discuss an approximation in which we set the
hyperparameters to speciﬁc values determined by maximizing the marginal likeli-
hood function obtained by ﬁrst integrating over the parameters w. This framework
is known in the statistics literature as empirical Bayes (Bernardo and Smith, 1994;
Gelman et al., 2004), or type 2 maximum likelihood (Berger, 1985), or generalized
maximum likelihood (Wahba, 1975), and in the machine learning literature is also
called the evidence approximation (Gull, 1989; MacKay, 1992a).
If we introduce hyperpriors over α and β, the predictive distribution is obtained
by marginalizing over w, α and β so that
p(t|t) =

p(t|w, β)p(w|t, α, β)p(α, β|t) dw dα dβ
(3.74)
where p(t|w, β) is given by (3.8) and p(w|t, α, β) is given by (3.49) with mN and
SN deﬁned by (3.53) and (3.54) respectively. Here we have omitted the dependence
on the input variable x to keep the notation uncluttered. If the posterior distribution
p(α, β|t) is sharply peaked around values α and β, then the predictive distribution is
obtained simply by marginalizing over w in which α and β are ﬁxed to the values α
and β, so that
p(t|t) ≃p(t|t, α, β) =

p(t|w, β)p(w|t, α, β) dw.
(3.75)


---
**Page 184**
166
3. LINEAR MODELS FOR REGRESSION
From Bayes’ theorem, the posterior distribution for α and β is given by
p(α, β|t) ∝p(t|α, β)p(α, β).
(3.76)
If the prior is relatively ﬂat, then in the evidence framework the values of α and
β are obtained by maximizing the marginal likelihood function p(t|α, β). We shall
proceed by evaluating the marginal likelihood for the linear basis function model and
then ﬁnding its maxima. This will allow us to determine values for these hyperpa-
rameters from the training data alone, without recourse to cross-validation. Recall
that the ratio α/β is analogous to a regularization parameter.
As an aside it is worth noting that, if we deﬁne conjugate (Gamma) prior distri-
butions over α and β, then the marginalization over these hyperparameters in (3.74)
can be performed analytically to give a Student’s t-distribution over w (see Sec-
tion 2.3.7). Although the resulting integral over w is no longer analytically tractable,
it might be thought that approximating this integral, for example using the Laplace
approximation discussed (Section 4.4) which is based on a local Gaussian approxi-
mation centred on the mode of the posterior distribution, might provide a practical
alternative to the evidence framework (Buntine and Weigend, 1991). However, the
integrand as a function of w typically has a strongly skewed mode so that the Laplace
approximation fails to capture the bulk of the probability mass, leading to poorer re-
sults than those obtained by maximizing the evidence (MacKay, 1999).
Returning to the evidence framework, we note that there are two approaches that
we can take to the maximization of the log evidence. We can evaluate the evidence
function analytically and then set its derivative equal to zero to obtain re-estimation
equations for α and β, which we shall do in Section 3.5.2. Alternatively we use a
technique called the expectation maximization (EM) algorithm, which will be dis-
cussed in Section 9.3.4 where we shall also show that these two approaches converge
to the same solution.
3.5.1
Evaluation of the evidence function
The marginal likelihood function p(t|α, β) is obtained by integrating over the
weight parameters w, so that
p(t|α, β) =

p(t|w, β)p(w|α) dw.
(3.77)
One way to evaluate this integral is to make use once again of the result (2.115)
for the conditional distribution in a linear-Gaussian model. Here we shall evaluate
Exercise 3.16
the integral instead by completing the square in the exponent and making use of the
standard form for the normalization coefﬁcient of a Gaussian.
From (3.11), (3.12), and (3.52), we can write the evidence function in the form
Exercise 3.17
p(t|α, β) =
 β
2π
N/2  α
2π
M/2 
exp {−E(w)} dw
(3.78)


---
**Page 185**
3.5. The Evidence Approximation
167
where M is the dimensionality of w, and we have deﬁned
E(w)
=
βED(w) + αEW (w)
=
β
2 ∥t −Φw∥2 + α
2 wTw.
(3.79)
We recognize (3.79) as being equal, up to a constant of proportionality, to the reg-
ularized sum-of-squares error function (3.27). We now complete the square over w
Exercise 3.18
giving
E(w) = E(mN) + 1
2(w −mN)TA(w −mN)
(3.80)
where we have introduced
A = αI + βΦTΦ
(3.81)
together with
E(mN) = β
2 ∥t −ΦmN∥2 + α
2 mT
NmN.
(3.82)
Note that A corresponds to the matrix of second derivatives of the error function
A = ∇∇E(w)
(3.83)
and is known as the Hessian matrix. Here we have also deﬁned mN given by
mN = βA−1ΦTt.
(3.84)
Using (3.54), we see that A = S−1
N , and hence (3.84) is equivalent to the previous
deﬁnition (3.53), and therefore represents the mean of the posterior distribution.
The integral over w can now be evaluated simply by appealing to the standard
result for the normalization coefﬁcient of a multivariate Gaussian, giving
Exercise 3.19

exp {−E(w)} dw
=
exp{−E(mN)}

exp

−1
2(w −mN)TA(w −mN)

dw
=
exp{−E(mN)}(2π)M/2|A|−1/2.
(3.85)
Using (3.78) we can then write the log of the marginal likelihood in the form
ln p(t|α, β) = M
2 ln α + N
2 ln β −E(mN) −1
2 ln |A| −N
2 ln(2π)
(3.86)
which is the required expression for the evidence function.
Returning to the polynomial regression problem, we can plot the model evidence
against the order of the polynomial, as shown in Figure 3.14. Here we have assumed
a prior of the form (1.65) with the parameter α ﬁxed at α = 5 × 10−3. The form
of this plot is very instructive. Referring back to Figure 1.4, we see that the M = 0
polynomial has very poor ﬁt to the data and consequently gives a relatively low value


---
**Page 186**
168
3. LINEAR MODELS FOR REGRESSION
Figure 3.14
Plot of the model evidence versus
the order M, for the polynomial re-
gression model, showing that the
evidence favours the model with
M = 3.
M
0
2
4
6
8
−26
−24
−22
−20
−18
for the evidence. Going to the M = 1 polynomial greatly improves the data ﬁt, and
hence the evidence is signiﬁcantly higher. However, in going to M = 2, the data
ﬁt is improved only very marginally, due to the fact that the underlying sinusoidal
function from which the data is generated is an odd function and so has no even terms
in a polynomial expansion. Indeed, Figure 1.5 shows that the residual data error is
reduced only slightly in going from M = 1 to M = 2. Because this richer model
suffers a greater complexity penalty, the evidence actually falls in going from M = 1
to M = 2. When we go to M = 3 we obtain a signiﬁcant further improvement in
data ﬁt, as seen in Figure 1.4, and so the evidence is increased again, giving the
highest overall evidence for any of the polynomials. Further increases in the value
of M produce only small improvements in the ﬁt to the data but suffer increasing
complexity penalty, leading overall to a decrease in the evidence values. Looking
again at Figure 1.5, we see that the generalization error is roughly constant between
M = 3 and M = 8, and it would be difﬁcult to choose between these models on
the basis of this plot alone. The evidence values, however, show a clear preference
for M = 3, since this is the simplest model which gives a good explanation for the
observed data.
3.5.2
Maximizing the evidence function
Let us ﬁrst consider the maximization of p(t|α, β) with respect to α. This can
be done by ﬁrst deﬁning the following eigenvector equation

βΦTΦ

ui = λiui.
(3.87)
From (3.81), it then follows that A has eigenvalues α+λi. Now consider the deriva-
tive of the term involving ln |A| in (3.86) with respect to α. We have
d
dα ln |A| = d
dα ln

i
(λi + α) = d
dα

i
ln(λi + α) =

i
1
λi + α.
(3.88)
Thus the stationary points of (3.86) with respect to α satisfy
0 = M
2α −1
2mT
NmN −1
2

i
1
λi + α.
(3.89)


---
**Page 187**
3.5. The Evidence Approximation
169
Multiplying through by 2α and rearranging, we obtain
αmT
NmN = M −α

i
1
λi + α = γ.
(3.90)
Since there are M terms in the sum over i, the quantity γ can be written
γ =

i
λi
α + λi
.
(3.91)
The interpretation of the quantity γ will be discussed shortly. From (3.90) we see
that the value of α that maximizes the marginal likelihood satisﬁes
Exercise 3.20
α =
γ
mT
NmN
.
(3.92)
Note that this is an implicit solution for α not only because γ depends on α, but also
because the mode mN of the posterior distribution itself depends on the choice of
α. We therefore adopt an iterative procedure in which we make an initial choice for
α and use this to ﬁnd mN, which is given by (3.53), and also to evaluate γ, which
is given by (3.91). These values are then used to re-estimate α using (3.92), and the
process repeated until convergence. Note that because the matrix ΦTΦ is ﬁxed, we
can compute its eigenvalues once at the start and then simply multiply these by β to
obtain the λi.
It should be emphasized that the value of α has been determined purely by look-
ing at the training data. In contrast to maximum likelihood methods, no independent
data set is required in order to optimize the model complexity.
We can similarly maximize the log marginal likelihood (3.86) with respect to β.
To do this, we note that the eigenvalues λi deﬁned by (3.87) are proportional to β,
and hence dλi/dβ = λi/β giving
d
dβ ln |A| = d
dβ

i
ln(λi + α) = 1
β

i
λi
λi + α = γ
β .
(3.93)
The stationary point of the marginal likelihood therefore satisﬁes
0 = N
2β −1
2
N

n=1

tn −mT
Nφ(xn)2 −γ
2β
(3.94)
and rearranging we obtain
Exercise 3.22
1
β =
1
N −γ
N

n=1

tn −mT
Nφ(xn)2 .
(3.95)
Again, this is an implicit solution for β and can be solved by choosing an initial
value for β and then using this to calculate mN and γ and then re-estimate β using
(3.95), repeating until convergence. If both α and β are to be determined from the
data, then their values can be re-estimated together after each update of γ.


---
**Page 188**
170
3. LINEAR MODELS FOR REGRESSION
Figure 3.15
Contours of the likelihood function (red)
and the prior (green) in which the axes in parameter
space have been rotated to align with the eigenvectors
ui of the Hessian. For α = 0, the mode of the poste-
rior is given by the maximum likelihood solution wML,
whereas for nonzero α the mode is at wMAP = mN. In
the direction w1 the eigenvalue λ1, deﬁned by (3.87), is
small compared with α and so the quantity λ1/(λ1 + α)
is close to zero, and the corresponding MAP value of
w1 is also close to zero. By contrast, in the direction w2
the eigenvalue λ2 is large compared with α and so the
quantity λ2/(λ2 +α) is close to unity, and the MAP value
of w2 is close to its maximum likelihood value.
u1
u2
w1
w2
wMAP
wML
3.5.3
Effective number of parameters
The result (3.92) has an elegant interpretation (MacKay, 1992a), which provides
insight into the Bayesian solution for α. To see this, consider the contours of the like-
lihood function and the prior as illustrated in Figure 3.15. Here we have implicitly
transformed to a rotated set of axes in parameter space aligned with the eigenvec-
tors ui deﬁned in (3.87). Contours of the likelihood function are then axis-aligned
ellipses. The eigenvalues λi measure the curvature of the likelihood function, and
so in Figure 3.15 the eigenvalue λ1 is small compared with λ2 (because a smaller
curvature corresponds to a greater elongation of the contours of the likelihood func-
tion). Because βΦTΦ is a positive deﬁnite matrix, it will have positive eigenvalues,
and so the ratio λi/(λi + α) will lie between 0 and 1. Consequently, the quantity γ
deﬁned by (3.91) will lie in the range 0 ⩽γ ⩽M. For directions in which λi ≫α,
the corresponding parameter wi will be close to its maximum likelihood value, and
the ratio λi/(λi + α) will be close to 1. Such parameters are called well determined
because their values are tightly constrained by the data. Conversely, for directions
in which λi ≪α, the corresponding parameters wi will be close to zero, as will the
ratios λi/(λi +α). These are directions in which the likelihood function is relatively
insensitive to the parameter value and so the parameter has been set to a small value
by the prior. The quantity γ deﬁned by (3.91) therefore measures the effective total
number of well determined parameters.
We can obtain some insight into the result (3.95) for re-estimating β by com-
paring it with the corresponding maximum likelihood result given by (3.21). Both
of these formulae express the variance (the inverse precision) as an average of the
squared differences between the targets and the model predictions. However, they
differ in that the number of data points N in the denominator of the maximum like-
lihood result is replaced by N −γ in the Bayesian result. We recall from (1.56) that
the maximum likelihood estimate of the variance for a Gaussian distribution over a


---
**Page 189**
3.5. The Evidence Approximation
171
single variable x is given by
σ2
ML = 1
N
N

n=1
(xn −µML)2
(3.96)
and that this estimate is biased because the maximum likelihood solution µML for
the mean has ﬁtted some of the noise on the data. In effect, this has used up one
degree of freedom in the model. The corresponding unbiased estimate is given by
(1.59) and takes the form
σ2
MAP =
1
N −1
N

n=1
(xn −µML)2.
(3.97)
We shall see in Section 10.1.3 that this result can be obtained from a Bayesian treat-
ment in which we marginalize over the unknown mean. The factor of N −1 in the
denominator of the Bayesian result takes account of the fact that one degree of free-
dom has been used in ﬁtting the mean and removes the bias of maximum likelihood.
Now consider the corresponding results for the linear regression model. The mean
of the target distribution is now given by the function wTφ(x), which contains M
parameters. However, not all of these parameters are tuned to the data. The effective
number of parameters that are determined by the data is γ, with the remaining M −γ
parameters set to small values by the prior. This is reﬂected in the Bayesian result
for the variance that has a factor N −γ in the denominator, thereby correcting for
the bias of the maximum likelihood result.
We can illustrate the evidence framework for setting hyperparameters using the
sinusoidal synthetic data set from Section 1.1, together with the Gaussian basis func-
tion model comprising 9 basis functions, so that the total number of parameters in
the model is given by M = 10 including the bias. Here, for simplicity of illustra-
tion, we have set β to its true value of 11.1 and then used the evidence framework to
determine α, as shown in Figure 3.16.
We can also see how the parameter α controls the magnitude of the parameters
{wi}, by plotting the individual parameters versus the effective number γ of param-
eters, as shown in Figure 3.17.
If we consider the limit N ≫M in which the number of data points is large in
relation to the number of parameters, then from (3.87) all of the parameters will be
well determined by the data because ΦTΦ involves an implicit sum over data points,
and so the eigenvalues λi increase with the size of the data set. In this case, γ = M,
and the re-estimation equations for α and β become
α
=
M
2EW (mN)
(3.98)
β
=
N
2ED(mN)
(3.99)
where EW and ED are deﬁned by (3.25) and (3.26), respectively. These results
can be used as an easy-to-compute approximation to the full evidence re-estimation


---
**Page 190**
172
3. LINEAR MODELS FOR REGRESSION
ln α
−5
0
5
ln α
−5
0
5
Figure 3.16
The left plot shows γ (red curve) and 2αEW (mN) (blue curve) versus ln α for the sinusoidal
synthetic data set. It is the intersection of these two curves that deﬁnes the optimum value for α given by the
evidence procedure. The right plot shows the corresponding graph of log evidence ln p(t|α, β) versus ln α (red
curve) showing that the peak coincides with the crossing point of the curves in the left plot. Also shown is the
test set error (blue curve) showing that the evidence maximum occurs close to the point of best generalization.
formulae, because they do not require evaluation of the eigenvalue spectrum of the
Hessian.
Figure 3.17
Plot of the 10 parameters wi
from the Gaussian basis function
model versus the effective num-
ber of parameters γ, in which the
hyperparameter α is varied in the
range 0 ⩽α ⩽∞causing γ to
vary in the range 0 ⩽γ ⩽M.
9
7
1
3
6
2
5
4
8
0
γ
wi
0
2
4
6
8
10
−2
−1
0
1
2
3.6. Limitations of Fixed Basis Functions
Throughout this chapter, we have focussed on models comprising a linear combina-
tion of ﬁxed, nonlinear basis functions. We have seen that the assumption of linearity
in the parameters led to a range of useful properties including closed-form solutions
to the least-squares problem, as well as a tractable Bayesian treatment. Furthermore,
for a suitable choice of basis functions, we can model arbitrary nonlinearities in the


---
**Page 191**
Exercises
173
mapping from input variables to targets. In the next chapter, we shall study an anal-
ogous class of models for classiﬁcation.
It might appear, therefore, that such linear models constitute a general purpose
framework for solving problems in pattern recognition. Unfortunately, there are
some signiﬁcant shortcomings with linear models, which will cause us to turn in
later chapters to more complex models such as support vector machines and neural
networks.
The difﬁculty stems from the assumption that the basis functions φj(x) are ﬁxed
before the training data set is observed and is a manifestation of the curse of dimen-
sionality discussed in Section 1.4. As a consequence, the number of basis functions
needs to grow rapidly, often exponentially, with the dimensionality D of the input
space.
Fortunately, there are two properties of real data sets that we can exploit to help
alleviate this problem. First of all, the data vectors {xn} typically lie close to a non-
linear manifold whose intrinsic dimensionality is smaller than that of the input space
as a result of strong correlations between the input variables. We will see an example
of this when we consider images of handwritten digits in Chapter 12. If we are using
localized basis functions, we can arrange that they are scattered in input space only
in regions containing data. This approach is used in radial basis function networks
and also in support vector and relevance vector machines. Neural network models,
which use adaptive basis functions having sigmoidal nonlinearities, can adapt the
parameters so that the regions of input space over which the basis functions vary
corresponds to the data manifold. The second property is that target variables may
have signiﬁcant dependence on only a small number of possible directions within the
data manifold. Neural networks can exploit this property by choosing the directions
in input space to which the basis functions respond.
Exercises
3.1
(⋆) www
Show that the ‘tanh’ function and the logistic sigmoid function (3.6)
are related by
tanh(a) = 2σ(2a) −1.
(3.100)
Hence show that a general linear combination of logistic sigmoid functions of the
form
y(x, w) = w0 +
M

j=1
wjσ
x −µj
s

(3.101)
is equivalent to a linear combination of ‘tanh’ functions of the form
y(x, u) = u0 +
M

j=1
uj tanh
x −µj
s

(3.102)
and ﬁnd expressions to relate the new parameters {u1, . . . , uM} to the original pa-
rameters {w1, . . . , wM}.


---
**Page 192**
174
3. LINEAR MODELS FOR REGRESSION
3.2
(⋆⋆) Show that the matrix
Φ(ΦTΦ)−1ΦT
(3.103)
takes any vector v and projects it onto the space spanned by the columns of Φ. Use
this result to show that the least-squares solution (3.15) corresponds to an orthogonal
projection of the vector t onto the manifold S as shown in Figure 3.2.
3.3
(⋆)
Consider a data set in which each data point tn is associated with a weighting
factor rn > 0, so that the sum-of-squares error function becomes
ED(w) = 1
2
N

n=1
rn

tn −wTφ(xn)
2 .
(3.104)
Find an expression for the solution w⋆that minimizes this error function. Give two
alternative interpretations of the weighted sum-of-squares error function in terms of
(i) data dependent noise variance and (ii) replicated data points.
3.4
(⋆) www
Consider a linear model of the form
y(x, w) = w0 +
D

i=1
wixi
(3.105)
together with a sum-of-squares error function of the form
ED(w) = 1
2
N

n=1
{y(xn, w) −tn}2 .
(3.106)
Now suppose that Gaussian noise ϵi with zero mean and variance σ2 is added in-
dependently to each of the input variables xi. By making use of E[ϵi] = 0 and
E[ϵiϵj] = δijσ2, show that minimizing ED averaged over the noise distribution is
equivalent to minimizing the sum-of-squares error for noise-free input variables with
the addition of a weight-decay regularization term, in which the bias parameter w0
is omitted from the regularizer.
3.5
(⋆) www
Using the technique of Lagrange multipliers, discussed in Appendix E,
show that minimization of the regularized error function (3.29) is equivalent to mini-
mizing the unregularized sum-of-squares error (3.12) subject to the constraint (3.30).
Discuss the relationship between the parameters η and λ.
3.6
(⋆) www
Consider a linear basis function regression model for a multivariate
target variable t having a Gaussian distribution of the form
p(t|W, Σ) = N(t|y(x, W), Σ)
(3.107)
where
y(x, W) = WTφ(x)
(3.108)


---
**Page 193**
Exercises
175
together with a training data set comprising input basis vectors φ(xn) and corre-
sponding target vectors tn, with n = 1, . . . , N. Show that the maximum likelihood
solution WML for the parameter matrix W has the property that each column is
given by an expression of the form (3.15), which was the solution for an isotropic
noise distribution. Note that this is independent of the covariance matrix Σ. Show
that the maximum likelihood solution for Σ is given by
Σ = 1
N
N

n=1

tn −WT
MLφ(xn)
 
tn −WT
MLφ(xn)
T .
(3.109)
3.7
(⋆) By using the technique of completing the square, verify the result (3.49) for the
posterior distribution of the parameters w in the linear basis function model in which
mN and SN are deﬁned by (3.50) and (3.51) respectively.
3.8
(⋆⋆) www
Consider the linear basis function model in Section 3.1, and suppose
that we have already observed N data points, so that the posterior distribution over
w is given by (3.49). This posterior can be regarded as the prior for the next obser-
vation. By considering an additional data point (xN+1, tN+1), and by completing
the square in the exponential, show that the resulting posterior distribution is again
given by (3.49) but with SN replaced by SN+1 and mN replaced by mN+1.
3.9
(⋆⋆)
Repeat the previous exercise but instead of completing the square by hand,
make use of the general result for linear-Gaussian models given by (2.116).
3.10
(⋆⋆) www
By making use of the result (2.115) to evaluate the integral in (3.57),
verify that the predictive distribution for the Bayesian linear regression model is
given by (3.58) in which the input-dependent variance is given by (3.59).
3.11
(⋆⋆) We have seen that, as the size of a data set increases, the uncertainty associated
with the posterior distribution over model parameters decreases. Make use of the
matrix identity (Appendix C)

M + vvT−1 = M−1 −(M−1v) 
vTM−1
1 + vTM−1v
(3.110)
to show that the uncertainty σ2
N(x) associated with the linear regression function
given by (3.59) satisﬁes
σ2
N+1(x) ⩽σ2
N(x).
(3.111)
3.12
(⋆⋆)
We saw in Section 2.3.6 that the conjugate prior for a Gaussian distribution
with unknown mean and unknown precision (inverse variance) is a normal-gamma
distribution. This property also holds for the case of the conditional Gaussian dis-
tribution p(t|x, w, β) of the linear regression model. If we consider the likelihood
function (3.10), then the conjugate prior for w and β is given by
p(w, β) = N(w|m0, β−1S0)Gam(β|a0, b0).
(3.112)


---
**Page 194**
176
3. LINEAR MODELS FOR REGRESSION
Show that the corresponding posterior distribution takes the same functional form,
so that
p(w, β|t) = N(w|mN, β−1SN)Gam(β|aN, bN)
(3.113)
and ﬁnd expressions for the posterior parameters mN, SN, aN, and bN.
3.13
(⋆⋆)
Show that the predictive distribution p(t|x, t) for the model discussed in Ex-
ercise 3.12 is given by a Student’s t-distribution of the form
p(t|x, t) = St(t|µ, λ, ν)
(3.114)
and obtain expressions for µ, λ and ν.
3.14
(⋆⋆)
In this exercise, we explore in more detail the properties of the equivalent
kernel deﬁned by (3.62), where SN is deﬁned by (3.54). Suppose that the basis
functions φj(x) are linearly independent and that the number N of data points is
greater than the number M of basis functions. Furthermore, let one of the basis
functions be constant, say φ0(x) = 1. By taking suitable linear combinations of
these basis functions, we can construct a new basis set ψj(x) spanning the same
space but that are orthonormal, so that
N

n=1
ψj(xn)ψk(xn) = Ijk
(3.115)
where Ijk is deﬁned to be 1 if j = k and 0 otherwise, and we take ψ0(x) = 1. Show
that for α = 0, the equivalent kernel can be written as k(x, x′) = ψ(x)Tψ(x′)
where ψ = (ψ1, . . . , ψM)T. Use this result to show that the kernel satisﬁes the
summation constraint
N

n=1
k(x, xn) = 1.
(3.116)
3.15
(⋆) www
Consider a linear basis function model for regression in which the pa-
rameters α and β are set using the evidence framework. Show that the function
E(mN) deﬁned by (3.82) satisﬁes the relation 2E(mN) = N.
3.16
(⋆⋆)
Derive the result (3.86) for the log evidence function p(t|α, β) of the linear
regression model by making use of (2.115) to evaluate the integral (3.77) directly.
3.17
(⋆)
Show that the evidence function for the Bayesian linear regression model can
be written in the form (3.78) in which E(w) is deﬁned by (3.79).
3.18
(⋆⋆) www
By completing the square over w, show that the error function (3.79)
in Bayesian linear regression can be written in the form (3.80).
3.19
(⋆⋆) Show that the integration over w in the Bayesian linear regression model gives
the result (3.85). Hence show that the log marginal likelihood is given by (3.86).


---
**Page 195**
Exercises
177
3.20
(⋆⋆) www
Starting from (3.86) verify all of the steps needed to show that maxi-
mization of the log marginal likelihood function (3.86) with respect to α leads to the
re-estimation equation (3.92).
3.21
(⋆⋆) An alternative way to derive the result (3.92) for the optimal value of α in the
evidence framework is to make use of the identity
d
dα ln |A| = Tr

A−1 d
dαA

.
(3.117)
Prove this identity by considering the eigenvalue expansion of a real, symmetric
matrix A, and making use of the standard results for the determinant and trace of
A expressed in terms of its eigenvalues (Appendix C). Then make use of (3.117) to
derive (3.92) starting from (3.86).
3.22
(⋆⋆)
Starting from (3.86) verify all of the steps needed to show that maximiza-
tion of the log marginal likelihood function (3.86) with respect to β leads to the
re-estimation equation (3.95).
3.23
(⋆⋆) www
Show that the marginal probability of the data, in other words the
model evidence, for the model described in Exercise 3.12 is given by
p(t) =
1
(2π)N/2
ba0
0
baN
N
Γ(aN)
Γ(a0)
|SN|1/2
|S0|1/2
(3.118)
by ﬁrst marginalizing with respect to w and then with respect to β.
3.24
(⋆⋆) Repeat the previous exercise but now use Bayes’ theorem in the form
p(t) = p(t|w, β)p(w, β)
p(w, β|t)
(3.119)
and then substitute for the prior and posterior distributions and the likelihood func-
tion in order to derive the result (3.118).


---
**Page 196**
4
Linear
Models for
Classiﬁcation
In the previous chapter, we explored a class of regression models having particularly
simple analytical and computational properties. We now discuss an analogous class
of models for solving classiﬁcation problems. The goal in classiﬁcation is to take an
input vector x and to assign it to one of K discrete classes Ck where k = 1, . . . , K.
In the most common scenario, the classes are taken to be disjoint, so that each input is
assigned to one and only one class. The input space is thereby divided into decision
regions whose boundaries are called decision boundaries or decision surfaces. In
this chapter, we consider linear models for classiﬁcation, by which we mean that the
decision surfaces are linear functions of the input vector x and hence are deﬁned
by (D −1)-dimensional hyperplanes within the D-dimensional input space. Data
sets whose classes can be separated exactly by linear decision surfaces are said to be
linearly separable.
For regression problems, the target variable t was simply the vector of real num-
bers whose values we wish to predict. In the case of classiﬁcation, there are various
179


---
**Page 197**
180
4. LINEAR MODELS FOR CLASSIFICATION
ways of using target values to represent class labels. For probabilistic models, the
most convenient, in the case of two-class problems, is the binary representation in
which there is a single target variable t ∈{0, 1} such that t = 1 represents class C1
and t = 0 represents class C2. We can interpret the value of t as the probability that
the class is C1, with the values of probability taking only the extreme values of 0 and
1. For K > 2 classes, it is convenient to use a 1-of-K coding scheme in which t is
a vector of length K such that if the class is Cj, then all elements tk of t are zero
except element tj, which takes the value 1. For instance, if we have K = 5 classes,
then a pattern from class 2 would be given the target vector
t = (0, 1, 0, 0, 0)T.
(4.1)
Again, we can interpret the value of tk as the probability that the class is Ck. For
nonprobabilistic models, alternative choices of target variable representation will
sometimes prove convenient.
In Chapter 1, we identiﬁed three distinct approaches to the classiﬁcation prob-
lem. The simplest involves constructing a discriminant function that directly assigns
each vector x to a speciﬁc class. A more powerful approach, however, models the
conditional probability distribution p(Ck|x) in an inference stage, and then subse-
quently uses this distribution to make optimal decisions. By separating inference
and decision, we gain numerous beneﬁts, as discussed in Section 1.5.4. There are
two different approaches to determining the conditional probabilities p(Ck|x). One
technique is to model them directly, for example by representing them as parametric
models and then optimizing the parameters using a training set. Alternatively, we
can adopt a generative approach in which we model the class-conditional densities
given by p(x|Ck), together with the prior probabilities p(Ck) for the classes, and then
we compute the required posterior probabilities using Bayes’ theorem
p(Ck|x) = p(x|Ck)p(Ck)
p(x)
.
(4.2)
We shall discuss examples of all three approaches in this chapter.
In the linear regression models considered in Chapter 3, the model prediction
y(x, w) was given by a linear function of the parameters w. In the simplest case,
the model is also linear in the input variables and therefore takes the form y(x) =
wTx+w0, so that y is a real number. For classiﬁcation problems, however, we wish
to predict discrete class labels, or more generally posterior probabilities that lie in
the range (0, 1). To achieve this, we consider a generalization of this model in which
we transform the linear function of w using a nonlinear function f( · ) so that
y(x) = f

wTx + w0

.
(4.3)
In the machine learning literature f( · ) is known as an activation function, whereas
its inverse is called a link function in the statistics literature. The decision surfaces
correspond to y(x) = constant, so that wTx + w0 = constant and hence the deci-
sion surfaces are linear functions of x, even if the function f(·) is nonlinear. For this
reason, the class of models described by (4.3) are called generalized linear models


---
**Page 198**
4.1. Discriminant Functions
181
(McCullagh and Nelder, 1989). Note, however, that in contrast to the models used
for regression, they are no longer linear in the parameters due to the presence of the
nonlinear function f(·). This will lead to more complex analytical and computa-
tional properties than for linear regression models. Nevertheless, these models are
still relatively simple compared to the more general nonlinear models that will be
studied in subsequent chapters.
The algorithms discussed in this chapter will be equally applicable if we ﬁrst
make a ﬁxed nonlinear transformation of the input variables using a vector of basis
functions φ(x) as we did for regression models in Chapter 3. We begin by consider-
ing classiﬁcation directly in the original input space x, while in Section 4.3 we shall
ﬁnd it convenient to switch to a notation involving basis functions for consistency
with later chapters.
4.1. Discriminant Functions
A discriminant is a function that takes an input vector x and assigns it to one of K
classes, denoted Ck. In this chapter, we shall restrict attention to linear discriminants,
namely those for which the decision surfaces are hyperplanes. To simplify the dis-
cussion, we consider ﬁrst the case of two classes and then investigate the extension
to K > 2 classes.
4.1.1
Two classes
The simplest representation of a linear discriminant function is obtained by tak-
ing a linear function of the input vector so that
y(x) = wTx + w0
(4.4)
where w is called a weight vector, and w0 is a bias (not to be confused with bias in
the statistical sense). The negative of the bias is sometimes called a threshold. An
input vector x is assigned to class C1 if y(x) ⩾0 and to class C2 otherwise. The cor-
responding decision boundary is therefore deﬁned by the relation y(x) = 0, which
corresponds to a (D −1)-dimensional hyperplane within the D-dimensional input
space. Consider two points xA and xB both of which lie on the decision surface.
Because y(xA) = y(xB) = 0, we have wT(xA −xB) = 0 and hence the vector w is
orthogonal to every vector lying within the decision surface, and so w determines the
orientation of the decision surface. Similarly, if x is a point on the decision surface,
then y(x) = 0, and so the normal distance from the origin to the decision surface is
given by
wTx
∥w∥= −w0
∥w∥.
(4.5)
We therefore see that the bias parameter w0 determines the location of the decision
surface. These properties are illustrated for the case of D = 2 in Figure 4.1.
Furthermore, we note that the value of y(x) gives a signed measure of the per-
pendicular distance r of the point x from the decision surface. To see this, consider


---
**Page 199**
182
4. LINEAR MODELS FOR CLASSIFICATION
Figure 4.1
Illustration of the geometry of a
linear discriminant function in two dimensions.
The decision surface, shown in red, is perpen-
dicular to w, and its displacement from the
origin is controlled by the bias parameter w0.
Also, the signed orthogonal distance of a gen-
eral point x from the decision surface is given
by y(x)/∥w∥.
x2
x1
w
x
y(x)
∥w∥
x⊥
−w0
∥w∥
y = 0
y < 0
y > 0
R2
R1
an arbitrary point x and let x⊥be its orthogonal projection onto the decision surface,
so that
x = x⊥+ r w
∥w∥.
(4.6)
Multiplying both sides of this result by wT and adding w0, and making use of y(x) =
wTx + w0 and y(x⊥) = wTx⊥+ w0 = 0, we have
r = y(x)
∥w∥.
(4.7)
This result is illustrated in Figure 4.1.
As with the linear regression models in Chapter 3, it is sometimes convenient
to use a more compact notation in which we introduce an additional dummy ‘input’
value x0 = 1 and then deﬁne w = (w0, w) and x = (x0, x) so that
y(x) = wTx.
(4.8)
In this case, the decision surfaces are D-dimensional hyperplanes passing through
the origin of the D + 1-dimensional expanded input space.
4.1.2
Multiple classes
Now consider the extension of linear discriminants to K > 2 classes. We might
be tempted be to build a K-class discriminant by combining a number of two-class
discriminant functions. However, this leads to some serious difﬁculties (Duda and
Hart, 1973) as we now show.
Consider the use of K−1 classiﬁers each of which solves a two-class problem of
separating points in a particular class Ck from points not in that class. This is known
as a one-versus-the-rest classiﬁer. The left-hand example in Figure 4.2 shows an


---
**Page 200**
4.1. Discriminant Functions
183
R1
R2
R3
?
C1
not C1
C2
not C2
R1
R2
R3
?
C1
C2
C1
C3
C2
C3
Figure 4.2
Attempting to construct a K class discriminant from a set of two class discriminants leads to am-
biguous regions, shown in green. On the left is an example involving the use of two discriminants designed to
distinguish points in class Ck from points not in class Ck. On the right is an example involving three discriminant
functions each of which is used to separate a pair of classes Ck and Cj.
example involving three classes where this approach leads to regions of input space
that are ambiguously classiﬁed.
An alternative is to introduce K(K −1)/2 binary discriminant functions, one
for every possible pair of classes. This is known as a one-versus-one classiﬁer. Each
point is then classiﬁed according to a majority vote amongst the discriminant func-
tions. However, this too runs into the problem of ambiguous regions, as illustrated
in the right-hand diagram of Figure 4.2.
We can avoid these difﬁculties by considering a single K-class discriminant
comprising K linear functions of the form
yk(x) = wT
k x + wk0
(4.9)
and then assigning a point x to class Ck if yk(x) > yj(x) for all j ̸= k. The decision
boundary between class Ck and class Cj is therefore given by yk(x) = yj(x) and
hence corresponds to a (D −1)-dimensional hyperplane deﬁned by
(wk −wj)Tx + (wk0 −wj0) = 0.
(4.10)
This has the same form as the decision boundary for the two-class case discussed in
Section 4.1.1, and so analogous geometrical properties apply.
The decision regions of such a discriminant are always singly connected and
convex. To see this, consider two points xA and xB both of which lie inside decision
region Rk, as illustrated in Figure 4.3. Any point x that lies on the line connecting
xA and xB can be expressed in the form
x = λxA + (1 −λ)xB
(4.11)


---
**Page 201**
184
4. LINEAR MODELS FOR CLASSIFICATION
Figure 4.3
Illustration of the decision regions for a mul-
ticlass linear discriminant, with the decision
boundaries shown in red.
If two points xA
and xB both lie inside the same decision re-
gion Rk, then any point bx that lies on the line
connecting these two points must also lie in
Rk, and hence the decision region must be
singly connected and convex.
Ri
Rj
Rk
xA
xB
ˆx
where 0 ⩽λ ⩽1. From the linearity of the discriminant functions, it follows that
yk(x) = λyk(xA) + (1 −λ)yk(xB).
(4.12)
Because both xA and xB lie inside Rk, it follows that yk(xA) > yj(xA), and
yk(xB) > yj(xB), for all j ̸= k, and hence yk(x) > yj(x), and so x also lies
inside Rk. Thus Rk is singly connected and convex.
Note that for two classes, we can either employ the formalism discussed here,
based on two discriminant functions y1(x) and y2(x), or else use the simpler but
equivalent formulation described in Section 4.1.1 based on a single discriminant
function y(x).
We now explore three approaches to learning the parameters of linear discrimi-
nant functions, based on least squares, Fisher’s linear discriminant, and the percep-
tron algorithm.
4.1.3
Least squares for classiﬁcation
In Chapter 3, we considered models that were linear functions of the parame-
ters, and we saw that the minimization of a sum-of-squares error function led to a
simple closed-form solution for the parameter values. It is therefore tempting to see
if we can apply the same formalism to classiﬁcation problems. Consider a general
classiﬁcation problem with K classes, with a 1-of-K binary coding scheme for the
target vector t. One justiﬁcation for using least squares in such a context is that it
approximates the conditional expectation E[t|x] of the target values given the input
vector. For the binary coding scheme, this conditional expectation is given by the
vector of posterior class probabilities. Unfortunately, however, these probabilities
are typically approximated rather poorly, indeed the approximations can have values
outside the range (0, 1), due to the limited ﬂexibility of a linear model as we shall
see shortly.
Each class Ck is described by its own linear model so that
yk(x) = wT
k x + wk0
(4.13)
where k = 1, . . . , K. We can conveniently group these together using vector nota-
tion so that
y(x) = ,
WTx
(4.14)


---
**Page 202**
4.1. Discriminant Functions
185
where ,
W is a matrix whose kth column comprises the D + 1-dimensional vector
wk = (wk0, wT
k )T and x is the corresponding augmented input vector (1, xT)T with
a dummy input x0 = 1. This representation was discussed in detail in Section 3.1. A
new input x is then assigned to the class for which the output yk = wT
k x is largest.
We now determine the parameter matrix ,
W by minimizing a sum-of-squares
error function, as we did for regression in Chapter 3. Consider a training data set
{xn, tn} where n = 1, . . . , N, and deﬁne a matrix T whose nth row is the vector tT
n,
together with a matrix X whose nth row is xT
n. The sum-of-squares error function
can then be written as
ED(,
W) = 1
2Tr

(X,
W −T)T(X,
W −T)

.
(4.15)
Setting the derivative with respect to ,
W to zero, and rearranging, we then obtain the
solution for ,
W in the form
,
W = (XT X)−1 XTT = X†T
(4.16)
where X† is the pseudo-inverse of the matrix X, as discussed in Section 3.1.1. We
then obtain the discriminant function in the form
y(x) = ,
WTx = TT 
X†T
x.
(4.17)
An interesting property of least-squares solutions with multiple target variables
is that if every target vector in the training set satisﬁes some linear constraint
aTtn + b = 0
(4.18)
for some constants a and b, then the model prediction for any value of x will satisfy
the same constraint so that
Exercise 4.2
aTy(x) + b = 0.
(4.19)
Thus if we use a 1-of-K coding scheme for K classes, then the predictions made
by the model will have the property that the elements of y(x) will sum to 1 for any
value of x. However, this summation constraint alone is not sufﬁcient to allow the
model outputs to be interpreted as probabilities because they are not constrained to
lie within the interval (0, 1).
The least-squares approach gives an exact closed-form solution for the discrimi-
nant function parameters. However, even as a discriminant function (where we use it
to make decisions directly and dispense with any probabilistic interpretation) it suf-
fers from some severe problems. We have already seen that least-squares solutions
Section 2.3.7
lack robustness to outliers, and this applies equally to the classiﬁcation application,
as illustrated in Figure 4.4. Here we see that the additional data points in the right-
hand ﬁgure produce a signiﬁcant change in the location of the decision boundary,
even though these point would be correctly classiﬁed by the original decision bound-
ary in the left-hand ﬁgure. The sum-of-squares error function penalizes predictions
that are ‘too correct’ in that they lie a long way on the correct side of the decision


---
**Page 203**
186
4. LINEAR MODELS FOR CLASSIFICATION
−4
−2
0
2
4
6
8
−8
−6
−4
−2
0
2
4
−4
−2
0
2
4
6
8
−8
−6
−4
−2
0
2
4
Figure 4.4
The left plot shows data from two classes, denoted by red crosses and blue circles, together with
the decision boundary found by least squares (magenta curve) and also by the logistic regression model (green
curve), which is discussed later in Section 4.3.2. The right-hand plot shows the corresponding results obtained
when extra data points are added at the bottom left of the diagram, showing that least squares is highly sensitive
to outliers, unlike logistic regression.
boundary. In Section 7.1.2, we shall consider several alternative error functions for
classiﬁcation and we shall see that they do not suffer from this difﬁculty.
However, problems with least squares can be more severe than simply lack of
robustness, as illustrated in Figure 4.5. This shows a synthetic data set drawn from
three classes in a two-dimensional input space (x1, x2), having the property that lin-
ear decision boundaries can give excellent separation between the classes. Indeed,
the technique of logistic regression, described later in this chapter, gives a satisfac-
tory solution as seen in the right-hand plot. However, the least-squares solution gives
poor results, with only a small region of the input space assigned to the green class.
The failure of least squares should not surprise us when we recall that it cor-
responds to maximum likelihood under the assumption of a Gaussian conditional
distribution, whereas binary target vectors clearly have a distribution that is far from
Gaussian. By adopting more appropriate probabilistic models, we shall obtain clas-
siﬁcation techniques with much better properties than least squares. For the moment,
however, we continue to explore alternative nonprobabilistic methods for setting the
parameters in the linear classiﬁcation models.
4.1.4
Fisher’s linear discriminant
One way to view a linear classiﬁcation model is in terms of dimensionality
reduction.
Consider ﬁrst the case of two classes, and suppose we take the D-


---
**Page 204**
4.1. Discriminant Functions
187
−6
−4
−2
0
2
4
6
−6
−4
−2
0
2
4
6
−6
−4
−2
0
2
4
6
−6
−4
−2
0
2
4
6
Figure 4.5
Example of a synthetic data set comprising three classes, with training data points denoted in red
(×), green (+), and blue (◦). Lines denote the decision boundaries, and the background colours denote the
respective classes of the decision regions. On the left is the result of using a least-squares discriminant. We see
that the region of input space assigned to the green class is too small and so most of the points from this class
are misclassiﬁed. On the right is the result of using logistic regressions as described in Section 4.3.2 showing
correct classiﬁcation of the training data.
dimensional input vector x and project it down to one dimension using
y = wTx.
(4.20)
If we place a threshold on y and classify y ⩾−w0 as class C1, and otherwise class
C2, then we obtain our standard linear classiﬁer discussed in the previous section.
In general, the projection onto one dimension leads to a considerable loss of infor-
mation, and classes that are well separated in the original D-dimensional space may
become strongly overlapping in one dimension. However, by adjusting the com-
ponents of the weight vector w, we can select a projection that maximizes the class
separation. To begin with, consider a two-class problem in which there are N1 points
of class C1 and N2 points of class C2, so that the mean vectors of the two classes are
given by
m1 = 1
N1

n ∈C1
xn,
m2 = 1
N2

n ∈C2
xn.
(4.21)
The simplest measure of the separation of the classes, when projected onto w, is the
separation of the projected class means. This suggests that we might choose w so as
to maximize
m2 −m1 = wT(m2 −m1)
(4.22)
where
mk = wTmk
(4.23)


---
**Page 205**
188
4. LINEAR MODELS FOR CLASSIFICATION
−2
2
6
−2
0
2
4
−2
2
6
−2
0
2
4
Figure 4.6
The left plot shows samples from two classes (depicted in red and blue) along with the histograms
resulting from projection onto the line joining the class means. Note that there is considerable class overlap in
the projected space. The right plot shows the corresponding projection based on the Fisher linear discriminant,
showing the greatly improved class separation.
is the mean of the projected data from class Ck. However, this expression can be
made arbitrarily large simply by increasing the magnitude of w.
To solve this
problem, we could constrain w to have unit length, so that 
i w2
i = 1. Using
a Lagrange multiplier to perform the constrained maximization, we then ﬁnd that
Appendix E
w ∝(m2 −m1). There is still a problem with this approach, however, as illustrated
Exercise 4.4
in Figure 4.6. This shows two classes that are well separated in the original two-
dimensional space (x1, x2) but that have considerable overlap when projected onto
the line joining their means. This difﬁculty arises from the strongly nondiagonal
covariances of the class distributions. The idea proposed by Fisher is to maximize
a function that will give a large separation between the projected class means while
also giving a small variance within each class, thereby minimizing the class overlap.
The projection formula (4.20) transforms the set of labelled data points in x
into a labelled set in the one-dimensional space y. The within-class variance of the
transformed data from class Ck is therefore given by
s2
k =

n∈Ck
(yn −mk)2
(4.24)
where yn = wTxn. We can deﬁne the total within-class variance for the whole
data set to be simply s2
1 + s2
2. The Fisher criterion is deﬁned to be the ratio of the
between-class variance to the within-class variance and is given by
J(w) = (m2 −m1)2
s2
1 + s2
2
.
(4.25)
We can make the dependence on w explicit by using (4.20), (4.23), and (4.24) to
rewrite the Fisher criterion in the form
Exercise 4.5


---
**Page 206**
4.1. Discriminant Functions
189
J(w) = wTSBw
wTSWw
(4.26)
where SB is the between-class covariance matrix and is given by
SB = (m2 −m1)(m2 −m1)T
(4.27)
and SW is the total within-class covariance matrix, given by
SW =

n∈C1
(xn −m1)(xn −m1)T +

n∈C2
(xn −m2)(xn −m2)T.
(4.28)
Differentiating (4.26) with respect to w, we ﬁnd that J(w) is maximized when
(wTSBw)SWw = (wTSWw)SBw.
(4.29)
From (4.27), we see that SBw is always in the direction of (m2−m1). Furthermore,
we do not care about the magnitude of w, only its direction, and so we can drop the
scalar factors (wTSBw) and (wTSWw). Multiplying both sides of (4.29) by S−1
W
we then obtain
w ∝S−1
W (m2 −m1).
(4.30)
Note that if the within-class covariance is isotropic, so that SW is proportional to the
unit matrix, we ﬁnd that w is proportional to the difference of the class means, as
discussed above.
The result (4.30) is known as Fisher’s linear discriminant, although strictly it
is not a discriminant but rather a speciﬁc choice of direction for projection of the
data down to one dimension. However, the projected data can subsequently be used
to construct a discriminant, by choosing a threshold y0 so that we classify a new
point as belonging to C1 if y(x) ⩾y0 and classify it as belonging to C2 otherwise.
For example, we can model the class-conditional densities p(y|Ck) using Gaussian
distributions and then use the techniques of Section 1.2.4 to ﬁnd the parameters
of the Gaussian distributions by maximum likelihood. Having found Gaussian ap-
proximations to the projected classes, the formalism of Section 1.5.1 then gives an
expression for the optimal threshold. Some justiﬁcation for the Gaussian assumption
comes from the central limit theorem by noting that y = wTx is the sum of a set of
random variables.
4.1.5
Relation to least squares
The least-squares approach to the determination of a linear discriminant was
based on the goal of making the model predictions as close as possible to a set of
target values. By contrast, the Fisher criterion was derived by requiring maximum
class separation in the output space. It is interesting to see the relationship between
these two approaches. In particular, we shall show that, for the two-class problem,
the Fisher criterion can be obtained as a special case of least squares.
So far we have considered 1-of-K coding for the target values. If, however, we
adopt a slightly different target coding scheme, then the least-squares solution for


---
**Page 207**
190
4. LINEAR MODELS FOR CLASSIFICATION
the weights becomes equivalent to the Fisher solution (Duda and Hart, 1973). In
particular, we shall take the targets for class C1 to be N/N1, where N1 is the number
of patterns in class C1, and N is the total number of patterns. This target value
approximates the reciprocal of the prior probability for class C1. For class C2, we
shall take the targets to be −N/N2, where N2 is the number of patterns in class C2.
The sum-of-squares error function can be written
E = 1
2
N

n=1

wTxn + w0 −tn
2 .
(4.31)
Setting the derivatives of E with respect to w0 and w to zero, we obtain respectively
N

n=1

wTxn + w0 −tn

=
0
(4.32)
N

n=1

wTxn + w0 −tn

xn
=
0.
(4.33)
From (4.32), and making use of our choice of target coding scheme for the tn, we
obtain an expression for the bias in the form
w0 = −wTm
(4.34)
where we have used
N

n=1
tn = N1
N
N1
−N2
N
N2
= 0
(4.35)
and where m is the mean of the total data set and is given by
m = 1
N
N

n=1
xn = 1
N (N1m1 + N2m2).
(4.36)
After some straightforward algebra, and again making use of the choice of tn, the
second equation (4.33) becomes
Exercise 4.6

SW + N1N2
N
SB

w = N(m1 −m2)
(4.37)
where SW is deﬁned by (4.28), SB is deﬁned by (4.27), and we have substituted for
the bias using (4.34). Using (4.27), we note that SBw is always in the direction of
(m2 −m1). Thus we can write
w ∝S−1
W (m2 −m1)
(4.38)
where we have ignored irrelevant scale factors. Thus the weight vector coincides
with that found from the Fisher criterion. In addition, we have also found an expres-
sion for the bias value w0 given by (4.34). This tells us that a new vector x should be
classiﬁed as belonging to class C1 if y(x) = wT(x−m) > 0 and class C2 otherwise.


---
**Page 208**
4.1. Discriminant Functions
191
4.1.6
Fisher’s discriminant for multiple classes
We now consider the generalization of the Fisher discriminant to K > 2 classes,
and we shall assume that the dimensionality D of the input space is greater than the
number K of classes. Next, we introduce D′ > 1 linear ‘features’ yk = wT
k x, where
k = 1, . . . , D′. These feature values can conveniently be grouped together to form
a vector y. Similarly, the weight vectors {wk} can be considered to be the columns
of a matrix W, so that
y = WTx.
(4.39)
Note that again we are not including any bias parameters in the deﬁnition of y. The
generalization of the within-class covariance matrix to the case of K classes follows
from (4.28) to give
SW =
K

k=1
Sk
(4.40)
where
Sk
=

n∈Ck
(xn −mk)(xn −mk)T
(4.41)
mk
=
1
Nk

n∈Ck
xn
(4.42)
and Nk is the number of patterns in class Ck. In order to ﬁnd a generalization of the
between-class covariance matrix, we follow Duda and Hart (1973) and consider ﬁrst
the total covariance matrix
ST =
N

n=1
(xn −m)(xn −m)T
(4.43)
where m is the mean of the total data set
m = 1
N
N

n=1
xn = 1
N
K

k=1
Nkmk
(4.44)
and N = 
k Nk is the total number of data points. The total covariance matrix can
be decomposed into the sum of the within-class covariance matrix, given by (4.40)
and (4.41), plus an additional matrix SB, which we identify as a measure of the
between-class covariance
ST = SW + SB
(4.45)
where
SB =
K

k=1
Nk(mk −m)(mk −m)T.
(4.46)


---
**Page 209**
192
4. LINEAR MODELS FOR CLASSIFICATION
These covariance matrices have been deﬁned in the original x-space. We can now
deﬁne similar matrices in the projected D′-dimensional y-space
sW =
K

k=1

n∈Ck
(yn −µk)(yn −µk)T
(4.47)
and
sB =
K

k=1
Nk(µk −µ)(µk −µ)T
(4.48)
where
µk = 1
Nk

n∈Ck
yn,
µ = 1
N
K

k=1
Nkµk.
(4.49)
Again we wish to construct a scalar that is large when the between-class covariance
is large and when the within-class covariance is small. There are now many possible
choices of criterion (Fukunaga, 1990). One example is given by
J(W) = Tr 
s−1
W sB

.
(4.50)
This criterion can then be rewritten as an explicit function of the projection matrix
W in the form
J(w) = Tr 
(WSWWT)−1(WSBWT)
.
(4.51)
Maximization of such criteria is straightforward, though somewhat involved, and is
discussed at length in Fukunaga (1990). The weight values are determined by those
eigenvectors of S−1
W SB that correspond to the D′ largest eigenvalues.
There is one important result that is common to all such criteria, which is worth
emphasizing. We ﬁrst note from (4.46) that SB is composed of the sum of K ma-
trices, each of which is an outer product of two vectors and therefore of rank 1. In
addition, only (K −1) of these matrices are independent as a result of the constraint
(4.44). Thus, SB has rank at most equal to (K −1) and so there are at most (K −1)
nonzero eigenvalues. This shows that the projection onto the (K −1)-dimensional
subspace spanned by the eigenvectors of SB does not alter the value of J(w), and
so we are therefore unable to ﬁnd more than (K −1) linear ‘features’ by this means
(Fukunaga, 1990).
4.1.7
The perceptron algorithm
Another example of a linear discriminant model is the perceptron of Rosenblatt
(1962), which occupies an important place in the history of pattern recognition al-
gorithms. It corresponds to a two-class model in which the input vector x is ﬁrst
transformed using a ﬁxed nonlinear transformation to give a feature vector φ(x),
and this is then used to construct a generalized linear model of the form
y(x) = f 
wTφ(x)
(4.52)


---
**Page 210**
4.1. Discriminant Functions
193
where the nonlinear activation function f(·) is given by a step function of the form
f(a) =

+1,
a ⩾0
−1,
a < 0.
(4.53)
The vector φ(x) will typically include a bias component φ0(x) = 1. In earlier
discussions of two-class classiﬁcation problems, we have focussed on a target coding
scheme in which t ∈{0, 1}, which is appropriate in the context of probabilistic
models. For the perceptron, however, it is more convenient to use target values
t = +1 for class C1 and t = −1 for class C2, which matches the choice of activation
function.
The algorithm used to determine the parameters w of the perceptron can most
easily be motivated by error function minimization. A natural choice of error func-
tion would be the total number of misclassiﬁed patterns. However, this does not lead
to a simple learning algorithm because the error is a piecewise constant function
of w, with discontinuities wherever a change in w causes the decision boundary to
move across one of the data points. Methods based on changing w using the gradi-
ent of the error function cannot then be applied, because the gradient is zero almost
everywhere.
We therefore consider an alternative error function known as the perceptron cri-
terion. To derive this, we note that we are seeking a weight vector w such that
patterns xn in class C1 will have wTφ(xn) > 0, whereas patterns xn in class C2
have wTφ(xn) < 0. Using the t ∈{−1, +1} target coding scheme it follows that
we would like all patterns to satisfy wTφ(xn)tn > 0. The perceptron criterion
associates zero error with any pattern that is correctly classiﬁed, whereas for a mis-
classiﬁed pattern xn it tries to minimize the quantity −wTφ(xn)tn. The perceptron
criterion is therefore given by
EP(w) = −

n∈M
wTφntn
(4.54)
Frank Rosenblatt
1928–1969
Rosenblatt’s perceptron played an
important role in the history of ma-
chine learning. Initially, Rosenblatt
simulated the perceptron on an IBM
704 computer at Cornell in 1957,
but by the early 1960s he had built
special-purpose hardware that provided a direct, par-
allel implementation of perceptron learning. Many of
his ideas were encapsulated in “Principles of Neuro-
dynamics: Perceptrons and the Theory of Brain Mech-
anisms” published in 1962.
Rosenblatt’s work was
criticized by Marvin Minksy, whose objections were
published in the book “Perceptrons”, co-authored with
Seymour Papert.
This book was widely misinter-
preted at the time as showing that neural networks
were fatally ﬂawed and could only learn solutions for
linearly separable problems.
In fact, it only proved
such limitations in the case of single-layer networks
such as the perceptron and merely conjectured (in-
correctly) that they applied to more general network
models. Unfortunately, however, this book contributed
to the substantial decline in research funding for neu-
ral computing, a situation that was not reversed un-
til the mid-1980s. Today, there are many hundreds,
if not thousands, of applications of neural networks
in widespread use, with examples in areas such as
handwriting recognition and information retrieval be-
ing used routinely by millions of people.


---
**Page 211**
194
4. LINEAR MODELS FOR CLASSIFICATION
where M denotes the set of all misclassiﬁed patterns. The contribution to the error
associated with a particular misclassiﬁed pattern is a linear function of w in regions
of w space where the pattern is misclassiﬁed and zero in regions where it is correctly
classiﬁed. The total error function is therefore piecewise linear.
We now apply the stochastic gradient descent algorithm to this error function.
Section 3.1.3
The change in the weight vector w is then given by
w(τ+1) = w(τ) −η∇EP(w) = w(τ) + ηφntn
(4.55)
where η is the learning rate parameter and τ is an integer that indexes the steps of
the algorithm. Because the perceptron function y(x, w) is unchanged if we multiply
w by a constant, we can set the learning rate parameter η equal to 1 without of
generality. Note that, as the weight vector evolves during training, the set of patterns
that are misclassiﬁed will change.
The perceptron learning algorithm has a simple interpretation, as follows. We
cycle through the training patterns in turn, and for each pattern xn we evaluate the
perceptron function (4.52). If the pattern is correctly classiﬁed, then the weight
vector remains unchanged, whereas if it is incorrectly classiﬁed, then for class C1
we add the vector φ(xn) onto the current estimate of weight vector w while for
class C2 we subtract the vector φ(xn) from w. The perceptron learning algorithm is
illustrated in Figure 4.7.
If we consider the effect of a single update in the perceptron learning algorithm,
we see that the contribution to the error from a misclassiﬁed pattern will be reduced
because from (4.55) we have
−w(τ+1)Tφntn = −w(τ)Tφntn −(φntn)Tφntn < −w(τ)Tφntn
(4.56)
where we have set η = 1, and made use of ∥φntn∥2 > 0. Of course, this does
not imply that the contribution to the error function from the other misclassiﬁed
patterns will have been reduced. Furthermore, the change in weight vector may have
caused some previously correctly classiﬁed patterns to become misclassiﬁed. Thus
the perceptron learning rule is not guaranteed to reduce the total error function at
each stage.
However, the perceptron convergence theorem states that if there exists an ex-
act solution (in other words, if the training data set is linearly separable), then the
perceptron learning algorithm is guaranteed to ﬁnd an exact solution in a ﬁnite num-
ber of steps. Proofs of this theorem can be found for example in Rosenblatt (1962),
Block (1962), Nilsson (1965), Minsky and Papert (1969), Hertz et al. (1991), and
Bishop (1995a). Note, however, that the number of steps required to achieve con-
vergence could still be substantial, and in practice, until convergence is achieved,
we will not be able to distinguish between a nonseparable problem and one that is
simply slow to converge.
Even when the data set is linearly separable, there may be many solutions, and
which one is found will depend on the initialization of the parameters and on the or-
der of presentation of the data points. Furthermore, for data sets that are not linearly
separable, the perceptron learning algorithm will never converge.


---
**Page 212**
4.1. Discriminant Functions
195
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
−1
−0.5
0
0.5
1
Figure 4.7
Illustration of the convergence of the perceptron learning algorithm, showing data points from two
classes (red and blue) in a two-dimensional feature space (φ1, φ2). The top left plot shows the initial parameter
vector w shown as a black arrow together with the corresponding decision boundary (black line), in which the
arrow points towards the decision region which classiﬁed as belonging to the red class. The data point circled
in green is misclassiﬁed and so its feature vector is added to the current weight vector, giving the new decision
boundary shown in the top right plot. The bottom left plot shows the next misclassiﬁed point to be considered,
indicated by the green circle, and its feature vector is again added to the weight vector giving the decision
boundary shown in the bottom right plot for which all data points are correctly classiﬁed.


---
**Page 213**
196
4. LINEAR MODELS FOR CLASSIFICATION
Figure 4.8
Illustration of the Mark 1 perceptron hardware. The photograph on the left shows how the inputs
were obtained using a simple camera system in which an input scene, in this case a printed character, was
illuminated by powerful lights, and an image focussed onto a 20 × 20 array of cadmium sulphide photocells,
giving a primitive 400 pixel image. The perceptron also had a patch board, shown in the middle photograph,
which allowed different conﬁgurations of input features to be tried. Often these were wired up at random to
demonstrate the ability of the perceptron to learn without the need for precise wiring, in contrast to a modern
digital computer. The photograph on the right shows one of the racks of adaptive weights. Each weight was
implemented using a rotary variable resistor, also called a potentiometer, driven by an electric motor thereby
allowing the value of the weight to be adjusted automatically by the learning algorithm.
Aside from difﬁculties with the learning algorithm, the perceptron does not pro-
vide probabilistic outputs, nor does it generalize readily to K > 2 classes. The most
important limitation, however, arises from the fact that (in common with all of the
models discussed in this chapter and the previous one) it is based on linear com-
binations of ﬁxed basis functions. More detailed discussions of the limitations of
perceptrons can be found in Minsky and Papert (1969) and Bishop (1995a).
Analogue hardware implementations of the perceptron were built by Rosenblatt,
based on motor-driven variable resistors to implement the adaptive parameters wj.
These are illustrated in Figure 4.8. The inputs were obtained from a simple camera
system based on an array of photo-sensors, while the basis functions φ could be
chosen in a variety of ways, for example based on simple ﬁxed functions of randomly
chosen subsets of pixels from the input image. Typical applications involved learning
to discriminate simple shapes or characters.
At the same time that the perceptron was being developed, a closely related
system called the adaline, which is short for ‘adaptive linear element’, was being
explored by Widrow and co-workers. The functional form of the model was the same
as for the perceptron, but a different approach to training was adopted (Widrow and
Hoff, 1960; Widrow and Lehr, 1990).
4.2. Probabilistic Generative Models
We turn next to a probabilistic view of classiﬁcation and show how models with
linear decision boundaries arise from simple assumptions about the distribution of
the data. In Section 1.5.4, we discussed the distinction between the discriminative
and the generative approaches to classiﬁcation. Here we shall adopt a generative


---
**Page 214**
4.2. Probabilistic Generative Models
197
Figure 4.9
Plot of the logistic sigmoid function
σ(a) deﬁned by (4.59), shown in
red, together with the scaled pro-
bit function Φ(λa), for λ2 = π/8,
shown in dashed blue, where Φ(a)
is deﬁned by (4.114).
The scal-
ing factor π/8 is chosen so that the
derivatives of the two curves are
equal for a = 0.
−5
0
5
0
0.5
1
approach in which we model the class-conditional densities p(x|Ck), as well as the
class priors p(Ck), and then use these to compute posterior probabilities p(Ck|x)
through Bayes’ theorem.
Consider ﬁrst of all the case of two classes. The posterior probability for class
C1 can be written as
p(C1|x)
=
p(x|C1)p(C1)
p(x|C1)p(C1) + p(x|C2)p(C2)
=
1
1 + exp(−a) = σ(a)
(4.57)
where we have deﬁned
a = ln p(x|C1)p(C1)
p(x|C2)p(C2)
(4.58)
and σ(a) is the logistic sigmoid function deﬁned by
σ(a) =
1
1 + exp(−a)
(4.59)
which is plotted in Figure 4.9. The term ‘sigmoid’ means S-shaped. This type of
function is sometimes also called a ‘squashing function’ because it maps the whole
real axis into a ﬁnite interval. The logistic sigmoid has been encountered already
in earlier chapters and plays an important role in many classiﬁcation algorithms. It
satisﬁes the following symmetry property
σ(−a) = 1 −σ(a)
(4.60)
as is easily veriﬁed. The inverse of the logistic sigmoid is given by
a = ln

σ
1 −σ

(4.61)
and is known as the logit function. It represents the log of the ratio of probabilities
ln [p(C1|x)/p(C2|x)] for the two classes, also known as the log odds.


---
**Page 215**
198
4. LINEAR MODELS FOR CLASSIFICATION
Note that in (4.57) we have simply rewritten the posterior probabilities in an
equivalent form, and so the appearance of the logistic sigmoid may seem rather vac-
uous. However, it will have signiﬁcance provided a(x) takes a simple functional
form. We shall shortly consider situations in which a(x) is a linear function of x, in
which case the posterior probability is governed by a generalized linear model.
For the case of K > 2 classes, we have
p(Ck|x)
=
p(x|Ck)p(Ck)

j p(x|Cj)p(Cj)
=
exp(ak)

j exp(aj)
(4.62)
which is known as the normalized exponential and can be regarded as a multiclass
generalization of the logistic sigmoid. Here the quantities ak are deﬁned by
ak = ln p(x|Ck)p(Ck).
(4.63)
The normalized exponential is also known as the softmax function, as it represents
a smoothed version of the ‘max’ function because, if ak ≫aj for all j ̸= k, then
p(Ck|x) ≃1, and p(Cj|x) ≃0.
We now investigate the consequences of choosing speciﬁc forms for the class-
conditional densities, looking ﬁrst at continuous input variables x and then dis-
cussing brieﬂy the case of discrete inputs.
4.2.1
Continuous inputs
Let us assume that the class-conditional densities are Gaussian and then explore
the resulting form for the posterior probabilities. To start with, we shall assume that
all classes share the same covariance matrix. Thus the density for class Ck is given
by
p(x|Ck) =
1
(2π)D/2
1
|Σ|1/2 exp

−1
2(x −µk)TΣ−1(x −µk)

.
(4.64)
Consider ﬁrst the case of two classes. From (4.57) and (4.58), we have
p(C1|x) = σ(wTx + w0)
(4.65)
where we have deﬁned
w
=
Σ−1(µ1 −µ2)
(4.66)
w0
=
−1
2µT
1 Σ−1µ1 + 1
2µT
2 Σ−1µ2 + ln p(C1)
p(C2).
(4.67)
We see that the quadratic terms in x from the exponents of the Gaussian densities
have cancelled (due to the assumption of common covariance matrices) leading to
a linear function of x in the argument of the logistic sigmoid. This result is illus-
trated for the case of a two-dimensional input space x in Figure 4.10. The resulting


---
**Page 216**
4.2. Probabilistic Generative Models
199
Figure 4.10
The left-hand plot shows the class-conditional densities for two classes, denoted red and blue.
On the right is the corresponding posterior probability p(C1|x), which is given by a logistic sigmoid of a linear
function of x. The surface in the right-hand plot is coloured using a proportion of red ink given by p(C1|x) and a
proportion of blue ink given by p(C2|x) = 1 −p(C1|x).
decision boundaries correspond to surfaces along which the posterior probabilities
p(Ck|x) are constant and so will be given by linear functions of x, and therefore
the decision boundaries are linear in input space. The prior probabilities p(Ck) enter
only through the bias parameter w0 so that changes in the priors have the effect of
making parallel shifts of the decision boundary and more generally of the parallel
contours of constant posterior probability.
For the general case of K classes we have, from (4.62) and (4.63),
ak(x) = wT
k x + wk0
(4.68)
where we have deﬁned
wk
=
Σ−1µk
(4.69)
wk0
=
−1
2µT
k Σ−1µk + ln p(Ck).
(4.70)
We see that the ak(x) are again linear functions of x as a consequence of the cancel-
lation of the quadratic terms due to the shared covariances. The resulting decision
boundaries, corresponding to the minimum misclassiﬁcation rate, will occur when
two of the posterior probabilities (the two largest) are equal, and so will be deﬁned
by linear functions of x, and so again we have a generalized linear model.
If we relax the assumption of a shared covariance matrix and allow each class-
conditional density p(x|Ck) to have its own covariance matrix Σk, then the earlier
cancellations will no longer occur, and we will obtain quadratic functions of x, giv-
ing rise to a quadratic discriminant. The linear and quadratic decision boundaries
are illustrated in Figure 4.11.


---
**Page 217**
200
4. LINEAR MODELS FOR CLASSIFICATION
−2
−1
0
1
2
−2.5
−2
−1.5
−1
−0.5
0
0.5
1
1.5
2
2.5
Figure 4.11
The left-hand plot shows the class-conditional densities for three classes each having a Gaussian
distribution, coloured red, green, and blue, in which the red and green classes have the same covariance matrix.
The right-hand plot shows the corresponding posterior probabilities, in which the RGB colour vector represents
the posterior probabilities for the respective three classes. The decision boundaries are also shown. Notice that
the boundary between the red and green classes, which have the same covariance matrix, is linear, whereas
those between the other pairs of classes are quadratic.
4.2.2
Maximum likelihood solution
Once we have speciﬁed a parametric functional form for the class-conditional
densities p(x|Ck), we can then determine the values of the parameters, together with
the prior class probabilities p(Ck), using maximum likelihood. This requires a data
set comprising observations of x along with their corresponding class labels.
Consider ﬁrst the case of two classes, each having a Gaussian class-conditional
density with a shared covariance matrix, and suppose we have a data set {xn, tn}
where n = 1, . . . , N. Here tn = 1 denotes class C1 and tn = 0 denotes class C2. We
denote the prior class probability p(C1) = π, so that p(C2) = 1 −π. For a data point
xn from class C1, we have tn = 1 and hence
p(xn, C1) = p(C1)p(xn|C1) = πN(xn|µ1, Σ).
Similarly for class C2, we have tn = 0 and hence
p(xn, C2) = p(C2)p(xn|C2) = (1 −π)N(xn|µ2, Σ).
Thus the likelihood function is given by
p(t|π, µ1, µ2, Σ) =
N

n=1
[πN(xn|µ1, Σ)]tn [(1 −π)N(xn|µ2, Σ)]1−tn
(4.71)
where t = (t1, . . . , tN)T. As usual, it is convenient to maximize the log of the
likelihood function. Consider ﬁrst the maximization with respect to π. The terms in


---
**Page 218**
4.2. Probabilistic Generative Models
201
the log likelihood function that depend on π are
N

n=1
{tn ln π + (1 −tn) ln(1 −π)} .
(4.72)
Setting the derivative with respect to π equal to zero and rearranging, we obtain
π = 1
N
N

n=1
tn = N1
N =
N1
N1 + N2
(4.73)
where N1 denotes the total number of data points in class C1, and N2 denotes the total
number of data points in class C2. Thus the maximum likelihood estimate for π is
simply the fraction of points in class C1 as expected. This result is easily generalized
to the multiclass case where again the maximum likelihood estimate of the prior
probability associated with class Ck is given by the fraction of the training set points
assigned to that class.
Exercise 4.9
Now consider the maximization with respect to µ1. Again we can pick out of
the log likelihood function those terms that depend on µ1 giving
N

n=1
tn ln N(xn|µ1, Σ) = −1
2
N

n=1
tn(xn −µ1)TΣ−1(xn −µ1) + const. (4.74)
Setting the derivative with respect to µ1 to zero and rearranging, we obtain
µ1 = 1
N1
N

n=1
tnxn
(4.75)
which is simply the mean of all the input vectors xn assigned to class C1. By a
similar argument, the corresponding result for µ2 is given by
µ2 = 1
N2
N

n=1
(1 −tn)xn
(4.76)
which again is the mean of all the input vectors xn assigned to class C2.
Finally, consider the maximum likelihood solution for the shared covariance
matrix Σ. Picking out the terms in the log likelihood function that depend on Σ, we
have
−1
2
N

n=1
tn ln |Σ| −1
2
N

n=1
tn(xn −µ1)TΣ−1(xn −µ1)
−1
2
N

n=1
(1 −tn) ln |Σ| −1
2
N

n=1
(1 −tn)(xn −µ2)TΣ−1(xn −µ2)
= −N
2 ln |Σ| −N
2 Tr

Σ−1S

(4.77)


---
**Page 219**
202
4. LINEAR MODELS FOR CLASSIFICATION
where we have deﬁned
S
=
N1
N S1 + N2
N S2
(4.78)
S1
=
1
N1

n∈C1
(xn −µ1)(xn −µ1)T
(4.79)
S2
=
1
N2

n∈C2
(xn −µ2)(xn −µ2)T.
(4.80)
Using the standard result for the maximum likelihood solution for a Gaussian distri-
bution, we see that Σ = S, which represents a weighted average of the covariance
matrices associated with each of the two classes separately.
This result is easily extended to the K class problem to obtain the corresponding
maximum likelihood solutions for the parameters in which each class-conditional
density is Gaussian with a shared covariance matrix. Note that the approach of ﬁtting
Exercise 4.10
Gaussian distributions to the classes is not robust to outliers, because the maximum
likelihood estimation of a Gaussian is not robust.
Section 2.3.7
4.2.3
Discrete features
Let us now consider the case of discrete feature values xi. For simplicity, we
begin by looking at binary feature values xi ∈{0, 1} and discuss the extension to
more general discrete features shortly. If there are D inputs, then a general distribu-
tion would correspond to a table of 2D numbers for each class, containing 2D −1
independent variables (due to the summation constraint). Because this grows expo-
nentially with the number of features, we might seek a more restricted representa-
tion. Here we will make the naive Bayes assumption in which the feature values are
Section 8.2.2
treated as independent, conditioned on the class Ck. Thus we have class-conditional
distributions of the form
p(x|Ck) =
D

i=1
µxi
ki(1 −µki)1−xi
(4.81)
which contain D independent parameters for each class. Substituting into (4.63) then
gives
ak(x) =
D

i=1
{xi ln µki + (1 −xi) ln(1 −µki)} + ln p(Ck)
(4.82)
which again are linear functions of the input values xi. For the case of K = 2 classes,
we can alternatively consider the logistic sigmoid formulation given by (4.57). Anal-
ogous results are obtained for discrete variables each of which can take M > 2
states.
Exercise 4.11
4.2.4
Exponential family
As we have seen, for both Gaussian distributed and discrete inputs, the posterior
class probabilities are given by generalized linear models with logistic sigmoid (K =


---
**Page 220**
4.3. Probabilistic Discriminative Models
203
2 classes) or softmax (K ⩾2 classes) activation functions. These are particular cases
of a more general result obtained by assuming that the class-conditional densities
p(x|Ck) are members of the exponential family of distributions.
Using the form (2.194) for members of the exponential family, we see that the
distribution of x can be written in the form
p(x|λk) = h(x)g(λk) exp 
λT
k u(x)
.
(4.83)
We now restrict attention to the subclass of such distributions for which u(x) = x.
Then we make use of (2.236) to introduce a scaling parameter s, so that we obtain
the restricted set of exponential family class-conditional densities of the form
p(x|λk, s) = 1
sh
1
sx

g(λk) exp
1
sλT
k x

.
(4.84)
Note that we are allowing each class to have its own parameter vector λk but we are
assuming that the classes share the same scale parameter s.
For the two-class problem, we substitute this expression for the class-conditional
densities into (4.58) and we see that the posterior class probability is again given by
a logistic sigmoid acting on a linear function a(x) which is given by
a(x) = (λ1 −λ2)Tx + ln g(λ1) −ln g(λ2) + ln p(C1) −ln p(C2).
(4.85)
Similarly, for the K-class problem, we substitute the class-conditional density ex-
pression into (4.63) to give
ak(x) = λT
k x + ln g(λk) + ln p(Ck)
(4.86)
and so again is a linear function of x.
4.3. Probabilistic Discriminative Models
For the two-class classiﬁcation problem, we have seen that the posterior probability
of class C1 can be written as a logistic sigmoid acting on a linear function of x, for a
wide choice of class-conditional distributions p(x|Ck). Similarly, for the multiclass
case, the posterior probability of class Ck is given by a softmax transformation of a
linear function of x. For speciﬁc choices of the class-conditional densities p(x|Ck),
we have used maximum likelihood to determine the parameters of the densities as
well as the class priors p(Ck) and then used Bayes’ theorem to ﬁnd the posterior class
probabilities.
However, an alternative approach is to use the functional form of the generalized
linear model explicitly and to determine its parameters directly by using maximum
likelihood. We shall see that there is an efﬁcient algorithm ﬁnding such solutions
known as iterative reweighted least squares, or IRLS.
The indirect approach to ﬁnding the parameters of a generalized linear model,
by ﬁtting class-conditional densities and class priors separately and then applying


---
**Page 221**
204
4. LINEAR MODELS FOR CLASSIFICATION
x1
x2
−1
0
1
−1
0
1
φ1
φ2
0
0.5
1
0
0.5
1
Figure 4.12
Illustration of the role of nonlinear basis functions in linear classiﬁcation models. The left plot
shows the original input space (x1, x2) together with data points from two classes labelled red and blue. Two
‘Gaussian’ basis functions φ1(x) and φ2(x) are deﬁned in this space with centres shown by the green crosses
and with contours shown by the green circles. The right-hand plot shows the corresponding feature space
(φ1, φ2) together with the linear decision boundary obtained given by a logistic regression model of the form
discussed in Section 4.3.2.
This corresponds to a nonlinear decision boundary in the original input space,
shown by the black curve in the left-hand plot.
Bayes’ theorem, represents an example of generative modelling, because we could
take such a model and generate synthetic data by drawing values of x from the
marginal distribution p(x). In the direct approach, we are maximizing a likelihood
function deﬁned through the conditional distribution p(Ck|x), which represents a
form of discriminative training. One advantage of the discriminative approach is
that there will typically be fewer adaptive parameters to be determined, as we shall
see shortly. It may also lead to improved predictive performance, particularly when
the class-conditional density assumptions give a poor approximation to the true dis-
tributions.
4.3.1
Fixed basis functions
So far in this chapter, we have considered classiﬁcation models that work di-
rectly with the original input vector x. However, all of the algorithms are equally
applicable if we ﬁrst make a ﬁxed nonlinear transformation of the inputs using a
vector of basis functions φ(x). The resulting decision boundaries will be linear in
the feature space φ, and these correspond to nonlinear decision boundaries in the
original x space, as illustrated in Figure 4.12. Classes that are linearly separable
in the feature space φ(x) need not be linearly separable in the original observation
space x. Note that as in our discussion of linear models for regression, one of the


---
**Page 222**
4.3. Probabilistic Discriminative Models
205
basis functions is typically set to a constant, say φ0(x) = 1, so that the correspond-
ing parameter w0 plays the role of a bias. For the remainder of this chapter, we shall
include a ﬁxed basis function transformation φ(x), as this will highlight some useful
similarities to the regression models discussed in Chapter 3.
For many problems of practical interest, there is signiﬁcant overlap between
the class-conditional densities p(x|Ck). This corresponds to posterior probabilities
p(Ck|x), which, for at least some values of x, are not 0 or 1. In such cases, the opti-
mal solution is obtained by modelling the posterior probabilities accurately and then
applying standard decision theory, as discussed in Chapter 1. Note that nonlinear
transformations φ(x) cannot remove such class overlap. Indeed, they can increase
the level of overlap, or create overlap where none existed in the original observation
space. However, suitable choices of nonlinearity can make the process of modelling
the posterior probabilities easier.
Such ﬁxed basis function models have important limitations, and these will be
Section 3.6
resolved in later chapters by allowing the basis functions themselves to adapt to the
data. Notwithstanding these limitations, models with ﬁxed nonlinear basis functions
play an important role in applications, and a discussion of such models will intro-
duce many of the key concepts needed for an understanding of their more complex
counterparts.
4.3.2
Logistic regression
We begin our treatment of generalized linear models by considering the problem
of two-class classiﬁcation. In our discussion of generative approaches in Section 4.2,
we saw that under rather general assumptions, the posterior probability of class C1
can be written as a logistic sigmoid acting on a linear function of the feature vector
φ so that
p(C1|φ) = y(φ) = σ

wTφ

(4.87)
with p(C2|φ) = 1 −p(C1|φ). Here σ(·) is the logistic sigmoid function deﬁned by
(4.59). In the terminology of statistics, this model is known as logistic regression,
although it should be emphasized that this is a model for classiﬁcation rather than
regression.
For an M-dimensional feature space φ, this model has M adjustable parameters.
By contrast, if we had ﬁtted Gaussian class conditional densities using maximum
likelihood, we would have used 2M parameters for the means and M(M + 1)/2
parameters for the (shared) covariance matrix. Together with the class prior p(C1),
this gives a total of M(M +5)/2+1 parameters, which grows quadratically with M,
in contrast to the linear dependence on M of the number of parameters in logistic
regression. For large values of M, there is a clear advantage in working with the
logistic regression model directly.
We now use maximum likelihood to determine the parameters of the logistic
regression model. To do this, we shall make use of the derivative of the logistic sig-
moid function, which can conveniently be expressed in terms of the sigmoid function
itself
Exercise 4.12
dσ
da = σ(1 −σ).
(4.88)


---
**Page 223**
206
4. LINEAR MODELS FOR CLASSIFICATION
For a data set {φn, tn}, where tn ∈{0, 1} and φn = φ(xn), with n =
1, . . . , N, the likelihood function can be written
p(t|w) =
N

n=1
ytn
n {1 −yn}1−tn
(4.89)
where t = (t1, . . . , tN)T and yn = p(C1|φn). As usual, we can deﬁne an error
function by taking the negative logarithm of the likelihood, which gives the cross-
entropy error function in the form
E(w) = −ln p(t|w) = −
N

n=1
{tn ln yn + (1 −tn) ln(1 −yn)}
(4.90)
where yn = σ(an) and an = wTφn. Taking the gradient of the error function with
respect to w, we obtain
Exercise 4.13
∇E(w) =
N

n=1
(yn −tn)φn
(4.91)
where we have made use of (4.88). We see that the factor involving the derivative
of the logistic sigmoid has cancelled, leading to a simpliﬁed form for the gradient
of the log likelihood. In particular, the contribution to the gradient from data point
n is given by the ‘error’ yn −tn between the target value and the prediction of the
model, times the basis function vector φn. Furthermore, comparison with (3.13)
shows that this takes precisely the same form as the gradient of the sum-of-squares
error function for the linear regression model.
Section 3.1.1
If desired, we could make use of the result (4.91) to give a sequential algorithm
in which patterns are presented one at a time, in which each of the weight vectors is
updated using (3.22) in which ∇En is the nth term in (4.91).
It is worth noting that maximum likelihood can exhibit severe over-ﬁtting for
data sets that are linearly separable. This arises because the maximum likelihood so-
lution occurs when the hyperplane corresponding to σ = 0.5, equivalent to wTφ =
0, separates the two classes and the magnitude of w goes to inﬁnity. In this case, the
logistic sigmoid function becomes inﬁnitely steep in feature space, corresponding to
a Heaviside step function, so that every training point from each class k is assigned
a posterior probability p(Ck|x) = 1. Furthermore, there is typically a continuum
Exercise 4.14
of such solutions because any separating hyperplane will give rise to the same pos-
terior probabilities at the training data points, as will be seen later in Figure 10.13.
Maximum likelihood provides no way to favour one such solution over another, and
which solution is found in practice will depend on the choice of optimization algo-
rithm and on the parameter initialization. Note that the problem will arise even if
the number of data points is large compared with the number of parameters in the
model, so long as the training data set is linearly separable. The singularity can be
avoided by inclusion of a prior and ﬁnding a MAP solution for w, or equivalently by
adding a regularization term to the error function.


---
**Page 224**
4.3. Probabilistic Discriminative Models
207
4.3.3
Iterative reweighted least squares
In the case of the linear regression models discussed in Chapter 3, the maxi-
mum likelihood solution, on the assumption of a Gaussian noise model, leads to a
closed-form solution. This was a consequence of the quadratic dependence of the
log likelihood function on the parameter vector w. For logistic regression, there
is no longer a closed-form solution, due to the nonlinearity of the logistic sigmoid
function. However, the departure from a quadratic form is not substantial. To be
precise, the error function is concave, as we shall see shortly, and hence has a unique
minimum. Furthermore, the error function can be minimized by an efﬁcient iterative
technique based on the Newton-Raphson iterative optimization scheme, which uses a
local quadratic approximation to the log likelihood function. The Newton-Raphson
update, for minimizing a function E(w), takes the form (Fletcher, 1987; Bishop and
Nabney, 2008)
w(new) = w(old) −H−1∇E(w).
(4.92)
where H is the Hessian matrix whose elements comprise the second derivatives of
E(w) with respect to the components of w.
Let us ﬁrst of all apply the Newton-Raphson method to the linear regression
model (3.3) with the sum-of-squares error function (3.12). The gradient and Hessian
of this error function are given by
∇E(w)
=
N

n=1
(wTφn −tn)φn = ΦTΦw −ΦTt
(4.93)
H = ∇∇E(w)
=
N

n=1
φnφT
n = ΦTΦ
(4.94)
where Φ is the N × M design matrix, whose nth row is given by φT
n. The Newton-
Section 3.1.1
Raphson update then takes the form
w(new)
=
w(old) −(ΦTΦ)−1 
ΦTΦw(old) −ΦTt
=
(ΦTΦ)−1ΦTt
(4.95)
which we recognize as the standard least-squares solution. Note that the error func-
tion in this case is quadratic and hence the Newton-Raphson formula gives the exact
solution in one step.
Now let us apply the Newton-Raphson update to the cross-entropy error function
(4.90) for the logistic regression model. From (4.91) we see that the gradient and
Hessian of this error function are given by
∇E(w)
=
N

n=1
(yn −tn)φn = ΦT(y −t)
(4.96)
H
=
∇∇E(w) =
N

n=1
yn(1 −yn)φnφT
n = ΦTRΦ
(4.97)

