# 01 - Introduction
*Pages 1-66 from Pattern Recognition and Machine Learning*

---
**Page 1**
Information Science and Statistics
Series Editors:
M. Jordan
J. Kleinberg
B. Scho¨lkopf


---
**Page 2**
Information Science and Statistics 
Akaike and Kitagawa: The Practice of Time Series Analysis. 
Bishop:  Pattern Recognition and Machine Learning. 
Cowell, Dawid, Lauritzen, and Spiegelhalter: Probabilistic Networks and
Expert Systems. 
Doucet, de Freitas, and Gordon: Sequential Monte Carlo Methods in Practice. 
Fine: Feedforward Neural Network Methodology. 
Hawkins and Olwell: Cumulative Sum Charts and Charting for Quality Improvement. 
Jensen: Bayesian Networks and Decision Graphs. 
Marchette: Computer Intrusion Detection and Network Monitoring:
A Statistical Viewpoint. 
Rubinstein and Kroese: The Cross-Entropy Method:  A Unified Approach to 
Combinatorial Optimization, Monte Carlo Simulation, and Machine Learning. 
Studený: Probabilistic Conditional Independence Structures.
Vapnik: The Nature of Statistical Learning Theory, Second Edition.  
Wallace: Statistical and Inductive Inference by Minimum Massage Length. 


---
**Page 3**
Christopher M. Bishop
Pattern Recognition and
Machine Learning


---
**Page 4**
Christopher M. Bishop F.R.Eng.
Assistant Director
Microsoft Research Ltd
Cambridge CB3 0FB, U.K.
cmbishop@microsoft.com
http://research.microsoft.com/cmbishop
Series Editors
Michael Jordan
Department of Computer
Science and Department
of Statistics
University of California,
Berkeley
Berkeley, CA 94720
USA
Professor Jon Kleinberg
Department of Computer
Science
Cornell University
Ithaca, NY 14853
USA
Bernhard Scho¨lkopf
Max Planck Institute for
Biological Cybernetics
Spemannstrasse 38
72076 Tu¨bingen
Germany
Library of Congress Control Number: 2006922522
ISBN-10: 0-387-31073-8
ISBN-13: 978-0387-31073-2
Printed on acid-free paper.
© 2006 Springer Science+Business Media, LLC
All rights reserved. This work may not be translated or copied in whole or in part without the written permission of the publisher
(Springer Science+Business Media, LLC, 233 Spring Street, New York, NY 10013, USA), except for brief excerpts in connection
with reviews or scholarly analysis. Use in connection with any form of information storage and retrieval, electronic adaptation,
computer software, or by similar or dissimilar methodology now known or hereafter developed is forbidden.
The use in this publication of trade names, trademarks, service marks, and similar terms, even if they are not identified as such,
is not to be taken as an expression of opinion as to whether or not they are subject to proprietary rights.
Printed in Singapore.
(KYO)
9 8 7 6 5 4 3 2 1
springer.com


---
**Page 5**
This book is dedicated to my family:
Jenna, Mark, and Hugh
Total eclipse of the sun, Antalya, Turkey, 29 March 2006.


---
**Page 6**
Preface
Pattern recognition has its origins in engineering, whereas machine learning grew
out of computer science. However, these activities can be viewed as two facets of
the same ﬁeld, and together they have undergone substantial development over the
past ten years. In particular, Bayesian methods have grown from a specialist niche to
become mainstream, while graphical models have emerged as a general framework
for describing and applying probabilistic models. Also, the practical applicability of
Bayesian methods has been greatly enhanced through the development of a range of
approximate inference algorithms such as variational Bayes and expectation propa-
gation. Similarly, new models based on kernels have had signiﬁcant impact on both
algorithms and applications.
This new textbook reﬂects these recent developments while providing a compre-
hensive introduction to the ﬁelds of pattern recognition and machine learning. It is
aimed at advanced undergraduates or ﬁrst year PhD students, as well as researchers
and practitioners, and assumes no previous knowledge of pattern recognition or ma-
chine learning concepts. Knowledge of multivariate calculus and basic linear algebra
is required, and some familiarity with probabilities would be helpful though not es-
sential as the book includes a self-contained introduction to basic probability theory.
Because this book has broad scope, it is impossible to provide a complete list of
references, and in particular no attempt has been made to provide accurate historical
attribution of ideas. Instead, the aim has been to give references that offer greater
detail than is possible here and that hopefully provide entry points into what, in some
cases, is a very extensive literature. For this reason, the references are often to more
recent textbooks and review articles rather than to original sources.
The book is supported by a great deal of additional material, including lecture
slides as well as the complete set of ﬁgures used in the book, and the reader is
encouraged to visit the book web site for the latest information:
http://research.microsoft.com/∼cmbishop/PRML
vii


---
**Page 7**
viii
PREFACE
Exercises
The exercises that appear at the end of every chapter form an important com-
ponent of the book. Each exercise has been carefully chosen to reinforce concepts
explained in the text or to develop and generalize them in signiﬁcant ways, and each
is graded according to difﬁculty ranging from (⋆), which denotes a simple exercise
taking a few minutes to complete, through to (⋆⋆⋆), which denotes a signiﬁcantly
more complex exercise.
It has been difﬁcult to know to what extent these solutions should be made
widely available. Those engaged in self study will ﬁnd worked solutions very ben-
eﬁcial, whereas many course tutors request that solutions be available only via the
publisher so that the exercises may be used in class. In order to try to meet these
conﬂicting requirements, those exercises that help amplify key points in the text, or
that ﬁll in important details, have solutions that are available as a PDF ﬁle from the
book web site. Such exercises are denoted by www . Solutions for the remaining
exercises are available to course tutors by contacting the publisher (contact details
are given on the book web site). Readers are strongly encouraged to work through
the exercises unaided, and to turn to the solutions only as required.
Although this book focuses on concepts and principles, in a taught course the
students should ideally have the opportunity to experiment with some of the key
algorithms using appropriate data sets. A companion volume (Bishop and Nabney,
2008) will deal with practical aspects of pattern recognition and machine learning,
and will be accompanied by Matlab software implementing most of the algorithms
discussed in this book.
Acknowledgements
First of all I would like to express my sincere thanks to Markus Svens´en who
has provided immense help with preparation of ﬁgures and with the typesetting of
the book in LATEX. His assistance has been invaluable.
I am very grateful to Microsoft Research for providing a highly stimulating re-
search environment and for giving me the freedom to write this book (the views and
opinions expressed in this book, however, are my own and are therefore not neces-
sarily the same as those of Microsoft or its afﬁliates).
Springer has provided excellent support throughout the ﬁnal stages of prepara-
tion of this book, and I would like to thank my commissioning editor John Kimmel
for his support and professionalism, as well as Joseph Piliero for his help in design-
ing the cover and the text format and MaryAnn Brickner for her numerous contribu-
tions during the production phase. The inspiration for the cover design came from a
discussion with Antonio Criminisi.
I also wish to thank Oxford University Press for permission to reproduce ex-
cerpts from an earlier textbook, Neural Networks for Pattern Recognition (Bishop,
1995a). The images of the Mark 1 perceptron and of Frank Rosenblatt are repro-
duced with the permission of Arvin Calspan Advanced Technology Center. I would
also like to thank Asela Gunawardana for plotting the spectrogram in Figure 13.1,
and Bernhard Sch¨olkopf for permission to use his kernel PCA code to plot Fig-
ure 12.17.


---
**Page 8**
PREFACE
ix
Many people have helped by proofreading draft material and providing com-
ments and suggestions, including Shivani Agarwal, C´edric Archambeau, Arik Azran,
Andrew Blake, Hakan Cevikalp, Michael Fourman, Brendan Frey, Zoubin Ghahra-
mani, Thore Graepel, Katherine Heller, Ralf Herbrich, Geoffrey Hinton, Adam Jo-
hansen, Matthew Johnson, Michael Jordan, Eva Kalyvianaki, Anitha Kannan, Julia
Lasserre, David Liu, Tom Minka, Ian Nabney, Tonatiuh Pena, Yuan Qi, Sam Roweis,
Balaji Sanjiya, Toby Sharp, Ana Costa e Silva, David Spiegelhalter, Jay Stokes, Tara
Symeonides, Martin Szummer, Marshall Tappen, Ilkay Ulusoy, Chris Williams, John
Winn, and Andrew Zisserman.
Finally, I would like to thank my wife Jenna who has been hugely supportive
throughout the several years it has taken to write this book.
Chris Bishop
Cambridge
February 2006


---
**Page 9**
Mathematical notation
I have tried to keep the mathematical content of the book to the minimum neces-
sary to achieve a proper understanding of the ﬁeld. However, this minimum level is
nonzero, and it should be emphasized that a good grasp of calculus, linear algebra,
and probability theory is essential for a clear understanding of modern pattern recog-
nition and machine learning techniques. Nevertheless, the emphasis in this book is
on conveying the underlying concepts rather than on mathematical rigour.
I have tried to use a consistent notation throughout the book, although at times
this means departing from some of the conventions used in the corresponding re-
search literature. Vectors are denoted by lower case bold Roman letters such as
x, and all vectors are assumed to be column vectors. A superscript T denotes the
transpose of a matrix or vector, so that xT will be a row vector. Uppercase bold
roman letters, such as M, denote matrices. The notation (w1, . . . , wM) denotes a
row vector with M elements, while the corresponding column vector is written as
w = (w1, . . . , wM)T.
The notation [a, b] is used to denote the closed interval from a to b, that is the
interval including the values a and b themselves, while (a, b) denotes the correspond-
ing open interval, that is the interval excluding a and b. Similarly, [a, b) denotes an
interval that includes a but excludes b. For the most part, however, there will be
little need to dwell on such reﬁnements as whether the end points of an interval are
included or not.
The M × M identity matrix (also known as the unit matrix) is denoted IM,
which will be abbreviated to I where there is no ambiguity about it dimensionality.
It has elements Iij that equal 1 if i = j and 0 if i ̸= j.
A functional is denoted f[y] where y(x) is some function. The concept of a
functional is discussed in Appendix D.
The notation g(x) = O(f(x)) denotes that |f(x)/g(x)| is bounded as x →∞.
For instance if g(x) = 3x2 + 2, then g(x) = O(x2).
The expectation of a function f(x, y) with respect to a random variable x is de-
noted by Ex[f(x, y)]. In situations where there is no ambiguity as to which variable
is being averaged over, this will be simpliﬁed by omitting the sufﬁx, for instance
xi


---
**Page 10**
xii
MATHEMATICAL NOTATION
E[x]. If the distribution of x is conditioned on another variable z, then the corre-
sponding conditional expectation will be written Ex[f(x)|z]. Similarly, the variance
is denoted var[f(x)], and for vector variables the covariance is written cov[x, y]. We
shall also use cov[x] as a shorthand notation for cov[x, x]. The concepts of expecta-
tions and covariances are introduced in Section 1.2.2.
If we have N values x1, . . . , xN of a D-dimensional vector x = (x1, . . . , xD)T,
we can combine the observations into a data matrix X in which the nth row of X
corresponds to the row vector xT
n. Thus the n, i element of X corresponds to the
ith element of the nth observation xn. For the case of one-dimensional variables we
shall denote such a matrix by x, which is a column vector whose nth element is xn.
Note that x (which has dimensionality N) uses a different typeface to distinguish it
from x (which has dimensionality D).


---
**Page 11**
Contents
Preface
vii
Mathematical notation
xi
Contents
xiii
1
Introduction
1
1.1
Example: Polynomial Curve Fitting . . . . . . . . . . . . . . . . .
4
1.2
Probability Theory . . . . . . . . . . . . . . . . . . . . . . . . . .
12
1.2.1
Probability densities
. . . . . . . . . . . . . . . . . . . . .
17
1.2.2
Expectations and covariances
. . . . . . . . . . . . . . . .
19
1.2.3
Bayesian probabilities
. . . . . . . . . . . . . . . . . . . .
21
1.2.4
The Gaussian distribution
. . . . . . . . . . . . . . . . . .
24
1.2.5
Curve ﬁtting re-visited . . . . . . . . . . . . . . . . . . . .
28
1.2.6
Bayesian curve ﬁtting
. . . . . . . . . . . . . . . . . . . .
30
1.3
Model Selection
. . . . . . . . . . . . . . . . . . . . . . . . . . .
32
1.4
The Curse of Dimensionality . . . . . . . . . . . . . . . . . . . . .
33
1.5
Decision Theory . . . . . . . . . . . . . . . . . . . . . . . . . . .
38
1.5.1
Minimizing the misclassiﬁcation rate
. . . . . . . . . . . .
39
1.5.2
Minimizing the expected loss
. . . . . . . . . . . . . . . .
41
1.5.3
The reject option . . . . . . . . . . . . . . . . . . . . . . .
42
1.5.4
Inference and decision . . . . . . . . . . . . . . . . . . . .
42
1.5.5
Loss functions for regression . . . . . . . . . . . . . . . . .
46
1.6
Information Theory . . . . . . . . . . . . . . . . . . . . . . . . . .
48
1.6.1
Relative entropy and mutual information
. . . . . . . . . .
55
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
58
xiii


---
**Page 12**
xiv
CONTENTS
2
Probability Distributions
67
2.1
Binary Variables . . . . . . . . . . . . . . . . . . . . . . . . . . .
68
2.1.1
The beta distribution . . . . . . . . . . . . . . . . . . . . .
71
2.2
Multinomial Variables . . . . . . . . . . . . . . . . . . . . . . . .
74
2.2.1
The Dirichlet distribution . . . . . . . . . . . . . . . . . . .
76
2.3
The Gaussian Distribution . . . . . . . . . . . . . . . . . . . . . .
78
2.3.1
Conditional Gaussian distributions . . . . . . . . . . . . . .
85
2.3.2
Marginal Gaussian distributions . . . . . . . . . . . . . . .
88
2.3.3
Bayes’ theorem for Gaussian variables . . . . . . . . . . . .
90
2.3.4
Maximum likelihood for the Gaussian . . . . . . . . . . . .
93
2.3.5
Sequential estimation . . . . . . . . . . . . . . . . . . . . .
94
2.3.6
Bayesian inference for the Gaussian . . . . . . . . . . . . .
97
2.3.7
Student’s t-distribution . . . . . . . . . . . . . . . . . . . .
102
2.3.8
Periodic variables . . . . . . . . . . . . . . . . . . . . . . .
105
2.3.9
Mixtures of Gaussians . . . . . . . . . . . . . . . . . . . .
110
2.4
The Exponential Family . . . . . . . . . . . . . . . . . . . . . . .
113
2.4.1
Maximum likelihood and sufﬁcient statistics
. . . . . . . .
116
2.4.2
Conjugate priors
. . . . . . . . . . . . . . . . . . . . . . .
117
2.4.3
Noninformative priors
. . . . . . . . . . . . . . . . . . . .
117
2.5
Nonparametric Methods . . . . . . . . . . . . . . . . . . . . . . .
120
2.5.1
Kernel density estimators . . . . . . . . . . . . . . . . . . .
122
2.5.2
Nearest-neighbour methods
. . . . . . . . . . . . . . . . .
124
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
127
3
Linear Models for Regression
137
3.1
Linear Basis Function Models . . . . . . . . . . . . . . . . . . . .
138
3.1.1
Maximum likelihood and least squares . . . . . . . . . . . .
140
3.1.2
Geometry of least squares
. . . . . . . . . . . . . . . . . .
143
3.1.3
Sequential learning . . . . . . . . . . . . . . . . . . . . . .
143
3.1.4
Regularized least squares . . . . . . . . . . . . . . . . . . .
144
3.1.5
Multiple outputs
. . . . . . . . . . . . . . . . . . . . . . .
146
3.2
The Bias-Variance Decomposition . . . . . . . . . . . . . . . . . .
147
3.3
Bayesian Linear Regression . . . . . . . . . . . . . . . . . . . . .
152
3.3.1
Parameter distribution
. . . . . . . . . . . . . . . . . . . .
152
3.3.2
Predictive distribution
. . . . . . . . . . . . . . . . . . . .
156
3.3.3
Equivalent kernel . . . . . . . . . . . . . . . . . . . . . . .
159
3.4
Bayesian Model Comparison . . . . . . . . . . . . . . . . . . . . .
161
3.5
The Evidence Approximation
. . . . . . . . . . . . . . . . . . . .
165
3.5.1
Evaluation of the evidence function . . . . . . . . . . . . .
166
3.5.2
Maximizing the evidence function . . . . . . . . . . . . . .
168
3.5.3
Effective number of parameters
. . . . . . . . . . . . . . .
170
3.6
Limitations of Fixed Basis Functions
. . . . . . . . . . . . . . . .
172
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
173


---
**Page 13**
CONTENTS
xv
4
Linear Models for Classiﬁcation
179
4.1
Discriminant Functions . . . . . . . . . . . . . . . . . . . . . . . .
181
4.1.1
Two classes . . . . . . . . . . . . . . . . . . . . . . . . . .
181
4.1.2
Multiple classes . . . . . . . . . . . . . . . . . . . . . . . .
182
4.1.3
Least squares for classiﬁcation . . . . . . . . . . . . . . . .
184
4.1.4
Fisher’s linear discriminant . . . . . . . . . . . . . . . . . .
186
4.1.5
Relation to least squares . . . . . . . . . . . . . . . . . . .
189
4.1.6
Fisher’s discriminant for multiple classes
. . . . . . . . . .
191
4.1.7
The perceptron algorithm . . . . . . . . . . . . . . . . . . .
192
4.2
Probabilistic Generative Models . . . . . . . . . . . . . . . . . . .
196
4.2.1
Continuous inputs
. . . . . . . . . . . . . . . . . . . . . .
198
4.2.2
Maximum likelihood solution
. . . . . . . . . . . . . . . .
200
4.2.3
Discrete features . . . . . . . . . . . . . . . . . . . . . . .
202
4.2.4
Exponential family . . . . . . . . . . . . . . . . . . . . . .
202
4.3
Probabilistic Discriminative Models . . . . . . . . . . . . . . . . .
203
4.3.1
Fixed basis functions . . . . . . . . . . . . . . . . . . . . .
204
4.3.2
Logistic regression . . . . . . . . . . . . . . . . . . . . . .
205
4.3.3
Iterative reweighted least squares
. . . . . . . . . . . . . .
207
4.3.4
Multiclass logistic regression . . . . . . . . . . . . . . . . .
209
4.3.5
Probit regression . . . . . . . . . . . . . . . . . . . . . . .
210
4.3.6
Canonical link functions . . . . . . . . . . . . . . . . . . .
212
4.4
The Laplace Approximation . . . . . . . . . . . . . . . . . . . . .
213
4.4.1
Model comparison and BIC
. . . . . . . . . . . . . . . . .
216
4.5
Bayesian Logistic Regression
. . . . . . . . . . . . . . . . . . . .
217
4.5.1
Laplace approximation . . . . . . . . . . . . . . . . . . . .
217
4.5.2
Predictive distribution
. . . . . . . . . . . . . . . . . . . .
218
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
220
5
Neural Networks
225
5.1
Feed-forward Network Functions
. . . . . . . . . . . . . . . . . .
227
5.1.1
Weight-space symmetries
. . . . . . . . . . . . . . . . . .
231
5.2
Network Training . . . . . . . . . . . . . . . . . . . . . . . . . . .
232
5.2.1
Parameter optimization . . . . . . . . . . . . . . . . . . . .
236
5.2.2
Local quadratic approximation . . . . . . . . . . . . . . . .
237
5.2.3
Use of gradient information
. . . . . . . . . . . . . . . . .
239
5.2.4
Gradient descent optimization . . . . . . . . . . . . . . . .
240
5.3
Error Backpropagation . . . . . . . . . . . . . . . . . . . . . . . .
241
5.3.1
Evaluation of error-function derivatives . . . . . . . . . . .
242
5.3.2
A simple example
. . . . . . . . . . . . . . . . . . . . . .
245
5.3.3
Efﬁciency of backpropagation . . . . . . . . . . . . . . . .
246
5.3.4
The Jacobian matrix
. . . . . . . . . . . . . . . . . . . . .
247
5.4
The Hessian Matrix . . . . . . . . . . . . . . . . . . . . . . . . . .
249
5.4.1
Diagonal approximation . . . . . . . . . . . . . . . . . . .
250
5.4.2
Outer product approximation . . . . . . . . . . . . . . . . .
251
5.4.3
Inverse Hessian . . . . . . . . . . . . . . . . . . . . . . . .
252


---
**Page 14**
xvi
CONTENTS
5.4.4
Finite differences . . . . . . . . . . . . . . . . . . . . . . .
252
5.4.5
Exact evaluation of the Hessian
. . . . . . . . . . . . . . .
253
5.4.6
Fast multiplication by the Hessian . . . . . . . . . . . . . .
254
5.5
Regularization in Neural Networks
. . . . . . . . . . . . . . . . .
256
5.5.1
Consistent Gaussian priors . . . . . . . . . . . . . . . . . .
257
5.5.2
Early stopping
. . . . . . . . . . . . . . . . . . . . . . . .
259
5.5.3
Invariances . . . . . . . . . . . . . . . . . . . . . . . . . .
261
5.5.4
Tangent propagation . . . . . . . . . . . . . . . . . . . . .
263
5.5.5
Training with transformed data . . . . . . . . . . . . . . . .
265
5.5.6
Convolutional networks
. . . . . . . . . . . . . . . . . . .
267
5.5.7
Soft weight sharing . . . . . . . . . . . . . . . . . . . . . .
269
5.6
Mixture Density Networks . . . . . . . . . . . . . . . . . . . . . .
272
5.7
Bayesian Neural Networks . . . . . . . . . . . . . . . . . . . . . .
277
5.7.1
Posterior parameter distribution . . . . . . . . . . . . . . .
278
5.7.2
Hyperparameter optimization
. . . . . . . . . . . . . . . .
280
5.7.3
Bayesian neural networks for classiﬁcation . . . . . . . . .
281
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
284
6
Kernel Methods
291
6.1
Dual Representations . . . . . . . . . . . . . . . . . . . . . . . . .
293
6.2
Constructing Kernels . . . . . . . . . . . . . . . . . . . . . . . . .
294
6.3
Radial Basis Function Networks . . . . . . . . . . . . . . . . . . .
299
6.3.1
Nadaraya-Watson model . . . . . . . . . . . . . . . . . . .
301
6.4
Gaussian Processes . . . . . . . . . . . . . . . . . . . . . . . . . .
303
6.4.1
Linear regression revisited . . . . . . . . . . . . . . . . . .
304
6.4.2
Gaussian processes for regression . . . . . . . . . . . . . .
306
6.4.3
Learning the hyperparameters . . . . . . . . . . . . . . . .
311
6.4.4
Automatic relevance determination
. . . . . . . . . . . . .
312
6.4.5
Gaussian processes for classiﬁcation . . . . . . . . . . . . .
313
6.4.6
Laplace approximation . . . . . . . . . . . . . . . . . . . .
315
6.4.7
Connection to neural networks . . . . . . . . . . . . . . . .
319
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
320
7
Sparse Kernel Machines
325
7.1
Maximum Margin Classiﬁers
. . . . . . . . . . . . . . . . . . . .
326
7.1.1
Overlapping class distributions . . . . . . . . . . . . . . . .
331
7.1.2
Relation to logistic regression . . . . . . . . . . . . . . . .
336
7.1.3
Multiclass SVMs . . . . . . . . . . . . . . . . . . . . . . .
338
7.1.4
SVMs for regression . . . . . . . . . . . . . . . . . . . . .
339
7.1.5
Computational learning theory . . . . . . . . . . . . . . . .
344
7.2
Relevance Vector Machines
. . . . . . . . . . . . . . . . . . . . .
345
7.2.1
RVM for regression . . . . . . . . . . . . . . . . . . . . . .
345
7.2.2
Analysis of sparsity . . . . . . . . . . . . . . . . . . . . . .
349
7.2.3
RVM for classiﬁcation . . . . . . . . . . . . . . . . . . . .
353
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
357


---
**Page 15**
CONTENTS
xvii
8
Graphical Models
359
8.1
Bayesian Networks . . . . . . . . . . . . . . . . . . . . . . . . . .
360
8.1.1
Example: Polynomial regression . . . . . . . . . . . . . . .
362
8.1.2
Generative models . . . . . . . . . . . . . . . . . . . . . .
365
8.1.3
Discrete variables . . . . . . . . . . . . . . . . . . . . . . .
366
8.1.4
Linear-Gaussian models . . . . . . . . . . . . . . . . . . .
370
8.2
Conditional Independence . . . . . . . . . . . . . . . . . . . . . .
372
8.2.1
Three example graphs
. . . . . . . . . . . . . . . . . . . .
373
8.2.2
D-separation
. . . . . . . . . . . . . . . . . . . . . . . . .
378
8.3
Markov Random Fields
. . . . . . . . . . . . . . . . . . . . . . .
383
8.3.1
Conditional independence properties . . . . . . . . . . . . .
383
8.3.2
Factorization properties
. . . . . . . . . . . . . . . . . . .
384
8.3.3
Illustration: Image de-noising
. . . . . . . . . . . . . . . .
387
8.3.4
Relation to directed graphs . . . . . . . . . . . . . . . . . .
390
8.4
Inference in Graphical Models . . . . . . . . . . . . . . . . . . . .
393
8.4.1
Inference on a chain
. . . . . . . . . . . . . . . . . . . . .
394
8.4.2
Trees
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
398
8.4.3
Factor graphs . . . . . . . . . . . . . . . . . . . . . . . . .
399
8.4.4
The sum-product algorithm . . . . . . . . . . . . . . . . . .
402
8.4.5
The max-sum algorithm
. . . . . . . . . . . . . . . . . . .
411
8.4.6
Exact inference in general graphs
. . . . . . . . . . . . . .
416
8.4.7
Loopy belief propagation . . . . . . . . . . . . . . . . . . .
417
8.4.8
Learning the graph structure . . . . . . . . . . . . . . . . .
418
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
418
9
Mixture Models and EM
423
9.1
K-means Clustering . . . . . . . . . . . . . . . . . . . . . . . . .
424
9.1.1
Image segmentation and compression . . . . . . . . . . . .
428
9.2
Mixtures of Gaussians . . . . . . . . . . . . . . . . . . . . . . . .
430
9.2.1
Maximum likelihood . . . . . . . . . . . . . . . . . . . . .
432
9.2.2
EM for Gaussian mixtures . . . . . . . . . . . . . . . . . .
435
9.3
An Alternative View of EM
. . . . . . . . . . . . . . . . . . . . .
439
9.3.1
Gaussian mixtures revisited
. . . . . . . . . . . . . . . . .
441
9.3.2
Relation to K-means . . . . . . . . . . . . . . . . . . . . .
443
9.3.3
Mixtures of Bernoulli distributions . . . . . . . . . . . . . .
444
9.3.4
EM for Bayesian linear regression . . . . . . . . . . . . . .
448
9.4
The EM Algorithm in General . . . . . . . . . . . . . . . . . . . .
450
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
455
10 Approximate Inference
461
10.1 Variational Inference . . . . . . . . . . . . . . . . . . . . . . . . .
462
10.1.1 Factorized distributions . . . . . . . . . . . . . . . . . . . .
464
10.1.2 Properties of factorized approximations . . . . . . . . . . .
466
10.1.3 Example: The univariate Gaussian . . . . . . . . . . . . . .
470
10.1.4 Model comparison . . . . . . . . . . . . . . . . . . . . . .
473
10.2 Illustration: Variational Mixture of Gaussians . . . . . . . . . . . .
474


---
**Page 16**
xviii
CONTENTS
10.2.1 Variational distribution . . . . . . . . . . . . . . . . . . . .
475
10.2.2 Variational lower bound
. . . . . . . . . . . . . . . . . . .
481
10.2.3 Predictive density . . . . . . . . . . . . . . . . . . . . . . .
482
10.2.4 Determining the number of components . . . . . . . . . . .
483
10.2.5 Induced factorizations
. . . . . . . . . . . . . . . . . . . .
485
10.3 Variational Linear Regression . . . . . . . . . . . . . . . . . . . .
486
10.3.1 Variational distribution . . . . . . . . . . . . . . . . . . . .
486
10.3.2 Predictive distribution
. . . . . . . . . . . . . . . . . . . .
488
10.3.3 Lower bound . . . . . . . . . . . . . . . . . . . . . . . . .
489
10.4 Exponential Family Distributions
. . . . . . . . . . . . . . . . . .
490
10.4.1 Variational message passing . . . . . . . . . . . . . . . . .
491
10.5 Local Variational Methods . . . . . . . . . . . . . . . . . . . . . .
493
10.6 Variational Logistic Regression
. . . . . . . . . . . . . . . . . . .
498
10.6.1 Variational posterior distribution . . . . . . . . . . . . . . .
498
10.6.2 Optimizing the variational parameters . . . . . . . . . . . .
500
10.6.3 Inference of hyperparameters
. . . . . . . . . . . . . . . .
502
10.7 Expectation Propagation . . . . . . . . . . . . . . . . . . . . . . .
505
10.7.1 Example: The clutter problem . . . . . . . . . . . . . . . .
511
10.7.2 Expectation propagation on graphs . . . . . . . . . . . . . .
513
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
517
11 Sampling Methods
523
11.1 Basic Sampling Algorithms
. . . . . . . . . . . . . . . . . . . . .
526
11.1.1 Standard distributions
. . . . . . . . . . . . . . . . . . . .
526
11.1.2 Rejection sampling . . . . . . . . . . . . . . . . . . . . . .
528
11.1.3 Adaptive rejection sampling . . . . . . . . . . . . . . . . .
530
11.1.4 Importance sampling . . . . . . . . . . . . . . . . . . . . .
532
11.1.5 Sampling-importance-resampling
. . . . . . . . . . . . . .
534
11.1.6 Sampling and the EM algorithm . . . . . . . . . . . . . . .
536
11.2 Markov Chain Monte Carlo
. . . . . . . . . . . . . . . . . . . . .
537
11.2.1 Markov chains
. . . . . . . . . . . . . . . . . . . . . . . .
539
11.2.2 The Metropolis-Hastings algorithm
. . . . . . . . . . . . .
541
11.3 Gibbs Sampling
. . . . . . . . . . . . . . . . . . . . . . . . . . .
542
11.4 Slice Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . .
546
11.5 The Hybrid Monte Carlo Algorithm . . . . . . . . . . . . . . . . .
548
11.5.1 Dynamical systems . . . . . . . . . . . . . . . . . . . . . .
548
11.5.2 Hybrid Monte Carlo
. . . . . . . . . . . . . . . . . . . . .
552
11.6 Estimating the Partition Function
. . . . . . . . . . . . . . . . . .
554
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
556
12 Continuous Latent Variables
559
12.1 Principal Component Analysis . . . . . . . . . . . . . . . . . . . .
561
12.1.1 Maximum variance formulation . . . . . . . . . . . . . . .
561
12.1.2 Minimum-error formulation . . . . . . . . . . . . . . . . .
563
12.1.3 Applications of PCA . . . . . . . . . . . . . . . . . . . . .
565
12.1.4 PCA for high-dimensional data
. . . . . . . . . . . . . . .
569


---
**Page 17**
CONTENTS
xix
12.2 Probabilistic PCA
. . . . . . . . . . . . . . . . . . . . . . . . . .
570
12.2.1 Maximum likelihood PCA . . . . . . . . . . . . . . . . . .
574
12.2.2 EM algorithm for PCA . . . . . . . . . . . . . . . . . . . .
577
12.2.3 Bayesian PCA
. . . . . . . . . . . . . . . . . . . . . . . .
580
12.2.4 Factor analysis . . . . . . . . . . . . . . . . . . . . . . . .
583
12.3
Kernel PCA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
586
12.4 Nonlinear Latent Variable Models . . . . . . . . . . . . . . . . . .
591
12.4.1 Independent component analysis . . . . . . . . . . . . . . .
591
12.4.2 Autoassociative neural networks . . . . . . . . . . . . . . .
592
12.4.3 Modelling nonlinear manifolds . . . . . . . . . . . . . . . .
595
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
599
13 Sequential Data
605
13.1
Markov Models . . . . . . . . . . . . . . . . . . . . . . . . . . . .
607
13.2 Hidden Markov Models
. . . . . . . . . . . . . . . . . . . . . . .
610
13.2.1 Maximum likelihood for the HMM
. . . . . . . . . . . . .
615
13.2.2 The forward-backward algorithm
. . . . . . . . . . . . . .
618
13.2.3 The sum-product algorithm for the HMM . . . . . . . . . .
625
13.2.4 Scaling factors . . . . . . . . . . . . . . . . . . . . . . . .
627
13.2.5 The Viterbi algorithm . . . . . . . . . . . . . . . . . . . . .
629
13.2.6 Extensions of the hidden Markov model . . . . . . . . . . .
631
13.3 Linear Dynamical Systems . . . . . . . . . . . . . . . . . . . . . .
635
13.3.1 Inference in LDS . . . . . . . . . . . . . . . . . . . . . . .
638
13.3.2 Learning in LDS . . . . . . . . . . . . . . . . . . . . . . .
642
13.3.3 Extensions of LDS . . . . . . . . . . . . . . . . . . . . . .
644
13.3.4 Particle ﬁlters . . . . . . . . . . . . . . . . . . . . . . . . .
645
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
646
14 Combining Models
653
14.1 Bayesian Model Averaging . . . . . . . . . . . . . . . . . . . . . .
654
14.2 Committees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
655
14.3 Boosting
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
657
14.3.1 Minimizing exponential error
. . . . . . . . . . . . . . . .
659
14.3.2 Error functions for boosting
. . . . . . . . . . . . . . . . .
661
14.4 Tree-based Models . . . . . . . . . . . . . . . . . . . . . . . . . .
663
14.5 Conditional Mixture Models . . . . . . . . . . . . . . . . . . . . .
666
14.5.1 Mixtures of linear regression models . . . . . . . . . . . . .
667
14.5.2 Mixtures of logistic models
. . . . . . . . . . . . . . . . .
670
14.5.3 Mixtures of experts . . . . . . . . . . . . . . . . . . . . . .
672
Exercises
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
674
Appendix A
Data Sets
677
Appendix B
Probability Distributions
685
Appendix C
Properties of Matrices
695


---
**Page 18**
xx
CONTENTS
Appendix D
Calculus of Variations
703
Appendix E
Lagrange Multipliers
707
References
711
Index
729


---
**Page 19**
1
Introduction
The problem of searching for patterns in data is a fundamental one and has a long and
successful history. For instance, the extensive astronomical observations of Tycho
Brahe in the 16th century allowed Johannes Kepler to discover the empirical laws of
planetary motion, which in turn provided a springboard for the development of clas-
sical mechanics. Similarly, the discovery of regularities in atomic spectra played a
key role in the development and veriﬁcation of quantum physics in the early twenti-
eth century. The ﬁeld of pattern recognition is concerned with the automatic discov-
ery of regularities in data through the use of computer algorithms and with the use of
these regularities to take actions such as classifying the data into different categories.
Consider the example of recognizing handwritten digits, illustrated in Figure 1.1.
Each digit corresponds to a 28×28 pixel image and so can be represented by a vector
x comprising 784 real numbers. The goal is to build a machine that will take such a
vector x as input and that will produce the identity of the digit 0, . . . , 9 as the output.
This is a nontrivial problem due to the wide variability of handwriting. It could be
1


---
**Page 20**
2
1. INTRODUCTION
Figure 1.1
Examples of hand-written dig-
its taken from US zip codes.
tackled using handcrafted rules or heuristics for distinguishing the digits based on
the shapes of the strokes, but in practice such an approach leads to a proliferation of
rules and of exceptions to the rules and so on, and invariably gives poor results.
Far better results can be obtained by adopting a machine learning approach in
which a large set of N digits {x1, . . . , xN} called a training set is used to tune the
parameters of an adaptive model. The categories of the digits in the training set
are known in advance, typically by inspecting them individually and hand-labelling
them. We can express the category of a digit using target vector t, which represents
the identity of the corresponding digit. Suitable techniques for representing cate-
gories in terms of vectors will be discussed later. Note that there is one such target
vector t for each digit image x.
The result of running the machine learning algorithm can be expressed as a
function y(x) which takes a new digit image x as input and that generates an output
vector y, encoded in the same way as the target vectors. The precise form of the
function y(x) is determined during the training phase, also known as the learning
phase, on the basis of the training data. Once the model is trained it can then de-
termine the identity of new digit images, which are said to comprise a test set. The
ability to categorize correctly new examples that differ from those used for train-
ing is known as generalization. In practical applications, the variability of the input
vectors will be such that the training data can comprise only a tiny fraction of all
possible input vectors, and so generalization is a central goal in pattern recognition.
For most practical applications, the original input variables are typically prepro-
cessed to transform them into some new space of variables where, it is hoped, the
pattern recognition problem will be easier to solve. For instance, in the digit recogni-
tion problem, the images of the digits are typically translated and scaled so that each
digit is contained within a box of a ﬁxed size. This greatly reduces the variability
within each digit class, because the location and scale of all the digits are now the
same, which makes it much easier for a subsequent pattern recognition algorithm
to distinguish between the different classes. This pre-processing stage is sometimes
also called feature extraction. Note that new test data must be pre-processed using
the same steps as the training data.
Pre-processing might also be performed in order to speed up computation. For
example, if the goal is real-time face detection in a high-resolution video stream,
the computer must handle huge numbers of pixels per second, and presenting these
directly to a complex pattern recognition algorithm may be computationally infeasi-
ble. Instead, the aim is to ﬁnd useful features that are fast to compute, and yet that


---
**Page 21**
1. INTRODUCTION
3
also preserve useful discriminatory information enabling faces to be distinguished
from non-faces. These features are then used as the inputs to the pattern recognition
algorithm. For instance, the average value of the image intensity over a rectangular
subregion can be evaluated extremely efﬁciently (Viola and Jones, 2004), and a set of
such features can prove very effective in fast face detection. Because the number of
such features is smaller than the number of pixels, this kind of pre-processing repre-
sents a form of dimensionality reduction. Care must be taken during pre-processing
because often information is discarded, and if this information is important to the
solution of the problem then the overall accuracy of the system can suffer.
Applications in which the training data comprises examples of the input vectors
along with their corresponding target vectors are known as supervised learning prob-
lems. Cases such as the digit recognition example, in which the aim is to assign each
input vector to one of a ﬁnite number of discrete categories, are called classiﬁcation
problems. If the desired output consists of one or more continuous variables, then
the task is called regression. An example of a regression problem would be the pre-
diction of the yield in a chemical manufacturing process in which the inputs consist
of the concentrations of reactants, the temperature, and the pressure.
In other pattern recognition problems, the training data consists of a set of input
vectors x without any corresponding target values. The goal in such unsupervised
learning problems may be to discover groups of similar examples within the data,
where it is called clustering, or to determine the distribution of data within the input
space, known as density estimation, or to project the data from a high-dimensional
space down to two or three dimensions for the purpose of visualization.
Finally, the technique of reinforcement learning (Sutton and Barto, 1998) is con-
cerned with the problem of ﬁnding suitable actions to take in a given situation in
order to maximize a reward. Here the learning algorithm is not given examples of
optimal outputs, in contrast to supervised learning, but must instead discover them
by a process of trial and error. Typically there is a sequence of states and actions in
which the learning algorithm is interacting with its environment. In many cases, the
current action not only affects the immediate reward but also has an impact on the re-
ward at all subsequent time steps. For example, by using appropriate reinforcement
learning techniques a neural network can learn to play the game of backgammon to a
high standard (Tesauro, 1994). Here the network must learn to take a board position
as input, along with the result of a dice throw, and produce a strong move as the
output. This is done by having the network play against a copy of itself for perhaps a
million games. A major challenge is that a game of backgammon can involve dozens
of moves, and yet it is only at the end of the game that the reward, in the form of
victory, is achieved. The reward must then be attributed appropriately to all of the
moves that led to it, even though some moves will have been good ones and others
less so. This is an example of a credit assignment problem. A general feature of re-
inforcement learning is the trade-off between exploration, in which the system tries
out new kinds of actions to see how effective they are, and exploitation, in which
the system makes use of actions that are known to yield a high reward. Too strong
a focus on either exploration or exploitation will yield poor results. Reinforcement
learning continues to be an active area of machine learning research. However, a


---
**Page 22**
4
1. INTRODUCTION
Figure 1.2
Plot of a training data set of N =
10 points, shown as blue circles,
each comprising an observation
of the input variable x along with
the corresponding target variable
t.
The green curve shows the
function sin(2πx) used to gener-
ate the data. Our goal is to pre-
dict the value of t for some new
value of x, without knowledge of
the green curve.
x
t
0
1
−1
0
1
detailed treatment lies beyond the scope of this book.
Although each of these tasks needs its own tools and techniques, many of the
key ideas that underpin them are common to all such problems. One of the main
goals of this chapter is to introduce, in a relatively informal way, several of the most
important of these concepts and to illustrate them using simple examples. Later in
the book we shall see these same ideas re-emerge in the context of more sophisti-
cated models that are applicable to real-world pattern recognition applications. This
chapter also provides a self-contained introduction to three important tools that will
be used throughout the book, namely probability theory, decision theory, and infor-
mation theory. Although these might sound like daunting topics, they are in fact
straightforward, and a clear understanding of them is essential if machine learning
techniques are to be used to best effect in practical applications.
1.1. Example: Polynomial Curve Fitting
We begin by introducing a simple regression problem, which we shall use as a run-
ning example throughout this chapter to motivate a number of key concepts. Sup-
pose we observe a real-valued input variable x and we wish to use this observation to
predict the value of a real-valued target variable t. For the present purposes, it is in-
structive to consider an artiﬁcial example using synthetically generated data because
we then know the precise process that generated the data for comparison against any
learned model. The data for this example is generated from the function sin(2πx)
with random noise included in the target values, as described in detail in Appendix A.
Now suppose that we are given a training set comprising N observations of x,
written x ≡(x1, . . . , xN)T, together with corresponding observations of the values
of t, denoted t ≡(t1, . . . , tN)T. Figure 1.2 shows a plot of a training set comprising
N = 10 data points. The input data set x in Figure 1.2 was generated by choos-
ing values of xn, for n = 1, . . . , N, spaced uniformly in range [0, 1], and the target
data set t was obtained by ﬁrst computing the corresponding values of the function


---
**Page 23**
1.1. Example: Polynomial Curve Fitting
5
sin(2πx) and then adding a small level of random noise having a Gaussian distri-
bution (the Gaussian distribution is discussed in Section 1.2.4) to each such point in
order to obtain the corresponding value tn. By generating data in this way, we are
capturing a property of many real data sets, namely that they possess an underlying
regularity, which we wish to learn, but that individual observations are corrupted by
random noise. This noise might arise from intrinsically stochastic (i.e. random) pro-
cesses such as radioactive decay but more typically is due to there being sources of
variability that are themselves unobserved.
Our goal is to exploit this training set in order to make predictions of the value
t of the target variable for some new value x of the input variable. As we shall see
later, this involves implicitly trying to discover the underlying function sin(2πx).
This is intrinsically a difﬁcult problem as we have to generalize from a ﬁnite data
set. Furthermore the observed data are corrupted with noise, and so for a given x
there is uncertainty as to the appropriate value for t. Probability theory, discussed
in Section 1.2, provides a framework for expressing such uncertainty in a precise
and quantitative manner, and decision theory, discussed in Section 1.5, allows us to
exploit this probabilistic representation in order to make predictions that are optimal
according to appropriate criteria.
For the moment, however, we shall proceed rather informally and consider a
simple approach based on curve ﬁtting. In particular, we shall ﬁt the data using a
polynomial function of the form
y(x, w) = w0 + w1x + w2x2 + . . . + wMxM =
M

j=0
wjxj
(1.1)
where M is the order of the polynomial, and xj denotes x raised to the power of j.
The polynomial coefﬁcients w0, . . . , wM are collectively denoted by the vector w.
Note that, although the polynomial function y(x, w) is a nonlinear function of x, it
is a linear function of the coefﬁcients w. Functions, such as the polynomial, which
are linear in the unknown parameters have important properties and are called linear
models and will be discussed extensively in Chapters 3 and 4.
The values of the coefﬁcients will be determined by ﬁtting the polynomial to the
training data. This can be done by minimizing an error function that measures the
misﬁt between the function y(x, w), for any given value of w, and the training set
data points. One simple choice of error function, which is widely used, is given by
the sum of the squares of the errors between the predictions y(xn, w) for each data
point xn and the corresponding target values tn, so that we minimize
E(w) = 1
2
N

n=1
{y(xn, w) −tn}2
(1.2)
where the factor of 1/2 is included for later convenience. We shall discuss the mo-
tivation for this choice of error function later in this chapter. For the moment we
simply note that it is a nonnegative quantity that would be zero if, and only if, the


---
**Page 24**
6
1. INTRODUCTION
Figure 1.3
The
error
function
(1.2)
corre-
sponds to (one half of) the sum of
the squares of the displacements
(shown by the vertical green bars)
of each data point from the function
y(x, w).
t
x
y(xn, w)
tn
xn
function y(x, w) were to pass exactly through each training data point. The geomet-
rical interpretation of the sum-of-squares error function is illustrated in Figure 1.3.
We can solve the curve ﬁtting problem by choosing the value of w for which
E(w) is as small as possible. Because the error function is a quadratic function of
the coefﬁcients w, its derivatives with respect to the coefﬁcients will be linear in the
elements of w, and so the minimization of the error function has a unique solution,
denoted by w⋆, which can be found in closed form. The resulting polynomial is
Exercise 1.1
given by the function y(x, w⋆).
There remains the problem of choosing the order M of the polynomial, and as
we shall see this will turn out to be an example of an important concept called model
comparison or model selection. In Figure 1.4, we show four examples of the results
of ﬁtting polynomials having orders M = 0, 1, 3, and 9 to the data set shown in
Figure 1.2.
We notice that the constant (M = 0) and ﬁrst order (M = 1) polynomials
give rather poor ﬁts to the data and consequently rather poor representations of the
function sin(2πx). The third order (M = 3) polynomial seems to give the best ﬁt
to the function sin(2πx) of the examples shown in Figure 1.4. When we go to a
much higher order polynomial (M = 9), we obtain an excellent ﬁt to the training
data. In fact, the polynomial passes exactly through each data point and E(w⋆) = 0.
However, the ﬁtted curve oscillates wildly and gives a very poor representation of
the function sin(2πx). This latter behaviour is known as over-ﬁtting.
As we have noted earlier, the goal is to achieve good generalization by making
accurate predictions for new data. We can obtain some quantitative insight into the
dependence of the generalization performance on M by considering a separate test
set comprising 100 data points generated using exactly the same procedure used
to generate the training set points but with new choices for the random noise values
included in the target values. For each choice of M, we can then evaluate the residual
value of E(w⋆) given by (1.2) for the training data, and we can also evaluate E(w⋆)
for the test data set. It is sometimes more convenient to use the root-mean-square


---
**Page 25**
1.1. Example: Polynomial Curve Fitting
7
x
t
M = 0
0
1
−1
0
1
x
t
M = 1
0
1
−1
0
1
x
t
M = 3
0
1
−1
0
1
x
t
M = 9
0
1
−1
0
1
Figure 1.4
Plots of polynomials having various orders M, shown as red curves, ﬁtted to the data set shown in
Figure 1.2.
(RMS) error deﬁned by
ERMS =

2E(w⋆)/N
(1.3)
in which the division by N allows us to compare different sizes of data sets on
an equal footing, and the square root ensures that ERMS is measured on the same
scale (and in the same units) as the target variable t. Graphs of the training and
test set RMS errors are shown, for various values of M, in Figure 1.5. The test
set error is a measure of how well we are doing in predicting the values of t for
new data observations of x. We note from Figure 1.5 that small values of M give
relatively large values of the test set error, and this can be attributed to the fact that
the corresponding polynomials are rather inﬂexible and are incapable of capturing
the oscillations in the function sin(2πx). Values of M in the range 3 ⩽M ⩽8
give small values for the test set error, and these also give reasonable representations
of the generating function sin(2πx), as can be seen, for the case of M = 3, from
Figure 1.4.


---
**Page 26**
8
1. INTRODUCTION
Figure 1.5
Graphs of the root-mean-square
error, deﬁned by (1.3), evaluated
on the training set and on an inde-
pendent test set for various values
of M.
M
ERMS
0
3
6
9
0
0.5
1
Training
Test
For M = 9, the training set error goes to zero, as we might expect because
this polynomial contains 10 degrees of freedom corresponding to the 10 coefﬁcients
w0, . . . , w9, and so can be tuned exactly to the 10 data points in the training set.
However, the test set error has become very large and, as we saw in Figure 1.4, the
corresponding function y(x, w⋆) exhibits wild oscillations.
This may seem paradoxical because a polynomial of given order contains all
lower order polynomials as special cases. The M = 9 polynomial is therefore capa-
ble of generating results at least as good as the M = 3 polynomial. Furthermore, we
might suppose that the best predictor of new data would be the function sin(2πx)
from which the data was generated (and we shall see later that this is indeed the
case). We know that a power series expansion of the function sin(2πx) contains
terms of all orders, so we might expect that results should improve monotonically as
we increase M.
We can gain some insight into the problem by examining the values of the co-
efﬁcients w⋆obtained from polynomials of various order, as shown in Table 1.1.
We see that, as M increases, the magnitude of the coefﬁcients typically gets larger.
In particular for the M = 9 polynomial, the coefﬁcients have become ﬁnely tuned
to the data by developing large positive and negative values so that the correspond-
Table 1.1
Table of the coefﬁcients w⋆for
polynomials of various order.
Observe how the typical mag-
nitude of the coefﬁcients in-
creases dramatically as the or-
der of the polynomial increases.
M = 0
M = 1
M = 6
M = 9
w⋆
0
0.19
0.82
0.31
0.35
w⋆
1
-1.27
7.99
232.37
w⋆
2
-25.43
-5321.83
w⋆
3
17.37
48568.31
w⋆
4
-231639.30
w⋆
5
640042.26
w⋆
6
-1061800.52
w⋆
7
1042400.18
w⋆
8
-557682.99
w⋆
9
125201.43


---
**Page 27**
1.1. Example: Polynomial Curve Fitting
9
x
t
N = 15
0
1
−1
0
1
x
t
N = 100
0
1
−1
0
1
Figure 1.6
Plots of the solutions obtained by minimizing the sum-of-squares error function using the M = 9
polynomial for N = 15 data points (left plot) and N = 100 data points (right plot). We see that increasing the
size of the data set reduces the over-ﬁtting problem.
ing polynomial function matches each of the data points exactly, but between data
points (particularly near the ends of the range) the function exhibits the large oscilla-
tions observed in Figure 1.4. Intuitively, what is happening is that the more ﬂexible
polynomials with larger values of M are becoming increasingly tuned to the random
noise on the target values.
It is also interesting to examine the behaviour of a given model as the size of the
data set is varied, as shown in Figure 1.6. We see that, for a given model complexity,
the over-ﬁtting problem become less severe as the size of the data set increases.
Another way to say this is that the larger the data set, the more complex (in other
words more ﬂexible) the model that we can afford to ﬁt to the data. One rough
heuristic that is sometimes advocated is that the number of data points should be
no less than some multiple (say 5 or 10) of the number of adaptive parameters in
the model. However, as we shall see in Chapter 3, the number of parameters is not
necessarily the most appropriate measure of model complexity.
Also, there is something rather unsatisfying about having to limit the number of
parameters in a model according to the size of the available training set. It would
seem more reasonable to choose the complexity of the model according to the com-
plexity of the problem being solved. We shall see that the least squares approach
to ﬁnding the model parameters represents a speciﬁc case of maximum likelihood
(discussed in Section 1.2.5), and that the over-ﬁtting problem can be understood as
a general property of maximum likelihood. By adopting a Bayesian approach, the
Section 3.4
over-ﬁtting problem can be avoided. We shall see that there is no difﬁculty from
a Bayesian perspective in employing models for which the number of parameters
greatly exceeds the number of data points. Indeed, in a Bayesian model the effective
number of parameters adapts automatically to the size of the data set.
For the moment, however, it is instructive to continue with the current approach
and to consider how in practice we can apply it to data sets of limited size where we


---
**Page 28**
10
1. INTRODUCTION
x
t
ln λ = −18
0
1
−1
0
1
x
t
ln λ = 0
0
1
−1
0
1
Figure 1.7
Plots of M = 9 polynomials ﬁtted to the data set shown in Figure 1.2 using the regularized error
function (1.4) for two values of the regularization parameter λ corresponding to ln λ = −18 and ln λ = 0. The
case of no regularizer, i.e., λ = 0, corresponding to ln λ = −∞, is shown at the bottom right of Figure 1.4.
may wish to use relatively complex and ﬂexible models. One technique that is often
used to control the over-ﬁtting phenomenon in such cases is that of regularization,
which involves adding a penalty term to the error function (1.2) in order to discourage
the coefﬁcients from reaching large values. The simplest such penalty term takes the
form of a sum of squares of all of the coefﬁcients, leading to a modiﬁed error function
of the form
E(w) = 1
2
N

n=1
{y(xn, w) −tn}2 + λ
2 ∥w∥2
(1.4)
where ∥w∥2 ≡wTw = w2
0 + w2
1 + . . . + w2
M, and the coefﬁcient λ governs the rel-
ative importance of the regularization term compared with the sum-of-squares error
term. Note that often the coefﬁcient w0 is omitted from the regularizer because its
inclusion causes the results to depend on the choice of origin for the target variable
(Hastie et al., 2001), or it may be included but with its own regularization coefﬁcient
(we shall discuss this topic in more detail in Section 5.5.1). Again, the error function
in (1.4) can be minimized exactly in closed form. Techniques such as this are known
Exercise 1.2
in the statistics literature as shrinkage methods because they reduce the value of the
coefﬁcients. The particular case of a quadratic regularizer is called ridge regres-
sion (Hoerl and Kennard, 1970). In the context of neural networks, this approach is
known as weight decay.
Figure 1.7 shows the results of ﬁtting the polynomial of order M = 9 to the
same data set as before but now using the regularized error function given by (1.4).
We see that, for a value of ln λ = −18, the over-ﬁtting has been suppressed and we
now obtain a much closer representation of the underlying function sin(2πx). If,
however, we use too large a value for λ then we again obtain a poor ﬁt, as shown in
Figure 1.7 for ln λ = 0. The corresponding coefﬁcients from the ﬁtted polynomials
are given in Table 1.2, showing that regularization has the desired effect of reducing


---
**Page 29**
1.1. Example: Polynomial Curve Fitting
11
Table 1.2
Table of the coefﬁcients w⋆for M =
9 polynomials with various values for
the regularization parameter λ. Note
that ln λ = −∞corresponds to a
model with no regularization, i.e., to
the graph at the bottom right in Fig-
ure 1.4. We see that, as the value of
λ increases, the typical magnitude of
the coefﬁcients gets smaller.
ln λ = −∞
ln λ = −18
ln λ = 0
w⋆
0
0.35
0.35
0.13
w⋆
1
232.37
4.74
-0.05
w⋆
2
-5321.83
-0.77
-0.06
w⋆
3
48568.31
-31.97
-0.05
w⋆
4
-231639.30
-3.89
-0.03
w⋆
5
640042.26
55.28
-0.02
w⋆
6
-1061800.52
41.32
-0.01
w⋆
7
1042400.18
-45.95
-0.00
w⋆
8
-557682.99
-91.53
0.00
w⋆
9
125201.43
72.68
0.01
the magnitude of the coefﬁcients.
The impact of the regularization term on the generalization error can be seen by
plotting the value of the RMS error (1.3) for both training and test sets against ln λ,
as shown in Figure 1.8. We see that in effect λ now controls the effective complexity
of the model and hence determines the degree of over-ﬁtting.
The issue of model complexity is an important one and will be discussed at
length in Section 1.3. Here we simply note that, if we were trying to solve a practical
application using this approach of minimizing an error function, we would have to
ﬁnd a way to determine a suitable value for the model complexity. The results above
suggest a simple way of achieving this, namely by taking the available data and
partitioning it into a training set, used to determine the coefﬁcients w, and a separate
validation set, also called a hold-out set, used to optimize the model complexity
(either M or λ). In many cases, however, this will prove to be too wasteful of
valuable training data, and we have to seek more sophisticated approaches.
Section 1.3
So far our discussion of polynomial curve ﬁtting has appealed largely to in-
tuition. We now seek a more principled approach to solving problems in pattern
recognition by turning to a discussion of probability theory. As well as providing the
foundation for nearly all of the subsequent developments in this book, it will also
Figure 1.8
Graph of the root-mean-square er-
ror (1.3) versus ln λ for the M = 9
polynomial.
ERMS
ln λ
−35
−30
−25
−20
0
0.5
1
Training
Test


---
**Page 30**
12
1. INTRODUCTION
give us some important insights into the concepts we have introduced in the con-
text of polynomial curve ﬁtting and will allow us to extend these to more complex
situations.
1.2. Probability Theory
A key concept in the ﬁeld of pattern recognition is that of uncertainty. It arises both
through noise on measurements, as well as through the ﬁnite size of data sets. Prob-
ability theory provides a consistent framework for the quantiﬁcation and manipula-
tion of uncertainty and forms one of the central foundations for pattern recognition.
When combined with decision theory, discussed in Section 1.5, it allows us to make
optimal predictions given all the information available to us, even though that infor-
mation may be incomplete or ambiguous.
We will introduce the basic concepts of probability theory by considering a sim-
ple example. Imagine we have two boxes, one red and one blue, and in the red box
we have 2 apples and 6 oranges, and in the blue box we have 3 apples and 1 orange.
This is illustrated in Figure 1.9. Now suppose we randomly pick one of the boxes
and from that box we randomly select an item of fruit, and having observed which
sort of fruit it is we replace it in the box from which it came. We could imagine
repeating this process many times. Let us suppose that in so doing we pick the red
box 40% of the time and we pick the blue box 60% of the time, and that when we
remove an item of fruit from a box we are equally likely to select any of the pieces
of fruit in the box.
In this example, the identity of the box that will be chosen is a random variable,
which we shall denote by B. This random variable can take one of two possible
values, namely r (corresponding to the red box) or b (corresponding to the blue
box). Similarly, the identity of the fruit is also a random variable and will be denoted
by F. It can take either of the values a (for apple) or o (for orange).
To begin with, we shall deﬁne the probability of an event to be the fraction
of times that event occurs out of the total number of trials, in the limit that the total
number of trials goes to inﬁnity. Thus the probability of selecting the red box is 4/10
Figure 1.9
We use a simple example of two
coloured boxes each containing fruit
(apples shown in green and or-
anges shown in orange) to intro-
duce the basic ideas of probability.


---
**Page 31**
1.2. Probability Theory
13
Figure 1.10
We can derive the sum and product rules of probability by
considering two random variables, X, which takes the values {xi} where
i = 1, . . . , M, and Y , which takes the values {yj} where j = 1, . . . , L.
In this illustration we have M = 5 and L = 3. If we consider a total
number N of instances of these variables, then we denote the number
of instances where X = xi and Y = yj by nij, which is the number of
points in the corresponding cell of the array. The number of points in
column i, corresponding to X = xi, is denoted by ci, and the number of
points in row j, corresponding to Y = yj, is denoted by rj.
}
}
ci
rj
yj
xi
nij
and the probability of selecting the blue box is 6/10. We write these probabilities
as p(B = r) = 4/10 and p(B = b) = 6/10. Note that, by deﬁnition, probabilities
must lie in the interval [0, 1]. Also, if the events are mutually exclusive and if they
include all possible outcomes (for instance, in this example the box must be either
red or blue), then we see that the probabilities for those events must sum to one.
We can now ask questions such as: “what is the overall probability that the se-
lection procedure will pick an apple?”, or “given that we have chosen an orange,
what is the probability that the box we chose was the blue one?”. We can answer
questions such as these, and indeed much more complex questions associated with
problems in pattern recognition, once we have equipped ourselves with the two el-
ementary rules of probability, known as the sum rule and the product rule. Having
obtained these rules, we shall then return to our boxes of fruit example.
In order to derive the rules of probability, consider the slightly more general ex-
ample shown in Figure 1.10 involving two random variables X and Y (which could
for instance be the Box and Fruit variables considered above). We shall suppose that
X can take any of the values xi where i = 1, . . . , M, and Y can take the values yj
where j = 1, . . . , L. Consider a total of N trials in which we sample both of the
variables X and Y , and let the number of such trials in which X = xi and Y = yj
be nij. Also, let the number of trials in which X takes the value xi (irrespective
of the value that Y takes) be denoted by ci, and similarly let the number of trials in
which Y takes the value yj be denoted by rj.
The probability that X will take the value xi and Y will take the value yj is
written p(X = xi, Y = yj) and is called the joint probability of X = xi and
Y = yj. It is given by the number of points falling in the cell i,j as a fraction of the
total number of points, and hence
p(X = xi, Y = yj) = nij
N .
(1.5)
Here we are implicitly considering the limit N →∞. Similarly, the probability that
X takes the value xi irrespective of the value of Y is written as p(X = xi) and is
given by the fraction of the total number of points that fall in column i, so that
p(X = xi) = ci
N .
(1.6)
Because the number of instances in column i in Figure 1.10 is just the sum of the
number of instances in each cell of that column, we have ci = 
j nij and therefore,


---
**Page 32**
14
1. INTRODUCTION
from (1.5) and (1.6), we have
p(X = xi) =
L

j=1
p(X = xi, Y = yj)
(1.7)
which is the sum rule of probability. Note that p(X = xi) is sometimes called the
marginal probability, because it is obtained by marginalizing, or summing out, the
other variables (in this case Y ).
If we consider only those instances for which X = xi, then the fraction of
such instances for which Y = yj is written p(Y = yj|X = xi) and is called the
conditional probability of Y = yj given X = xi. It is obtained by ﬁnding the
fraction of those points in column i that fall in cell i,j and hence is given by
p(Y = yj|X = xi) = nij
ci
.
(1.8)
From (1.5), (1.6), and (1.8), we can then derive the following relationship
p(X = xi, Y = yj)
=
nij
N = nij
ci
· ci
N
=
p(Y = yj|X = xi)p(X = xi)
(1.9)
which is the product rule of probability.
So far we have been quite careful to make a distinction between a random vari-
able, such as the box B in the fruit example, and the values that the random variable
can take, for example r if the box were the red one. Thus the probability that B takes
the value r is denoted p(B = r). Although this helps to avoid ambiguity, it leads
to a rather cumbersome notation, and in many cases there will be no need for such
pedantry. Instead, we may simply write p(B) to denote a distribution over the ran-
dom variable B, or p(r) to denote the distribution evaluated for the particular value
r, provided that the interpretation is clear from the context.
With this more compact notation, we can write the two fundamental rules of
probability theory in the following form.
The Rules of Probability
sum rule
p(X) =

Y
p(X, Y )
(1.10)
product rule
p(X, Y ) = p(Y |X)p(X).
(1.11)
Here p(X, Y ) is a joint probability and is verbalized as “the probability of X and
Y ”. Similarly, the quantity p(Y |X) is a conditional probability and is verbalized as
“the probability of Y given X”, whereas the quantity p(X) is a marginal probability


---
**Page 33**
1.2. Probability Theory
15
and is simply “the probability of X”. These two simple rules form the basis for all
of the probabilistic machinery that we use throughout this book.
From the product rule, together with the symmetry property p(X, Y ) = p(Y, X),
we immediately obtain the following relationship between conditional probabilities
p(Y |X) = p(X|Y )p(Y )
p(X)
(1.12)
which is called Bayes’ theorem and which plays a central role in pattern recognition
and machine learning. Using the sum rule, the denominator in Bayes’ theorem can
be expressed in terms of the quantities appearing in the numerator
p(X) =

Y
p(X|Y )p(Y ).
(1.13)
We can view the denominator in Bayes’ theorem as being the normalization constant
required to ensure that the sum of the conditional probability on the left-hand side of
(1.12) over all values of Y equals one.
In Figure 1.11, we show a simple example involving a joint distribution over two
variables to illustrate the concept of marginal and conditional distributions.
Here
a ﬁnite sample of N = 60 data points has been drawn from the joint distribution
and is shown in the top left. In the top right is a histogram of the fractions of data
points having each of the two values of Y . From the deﬁnition of probability, these
fractions would equal the corresponding probabilities p(Y ) in the limit N →∞. We
can view the histogram as a simple way to model a probability distribution given only
a ﬁnite number of points drawn from that distribution. Modelling distributions from
data lies at the heart of statistical pattern recognition and will be explored in great
detail in this book. The remaining two plots in Figure 1.11 show the corresponding
histogram estimates of p(X) and p(X|Y = 1).
Let us now return to our example involving boxes of fruit. For the moment, we
shall once again be explicit about distinguishing between the random variables and
their instantiations. We have seen that the probabilities of selecting either the red or
the blue boxes are given by
p(B = r)
=
4/10
(1.14)
p(B = b)
=
6/10
(1.15)
respectively. Note that these satisfy p(B = r) + p(B = b) = 1.
Now suppose that we pick a box at random, and it turns out to be the blue box.
Then the probability of selecting an apple is just the fraction of apples in the blue
box which is 3/4, and so p(F = a|B = b) = 3/4. In fact, we can write out all four
conditional probabilities for the type of fruit, given the selected box
p(F = a|B = r)
=
1/4
(1.16)
p(F = o|B = r)
=
3/4
(1.17)
p(F = a|B = b)
=
3/4
(1.18)
p(F = o|B = b)
=
1/4.
(1.19)


---
**Page 34**
16
1. INTRODUCTION
p(X,Y )
X
Y = 2
Y = 1
p(Y )
p(X)
X
X
p(X|Y = 1)
Figure 1.11
An illustration of a distribution over two variables, X, which takes 9 possible values, and Y , which
takes two possible values. The top left ﬁgure shows a sample of 60 points drawn from a joint probability distri-
bution over these variables. The remaining ﬁgures show histogram estimates of the marginal distributions p(X)
and p(Y ), as well as the conditional distribution p(X|Y = 1) corresponding to the bottom row in the top left
ﬁgure.
Again, note that these probabilities are normalized so that
p(F = a|B = r) + p(F = o|B = r) = 1
(1.20)
and similarly
p(F = a|B = b) + p(F = o|B = b) = 1.
(1.21)
We can now use the sum and product rules of probability to evaluate the overall
probability of choosing an apple
p(F = a)
=
p(F = a|B = r)p(B = r) + p(F = a|B = b)p(B = b)
=
1
4 × 4
10 + 3
4 × 6
10 = 11
20
(1.22)
from which it follows, using the sum rule, that p(F = o) = 1 −11/20 = 9/20.


---
**Page 35**
1.2. Probability Theory
17
Suppose instead we are told that a piece of fruit has been selected and it is an
orange, and we would like to know which box it came from. This requires that
we evaluate the probability distribution over boxes conditioned on the identity of
the fruit, whereas the probabilities in (1.16)–(1.19) give the probability distribution
over the fruit conditioned on the identity of the box. We can solve the problem of
reversing the conditional probability by using Bayes’ theorem to give
p(B = r|F = o) = p(F = o|B = r)p(B = r)
p(F = o)
= 3
4 × 4
10 × 20
9 = 2
3.
(1.23)
From the sum rule, it then follows that p(B = b|F = o) = 1 −2/3 = 1/3.
We can provide an important interpretation of Bayes’ theorem as follows. If
we had been asked which box had been chosen before being told the identity of
the selected item of fruit, then the most complete information we have available is
provided by the probability p(B). We call this the prior probability because it is the
probability available before we observe the identity of the fruit. Once we are told that
the fruit is an orange, we can then use Bayes’ theorem to compute the probability
p(B|F), which we shall call the posterior probability because it is the probability
obtained after we have observed F. Note that in this example, the prior probability
of selecting the red box was 4/10, so that we were more likely to select the blue box
than the red one. However, once we have observed that the piece of selected fruit is
an orange, we ﬁnd that the posterior probability of the red box is now 2/3, so that
it is now more likely that the box we selected was in fact the red one. This result
accords with our intuition, as the proportion of oranges is much higher in the red box
than it is in the blue box, and so the observation that the fruit was an orange provides
signiﬁcant evidence favouring the red box. In fact, the evidence is sufﬁciently strong
that it outweighs the prior and makes it more likely that the red box was chosen
rather than the blue one.
Finally, we note that if the joint distribution of two variables factorizes into the
product of the marginals, so that p(X, Y ) = p(X)p(Y ), then X and Y are said to
be independent. From the product rule, we see that p(Y |X) = p(Y ), and so the
conditional distribution of Y given X is indeed independent of the value of X. For
instance, in our boxes of fruit example, if each box contained the same fraction of
apples and oranges, then p(F|B) = P(F), so that the probability of selecting, say,
an apple is independent of which box is chosen.
1.2.1
Probability densities
As well as considering probabilities deﬁned over discrete sets of events, we
also wish to consider probabilities with respect to continuous variables. We shall
limit ourselves to a relatively informal discussion. If the probability of a real-valued
variable x falling in the interval (x, x + δx) is given by p(x)δx for δx →0, then
p(x) is called the probability density over x. This is illustrated in Figure 1.12. The
probability that x will lie in an interval (a, b) is then given by
p(x ∈(a, b)) =
 b
a
p(x) dx.
(1.24)


---
**Page 36**
18
1. INTRODUCTION
Figure 1.12
The concept of probability for
discrete variables can be ex-
tended to that of a probability
density p(x) over a continuous
variable x and is such that the
probability of x lying in the inter-
val (x, x+δx) is given by p(x)δx
for δx →0.
The probability
density can be expressed as the
derivative of a cumulative distri-
bution function P(x).
x
δx
p(x)
P(x)
Because probabilities are nonnegative, and because the value of x must lie some-
where on the real axis, the probability density p(x) must satisfy the two conditions
p(x)
⩾
0
(1.25)
 ∞
−∞
p(x) dx
=
1.
(1.26)
Under a nonlinear change of variable, a probability density transforms differently
from a simple function, due to the Jacobian factor. For instance, if we consider
a change of variables x = g(y), then a function f(x) becomes f(y) = f(g(y)).
Now consider a probability density px(x) that corresponds to a density py(y) with
respect to the new variable y, where the sufﬁces denote the fact that px(x) and py(y)
are different densities. Observations falling in the range (x, x + δx) will, for small
values of δx, be transformed into the range (y, y + δy) where px(x)δx ≃py(y)δy,
and hence
py(y)
=
px(x)

dx
dy

=
px(g(y)) |g′(y)| .
(1.27)
One consequence of this property is that the concept of the maximum of a probability
density is dependent on the choice of variable.
Exercise 1.4
The probability that x lies in the interval (−∞, z) is given by the cumulative
distribution function deﬁned by
P(z) =
 z
−∞
p(x) dx
(1.28)
which satisﬁes P ′(x) = p(x), as shown in Figure 1.12.
If we have several continuous variables x1, . . . , xD, denoted collectively by the
vector x, then we can deﬁne a joint probability density p(x) = p(x1, . . . , xD) such


---
**Page 37**
1.2. Probability Theory
19
that the probability of x falling in an inﬁnitesimal volume δx containing the point x
is given by p(x)δx. This multivariate probability density must satisfy
p(x)
⩾
0
(1.29)

p(x) dx
=
1
(1.30)
in which the integral is taken over the whole of x space. We can also consider joint
probability distributions over a combination of discrete and continuous variables.
Note that if x is a discrete variable, then p(x) is sometimes called a probability
mass function because it can be regarded as a set of ‘probability masses’ concentrated
at the allowed values of x.
The sum and product rules of probability, as well as Bayes’ theorem, apply
equally to the case of probability densities, or to combinations of discrete and con-
tinuous variables. For instance, if x and y are two real variables, then the sum and
product rules take the form
p(x)
=

p(x, y) dy
(1.31)
p(x, y)
=
p(y|x)p(x).
(1.32)
A formal justiﬁcation of the sum and product rules for continuous variables (Feller,
1966) requires a branch of mathematics called measure theory and lies outside the
scope of this book. Its validity can be seen informally, however, by dividing each
real variable into intervals of width ∆and considering the discrete probability dis-
tribution over these intervals. Taking the limit ∆→0 then turns sums into integrals
and gives the desired result.
1.2.2
Expectations and covariances
One of the most important operations involving probabilities is that of ﬁnding
weighted averages of functions. The average value of some function f(x) under a
probability distribution p(x) is called the expectation of f(x) and will be denoted by
E[f]. For a discrete distribution, it is given by
E[f] =

x
p(x)f(x)
(1.33)
so that the average is weighted by the relative probabilities of the different values
of x. In the case of continuous variables, expectations are expressed in terms of an
integration with respect to the corresponding probability density
E[f] =

p(x)f(x) dx.
(1.34)
In either case, if we are given a ﬁnite number N of points drawn from the probability
distribution or probability density, then the expectation can be approximated as a


---
**Page 38**
20
1. INTRODUCTION
ﬁnite sum over these points
E[f] ≃1
N
N

n=1
f(xn).
(1.35)
We shall make extensive use of this result when we discuss sampling methods in
Chapter 11. The approximation in (1.35) becomes exact in the limit N →∞.
Sometimes we will be considering expectations of functions of several variables,
in which case we can use a subscript to indicate which variable is being averaged
over, so that for instance
Ex[f(x, y)]
(1.36)
denotes the average of the function f(x, y) with respect to the distribution of x. Note
that Ex[f(x, y)] will be a function of y.
We can also consider a conditional expectation with respect to a conditional
distribution, so that
Ex[f|y] =

x
p(x|y)f(x)
(1.37)
with an analogous deﬁnition for continuous variables.
The variance of f(x) is deﬁned by
var[f] = E

(f(x) −E[f(x)])2	
(1.38)
and provides a measure of how much variability there is in f(x) around its mean
value E[f(x)]. Expanding out the square, we see that the variance can also be written
in terms of the expectations of f(x) and f(x)2
Exercise 1.5
var[f] = E[f(x)2] −E[f(x)]2.
(1.39)
In particular, we can consider the variance of the variable x itself, which is given by
var[x] = E[x2] −E[x]2.
(1.40)
For two random variables x and y, the covariance is deﬁned by
cov[x, y]
=
Ex,y [{x −E[x]} {y −E[y]}]
=
Ex,y[xy] −E[x]E[y]
(1.41)
which expresses the extent to which x and y vary together. If x and y are indepen-
dent, then their covariance vanishes.
Exercise 1.6
In the case of two vectors of random variables x and y, the covariance is a matrix
cov[x, y]
=
Ex,y

{x −E[x]}{yT −E[yT]}
	
=
Ex,y[xyT] −E[x]E[yT].
(1.42)
If we consider the covariance of the components of a vector x with each other, then
we use a slightly simpler notation cov[x] ≡cov[x, x].


---
**Page 39**
1.2. Probability Theory
21
1.2.3
Bayesian probabilities
So far in this chapter, we have viewed probabilities in terms of the frequencies
of random, repeatable events. We shall refer to this as the classical or frequentist
interpretation of probability. Now we turn to the more general Bayesian view, in
which probabilities provide a quantiﬁcation of uncertainty.
Consider an uncertain event, for example whether the moon was once in its own
orbit around the sun, or whether the Arctic ice cap will have disappeared by the end
of the century. These are not events that can be repeated numerous times in order
to deﬁne a notion of probability as we did earlier in the context of boxes of fruit.
Nevertheless, we will generally have some idea, for example, of how quickly we
think the polar ice is melting. If we now obtain fresh evidence, for instance from a
new Earth observation satellite gathering novel forms of diagnostic information, we
may revise our opinion on the rate of ice loss. Our assessment of such matters will
affect the actions we take, for instance the extent to which we endeavour to reduce
the emission of greenhouse gasses. In such circumstances, we would like to be able
to quantify our expression of uncertainty and make precise revisions of uncertainty in
the light of new evidence, as well as subsequently to be able to take optimal actions
or decisions as a consequence. This can all be achieved through the elegant, and very
general, Bayesian interpretation of probability.
The use of probability to represent uncertainty, however, is not an ad-hoc choice,
but is inevitable if we are to respect common sense while making rational coherent
inferences. For instance, Cox (1946) showed that if numerical values are used to
represent degrees of belief, then a simple set of axioms encoding common sense
properties of such beliefs leads uniquely to a set of rules for manipulating degrees of
belief that are equivalent to the sum and product rules of probability. This provided
the ﬁrst rigorous proof that probability theory could be regarded as an extension of
Boolean logic to situations involving uncertainty (Jaynes, 2003). Numerous other
authors have proposed different sets of properties or axioms that such measures of
uncertainty should satisfy (Ramsey, 1931; Good, 1950; Savage, 1961; deFinetti,
1970; Lindley, 1982). In each case, the resulting numerical quantities behave pre-
cisely according to the rules of probability. It is therefore natural to refer to these
quantities as (Bayesian) probabilities.
In the ﬁeld of pattern recognition, too, it is helpful to have a more general no-
Thomas Bayes
1701–1761
Thomas Bayes was born in Tun-
bridge Wells and was a clergyman
as well as an amateur scientist and
a mathematician. He studied logic
and theology at Edinburgh Univer-
sity and was elected Fellow of the
Royal Society in 1742. During the 18th century, is-
sues regarding probability arose in connection with
gambling and with the new concept of insurance. One
particularly important problem concerned so-called in-
verse probability. A solution was proposed by Thomas
Bayes in his paper ‘Essay towards solving a problem
in the doctrine of chances’, which was published in
1764, some three years after his death, in the Philo-
sophical Transactions of the Royal Society.
In fact,
Bayes only formulated his theory for the case of a uni-
form prior, and it was Pierre-Simon Laplace who inde-
pendently rediscovered the theory in general form and
who demonstrated its broad applicability.


---
**Page 40**
22
1. INTRODUCTION
tion of probability. Consider the example of polynomial curve ﬁtting discussed in
Section 1.1. It seems reasonable to apply the frequentist notion of probability to the
random values of the observed variables tn. However, we would like to address and
quantify the uncertainty that surrounds the appropriate choice for the model param-
eters w. We shall see that, from a Bayesian perspective, we can use the machinery
of probability theory to describe the uncertainty in model parameters such as w, or
indeed in the choice of model itself.
Bayes’ theorem now acquires a new signiﬁcance. Recall that in the boxes of fruit
example, the observation of the identity of the fruit provided relevant information
that altered the probability that the chosen box was the red one. In that example,
Bayes’ theorem was used to convert a prior probability into a posterior probability
by incorporating the evidence provided by the observed data. As we shall see in
detail later, we can adopt a similar approach when making inferences about quantities
such as the parameters w in the polynomial curve ﬁtting example. We capture our
assumptions about w, before observing the data, in the form of a prior probability
distribution p(w). The effect of the observed data D = {t1, . . . , tN} is expressed
through the conditional probability p(D|w), and we shall see later, in Section 1.2.5,
how this can be represented explicitly. Bayes’ theorem, which takes the form
p(w|D) = p(D|w)p(w)
p(D)
(1.43)
then allows us to evaluate the uncertainty in w after we have observed D in the form
of the posterior probability p(w|D).
The quantity p(D|w) on the right-hand side of Bayes’ theorem is evaluated for
the observed data set D and can be viewed as a function of the parameter vector
w, in which case it is called the likelihood function. It expresses how probable the
observed data set is for different settings of the parameter vector w. Note that the
likelihood is not a probability distribution over w, and its integral with respect to w
does not (necessarily) equal one.
Given this deﬁnition of likelihood, we can state Bayes’ theorem in words
posterior ∝likelihood × prior
(1.44)
where all of these quantities are viewed as functions of w. The denominator in
(1.43) is the normalization constant, which ensures that the posterior distribution
on the left-hand side is a valid probability density and integrates to one. Indeed,
integrating both sides of (1.43) with respect to w, we can express the denominator
in Bayes’ theorem in terms of the prior distribution and the likelihood function
p(D) =

p(D|w)p(w) dw.
(1.45)
In both the Bayesian and frequentist paradigms, the likelihood function p(D|w)
plays a central role. However, the manner in which it is used is fundamentally dif-
ferent in the two approaches. In a frequentist setting, w is considered to be a ﬁxed
parameter, whose value is determined by some form of ‘estimator’, and error bars


---
**Page 41**
1.2. Probability Theory
23
on this estimate are obtained by considering the distribution of possible data sets D.
By contrast, from the Bayesian viewpoint there is only a single data set D (namely
the one that is actually observed), and the uncertainty in the parameters is expressed
through a probability distribution over w.
A widely used frequentist estimator is maximum likelihood, in which w is set
to the value that maximizes the likelihood function p(D|w). This corresponds to
choosing the value of w for which the probability of the observed data set is maxi-
mized. In the machine learning literature, the negative log of the likelihood function
is called an error function. Because the negative logarithm is a monotonically de-
creasing function, maximizing the likelihood is equivalent to minimizing the error.
One approach to determining frequentist error bars is the bootstrap (Efron, 1979;
Hastie et al., 2001), in which multiple data sets are created as follows. Suppose our
original data set consists of N data points X = {x1, . . . , xN}. We can create a new
data set XB by drawing N points at random from X, with replacement, so that some
points in X may be replicated in XB, whereas other points in X may be absent from
XB. This process can be repeated L times to generate L data sets each of size N and
each obtained by sampling from the original data set X. The statistical accuracy of
parameter estimates can then be evaluated by looking at the variability of predictions
between the different bootstrap data sets.
One advantage of the Bayesian viewpoint is that the inclusion of prior knowl-
edge arises naturally. Suppose, for instance, that a fair-looking coin is tossed three
times and lands heads each time. A classical maximum likelihood estimate of the
probability of landing heads would give 1, implying that all future tosses will land
Section 2.1
heads! By contrast, a Bayesian approach with any reasonable prior will lead to a
much less extreme conclusion.
There has been much controversy and debate associated with the relative mer-
its of the frequentist and Bayesian paradigms, which have not been helped by the
fact that there is no unique frequentist, or even Bayesian, viewpoint. For instance,
one common criticism of the Bayesian approach is that the prior distribution is of-
ten selected on the basis of mathematical convenience rather than as a reﬂection of
any prior beliefs. Even the subjective nature of the conclusions through their de-
pendence on the choice of prior is seen by some as a source of difﬁculty. Reducing
the dependence on the prior is one motivation for so-called noninformative priors.
Section 2.4.3
However, these lead to difﬁculties when comparing different models, and indeed
Bayesian methods based on poor choices of prior can give poor results with high
conﬁdence. Frequentist evaluation methods offer some protection from such prob-
lems, and techniques such as cross-validation remain useful in areas such as model
Section 1.3
comparison.
This book places a strong emphasis on the Bayesian viewpoint, reﬂecting the
huge growth in the practical importance of Bayesian methods in the past few years,
while also discussing useful frequentist concepts as required.
Although the Bayesian framework has its origins in the 18th century, the prac-
tical application of Bayesian methods was for a long time severely limited by the
difﬁculties in carrying through the full Bayesian procedure, particularly the need to
marginalize (sum or integrate) over the whole of parameter space, which, as we shall


---
**Page 42**
24
1. INTRODUCTION
see, is required in order to make predictions or to compare different models. The
development of sampling methods, such as Markov chain Monte Carlo (discussed in
Chapter 11) along with dramatic improvements in the speed and memory capacity
of computers, opened the door to the practical use of Bayesian techniques in an im-
pressive range of problem domains. Monte Carlo methods are very ﬂexible and can
be applied to a wide range of models. However, they are computationally intensive
and have mainly been used for small-scale problems.
More recently, highly efﬁcient deterministic approximation schemes such as
variational Bayes and expectation propagation (discussed in Chapter 10) have been
developed. These offer a complementary alternative to sampling methods and have
allowed Bayesian techniques to be used in large-scale applications (Blei et al., 2003).
1.2.4
The Gaussian distribution
We shall devote the whole of Chapter 2 to a study of various probability dis-
tributions and their key properties. It is convenient, however, to introduce here one
of the most important probability distributions for continuous variables, called the
normal or Gaussian distribution. We shall make extensive use of this distribution in
the remainder of this chapter and indeed throughout much of the book.
For the case of a single real-valued variable x, the Gaussian distribution is de-
ﬁned by
N

x|µ, σ2
=
1
(2πσ2)1/2 exp

−1
2σ2 (x −µ)2

(1.46)
which is governed by two parameters: µ, called the mean, and σ2, called the vari-
ance. The square root of the variance, given by σ, is called the standard deviation,
and the reciprocal of the variance, written as β = 1/σ2, is called the precision. We
shall see the motivation for these terms shortly. Figure 1.13 shows a plot of the
Gaussian distribution.
From the form of (1.46) we see that the Gaussian distribution satisﬁes
N(x|µ, σ2) > 0.
(1.47)
Also it is straightforward to show that the Gaussian is normalized, so that
Exercise 1.7
Pierre-Simon Laplace
1749–1827
It is said that Laplace was seri-
ously lacking in modesty and at one
point declared himself to be the
best mathematician in France at the
time, a claim that was arguably true.
As well as being proliﬁc in mathe-
matics, he also made numerous contributions to as-
tronomy, including the nebular hypothesis by which the
earth is thought to have formed from the condensa-
tion and cooling of a large rotating disk of gas and
dust. In 1812 he published the ﬁrst edition of Th´eorie
Analytique des Probabilit´es, in which Laplace states
that “probability theory is nothing but common sense
reduced to calculation”. This work included a discus-
sion of the inverse probability calculation (later termed
Bayes’ theorem by Poincar´e), which he used to solve
problems in life expectancy, jurisprudence, planetary
masses, triangulation, and error estimation.


---
**Page 43**
1.2. Probability Theory
25
Figure 1.13
Plot of the univariate Gaussian
showing the mean µ and the
standard deviation σ.
N(x|µ, σ2)
x
2σ
µ
 ∞
−∞
N 
x|µ, σ2
dx = 1.
(1.48)
Thus (1.46) satisﬁes the two requirements for a valid probability density.
We can readily ﬁnd expectations of functions of x under the Gaussian distribu-
tion. In particular, the average value of x is given by
Exercise 1.8
E[x] =
 ∞
−∞
N 
x|µ, σ2
x dx = µ.
(1.49)
Because the parameter µ represents the average value of x under the distribution, it
is referred to as the mean. Similarly, for the second order moment
E[x2] =
 ∞
−∞
N

x|µ, σ2
x2 dx = µ2 + σ2.
(1.50)
From (1.49) and (1.50), it follows that the variance of x is given by
var[x] = E[x2] −E[x]2 = σ2
(1.51)
and hence σ2 is referred to as the variance parameter. The maximum of a distribution
is known as its mode. For a Gaussian, the mode coincides with the mean.
Exercise 1.9
We are also interested in the Gaussian distribution deﬁned over a D-dimensional
vector x of continuous variables, which is given by
N(x|µ, Σ) =
1
(2π)D/2
1
|Σ|1/2 exp

−1
2(x −µ)TΣ−1(x −µ)

(1.52)
where the D-dimensional vector µ is called the mean, the D × D matrix Σ is called
the covariance, and |Σ| denotes the determinant of Σ. We shall make use of the
multivariate Gaussian distribution brieﬂy in this chapter, although its properties will
be studied in detail in Section 2.3.


---
**Page 44**
26
1. INTRODUCTION
Figure 1.14
Illustration of the likelihood function for
a Gaussian distribution, shown by the
red curve. Here the black points de-
note a data set of values {xn}, and
the likelihood function given by (1.53)
corresponds to the product of the blue
values. Maximizing the likelihood in-
volves adjusting the mean and vari-
ance of the Gaussian so as to maxi-
mize this product.
x
p(x)
xn
N(xn|µ, σ2)
Now suppose that we have a data set of observations x = (x1, . . . , xN)T, rep-
resenting N observations of the scalar variable x. Note that we are using the type-
face x to distinguish this from a single observation of the vector-valued variable
(x1, . . . , xD)T, which we denote by x. We shall suppose that the observations are
drawn independently from a Gaussian distribution whose mean µ and variance σ2
are unknown, and we would like to determine these parameters from the data set.
Data points that are drawn independently from the same distribution are said to be
independent and identically distributed, which is often abbreviated to i.i.d. We have
seen that the joint probability of two independent events is given by the product of
the marginal probabilities for each event separately. Because our data set x is i.i.d.,
we can therefore write the probability of the data set, given µ and σ2, in the form
p(x|µ, σ2) =
N

n=1
N 
xn|µ, σ2
.
(1.53)
When viewed as a function of µ and σ2, this is the likelihood function for the Gaus-
sian and is interpreted diagrammatically in Figure 1.14.
One common criterion for determining the parameters in a probability distribu-
tion using an observed data set is to ﬁnd the parameter values that maximize the
likelihood function. This might seem like a strange criterion because, from our fore-
going discussion of probability theory, it would seem more natural to maximize the
probability of the parameters given the data, not the probability of the data given the
parameters. In fact, these two criteria are related, as we shall discuss in the context
of curve ﬁtting.
Section 1.2.5
For the moment, however, we shall determine values for the unknown parame-
ters µ and σ2 in the Gaussian by maximizing the likelihood function (1.53). In prac-
tice, it is more convenient to maximize the log of the likelihood function. Because
the logarithm is a monotonically increasing function of its argument, maximization
of the log of a function is equivalent to maximization of the function itself. Taking
the log not only simpliﬁes the subsequent mathematical analysis, but it also helps
numerically because the product of a large number of small probabilities can easily
underﬂow the numerical precision of the computer, and this is resolved by computing
instead the sum of the log probabilities. From (1.46) and (1.53), the log likelihood


---
**Page 45**
1.2. Probability Theory
27
function can be written in the form
ln p 
x|µ, σ2
= −1
2σ2
N

n=1
(xn −µ)2 −N
2 ln σ2 −N
2 ln(2π).
(1.54)
Maximizing (1.54) with respect to µ, we obtain the maximum likelihood solution
given by
Exercise 1.11
µML = 1
N
N

n=1
xn
(1.55)
which is the sample mean, i.e., the mean of the observed values {xn}. Similarly,
maximizing (1.54) with respect to σ2, we obtain the maximum likelihood solution
for the variance in the form
σ2
ML = 1
N
N

n=1
(xn −µML)2
(1.56)
which is the sample variance measured with respect to the sample mean µML. Note
that we are performing a joint maximization of (1.54) with respect to µ and σ2, but
in the case of the Gaussian distribution the solution for µ decouples from that for σ2
so that we can ﬁrst evaluate (1.55) and then subsequently use this result to evaluate
(1.56).
Later in this chapter, and also in subsequent chapters, we shall highlight the sig-
niﬁcant limitations of the maximum likelihood approach. Here we give an indication
of the problem in the context of our solutions for the maximum likelihood param-
eter settings for the univariate Gaussian distribution. In particular, we shall show
that the maximum likelihood approach systematically underestimates the variance
of the distribution. This is an example of a phenomenon called bias and is related
to the problem of over-ﬁtting encountered in the context of polynomial curve ﬁtting.
Section 1.1
We ﬁrst note that the maximum likelihood solutions µML and σ2
ML are functions of
the data set values x1, . . . , xN. Consider the expectations of these quantities with
respect to the data set values, which themselves come from a Gaussian distribution
with parameters µ and σ2. It is straightforward to show that
Exercise 1.12
E[µML]
=
µ
(1.57)
E[σ2
ML]
=
N −1
N

σ2
(1.58)
so that on average the maximum likelihood estimate will obtain the correct mean but
will underestimate the true variance by a factor (N −1)/N. The intuition behind
this result is given by Figure 1.15.
From (1.58) it follows that the following estimate for the variance parameter is
unbiased
σ2 =
N
N −1σ2
ML =
1
N −1
N

n=1
(xn −µML)2.
(1.59)


---
**Page 46**
28
1. INTRODUCTION
Figure 1.15
Illustration of how bias arises in using max-
imum likelihood to determine the variance
of a Gaussian.
The green curve shows
the true Gaussian distribution from which
data is generated, and the three red curves
show the Gaussian distributions obtained
by ﬁtting to three data sets, each consist-
ing of two data points shown in blue, us-
ing the maximum likelihood results (1.55)
and (1.56). Averaged across the three data
sets, the mean is correct, but the variance
is systematically under-estimated because
it is measured relative to the sample mean
and not relative to the true mean.
(a)
(b)
(c)
In Section 10.1.3, we shall see how this result arises automatically when we adopt a
Bayesian approach.
Note that the bias of the maximum likelihood solution becomes less signiﬁcant
as the number N of data points increases, and in the limit N →∞the maximum
likelihood solution for the variance equals the true variance of the distribution that
generated the data. In practice, for anything other than small N, this bias will not
prove to be a serious problem. However, throughout this book we shall be interested
in more complex models with many parameters, for which the bias problems asso-
ciated with maximum likelihood will be much more severe. In fact, as we shall see,
the issue of bias in maximum likelihood lies at the root of the over-ﬁtting problem
that we encountered earlier in the context of polynomial curve ﬁtting.
1.2.5
Curve ﬁtting re-visited
We have seen how the problem of polynomial curve ﬁtting can be expressed in
terms of error minimization. Here we return to the curve ﬁtting example and view it
Section 1.1
from a probabilistic perspective, thereby gaining some insights into error functions
and regularization, as well as taking us towards a full Bayesian treatment.
The goal in the curve ﬁtting problem is to be able to make predictions for the
target variable t given some new value of the input variable x on the basis of a set of
training data comprising N input values x = (x1, . . . , xN)T and their corresponding
target values t = (t1, . . . , tN)T. We can express our uncertainty over the value of
the target variable using a probability distribution. For this purpose, we shall assume
that, given the value of x, the corresponding value of t has a Gaussian distribution
with a mean equal to the value y(x, w) of the polynomial curve given by (1.1). Thus
we have
p(t|x, w, β) = N 
t|y(x, w), β−1
(1.60)
where, for consistency with the notation in later chapters, we have deﬁned a preci-
sion parameter β corresponding to the inverse variance of the distribution. This is
illustrated schematically in Figure 1.16.


---
**Page 47**
1.2. Probability Theory
29
Figure 1.16
Schematic illustration of a Gaus-
sian conditional distribution for t given x given by
(1.60), in which the mean is given by the polyno-
mial function y(x, w), and the precision is given
by the parameter β, which is related to the vari-
ance by β−1 = σ2.
t
x
x0
2σ
y(x0, w)
y(x, w)
p(t|x0, w, β)
We now use the training data {x, t} to determine the values of the unknown
parameters w and β by maximum likelihood. If the data are assumed to be drawn
independently from the distribution (1.60), then the likelihood function is given by
p(t|x, w, β) =
N

n=1
N

tn|y(xn, w), β−1
.
(1.61)
As we did in the case of the simple Gaussian distribution earlier, it is convenient to
maximize the logarithm of the likelihood function. Substituting for the form of the
Gaussian distribution, given by (1.46), we obtain the log likelihood function in the
form
ln p(t|x, w, β) = −β
2
N

n=1
{y(xn, w) −tn}2 + N
2 ln β −N
2 ln(2π).
(1.62)
Consider ﬁrst the determination of the maximum likelihood solution for the polyno-
mial coefﬁcients, which will be denoted by wML. These are determined by maxi-
mizing (1.62) with respect to w. For this purpose, we can omit the last two terms
on the right-hand side of (1.62) because they do not depend on w. Also, we note
that scaling the log likelihood by a positive constant coefﬁcient does not alter the
location of the maximum with respect to w, and so we can replace the coefﬁcient
β/2 with 1/2. Finally, instead of maximizing the log likelihood, we can equivalently
minimize the negative log likelihood. We therefore see that maximizing likelihood is
equivalent, so far as determining w is concerned, to minimizing the sum-of-squares
error function deﬁned by (1.2). Thus the sum-of-squares error function has arisen as
a consequence of maximizing likelihood under the assumption of a Gaussian noise
distribution.
We can also use maximum likelihood to determine the precision parameter β of
the Gaussian conditional distribution. Maximizing (1.62) with respect to β gives
1
βML
= 1
N
N

n=1
{y(xn, wML) −tn}2 .
(1.63)


---
**Page 48**
30
1. INTRODUCTION
Again we can ﬁrst determine the parameter vector wML governing the mean and sub-
sequently use this to ﬁnd the precision βML as was the case for the simple Gaussian
distribution.
Section 1.2.4
Having determined the parameters w and β, we can now make predictions for
new values of x. Because we now have a probabilistic model, these are expressed
in terms of the predictive distribution that gives the probability distribution over t,
rather than simply a point estimate, and is obtained by substituting the maximum
likelihood parameters into (1.60) to give
p(t|x, wML, βML) = N 
t|y(x, wML), β−1
ML

.
(1.64)
Now let us take a step towards a more Bayesian approach and introduce a prior
distribution over the polynomial coefﬁcients w. For simplicity, let us consider a
Gaussian distribution of the form
p(w|α) = N(w|0, α−1I) =
 α
2π
(M+1)/2
exp

−α
2 wTw

(1.65)
where α is the precision of the distribution, and M +1 is the total number of elements
in the vector w for an M th order polynomial. Variables such as α, which control
the distribution of model parameters, are called hyperparameters. Using Bayes’
theorem, the posterior distribution for w is proportional to the product of the prior
distribution and the likelihood function
p(w|x, t, α, β) ∝p(t|x, w, β)p(w|α).
(1.66)
We can now determine w by ﬁnding the most probable value of w given the data,
in other words by maximizing the posterior distribution. This technique is called
maximum posterior, or simply MAP. Taking the negative logarithm of (1.66) and
combining with (1.62) and (1.65), we ﬁnd that the maximum of the posterior is
given by the minimum of
β
2
N

n=1
{y(xn, w) −tn}2 + α
2 wTw.
(1.67)
Thus we see that maximizing the posterior distribution is equivalent to minimizing
the regularized sum-of-squares error function encountered earlier in the form (1.4),
with a regularization parameter given by λ = α/β.
1.2.6
Bayesian curve ﬁtting
Although we have included a prior distribution p(w|α), we are so far still mak-
ing a point estimate of w and so this does not yet amount to a Bayesian treatment. In
a fully Bayesian approach, we should consistently apply the sum and product rules
of probability, which requires, as we shall see shortly, that we integrate over all val-
ues of w. Such marginalizations lie at the heart of Bayesian methods for pattern
recognition.


---
**Page 49**
1.2. Probability Theory
31
In the curve ﬁtting problem, we are given the training data x and t, along with
a new test point x, and our goal is to predict the value of t. We therefore wish
to evaluate the predictive distribution p(t|x, x, t). Here we shall assume that the
parameters α and β are ﬁxed and known in advance (in later chapters we shall discuss
how such parameters can be inferred from data in a Bayesian setting).
A Bayesian treatment simply corresponds to a consistent application of the sum
and product rules of probability, which allow the predictive distribution to be written
in the form
p(t|x, x, t) =

p(t|x, w)p(w|x, t) dw.
(1.68)
Here p(t|x, w) is given by (1.60), and we have omitted the dependence on α and
β to simplify the notation. Here p(w|x, t) is the posterior distribution over param-
eters, and can be found by normalizing the right-hand side of (1.66). We shall see
in Section 3.3 that, for problems such as the curve-ﬁtting example, this posterior
distribution is a Gaussian and can be evaluated analytically. Similarly, the integra-
tion in (1.68) can also be performed analytically with the result that the predictive
distribution is given by a Gaussian of the form
p(t|x, x, t) = N

t|m(x), s2(x)

(1.69)
where the mean and variance are given by
m(x)
=
βφ(x)TS
N

n=1
φ(xn)tn
(1.70)
s2(x)
=
β−1 + φ(x)TSφ(x).
(1.71)
Here the matrix S is given by
S−1 = αI + β
N

n=1
φ(xn)φ(x)T
(1.72)
where I is the unit matrix, and we have deﬁned the vector φ(x) with elements
φi(x) = xi for i = 0, . . . , M.
We see that the variance, as well as the mean, of the predictive distribution in
(1.69) is dependent on x. The ﬁrst term in (1.71) represents the uncertainty in the
predicted value of t due to the noise on the target variables and was expressed already
in the maximum likelihood predictive distribution (1.64) through β−1
ML. However, the
second term arises from the uncertainty in the parameters w and is a consequence
of the Bayesian treatment. The predictive distribution for the synthetic sinusoidal
regression problem is illustrated in Figure 1.17.


---
**Page 50**
32
1. INTRODUCTION
Figure 1.17
The predictive distribution result-
ing from a Bayesian treatment of
polynomial curve ﬁtting using an
M = 9 polynomial, with the ﬁxed
parameters α = 5 × 10−3 and β =
11.1 (corresponding to the known
noise variance), in which the red
curve denotes the mean of the
predictive distribution and the red
region corresponds to ±1 stan-
dard deviation around the mean.
x
t
0
1
−1
0
1
1.3. Model Selection
In our example of polynomial curve ﬁtting using least squares, we saw that there was
an optimal order of polynomial that gave the best generalization. The order of the
polynomial controls the number of free parameters in the model and thereby governs
the model complexity. With regularized least squares, the regularization coefﬁcient
λ also controls the effective complexity of the model, whereas for more complex
models, such as mixture distributions or neural networks there may be multiple pa-
rameters governing complexity. In a practical application, we need to determine
the values of such parameters, and the principal objective in doing so is usually to
achieve the best predictive performance on new data. Furthermore, as well as ﬁnd-
ing the appropriate values for complexity parameters within a given model, we may
wish to consider a range of different types of model in order to ﬁnd the best one for
our particular application.
We have already seen that, in the maximum likelihood approach, the perfor-
mance on the training set is not a good indicator of predictive performance on un-
seen data due to the problem of over-ﬁtting. If data is plentiful, then one approach is
simply to use some of the available data to train a range of models, or a given model
with a range of values for its complexity parameters, and then to compare them on
independent data, sometimes called a validation set, and select the one having the
best predictive performance. If the model design is iterated many times using a lim-
ited size data set, then some over-ﬁtting to the validation data can occur and so it may
be necessary to keep aside a third test set on which the performance of the selected
model is ﬁnally evaluated.
In many applications, however, the supply of data for training and testing will be
limited, and in order to build good models, we wish to use as much of the available
data as possible for training. However, if the validation set is small, it will give a
relatively noisy estimate of predictive performance. One solution to this dilemma is
to use cross-validation, which is illustrated in Figure 1.18. This allows a proportion
(S −1)/S of the available data to be used for training while making use of all of the


---
**Page 51**
1.4. The Curse of Dimensionality
33
Figure 1.18
The technique of S-fold cross-validation, illus-
trated here for the case of S = 4, involves tak-
ing the available data and partitioning it into S
groups (in the simplest case these are of equal
size). Then S −1 of the groups are used to train
a set of models that are then evaluated on the re-
maining group. This procedure is then repeated
for all S possible choices for the held-out group,
indicated here by the red blocks, and the perfor-
mance scores from the S runs are then averaged.
run 1
run 2
run 3
run 4
data to assess performance. When data is particularly scarce, it may be appropriate
to consider the case S = N, where N is the total number of data points, which gives
the leave-one-out technique.
One major drawback of cross-validation is that the number of training runs that
must be performed is increased by a factor of S, and this can prove problematic for
models in which the training is itself computationally expensive. A further problem
with techniques such as cross-validation that use separate data to assess performance
is that we might have multiple complexity parameters for a single model (for in-
stance, there might be several regularization parameters). Exploring combinations
of settings for such parameters could, in the worst case, require a number of training
runs that is exponential in the number of parameters. Clearly, we need a better ap-
proach. Ideally, this should rely only on the training data and should allow multiple
hyperparameters and model types to be compared in a single training run. We there-
fore need to ﬁnd a measure of performance which depends only on the training data
and which does not suffer from bias due to over-ﬁtting.
Historically various ‘information criteria’ have been proposed that attempt to
correct for the bias of maximum likelihood by the addition of a penalty term to
compensate for the over-ﬁtting of more complex models. For example, the Akaike
information criterion, or AIC (Akaike, 1974), chooses the model for which the quan-
tity
ln p(D|wML) −M
(1.73)
is largest. Here p(D|wML) is the best-ﬁt log likelihood, and M is the number of
adjustable parameters in the model. A variant of this quantity, called the Bayesian
information criterion, or BIC, will be discussed in Section 4.4.1. Such criteria do
not take account of the uncertainty in the model parameters, however, and in practice
they tend to favour overly simple models. We therefore turn in Section 3.4 to a fully
Bayesian approach where we shall see how complexity penalties arise in a natural
and principled way.
1.4. The Curse of Dimensionality
In the polynomial curve ﬁtting example we had just one input variable x. For prac-
tical applications of pattern recognition, however, we will have to deal with spaces


---
**Page 52**
34
1. INTRODUCTION
Figure 1.19
Scatter plot of the oil ﬂow data
for input variables x6 and x7, in
which red denotes the ‘homoge-
nous’ class, green denotes the
‘annular’ class, and blue denotes
the ‘laminar’ class.
Our goal is
to classify the new test point de-
noted by ‘×’.
x6
x7
0
0.25
0.5
0.75
1
0
0.5
1
1.5
2
of high dimensionality comprising many input variables. As we now discuss, this
poses some serious challenges and is an important factor inﬂuencing the design of
pattern recognition techniques.
In order to illustrate the problem we consider a synthetically generated data set
representing measurements taken from a pipeline containing a mixture of oil, wa-
ter, and gas (Bishop and James, 1993). These three materials can be present in one
of three different geometrical conﬁgurations known as ‘homogenous’, ‘annular’, and
‘laminar’, and the fractions of the three materials can also vary. Each data point com-
prises a 12-dimensional input vector consisting of measurements taken with gamma
ray densitometers that measure the attenuation of gamma rays passing along nar-
row beams through the pipe. This data set is described in detail in Appendix A.
Figure 1.19 shows 100 points from this data set on a plot showing two of the mea-
surements x6 and x7 (the remaining ten input values are ignored for the purposes of
this illustration). Each data point is labelled according to which of the three geomet-
rical classes it belongs to, and our goal is to use this data as a training set in order to
be able to classify a new observation (x6, x7), such as the one denoted by the cross
in Figure 1.19. We observe that the cross is surrounded by numerous red points, and
so we might suppose that it belongs to the red class. However, there are also plenty
of green points nearby, so we might think that it could instead belong to the green
class. It seems unlikely that it belongs to the blue class. The intuition here is that the
identity of the cross should be determined more strongly by nearby points from the
training set and less strongly by more distant points. In fact, this intuition turns out
to be reasonable and will be discussed more fully in later chapters.
How can we turn this intuition into a learning algorithm? One very simple ap-
proach would be to divide the input space into regular cells, as indicated in Fig-
ure 1.20. When we are given a test point and we wish to predict its class, we ﬁrst
decide which cell it belongs to, and we then ﬁnd all of the training data points that


---
**Page 53**
1.4. The Curse of Dimensionality
35
Figure 1.20
Illustration of a simple approach
to the solution of a classiﬁcation
problem in which the input space
is divided into cells and any new
test point is assigned to the class
that has a majority number of rep-
resentatives in the same cell as
the test point.
As we shall see
shortly, this simplistic approach
has some severe shortcomings.
x6
x7
0
0.25
0.5
0.75
1
0
0.5
1
1.5
2
fall in the same cell. The identity of the test point is predicted as being the same
as the class having the largest number of training points in the same cell as the test
point (with ties being broken at random).
There are numerous problems with this naive approach, but one of the most se-
vere becomes apparent when we consider its extension to problems having larger
numbers of input variables, corresponding to input spaces of higher dimensionality.
The origin of the problem is illustrated in Figure 1.21, which shows that, if we divide
a region of a space into regular cells, then the number of such cells grows exponen-
tially with the dimensionality of the space. The problem with an exponentially large
number of cells is that we would need an exponentially large quantity of training data
in order to ensure that the cells are not empty. Clearly, we have no hope of applying
such a technique in a space of more than a few variables, and so we need to ﬁnd a
more sophisticated approach.
We can gain further insight into the problems of high-dimensional spaces by
returning to the example of polynomial curve ﬁtting and considering how we would
Section 1.1
Figure 1.21
Illustration
of
the
curse of dimensionality,
showing
how the number of regions of a
regular
grid
grows
exponentially
with the dimensionality D of the
space. For clarity, only a subset of
the cubical regions are shown for
D = 3.
x1
D = 1
x1
x2
D = 2
x1
x2
x3
D = 3


---
**Page 54**
36
1. INTRODUCTION
extend this approach to deal with input spaces having several variables. If we have
D input variables, then a general polynomial with coefﬁcients up to order 3 would
take the form
y(x, w) = w0 +
D

i=1
wixi +
D

i=1
D

j=1
wijxixj +
D

i=1
D

j=1
D

k=1
wijkxixjxk. (1.74)
As D increases, so the number of independent coefﬁcients (not all of the coefﬁcients
are independent due to interchange symmetries amongst the x variables) grows pro-
portionally to D3. In practice, to capture complex dependencies in the data, we may
need to use a higher-order polynomial. For a polynomial of order M, the growth in
the number of coefﬁcients is like DM. Although this is now a power law growth,
Exercise 1.16
rather than an exponential growth, it still points to the method becoming rapidly
unwieldy and of limited practical utility.
Our geometrical intuitions, formed through a life spent in a space of three di-
mensions, can fail badly when we consider spaces of higher dimensionality. As a
simple example, consider a sphere of radius r = 1 in a space of D dimensions, and
ask what is the fraction of the volume of the sphere that lies between radius r = 1−ϵ
and r = 1. We can evaluate this fraction by noting that the volume of a sphere of
radius r in D dimensions must scale as rD, and so we write
VD(r) = KDrD
(1.75)
where the constant KD depends only on D. Thus the required fraction is given by
Exercise 1.18
VD(1) −VD(1 −ϵ)
VD(1)
= 1 −(1 −ϵ)D
(1.76)
which is plotted as a function of ϵ for various values of D in Figure 1.22. We see
that, for large D, this fraction tends to 1 even for small values of ϵ. Thus, in spaces
of high dimensionality, most of the volume of a sphere is concentrated in a thin shell
near the surface!
As a further example, of direct relevance to pattern recognition, consider the
behaviour of a Gaussian distribution in a high-dimensional space. If we transform
from Cartesian to polar coordinates, and then integrate out the directional variables,
we obtain an expression for the density p(r) as a function of radius r from the origin.
Exercise 1.20
Thus p(r)δr is the probability mass inside a thin shell of thickness δr located at
radius r. This distribution is plotted, for various values of D, in Figure 1.23, and we
see that for large D the probability mass of the Gaussian is concentrated in a thin
shell.
The severe difﬁculty that can arise in spaces of many dimensions is sometimes
called the curse of dimensionality (Bellman, 1961). In this book, we shall make ex-
tensive use of illustrative examples involving input spaces of one or two dimensions,
because this makes it particularly easy to illustrate the techniques graphically. The
reader should be warned, however, that not all intuitions developed in spaces of low
dimensionality will generalize to spaces of many dimensions.


---
**Page 55**
1.4. The Curse of Dimensionality
37
Figure 1.22
Plot of the fraction of the volume of
a sphere lying in the range r = 1−ϵ
to r = 1 for various values of the
dimensionality D.
ϵ
volume fraction
D = 1
D = 2
D = 5
D = 20
0
0.2
0.4
0.6
0.8
1
0
0.2
0.4
0.6
0.8
1
Although the curse of dimensionality certainly raises important issues for pat-
tern recognition applications, it does not prevent us from ﬁnding effective techniques
applicable to high-dimensional spaces. The reasons for this are twofold. First, real
data will often be conﬁned to a region of the space having lower effective dimension-
ality, and in particular the directions over which important variations in the target
variables occur may be so conﬁned. Second, real data will typically exhibit some
smoothness properties (at least locally) so that for the most part small changes in the
input variables will produce small changes in the target variables, and so we can ex-
ploit local interpolation-like techniques to allow us to make predictions of the target
variables for new values of the input variables. Successful pattern recognition tech-
niques exploit one or both of these properties. Consider, for example, an application
in manufacturing in which images are captured of identical planar objects on a con-
veyor belt, in which the goal is to determine their orientation. Each image is a point
Figure 1.23
Plot of the probability density with
respect to radius r of a Gaus-
sian distribution for various values
of the dimensionality D.
In a
high-dimensional space, most of the
probability mass of a Gaussian is lo-
cated within a thin shell at a speciﬁc
radius.
D = 1
D = 2
D = 20
r
p(r)
0
2
4
0
1
2


---
**Page 56**
38
1. INTRODUCTION
in a high-dimensional space whose dimensionality is determined by the number of
pixels. Because the objects can occur at different positions within the image and
in different orientations, there are three degrees of freedom of variability between
images, and a set of images will live on a three dimensional manifold embedded
within the high-dimensional space. Due to the complex relationships between the
object position or orientation and the pixel intensities, this manifold will be highly
nonlinear. If the goal is to learn a model that can take an input image and output the
orientation of the object irrespective of its position, then there is only one degree of
freedom of variability within the manifold that is signiﬁcant.
1.5. Decision Theory
We have seen in Section 1.2 how probability theory provides us with a consistent
mathematical framework for quantifying and manipulating uncertainty. Here we
turn to a discussion of decision theory that, when combined with probability theory,
allows us to make optimal decisions in situations involving uncertainty such as those
encountered in pattern recognition.
Suppose we have an input vector x together with a corresponding vector t of
target variables, and our goal is to predict t given a new value for x. For regression
problems, t will comprise continuous variables, whereas for classiﬁcation problems
t will represent class labels. The joint probability distribution p(x, t) provides a
complete summary of the uncertainty associated with these variables. Determination
of p(x, t) from a set of training data is an example of inference and is typically a
very difﬁcult problem whose solution forms the subject of much of this book. In
a practical application, however, we must often make a speciﬁc prediction for the
value of t, or more generally take a speciﬁc action based on our understanding of the
values t is likely to take, and this aspect is the subject of decision theory.
Consider, for example, a medical diagnosis problem in which we have taken an
X-ray image of a patient, and we wish to determine whether the patient has cancer
or not. In this case, the input vector x is the set of pixel intensities in the image,
and output variable t will represent the presence of cancer, which we denote by the
class C1, or the absence of cancer, which we denote by the class C2. We might, for
instance, choose t to be a binary variable such that t = 0 corresponds to class C1 and
t = 1 corresponds to class C2. We shall see later that this choice of label values is
particularly convenient for probabilistic models. The general inference problem then
involves determining the joint distribution p(x, Ck), or equivalently p(x, t), which
gives us the most complete probabilistic description of the situation. Although this
can be a very useful and informative quantity, in the end we must decide either to
give treatment to the patient or not, and we would like this choice to be optimal
in some appropriate sense (Duda and Hart, 1973). This is the decision step, and
it is the subject of decision theory to tell us how to make optimal decisions given
the appropriate probabilities. We shall see that the decision stage is generally very
simple, even trivial, once we have solved the inference problem.
Here we give an introduction to the key ideas of decision theory as required for


---
**Page 57**
1.5. Decision Theory
39
the rest of the book. Further background, as well as more detailed accounts, can be
found in Berger (1985) and Bather (2000).
Before giving a more detailed analysis, let us ﬁrst consider informally how we
might expect probabilities to play a role in making decisions. When we obtain the
X-ray image x for a new patient, our goal is to decide which of the two classes to
assign to the image. We are interested in the probabilities of the two classes given
the image, which are given by p(Ck|x). Using Bayes’ theorem, these probabilities
can be expressed in the form
p(Ck|x) = p(x|Ck)p(Ck)
p(x)
.
(1.77)
Note that any of the quantities appearing in Bayes’ theorem can be obtained from
the joint distribution p(x, Ck) by either marginalizing or conditioning with respect to
the appropriate variables. We can now interpret p(Ck) as the prior probability for the
class Ck, and p(Ck|x) as the corresponding posterior probability. Thus p(C1) repre-
sents the probability that a person has cancer, before we take the X-ray measurement.
Similarly, p(C1|x) is the corresponding probability, revised using Bayes’ theorem in
light of the information contained in the X-ray. If our aim is to minimize the chance
of assigning x to the wrong class, then intuitively we would choose the class having
the higher posterior probability. We now show that this intuition is correct, and we
also discuss more general criteria for making decisions.
1.5.1
Minimizing the misclassiﬁcation rate
Suppose that our goal is simply to make as few misclassiﬁcations as possible.
We need a rule that assigns each value of x to one of the available classes. Such a
rule will divide the input space into regions Rk called decision regions, one for each
class, such that all points in Rk are assigned to class Ck. The boundaries between
decision regions are called decision boundaries or decision surfaces. Note that each
decision region need not be contiguous but could comprise some number of disjoint
regions. We shall encounter examples of decision boundaries and decision regions in
later chapters. In order to ﬁnd the optimal decision rule, consider ﬁrst of all the case
of two classes, as in the cancer problem for instance. A mistake occurs when an input
vector belonging to class C1 is assigned to class C2 or vice versa. The probability of
this occurring is given by
p(mistake)
=
p(x ∈R1, C2) + p(x ∈R2, C1)
=

R1
p(x, C2) dx +

R2
p(x, C1) dx.
(1.78)
We are free to choose the decision rule that assigns each point x to one of the two
classes. Clearly to minimize p(mistake) we should arrange that each x is assigned to
whichever class has the smaller value of the integrand in (1.78). Thus, if p(x, C1) >
p(x, C2) for a given value of x, then we should assign that x to class C1. From the
product rule of probability we have p(x, Ck) = p(Ck|x)p(x). Because the factor
p(x) is common to both terms, we can restate this result as saying that the minimum


---
**Page 58**
40
1. INTRODUCTION
R1
R2
x0
x
p(x, C1)
p(x, C2)
x
Figure 1.24
Schematic illustration of the joint probabilities p(x, Ck) for each of two classes plotted
against x, together with the decision boundary x = bx. Values of x ⩾bx are classiﬁed as
class C2 and hence belong to decision region R2, whereas points x < bx are classiﬁed
as C1 and belong to R1. Errors arise from the blue, green, and red regions, so that for
x < bx the errors are due to points from class C2 being misclassiﬁed as C1 (represented by
the sum of the red and green regions), and conversely for points in the region x ⩾bx the
errors are due to points from class C1 being misclassiﬁed as C2 (represented by the blue
region). As we vary the location bx of the decision boundary, the combined areas of the
blue and green regions remains constant, whereas the size of the red region varies. The
optimal choice for bx is where the curves for p(x, C1) and p(x, C2) cross, corresponding to
bx = x0, because in this case the red region disappears. This is equivalent to the minimum
misclassiﬁcation rate decision rule, which assigns each value of x to the class having the
higher posterior probability p(Ck|x).
probability of making a mistake is obtained if each value of x is assigned to the class
for which the posterior probability p(Ck|x) is largest. This result is illustrated for
two classes, and a single input variable x, in Figure 1.24.
For the more general case of K classes, it is slightly easier to maximize the
probability of being correct, which is given by
p(correct)
=
K

k=1
p(x ∈Rk, Ck)
=
K

k=1

Rk
p(x, Ck) dx
(1.79)
which is maximized when the regions Rk are chosen such that each x is assigned
to the class for which p(x, Ck) is largest. Again, using the product rule p(x, Ck) =
p(Ck|x)p(x), and noting that the factor of p(x) is common to all terms, we see
that each x should be assigned to the class having the largest posterior probability
p(Ck|x).


---
**Page 59**
1.5. Decision Theory
41
Figure 1.25
An example of a loss matrix with ele-
ments Lkj for the cancer treatment problem. The rows
correspond to the true class, whereas the columns cor-
respond to the assignment of class made by our deci-
sion criterion.
 cancer
normal
cancer
0
1000
normal
1
0

1.5.2
Minimizing the expected loss
For many applications, our objective will be more complex than simply mini-
mizing the number of misclassiﬁcations. Let us consider again the medical diagnosis
problem. We note that, if a patient who does not have cancer is incorrectly diagnosed
as having cancer, the consequences may be some patient distress plus the need for
further investigations. Conversely, if a patient with cancer is diagnosed as healthy,
the result may be premature death due to lack of treatment. Thus the consequences
of these two types of mistake can be dramatically different. It would clearly be better
to make fewer mistakes of the second kind, even if this was at the expense of making
more mistakes of the ﬁrst kind.
We can formalize such issues through the introduction of a loss function, also
called a cost function, which is a single, overall measure of loss incurred in taking
any of the available decisions or actions. Our goal is then to minimize the total loss
incurred. Note that some authors consider instead a utility function, whose value
they aim to maximize. These are equivalent concepts if we take the utility to be
simply the negative of the loss, and throughout this text we shall use the loss function
convention. Suppose that, for a new value of x, the true class is Ck and that we assign
x to class Cj (where j may or may not be equal to k). In so doing, we incur some
level of loss that we denote by Lkj, which we can view as the k, j element of a loss
matrix. For instance, in our cancer example, we might have a loss matrix of the form
shown in Figure 1.25. This particular loss matrix says that there is no loss incurred
if the correct decision is made, there is a loss of 1 if a healthy patient is diagnosed as
having cancer, whereas there is a loss of 1000 if a patient having cancer is diagnosed
as healthy.
The optimal solution is the one which minimizes the loss function. However,
the loss function depends on the true class, which is unknown. For a given input
vector x, our uncertainty in the true class is expressed through the joint probability
distribution p(x, Ck) and so we seek instead to minimize the average loss, where the
average is computed with respect to this distribution, which is given by
E[L] =

k

j

Rj
Lkjp(x, Ck) dx.
(1.80)
Each x can be assigned independently to one of the decision regions Rj. Our goal
is to choose the regions Rj in order to minimize the expected loss (1.80), which
implies that for each x we should minimize 
k Lkjp(x, Ck). As before, we can use
the product rule p(x, Ck) = p(Ck|x)p(x) to eliminate the common factor of p(x).
Thus the decision rule that minimizes the expected loss is the one that assigns each


---
**Page 60**
42
1. INTRODUCTION
Figure 1.26
Illustration of the reject option. Inputs
x such that the larger of the two poste-
rior probabilities is less than or equal to
some threshold θ will be rejected.
x
p(C1|x)
p(C2|x)
0.0
1.0
θ
reject region
new x to the class j for which the quantity

k
Lkjp(Ck|x)
(1.81)
is a minimum. This is clearly trivial to do, once we know the posterior class proba-
bilities p(Ck|x).
1.5.3
The reject option
We have seen that classiﬁcation errors arise from the regions of input space
where the largest of the posterior probabilities p(Ck|x) is signiﬁcantly less than unity,
or equivalently where the joint distributions p(x, Ck) have comparable values. These
are the regions where we are relatively uncertain about class membership. In some
applications, it will be appropriate to avoid making decisions on the difﬁcult cases
in anticipation of a lower error rate on those examples for which a classiﬁcation de-
cision is made. This is known as the reject option. For example, in our hypothetical
medical illustration, it may be appropriate to use an automatic system to classify
those X-ray images for which there is little doubt as to the correct class, while leav-
ing a human expert to classify the more ambiguous cases. We can achieve this by
introducing a threshold θ and rejecting those inputs x for which the largest of the
posterior probabilities p(Ck|x) is less than or equal to θ. This is illustrated for the
case of two classes, and a single continuous input variable x, in Figure 1.26. Note
that setting θ = 1 will ensure that all examples are rejected, whereas if there are K
classes then setting θ < 1/K will ensure that no examples are rejected. Thus the
fraction of examples that get rejected is controlled by the value of θ.
We can easily extend the reject criterion to minimize the expected loss, when
a loss matrix is given, taking account of the loss incurred when a reject decision is
made.
Exercise 1.24
1.5.4
Inference and decision
We have broken the classiﬁcation problem down into two separate stages, the
inference stage in which we use training data to learn a model for p(Ck|x), and the


---
**Page 61**
1.5. Decision Theory
43
subsequent decision stage in which we use these posterior probabilities to make op-
timal class assignments. An alternative possibility would be to solve both problems
together and simply learn a function that maps inputs x directly into decisions. Such
a function is called a discriminant function.
In fact, we can identify three distinct approaches to solving decision problems,
all of which have been used in practical applications. These are given, in decreasing
order of complexity, by:
(a) First solve the inference problem of determining the class-conditional densities
p(x|Ck) for each class Ck individually. Also separately infer the prior class
probabilities p(Ck). Then use Bayes’ theorem in the form
p(Ck|x) = p(x|Ck)p(Ck)
p(x)
(1.82)
to ﬁnd the posterior class probabilities p(Ck|x). As usual, the denominator
in Bayes’ theorem can be found in terms of the quantities appearing in the
numerator, because
p(x) =

k
p(x|Ck)p(Ck).
(1.83)
Equivalently, we can model the joint distribution p(x, Ck) directly and then
normalize to obtain the posterior probabilities. Having found the posterior
probabilities, we use decision theory to determine class membership for each
new input x. Approaches that explicitly or implicitly model the distribution of
inputs as well as outputs are known as generative models, because by sampling
from them it is possible to generate synthetic data points in the input space.
(b) First solve the inference problem of determining the posterior class probabilities
p(Ck|x), and then subsequently use decision theory to assign each new x to
one of the classes. Approaches that model the posterior probabilities directly
are called discriminative models.
(c) Find a function f(x), called a discriminant function, which maps each input x
directly onto a class label. For instance, in the case of two-class problems,
f(·) might be binary valued and such that f = 0 represents class C1 and f = 1
represents class C2. In this case, probabilities play no role.
Let us consider the relative merits of these three alternatives. Approach (a) is the
most demanding because it involves ﬁnding the joint distribution over both x and
Ck. For many applications, x will have high dimensionality, and consequently we
may need a large training set in order to be able to determine the class-conditional
densities to reasonable accuracy. Note that the class priors p(Ck) can often be esti-
mated simply from the fractions of the training set data points in each of the classes.
One advantage of approach (a), however, is that it also allows the marginal density
of data p(x) to be determined from (1.83). This can be useful for detecting new data
points that have low probability under the model and for which the predictions may


---
**Page 62**
44
1. INTRODUCTION
p(x|C1)
p(x|C2)
x
class densities
0
0.2
0.4
0.6
0.8
1
0
1
2
3
4
5
x
p(C1|x)
p(C2|x)
0
0.2
0.4
0.6
0.8
1
0
0.2
0.4
0.6
0.8
1
1.2
Figure 1.27
Example of the class-conditional densities for two classes having a single input variable x (left
plot) together with the corresponding posterior probabilities (right plot). Note that the left-hand mode of the
class-conditional density p(x|C1), shown in blue on the left plot, has no effect on the posterior probabilities. The
vertical green line in the right plot shows the decision boundary in x that gives the minimum misclassiﬁcation
rate.
be of low accuracy, which is known as outlier detection or novelty detection (Bishop,
1994; Tarassenko, 1995).
However, if we only wish to make classiﬁcation decisions, then it can be waste-
ful of computational resources, and excessively demanding of data, to ﬁnd the joint
distribution p(x, Ck) when in fact we only really need the posterior probabilities
p(Ck|x), which can be obtained directly through approach (b). Indeed, the class-
conditional densities may contain a lot of structure that has little effect on the pos-
terior probabilities, as illustrated in Figure 1.27. There has been much interest in
exploring the relative merits of generative and discriminative approaches to machine
learning, and in ﬁnding ways to combine them (Jebara, 2004; Lasserre et al., 2006).
An even simpler approach is (c) in which we use the training data to ﬁnd a
discriminant function f(x) that maps each x directly onto a class label, thereby
combining the inference and decision stages into a single learning problem. In the
example of Figure 1.27, this would correspond to ﬁnding the value of x shown by
the vertical green line, because this is the decision boundary giving the minimum
probability of misclassiﬁcation.
With option (c), however, we no longer have access to the posterior probabilities
p(Ck|x). There are many powerful reasons for wanting to compute the posterior
probabilities, even if we subsequently use them to make decisions. These include:
Minimizing risk. Consider a problem in which the elements of the loss matrix are
subjected to revision from time to time (such as might occur in a ﬁnancial


---
**Page 63**
1.5. Decision Theory
45
application). If we know the posterior probabilities, we can trivially revise the
minimum risk decision criterion by modifying (1.81) appropriately. If we have
only a discriminant function, then any change to the loss matrix would require
that we return to the training data and solve the classiﬁcation problem afresh.
Reject option. Posterior probabilities allow us to determine a rejection criterion that
will minimize the misclassiﬁcation rate, or more generally the expected loss,
for a given fraction of rejected data points.
Compensating for class priors. Consider our medical X-ray problem again, and
suppose that we have collected a large number of X-ray images from the gen-
eral population for use as training data in order to build an automated screening
system. Because cancer is rare amongst the general population, we might ﬁnd
that, say, only 1 in every 1,000 examples corresponds to the presence of can-
cer. If we used such a data set to train an adaptive model, we could run into
severe difﬁculties due to the small proportion of the cancer class. For instance,
a classiﬁer that assigned every point to the normal class would already achieve
99.9% accuracy and it would be difﬁcult to avoid this trivial solution. Also,
even a large data set will contain very few examples of X-ray images corre-
sponding to cancer, and so the learning algorithm will not be exposed to a
broad range of examples of such images and hence is not likely to generalize
well. A balanced data set in which we have selected equal numbers of exam-
ples from each of the classes would allow us to ﬁnd a more accurate model.
However, we then have to compensate for the effects of our modiﬁcations to
the training data. Suppose we have used such a modiﬁed data set and found
models for the posterior probabilities. From Bayes’ theorem (1.82), we see that
the posterior probabilities are proportional to the prior probabilities, which we
can interpret as the fractions of points in each class. We can therefore simply
take the posterior probabilities obtained from our artiﬁcially balanced data set
and ﬁrst divide by the class fractions in that data set and then multiply by the
class fractions in the population to which we wish to apply the model. Finally,
we need to normalize to ensure that the new posterior probabilities sum to one.
Note that this procedure cannot be applied if we have learned a discriminant
function directly instead of determining posterior probabilities.
Combining models. For complex applications, we may wish to break the problem
into a number of smaller subproblems each of which can be tackled by a sep-
arate module. For example, in our hypothetical medical diagnosis problem,
we may have information available from, say, blood tests as well as X-ray im-
ages. Rather than combine all of this heterogeneous information into one huge
input space, it may be more effective to build one system to interpret the X-
ray images and a different one to interpret the blood data. As long as each of
the two models gives posterior probabilities for the classes, we can combine
the outputs systematically using the rules of probability. One simple way to
do this is to assume that, for each class separately, the distributions of inputs
for the X-ray images, denoted by xI, and the blood data, denoted by xB, are


---
**Page 64**
46
1. INTRODUCTION
independent, so that
p(xI, xB|Ck) = p(xI|Ck)p(xB|Ck).
(1.84)
This is an example of conditional independence property, because the indepen-
Section 8.2
dence holds when the distribution is conditioned on the class Ck. The posterior
probability, given both the X-ray and blood data, is then given by
p(Ck|xI, xB)
∝
p(xI, xB|Ck)p(Ck)
∝
p(xI|Ck)p(xB|Ck)p(Ck)
∝
p(Ck|xI)p(Ck|xB)
p(Ck)
(1.85)
Thus we need the class prior probabilities p(Ck), which we can easily estimate
from the fractions of data points in each class, and then we need to normalize
the resulting posterior probabilities so they sum to one. The particular condi-
tional independence assumption (1.84) is an example of the naive Bayes model.
Section 8.2.2
Note that the joint marginal distribution p(xI, xB) will typically not factorize
under this model. We shall see in later chapters how to construct models for
combining data that do not require the conditional independence assumption
(1.84).
1.5.5
Loss functions for regression
So far, we have discussed decision theory in the context of classiﬁcation prob-
lems. We now turn to the case of regression problems, such as the curve ﬁtting
example discussed earlier. The decision stage consists of choosing a speciﬁc esti-
Section 1.1
mate y(x) of the value of t for each input x. Suppose that in doing so, we incur a
loss L(t, y(x)). The average, or expected, loss is then given by
E[L] =

L(t, y(x))p(x, t) dx dt.
(1.86)
A common choice of loss function in regression problems is the squared loss given
by L(t, y(x)) = {y(x) −t}2. In this case, the expected loss can be written
E[L] =

{y(x) −t}2p(x, t) dx dt.
(1.87)
Our goal is to choose y(x) so as to minimize E[L]. If we assume a completely
ﬂexible function y(x), we can do this formally using the calculus of variations to
Appendix D
give
δE[L]
δy(x) = 2

{y(x) −t}p(x, t) dt = 0.
(1.88)
Solving for y(x), and using the sum and product rules of probability, we obtain
y(x) =

tp(x, t) dt
p(x)
=

tp(t|x) dt = Et[t|x]
(1.89)


---
**Page 65**
1.5. Decision Theory
47
Figure 1.28
The regression function y(x),
which minimizes the expected
squared loss, is given by the
mean of the conditional distri-
bution p(t|x).
t
x
x0
y(x0)
y(x)
p(t|x0)
which is the conditional average of t conditioned on x and is known as the regression
function. This result is illustrated in Figure 1.28. It can readily be extended to mul-
tiple target variables represented by the vector t, in which case the optimal solution
is the conditional average y(x) = Et[t|x].
Exercise 1.25
We can also derive this result in a slightly different way, which will also shed
light on the nature of the regression problem. Armed with the knowledge that the
optimal solution is the conditional expectation, we can expand the square term as
follows
{y(x) −t}2 = {y(x) −E[t|x] + E[t|x] −t}2
=
{y(x) −E[t|x]}2 + 2{y(x) −E[t|x]}{E[t|x] −t} + {E[t|x] −t}2
where, to keep the notation uncluttered, we use E[t|x] to denote Et[t|x]. Substituting
into the loss function and performing the integral over t, we see that the cross-term
vanishes and we obtain an expression for the loss function in the form
E[L] =

{y(x) −E[t|x]}2 p(x) dx +

{E[t|x] −t}2p(x) dx.
(1.90)
The function y(x) we seek to determine enters only in the ﬁrst term, which will be
minimized when y(x) is equal to E[t|x], in which case this term will vanish. This
is simply the result that we derived previously and that shows that the optimal least
squares predictor is given by the conditional mean. The second term is the variance
of the distribution of t, averaged over x. It represents the intrinsic variability of
the target data and can be regarded as noise. Because it is independent of y(x), it
represents the irreducible minimum value of the loss function.
As with the classiﬁcation problem, we can either determine the appropriate prob-
abilities and then use these to make optimal decisions, or we can build models that
make decisions directly. Indeed, we can identify three distinct approaches to solving
regression problems given, in order of decreasing complexity, by:
(a) First solve the inference problem of determining the joint density p(x, t). Then
normalize to ﬁnd the conditional density p(t|x), and ﬁnally marginalize to ﬁnd
the conditional mean given by (1.89).


---
**Page 66**
48
1. INTRODUCTION
(b) First solve the inference problem of determining the conditional density p(t|x),
and then subsequently marginalize to ﬁnd the conditional mean given by (1.89).
(c) Find a regression function y(x) directly from the training data.
The relative merits of these three approaches follow the same lines as for classiﬁca-
tion problems above.
The squared loss is not the only possible choice of loss function for regression.
Indeed, there are situations in which squared loss can lead to very poor results and
where we need to develop more sophisticated approaches. An important example
concerns situations in which the conditional distribution p(t|x) is multimodal, as
often arises in the solution of inverse problems. Here we consider brieﬂy one simple
Section 5.6
generalization of the squared loss, called the Minkowski loss, whose expectation is
given by
E[Lq] =

|y(x) −t|qp(x, t) dx dt
(1.91)
which reduces to the expected squared loss for q = 2. The function |y −t|q is
plotted against y −t for various values of q in Figure 1.29. The minimum of E[Lq]
is given by the conditional mean for q = 2, the conditional median for q = 1, and
the conditional mode for q →0.
Exercise 1.27
1.6. Information Theory
In this chapter, we have discussed a variety of concepts from probability theory and
decision theory that will form the foundations for much of the subsequent discussion
in this book. We close this chapter by introducing some additional concepts from
the ﬁeld of information theory, which will also prove useful in our development of
pattern recognition and machine learning techniques. Again, we shall focus only on
the key concepts, and we refer the reader elsewhere for more detailed discussions
(Viterbi and Omura, 1979; Cover and Thomas, 1991; MacKay, 2003) .
We begin by considering a discrete random variable x and we ask how much
information is received when we observe a speciﬁc value for this variable. The
amount of information can be viewed as the ‘degree of surprise’ on learning the
value of x. If we are told that a highly improbable event has just occurred, we will
have received more information than if we were told that some very likely event
has just occurred, and if we knew that the event was certain to happen we would
receive no information. Our measure of information content will therefore depend
on the probability distribution p(x), and we therefore look for a quantity h(x) that
is a monotonic function of the probability p(x) and that expresses the information
content. The form of h(·) can be found by noting that if we have two events x
and y that are unrelated, then the information gain from observing both of them
should be the sum of the information gained from each of them separately, so that
h(x, y) = h(x) + h(y). Two unrelated events will be statistically independent and
so p(x, y) = p(x)p(y). From these two relationships, it is easily shown that h(x)
must be given by the logarithm of p(x) and so we have
Exercise 1.28

