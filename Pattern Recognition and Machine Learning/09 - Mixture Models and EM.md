# 09 - Mixture Models and EM
*Pages 423-460 from Pattern Recognition and Machine Learning*

---
**Page 423**
8.4. Inference in Graphical Models
407
as illustrated in Figure 8.49(b).
At this point, it is worth pausing to summarize the particular version of the sum-
product algorithm obtained so far for evaluating the marginal p(x). We start by
viewing the variable node x as the root of the factor graph and initiating messages
at the leaves of the graph using (8.70) and (8.71). The message passing steps (8.66)
and (8.69) are then applied recursively until messages have been propagated along
every link, and the root node has received messages from all of its neighbours. Each
node can send a message towards the root once it has received messages from all
of its other neighbours. Once the root node has received messages from all of its
neighbours, the required marginal can be evaluated using (8.63). We shall illustrate
this process shortly.
To see that each node will always receive enough messages to be able to send out
a message, we can use a simple inductive argument as follows. Clearly, for a graph
comprising a variable root node connected directly to several factor leaf nodes, the
algorithm trivially involves sending messages of the form (8.71) directly from the
leaves to the root. Now imagine building up a general graph by adding nodes one at
a time, and suppose that for some particular graph we have a valid algorithm. When
one more (variable or factor) node is added, it can be connected only by a single
link because the overall graph must remain a tree, and so the new node will be a leaf
node. It therefore sends a message to the node to which it is linked, which in turn
will therefore receive all the messages it requires in order to send its own message
towards the root, and so again we have a valid algorithm, thereby completing the
proof.
Now suppose we wish to ﬁnd the marginals for every variable node in the graph.
This could be done by simply running the above algorithm afresh for each such node.
However, this would be very wasteful as many of the required computations would
be repeated. We can obtain a much more efﬁcient procedure by ‘overlaying’ these
multiple message passing algorithms to obtain the general sum-product algorithm
as follows. Arbitrarily pick any (variable or factor) node and designate it as the
root. Propagate messages from the leaves to the root as before. At this point, the
root node will have received messages from all of its neighbours. It can therefore
send out messages to all of its neighbours. These in turn will then have received
messages from all of their neighbours and so can send out messages along the links
going away from the root, and so on. In this way, messages are passed outwards
from the root all the way to the leaves. By now, a message will have passed in
both directions across every link in the graph, and every node will have received
a message from all of its neighbours. Again a simple inductive argument can be
used to verify the validity of this message passing protocol. Because every variable
Exercise 8.20
node will have received messages from all of its neighbours, we can readily calculate
the marginal distribution for every variable in the graph. The number of messages
that have to be computed is given by twice the number of links in the graph and
so involves only twice the computation involved in ﬁnding a single marginal. By
comparison, if we had run the sum-product algorithm separately for each node, the
amount of computation would grow quadratically with the size of the graph. Note
that this algorithm is in fact independent of which node was designated as the root,


---
**Page 424**
408
8. GRAPHICAL MODELS
Figure 8.50
The sum-product algorithm can be viewed
purely in terms of messages sent out by factor
nodes to other factor nodes. In this example,
the outgoing message shown by the blue arrow
is obtained by taking the product of all the in-
coming messages shown by green arrows, mul-
tiplying by the factor fs, and marginalizing over
the variables x1 and x2.
fs
x1
x2
x3
and indeed the notion of one node having a special status was introduced only as a
convenient way to explain the message passing protocol.
Next suppose we wish to ﬁnd the marginal distributions p(xs) associated with
the sets of variables belonging to each of the factors. By a similar argument to that
used above, it is easy to see that the marginal associated with a factor is given by the
Exercise 8.21
product of messages arriving at the factor node and the local factor at that node
p(xs) = fs(xs)

i∈ne(fs)
µxi→fs(xi)
(8.72)
in complete analogy with the marginals at the variable nodes. If the factors are
parameterized functions and we wish to learn the values of the parameters using
the EM algorithm, then these marginals are precisely the quantities we will need to
calculate in the E step, as we shall see in detail when we discuss the hidden Markov
model in Chapter 13.
The message sent by a variable node to a factor node, as we have seen, is simply
the product of the incoming messages on other links. We can if we wish view the
sum-product algorithm in a slightly different form by eliminating messages from
variable nodes to factor nodes and simply considering messages that are sent out by
factor nodes. This is most easily seen by considering the example in Figure 8.50.
So far, we have rather neglected the issue of normalization. If the factor graph
was derived from a directed graph, then the joint distribution is already correctly nor-
malized, and so the marginals obtained by the sum-product algorithm will similarly
be normalized correctly. However, if we started from an undirected graph, then in
general there will be an unknown normalization coefﬁcient 1/Z. As with the simple
chain example of Figure 8.38, this is easily handled by working with an unnormal-
ized version p(x) of the joint distribution, where p(x) = p(x)/Z. We ﬁrst run the
sum-product algorithm to ﬁnd the corresponding unnormalized marginals p(xi). The
coefﬁcient 1/Z is then easily obtained by normalizing any one of these marginals,
and this is computationally efﬁcient because the normalization is done over a single
variable rather than over the entire set of variables as would be required to normalize
p(x) directly.
At this point, it may be helpful to consider a simple example to illustrate the
operation of the sum-product algorithm. Figure 8.51 shows a simple 4-node factor


---
**Page 425**
8.4. Inference in Graphical Models
409
Figure 8.51
A simple factor graph used to illustrate the
sum-product algorithm.
x1
x2
x3
x4
fa
fb
fc
graph whose unnormalized joint distribution is given by
p(x) = fa(x1, x2)fb(x2, x3)fc(x2, x4).
(8.73)
In order to apply the sum-product algorithm to this graph, let us designate node x3
as the root, in which case there are two leaf nodes x1 and x4. Starting with the leaf
nodes, we then have the following sequence of six messages
µx1→fa(x1)
=
1
(8.74)
µfa→x2(x2)
=

x1
fa(x1, x2)
(8.75)
µx4→fc(x4)
=
1
(8.76)
µfc→x2(x2)
=

x4
fc(x2, x4)
(8.77)
µx2→fb(x2)
=
µfa→x2(x2)µfc→x2(x2)
(8.78)
µfb→x3(x3)
=

x2
fb(x2, x3)µx2→fb.
(8.79)
The direction of ﬂow of these messages is illustrated in Figure 8.52. Once this mes-
sage propagation is complete, we can then propagate messages from the root node
out to the leaf nodes, and these are given by
µx3→fb(x3)
=
1
(8.80)
µfb→x2(x2)
=

x3
fb(x2, x3)
(8.81)
µx2→fa(x2)
=
µfb→x2(x2)µfc→x2(x2)
(8.82)
µfa→x1(x1)
=

x2
fa(x1, x2)µx2→fa(x2)
(8.83)
µx2→fc(x2)
=
µfa→x2(x2)µfb→x2(x2)
(8.84)
µfc→x4(x4)
=

x2
fc(x2, x4)µx2→fc(x2).
(8.85)


---
**Page 426**
410
8. GRAPHICAL MODELS
x1
x2
x3
x4
(a)
x1
x2
x3
x4
(b)
Figure 8.52
Flow of messages for the sum-product algorithm applied to the example graph in Figure 8.51. (a)
From the leaf nodes x1 and x4 towards the root node x3. (b) From the root node towards the leaf nodes.
One message has now passed in each direction across each link, and we can now
evaluate the marginals. As a simple check, let us verify that the marginal p(x2) is
given by the correct expression. Using (8.63) and substituting for the messages using
the above results, we have
p(x2)
=
µfa→x2(x2)µfb→x2(x2)µfc→x2(x2)
=

x1
fa(x1, x2)
 
x3
fb(x2, x3)
 
x4
fc(x2, x4)
 
=

x1

x2

x4
fa(x1, x2)fb(x2, x3)fc(x2, x4)
=

x1

x3

x4
p(x)
(8.86)
as required.
So far, we have assumed that all of the variables in the graph are hidden. In most
practical applications, a subset of the variables will be observed, and we wish to cal-
culate posterior distributions conditioned on these observations. Observed nodes are
easily handled within the sum-product algorithm as follows. Suppose we partition x
into hidden variables h and observed variables v, and that the observed value of v
is denoted v. Then we simply multiply the joint distribution p(x) by 
i I(vi,vi),
where I(v,v) = 1 if v = v and I(v,v) = 0 otherwise. This product corresponds
to p(h, v = v) and hence is an unnormalized version of p(h|v = v). By run-
ning the sum-product algorithm, we can efﬁciently calculate the posterior marginals
p(hi|v = v) up to a normalization coefﬁcient whose value can be found efﬁciently
using a local computation. Any summations over variables in v then collapse into a
single term.
We have assumed throughout this section that we are dealing with discrete vari-
ables. However, there is nothing speciﬁc to discrete variables either in the graphical
framework or in the probabilistic construction of the sum-product algorithm. For


---
**Page 427**
8.4. Inference in Graphical Models
411
Table 8.1
Example of a joint distribution over two binary variables for
which the maximum of the joint distribution occurs for dif-
ferent variable values compared to the maxima of the two
marginals.
x = 0
x = 1
y = 0
0.3
0.4
y = 1
0.3
0.0
continuous variables the summations are simply replaced by integrations. We shall
give an example of the sum-product algorithm applied to a graph of linear-Gaussian
variables when we consider linear dynamical systems.
Section 13.3
8.4.5
The max-sum algorithm
The sum-product algorithm allows us to take a joint distribution p(x) expressed
as a factor graph and efﬁciently ﬁnd marginals over the component variables. Two
other common tasks are to ﬁnd a setting of the variables that has the largest prob-
ability and to ﬁnd the value of that probability. These can be addressed through a
closely related algorithm called max-sum, which can be viewed as an application of
dynamic programming in the context of graphical models (Cormen et al., 2001).
A simple approach to ﬁnding latent variable values having high probability
would be to run the sum-product algorithm to obtain the marginals p(xi) for ev-
ery variable, and then, for each marginal in turn, to ﬁnd the value x⋆
i that maximizes
that marginal. However, this would give the set of values that are individually the
most probable. In practice, we typically wish to ﬁnd the set of values that jointly
have the largest probability, in other words the vector xmax that maximizes the joint
distribution, so that
xmax = arg max
x
p(x)
(8.87)
for which the corresponding value of the joint probability will be given by
p(xmax) = max
x
p(x).
(8.88)
In general, xmax is not the same as the set of x⋆
i values, as we can easily show using
a simple example. Consider the joint distribution p(x, y) over two binary variables
x, y ∈{0, 1} given in Table 8.1. The joint distribution is maximized by setting x =
1 and y = 0, corresponding the value 0.4. However, the marginal for p(x), obtained
by summing over both values of y, is given by p(x = 0) = 0.6 and p(x = 1) = 0.4,
and similarly the marginal for y is given by p(y = 0) = 0.7 and p(y = 1) = 0.3,
and so the marginals are maximized by x = 0 and y = 0, which corresponds to a
value of 0.3 for the joint distribution. In fact, it is not difﬁcult to construct examples
for which the set of individually most probable values has probability zero under the
joint distribution.
Exercise 8.27
We therefore seek an efﬁcient algorithm for ﬁnding the value of x that maxi-
mizes the joint distribution p(x) and that will allow us to obtain the value of the
joint distribution at its maximum. To address the second of these problems, we shall
simply write out the max operator in terms of its components
max
x
p(x) = max
x1 . . . max
xM p(x)
(8.89)


---
**Page 428**
412
8. GRAPHICAL MODELS
where M is the total number of variables, and then substitute for p(x) using its
expansion in terms of a product of factors. In deriving the sum-product algorithm,
we made use of the distributive law (8.53) for multiplication. Here we make use of
the analogous law for the max operator
max(ab, ac) = a max(b, c)
(8.90)
which holds if a ⩾0 (as will always be the case for the factors in a graphical model).
This allows us to exchange products with maximizations.
Consider ﬁrst the simple example of a chain of nodes described by (8.49). The
evaluation of the probability maximum can be written as
max
x
p(x) = 1
Z max
x1 · · · max
xN [ψ1,2(x1, x2) · · · ψN−1,N(xN−1, xN)]
=
1
Z max
x1

ψ1,2(x1, x2)

· · · max
xN ψN−1,N(xN−1, xN)

.
As with the calculation of marginals, we see that exchanging the max and product
operators results in a much more efﬁcient computation, and one that is easily inter-
preted in terms of messages passed from node xN backwards along the chain to node
x1.
We can readily generalize this result to arbitrary tree-structured factor graphs
by substituting the expression (8.59) for the factor graph expansion into (8.89) and
again exchanging maximizations with products. The structure of this calculation is
identical to that of the sum-product algorithm, and so we can simply translate those
results into the present context. In particular, suppose that we designate a particular
variable node as the ‘root’ of the graph. Then we start a set of messages propagating
inwards from the leaves of the tree towards the root, with each node sending its
message towards the root once it has received all incoming messages from its other
neighbours. The ﬁnal maximization is performed over the product of all messages
arriving at the root node, and gives the maximum value for p(x). This could be called
the max-product algorithm and is identical to the sum-product algorithm except that
summations are replaced by maximizations. Note that at this stage, messages have
been sent from leaves to the root, but not in the other direction.
In practice, products of many small probabilities can lead to numerical under-
ﬂow problems, and so it is convenient to work with the logarithm of the joint distri-
bution. The logarithm is a monotonic function, so that if a > b then ln a > ln b, and
hence the max operator and the logarithm function can be interchanged, so that
ln

max
x
p(x)

= max
x
ln p(x).
(8.91)
The distributive property is preserved because
max(a + b, a + c) = a + max(b, c).
(8.92)
Thus taking the logarithm simply has the effect of replacing the products in the
max-product algorithm with sums, and so we obtain the max-sum algorithm. From


---
**Page 429**
8.4. Inference in Graphical Models
413
the results (8.66) and (8.69) derived earlier for the sum-product algorithm, we can
readily write down the max-sum algorithm in terms of message passing simply by
replacing ‘sum’ with ‘max’ and replacing products with sums of logarithms to give
µf→x(x)
=
max
x1,...,xM
⎡
⎣ln f(x, x1, . . . , xM) +

m∈ne(fs)\x
µxm→f(xm)
⎤
⎦(8.93)
µx→f(x)
=

l∈ne(x)\f
µfl→x(x).
(8.94)
The initial messages sent by the leaf nodes are obtained by analogy with (8.70) and
(8.71) and are given by
µx→f(x)
=
0
(8.95)
µf→x(x)
=
ln f(x)
(8.96)
while at the root node the maximum probability can then be computed, by analogy
with (8.63), using
pmax = max
x
⎡
⎣
s∈ne(x)
µfs→x(x)
⎤
⎦.
(8.97)
So far, we have seen how to ﬁnd the maximum of the joint distribution by prop-
agating messages from the leaves to an arbitrarily chosen root node. The result will
be the same irrespective of which node is chosen as the root. Now we turn to the
second problem of ﬁnding the conﬁguration of the variables for which the joint dis-
tribution attains this maximum value. So far, we have sent messages from the leaves
to the root. The process of evaluating (8.97) will also give the value xmax for the
most probable value of the root node variable, deﬁned by
xmax = arg max
x
⎡
⎣
s∈ne(x)
µfs→x(x)
⎤
⎦.
(8.98)
At this point, we might be tempted simply to continue with the message passing al-
gorithm and send messages from the root back out to the leaves, using (8.93) and
(8.94), then apply (8.98) to all of the remaining variable nodes. However, because
we are now maximizing rather than summing, it is possible that there may be mul-
tiple conﬁgurations of x all of which give rise to the maximum value for p(x). In
such cases, this strategy can fail because it is possible for the individual variable
values obtained by maximizing the product of messages at each node to belong to
different maximizing conﬁgurations, giving an overall conﬁguration that no longer
corresponds to a maximum.
The problem can be resolved by adopting a rather different kind of message
passing from the root node to the leaves. To see how this works, let us return once
again to the simple chain example of N variables x1, . . . , xN each having K states,


---
**Page 430**
414
8. GRAPHICAL MODELS
Figure 8.53
A lattice, or trellis, diagram show-
ing explicitly the K possible states (one per row
of the diagram) for each of the variables xn in the
chain model. In this illustration K = 3. The ar-
row shows the direction of message passing in the
max-product algorithm. For every state k of each
variable xn (corresponding to column n of the dia-
gram) the function φ(xn) deﬁnes a unique state at
the previous variable, indicated by the black lines.
The two paths through the lattice correspond to
conﬁgurations that give the global maximum of the
joint probability distribution, and either of these
can be found by tracing back along the black lines
in the opposite direction to the arrow.
k = 1
k = 2
k = 3
n −2
n −1
n
n + 1
corresponding to the graph shown in Figure 8.38. Suppose we take node xN to be
the root node. Then in the ﬁrst phase, we propagate messages from the leaf node x1
to the root node using
µxn→fn,n+1(xn)
=
µfn−1,n→xn(xn)
µfn−1,n→xn(xn)
=
max
xn−1

ln fn−1,n(xn−1, xn) + µxn−1→fn−1,n(xn)	
which follow from applying (8.94) and (8.93) to this particular graph. The initial
message sent from the leaf node is simply
µx1→f1,2(x1) = 0.
(8.99)
The most probable value for xN is then given by
xmax
N
= arg max
xN

µfN−1,N→xN(xN)	
.
(8.100)
Now we need to determine the states of the previous variables that correspond to the
same maximizing conﬁguration. This can be done by keeping track of which values
of the variables gave rise to the maximum state of each variable, in other words by
storing quantities given by
φ(xn) = arg max
xn−1

ln fn−1,n(xn−1, xn) + µxn−1→fn−1,n(xn)	
.
(8.101)
To understand better what is happening, it is helpful to represent the chain of vari-
ables in terms of a lattice or trellis diagram as shown in Figure 8.53. Note that this
is not a probabilistic graphical model because the nodes represent individual states
of variables, while each variable corresponds to a column of such states in the di-
agram. For each state of a given variable, there is a unique state of the previous
variable that maximizes the probability (ties are broken either systematically or at
random), corresponding to the function φ(xn) given by (8.101), and this is indicated


---
**Page 431**
8.4. Inference in Graphical Models
415
by the lines connecting the nodes. Once we know the most probable value of the ﬁ-
nal node xN, we can then simply follow the link back to ﬁnd the most probable state
of node xN−1 and so on back to the initial node x1. This corresponds to propagating
a message back down the chain using
xmax
n−1 = φ(xmax
n
)
(8.102)
and is known as back-tracking. Note that there could be several values of xn−1 all
of which give the maximum value in (8.101). Provided we chose one of these values
when we do the back-tracking, we are assured of a globally consistent maximizing
conﬁguration.
In Figure 8.53, we have indicated two paths, each of which we shall suppose
corresponds to a global maximum of the joint probability distribution. If k = 2
and k = 3 each represent possible values of xmax
N
, then starting from either state
and tracing back along the black lines, which corresponds to iterating (8.102), we
obtain a valid global maximum conﬁguration. Note that if we had run a forward
pass of max-sum message passing followed by a backward pass and then applied
(8.98) at each node separately, we could end up selecting some states from one path
and some from the other path, giving an overall conﬁguration that is not a global
maximizer. We see that it is necessary instead to keep track of the maximizing states
during the forward pass using the functions φ(xn) and then use back-tracking to ﬁnd
a consistent solution.
The extension to a general tree-structured factor graph should now be clear. If
a message is sent from a factor node f to a variable node x, a maximization is
performed over all other variable nodes x1, . . . , xM that are neighbours of that fac-
tor node, using (8.93). When we perform this maximization, we keep a record of
which values of the variables x1, . . . , xM gave rise to the maximum. Then in the
back-tracking step, having found xmax, we can then use these stored values to as-
sign consistent maximizing states xmax
1
, . . . , xmax
M . The max-sum algorithm, with
back-tracking, gives an exact maximizing conﬁguration for the variables provided
the factor graph is a tree. An important application of this technique is for ﬁnding
the most probable sequence of hidden states in a hidden Markov model, in which
case it is known as the Viterbi algorithm.
Section 13.2
As with the sum-product algorithm, the inclusion of evidence in the form of
observed variables is straightforward. The observed variables are clamped to their
observed values, and the maximization is performed over the remaining hidden vari-
ables. This can be shown formally by including identity functions for the observed
variables into the factor functions, as we did for the sum-product algorithm.
It is interesting to compare max-sum with the iterated conditional modes (ICM)
algorithm described on page 389. Each step in ICM is computationally simpler be-
cause the ‘messages’ that are passed from one node to the next comprise a single
value consisting of the new state of the node for which the conditional distribution
is maximized. The max-sum algorithm is more complex because the messages are
functions of node variables x and hence comprise a set of K values for each pos-
sible state of x. Unlike max-sum, however, ICM is not guaranteed to ﬁnd a global
maximum even for tree-structured graphs.


---
**Page 432**
416
8. GRAPHICAL MODELS
8.4.6
Exact inference in general graphs
The sum-product and max-sum algorithms provide efﬁcient and exact solutions
to inference problems in tree-structured graphs. For many practical applications,
however, we have to deal with graphs having loops.
The message passing framework can be generalized to arbitrary graph topolo-
gies, giving an exact inference procedure known as the junction tree algorithm (Lau-
ritzen and Spiegelhalter, 1988; Jordan, 2007). Here we give a brief outline of the
key steps involved. This is not intended to convey a detailed understanding of the
algorithm, but rather to give a ﬂavour of the various stages involved. If the starting
point is a directed graph, it is ﬁrst converted to an undirected graph by moraliza-
tion, whereas if starting from an undirected graph this step is not required. Next the
graph is triangulated, which involves ﬁnding chord-less cycles containing four or
more nodes and adding extra links to eliminate such chord-less cycles. For instance,
in the graph in Figure 8.36, the cycle A–C–B–D–A is chord-less a link could be
added between A and B or alternatively between C and D. Note that the joint dis-
tribution for the resulting triangulated graph is still deﬁned by a product of the same
potential functions, but these are now considered to be functions over expanded sets
of variables. Next the triangulated graph is used to construct a new tree-structured
undirected graph called a join tree, whose nodes correspond to the maximal cliques
of the triangulated graph, and whose links connect pairs of cliques that have vari-
ables in common. The selection of which pairs of cliques to connect in this way is
important and is done so as to give a maximal spanning tree deﬁned as follows. Of
all possible trees that link up the cliques, the one that is chosen is one for which the
weight of the tree is largest, where the weight for a link is the number of nodes shared
by the two cliques it connects, and the weight for the tree is the sum of the weights
for the links. If the tree is condensed, so that any clique that is a subset of another
clique is absorbed into the larger clique, this gives a junction tree. As a consequence
of the triangulation step, the resulting tree satisﬁes the running intersection property,
which means that if a variable is contained in two cliques, then it must also be con-
tained in every clique on the path that connects them. This ensures that inference
about variables will be consistent across the graph. Finally, a two-stage message
passing algorithm, essentially equivalent to the sum-product algorithm, can now be
applied to this junction tree in order to ﬁnd marginals and conditionals. Although
the junction tree algorithm sounds complicated, at its heart is the simple idea that
we have used already of exploiting the factorization properties of the distribution to
allow sums and products to be interchanged so that partial summations can be per-
formed, thereby avoiding having to work directly with the joint distribution. The
role of the junction tree is to provide a precise and efﬁcient way to organize these
computations. It is worth emphasizing that this is achieved using purely graphical
operations!
The junction tree is exact for arbitrary graphs and is efﬁcient in the sense that
for a given graph there does not in general exist a computationally cheaper approach.
Unfortunately, the algorithm must work with the joint distributions within each node
(each of which corresponds to a clique of the triangulated graph) and so the compu-
tational cost of the algorithm is determined by the number of variables in the largest


---
**Page 433**
8.4. Inference in Graphical Models
417
clique and will grow exponentially with this number in the case of discrete variables.
An important concept is the treewidth of a graph (Bodlaender, 1993), which is de-
ﬁned in terms of the number of variables in the largest clique. In fact, it is deﬁned to
be as one less than the size of the largest clique, to ensure that a tree has a treewidth
of 1. Because there in general there can be multiple different junction trees that can
be constructed from a given starting graph, the treewidth is deﬁned by the junction
tree for which the largest clique has the fewest variables. If the treewidth of the
original graph is high, the junction tree algorithm becomes impractical.
8.4.7
Loopy belief propagation
For many problems of practical interest, it will not be feasible to use exact in-
ference, and so we need to exploit effective approximation methods. An important
class of such approximations, that can broadly be called variational methods, will be
discussed in detail in Chapter 10. Complementing these deterministic approaches is
a wide range of sampling methods, also called Monte Carlo methods, that are based
on stochastic numerical sampling from distributions and that will be discussed at
length in Chapter 11.
Here we consider one simple approach to approximate inference in graphs with
loops, which builds directly on the previous discussion of exact inference in trees.
The idea is simply to apply the sum-product algorithm even though there is no guar-
antee that it will yield good results. This approach is known as loopy belief propa-
gation (Frey and MacKay, 1998) and is possible because the message passing rules
(8.66) and (8.69) for the sum-product algorithm are purely local. However, because
the graph now has cycles, information can ﬂow many times around the graph. For
some models, the algorithm will converge, whereas for others it will not.
In order to apply this approach, we need to deﬁne a message passing schedule.
Let us assume that one message is passed at a time on any given link and in any
given direction. Each message sent from a node replaces any previous message sent
in the same direction across the same link and will itself be a function only of the
most recent messages received by that node at previous steps of the algorithm.
We have seen that a message can only be sent across a link from a node when
all other messages have been received by that node across its other links. Because
there are loops in the graph, this raises the problem of how to initiate the message
passing algorithm. To resolve this, we suppose that an initial message given by the
unit function has been passed across every link in each direction. Every node is then
in a position to send a message.
There are now many possible ways to organize the message passing schedule.
For example, the ﬂooding schedule simultaneously passes a message across every
link in both directions at each time step, whereas schedules that pass one message at
a time are called serial schedules.
Following Kschischnang et al. (2001), we will say that a (variable or factor)
node a has a message pending on its link to a node b if node a has received any
message on any of its other links since the last time it send a message to b. Thus,
when a node receives a message on one of its links, this creates pending messages
on all of its other links. Only pending messages need to be transmitted because


---
**Page 434**
418
8. GRAPHICAL MODELS
other messages would simply duplicate the previous message on the same link. For
graphs that have a tree structure, any schedule that sends only pending messages
will eventually terminate once a message has passed in each direction across every
link. At this point, there are no pending messages, and the product of the received
Exercise 8.29
messages at every variable give the exact marginal. In graphs having loops, however,
the algorithm may never terminate because there might always be pending messages,
although in practice it is generally found to converge within a reasonable time for
most applications. Once the algorithm has converged, or once it has been stopped
if convergence is not observed, the (approximate) local marginals can be computed
using the product of the most recently received incoming messages to each variable
node or factor node on every link.
In some applications, the loopy belief propagation algorithm can give poor re-
sults, whereas in other applications it has proven to be very effective. In particular,
state-of-the-art algorithms for decoding certain kinds of error-correcting codes are
equivalent to loopy belief propagation (Gallager, 1963; Berrou et al., 1993; McEliece
et al., 1998; MacKay and Neal, 1999; Frey, 1998).
8.4.8
Learning the graph structure
In our discussion of inference in graphical models, we have assumed that the
structure of the graph is known and ﬁxed. However, there is also interest in go-
ing beyond the inference problem and learning the graph structure itself from data
(Friedman and Koller, 2003). This requires that we deﬁne a space of possible struc-
tures as well as a measure that can be used to score each structure.
From a Bayesian viewpoint, we would ideally like to compute a posterior dis-
tribution over graph structures and to make predictions by averaging with respect
to this distribution. If we have a prior p(m) over graphs indexed by m, then the
posterior distribution is given by
p(m|D) ∝p(m)p(D|m)
(8.103)
where D is the observed data set. The model evidence p(D|m) then provides the
score for each model. However, evaluation of the evidence involves marginalization
over the latent variables and presents a challenging computational problem for many
models.
Exploring the space of structures can also be problematic. Because the number
of different graph structures grows exponentially with the number of nodes, it is
often necessary to resort to heuristics to ﬁnd good candidates.
Exercises
8.1
(⋆) www
By marginalizing out the variables in order, show that the representation
(8.5) for the joint distribution of a directed graph is correctly normalized, provided
each of the conditional distributions is normalized.
8.2
(⋆) www
Show that the property of there being no directed cycles in a directed
graph follows from the statement that there exists an ordered numbering of the nodes
such that for each node there are no links going to a lower-numbered node.


---
**Page 435**
Exercises
419
Table 8.2
The joint distribution over three binary variables.
a
b
c
p(a, b, c)
0
0
0
0.192
0
0
1
0.144
0
1
0
0.048
0
1
1
0.216
1
0
0
0.192
1
0
1
0.064
1
1
0
0.048
1
1
1
0.096
8.3
(⋆⋆)
Consider three binary variables a, b, c ∈{0, 1} having the joint distribution
given in Table 8.2. Show by direct evaluation that this distribution has the property
that a and b are marginally dependent, so that p(a, b) ̸= p(a)p(b), but that they
become independent when conditioned on c, so that p(a, b|c) = p(a|c)p(b|c) for
both c = 0 and c = 1.
8.4
(⋆⋆)
Evaluate the distributions p(a), p(b|c), and p(c|a) corresponding to the joint
distribution given in Table 8.2. Hence show by direct evaluation that p(a, b, c) =
p(a)p(c|a)p(b|c). Draw the corresponding directed graph.
8.5
(⋆) www
Draw a directed probabilistic graphical model corresponding to the
relevance vector machine described by (7.79) and (7.80).
8.6
(⋆) For the model shown in Figure 8.13, we have seen that the number of parameters
required to specify the conditional distribution p(y|x1, . . . , xM), where xi ∈{0, 1},
could be reduced from 2M to M +1 by making use of the logistic sigmoid represen-
tation (8.10). An alternative representation (Pearl, 1988) is given by
p(y = 1|x1, . . . , xM) = 1 −(1 −µ0)
M

i=1
(1 −µi)xi
(8.104)
where the parameters µi represent the probabilities p(xi = 1), and µ0 is an additional
parameters satisfying 0 ⩽µ0 ⩽1. The conditional distribution (8.104) is known as
the noisy-OR. Show that this can be interpreted as a ‘soft’ (probabilistic) form of the
logical OR function (i.e., the function that gives y = 1 whenever at least one of the
xi = 1). Discuss the interpretation of µ0.
8.7
(⋆⋆) Using the recursion relations (8.15) and (8.16), show that the mean and covari-
ance of the joint distribution for the graph shown in Figure 8.14 are given by (8.17)
and (8.18), respectively.
8.8
(⋆) www
Show that a ⊥⊥b, c | d implies a ⊥⊥b | d.
8.9
(⋆) www
Using the d-separation criterion, show that the conditional distribution
for a node x in a directed graph, conditioned on all of the nodes in the Markov
blanket, is independent of the remaining variables in the graph.


---
**Page 436**
420
8. GRAPHICAL MODELS
Figure 8.54
Example of a graphical model used to explore the con-
ditional independence properties of the head-to-head
path a–c–b when a descendant of c, namely the node
d, is observed.
c
a
b
d
8.10
(⋆) Consider the directed graph shown in Figure 8.54 in which none of the variables
is observed. Show that a ⊥⊥b | ∅. Suppose we now observe the variable d. Show
that in general a ̸⊥⊥b | d.
8.11
(⋆⋆) Consider the example of the car fuel system shown in Figure 8.21, and suppose
that instead of observing the state of the fuel gauge G directly, the gauge is seen by
the driver D who reports to us the reading on the gauge. This report is either that the
gauge shows full D = 1 or that it shows empty D = 0. Our driver is a bit unreliable,
as expressed through the following probabilities
p(D = 1|G = 1)
=
0.9
(8.105)
p(D = 0|G = 0)
=
0.9.
(8.106)
Suppose that the driver tells us that the fuel gauge shows empty, in other words
that we observe D = 0. Evaluate the probability that the tank is empty given only
this observation. Similarly, evaluate the corresponding probability given also the
observation that the battery is ﬂat, and note that this second probability is lower.
Discuss the intuition behind this result, and relate the result to Figure 8.54.
8.12
(⋆) www
Show that there are 2M(M−1)/2 distinct undirected graphs over a set of
M distinct random variables. Draw the 8 possibilities for the case of M = 3.
8.13
(⋆)
Consider the use of iterated conditional modes (ICM) to minimize the energy
function given by (8.42). Write down an expression for the difference in the values
of the energy associated with the two states of a particular variable xj, with all other
variables held ﬁxed, and show that it depends only on quantities that are local to xj
in the graph.
8.14
(⋆)
Consider a particular case of the energy function given by (8.42) in which the
coefﬁcients β = h = 0. Show that the most probable conﬁguration of the latent
variables is given by xi = yi for all i.
8.15
(⋆⋆) www
Show that the joint distribution p(xn−1, xn) for two neighbouring
nodes in the graph shown in Figure 8.38 is given by an expression of the form (8.58).


---
**Page 437**
Exercises
421
8.16
(⋆⋆)
Consider the inference problem of evaluating p(xn|xN) for the graph shown
in Figure 8.38, for all nodes n ∈{1, . . . , N −1}. Show that the message passing
algorithm discussed in Section 8.4.1 can be used to solve this efﬁciently, and discuss
which messages are modiﬁed and in what way.
8.17
(⋆⋆)
Consider a graph of the form shown in Figure 8.38 having N = 5 nodes, in
which nodes x3 and x5 are observed. Use d-separation to show that x2 ⊥⊥x5 | x3.
Show that if the message passing algorithm of Section 8.4.1 is applied to the evalu-
ation of p(x2|x3, x5), the result will be independent of the value of x5.
8.18
(⋆⋆) www
Show that a distribution represented by a directed tree can trivially
be written as an equivalent distribution over the corresponding undirected tree. Also
show that a distribution expressed as an undirected tree can, by suitable normaliza-
tion of the clique potentials, be written as a directed tree. Calculate the number of
distinct directed trees that can be constructed from a given undirected tree.
8.19
(⋆⋆)
Apply the sum-product algorithm derived in Section 8.4.4 to the chain-of-
nodes model discussed in Section 8.4.1 and show that the results (8.54), (8.55), and
(8.57) are recovered as a special case.
8.20
(⋆) www
Consider the message passing protocol for the sum-product algorithm on
a tree-structured factor graph in which messages are ﬁrst propagated from the leaves
to an arbitrarily chosen root node and then from the root node out to the leaves. Use
proof by induction to show that the messages can be passed in such an order that
at every step, each node that must send a message has received all of the incoming
messages necessary to construct its outgoing messages.
8.21
(⋆⋆) www
Show that the marginal distributions p(xs) over the sets of variables
xs associated with each of the factors fx(xs) in a factor graph can be found by ﬁrst
running the sum-product message passing algorithm and then evaluating the required
marginals using (8.72).
8.22
(⋆)
Consider a tree-structured factor graph, in which a given subset of the variable
nodes form a connected subgraph (i.e., any variable node of the subset is connected
to at least one of the other variable nodes via a single factor node). Show how the
sum-product algorithm can be used to compute the marginal distribution over that
subset.
8.23
(⋆⋆) www
In Section 8.4.4, we showed that the marginal distribution p(xi) for a
variable node xi in a factor graph is given by the product of the messages arriving at
this node from neighbouring factor nodes in the form (8.63). Show that the marginal
p(xi) can also be written as the product of the incoming message along any one of
the links with the outgoing message along the same link.
8.24
(⋆⋆)
Show that the marginal distribution for the variables xs in a factor fs(xs) in
a tree-structured factor graph, after running the sum-product message passing algo-
rithm, can be written as the product of the message arriving at the factor node along
all its links, times the local factor f(xs), in the form (8.72).


---
**Page 438**
422
8. GRAPHICAL MODELS
8.25
(⋆⋆)
In (8.86), we veriﬁed that the sum-product algorithm run on the graph in
Figure 8.51 with node x3 designated as the root node gives the correct marginal for
x2. Show that the correct marginals are obtained also for x1 and x3. Similarly, show
that the use of the result (8.72) after running the sum-product algorithm on this graph
gives the correct joint distribution for x1, x2.
8.26
(⋆) Consider a tree-structured factor graph over discrete variables, and suppose we
wish to evaluate the joint distribution p(xa, xb) associated with two variables xa and
xb that do not belong to a common factor. Deﬁne a procedure for using the sum-
product algorithm to evaluate this joint distribution in which one of the variables is
successively clamped to each of its allowed values.
8.27
(⋆⋆) Consider two discrete variables x and y each having three possible states, for
example x, y ∈{0, 1, 2}. Construct a joint distribution p(x, y) over these variables
having the property that the value x that maximizes the marginal p(x), along with
the value y that maximizes the marginal p(y), together have probability zero under
the joint distribution, so that p(x,y) = 0.
8.28
(⋆⋆) www
The concept of a pending message in the sum-product algorithm for
a factor graph was deﬁned in Section 8.4.7. Show that if the graph has one or more
cycles, there will always be at least one pending message irrespective of how long
the algorithm runs.
8.29
(⋆⋆) www
Show that if the sum-product algorithm is run on a factor graph with a
tree structure (no loops), then after a ﬁnite number of messages have been sent, there
will be no pending messages.


---
**Page 439**
9
Mixture Models
and EM
If we deﬁne a joint distribution over observed and latent variables, the correspond-
ing distribution of the observed variables alone is obtained by marginalization. This
allows relatively complex marginal distributions over observed variables to be ex-
pressed in terms of more tractable joint distributions over the expanded space of
observed and latent variables. The introduction of latent variables thereby allows
complicated distributions to be formed from simpler components. In this chapter,
we shall see that mixture distributions, such as the Gaussian mixture discussed in
Section 2.3.9, can be interpreted in terms of discrete latent variables. Continuous
latent variables will form the subject of Chapter 12.
As well as providing a framework for building more complex probability dis-
tributions, mixture models can also be used to cluster data. We therefore begin our
discussion of mixture distributions by considering the problem of ﬁnding clusters
in a set of data points, which we approach ﬁrst using a nonprobabilistic technique
called the K-means algorithm (Lloyd, 1982). Then we introduce the latent variable
Section 9.1
423


---
**Page 440**
424
9. MIXTURE MODELS AND EM
view of mixture distributions in which the discrete latent variables can be interpreted
as deﬁning assignments of data points to speciﬁc components of the mixture. A gen-
Section 9.2
eral technique for ﬁnding maximum likelihood estimators in latent variable models
is the expectation-maximization (EM) algorithm. We ﬁrst of all use the Gaussian
mixture distribution to motivate the EM algorithm in a fairly informal way, and then
we give a more careful treatment based on the latent variable viewpoint. We shall
Section 9.3
see that the K-means algorithm corresponds to a particular nonprobabilistic limit of
EM applied to mixtures of Gaussians. Finally, we discuss EM in some generality.
Section 9.4
Gaussian mixture models are widely used in data mining, pattern recognition,
machine learning, and statistical analysis. In many applications, their parameters are
determined by maximum likelihood, typically using the EM algorithm. However, as
we shall see there are some signiﬁcant limitations to the maximum likelihood ap-
proach, and in Chapter 10 we shall show that an elegant Bayesian treatment can be
given using the framework of variational inference. This requires little additional
computation compared with EM, and it resolves the principal difﬁculties of maxi-
mum likelihood while also allowing the number of components in the mixture to be
inferred automatically from the data.
9.1. K-means Clustering
We begin by considering the problem of identifying groups, or clusters, of data points
in a multidimensional space. Suppose we have a data set {x1, . . . , xN} consisting
of N observations of a random D-dimensional Euclidean variable x. Our goal is to
partition the data set into some number K of clusters, where we shall suppose for
the moment that the value of K is given. Intuitively, we might think of a cluster as
comprising a group of data points whose inter-point distances are small compared
with the distances to points outside of the cluster. We can formalize this notion by
ﬁrst introducing a set of D-dimensional vectors µk, where k = 1, . . . , K, in which
µk is a prototype associated with the kth cluster. As we shall see shortly, we can
think of the µk as representing the centres of the clusters. Our goal is then to ﬁnd
an assignment of data points to clusters, as well as a set of vectors {µk}, such that
the sum of the squares of the distances of each data point to its closest vector µk, is
a minimum.
It is convenient at this point to deﬁne some notation to describe the assignment
of data points to clusters. For each data point xn, we introduce a corresponding set
of binary indicator variables rnk ∈{0, 1}, where k = 1, . . . , K describing which of
the K clusters the data point xn is assigned to, so that if data point xn is assigned to
cluster k then rnk = 1, and rnj = 0 for j ̸= k. This is known as the 1-of-K coding
scheme. We can then deﬁne an objective function, sometimes called a distortion
measure, given by
J =
N

n=1
K

k=1
rnk∥xn −µk∥2
(9.1)
which represents the sum of the squares of the distances of each data point to its


---
**Page 441**
9.1. K-means Clustering
425
assigned vector µk. Our goal is to ﬁnd values for the {rnk} and the {µk} so as to
minimize J. We can do this through an iterative procedure in which each iteration
involves two successive steps corresponding to successive optimizations with respect
to the rnk and the µk. First we choose some initial values for the µk. Then in the ﬁrst
phase we minimize J with respect to the rnk, keeping the µk ﬁxed. In the second
phase we minimize J with respect to the µk, keeping rnk ﬁxed. This two-stage
optimization is then repeated until convergence. We shall see that these two stages
of updating rnk and updating µk correspond respectively to the E (expectation) and
M (maximization) steps of the EM algorithm, and to emphasize this we shall use the
Section 9.4
terms E step and M step in the context of the K-means algorithm.
Consider ﬁrst the determination of the rnk. Because J in (9.1) is a linear func-
tion of rnk, this optimization can be performed easily to give a closed form solution.
The terms involving different n are independent and so we can optimize for each
n separately by choosing rnk to be 1 for whichever value of k gives the minimum
value of ∥xn −µk∥2. In other words, we simply assign the nth data point to the
closest cluster centre. More formally, this can be expressed as
rnk =
1
if k = arg minj ∥xn −µj∥2
0
otherwise.
(9.2)
Now consider the optimization of the µk with the rnk held ﬁxed. The objective
function J is a quadratic function of µk, and it can be minimized by setting its
derivative with respect to µk to zero giving
2
N

n=1
rnk(xn −µk) = 0
(9.3)
which we can easily solve for µk to give
µk =

n rnkxn

n rnk
.
(9.4)
The denominator in this expression is equal to the number of points assigned to
cluster k, and so this result has a simple interpretation, namely set µk equal to the
mean of all of the data points xn assigned to cluster k. For this reason, the procedure
is known as the K-means algorithm.
The two phases of re-assigning data points to clusters and re-computing the clus-
ter means are repeated in turn until there is no further change in the assignments (or
until some maximum number of iterations is exceeded). Because each phase reduces
the value of the objective function J, convergence of the algorithm is assured. How-
Exercise 9.1
ever, it may converge to a local rather than global minimum of J. The convergence
properties of the K-means algorithm were studied by MacQueen (1967).
The K-means algorithm is illustrated using the Old Faithful data set in Fig-
Appendix A
ure 9.1. For the purposes of this example, we have made a linear re-scaling of the
data, known as standardizing, such that each of the variables has zero mean and
unit standard deviation. For this example, we have chosen K = 2, and so in this


---
**Page 442**
426
9. MIXTURE MODELS AND EM
(a)
−2
0
2
−2
0
2
(b)
−2
0
2
−2
0
2
(c)
−2
0
2
−2
0
2
(d)
−2
0
2
−2
0
2
(e)
−2
0
2
−2
0
2
(f)
−2
0
2
−2
0
2
(g)
−2
0
2
−2
0
2
(h)
−2
0
2
−2
0
2
(i)
−2
0
2
−2
0
2
Figure 9.1
Illustration of the K-means algorithm using the re-scaled Old Faithful data set. (a) Green points
denote the data set in a two-dimensional Euclidean space. The initial choices for centres µ1 and µ2 are shown
by the red and blue crosses, respectively. (b) In the initial E step, each data point is assigned either to the red
cluster or to the blue cluster, according to which cluster centre is nearer. This is equivalent to classifying the
points according to which side of the perpendicular bisector of the two cluster centres, shown by the magenta
line, they lie on. (c) In the subsequent M step, each cluster centre is re-computed to be the mean of the points
assigned to the corresponding cluster. (d)–(i) show successive E and M steps through to ﬁnal convergence of
the algorithm.


---
**Page 443**
9.1. K-means Clustering
427
Figure 9.2
Plot of the cost function J given by
(9.1) after each E step (blue points)
and M step (red points) of the K-
means algorithm for the example
shown in Figure 9.1.
The algo-
rithm has converged after the third
M step, and the ﬁnal EM cycle pro-
duces no changes in either the as-
signments or the prototype vectors.
J
1
2
3
4
0
500
1000
case, the assignment of each data point to the nearest cluster centre is equivalent to a
classiﬁcation of the data points according to which side they lie of the perpendicular
bisector of the two cluster centres. A plot of the cost function J given by (9.1) for
the Old Faithful example is shown in Figure 9.2.
Note that we have deliberately chosen poor initial values for the cluster centres
so that the algorithm takes several steps before convergence. In practice, a better
initialization procedure would be to choose the cluster centres µk to be equal to a
random subset of K data points. It is also worth noting that the K-means algorithm
itself is often used to initialize the parameters in a Gaussian mixture model before
applying the EM algorithm.
Section 9.2.2
A direct implementation of the K-means algorithm as discussed here can be
relatively slow, because in each E step it is necessary to compute the Euclidean dis-
tance between every prototype vector and every data point. Various schemes have
been proposed for speeding up the K-means algorithm, some of which are based on
precomputing a data structure such as a tree such that nearby points are in the same
subtree (Ramasubramanian and Paliwal, 1990; Moore, 2000). Other approaches
make use of the triangle inequality for distances, thereby avoiding unnecessary dis-
tance calculations (Hodgson, 1998; Elkan, 2003).
So far, we have considered a batch version of K-means in which the whole data
set is used together to update the prototype vectors. We can also derive an on-line
stochastic algorithm (MacQueen, 1967) by applying the Robbins-Monro procedure
Section 2.3.5
to the problem of ﬁnding the roots of the regression function given by the derivatives
of J in (9.1) with respect to µk. This leads to a sequential update in which, for each
Exercise 9.2
data point xn in turn, we update the nearest prototype µk using
µnew
k
= µold
k
+ ηn(xn −µold
k )
(9.5)
where ηn is the learning rate parameter, which is typically made to decrease mono-
tonically as more data points are considered.
The K-means algorithm is based on the use of squared Euclidean distance as the
measure of dissimilarity between a data point and a prototype vector. Not only does
this limit the type of data variables that can be considered (it would be inappropriate
for cases where some or all of the variables represent categorical labels for instance),


---
**Page 444**
428
9. MIXTURE MODELS AND EM
but it can also make the determination of the cluster means nonrobust to outliers. We
Section 2.3.7
can generalize the K-means algorithm by introducing a more general dissimilarity
measure V(x, x′) between two vectors x and x′ and then minimizing the following
distortion measure
J =
N

n=1
K

k=1
rnkV(xn, µk)
(9.6)
which gives the K-medoids algorithm. The E step again involves, for given cluster
prototypes µk, assigning each data point to the cluster for which the dissimilarity to
the corresponding prototype is smallest. The computational cost of this is O(KN),
as is the case for the standard K-means algorithm. For a general choice of dissimi-
larity measure, the M step is potentially more complex than for K-means, and so it
is common to restrict each cluster prototype to be equal to one of the data vectors as-
signed to that cluster, as this allows the algorithm to be implemented for any choice
of dissimilarity measure V(·, ·) so long as it can be readily evaluated. Thus the M
step involves, for each cluster k, a discrete search over the Nk points assigned to that
cluster, which requires O(N 2
k) evaluations of V(·, ·).
One notable feature of the K-means algorithm is that at each iteration, every
data point is assigned uniquely to one, and only one, of the clusters. Whereas some
data points will be much closer to a particular centre µk than to any other centre,
there may be other data points that lie roughly midway between cluster centres. In
the latter case, it is not clear that the hard assignment to the nearest cluster is the
most appropriate. We shall see in the next section that by adopting a probabilistic
approach, we obtain ‘soft’ assignments of data points to clusters in a way that reﬂects
the level of uncertainty over the most appropriate assignment. This probabilistic
formulation brings with it numerous beneﬁts.
9.1.1
Image segmentation and compression
As an illustration of the application of the K-means algorithm, we consider
the related problems of image segmentation and image compression. The goal of
segmentation is to partition an image into regions each of which has a reasonably
homogeneous visual appearance or which corresponds to objects or parts of objects
(Forsyth and Ponce, 2003). Each pixel in an image is a point in a 3-dimensional space
comprising the intensities of the red, blue, and green channels, and our segmentation
algorithm simply treats each pixel in the image as a separate data point. Note that
strictly this space is not Euclidean because the channel intensities are bounded by
the interval [0, 1]. Nevertheless, we can apply the K-means algorithm without difﬁ-
culty. We illustrate the result of running K-means to convergence, for any particular
value of K, by re-drawing the image replacing each pixel vector with the {R, G, B}
intensity triplet given by the centre µk to which that pixel has been assigned. Results
for various values of K are shown in Figure 9.3. We see that for a given value of K,
the algorithm is representing the image using a palette of only K colours. It should
be emphasized that this use of K-means is not a particularly sophisticated approach
to image segmentation, not least because it takes no account of the spatial proximity
of different pixels. The image segmentation problem is in general extremely difﬁcult


---
**Page 445**
9.1. K-means Clustering
429
K = 2
K = 3
K = 10
Original image
Figure 9.3
Two examples of the application of the K-means clustering algorithm to image segmentation show-
ing the initial images together with their K-means segmentations obtained using various values of K. This
also illustrates of the use of vector quantization for data compression, in which smaller values of K give higher
compression at the expense of poorer image quality.
and remains the subject of active research and is introduced here simply to illustrate
the behaviour of the K-means algorithm.
We can also use the result of a clustering algorithm to perform data compres-
sion. It is important to distinguish between lossless data compression, in which
the goal is to be able to reconstruct the original data exactly from the compressed
representation, and lossy data compression, in which we accept some errors in the
reconstruction in return for higher levels of compression than can be achieved in the
lossless case. We can apply the K-means algorithm to the problem of lossy data
compression as follows. For each of the N data points, we store only the identity
k of the cluster to which it is assigned. We also store the values of the K clus-
ter centres µk, which typically requires signiﬁcantly less data, provided we choose
K ≪N. Each data point is then approximated by its nearest centre µk. New data
points can similarly be compressed by ﬁrst ﬁnding the nearest µk and then storing
the label k instead of the original data vector. This framework is often called vector
quantization, and the vectors µk are called code-book vectors.


---
**Page 446**
430
9. MIXTURE MODELS AND EM
The image segmentation problem discussed above also provides an illustration
of the use of clustering for data compression. Suppose the original image has N
pixels comprising {R, G, B} values each of which is stored with 8 bits of precision.
Then to transmit the whole image directly would cost 24N bits. Now suppose we
ﬁrst run K-means on the image data, and then instead of transmitting the original
pixel intensity vectors we transmit the identity of the nearest vector µk. Because
there are K such vectors, this requires log2 K bits per pixel. We must also transmit
the K code book vectors µk, which requires 24K bits, and so the total number of
bits required to transmit the image is 24K + N log2 K (rounding up to the nearest
integer). The original image shown in Figure 9.3 has 240 × 180 = 43, 200 pixels
and so requires 24 × 43, 200 = 1, 036, 800 bits to transmit directly. By comparison,
the compressed images require 43, 248 bits (K = 2), 86, 472 bits (K = 3), and
173, 040 bits (K = 10), respectively, to transmit. These represent compression ratios
compared to the original image of 4.2%, 8.3%, and 16.7%, respectively. We see that
there is a trade-off between degree of compression and image quality. Note that our
aim in this example is to illustrate the K-means algorithm. If we had been aiming to
produce a good image compressor, then it would be more fruitful to consider small
blocks of adjacent pixels, for instance 5×5, and thereby exploit the correlations that
exist in natural images between nearby pixels.
9.2. Mixtures of Gaussians
In Section 2.3.9 we motivated the Gaussian mixture model as a simple linear super-
position of Gaussian components, aimed at providing a richer class of density mod-
els than the single Gaussian. We now turn to a formulation of Gaussian mixtures in
terms of discrete latent variables. This will provide us with a deeper insight into this
important distribution, and will also serve to motivate the expectation-maximization
algorithm.
Recall from (2.188) that the Gaussian mixture distribution can be written as a
linear superposition of Gaussians in the form
p(x) =
K

k=1
πkN(x|µk, Σk).
(9.7)
Let us introduce a K-dimensional binary random variable z having a 1-of-K repre-
sentation in which a particular element zk is equal to 1 and all other elements are
equal to 0. The values of zk therefore satisfy zk ∈{0, 1} and 
k zk = 1, and we
see that there are K possible states for the vector z according to which element is
nonzero. We shall deﬁne the joint distribution p(x, z) in terms of a marginal dis-
tribution p(z) and a conditional distribution p(x|z), corresponding to the graphical
model in Figure 9.4. The marginal distribution over z is speciﬁed in terms of the
mixing coefﬁcients πk, such that
p(zk = 1) = πk


---
**Page 447**
9.2. Mixtures of Gaussians
431
Figure 9.4
Graphical representation of a mixture model, in which
the joint distribution is expressed in the form p(x, z) =
p(z)p(x|z).
x
z
where the parameters {πk} must satisfy
0 ⩽πk ⩽1
(9.8)
together with
K

k=1
πk = 1
(9.9)
in order to be valid probabilities. Because z uses a 1-of-K representation, we can
also write this distribution in the form
p(z) =
K

k=1
πzk
k .
(9.10)
Similarly, the conditional distribution of x given a particular value for z is a Gaussian
p(x|zk = 1) = N(x|µk, Σk)
which can also be written in the form
p(x|z) =
K

k=1
N(x|µk, Σk)zk.
(9.11)
The joint distribution is given by p(z)p(x|z), and the marginal distribution of x is
then obtained by summing the joint distribution over all possible states of z to give
Exercise 9.3
p(x) =

z
p(z)p(x|z) =
K

k=1
πkN(x|µk, Σk)
(9.12)
where we have made use of (9.10) and (9.11). Thus the marginal distribution of x is
a Gaussian mixture of the form (9.7). If we have several observations x1, . . . , xN,
then, because we have represented the marginal distribution in the form p(x) =

z p(x, z), it follows that for every observed data point xn there is a corresponding
latent variable zn.
We have therefore found an equivalent formulation of the Gaussian mixture in-
volving an explicit latent variable. It might seem that we have not gained much
by doing so. However, we are now able to work with the joint distribution p(x, z)


---
**Page 448**
432
9. MIXTURE MODELS AND EM
instead of the marginal distribution p(x), and this will lead to signiﬁcant simpliﬁca-
tions, most notably through the introduction of the expectation-maximization (EM)
algorithm.
Another quantity that will play an important role is the conditional probability
of z given x. We shall use γ(zk) to denote p(zk = 1|x), whose value can be found
using Bayes’ theorem
γ(zk) ≡p(zk = 1|x)
=
p(zk = 1)p(x|zk = 1)
K

j=1
p(zj = 1)p(x|zj = 1)
=
πkN(x|µk, Σk)
K

j=1
πjN(x|µj, Σj)
.
(9.13)
We shall view πk as the prior probability of zk = 1, and the quantity γ(zk) as the
corresponding posterior probability once we have observed x. As we shall see later,
γ(zk) can also be viewed as the responsibility that component k takes for ‘explain-
ing’ the observation x.
We can use the technique of ancestral sampling to generate random samples
Section 8.1.2
distributed according to the Gaussian mixture model. To do this, we ﬁrst generate a
value for z, which we denote z, from the marginal distribution p(z) and then generate
a value for x from the conditional distribution p(x|z). Techniques for sampling from
standard distributions are discussed in Chapter 11. We can depict samples from the
joint distribution p(x, z) by plotting points at the corresponding values of x and
then colouring them according to the value of z, in other words according to which
Gaussian component was responsible for generating them, as shown in Figure 9.5(a).
Similarly samples from the marginal distribution p(x) are obtained by taking the
samples from the joint distribution and ignoring the values of z. These are illustrated
in Figure 9.5(b) by plotting the x values without any coloured labels.
We can also use this synthetic data set to illustrate the ‘responsibilities’ by eval-
uating, for every data point, the posterior probability for each component in the
mixture distribution from which this data set was generated. In particular, we can
represent the value of the responsibilities γ(znk) associated with data point xn by
plotting the corresponding point using proportions of red, blue, and green ink given
by γ(znk) for k = 1, 2, 3, respectively, as shown in Figure 9.5(c). So, for instance,
a data point for which γ(zn1) = 1 will be coloured red, whereas one for which
γ(zn2) = γ(zn3) = 0.5 will be coloured with equal proportions of blue and green
ink and so will appear cyan. This should be compared with Figure 9.5(a) in which
the data points were labelled using the true identity of the component from which
they were generated.
9.2.1
Maximum likelihood
Suppose we have a data set of observations {x1, . . . , xN}, and we wish to model
this data using a mixture of Gaussians. We can represent this data set as an N × D


---
**Page 449**
9.2. Mixtures of Gaussians
433
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
(c)
0
0.5
1
0
0.5
1
Figure 9.5
Example of 500 points drawn from the mixture of 3 Gaussians shown in Figure 2.23. (a) Samples
from the joint distribution p(z)p(x|z) in which the three states of z, corresponding to the three components of the
mixture, are depicted in red, green, and blue, and (b) the corresponding samples from the marginal distribution
p(x), which is obtained by simply ignoring the values of z and just plotting the x values. The data set in (a) is
said to be complete, whereas that in (b) is incomplete. (c) The same samples in which the colours represent the
value of the responsibilities γ(znk) associated with data point xn, obtained by plotting the corresponding point
using proportions of red, blue, and green ink given by γ(znk) for k = 1, 2, 3, respectively
matrix X in which the nth row is given by xT
n. Similarly, the corresponding latent
variables will be denoted by an N × K matrix Z with rows zT
n. If we assume that
the data points are drawn independently from the distribution, then we can express
the Gaussian mixture model for this i.i.d. data set using the graphical representation
shown in Figure 9.6. From (9.7) the log of the likelihood function is given by
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
.
(9.14)
Before discussing how to maximize this function, it is worth emphasizing that
there is a signiﬁcant problem associated with the maximum likelihood framework
applied to Gaussian mixture models, due to the presence of singularities. For sim-
plicity, consider a Gaussian mixture whose components have covariance matrices
given by Σk = σ2
kI, where I is the unit matrix, although the conclusions will hold
for general covariance matrices. Suppose that one of the components of the mixture
model, let us say the jth component, has its mean µj exactly equal to one of the data
Figure 9.6
Graphical representation of a Gaussian mixture model
for a set of N i.i.d. data points {xn}, with corresponding
latent points {zn}, where n = 1, . . . , N.
xn
zn
N
µ
Σ
π


---
**Page 450**
434
9. MIXTURE MODELS AND EM
Figure 9.7
Illustration of how singularities in the
likelihood function arise with mixtures
of Gaussians.
This should be com-
pared with the case of a single Gaus-
sian shown in Figure 1.14 for which no
singularities arise.
x
p(x)
points so that µj = xn for some value of n. This data point will then contribute a
term in the likelihood function of the form
N(xn|xn, σ2
jI) =
1
(2π)1/2
1
σj
.
(9.15)
If we consider the limit σj →0, then we see that this term goes to inﬁnity and
so the log likelihood function will also go to inﬁnity. Thus the maximization of
the log likelihood function is not a well posed problem because such singularities
will always be present and will occur whenever one of the Gaussian components
‘collapses’ onto a speciﬁc data point. Recall that this problem did not arise in the
case of a single Gaussian distribution. To understand the difference, note that if a
single Gaussian collapses onto a data point it will contribute multiplicative factors
to the likelihood function arising from the other data points and these factors will go
to zero exponentially fast, giving an overall likelihood that goes to zero rather than
inﬁnity. However, once we have (at least) two components in the mixture, one of
the components can have a ﬁnite variance and therefore assign ﬁnite probability to
all of the data points while the other component can shrink onto one speciﬁc data
point and thereby contribute an ever increasing additive value to the log likelihood.
This is illustrated in Figure 9.7. These singularities provide another example of the
severe over-ﬁtting that can occur in a maximum likelihood approach. We shall see
that this difﬁculty does not occur if we adopt a Bayesian approach. For the moment,
Section 10.1
however, we simply note that in applying maximum likelihood to Gaussian mixture
models we must take steps to avoid ﬁnding such pathological solutions and instead
seek local maxima of the likelihood function that are well behaved. We can hope to
avoid the singularities by using suitable heuristics, for instance by detecting when a
Gaussian component is collapsing and resetting its mean to a randomly chosen value
while also resetting its covariance to some large value, and then continuing with the
optimization.
A further issue in ﬁnding maximum likelihood solutions arises from the fact
that for any given maximum likelihood solution, a K-component mixture will have
a total of K! equivalent solutions corresponding to the K! ways of assigning K
sets of parameters to K components. In other words, for any given (nondegenerate)
point in the space of parameter values there will be a further K!−1 additional points
all of which give rise to exactly the same distribution. This problem is known as


---
**Page 451**
9.2. Mixtures of Gaussians
435
identiﬁability (Casella and Berger, 2002) and is an important issue when we wish to
interpret the parameter values discovered by a model. Identiﬁability will also arise
when we discuss models having continuous latent variables in Chapter 12. However,
for the purposes of ﬁnding a good density model, it is irrelevant because any of the
equivalent solutions is as good as any other.
Maximizing the log likelihood function (9.14) for a Gaussian mixture model
turns out to be a more complex problem than for the case of a single Gaussian. The
difﬁculty arises from the presence of the summation over k that appears inside the
logarithm in (9.14), so that the logarithm function no longer acts directly on the
Gaussian. If we set the derivatives of the log likelihood to zero, we will no longer
obtain a closed form solution, as we shall see shortly.
One approach is to apply gradient-based optimization techniques (Fletcher, 1987;
Nocedal and Wright, 1999; Bishop and Nabney, 2008). Although gradient-based
techniques are feasible, and indeed will play an important role when we discuss
mixture density networks in Chapter 5, we now consider an alternative approach
known as the EM algorithm which has broad applicability and which will lay the
foundations for a discussion of variational inference techniques in Chapter 10.
9.2.2
EM for Gaussian mixtures
An elegant and powerful method for ﬁnding maximum likelihood solutions for
models with latent variables is called the expectation-maximization algorithm, or EM
algorithm (Dempster et al., 1977; McLachlan and Krishnan, 1997). Later we shall
give a general treatment of EM, and we shall also show how EM can be generalized
to obtain the variational inference framework. Initially, we shall motivate the EM
Section 10.1
algorithm by giving a relatively informal treatment in the context of the Gaussian
mixture model. We emphasize, however, that EM has broad applicability, and indeed
it will be encountered in the context of a variety of different models in this book.
Let us begin by writing down the conditions that must be satisﬁed at a maximum
of the likelihood function. Setting the derivatives of ln p(X|π, µ, Σ) in (9.14) with
respect to the means µk of the Gaussian components to zero, we obtain
0 = −
N

n=1
πkN(xn|µk, Σk)

j πjN(xn|µj, Σj)
(
)*
+
γ(znk)
Σk(xn −µk)
(9.16)
where we have made use of the form (2.43) for the Gaussian distribution. Note that
the posterior probabilities, or responsibilities, given by (9.13) appear naturally on
the right-hand side. Multiplying by Σ−1
k
(which we assume to be nonsingular) and
rearranging we obtain
µk = 1
Nk
N

n=1
γ(znk)xn
(9.17)
where we have deﬁned
Nk =
N

n=1
γ(znk).
(9.18)


---
**Page 452**
436
9. MIXTURE MODELS AND EM
We can interpret Nk as the effective number of points assigned to cluster k. Note
carefully the form of this solution. We see that the mean µk for the kth Gaussian
component is obtained by taking a weighted mean of all of the points in the data set,
in which the weighting factor for data point xn is given by the posterior probability
γ(znk) that component k was responsible for generating xn.
If we set the derivative of ln p(X|π, µ, Σ) with respect to Σk to zero, and follow
a similar line of reasoning, making use of the result for the maximum likelihood
solution for the covariance matrix of a single Gaussian, we obtain
Section 2.3.4
Σk = 1
Nk
N

n=1
γ(znk)(xn −µk)(xn −µk)T
(9.19)
which has the same form as the corresponding result for a single Gaussian ﬁtted to
the data set, but again with each data point weighted by the corresponding poste-
rior probability and with the denominator given by the effective number of points
associated with the corresponding component.
Finally, we maximize ln p(X|π, µ, Σ) with respect to the mixing coefﬁcients
πk. Here we must take account of the constraint (9.9), which requires the mixing
coefﬁcients to sum to one. This can be achieved using a Lagrange multiplier and
Appendix E
maximizing the following quantity
ln p(X|π, µ, Σ) + λ
 K

k=1
πk −1

(9.20)
which gives
0 =
N

n=1
N(xn|µk, Σk)

j πjN(xn|µj, Σj) + λ
(9.21)
where again we see the appearance of the responsibilities. If we now multiply both
sides by πk and sum over k making use of the constraint (9.9), we ﬁnd λ = −N.
Using this to eliminate λ and rearranging we obtain
πk = Nk
N
(9.22)
so that the mixing coefﬁcient for the kth component is given by the average respon-
sibility which that component takes for explaining the data points.
It is worth emphasizing that the results (9.17), (9.19), and (9.22) do not con-
stitute a closed-form solution for the parameters of the mixture model because the
responsibilities γ(znk) depend on those parameters in a complex way through (9.13).
However, these results do suggest a simple iterative scheme for ﬁnding a solution to
the maximum likelihood problem, which as we shall see turns out to be an instance
of the EM algorithm for the particular case of the Gaussian mixture model. We
ﬁrst choose some initial values for the means, covariances, and mixing coefﬁcients.
Then we alternate between the following two updates that we shall call the E step


---
**Page 453**
9.2. Mixtures of Gaussians
437
(a)
−2
0
2
−2
0
2
(b)
−2
0
2
−2
0
2
(c)
L = 1
−2
0
2
−2
0
2
(d)
L = 2
−2
0
2
−2
0
2
(e)
L = 5
−2
0
2
−2
0
2
(f)
L = 20
−2
0
2
−2
0
2
Figure 9.8
Illustration of the EM algorithm using the Old Faithful set as used for the illustration of the K-means
algorithm in Figure 9.1. See the text for details.
and the M step, for reasons that will become apparent shortly. In the expectation
step, or E step, we use the current values for the parameters to evaluate the posterior
probabilities, or responsibilities, given by (9.13). We then use these probabilities in
the maximization step, or M step, to re-estimate the means, covariances, and mix-
ing coefﬁcients using the results (9.17), (9.19), and (9.22). Note that in so doing
we ﬁrst evaluate the new means using (9.17) and then use these new values to ﬁnd
the covariances using (9.19), in keeping with the corresponding result for a single
Gaussian distribution. We shall show that each update to the parameters resulting
from an E step followed by an M step is guaranteed to increase the log likelihood
function. In practice, the algorithm is deemed to have converged when the change
Section 9.4
in the log likelihood function, or alternatively in the parameters, falls below some
threshold. We illustrate the EM algorithm for a mixture of two Gaussians applied to
the rescaled Old Faithful data set in Figure 9.8.
Here a mixture of two Gaussians
is used, with centres initialized using the same values as for the K-means algorithm
in Figure 9.1, and with precision matrices initialized to be proportional to the unit
matrix. Plot (a) shows the data points in green, together with the initial conﬁgura-
tion of the mixture model in which the one standard-deviation contours for the two


---
**Page 454**
438
9. MIXTURE MODELS AND EM
Gaussian components are shown as blue and red circles. Plot (b) shows the result
of the initial E step, in which each data point is depicted using a proportion of blue
ink equal to the posterior probability of having been generated from the blue com-
ponent, and a corresponding proportion of red ink given by the posterior probability
of having been generated by the red component. Thus, points that have a signiﬁcant
probability for belonging to either cluster appear purple. The situation after the ﬁrst
M step is shown in plot (c), in which the mean of the blue Gaussian has moved to
the mean of the data set, weighted by the probabilities of each data point belonging
to the blue cluster, in other words it has moved to the centre of mass of the blue ink.
Similarly, the covariance of the blue Gaussian is set equal to the covariance of the
blue ink. Analogous results hold for the red component. Plots (d), (e), and (f) show
the results after 2, 5, and 20 complete cycles of EM, respectively. In plot (f) the
algorithm is close to convergence.
Note that the EM algorithm takes many more iterations to reach (approximate)
convergence compared with the K-means algorithm, and that each cycle requires
signiﬁcantly more computation. It is therefore common to run the K-means algo-
rithm in order to ﬁnd a suitable initialization for a Gaussian mixture model that is
subsequently adapted using EM. The covariance matrices can conveniently be ini-
tialized to the sample covariances of the clusters found by the K-means algorithm,
and the mixing coefﬁcients can be set to the fractions of data points assigned to the
respective clusters. As with gradient-based approaches for maximizing the log like-
lihood, techniques must be employed to avoid singularities of the likelihood function
in which a Gaussian component collapses onto a particular data point. It should be
emphasized that there will generally be multiple local maxima of the log likelihood
function, and that EM is not guaranteed to ﬁnd the largest of these maxima. Because
the EM algorithm for Gaussian mixtures plays such an important role, we summarize
it below.
EM for Gaussian Mixtures
Given a Gaussian mixture model, the goal is to maximize the likelihood function
with respect to the parameters (comprising the means and covariances of the
components and the mixing coefﬁcients).
1. Initialize the means µk, covariances Σk and mixing coefﬁcients πk, and
evaluate the initial value of the log likelihood.
2. E step. Evaluate the responsibilities using the current parameter values
γ(znk) =
πkN(xn|µk, Σk)
K

j=1
πjN(xn|µj, Σj)
.
(9.23)


---
**Page 455**
9.3. An Alternative View of EM
439
3. M step. Re-estimate the parameters using the current responsibilities
µnew
k
=
1
Nk
N

n=1
γ(znk)xn
(9.24)
Σnew
k
=
1
Nk
N

n=1
γ(znk) (xn −µnew
k
) (xn −µnew
k
)T
(9.25)
πnew
k
=
Nk
N
(9.26)
where
Nk =
N

n=1
γ(znk).
(9.27)
4. Evaluate the log likelihood
ln p(X|µ, Σ, π) =
N

n=1
ln
 K

k=1
πkN(xn|µk, Σk)

(9.28)
and check for convergence of either the parameters or the log likelihood. If
the convergence criterion is not satisﬁed return to step 2.
9.3. An Alternative View of EM
In this section, we present a complementary view of the EM algorithm that recog-
nizes the key role played by latent variables. We discuss this approach ﬁrst of all
in an abstract setting, and then for illustration we consider once again the case of
Gaussian mixtures.
The goal of the EM algorithm is to ﬁnd maximum likelihood solutions for mod-
els having latent variables. We denote the set of all observed data by X, in which the
nth row represents xT
n, and similarly we denote the set of all latent variables by Z,
with a corresponding row zT
n. The set of all model parameters is denoted by θ, and
so the log likelihood function is given by
ln p(X|θ) = ln

Z
p(X, Z|θ)

.
(9.29)
Note that our discussion will apply equally well to continuous latent variables simply
by replacing the sum over Z with an integral.
A key observation is that the summation over the latent variables appears inside
the logarithm. Even if the joint distribution p(X, Z|θ) belongs to the exponential


---
**Page 456**
440
9. MIXTURE MODELS AND EM
family, the marginal distribution p(X|θ) typically does not as a result of this sum-
mation. The presence of the sum prevents the logarithm from acting directly on the
joint distribution, resulting in complicated expressions for the maximum likelihood
solution.
Now suppose that, for each observation in X, we were told the corresponding
value of the latent variable Z. We shall call {X, Z} the complete data set, and we
shall refer to the actual observed data X as incomplete, as illustrated in Figure 9.5.
The likelihood function for the complete data set simply takes the form ln p(X, Z|θ),
and we shall suppose that maximization of this complete-data log likelihood function
is straightforward.
In practice, however, we are not given the complete data set {X, Z}, but only
the incomplete data X. Our state of knowledge of the values of the latent variables
in Z is given only by the posterior distribution p(Z|X, θ). Because we cannot use
the complete-data log likelihood, we consider instead its expected value under the
posterior distribution of the latent variable, which corresponds (as we shall see) to the
E step of the EM algorithm. In the subsequent M step, we maximize this expectation.
If the current estimate for the parameters is denoted θold, then a pair of successive
E and M steps gives rise to a revised estimate θnew. The algorithm is initialized by
choosing some starting value for the parameters θ0. The use of the expectation may
seem somewhat arbitrary. However, we shall see the motivation for this choice when
we give a deeper treatment of EM in Section 9.4.
In the E step, we use the current parameter values θold to ﬁnd the posterior
distribution of the latent variables given by p(Z|X, θold). We then use this posterior
distribution to ﬁnd the expectation of the complete-data log likelihood evaluated for
some general parameter value θ. This expectation, denoted Q(θ, θold), is given by
Q(θ, θold) =

Z
p(Z|X, θold) ln p(X, Z|θ).
(9.30)
In the M step, we determine the revised parameter estimate θnew by maximizing this
function
θnew = arg max
θ
Q(θ, θold).
(9.31)
Note that in the deﬁnition of Q(θ, θold), the logarithm acts directly on the joint
distribution p(X, Z|θ), and so the corresponding M-step maximization will, by sup-
position, be tractable.
The general EM algorithm is summarized below. It has the property, as we shall
show later, that each cycle of EM will increase the incomplete-data log likelihood
(unless it is already at a local maximum).
Section 9.4
The General EM Algorithm
Given a joint distribution p(X, Z|θ) over observed variables X and latent vari-
ables Z, governed by parameters θ, the goal is to maximize the likelihood func-
tion p(X|θ) with respect to θ.
1. Choose an initial setting for the parameters θold.


---
**Page 457**
9.3. An Alternative View of EM
441
2. E step Evaluate p(Z|X, θold).
3. M step Evaluate θnew given by
θnew = arg max
θ
Q(θ, θold)
(9.32)
where
Q(θ, θold) =

Z
p(Z|X, θold) ln p(X, Z|θ).
(9.33)
4. Check for convergence of either the log likelihood or the parameter values.
If the convergence criterion is not satisﬁed, then let
θold ←θnew
(9.34)
and return to step 2.
The EM algorithm can also be used to ﬁnd MAP (maximum posterior) solutions
for models in which a prior p(θ) is deﬁned over the parameters. In this case the E
Exercise 9.4
step remains the same as in the maximum likelihood case, whereas in the M step the
quantity to be maximized is given by Q(θ, θold) + ln p(θ). Suitable choices for the
prior will remove the singularities of the kind illustrated in Figure 9.7.
Here we have considered the use of the EM algorithm to maximize a likelihood
function when there are discrete latent variables. However, it can also be applied
when the unobserved variables correspond to missing values in the data set. The
distribution of the observed values is obtained by taking the joint distribution of all
the variables and then marginalizing over the missing ones. EM can then be used
to maximize the corresponding likelihood function. We shall show an example of
the application of this technique in the context of principal component analysis in
Figure 12.11. This will be a valid procedure if the data values are missing at random,
meaning that the mechanism causing values to be missing does not depend on the
unobserved values. In many situations this will not be the case, for instance if a
sensor fails to return a value whenever the quantity it is measuring exceeds some
threshold.
9.3.1
Gaussian mixtures revisited
We now consider the application of this latent variable view of EM to the spe-
ciﬁc case of a Gaussian mixture model. Recall that our goal is to maximize the log
likelihood function (9.14), which is computed using the observed data set X, and we
saw that this was more difﬁcult than for the case of a single Gaussian distribution
due to the presence of the summation over k that occurs inside the logarithm. Sup-
pose then that in addition to the observed data set X, we were also given the values
of the corresponding discrete variables Z. Recall that Figure 9.5(a) shows a ‘com-
plete’ data set (i.e., one that includes labels showing which component generated
each data point) while Figure 9.5(b) shows the corresponding ‘incomplete’ data set.
The graphical model for the complete data is shown in Figure 9.9.


---
**Page 458**
442
9. MIXTURE MODELS AND EM
Figure 9.9
This shows the same graph as in Figure 9.6 except that
we now suppose that the discrete variables zn are ob-
served, as well as the data variables xn.
xn
zn
N
µ
Σ
π
Now consider the problem of maximizing the likelihood for the complete data
set {X, Z}. From (9.10) and (9.11), this likelihood function takes the form
p(X, Z|µ, Σ, π) =
N

n=1
K

k=1
πznk
k
N(xn|µk, Σk)znk
(9.35)
where znk denotes the kth component of zn. Taking the logarithm, we obtain
ln p(X, Z|µ, Σ, π) =
N

n=1
K

k=1
znk {ln πk + ln N(xn|µk, Σk)} .
(9.36)
Comparison with the log likelihood function (9.14) for the incomplete data shows
that the summation over k and the logarithm have been interchanged. The loga-
rithm now acts directly on the Gaussian distribution, which itself is a member of
the exponential family. Not surprisingly, this leads to a much simpler solution to
the maximum likelihood problem, as we now show. Consider ﬁrst the maximization
with respect to the means and covariances. Because zn is a K-dimensional vec-
tor with all elements equal to 0 except for a single element having the value 1, the
complete-data log likelihood function is simply a sum of K independent contribu-
tions, one for each mixture component. Thus the maximization with respect to a
mean or a covariance is exactly as for a single Gaussian, except that it involves only
the subset of data points that are ‘assigned’ to that component. For the maximization
with respect to the mixing coefﬁcients, we note that these are coupled for different
values of k by virtue of the summation constraint (9.9). Again, this can be enforced
using a Lagrange multiplier as before, and leads to the result
πk = 1
N
N

n=1
znk
(9.37)
so that the mixing coefﬁcients are equal to the fractions of data points assigned to
the corresponding components.
Thus we see that the complete-data log likelihood function can be maximized
trivially in closed form. In practice, however, we do not have values for the latent
variables so, as discussed earlier, we consider the expectation, with respect to the
posterior distribution of the latent variables, of the complete-data log likelihood.


---
**Page 459**
9.3. An Alternative View of EM
443
Using (9.10) and (9.11) together with Bayes’ theorem, we see that this posterior
distribution takes the form
p(Z|X, µ, Σ, π) ∝
N

n=1
K

k=1
[πkN(xn|µk, Σk)]znk .
(9.38)
and hence factorizes over n so that under the posterior distribution the {zn} are
independent. This is easily veriﬁed by inspection of the directed graph in Figure 9.6
Exercise 9.5
and making use of the d-separation criterion. The expected value of the indicator
Section 8.2
variable znk under this posterior distribution is then given by
E[znk]
=

znk
znk [πkN(xn|µk, Σk)]znk

znj

πjN(xn|µj, Σj)
	znj
=
πkN(xn|µk, Σk)
K

j=1
πjN(xn|µj, Σj)
= γ(znk)
(9.39)
which is just the responsibility of component k for data point xn. The expected value
of the complete-data log likelihood function is therefore given by
EZ[ln p(X, Z|µ, Σ, π)] =
N

n=1
K

k=1
γ(znk) {ln πk + ln N(xn|µk, Σk)} .
(9.40)
We can now proceed as follows. First we choose some initial values for the param-
eters µold, Σold and πold, and use these to evaluate the responsibilities (the E step).
We then keep the responsibilities ﬁxed and maximize (9.40) with respect to µk, Σk
and πk (the M step). This leads to closed form solutions for µnew, Σnew and πnew
given by (9.17), (9.19), and (9.22) as before. This is precisely the EM algorithm for
Exercise 9.8
Gaussian mixtures as derived earlier. We shall gain more insight into the role of the
expected complete-data log likelihood function when we give a proof of convergence
of the EM algorithm in Section 9.4.
9.3.2
Relation to K-means
Comparison of the K-means algorithm with the EM algorithm for Gaussian
mixtures shows that there is a close similarity. Whereas the K-means algorithm
performs a hard assignment of data points to clusters, in which each data point is
associated uniquely with one cluster, the EM algorithm makes a soft assignment
based on the posterior probabilities. In fact, we can derive the K-means algorithm
as a particular limit of EM for Gaussian mixtures as follows.
Consider a Gaussian mixture model in which the covariance matrices of the
mixture components are given by ϵI, where ϵ is a variance parameter that is shared


---
**Page 460**
444
9. MIXTURE MODELS AND EM
by all of the components, and I is the identity matrix, so that
p(x|µk, Σk) =
1
(2πϵ)1/2 exp

−1
2ϵ∥x −µk∥2

.
(9.41)
We now consider the EM algorithm for a mixture of K Gaussians of this form in
which we treat ϵ as a ﬁxed constant, instead of a parameter to be re-estimated. From
(9.13) the posterior probabilities, or responsibilities, for a particular data point xn,
are given by
γ(znk) =
πk exp {−∥xn −µk∥2/2ϵ}

j πj exp 
−∥xn −µj∥2/2ϵ.
(9.42)
If we consider the limit ϵ →0, we see that in the denominator the term for which
∥xn −µj∥2 is smallest will go to zero most slowly, and hence the responsibilities
γ(znk) for the data point xn all go to zero except for term j, for which the responsi-
bility γ(znj) will go to unity. Note that this holds independently of the values of the
πk so long as none of the πk is zero. Thus, in this limit, we obtain a hard assignment
of data points to clusters, just as in the K-means algorithm, so that γ(znk) →rnk
where rnk is deﬁned by (9.2). Each data point is thereby assigned to the cluster
having the closest mean.
The EM re-estimation equation for the µk, given by (9.17), then reduces to the
K-means result (9.4). Note that the re-estimation formula for the mixing coefﬁcients
(9.22) simply re-sets the value of πk to be equal to the fraction of data points assigned
to cluster k, although these parameters no longer play an active role in the algorithm.
Finally, in the limit ϵ →0 the expected complete-data log likelihood, given by
(9.40), becomes
Exercise 9.11
EZ[ln p(X, Z|µ, Σ, π)] →−1
2
N

n=1
K

k=1
rnk∥xn −µk∥2 + const.
(9.43)
Thus we see that in this limit, maximizing the expected complete-data log likelihood
is equivalent to minimizing the distortion measure J for the K-means algorithm
given by (9.1).
Note that the K-means algorithm does not estimate the covariances of the clus-
ters but only the cluster means. A hard-assignment version of the Gaussian mixture
model with general covariance matrices, known as the elliptical K-means algorithm,
has been considered by Sung and Poggio (1994).
9.3.3
Mixtures of Bernoulli distributions
So far in this chapter, we have focussed on distributions over continuous vari-
ables described by mixtures of Gaussians. As a further example of mixture mod-
elling, and to illustrate the EM algorithm in a different context, we now discuss mix-
tures of discrete binary variables described by Bernoulli distributions. This model
is also known as latent class analysis (Lazarsfeld and Henry, 1968; McLachlan and
Peel, 2000). As well as being of practical importance in its own right, our discus-
sion of Bernoulli mixtures will also lay the foundation for a consideration of hidden
Markov models over discrete variables.
Section 13.2

