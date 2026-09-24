# ShellMetric: Final Implementation and Experiment Specification

> **Status:** Final design approved for implementation, 21 September 2026.  
> **Authority:** This is the only design document for the new method. It supersedes
> `MULTISHELL_RESEARCH_IMPLEMENTATION_SPEC.md` wherever that document uses class
> prototypes, codebooks, fixed class directions, manual shell maps, or
> nearest-prototype training. The existing CA-MSPC code is a legacy baseline, not
> the implementation target described here.

## 1. Exact research object

The proposed method is **supervised, prototype-free, multi-shell metric
learning**. Its working name is **ShellMetric**.

It learns an embedding

\[
z=f_\theta(x)\in\mathbb R^d
\]

whose norm places a sample near a class-assigned shell and whose direction is
free to organize naturally. A class occupies a localized patch on a shell. It
is not assigned an angular coordinate, direction, center, proxy, or prototype.
Several classes share each shell.

The preliminary confusion matrix decides which classes use which shells:

- easy classes are always assigned inward;
- hard classes are always assigned outward;
- inner shells contain fewer classes;
- outer shells contain more classes;
- shell radii are learned continuously during representation training for
  \(S>1\), while \(\rho_1=1\) for \(S=1\);
- class-to-shell assignments are frozen before representation training;
- no manual class assignment exists.

Class labels are deliberately used. They are used to train the preliminary CE
model, construct confusion, assign classes to shells, select positive and
negative metric pairs, choose each sample's shell, and later train a new linear
decoder while the encoder remains frozen. “Natural clusters” means **no
prescribed angular class point**;
it does not mean unsupervised learning.

### 1.1 Locked method families

There are only two proposed method families:

1. **ShellMetric-FixedS:** the experiment supplies a shell count \(S\), while
   capacities, class assignment, and radii remain automatic.
2. **ShellMetric-AutoK:** the confusion data automatically select \(S\). This is
   the flagship method used across datasets, dimensions, and backbones.

`FixedS` is a sweep family, not one run fixed at \(S_{\max}\). A single suite
command must accept multiple counts and launch all of them.

### 1.2 Explicitly forbidden in the new method

The new representation-training path must not contain:

- a class prototype, proxy, centroid parameter, or class direction;
- an angular target for any class;
- a trainable tensor of class points with shape `[C, d]`;
- prototype-distance cross-entropy or prototype compactness;
- a classifier or decoder head attached during representation training;
- manual class-to-shell assignment, including a hidden config escape hatch;
- fixed radius candidate tables;
- nearest-prototype inference;
- a loss that spreads every individual class across an entire shell;
- surface-uniformity, center-loss, or other optional regularizers in version 1.

Learned shell radii are allowed because there are only \(S\) shared scalar
radii, not one representative per class.

## 2. End-to-end data flow

The pipeline is strictly staged:

```text
fixed train/validation/test manifest
        |
training split only
        v
cross-fitted ordinary CE pilot
        |
OOF probabilities + true training labels
        v
soft symmetric confusion W and class difficulty h
        |
        +--> FixedS plans for every requested S
        |
        +--> AutoK selection and AutoK plan
        v
train a fresh prototype-free representation encoder for each plan
        |
freeze encoder and batch-normalization state
        +--> raw Euclidean kNN/retrieval evaluation
        +--> ShellMetric-only shell-then-cosine kNN
        +--> separately trained post-hoc linear softmax probe
```

The CE pilot is discarded after producing out-of-fold predictions. It is not a
warm start for ShellMetric. The shell plan is not updated online.

The primary publication pipeline uses the ordinary ReLU ResNet and
`linear_no_bias` output so every in-repository method has a conventional,
architecture-matched comparison. Section 5.4 is a bounded auxiliary
architecture study run only after the primary ShellMetric hyperparameters and
official ReLU-derived pilot/plan are frozen. It does not feed a
validation-selected architecture back into the pilot or silently redefine the
flagship method.

## 3. Leakage-safe preliminary confusion

### 3.1 Splits

Create one immutable, stratified split manifest per dataset:

- `train`: pilot construction, representation training, galleries, and
  post-hoc-probe training;
- `validation`: checkpoint/hyperparameter/post-hoc-probe selection only;
- `test`: final reporting only.

Use split seed `12345`. For MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100,
reserve a stratified 10% of the official training set as validation and retain
the official test set as sealed test. For CUB-200, preserve the official
train/test split and reserve 10% of the official training examples per class
for validation. For TinyImageNet, reserve 10% of the official train
split for validation and use the labeled official validation split as test.
Save dataset version/checksums and exact sample IDs; never infer a split again
from the random seed once the manifest exists.

No validation or test label, prediction, embedding, or metric may influence
the confusion matrix, AutoK, or class assignment. Validation may select
predeclared hyperparameters and checkpoints after the shell plan is frozen.
Test data may influence nothing until final reporting.

### 3.2 Pilot

Use an ordinary affine-softmax classifier trained with cross-entropy:

- take the backbone's standard final pooled feature of width \(H\) and attach
  `Linear(H,C)` directly; do not insert the downstream \(d\)-dimensional
  bottleneck, shell module, metric projection, or margin head;
- same backbone family and data preprocessing as the downstream experiment;
- stratified 3-fold cross-fitting on the training split;
- publication runs use pilot seeds `[0, 1, 2]`;
- every training sample is predicted only by models that did not train on it;
- average raw OOF softmax probabilities over pilot seeds;
- use deterministic evaluation transforms when producing OOF probabilities.

Train each pilot fold with the applicable classification optimizer,
augmentation, and epoch schedule in Section 8.4. The pilot has an ordinary
batch size of 128 and uses the final scheduled checkpoint; the held-out fold
and global validation split are never used for pilot checkpoint selection. It
never uses the later metric sampler.

Do not fit a temperature on the pooled OOF labels: doing so would let each
sample's label alter the calibration applied to its own nominally held-out
prediction. Version 1 uses uncalibrated OOF softmax probabilities instead of a
nested calibration stage.

Do not mix CE, ArcFace, CosFace, or metric losses to make the pilot. A single
CE definition keeps the source of confusion interpretable.

Cache the OOF artifact once per dataset/backbone/inner-activation/pilot
configuration. Reuse it across downstream dimensions, shell counts, losses,
controls, and encoder seeds. Section 5.4 is an explicit controlled exception:
all activation candidates consume the same ReLU-derived pilot/plan, and no
candidate-specific pilot is built.

### 3.3 Confusion and difficulty

For \(C\) classes, let \(p_j(x)\) be the averaged raw OOF softmax probability
for class \(j\). Define directed soft confusion

\[
Q_{ij}=\frac{1}{N_i}\sum_{x:y=i}p_j(x).
\]

Then define the raw symmetric off-diagonal confusion graph

\[
W_{ij}=\begin{cases}
\tfrac12(Q_{ij}+Q_{ji}),&i\ne j,\\
0,&i=j.
\end{cases}
\]

Class difficulty is

\[
h_i=\sum_{j\ne i}W_{ij}.
\]

Use this **raw** \(W\) for difficulty and AutoK. Do not add a uniform floor to
it. For the negative-pair margin, normalize it as

\[
\widehat W_{ij}=\frac{W_{ij}}
{\max_{a<b}W_{ab}+\varepsilon}.
\]

If every off-diagonal element is numerically zero, set \(\widehat W=0\). The
base negative margin still protects every class pair, so no confusion floor is
needed.

The confusion artifact must save `Q`, `W`, `W_hat`, `h`, class order, sample
IDs, fold IDs, pilot seeds, split hash, and a content hash.

## 4. Automatic shell planning

### 4.1 Maximum allowed shell count

For \(C\) classes,

\[
S_{\max}(C)=\min\bigl(C,\lceil\sqrt C\rceil\bigr).
\]

Examples are \(S_{\max}(10)=4\), \(S_{\max}(100)=10\), and
\(S_{\max}(1000)=32\). This is a safety/search bound, not the FixedS shell
count.

### 4.2 Unequal capacities

For a requested \(S\), shell indices run from \(s=1\) (inner) to \(S\)
(outer). Use the fixed capacity prior

\[
w_s=s^2.
\]

This is a controlled prior inspired by surface-area growth in the primary
three-dimensional setting. It remains dimension-independent so changing
embedding dimension does not silently change the class assignment. It is not
claimed to be the exact hypersurface-area formula in every dimension.

Convert the weights to deterministic integer capacities as follows:

1. Initialize \(n_s=1\) for every shell.
2. For each of the remaining \(C-S\) class slots, increment the shell

   \[
   s^*=\arg\max_s\frac{s^2}{n_s+1}.
   \]

3. Break exact ties in favor of the larger shell index.

Implement the comparison with integer cross-multiplication rather than
floating division. To compare candidates \(s\) and \(t\), compare
`s*s*(n_t+1)` with `t*t*(n_s+1)` and then apply the outward tie-break. This
makes capacities platform-independent.

The result must satisfy

\[
n_s\ge1,\qquad \sum_s n_s=C,\qquad
n_1\le n_2\le\cdots\le n_S.
\]

For \(C=10,S=4\), the required result is `[1, 1, 3, 5]`.

### 4.3 Class assignment for any fixed S

Sort classes by `(difficulty, class_id)` in ascending order. Fill contiguous
capacity-sized chunks from the inner shell outward. Therefore, apart from exact
difficulty ties, a harder class can never be assigned inward of an easier
class. The most difficult classes are always on the outermost shell.

Only the shell index is assigned. There is no within-shell angular assignment.

### 4.4 FixedS sweep

The suite runner must support all four forms:

```yaml
study:
  fixed_shell_counts: compact        # automatic economical sweep
# fixed_shell_counts: all            # every integer 1..Smax
# fixed_shell_counts: [1, 2, 4, 8]   # explicit user list
# fixed_shell_counts: none            # AutoK-only benchmark cell
```

`compact` expands as follows:

- if \(S_{\max}\le6\), use every integer from 1 through \(S_{\max}\);
- otherwise use the unique sorted values
  `[1, ceil(Smax/4), ceil(Smax/2), ceil(3*Smax/4), Smax]`.

Examples:

| Classes | \(S_{\max}\) | Compact sweep |
|---:|---:|---|
| 10 | 4 | 1, 2, 3, 4 |
| 100 | 10 | 1, 3, 5, 8, 10 |
| 200 | 15 | 1, 4, 8, 12, 15 |
| 1000 | 32 | 1, 8, 16, 24, 32 |

Validate every explicit value against \([1,S_{\max}]\), then sort and
deduplicate it. An explicit list is honored exactly. `compact` and `all`
include \(S=1\) as shown. `none` (or `[]`) launches no FixedS jobs. The
canonical control suite sets `ensure_one_shell_control: true`, which adds
`FixedS(1)` only if it is absent. Other experiment cells leave that option
false. `all` is deliberately available when compute permits a full curve.

AutoK always evaluates every integer in \([1,S_{\max}]\); the FixedS sweep
list must never restrict AutoK's candidates.

### 4.5 AutoK: bootstrap predictive-risk selection

AutoK uses only cached OOF probabilities and their labels. It does not inspect
representation training or validation accuracy.

Use \(B=200\) deterministic, paired, class-stratified bootstrap repetitions.
For every repetition \(b\), independently resample the saved OOF examples
within each true class twice, producing difficulties \(h^{A,b}\) and
\(h^{B,b}\). For each resample, recompute \(Q\), then symmetric \(W\), then
\(h\) exactly as in Section 3.3. Use a bootstrap seed derived from the
confusion artifact hash.

Perform planning in float64. Let

\[
h_{min}=\min_i h_i^{full},\qquad
D_h=\max_i h_i^{full}-h_{min}.
\]

If \(D_h<10^{-12}\), immediately return \(S^*=1\). Otherwise transform every
full-data and bootstrap difficulty using

\[
\widetilde h=(h-h_{min})/D_h.
\]

Do not clip bootstrap values that fall outside `[0,1]`, and do not normalize
each bootstrap independently. Use \(\widetilde h\), not raw \(h\), in all
AutoK sorting, shell means, and risks below. Break bootstrap sorting ties by
class ID.

For each candidate \(S=1,\ldots,S_{\max}\) and each repetition:

1. Construct the deterministic \(s^2\) capacities.
2. Sort classes by \(\widetilde h^{A,b}\) and assign contiguous chunks.
3. Compute the mean normalized difficulty
   \(\widetilde\mu_s^{A,b}\) in each resulting shell.
4. Evaluate that plan on the independent bootstrap:

   \[
   R_S^{A\rightarrow B,b}=\frac1C\sum_i
   \left(\widetilde h_i^{B,b}
   -\widetilde\mu_{a^{A,b}(i)}^{A,b}\right)^2.
   \]

5. Exchange A and B and define

   \[
   R_S^b=\tfrac12\left(R_S^{A\rightarrow B,b}
   +R_S^{B\rightarrow A,b}\right).
   \]

Aggregate

\[
\bar R_S=\frac1B\sum_bR_S^b,
\qquad
SE_S=\frac{\operatorname{sd}_b(R_S^b)}{\sqrt B}.
\]

Let \(S_{best}\) minimize \(\bar R_S\), breaking exact ties toward smaller
\(S\). Apply the one-standard-error rule:

\[
S^*=\min\left\{S:\bar R_S\le
\bar R_{S_{best}}+SE_{S_{best}}\right\}.
\]

Finally, recompute capacities and class assignment from the full, unbootstrapped
\(h\) using \(S^*\). Save all candidate risks, standard errors, the chosen
count, bootstrap seed, `plan_provenance_hash`, and `plan_semantic_hash` as
defined in Section 7.

This criterion has no hand-tuned shell-count penalty. A saturated candidate
cannot win merely by fitting every class separately because it is scored on an
independent resample. Flat or unstable difficulties favor one shell; stable
difficulty bands justify more shells.

## 5. Representation model and objective

### 5.1 Encoder

The primary canonical output is a bias-free raw Cartesian projection from the
backbone feature \(h\in\mathbb R^H\):

\[
z=W h,\qquad W\in\mathbb R^{d\times H},\qquad \text{bias}=\text{false}.
\]

All primary scratch ResNets use `nn.ReLU(inplace=False)` at their standard
activation sites. This makes the primary AutoK run exactly reusable as the
Stage-A ReLU arm; it is a memory-policy choice, not an extra architecture.

The origin is meaningful, so the geometric output layer must not learn a
translation bias. ReLU inside a CNN backbone does **not** restrict \(z\) to the
positive orthant: the final signed linear projection can still produce every
direction. The prohibition is on applying ReLU—or any other coordinate-wise
nonlinearity—directly to the final \(d\)-dimensional coordinates.

There is no final coordinate-wise activation, BatchNorm, L2 normalization,
softmax, or classifier. Section 5.4 defines one explicitly isolated auxiliary
rotation-equivariant radial-gate candidate. Compute

\[
r_i=\sqrt{\lVert z_i\rVert_2^2+\varepsilon},
\qquad u_i=z_i/r_i
\]

only where a loss or diagnostic needs radius/direction. Never replace the raw
embedding by \(u\) for ShellMetric training, Euclidean kNN, or retrieval.

### 5.2 Learned ordered radii

For \(S>1\), fix \(\gamma_1=0\) and learn
\(\gamma_2,\ldots,\gamma_S\). Define

\[
q=\operatorname{softmax}(\gamma),\qquad
g_s=\frac{\kappa_g}{S}+(1-\kappa_g)q_s,
\qquad
t_s=\sum_{k=1}^{s}g_k,
\]

\[
\rho_s=\frac{t_s}
{\sqrt{\sum_{k=1}^{S}(n_k/C)t_k^2}}.
\]

Use \(\kappa_g=0.10\). The simplex gaps sum to one, removing the redundant
global gap scale; their fixed floor prevents learned shells from becoming
numerically identical. This guarantees positive, strictly ordered radii and
fixes class-weighted RMS radius to one, preventing global scale cheating
without a fixed radius list. Initialize every `gamma` to zero, giving equal
gaps (`t_s` linear in shell index before RMS normalization). Do not apply
weight decay to `gamma`.

For \(S=1\), set \(\rho_1=1\) and create no radius parameter. Do not combine
the RMS constraint with another fixed maximum-radius constraint.

### 5.3 Prototype-free supervised metric loss

Use a class-balanced \(P\times K\) batch sampler with \(P\ge2\) classes and
\(K\ge2\) samples per selected class. Use
`P=min(C,32)` and `K=max(4,floor(128/P))`, as specified in Section 8.4.
Average positive and negative terms separately so the larger number of
negative pairs does not change their relative weight.

For every unordered in-batch pair, let

\[
d_{ij}=\lVert z_i-z_j\rVert_2.
\]

For same-class pairs \(\mathcal P\), use a saturated positive margin:

\[
L_+=\frac1{|\mathcal P|}\sum_{(i,j)\in\mathcal P}
\left[\max(0,d_{ij}-m_+)\right]^2.
\]

For different-class pairs \(\mathcal N\), use

\[
m^-_{cd}=m_0+\Delta m\,\widehat W_{cd},
\]

\[
L_-=\frac1{|\mathcal N|}\sum_{(i,j)\in\mathcal N}
\left[\max(0,m^-_{y_i y_j}-d_{ij})\right]^2.
\]

The radial shell loss is

\[
L_{shell}=\frac1B\sum_i
\operatorname{SmoothL1}_{\beta}
\left(\log r_i-\log\rho_{a(y_i)}\right).
\]

The complete representation objective is

\[
L_{ours}=L_+ + \lambda_-L_-+\lambda_rL_{shell}.
\]

Numerical defaults, expressed in the RMS-radius-one scale, are:

```yaml
positive_margin: 0.50       # m+
negative_margin_base: 1.00  # m0
confusion_margin_delta: 0.50
negative_weight: 1.00       # lambda-
shell_weight: 1.00          # lambda-r
shell_huber_beta: 0.10
epsilon: 1.0e-8
```

These are starting defaults, not claims of universal optimality. Tune them
once on the canonical CIFAR-100 validation setting under a fixed budget, then
reuse the chosen values for every FixedS count, AutoK, and all three controls.
Do not tune a separate loss for each shell count.

The positive hinge makes a same-class cloud sufficiently compact but stops
pulling once it fits inside the margin. The shell loss has only a radial
gradient and leaves the class direction unconstrained. Confusion-weighted
negative pairs cause different classes—especially commonly confused ones—to
separate naturally. This combination should produce localized surface patches
instead of prototype-centered spokes.

Use a normal variance-preserving initialization for the Cartesian projection;
zero initialization is forbidden. Abort before training if an initial batch has
zero or non-finite embedding variance, because the Euclidean hinge and radial
norm can otherwise begin at a collapsed stationary point.

### 5.4 Bounded inner-activation and output-map study

The activation question applies to the **inner encoder layers as well as the
output map**, but these are two different questions. A coordinate-wise
activation is generally not rotation-equivariant: for an arbitrary orthogonal
matrix \(R\), ReLU, GELU, SiLU, and tanh usually satisfy
\(\phi(Rv)\ne R\phi(v)\). This does not mean that a hidden ReLU confines the
final embedding to one orthant—a later signed linear map can recombine its
channels—but it does introduce axes in that hidden channel basis. No
coordinate-wise activation is therefore to be described as “radially
symmetric.”

Run the following two auxiliary stages sequentially, only after the ordinary
ReLU + `linear_no_bias` AutoK loss hyperparameters are validation-tuned and
locked. Use those same locked loss values for every candidate; do not retune
per activation or output map. Do **not** form a Cartesian product.

#### Stage A: inner activation

Use CIFAR-100, a scratch CIFAR ResNet-18, AutoK, \(d=3\), seeds `[0,1,2]`,
the frozen official ReLU-derived pilot/AutoK plan, and a fixed
`linear_no_bias` output. Compare exactly:

1. `relu`;
2. `silu`;
3. `gelu_exact` (`approximate="none"`); and
4. `channel_radial_silu`, the following parameter-free channel-radial map.

Instantiate ReLU and SiLU with `inplace=False`; activation choice must not
change residual-buffer aliasing or autograd behavior.

For a feature tensor \(x\in\mathbb R^{B\times C\times H\times W}\), apply it
independently at each sample and spatial position:

\[
q_{bhw}=\sqrt{\frac1C\sum_c x_{bchw}^2+\varepsilon_a},
\qquad
g(q)=2\,\sigma(q-1),
\]

\[
A(x)_{bchw}=g(q_{bhw})x_{bchw},
\qquad \varepsilon_a=10^{-6}.
\]

It has zero trainable parameters, preserves signs and channel direction,
satisfies \(g(1)=1\), has a bounded multiplier, transforms radius
monotonically, and obeys \(A(Qx)=QA(x)\) for an orthogonal channel rotation
\(Q\). Compute the reduction, square root, and sigmoid in FP32 under mixed
precision, then cast the multiplier back to the input dtype.

Replace all 17 standard activation calls in ResNet-18: one after the stem
normalization, one after the first normalization in each of eight BasicBlocks,
and one after each of their eight residual additions. Do not place an
activation in a shortcut, after the embedding projection, or inside a loss.
Keep topology, normalization layers, initialized weight tensors, optimizer,
epochs, augmentations, batches, ShellMetric loss, and the bias-free output map
identical. All four activations are parameter-free; report throughput because
the radial channel reduction has a larger constant cost.

Use that one cached plan plus paired data-order manifests for all four
candidates, so the screen changes only the representation encoder's
activation. This estimates the downstream activation effect under one fixed
ReLU-derived shell problem; it does not compare four activation-specific
confusion matrices.
Let \(a^*\) have the highest mean validation raw-Euclidean 1-NN and retain every
candidate whose mean is within `0.2` percentage points of \(a^*\). If this set
has more than one member, choose the highest mean validation mAP@R, then the
fewest undefined per-class spoke fractions, then the lowest macro mean spoke
fraction over defined classes. For a final exact tie, prefer `relu` if it
remains in the tied set; otherwise choose the first tied name in the declared
candidate order. Test data must not participate. Lock this winner only as the
input to Stage B; do not regenerate confusion or propagate it into the primary
benchmark.

This activation alone does not make the complete CNN rotation-equivariant:
ordinary convolutions and per-channel BatchNorm still select a channel basis.
It also does not guarantee shells or angular separation; those remain effects
to be tested from the objective. High-dimensional norm concentration may make
the radial map nearly linear, and it may optimize worse than ReLU.

#### Stage B: final output map

With the Stage-A winner and the same official pilot/plan frozen, compare only:

1. `linear_no_bias`, \(z=Wh\); and
2. `radial_power_gate`, which first computes \(v=Wh\) with `bias=False`, then
   learns one global scalar \(\beta\):

   \[
   r=\lVert v\rVert_2,
   \qquad
   p_\beta=\exp\!\left((\log 2)\tanh\beta\right)\in(0.5,2),
   \qquad
   z=\exp\!\left((p_\beta-1)\log(r+\varepsilon_g)\right)v.
   \]

Use \(\varepsilon_g=10^{-6}\) and initialize \(\beta=0\), so the gate begins
as the identity. It adds one scalar, preserves direction and the origin, maps
radius monotonically, and satisfies \(A(Rv)=RA(v)\) for every orthogonal
\(R\). Under mixed precision, compute the norm, logarithm, exponent, and scalar
gate in FP32 before casting the result back to the projection dtype. Neither
candidate applies a coordinate-wise final activation.

Use the same CIFAR-100/ResNet-18/AutoK/3D cell, seeds, plan, and paired batch
construction as Stage A. Apply the same validation selection rule, replacing
the final Stage-A name tie-break by `linear_no_bias`. The Stage-B linear entry
aliases the winning Stage-A encoder; do not retrain it. Likewise, Stage-A's
ReLU entry aliases the primary AutoK seeds `[0,1,2]`. Thus the decision tables
contain four activation entries plus two output-map entries, but semantic
deduplication yields five unique architecture configurations and only four new
three-seed configurations beyond the primary run. This is not an axis crossed
with FixedS, controls, losses, dimensions, datasets, or backbones.

Train the `radial_power_gate` arm **from scratch** from the same seeded initial
backbone/projection state (or the same pinned pretrained starting checkpoint),
plan, batches, and data order as its linear counterpart. Never attach the gate
to or fine-tune the already trained Stage-A winner.

Report every candidate's validation result in a separate
architecture-ablation table. After both decisions are frozen, evaluate only the
final auxiliary configuration on sealed test data; do not test-select among the
six decision entries. Selection uses seeds `[0,1,2]`; after both decisions are
frozen, materialize seeds `[3,4]` only for the chosen auxiliary configuration
and report its five-seed test statistics; if it equals primary, reuse primary
seeds `[3,4]`. Nonselected candidates remain validation-only. Every primary
scratch-ResNet method remains ReLU +
`linear_no_bias` in the main tables. Pretrained CNNs and transformers retain
their native inner activations but still use the shared linear output where the
matched protocol applies. HyperSpaceX-Official retains its complete native
architecture. If the final auxiliary winner is semantically identical to the
primary ReLU + linear run, record it as an alias and state that no distinct
architecture variant won; do not create a duplicate
`ShellMetric-ActivationVariant` row. Otherwise use that separate name. The
auxiliary variant must not replace the flagship post hoc or be carried across
the full matrix. A strong result may motivate a separately preregistered
follow-up; it is not another version-1 method family or one of the three causal
controls.

## 6. Training and decoding are separate experiments

### 6.1 ShellMetric representation stage

Train only:

- the backbone with `linear_no_bias` output: primary scratch ResNets use
  `ReLU(inplace=False)`, pretrained CNNs/transformers retain their pinned native
  inner activation, and Section 5.4 alone may use its named auxiliary
  architecture; and
- the \(S-1\) trainable gap logits that determine \(S\) shared ordered radii
  when \(S>1\).

There is no decoder. Save the encoder and radii as the representation
checkpoint. Select its checkpoint by validation raw-Euclidean 1-NN accuracy;
use the same rule for **every method** in the common representation and common
probe tables. For the separate native-head table, CE/LS/ArcFace/CosFace and
HyperSpaceX-Matched may instead use the checkpoint maximizing their native
validation top-1 accuracy. Save both checkpoint selections from the same
training history, label them explicitly, and never compare a native-selected
number as though it used the common selection rule. HyperSpaceX-Official follows
its pinned upstream checkpoint/selection rule and remains in its separate
reference table.

Section 8 classification baselines may train their declared native
classification/margin heads, but those heads are excluded from the saved common
representation. Their common affine probe is always newly initialized and
trained only after the encoder is frozen, exactly as for ShellMetric.

### 6.2 Decoder-free metric evaluation

Before fitting any post-hoc probe, evaluate raw embeddings with:

- Euclidean 1-NN top-1 accuracy (primary representation endpoint);
- Euclidean kNN with \(k\in\{1,3,5,11\}\), selecting \(k\) on validation and
  breaking a validation-accuracy tie toward smaller \(k\);
- Recall@1, Recall@5, and mAP@R;
- macro-F1, balanced accuracy, and worst-class accuracy for kNN;
- cosine 1-NN only as a diagnostic of how much information is lost when radius
  is removed.

The gallery is the training embedding set. Validation/test examples are
queries. Generate all cached embeddings in model evaluation mode with
deterministic evaluation transforms. Use unweighted majority voting over
squared Euclidean distance. Order equal-distance gallery candidates by stable
gallery sample ID before taking the top \(k\). Break a vote tie by the smallest
summed neighbor distance among tied classes, then by class ID. Report every
predefined \(k\) on validation, report 1-NN on test unconditionally, and report
one additional test result for the \(k\) selected on validation. Do not use
k-means.
Nearest-centroid results may be diagnostic only, because a centroid recreates
a one-point-per-class assumption.

### 6.3 ShellMetric-native shell-then-cosine kNN

In addition to the common evaluations, report one inference rule available
**only to ShellMetric runs that contain learned radii for \(S>1\), or
\(\rho_1=1\) for \(S=1\), and a frozen class-to-shell plan**. It has no
trainable classifier, prototype, centroid, or class direction. Use the same
validation-Euclidean-1NN-selected representation
checkpoint as the common tables; select only \(k\) on validation and do not
reselect an encoder checkpoint for this native classifier.

For a query embedding \(z\), first choose the nearest learned shell in the same
log-radius geometry used by the shell loss:

\[
r(z)=\sqrt{\lVert z\rVert_2^2+\varepsilon},
\qquad
\hat s(z)=\arg\min_{s\in\{1,\ldots,S\}}
\left|\log r(z)-\log\rho_s\right|.
\]

Thus adjacent decision boundaries in the smoothed radius \(r(z)\) are the
geometric means \(\sqrt{\rho_s\rho_{s+1}}\); in raw norm they are
\(\sqrt{\max(\rho_s\rho_{s+1}-\varepsilon,0)}\). Break an exact
shell-distance tie toward the smaller shell index. Restrict the
training gallery to

\[
\mathcal G_{\hat s}=\{(z_g,y_g):a(y_g)=\hat s\},
\]

where \(a(y_g)\) is the frozen **class assignment**, not a shell inferred again
from the gallery sample's observed radius. Define a safe direction

\[
u_c(z)=\begin{cases}
z/\lVert z\rVert_2,&\lVert z\rVert_2\ge\varepsilon_c,\\
0,&\text{otherwise},
\end{cases}
\qquad \varepsilon_c=10^{-8},
\]

and rank only that gallery by cosine distance

\[
d_{cos}(z,z_g)=1-u_c(z)^Tu_c(z_g).
\]

Call this classifier `shell_then_cosine_knn`. Evaluate
\(k\in\{1,3,5,11\}\), always report 1-NN, and choose one additional \(k\) by
validation top-1 exactly as in Section 6.2. Whenever a selected-shell gallery
has fewer than \(k\) eligible samples, use and record
\(k_{eff}=\min(k,|\mathcal G_{\hat s}|)\). Order equal-distance neighbors by
stable gallery sample ID; break vote ties by the smallest summed cosine
distance and then class ID. Assert that every shell gallery is nonempty. Cache
normalized gallery directions partitioned by shell and perform the search
blockwise at high \(d\). Count and report queries/gallery vectors with norm
below \(\varepsilon_c\); their direction is the zero vector by definition,
and any nonzero count is a geometry warning rather than silently ignored.

For both mandatory \(k=1\) and the validation-selected \(k\), report three
decomposed quantities in addition to end-to-end classification:

- shell-selection accuracy, \(\Pr[\hat s(z)=a(y)]\);
- cosine-kNN class accuracy conditional on a correct predicted shell; and
- oracle-shell cosine-kNN accuracy using \(a(y)\) only as a diagnostic upper
  bound, never as the deployable prediction.

For the deployable result also report top-1, macro-F1, balanced accuracy, and
worst-class accuracy. Save the numerator and denominator of every conditional
metric; if no query has the conditioning event, report `not applicable` rather
than zero or NaN-derived arithmetic.

This native classifier is required for ShellMetric-FixedS (including \(S=1\)),
ShellMetric-AutoK, and ShellMetric-AutoK-ShuffledConfusion. With \(S=1\) it
reduces to global cosine kNN. It is `not applicable` to
`ShellMetric-AutoK-NoShellLoss`, because that control deliberately has no
radii, and to every non-ShellMetric baseline. A wrong shell prediction is
intentionally unrecoverable: the classifier tests whether radial location and
within-shell angular neighborhoods jointly carry class information.

### 6.4 Common trainable linear probe on the frozen encoder

Then freeze the encoder completely, including BatchNorm statistics, and train
from scratch the same post-hoc affine softmax probe (decoder) for every method:

\[
\ell=W_{probe}z+b,
\qquad W_{probe}\in\mathbb R^{C\times d}.
\]

This is why a three-dimensional embedding can classify 10, 1,000, or more
classes: three coordinates are expanded into \(C\) logits. No nonlinear
decoder is required.

Train only `W_probe` and `b` with CE on cached deterministic training
embeddings. The canonical probe uses AdamW, learning rate `1e-2`, batch size
`min(1024, N_train)`, at most 200 epochs, and early stopping with patience 20
on validation top-1. Search weight decay over
`[0, 1e-6, 1e-5, 1e-4, 1e-3]`; break validation ties toward stronger
regularization, then the earlier epoch. Derive one deterministic probe seed
from the encoder run hash and use the identical grid/budget for every method.
Verify that the encoder checksum and BatchNorm buffers are unchanged. Store
the probe checkpoint separately.

The same trained encoder supplies both the metric and classification results;
do not retrain it for the second table.

## 7. Proposed configurations and the only controls

In the canonical control cell, let \(F\) be the deduplicated FixedS count set
after `ensure_one_shell_control` has added \(S=1\) if necessary. The reporting
rows per seed are:

1. `ShellMetric-FixedS(S)` for every \(S\in F\);
2. `ShellMetric-AutoK`;
3. `ShellMetric-AutoK-ShuffledConfusion`;
4. `ShellMetric-AutoK-NoShellLoss`.

AutoK selects a count and then uses exactly the FixedS planning/training rule.
Therefore, if \(S^*\in F\), `AutoK` and `FixedS(S*)` must reference the same
encoder artifact while retaining both reporting/provenance labels. Deduplicate
by the complete resolved training semantics and hashes, not by display name.

Maintain two distinct plan hashes:

- `plan_provenance_hash` hashes the complete plan artifact, including display
  family, FixedS/AutoK selection route, bootstrap risks, parent hashes, and
  other reporting provenance;
- `plan_semantic_hash` hashes only the canonical byte representation of values
  that can change representation training: ordered class IDs, \(S\), capacities,
  class-to-shell assignment, the exact \(\widehat W\) used for pair margins,
  and radius parameterization/initialization settings.

The encoder-training job hash combines `plan_semantic_hash` with the resolved
split/data transforms, backbone and starting weights, activation/output map,
embedding dimension, loss, sampler and batch-order manifest, optimizer/schedule,
training budget, precision mode, and seed. It excludes display names,
selection-route metadata, and `plan_provenance_hash`. Use this job hash for
deduplication and resume. This is what permits AutoK/FixedS, Stage-A ReLU, and
Stage-B linear reporting aliases without erasing their different provenance.

There are \(|F|+3\) reporting rows, but only

\[
|F|+2+\mathbf 1[S^*\notin F]
\]

unique encoder-training jobs: the FixedS jobs, up to one otherwise-uncovered
AutoK job, shuffled confusion, and no-shell-loss. For 10 classes with
`F=[1,2,3,4]`, there are seven reporting rows but six unique training jobs.
In a non-control cell there are \(|F|+1\) reporting rows and
\(|F|+\mathbf 1[S^*\notin F]\) unique jobs.

### 7.1 One-shell control

Reuse `ShellMetric-FixedS(S=1)`; never launch a duplicate control under a
second name.

- set \(\rho_1=1\);
- retain real confusion-weighted positive/negative metric learning;
- retain the shell loss, architecture, optimizer, and budget.

This tests multiple radial levels against one shared hypersphere.

### 7.2 Shuffled-confusion control

First freeze the real AutoK shell count \(S^*\) and its capacities. For a
deterministic class permutation \(\pi\), construct

\[
W^\pi_{ij}=W_{\pi(i),\pi(j)}.
\]

Apply the same permutation to rows and columns, recompute difficulty ordering
and assignment from \(W^\pi\), and use \(W^\pi\) in negative margins. Do not
rerun AutoK and do not change \(S^*\), capacities, architecture, or budget.
Pair each ordinary training seed with one deterministic permutation; do not add
a separate shuffle-repeat factorial.

This preserves the numerical structure of confusion while attaching it to the
wrong class identities. Generate a seeded derangement, reject the identity,
recompute \(\widehat W^\pi\), and verify that \(W^\pi\ne W\) and that either
the assignment or pair margins change. Retry deterministically up to 100 times;
if no valid permutation exists because the graph is permutation-invariant,
mark the control `not applicable`. This control tests the combined use of
confusion in assignment and margins; do not claim it isolates assignment alone.

### 7.3 Shell-loss-removal control

Name this run `ShellMetric-AutoK-NoShellLoss` and record its parent AutoK
confusion and plan hashes. Never instantiate one no-shell run per FixedS count.

Remove the radii module and set \(\lambda_r=0\). Retain the real
confusion-weighted prototype-free metric loss and every other training choice.
The saved AutoK plan has provenance value but no training role in this control.

This tests explicit radial geometry against unconstrained confusion-aware
metric learning. It differs from one shell because sample norms are not
constrained.

If AutoK itself selects one shell, AutoK, FixedS(1), and the one-shell control
all reference the same encoder artifact. Mark the causal one-shell comparison
`not applicable` rather than retraining it.

Do not cross the controls with every shell count, dataset, dimension, or
backbone. Run them once in the canonical ablation cell described below.

## 8. Baselines and fair comparison

Use exactly these initial baselines at the same embedding dimension and with
the same backbone, inner activation, `linear_no_bias` output map, split,
augmentations, optimizer budget, and seeds. Scratch-ResNet matched cells use
`ReLU(inplace=False)`; pretrained CNN/transformer cells retain the pinned
backbone's native activation for every matched method. HyperSpaceX is split
into matched and official protocols below; never mix them in one comparison
row.

### 8.1 Classification/native-head baselines

- cross-entropy;
- CE with label smoothing (`0.1` unless validation changes it);
- ArcFace;
- CosFace;
- `HyperSpaceX-Matched`: DistArc adapted through a thin wrapper to the shared
  dataset, backbone, dimension, budget, and split; and
- `HyperSpaceX-Official`: one unchanged pinned upstream configuration, reported
  only in a separate reference/reproduction table.

### 8.2 Metric-learning baselines

- supervised pairwise contrastive loss;
- supervised contrastive learning (SupCon);
- batch-hard triplet loss;
- Multi-Similarity loss.

Do not initially add Center Loss, OPL, normalized softmax, CE+shell,
triplet+shell, semantic assignments, surface uniformity, or other combinations.
They would obscure the main causal study.

For every in-repository or matched baseline, define the evaluated representation
as the output of the shared \(d\)-dimensional projection/output map **before** any
method-specific L2 normalization, margin head, proxy operation, or classifier.
This same raw tensor is used for common Euclidean evaluation. ArcFace,
CosFace, SupCon, and Multi-Similarity may normalize a copy only inside their
training loss. For HyperSpaceX, save its raw \(d\)-dimensional embedding before
the scaled-proxy decoder.

Classification methods also report their native head. Contrastive, SupCon,
triplet, and Multi-Similarity have no separate native classifier; they report
decoder-free raw-Euclidean kNN/retrieval and the common trainable linear probe
on their frozen encoder.
ShellMetric alone additionally reports `shell_then_cosine_knn` from Section
6.3; no baseline receives this classifier because no baseline owns ShellMetric's
frozen class-to-shell plan.
Both HyperSpaceX protocols report their native nearest-scaled-proxy decoder and
the common evaluation when the upstream artifact exposes compatible raw
embeddings. HyperSpaceX-Official common metrics remain supplemental inside its
fourth reference table and never enter controlled statistics. Their proxies
must never be imported into ShellMetric.

ShellMetric and all four metric baselines must use the same seeded
\(P\times K\) sampler and paired batch-index manifests. Native classification
baselines use one shared ordinary shuffled sampler. This prevents batch
composition from masquerading as a loss effect.

### 8.3 Canonical baseline definitions

Use these starting definitions; any tuning consumes the common validation
budget described below.

| Method | Locked initial definition |
|---|---|
| CE | affine `Linear(d,C)` head and ordinary cross-entropy |
| CE + label smoothing | same head, smoothing `0.1` |
| ArcFace | normalized copy of `z`, scale `30`, angular margin `0.50` |
| CosFace | normalized copy of `z`, scale `30`, cosine margin `0.35` |
| Pair contrastive | same-class `d^2`; different-class `[1-d]_+^2` on raw Euclidean `z` |
| SupCon | two augmented views, normalized copy of `z`, temperature `0.07` |
| Batch-hard triplet | hardest positive/negative in the common PK batch, raw Euclidean `z`, margin `0.20` |
| Multi-Similarity | normalized copy of `z`; `alpha=2`, `beta=50`, `base=0.5`, miner epsilon `0.1` |
| HyperSpaceX-Matched | pinned DistArc implementation, adapted to the shared split/backbone/dimension/budget; record every deviation and never label it an official reproduction |
| HyperSpaceX-Official | one pinned upstream command/checkpoint protocol reproduced without shared-architecture edits; report in a separate reference table |

Only `HyperSpaceX-Matched` enters paired matched-method tables and statistics.
`HyperSpaceX-Official` establishes that the wrapper can reproduce an upstream
setting and provides context, but its native architecture/data protocol makes
it ineligible for claims based on a controlled loss comparison.

Give every method except the unchanged `HyperSpaceX-Official` reproduction at
most 12 deterministic validation trials, including its default. The search
space and seeds must be written before test access. Tune ShellMetric once on
CIFAR-100/ResNet-18/3D AutoK with the primary ReLU + linear architecture and
reuse it for FixedS, controls, and the Section 5.4 activation study. The same
principle applies to each baseline: do not retune for every dimension or
dataset. Store the complete trial table, including failed trials.

### 8.4 Shared optimizer and preprocessing defaults

These defaults make the first implementation executable; a changed value is a
recorded experimental protocol change, not an implicit implementation choice.

| Cell | Input and augmentation | Initialization | Optimizer/schedule |
|---|---|---|---|
| MNIST/Fashion smoke | native 28×28, tensor normalization; no horizontal flip | scratch small CNN | AdamW, lr `1e-3`, wd `1e-4`, 30 epochs, cosine |
| CIFAR-10/100 | random 32×32 crop with padding 4, horizontal flip, dataset normalization; deterministic normalized eval | scratch ResNet with 3×3 stride-1 stem and no max-pool | SGD, lr `0.1`, momentum `0.9`, wd `5e-4`, 5-epoch warmup, 200 epochs, cosine |
| TinyImageNet | random 64×64 crop with padding 8, horizontal flip, ImageNet normalization; deterministic normalized eval | scratch ResNet with 3×3 stride-1 stem and no max-pool | same SGD protocol as CIFAR, 200 epochs |
| CUB-200 | random-resized crop 224, horizontal flip, ImageNet normalization; resize 256 + center crop 224 for eval | ImageNet-1K pretrained ResNet | SGD, lr `0.01`, momentum `0.9`, wd `1e-4`, 5-epoch warmup, 100 epochs, cosine |
| ViT-S/16 CIFAR-100 check | train: `RandomResizedCrop(224, scale=(0.08,1.0), bicubic)` + horizontal flip; eval: bicubic resize 256 + center crop 224; normalize with the pinned checkpoint's recorded mean/std | pinned ImageNet-1K pretrained weights | AdamW, lr `5e-5`, wd `0.05`, 5-epoch warmup, 100 epochs, cosine, grad clip `1.0` |

Use target batch size 128 for CNN cells. For metric methods choose
`P=min(C,32)` and `K=max(4,floor(128/P))`; draw with replacement when needed.
Define an epoch as `ceil(N_train/128)` optimizer steps for every method, so a
slightly different `P*K` does not change the update budget. Classification
baselines use batch size 128. Scale learning rate linearly only when hardware
forces a different effective batch size, and record that fact. Apply no CNN
gradient clipping. Mixed precision may be enabled only if it is enabled for
every method in the cell. Save a checkpoint every epoch.

### 8.5 Encoder inventory and parameter scaling

“Encoder size” below means trainable parameter count, excluding buffers,
optimizer state, the final embedding map, and any classifier. Assert these
counts in code after the exact library versions are pinned; a mismatch is an
architecture change, not harmless metadata.

| Encoder | Role/input variant | Pooled width \(H\) | Backbone parameters | Native inner activation |
|---|---|---:|---:|---|
| local small CNN | MNIST/Fashion smoke only | 128 | 109,184 | native ReLU; not part of Stage A |
| ResNet-18, 3×3 stem/no max-pool | primary CIFAR/TinyImageNet scratch encoder | 512 | 11,168,832 | primary ReLU; Stage A is an auxiliary replacement study |
| ResNet-18, standard ImageNet stem | pretrained CUB encoder | 512 | 11,176,512 | native ReLU; do not replace after pretraining |
| ResNet-50, 3×3 stem/no max-pool | scratch backbone confirmation | 2,048 | 23,500,352 | native ReLU; not part of Stage A |
| ResNet-50, standard ImageNet stem | pretrained variant if later used | 2,048 | 23,508,032 | native ReLU |
| ViT-S/16 at 224 pixels | transformer confirmation | 384 | 21,665,664 expected | native GELU |

The ViT number assumes a non-distilled `vit_small_patch16_224` with 12 blocks,
width 384, MLP ratio 4, and no classifier. Pin the exact implementation/model
ID and pretrained checkpoint before coding the experiment; until then, treat
the count as an expected value rather than a verified run artifact.

The existing local `MLPBackbone(784→256→128)` has \(H=128\), 233,856
parameters, and ReLU, but it is legacy/debug code and is not a publication
encoder. HyperSpaceX's paper also mentions the following reference-only
encoders; they are not ShellMetric backbones:

| HyperSpaceX reference encoder | Typical visual parameters | Typical native width/output | Status |
|---|---:|---|---|
| iResNet-50 | ≈43.6M | commonly a 512-D face embedding | variant-dependent |
| CLIP RN101 | ≈56M | usually 512-D projected output | checkpoint-dependent |
| CLIP ViT-B | ≈86–88M | width 768, usually 512-D projected output | patch/checkpoint not yet pinned |
| CLIP ViT-L | ≈303–304M | width 1,024, usually 768-D projected output | patch/checkpoint not yet pinned |

These values are orientation only. The exact patch size, projection width,
checkpoint, and upstream commit must be read from the pinned official
implementation; do not use approximate values in result tables.

For the bias-free linear output map, the additional parameter count is exactly
\(Hd\). The radial power gate adds one more scalar, and learned radii add
\(S-1\) scalars. The output-map cost at the low and highest required dimensions
is:

| Backbone width \(H\) | Linear map at \(d=3\) | Linear map at \(d=1024\) |
|---:|---:|---:|
| 128 (small CNN/legacy MLP) | 384 | 131,072 |
| 384 (ViT-S/16) | 1,152 | 393,216 |
| 512 (ResNet-18) | 1,536 | 524,288 |
| 2,048 (ResNet-50) | 6,144 | 2,097,152 |

### 8.6 Decoder and metric-evaluation complexity

The common post-hoc trainable classifier is the affine probe from Section 6.4,
not part of representation training. For \(C\) classes and embedding dimension
\(d\), it has

\[
N_{probe}=Cd+C=C(d+1)
\]

parameters, requires approximately \(Cd\) multiply-accumulates per sample,
and stores \(4C(d+1)\) bytes in FP32. This count excludes \(C\) bias additions
and softmax/argmax's \(O(C)\) work; one MAC is approximately two FLOPs.

| Classes \(C\) | \(d\) | Probe parameters | Approx. FP32 weights | MACs/sample |
|---:|---:|---:|---:|---:|
| 10 | 3 | 40 | 160 B | 30 |
| 10 | 1,024 | 10,250 | 40.0 KiB | 10,240 |
| 100 | 3 | 400 | 1.56 KiB | 300 |
| 100 | 1,024 | 102,500 | 400 KiB | 102,400 |
| 1,000 | 3 | 4,000 | 15.6 KiB | 3,000 |
| 1,000 | 1,024 | 1,025,000 | 3.91 MiB | 1,024,000 |

An ArcFace/CosFace class-weight matrix has \(Cd\) parameters without bias and
the same \(O(Cd)\) leading compute. HyperSpaceX's proxy decoder likewise stores
and compares \(O(Cd)\) class values, subject to its pinned implementation.

Because embeddings are cached and the encoder is frozen, one probe-training
epoch costs \(O(N_{train}Cd)\), and \(E\) epochs cost
\(O(EN_{train}Cd)\). With AdamW, parameters, gradients, and two moment tensors
use roughly four times the weight bytes in the table, excluding framework
overhead. The preliminary pilot is different: its classifier maps \(H\) to
\(C\), so it has \(C(H+1)\) parameters rather than \(C(d+1)\).

The ShellMetric-native classifier adds no trainable parameters. Pre-index the
gallery by frozen shell assignment. A query costs \(O(S)\) for shell selection
and \(O(N_{\hat s}d)\) for cosine search within the selected shell, where
\(N_{\hat s}\) is that shell's gallery size; worst case is the global
\(O(N_gd)\) search. It stores the same \(O(N_gd)\) embeddings plus \(O(C+N_g)\)
integer shell/index metadata.

Exact brute-force kNN has no trainable decoder parameters, but stores
\(O(N_gd)\) gallery values and costs \(O(N_gd)\) per query. At \(d=1024\), a
45,000-example CIFAR-100 training gallery occupies about 175.8 MiB in FP32;
10,000 test queries require 450 million query-gallery pairs, or 460.8 billion
coordinate distance terms. Therefore implement exact kNN with query/gallery
blocking and top-k merging; never materialize the full distance tensor. If an
approximate index is later used, report it separately and never substitute it
for the required exact result.

Report three separate tables:

1. **Representation quality:** common raw Euclidean kNN/retrieval metrics for
   all methods.
2. **Common classification:** an identical newly trained affine probe for all
   methods, with each encoder frozen.
3. **Method-specific native classification:** ShellMetric's
   `shell_then_cosine_knn`, native CE/LS/ArcFace/CosFace heads, and the
   HyperSpaceX-Matched native decoder. These mechanisms are deliberately
   different; do not mix this table with common-probe or controlled-decoder
   claims.

Place `HyperSpaceX-Official` in a fourth, clearly labeled reproduction/reference
table. Only `HyperSpaceX-Matched` may appear in the three controlled comparison
tables above.

The local `HyperSpaceX.pdf` is a protocol reference. Use one pinned official
repository commit for both HyperSpaceX protocols, reproduce the unchanged
reference setting before trusting the matched adapter, and do not copy its
training geometry into the proposed method.

## 9. Experiment program without a combinatorial explosion

### 9.1 Stages

| Stage | Setting | Required runs |
|---|---|---|
| Smoke | MNIST and Fashion-MNIST, small CNN, \(d=3\), one seed | Pipeline correctness only; not paper evidence |
| Primary AutoK tuning | CIFAR-100, scratch ResNet-18, ReLU + linear, \(d=3\), seeds `[0,1,2]` | Build/freeze the official pilot and AutoK plan; tune ShellMetric within the declared budget and lock the loss config |
| Inner-activation ablation | CIFAR-100, scratch ResNet-18, AutoK, \(d=3\), seeds `[0,1,2]`; primary loss values and ReLU-derived plan already locked | ReLU, SiLU, exact GELU, and channel-radial SiLU with a fixed bias-free linear output; select one only for the next auxiliary stage |
| Output-map ablation | same cell with the Stage-A activation selected, seeds `[0,1,2]` | Bias-free linear versus radial power gate; after selection add seeds `[3,4]` only for the auxiliary winner |
| Canonical shell study | CIFAR-10 and CIFAR-100, ResNet-18, \(d=3\); reuse seeds `[0,1,2]`, then add `[3,4]` | FixedS compact sweep plus AutoK; report the entire shell-count curve |
| Essential controls | CIFAR-100, ResNet-18, \(d=3\), five seeds | Exactly one-shell, shuffled-confusion, and no-shell-loss controls |
| Core benchmark | CIFAR-10/100, ResNet-18, \(d\in\{3,32,128\}\), five seeds | All in-repository baselines plus AutoK; HyperSpaceX-Matched joins only after its external pin gate; reuse canonical runs |
| Dataset generalization | TinyImageNet and CUB-200, ResNet-18, \(d=128\), five seeds | All in-repository baselines plus AutoK; add HyperSpaceX-Matched after its gate |
| Dimension curve | CIFAR-100, ResNet-18, five seeds | Required now: AutoK, CE, ArcFace, SupCon, and batch-hard triplet at \(d\in\{2,3,8,16,32,128,512,1024\}\); add HyperSpaceX-Matched after its gate; run only new dimensions `2,8,16,512,1024` and reuse core artifacts at `3,32,128` |
| Backbone check | CIFAR-100, \(d=128\), scratch CIFAR-stem ResNet-50 and pretrained ViT-S/16, five seeds | ResNet-50: CE, ArcFace, SupCon, triplet, AutoK; ViT repeats these only after provider, model ID, dependency version, weight checksum, and preprocessing are pinned; HyperSpaceX-Matched joins after its upstream pin gate |
| HyperSpaceX official reproduction | one upstream paper/repository setting | Gated until the official commit/environment/command are pinned; then run unchanged and report separately from paired statistics |
| Scale-up | ImageNet-1K | Deferred until every preceding gate passes |

Use ResNet-18 as the primary backbone. A stronger CNN and one transformer are
confirmation studies, not a full reproduction of HyperSpaceX's backbone
matrix.

An unresolved external pin must produce `blocked_external_pin` in the dry-run
manifest; the runner must never choose a “latest” checkpoint or commit. The ViT
and HyperSpaceX-Official gates do not block core ShellMetric implementation,
ResNet experiments, or unit/integration tests. They block only their named
publication cells until an immutable pin manifest is approved.

Before HyperSpaceX integration, the dimension curve contains five methods ×
eight dimensions × five seeds = 200 reporting cells. Reusing the 75 cells at
\(d\in\{3,32,128\}\) leaves 125 new jobs. HyperSpaceX-Matched later adds 40
cells; if its 15 core-dimension cells already exist, only 25 are new, producing
the final six-method total of 240 cells and 150 new jobs. The dry run must print
the applicable counts before launch.

The FixedS family is fully reported only in the shell-count study. Carry AutoK
to the broad benchmark. Select no “best FixedS” using test results.

Controls are enabled only in the CIFAR-100/ResNet-18/3D canonical ablation
suite. AutoK-only benchmark cells use `fixed_shell_counts: none` and disable
all controls. Reuse every overlapping run across stages by semantic hash.
The two architecture-ablation stages follow the primary AutoK tuning and are
not crossed with any FixedS count, control, baseline, dimension, dataset, or
backbone.

### 9.2 Metric-learning claim boundary

The classification datasets above establish supervised closed-set metric
structure. If the paper claims general metric learning for unseen classes, add
a class-disjoint retrieval protocol such as SOP or a standard class-disjoint
CUB/Cars split. Do not make that claim from closed-set kNN alone.

### 9.3 Statistics

- use paired seeds and identical split manifests;
- for primary methods/baselines, use seeds `[0,1,2]` for screening, then add
  `[3,4]` for final tables rather than launching a separate five-seed study;
- for Section 5.4, select using `[0,1,2]` and add `[3,4]` only to the frozen
  auxiliary winner; nonselected candidates remain validation-only;
- report mean, standard deviation, and paired 95% confidence intervals;
- report every fixed shell count, not only the winner;
- use validation only for predeclared hyperparameters and checkpoints;
- lock the study plan before the final test evaluation.

## 10. Configuration and suite runner contract

A resolved study configuration should contain at least:

```yaml
method: shellmetric

data:
  dataset: cifar100
  split_manifest: artifacts/splits/cifar100.json

model:
  backbone: resnet18
  backbone_activation: relu
  embedding_dim: 3
  embedding_head: linear_no_bias
  output_bias: false

pilot:
  loss: cross_entropy
  folds: 3
  seeds: [0, 1, 2]
  temperature_calibration: false

shells:
  smax_rule: ceil_sqrt
  capacity_rule: index_squared
  assignment_rule: difficulty_sorted
  radii: learned_simplex_gaps_ordered_rms
  radius_gap_floor_fraction: 0.10
  auto_k:
    bootstrap_repeats: 200
    rule: paired_predictive_risk_one_se

loss:
  name: shellmetric_pair_margin
  positive_margin: 0.50
  negative_margin_base: 1.00
  confusion_margin_delta: 0.50
  negative_weight: 1.00
  shell_weight: 1.00
  shell_huber_beta: 0.10

sampling:
  classes_per_batch: 32
  samples_per_class: 4

study:
  fixed_shell_counts: compact
  ensure_one_shell_control: true
  run_auto_k: true
  controls: [shuffled_confusion, no_shell_loss]
  seeds: [0, 1, 2, 3, 4]

evaluation:
  euclidean_knn_k: [1, 3, 5, 11]
  retrieval: [recall_at_1, recall_at_5, map_at_r]
  shellmetric_native_classifier: shell_then_cosine_knn
  shellmetric_native_k: [1, 3, 5, 11]
  train_linear_probe_on_frozen_encoder: true
```

This example is the canonical control suite, so `ensure_one_shell_control` adds
`FixedS(1)` if needed. AutoK-only benchmark configs set
`fixed_shell_counts: none`, `ensure_one_shell_control: false`, and
`controls: []`.

The Stage-A config sets `fixed_shell_counts: none`, disables controls, fixes
`embedding_head: linear_no_bias`, and expands `backbone_activation` over
`[relu, silu, gelu_exact, channel_radial_silu]`. It must reference the frozen
primary loss-config hash and both hashes of the official ReLU-derived plan.
Save an immutable, content-hashed decision artifact containing all validation
inputs, candidate metrics, the tie-breaking path, and the winner.

The Stage-B config fixes that exact activation and expands `embedding_head`
over `[linear_no_bias, radial_power_gate]`. Save a second content-hashed
decision artifact. Only the separately named auxiliary result config contains
the two literal winning names; primary configs remain `relu` and
`linear_no_bias`. Never use a dynamic `best` value that could be recomputed
after test access. The suite must reject an activation/head Cartesian-product
request in these stages.

HyperSpaceX configs require one explicit, mutually exclusive protocol:

- `method: hyperspacex_matched` records the shared backbone/dimension/split and
  an immutable manifest of every deviation from upstream; or
- `method: hyperspacex_official` pins upstream commit, environment, full
  command, checkpoint/selection rule, and native data protocol, and rejects
  shared backbone, dimension, or budget overrides.

The two protocols must never share a run ID, artifact namespace, or result row.

Implement one restartable command with a dry-run manifest, for example:

```text
python scripts/run_shellmetric_study.py \
  --config configs/shellmetric/cifar100_shellmetric.yaml \
  --fixed-shell-counts compact \
  --resume
```

The CLI must also accept `--fixed-shell-counts compact`,
`--fixed-shell-counts all`, and `--fixed-shell-counts none`. Before training,
print and save every reporting row and every deduplicated encoder job, its seed,
parent plan, aliases, and whether it is a control. Resume by semantic artifact
hash, not merely by directory name.

When `method: shellmetric`, schema validation must reject manual assignment,
fixed class directions, prototype/proxy losses, a representation-stage
classifier, invalid shell counts, and any test-derived planning option.
Baseline-specific schemas permit only the heads/proxies declared in Section 8
and must keep them outside the ShellMetric namespace.

## 11. Repository update plan

The current repository implements prototype-based CA-MSPC. Preserve it only as
a clearly named legacy baseline; do not gradually reinterpret its codebook as
ShellMetric.

### 11.1 Reuse after verification

- split manifests and leakage checks;
- dataset loaders and transforms;
- CE cross-fitting and OOF artifact machinery;
- soft-confusion utilities (but not pooled-OOF temperature fitting);
- Cartesian backbone/projection construction only after replacing the current
  affine output with `bias=False` and verifying no output activation;
- generic artifact hashing, progress, resume, and reproducibility utilities;
- existing conventional classifier baselines.

The current confusion artifact must be adjusted so shell difficulty uses raw
off-diagonal symmetric `W`, not its uniform-floor normalized degree.

### 11.2 Do not call from the new path

- `src/multishell/codebook/*` prototype directions or prototype optimization;
- prototype arrays or fixed codebook artifacts;
- `distance_cross_entropy`, `prototype_compactness`, or
  `distance_ce_compact`;
- `nearest_prototype`, cosine-prototype, or shell-first prototype decoders;
- legacy manual, random, reverse-difficulty, semantic, or fixed-radius configs.

They may remain solely to reproduce the old baseline.

### 11.3 Add clean ShellMetric modules

Recommended boundaries are:

```text
src/multishell/shellmetric/
  activations.py   # activation registry and parameter-free channel-radial map
  heads.py         # bias-free linear output and equivariant radial power gate
  plan.py          # Smax, capacities, difficulty ordering, FixedS plans
  autok.py         # paired bootstrap predictive-risk selection
  radii.py         # ordered RMS-normalized shared radii
  sampler.py       # deterministic P x K sampler
  loss.py          # positive, negative, and radial losses
  train.py         # decoder-free representation training
  probe.py         # train affine probe while keeping encoder frozen
  evaluate.py      # raw metric and geometry evaluation
  study.py         # expansion, deduplication, resume, manifests
```

Add new configs under `configs/shellmetric/` and a new
`scripts/run_shellmetric_study.py`. The new modules may import generic legacy
utilities but must not import `multishell.codebook` or prototype decoders.

## 12. Required artifacts

Every result must be traceable. Save:

- immutable split manifest and sample IDs;
- OOF logits/probabilities, folds, and pilot seeds;
- `Q`, raw `W`, normalized `W_hat`, difficulty, and confusion hash;
- for every plan: \(C,S_{max},S\), capacities, class assignment, sorted class
  IDs, planning rule, upstream hashes, `plan_provenance_hash`, and
  `plan_semantic_hash`;
- for AutoK: all bootstrap risks/SEs, chosen count, and bootstrap seed;
- for shuffled confusion: its exact permutation and parent AutoK plan hash;
- for the suite: reporting-row IDs, semantic training-job IDs, alias/dedup
  relationships, and control-parent hashes;
- for each Section 5.4 stage: candidate metrics, paired seed/batch hashes,
  fixed parent plan/loss hashes, deterministic tie-break trace, selected name,
  decision hash, parameter count, and throughput;
- for HyperSpaceX: protocol kind, upstream URL/commit, environment, full
  command, checkpoint/selection rule, and—only for Matched—the complete shared
  override/deviation manifest;
- fully resolved config, run ID, seed, software versions, dataset/weight
  checksums, and code revision;
- encoder checkpoint and learned radius history;
- inner-activation name, output-map name, parameter count, and radial-gate beta
  when applicable;
- cached train/validation/test embeddings and labels from the frozen encoder;
- ShellMetric-native predicted shell, candidate-gallery size, neighbors,
  selected \(k\), predictions, shell/conditional/oracle metrics, and applicable
  plan/radius hashes;
- common linear-probe checkpoint stored separately;
- per-epoch losses and all final metric/geometry results;
- wall time, trainable parameters, gallery memory, and inference latency.

Never serialize class prototypes for a ShellMetric run because none should
exist.

## 13. Required geometry diagnostics

In raw embedding space report:

- learned radii and adjacent radius gaps;
- class and sample occupancy per shell;
- mean/quantiles of `abs(log(r)-log(target_radius))`;
- nearest-shell adherence rate;
- within-class radial and tangential spread;
- same-class and different-class distance distributions;
- Euclidean-versus-cosine 1-NN difference;
- covariance spectrum and rotationally invariant dimension utilization; and
- AutoK stability and FixedS shell-count curve.

For the global centered embedding covariance with eigenvalues \(\lambda_j\),
report the participation ratio

\[
d_{eff}=\frac{(\sum_j\lambda_j)^2}{\sum_j\lambda_j^2+\varepsilon},
\qquad d_{eff}/d.
\]

Report the same quantity for pooled within-class residuals. This is mandatory
on the dimension curve so a nominal 512D/1024D model cannot be described as
using the high-dimensional space when its effective rank remains small.

For class \(c\), if \(\lVert\bar z_c\rVert_2\ge10^{-8}\), let
\(v_c=\bar z_c/\lVert\bar z_c\rVert_2\) and let
\(e_i=z_i-\bar z_c\). Define

\[
V_{rad,c}=\mathbb E[(e_i^Tv_c)^2],
\]

\[
V_{tan,c}=\mathbb E[\lVert e_i-(e_i^Tv_c)v_c\rVert_2^2],
\]

and report

\[
\operatorname{spoke\_fraction}_c=
\frac{V_{rad,c}}{V_{rad,c}+V_{tan,c}+\varepsilon}.
\]

The desired geometry has low radial residual/spoke fraction and separable,
localized angular patches. Do not interpret a single class covering the whole
sphere as success. If a class mean norm is below \(10^{-8}\), report its spoke
fraction as undefined and report the number of undefined classes; do not invent
a fallback direction.

## 14. Tests required before large runs

### 14.1 Unit tests

- `Smax(10)==4`, `Smax(100)==10`, and `Smax(1000)==32`;
- capacities are deterministic, positive, nondecreasing, and sum to \(C\);
- capacity priorities and ties are identical across float32/float64 platforms
  because selection uses integer cross-products;
- capacities for \((C,S)=(10,4)\) equal `[1,1,3,5]`;
- harder classes never receive a smaller shell index than easier classes;
- AutoK is deterministic, uses all integers through \(S_{max}\), and returns a
  valid count;
- flat difficulty returns one shell;
- shuffling uses a non-identity derangement, preserves symmetry, diagonal
  zeros, and the multiset of matrix values, and changes assignment or margins;
- simplex gaps sum to one, radii are positive and strictly increasing, every
  normalized adjacent gap is at least \(\kappa_g/S\), and class-weighted RMS
  radius equals one;
- the shell loss is invariant to angular rotation at fixed norm;
- its gradient has no tangential component within numerical tolerance;
- positive loss is zero inside its margin;
- a more-confused negative pair never receives a smaller margin;
- each sampled batch contains valid positive and negative pairs;
- the initialized Cartesian projection produces finite, nonzero embedding
  variance on a real batch;
- the complete loss is invariant to a common orthogonal rotation;
- proposed checkpoints contain no decoder, proxy, class direction, or class
  point parameter;
- no ShellMetric embedding output map has a translation bias or
  coordinate-wise final activation;
- all four Stage-A activations have zero trainable parameters and leave the
  ResNet-18 parameter count identical;
- ResNet-18 exposes exactly the 17 intended replaceable activation sites, with
  none in a shortcut or after the embedding projection;
- `channel_radial_silu` preserves shape/device/dtype, maps zero to zero, has
  finite forward/backward values near zero, passes float64 gradcheck, satisfies
  \(g(1)=1\), preserves direction, and has monotonically increasing output
  radius;
- in float64, `channel_radial_silu` satisfies \(A(Qx)=QA(x)\) for random
  orthogonal channel rotations \(Q\);
- the radial power gate is identity at \(\beta=0\), preserves the direction of
  every nonzero vector, and has finite outputs/gradients at zero, tiny, and
  large radii;
- the radial gate has exactly one scalar beyond the \(Hd\)-parameter base map,
  always gives \(p_\beta\in(0.5,2)\), and maps input radius monotonically;
- in float64, the radial gate satisfies \(A(Rv)=RA(v)\) for random orthogonal
  \(R\), while a negative-test final coordinate-wise ReLU does not;
- every pinned encoder exposes the width and parameter count in Section 8.5,
  and linear/radial output maps contain exactly \(Hd\)/\(Hd+1\) parameters;
- `shell_then_cosine_knn` chooses the nearest log-radius shell with the declared
  inner-shell tie-break, admits only gallery samples whose class assignment
  equals that shell, and contains no trainable parameter/class representative;
- its safe direction is exactly zero below \(\varepsilon_c\), finite at zero,
  and unit norm otherwise; its smoothed-radius boundary matches the declared
  raw-norm boundary;
- it uses \(k_{eff}=\min(k,|\mathcal G_{\hat s}|)\) and resolves equal
  validation accuracy toward smaller \(k\);
- its neighbor/vote ties follow sample ID, summed cosine distance, then class
  ID; on a fixed CPU-float64 fixture, blockwise and dense implementations give
  identical neighbor IDs/predictions and distances within `1e-12`;
- with \(S=1\), `shell_then_cosine_knn` equals global cosine kNN;
- effective-rank diagnostics return one for rank-one covariance, \(d\) for an
  isotropic \(d\)-dimensional fixture, and remain finite for zero covariance;

### 14.2 Integration and leakage tests

- synthetic end-to-end: OOF pilot → confusion → FixedS/AutoK plan → encoder →
  raw evaluation → ShellMetric-native shell/cosine evaluation → trainable
  linear probe on the frozen encoder;
- changing validation/test labels cannot alter the confusion or plan;
- explicit FixedS lists are honored exactly unless the canonical control flag
  requests `S=1`;
- the suite emits each requested reporting row once but trains semantically
  identical AutoK/FixedS plans only once, retaining both aliases;
- semantically identical AutoK/FixedS plans have equal `plan_semantic_hash` but
  different `plan_provenance_hash`; changing only display/selection metadata
  cannot change the encoder-job hash, while changing any training-relevant
  tensor/configuration must change it;
- AutoK candidates are independent of the FixedS sweep list;
- controls appear only in the canonical ablation suite;
- shuffled confusion retains the real AutoK count and capacities and is not a
  no-op;
- no-shell-loss creates no radius parameters and contributes exactly zero
  radial loss;
- encoder parameters and BatchNorm buffers are byte-identical before and after
  probe training;
- blockwise exact kNN produces the same neighbors, distances, votes, and tie
  breaks as a dense reference on a small fixture;
- the ShellMetric-native classifier is emitted only for runs with usable
  plan/radius artifacts, is `not applicable` for no-shell-loss and baselines,
  selects \(k\) using validation only, and never uses oracle shell labels for
  deployable predictions;
- Stage A varies only the inner activation, Stage B varies only the output map,
  and the suite rejects their Cartesian product;
- Stage-A ReLU aliases the primary AutoK artifacts, Stage-B linear aliases the
  selected Stage-A artifacts, and an auxiliary winner identical to primary
  creates no misleading duplicate result row;
- every Stage-A/B candidate references the same official ReLU-derived plan and
  locked primary loss config; primary scratch-ResNet configs remain ReLU +
  linear, while pretrained/transformer cells retain native inner activations;
- only the frozen Stage-B winner may receive seeds `[3,4]` or sealed-test
  evaluation; every nonselected architecture candidate remains validation-only;
- HyperSpaceX-Official and HyperSpaceX-Matched cannot alias, Official cannot
  enter paired tables/confidence intervals, and all proxy tensors remain inside
  their baseline artifact namespace;
- the dimension suite expands exactly
  `[2,3,8,16,32,128,512,1024]`, includes `1024`, and deduplicates the
  `3`, `32`, and `128` jobs already present in the core benchmark;
- resuming reproduces the same plan, radii state, and run hash;
- common tables and ShellMetric's native classifier use the
  validation-Euclidean-1NN-selected checkpoint; CE/LS/ArcFace/CosFace and
  HyperSpaceX-Matched native tables may use explicitly separate
  native-validation-selected checkpoints, while HyperSpaceX-Official follows
  its pinned rule;
- raw rather than L2-normalized embeddings reach Euclidean evaluation.

## 15. Implementation sequence and acceptance gates

1. **Planning:** implement confusion artifacts, capacities, FixedS expansion,
   AutoK, and their tests without touching training.
2. **Geometry:** implement learned radii, shell loss, pair loss, and PK sampler;
   pass rotation/gradient/invariant tests.
3. **Training:** build the new decoder-free trainer and synthetic integration
   run. Confirm no class-point parameter exists.
4. **Evaluation:** add raw kNN/retrieval, geometry metrics, ShellMetric-native
   shell-then-cosine kNN, and the trainable linear probe; prove the frozen
   encoder cannot mutate during probing.
5. **Suite:** implement manifest expansion, deduplication, caching, dry-run, and
   resume. Confirm one command executes a multi-count FixedS sweep.
6. **Smoke:** MNIST/Fashion-MNIST only for failures and visualization.
7. **Canonical primary study:** build the official ReLU-derived pilot/plan, run
   CIFAR-100/ResNet-18/3D with ReLU + linear, tune within the declared budget,
   inspect geometry, and lock the loss hyperparameters.
8. **Inner-activation ablation:** run the four Stage-A candidates from Section
   5.4 against the frozen primary plan/loss, save the decision artifact, and
   select one only for Stage B.
9. **Output-map ablation:** run the two Stage-B candidates, save the decision
   artifact, and report the auxiliary winner without a Stage-A × Stage-B grid.
10. **External pin gates:** record the exact ViT provider/model/dependency/weight
    hash and HyperSpaceX commit/environment/command before their cells. Reproduce
    one official HyperSpaceX setting before any HyperSpaceX-Matched job; until
    then mark only those cells `blocked_external_pin`.
11. **Publication matrix:** execute all ungated rows in Section 9, including the
    required dimension curve through \(d=1024\), then execute gated cells only
    after their immutable pins exist.

The implementation is complete only when:

- one command can run an arbitrary FixedS list, AutoK, and the controls;
- there is no manual assignment API;
- the representation checkpoint has no decoder or class representative;
- every primary scratch-ResNet matched method uses `ReLU(inplace=False)` and
  `linear_no_bias`; pretrained/transformer matched cells retain native inner
  activations; the auxiliary decision is validation-locked and separately
  named; no ShellMetric embedding output has bias or a coordinate-wise final
  activation;
- AutoK provenance proves that only training OOF confusion selected \(S\);
- hard classes are demonstrably outward and capacity grows outward;
- radii are learned, ordered, and scale-constrained;
- raw metric results are produced before any method-specific classifier or
  post-hoc-probe results;
- ShellMetric-native classification first predicts a shell from radius, then
  searches only that shell by cosine kNN, with no prototype or trainable head;
- the post-hoc affine probe is trained strictly after freezing the encoder;
- each control differs from the flagship in exactly its stated factor;
- exact dimension results include \(d=1024\) with blockwise evaluation;
- all outputs are linked to split, confusion, plan, config, and seed hashes.

## 16. Paper-facing claim, kept narrow

The intended contribution is not that radial embeddings, metric learning,
margins, or confusion matrices are individually new. The candidate novelty is
their specific combination:

> a prototype-free supervised metric space with automatically selected,
> confusion-ordered radial shells, learned shared radii, outward-increasing
> class capacity, and unconstrained natural angular class patches.

The parameter-free shell-then-cosine kNN rule is the method's native inference
mechanism: radius chooses a shell and angular neighbors choose a class within
it. It is not used to create clusters during training, and its result is kept
separate from the common Euclidean and affine-probe comparisons.

HyperSpaceX remains the closest explicit radial-angular comparator, but it uses
class proxies and a proxy-based native decoder. Publication novelty and wording
must still be confirmed by a final literature review after the method is
implemented and the essential controls succeed.
