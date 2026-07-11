#!/usr/bin/env python3
"""Apply selective reviewer-closure edits to TA-DVFG LaTeX sources."""
from __future__ import annotations
import difflib,re,sys
from pathlib import Path


def replace_once(text, old, new, label):
    c=text.count(old)
    if c!=1: raise RuntimeError(f"{label}: expected 1 match, found {c}")
    return text.replace(old,new,1)


def replace_regex_once(text, pattern, new, label):
    updated,c=re.subn(pattern,lambda _:new,text,count=1,flags=re.MULTILINE|re.DOTALL)
    if c!=1: raise RuntimeError(f"{label}: expected 1 match, found {c}")
    return updated


def revise_main(text):
    text=replace_once(text,"""Does learned sparsity improve over dense, fixed matched-sparse, and
no-collaboration references?""","""Does learned sparsity improve over dense, fixed matched-sparse, adaptive,
and centralized prediction-selection references?""","main Q1")
    text=replace_once(text,"""Global Top-$k$ is the strongest centralized prediction-selection reference.
Best Single is the no-collaboration reference, while Ring, Random, Expander,
Adaptive Pairwise, Adaptive Complementarity, and Full Mesh serve as P2P
comparisons.""","""Global Top-$k$ is the strongest centralized prediction-selection reference.
Ring, Random, Expander, Adaptive Pairwise, Adaptive Complementarity, and Full
Mesh serve as P2P comparisons. Best Single is retained as the no-collaboration
reference for MovieLens, where it is evaluated under the same chronological
split.""","main baseline scope")
    text=replace_once(text,"""We also
do not assume identical confidence calibration across parties; predictor
reliability is estimated from held-out validation performance rather than from
raw confidence alone.""","""We do not require identical raw-confidence scales for reliability ranking:
predictor reliability is estimated from held-out validation performance rather
than from raw confidence alone. This weighting is not a per-party calibration
procedure.""","main calibration wording")
    text=replace_once(text,"""uses $k=5$. We audited all 90 headline HGB result rows and verified that their
resolved configurations matched these defaults. The full audit is provided in
the supplementary material.""","""uses $k=5$. The primary protocol uses the held-out validation split for
reliability estimation, candidate-topology scoring, and validation-based epoch
selection; test labels remain reserved for final reporting. We audited all 90
headline HGB result rows and verified that their resolved configurations matched
these defaults. The full audit is provided in the supplementary material.""","main validation flow")
    text=replace_once(text,"""predictions, validation signals, and optional logit gradients may still leak
information. Third, HGB provides controlled graph-view constructions, while""","""predictions, validation signals, and optional logit gradients may still leak
information. Compatible output semantics also do not ensure identical
calibration across predictors, and the current reliability weighting does not
explicitly calibrate per-party distributions. Third, HGB provides controlled
graph-view constructions, while""","main calibration limitation")
    return text

ADAPTIVE=r'''\paragraph{Adaptive P2P baselines.}
Let $p_i(x)$ be party $i$'s validation probability vector and
$\widehat y_i(x)=\arg\max_c p_{ic}(x)$. For validation set
$\mathcal{D}_{\mathrm{val}}$, define the standalone accuracy $a_i$, pair-average
accuracy $a_{ij}$, reliability $\rho_{ij}$, and pair gain $g_{ij}$ as
\begin{align}
a_i &= \frac{1}{|\mathcal{D}_{\mathrm{val}}|}
\sum_x \mathbf{1}[\widehat y_i(x)=y(x)], \nonumber\\
a_{ij} &= \frac{1}{|\mathcal{D}_{\mathrm{val}}|}
\sum_x \mathbf{1}\!\left[
\arg\max_c \tfrac12\bigl(p_{ic}(x)+p_{jc}(x)\bigr)=y(x)
\right], \nonumber\\
\rho_{ij} &= \tfrac12(a_i+a_j),
\qquad
g_{ij}=a_{ij}-\max(a_i,a_j).
\end{align}
We further use the corrective rate $c_{ij}$ (exactly one endpoint is correct),
the jointly-wrong rate $w_{ij}$, and the mean Jensen--Shannon disagreement
$d_{ij}$:
\begin{align}
c_{ij} &= \frac{1}{|\mathcal{D}_{\mathrm{val}}|}
\sum_x \mathbf{1}[\widehat y_i(x)=y(x)\ \mathrm{xor}\
\widehat y_j(x)=y(x)], \nonumber\\
w_{ij} &= \frac{1}{|\mathcal{D}_{\mathrm{val}}|}
\sum_x \mathbf{1}[\widehat y_i(x)\ne y(x),\widehat y_j(x)\ne y(x)],
\nonumber\\
d_{ij} &= \frac{1}{|\mathcal{D}_{\mathrm{val}}|}
\sum_x \operatorname{JSD}\!\left(p_i(x),p_j(x)\right).
\end{align}
The production cached-path utilities are
\begin{align}
u^{\mathrm{pair}}_{ij}
&=0.45a_{ij}+0.25\rho_{ij}+0.05d_{ij}, \nonumber\\
u^{\mathrm{comp}}_{ij}
&=0.45a_{ij}+0.20g_{ij}+0.25\rho_{ij}
 +0.30c_{ij}+0.05d_{ij}-0.20w_{ij}.
\end{align}
Embedding-diversity terms are inactive because the cached HGB and MovieLens
paths do not store embeddings. Each unordered pair is scored once, candidates
are sorted by descending utility, and edges are admitted in that fixed order
subject to the configured edge budget and degree limit. Exact utility ties are
resolved by descending party indices, matching the implementation's reverse
sort over $(u_{ij},i,j)$. Neither baseline inserts a candidate, reruns
whole-graph consensus, and recomputes deployment utility; neither uses
\method{}'s graph-level gain-threshold stopping rule. After construction, both
use the same consensus and topology-constrained active readout as the other P2P
methods.'''


def revise_supp(text):
    text=replace_once(text,"""learned sparsity improves over dense, communication-matched, and
no-collaboration references;""","""learned sparsity improves over dense, communication-matched, adaptive,
and centralized prediction-selection references;""","supp Q1")
    text=replace_once(text,"""Isolated parties retain their current prediction. The reported configuration
uses one consensus step and self-weight $\eta=0.85$.""","""Isolated parties retain their current prediction. The reported configuration
uses one consensus step and self-weight $\eta=0.85$.

\paragraph{Output-calibration boundary.}
Compatible class semantics make prediction vectors dimensionally exchangeable,
but do not guarantee identical calibration across independently trained
predictors. The reliability scores above are global validation metrics used for
weighting; they are not a per-party calibration transform. Calibration-aware
consensus is outside the present method definition.""","supp calibration")
    text=replace_once(text,"""\subsection{Greedy Topology Selection}

The constrained argmax in the main paper defines a conceptual optimum""","""\paragraph{Validation-label flow.}
In the primary cached protocol, the held-out validation split supplies labels
for reliability estimation, candidate-topology scoring, and validation-based
epoch selection. Test labels are not used for any of these decisions and are
reserved for final reporting.

\subsection{Greedy Topology Selection}

The constrained argmax in the main paper defines a conceptual optimum""","supp validation flow")
    return replace_regex_once(text,r"\\paragraph\{Adaptive P2P baselines\.\}\nAdaptive Pairwise assigns each candidate edge.*?gain-threshold stopping rule\.",ADAPTIVE,"adaptive spec")


def diff(old,new,a,b):
    return ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=a,tofile=b))

root=Path(__file__).resolve().parent
mainp=root/'paper/source/main.tex'; suppp=root/'paper/source/supplementary.tex'
main=mainp.read_text(encoding='utf-8'); supp=suppp.read_text(encoding='utf-8')
main2=revise_main(main); supp2=revise_supp(supp)
mainout=root/'paper/source/main_reviewer_ready.tex'; suppout=root/'paper/source/supplementary_reviewer_ready.tex'
mainout.write_text(main2,encoding='utf-8',newline='\n'); suppout.write_text(supp2,encoding='utf-8',newline='\n')
out=root/'outputs/reviewer_closure_20260711'; out.mkdir(parents=True,exist_ok=True)
(out/'selective_paper_edits.diff').write_text(diff(main,main2,'main.tex','main_reviewer_ready.tex')+'\n'+diff(supp,supp2,'supplementary.tex','supplementary_reviewer_ready.tex'),encoding='utf-8')
print(mainout); print(suppout)
