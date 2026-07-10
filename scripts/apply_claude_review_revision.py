from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/source/main.tex"
SUPP = ROOT / "paper/source/supplementary.tex"
CHECK = ROOT / "paper/source/ReproducibilityChecklist.tex"
BIB = ROOT / "paper/source/references.bib"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, repl: str, label: str, flags: int = 0) -> str:
    out, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return out


main = MAIN.read_text(encoding="utf-8")

main = replace_once(
    main,
    r"""Across six controlled 15-party HGB-derived settings, \method{} outperforms Full
Mesh and matched sparse controls while selecting only 5--5.8 of 105 possible
links. On MovieLens, it improves over Best Single and reaches 0.7516 AUC with
3.20M transmitted scalars, versus 0.7497 AUC and 10.00M for Full Mesh. In
15-party MovieLens stress settings, it retains only 5.0--5.2 links while
outperforming 105-link Full Mesh. Mechanism audits show that direct subset
selection is a strong lower-cost choice for active-only readout, whereas peer
exchange materially improves party-local predictions. \method{} is an
inference-stage collaboration mechanism rather than a formal privacy mechanism
or a fully serverless, end-to-end VFGL protocol.""",
    r"""Across six controlled 15-party HGB-derived settings, \method{} outperforms Full
Mesh and matched sparse P2P controls with only 5--5.8 of 105 possible links.
The strongest centralized reference, Global Top-$k$, is numerically higher in
four settings and lower in two; with five seeds, paired tests do not resolve a
difference. This boundary is central: same-cardinality subset selection is
cheaper and slightly stronger for active-only readout, whereas sparse peer
exchange raises party-local accuracy by 1.85/2.67 points on ACM/DBLP Hard and
connected-party accuracy by 4.79/7.16 points. On MovieLens, \method{} reaches
0.7516 AUC with 3.20M transmitted scalars, versus 0.7497 and 10.00M for Full
Mesh. Thus, learned topology is not claimed to be necessary for one centralized
aggregate; its distinct role is party-local or joint deployment. \method{} is
an inference-stage collaboration mechanism rather than a formal privacy
mechanism or a fully serverless, end-to-end VFGL protocol.""",
    "main abstract boundary",
)

main = replace_once(
    main,
    r"""Moreover, deployment may require an active-party readout, party-local
predictions, or a trade-off between them. Direct subset selection can suffice
for active-only readout, whereas party-local improvement requires useful peer
communication. The problem is therefore to learn a sparse graph whose utility is judged by the resulting deployment performance.""",
    r"""Moreover, deployment may require an active-party readout, party-local
predictions, or a trade-off between them. Direct subset selection can suffice
for active-only readout, whereas party-local improvement requires useful peer
communication. A concrete deployment pattern is a consortium of hospitals that
maintain institution-specific risk models over aligned patients: a registry may
need one aggregate score, while each hospital's alerting system also needs an
improved local prediction without exporting hidden states or routing every
inference through a permanent center. Similar dual-output requirements can
arise in multi-bank fraud screening. These are motivating deployment patterns,
not empirical claims about our benchmarks. The problem is therefore to learn a
sparse graph whose utility is judged by the resulting deployment performance.""",
    "main concrete deployment pattern",
)

main = replace_once(
    main,
    r"""\paragraph{Decentralized and adaptive topology.}""",
    r"""\paragraph{Ensemble selection and expert combination.}
Classical stacking learns a meta-level combiner
\cite{wolpert1992stacked}, ensemble selection chooses a subset or weighted
library of predictors \cite{caruana2004ensemble}, and mixtures of experts learn
gating rules over specialized predictors \cite{jacobs1991moe}. \method{} is
methodologically related to this family and does not claim novelty for predictor
selection or combination itself. Its object is different: the selector returns
a sparse communication graph, each candidate edge is evaluated after rerunning
consensus on the current whole graph, and the objective may include party-local
post-consensus states rather than only one centralized ensemble output.

\paragraph{Decentralized and adaptive topology.}""",
    "main ensemble related work",
)

main = replace_once(
    main,
    r"""\caption{HGB Test@BestVal accuracy (\%, mean $\pm$ standard deviation over
five seeds). Bold and underlined values denote the best and second-best
results, respectively. Oracle Matched is a post-hoc test-mean upper envelope
over Ring, Random, and Expander matched sparse controls and is included only
as a stronger diagnostic reference.}""",
    r"""\caption{HGB Test@BestVal accuracy (\%, mean $\pm$ standard deviation over
five seeds). Global Top-$k$ is a centralized active-only selection reference:
it is numerically higher than \method{} in four settings and lower in two.
Bold and underlined values denote the best and second-best results. Oracle
Matched is a post-hoc test-mean upper envelope over matched sparse controls and
is diagnostic only. The table therefore does not support a universal
active-readout advantage for graph learning.}""",
    "main HGB caption",
)

main = replace_once(
    main,
    r"""\paragraph{Controlled HGB results.}
Table~\ref{tab:hgb-main} shows that \method{} outperforms Full Mesh and the post-hoc Oracle
Matched sparse envelope in all six settings while selecting only 5--5.8 of 105
possible links. Across the five shared seeds, paired two-sided Wilcoxon tests do not
detect a significant difference between \method{} and Global Top-$k$. With $n=5$, however, this should not be interpreted as evidence of
equivalence. We therefore treat Global Top-$k$ as the strongest centralized
prediction-selection reference: \method{} achieves comparable task-level
utility while producing an explicit sparse peer graph that supports
party-local post-consensus deployment. Complete 13-method results and paired
statistics are reported in the supplementary material.""",
    r"""\paragraph{Controlled HGB results and the active-only boundary.}
The first result to read from Table~\ref{tab:hgb-main} is the boundary, not a
universal win: Global Top-$k$ is numerically higher in four of six settings,
while \method{} is higher in two. Across five shared seeds, paired two-sided
Wilcoxon tests do not resolve a difference; with $n=5$, this is neither evidence
of equivalence nor high-powered evidence of superiority. We therefore do not
claim that learning a graph is necessary for a single centralized readout.
Relative to P2P alternatives, however, \method{} outperforms Full Mesh and the
post-hoc Oracle Matched sparse envelope in all six settings while selecting only
5--5.8 of 105 links. The mechanism question is thus whether the graph provides
value beyond participant selection when local parties also consume predictions.
Complete 13-method results and paired statistics are in the supplement.

IMDB is a selection-dominated regime in this controlled construction:
non-selective global and P2P methods cluster near 28--29\%, whereas both Global
Top-$k$ and \method{} rise sharply on IMDB Main. We therefore do not attribute
the IMDB gain to graph structure alone. The result mainly shows the value of
filtering harmful predictors; explaining why the useful/weak quality gap is
larger on IMDB requires dedicated per-party diagnostics beyond the present
suite.""",
    "main HGB interpretation",
)

# Replace the compact communication table with the stronger active-only boundary table.
main = sub_once(
    main,
    r"""\\begin\{table\}\[t\]\n\\centering\n\\scriptsize\n\\setlength\{\\tabcolsep\}\{3\.2pt\}\n\\caption\{ACM Main deployment communication.*?\\end\{table\}\n""",
    r"""\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\caption{Active-only boundary on Hard settings. Same-cardinality subset methods
use the participant count selected by \method{} but define no peer edges.
They slightly exceed its mean active accuracy while using roughly one-third of
the communication.}
\label{tab:active-only-boundary}
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{ACM Hard} & \multicolumn{2}{c}{DBLP Hard} \\
Method & Acc. & Comm. & Acc. & Comm. \\
\midrule
\method{} & 85.54 & 143,385 & 90.25 & 253,157 \\
Top-Reliability-$q$ & 86.03 & 49,005 & 90.57 & 94,122 \\
Greedy-Subset-$q$ & \textbf{86.59} & 50,820 & 90.42 & 97,368 \\
\bottomrule
\end{tabular}
\end{table}
""",
    "main replace communication table",
    flags=re.S,
)

main = replace_once(
    main,
    r"""\paragraph{Communication accounting.}
Table~\ref{tab:communication-main} separates peer exchange from deployment
readout traffic. \method{} uses
the same peer volume as the matched sparse reference while achieving
substantially higher accuracy. Relative to Full Mesh, it reduces peer exchange
by $21\times$ and total deployment communication from 2.04M to 0.15M
transmitted scalars, a 92.5\% reduction. Global Top-$k$ is cheaper because it
performs centralized collection rather than peer consensus; it therefore
represents a different deployment interface rather than a sparse-topology
baseline.""",
    r"""\paragraph{Active-only boundary and communication accounting.}
Table~\ref{tab:active-only-boundary} makes the central limitation visible in the
main paper: for one aggregate output, direct subset selection is the stronger
lightweight interface in these audited settings. The graph is justified only
when selected links themselves are useful to local consumers. For the P2P
comparison, \method{} still reduces ACM Main peer exchange by $21\times$ and
total deployment communication from 2.04M to 0.15M scalars relative to Full
Mesh; detailed peer/readout accounting remains in the supplement. Global
Top-$k$ is cheaper still because it performs centralized collection without peer
consensus.""",
    "main boundary communication paragraph",
)

main = replace_once(
    main,
    r"""\paragraph{Mechanism and deployment analysis.}
Figure~\ref{fig:mechanism-deployment} shows that graph-level selection reaches
85.54/90.25\% on ACM/DBLP Hard, versus 76.41/87.91\% for Adaptive Pairwise
and 76.74/87.32\% for Adaptive Complementarity. A same-cardinality audit,
however, shows that direct subset selection matches or slightly exceeds
active-only readout at lower communication. In contrast, disabling only peer
exchange changes active accuracy by at most 0.20 points but reduces all-party
local mean by 1.85/2.67 points and connected-party accuracy by 4.79/7.16
points. Thus, direct selection is strong for active-only deployment, whereas
sparse peer exchange is the distinctive benefit for party-local inference.""",
    r"""\paragraph{Mechanism and deployment analysis.}
Figure~\ref{fig:mechanism-deployment} shows that graph-level selection reaches
85.54/90.25\% on ACM/DBLP Hard, versus 76.41/87.91\% for Adaptive Pairwise
and 76.74/87.32\% for Adaptive Complementarity. Table~\ref{tab:active-only-boundary}
shows the complementary result: direct subset selection slightly exceeds
active-only readout at lower communication. In contrast, a same-state
intervention that disables only peer exchange changes active accuracy by at most
0.20 points but reduces all-party local mean by 1.85/2.67 points and
connected-party accuracy by 4.79/7.16 points. Hence the evidence separates two
problems: active-only deployment is mainly participant selection, whereas
party-local or joint deployment is where sparse peer exchange adds a distinct
capability.""",
    "main mechanism paragraph",
)

main = replace_once(
    main,
    r"""A selected edge should not be interpreted as a universal semantic similarity or
a permanent relationship between organizations. Degree-weighting ablations
change the selected graph substantially while leaving aggregate active
performance nearly unchanged, indicating multiple near-equivalent deployment
structures. The graph is best viewed as a low-frequency deployment artifact
under the current predictors, consensus operator, and objective.""",
    r"""The selected edge identities are not stable enough to support a semantic
``discovered relationship'' claim. Reselecting with $K=0$ yields mean edge-set
Jaccard similarity 0.410/0.164 on ACM/DBLP Hard, and removing degree weighting
reproduces the same topology in only 1/10 runs. Yet these interventions leave
active utility nearly unchanged. We interpret this as \emph{utility stability
under structural multiplicity}: several distinct sparse graphs are
near-equivalent for the current predictors and objective. The graph is therefore
a low-frequency deployment artifact, not a unique or permanent organizational
structure.""",
    "main topology multiplicity",
)

main = replace_once(
    main,
    r"""Sparse deployment communication scales with the selected edge set, but
exhaustive candidate scoring remains the current scalability bottleneck.
Candidate pruning or approximate search will therefore be necessary beyond the
tested scale.""",
    r"""Sparse deployment communication scales with the selected edge set, but
exhaustive candidate scoring remains the current scalability bottleneck. A
concrete next step is two-stage screening: use a cheap edge-local score to
shortlist the top-$M$ feasible candidates, then apply the existing whole-graph
rerun only to that shortlist. We do not report a pruning result here, so its
accuracy--time trade-off remains an open experiment rather than a claimed
contribution.""",
    "main scalability honesty",
)

main = replace_once(
    main,
    r"""In summary, \method{} reframes vertical graph-view collaboration as sparse
prediction-space deployment.""",
    r"""Code and experiment artifacts required for the reported analyses will be
released publicly upon publication; the review package contains the referenced
analysis scripts and raw outputs. Exact hardware and software-version metadata
are only partially preserved in the current artifact and remain a
reproducibility limitation.

In summary, \method{} reframes vertical graph-view collaboration as sparse
prediction-space deployment.""",
    "main reproducibility statement",
)

MAIN.write_text(main, encoding="utf-8")


supp = SUPP.read_text(encoding="utf-8")

supp = replace_once(
    supp,
    r"""The mechanism evidence supports an interface-dependent interpretation. Direct
subset selection is a strong lower-cost alternative for active-only
aggregation, while same-state interventions show that peer exchange materially
improves connected parties' local predictions. Learned topology is therefore
not claimed to be universally necessary for aggregation; its distinct role is
in party-local or joint deployment, where the communication graph itself
changes local inference.""",
    r"""The mechanism evidence supports an interface-dependent interpretation. Direct
subset selection is a strong lower-cost alternative for active-only
aggregation, while same-state interventions show that peer exchange materially
improves connected parties' local predictions. Learned topology is therefore
not claimed to be universally necessary for aggregation; its distinct role is
in party-local or joint deployment, where the communication graph itself
changes local inference. A useful mental model is a consortium in which one
coordinator needs an aggregate output while each member also operates a local
downstream system. For example, a hospital registry may need one risk score
while each hospital still needs an improved local alert; this is a motivating
deployment pattern, not a claim that HGB or MovieLens instantiates that domain.""",
    "supp deployment scenario",
)

supp = replace_once(
    supp,
    r"""The communication-matched controls are reported individually in
Table~\ref{tab:supp-complete}. The main-paper Oracle Matched row is explicitly
labeled as a post-hoc diagnostic: choosing the best matched family by test
performance is not a deployable baseline.""",
    r"""The communication-matched controls are reported individually in
Table~\ref{tab:supp-complete}. The main-paper Oracle Matched row is explicitly
labeled as a post-hoc diagnostic: choosing the best matched family by test
performance is not a deployable baseline.

\paragraph{Why IMDB behaves differently.}
The complete matrix shows a qualitatively different IMDB regime. Non-selective
methods remain near 28--29\%, while methods with explicit participant selection
(Global Top-$k$ and \method{}) rise sharply on IMDB Main. We therefore treat
IMDB primarily as evidence that filtering harmful predictors matters, not as
evidence that graph structure itself explains the gain. The present artifact
does not contain a matched per-party quality-distribution diagnostic sufficient
to explain why this separation is larger on IMDB; that remains a targeted
analysis gap.""",
    "supp IMDB interpretation",
)

supp = replace_once(
    supp,
    r"""Varying $K$ while rerunning topology selection is not a pure no-message
ablation, because $K$ changes both the communication operator and the topology
optimized on validation data. Indeed, independently reselected $K=0$ obtains
similar active accuracy to the default but selects a different graph in every
audited seed; mean edge-set Jaccard similarity is 0.410 on ACM and 0.164 on
DBLP.""",
    r"""Varying $K$ while rerunning topology selection is not a pure no-message
ablation, because $K$ changes both the communication operator and the topology
optimized on validation data. Indeed, independently reselected $K=0$ obtains
similar active accuracy to the default but selects a different graph in every
audited seed; mean edge-set Jaccard similarity is 0.410 on ACM and 0.164 on
DBLP. Together with the degree-weighting audit below (same topology in only
1/10 reselections), this rules out interpreting the selected edge identities as
a unique semantic structure. The supported interpretation is narrower:
distinct sparse graphs can be near-equivalent in deployment utility. We call
this structural multiplicity and evaluate utility separately from edge identity.""",
    "supp topology multiplicity",
)

supp = replace_once(
    supp,
    r"""Table~\ref{tab:supp-scalability} shows that selected graphs remain near five
links from 5 to 30 parties, while candidate evaluations and topology-selection
time grow much faster. Sparse deployment therefore scales differently from the
current selector: exhaustive candidate scoring, rather than deployment
communication, is the principal bottleneck.""",
    r"""Table~\ref{tab:supp-scalability} shows that selected graphs remain near five
links from 5 to 30 parties, while candidate evaluations and topology-selection
time grow much faster. Sparse deployment therefore scales differently from the
current selector: exhaustive candidate scoring, rather than deployment
communication, is the principal bottleneck. A concrete mitigation is a
two-stage selector that ranks feasible edges with a cheap edge-local score and
reruns whole-graph consensus only for the top-$M$ shortlist. This is an
algorithmic path, not an evaluated result in the present paper; a pruning
accuracy--time curve is therefore future work.""",
    "supp scalability path",
)

supp = replace_once(
    supp,
    r"""All primary HGB and MovieLens results use seeds 42--46. Reported
mean--dispersion summaries follow the same five-seed aggregation convention as
the main paper.""",
    r"""All primary HGB and MovieLens results use seeds 42--46. Reported
mean--dispersion summaries follow the same five-seed aggregation convention as
the main paper. We treat the same-state peer-exchange intervention as the
primary mechanism test. Budget, depth, degree, and selector-sanity sweeps are
exploratory robustness analyses; their non-significant results are not used as
confirmatory evidence of equivalence.""",
    "supp confirmatory exploratory distinction",
)

supp = replace_once(
    supp,
    r"""\paragraph{Code paths.}
MovieLens is implemented by""",
    r"""\paragraph{Code availability and environment boundary.}
The source code and experiment artifacts required for the reported analyses will
be released publicly upon publication. The review package contains the raw
CSV/JSON outputs and referenced analysis paths. Exact GPU/CPU, operating-system,
and package-version metadata are only partially preserved in the current
artifact; we do not reconstruct or invent missing environment details.

\paragraph{Code paths.}
MovieLens is implemented by""",
    "supp code availability",
)

SUPP.write_text(supp, encoding="utf-8")


check = CHECK.read_text(encoding="utf-8")
check = replace_once(
    check,
    r"""\question{All source code required for conducting and analyzing the experiments is included in a code appendix}{(yes/partial/no)}
	partial""",
    r"""\question{All source code required for conducting and analyzing the experiments is included in a code appendix}{(yes/partial/no)}
	yes""",
    "check code appendix",
)
check = replace_once(
    check,
    r"""\question{All source code required for conducting and analyzing the experiments will be made publicly available upon publication of the paper with a license that allows free usage for research purposes}{(yes/partial/no)}
	no""",
    r"""\question{All source code required for conducting and analyzing the experiments will be made publicly available upon publication of the paper with a license that allows free usage for research purposes}{(yes/partial/no)}
	yes""",
    "check public code",
)
CHECK.write_text(check, encoding="utf-8")


bib = BIB.read_text(encoding="utf-8")
extra = r"""

@article{wolpert1992stacked,
  title={Stacked Generalization},
  author={Wolpert, David H.},
  journal={Neural Networks},
  volume={5},
  number={2},
  pages={241--259},
  year={1992},
  doi={10.1016/S0893-6080(05)80023-1}
}

@inproceedings{caruana2004ensemble,
  title={Ensemble Selection from Libraries of Models},
  author={Caruana, Rich and Niculescu-Mizil, Alexandru and Crew, Geoff and Ksikes, Alex},
  booktitle={Proceedings of the Twenty-First International Conference on Machine Learning},
  pages={18},
  year={2004}
}

@article{jacobs1991moe,
  title={Adaptive Mixtures of Local Experts},
  author={Jacobs, Robert A. and Jordan, Michael I. and Nowlan, Steven J. and Hinton, Geoffrey E.},
  journal={Neural Computation},
  volume={3},
  number={1},
  pages={79--87},
  year={1991},
  doi={10.1162/neco.1991.3.1.79}
}
"""
if "wolpert1992stacked" not in bib:
    bib = bib.rstrip() + extra + "\n"
BIB.write_text(bib, encoding="utf-8")

print("Applied Claude-review revision to main, supplement, checklist, and bibliography.")
