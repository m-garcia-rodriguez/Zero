# %% [markdown]
# # Zero vs one: is what we see for 0 specific to zero, or shared with the other rule table?
#
# Every major inspection run on the 0-facts is repeated here with the **1-facts** as a separate
# group, as asked in the review of October 2026. Both tables are solved by a rule rather than by
# retrieval (n×0 = 0, n×1 = n), so a pattern that appears for 0 **and not** for 1 cannot be
# explained by rule use alone and is a candidate zero-specific effect (DECISIONS #42-#47).
#
# Groups used throughout (DECISIONS #42):
# * **0-facts**: at least one operand is 0. 0×1 and 1×0 are 0-facts (the zero rule fixes the answer, #36).
# * **1-facts**: no 0, at least one operand is 1.
# * **Retrieval**: both operands 2-9.
#
# Sections
# 1. RT vs error rate per fact (the first figure), with 1 as its own colour, ES_2025 and ES_2024
# 2. The same split by data-collection wave inside each age (reviewer question 4)
# 3. What the wrong answer is: the error taxonomy on the 1-facts next to the 0-facts
# 4. RT distribution by operand order: 1×n vs n×1 next to 0×n vs n×0
# 5. The order effect at trial level (fixed effects), 0 and 1 in one model
# 6. Sequential consistency by age and by transfer condition, ×0 next to ×1
# 7. Data checks behind the reviewer questions 1-3 (signs, Help, mis-recorded answers)
# 8. Sanity checks

# %%
import logging, re, sys, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path.cwd().parent if Path.cwd().name == "analysis" else Path.cwd()
sys.path.insert(0, str(ROOT))
from binary_stats import contrast_2x2, tetrachoric, tetrachoric_ci, wilson as wilson01
from cached_query import cached_query
from paired_stats import holm, signed_rank, stars
from plot_style import (C_DIR, C_DIR_ONE, C_ONE, C_OTHER, C_TYPES, C_ZERO, FONT_BIG, FONT_MED,
                        paint_it_black, set_style)

Q, PLOTS = ROOT / "queries", ROOT / "analysis" / "plots"; PLOTS.mkdir(exist_ok=True)
YEARS, YEAR, SEED, CAP = ["ES_2025", "ES_2024"], "ES_2025", 20261005, 14
GROUPS = ["0-facts", "1-facts", "Retrieval"]
GCOL = {"0-facts": C_ZERO, "1-facts": C_ONE, "Retrieval": C_OTHER}
rng = np.random.default_rng(SEED)
set_style()
for noisy in ("fontTools", "matplotlib.font_manager"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*singleton fixed effect.*")
warnings.filterwarnings("ignore", message="(?s).*multicollinearity.*")
pd.set_option("display.width", 200)
fmt = lambda p: "< .001" if p < .001 else f"= {p:.3f}"
compact = lambda n: f"{n / 1000:.0f}k" if n >= 1000 else f"{n:.0f}"


def _sub(name, year=YEAR, cap=None):
    s = (Q / name).read_text(encoding="utf-8")
    s = re.sub(r"'[A-Z_0-9-]+'(\s+-- \{YEAR\})", f"'{year}'\\1", s)
    return s if cap is None else re.sub(r"<= \d+(\s+-- \{CAP\})", f"<= {cap}\\1", s)


def cq(name, **kw):
    return cached_query(Q / name, cache_dir=ROOT / "cache", sql_text=_sub(name, **kw))


def group_of(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.select([(a == 0) | (b == 0), (a == 1) | (b == 1)], GROUPS[:2], default=GROUPS[2])


def save(fig, stem):
    fig.savefig(PLOTS / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## 1. RT vs error rate per directed fact, 1-facts in their own colour
# Same data and filters as `rt_error_scatter.ipynb` (DECISIONS #1-#13). Only the colouring changes.
# Note on age 8: the 1×n direction IS administered at age 8 (8 facts), n×1 and every 0-fact are not.
# DECISIONS #12 said "neither the 0-facts nor the 1-facts"; that is corrected in #42.

# %%
def load_facts(year):
    df = cq("rt_error_by_fact_age.sql", year=year)
    df["group"] = group_of(df.left_factor, df.right_factor)
    return df


def plot_facts(df, stem):
    ages = sorted(df.course_age.unique())
    ncols = 4; nrows = -(-len(ages) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, age in zip(axes.flat, ages):
        d = df[df.course_age.eq(age)]
        for g, s, ec in [("Retrieval", 55, "none"), ("1-facts", 70, "black"), ("0-facts", 70, "black")]:
            x = d[d.group.eq(g)]
            ax.scatter(x.median_rt_correct, x.error_rate, s=s, color=GCOL[g], edgecolor=ec, linewidth=.5,
                       label={"Retrieval": "Other facts", "1-facts": "Facts with 1", "0-facts": "Facts with 0"}[g],
                       zorder={"Retrieval": 1, "1-facts": 2, "0-facts": 3}[g])
        ax.set_title(f"Age {age}", fontsize=FONT_MED, color="black")
        if not d.group.eq("0-facts").any():
            ax.text(.97, .04, "0-facts and n×1 not administered", transform=ax.transAxes, ha="right",
                    fontsize=12, color="black")
        sns.despine(ax=ax)
    for ax in axes.flat[len(ages):]:
        ax.set_visible(False)
    h, l = axes.flat[1].get_legend_handles_labels()
    axes.flat[1].legend(h[::-1], l[::-1], frameon=False, fontsize=FONT_MED, loc="upper left")
    fig.supxlabel("Median response time (s)", fontsize=FONT_BIG, color="black")
    fig.supylabel("Error rate", fontsize=FONT_BIG, color="black")
    fig.tight_layout(); paint_it_black(axes); save(fig, stem)


def group_summary(df):
    '''Median over the facts of each group, per age, plus the fact-level 0 vs 1 Mann-Whitney test.'''
    rows = []
    for age, d in df.groupby("course_age"):
        r = {"course_age": age}
        for g in GROUPS:
            x = d[d.group.eq(g)]
            r[f"facts_{g[:3]}"] = len(x)
            r[f"err_{g[:3]}"] = 100 * x.error_rate.median()
            r[f"rt_{g[:3]}"] = x.median_rt_correct.median()
        z, o = d[d.group.eq("0-facts")], d[d.group.eq("1-facts")]
        if len(z) and len(o):
            r["p_err_0v1"] = mannwhitneyu(z.error_rate, o.error_rate).pvalue
            r["p_rt_0v1"] = mannwhitneyu(z.median_rt_correct, o.median_rt_correct).pvalue
        rows.append(r)
    out = pd.DataFrame(rows).set_index("course_age")
    for c in ("p_err_0v1", "p_rt_0v1"):
        ok = out[c].notna()
        out.loc[ok, c + "_holm"] = holm(out.loc[ok, c].to_numpy())
    return out


facts = {y: load_facts(y) for y in YEARS}
summ = {}
for y in YEARS:
    plot_facts(facts[y], f"rt_error_scatter_zero_one_by_age_{y}")
    summ[y] = group_summary(facts[y])
    print(y); display(summ[y].round(3))

# %% [markdown]
# Pooled over ages, response-weighted (so each group is summarised by its responses, not its facts):

# %%
def pooled(df):
    g = df.assign(err=df.n_errors).groupby("group")
    return pd.DataFrame({"facts_x_ages": g.size(), "responses": g.n_responses.sum(),
                         "error_rate_%": 100 * g.n_errors.sum() / g.n_responses.sum(),
                         "median_rt_of_fact_medians": g.median_rt_correct.median()}).round(2)

for y in YEARS:
    print(y, "(ages 9-15, where both rule tables are administered)")
    display(pooled(facts[y][facts[y].course_age.ge(9)]))

# %%
# A compact version of the scatter: group medians per age, 0 vs 1 vs retrieval, both outcomes
def group_lines(stem):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for col, y in enumerate(YEARS):
        s = summ[y]
        for row, (pre, lab) in enumerate([("err", "Median error rate of the facts (%)"),
                                          ("rt", "Median RT of the facts (s)")]):
            ax = axes[row, col]
            for g in GROUPS:
                v = s[f"{pre}_{g[:3]}"]
                ax.plot(v.index, v, "o-", color=GCOL[g] if g != "Retrieval" else "#8A8A8A", lw=2,
                        markersize=7, label={"0-facts": "Facts with 0", "1-facts": "Facts with 1",
                                             "Retrieval": "Other facts (2-9)"}[g])
            sig = s[f"p_{pre}_0v1_holm"]
            top = s[[f"{pre}_0-f", f"{pre}_1-f"]].max(axis=1)
            for age, p in sig.dropna().items():
                ax.text(age, top[age] * 1.03, stars(p), ha="center", fontsize=FONT_MED - 5, color="black")
            ax.set_ylabel(lab if col == 0 else "", fontsize=FONT_MED)
            if row == 0:
                ax.set_title(y, fontsize=FONT_MED, color="black")
            sns.despine(ax=ax)
    for ax in axes[1]:
        ax.set_xlabel("Age", fontsize=FONT_MED); ax.set_xticks(range(8, 16))
    axes[0, 0].legend(frameon=False, fontsize=FONT_MED - 3)
    fig.text(.5, -.01, "Stars (above the higher of the two rule lines): facts with 0 vs facts with 1, Mann-Whitney over facts, Holm over ages",
             ha="center", fontsize=FONT_MED - 4)
    fig.tight_layout(); paint_it_black(axes); save(fig, stem)

group_lines("group_medians_zero_one_by_age")


# %% [markdown]
# ## 2. Age split by data-collection wave (reviewer question 4)
# The A998 test is administered in waves (DECISIONS #47): ES_2025 has three, Nov-Dec 2025 (T1),
# Mar 2026 (T2) and May-Jun 2026 (T3); ES_2024 has two, Dec 2024 (T1) and May-Jun 2025 (T3), and ages
# 12-15 were tested only in T3 that year. Each age x wave cell is one point on a school-time axis.
# **The item set itself changes with the wave**: at age 9 the 0-facts and the n×1 direction appear only
# in T3 (72 facts in T1 and T2, 100 in T3), and at age 8 n×1 is absent everywhere. So "age 9" in the
# pooled figures is, for the 0-facts, really "end of age 9".

# %%
def load_wave(year):
    df = cq("rt_error_by_fact_age_wave.sql", year=year)
    df["group"] = group_of(df.left_factor, df.right_factor)
    return df

waves = {y: load_wave(y) for y in YEARS}

def wave_table(df):
    g = df.groupby(["course_age", "wave", "group"])
    out = pd.DataFrame({"facts": g.size(), "responses": g.n_responses.sum(),
                        "error_%": 100 * g.n_errors.sum() / g.n_responses.sum(),
                        "error_%_no_help": 100 * (g.n_errors.sum() - g.n_help.sum()) /
                                           (g.n_responses.sum() - g.n_help.sum()),
                        "median_rt": g.median_rt_correct.median()})
    return out

wt = {y: wave_table(waves[y]) for y in YEARS}
display(wt["ES_2025"].unstack("group")[["facts", "error_%", "median_rt"]].round(2))

# %%
def wave_figure(stem):
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True)
    order = [(a, w) for a in range(8, 16) for w in ("T1", "T2", "T3")]
    xpos = {k: i for i, k in enumerate(order)}
    for y, ls, mk in [("ES_2025", "-", "o"), ("ES_2024", ":", "s")]:
        t = wt[y].reset_index()
        for g in GROUPS:
            d = t[t.group.eq(g)].copy(); d["x"] = [xpos[(a, w)] for a, w in zip(d.course_age, d.wave)]
            d = d.sort_values("x")
            col = GCOL[g] if g != "Retrieval" else "#8A8A8A"
            for ax, v in [(axes[0], "error_%"), (axes[1], "median_rt")]:
                for _, seg in d.groupby("course_age"):              # join waves inside an age only
                    ax.plot(seg.x, seg[v], ls=ls, marker=mk, color=col, lw=2, markersize=6,
                            label=f"{ {'0-facts': 'Facts with 0', '1-facts': 'Facts with 1', 'Retrieval': 'Other facts'}[g]} ({y})"
                            if _ == d.course_age.min() else None)
    for a in range(8, 16):
        for ax in axes:
            ax.axvspan(3 * (a - 8) - .5, 3 * (a - 8) + 2.5, color="#F2F2F2" if a % 2 else "white", zorder=0)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([f"{w}" for a, w in order], fontsize=FONT_MED - 5)
    for a in range(8, 16):
        axes[1].text(3 * (a - 8) + 1, -.16, f"Age {a}", transform=axes[1].get_xaxis_transform(),
                     ha="center", fontsize=FONT_MED - 2)
    axes[0].set_ylabel("Error rate (%)", fontsize=FONT_MED)
    axes[1].set_ylabel("Median RT of the facts (s)", fontsize=FONT_MED)
    axes[0].legend(frameon=False, fontsize=FONT_MED - 6, ncol=2)
    for ax in axes: sns.despine(ax=ax)
    fig.tight_layout(); paint_it_black(axes); save(fig, stem)

wave_figure("error_rt_by_age_and_wave_zero_one")


# %% [markdown]
# ## 3. What the wrong answer is: 1-facts next to 0-facts
# Same data and filters as `error_types.ipynb` (DECISIONS #14-#17), ES_2025, ages 9-15 (the ages where
# both tables are administered in both orders). The 1-facts are 1×n and n×1 with n = 2-9. 0×1 / 1×0 belong
# to both rules and are reported apart. The classes (DECISIONS #44):
#
# | 0-facts (correct 0)                              | 1-facts (correct n)                          |
# |--------------------------------------------------|----------------------------------------------|
# | `= n`  copies the OTHER operand (= a + b too)    | `= 1`  copies the OTHER (rule) operand        |
# | `= 1`                                            | `= n + 1`  additive (a + b)                   |
# | other single digit                               | `= 0`  the zero rule applied                   |
# | ≥ 10                                             | anything else                                 |
#
# Why "copies the other operand": the correct answer of BOTH rules is one of the two operands (0 for
# n×0, n for n×1). The typical 0-fact error returns the wrong one of the two. On a 0-fact that answer is
# also the additive one (0 + n = n) and the 1-rule one (treating 0 like 1), so it cannot be split there;
# on a 1-fact the three are different answers (1, n + 1, n), which is what makes the 1-facts informative.

# %%
ans = cq("answers_by_fact_age.sql", year=YEAR)
ans = ans[ans.groupby(["course_age", "operation"]).n_responses.transform("sum").ge(30)].copy()
a, b, x = ans.left_factor, ans.right_factor, ans.answer
ans["group"] = group_of(a, b)
ans["is_error"] = ans.statement_result.ne("Correct")
ans["answered"] = ans.answer_kind.eq("int")
ans["direction"] = np.select([a.eq(0) & b.gt(0), b.eq(0) & a.gt(0), a.eq(1) & b.gt(1), b.eq(1) & a.gt(1)],
                             ["0×n", "n×0", "1×n", "n×1"], default="-")
ans["n"] = np.where(ans.group.eq("0-facts"), np.where(a.eq(0), b, a), np.where(a.eq(1), b, a))   # non-rule operand
T0 = ["= n (other operand)", "= 1", "Other digit", "≥ 10"]
T1 = ["= 1 (other operand)", "= n+1 (additive)", "= 0 (zero rule)", "Other"]
ans["error_type"] = np.where(
    ans.group.eq("0-facts"),
    np.select([~ans.answered, x.eq(a + b), x.eq(1), x.between(2, 9)], ["No answer"] + T0[:3], default=T0[3]),
    np.select([~ans.answered, x.eq(1), x.eq(a + b), x.eq(0)], ["No answer"] + T1[:3], default=T1[3]))
rule = ans[ans.direction.ne("-") & ans.course_age.ge(9)]
rule = rule[~(rule.group.eq("0-facts") & rule.n.eq(1))]          # 0×1 / 1×0: both rules at once

def share(d, by):
    e = d[d.is_error]
    cnt = lambda m: e[m].groupby(by).n_responses.sum()
    out = pd.DataFrame({"responses": d.groupby(by).n_responses.sum(),
                        "errors": e.groupby(by).n_responses.sum(),
                        "answered_errors": cnt(e.answered),
                        "copy_other": cnt(e.error_type.str.contains("other operand")),
                        "additive_1facts": cnt(e.error_type.eq("= n+1 (additive)")),
                        "zero_rule_on_1": cnt(e.error_type.eq("= 0 (zero rule)")),
                        "one_on_0": cnt(e.group.eq("0-facts") & e.error_type.eq("= 1"))}).fillna(0)
    out["error_%"] = 100 * out.errors / out.responses
    out["copy_other_%_resp"] = 100 * out.copy_other / out.responses
    out["copy_other_%_err"] = 100 * out.copy_other / out.answered_errors
    out["additive_%_err"] = 100 * out.additive_1facts / out.answered_errors
    return out

display(share(rule, ["group"]).round(2))
display(share(rule, ["direction"]).round(2))
display(share(rule, ["group", "course_age"]).round(2))
both = ans[ans.operation.isin(["0×1", "1×0"]) & ans.is_error & ans.answered & ans.course_age.ge(9)]
print("0×1 and 1×0 (both rules): % of their answered errors by answer")
display((both.groupby("answer").n_responses.sum() / both.n_responses.sum() * 100).round(1).head(4))

# %%
def mix(d, types):
    m = (d[d.is_error & d.answered].pivot_table(index=["direction", "course_age"], columns="error_type",
                                                values="n_responses", aggfunc="sum")
         .reindex(columns=types).fillna(0))
    return 100 * m.div(m.sum(axis=1), axis=0)

def error_type_figure(stem):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for row, (grp, types, dirs) in enumerate([("0-facts", T0, ["0×n", "n×0"]), ("1-facts", T1, ["1×n", "n×1"])]):
        m = mix(rule[rule.group.eq(grp)], types)
        for ax, dname in zip(axes[row], dirs):
            mm = m.loc[dname]; bottom = np.zeros(len(mm))
            for col, color in zip(types, C_TYPES):
                ax.bar(mm.index.astype(str), mm[col], bottom=bottom, color=color, label=col, width=.72)
                bottom += mm[col].to_numpy()
            ax.set_title(dname, fontsize=FONT_MED, color="black"); ax.set_ylim(0, 100); sns.despine(ax=ax)
        axes[row, 1].legend(frameon=False, fontsize=FONT_MED - 4, loc="upper left", bbox_to_anchor=(1, 1))
    fig.supxlabel("Age", fontsize=FONT_BIG, color="black")
    fig.supylabel("% of answered errors", fontsize=FONT_BIG, color="black")
    fig.tight_layout(); paint_it_black(axes); save(fig, stem)

def copy_figure(stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    cols = {"0×n": plt.cm.viridis(.05), "n×0": plt.cm.viridis(.32), "1×n": plt.cm.viridis(.62), "n×1": plt.cm.viridis(.88)}
    labs = {"0×n": "0×n answered n", "n×0": "n×0 answered n", "1×n": "1×n answered 1", "n×1": "n×1 answered 1"}
    for ax, by, xl in [(axes[0], "course_age", "Age"), (axes[1], "n", "Non-rule operand n")]:
        tab = share(rule, ["direction", by])
        for dname in labs:
            g = tab.loc[dname]; k, n = g.copy_other.to_numpy(), g.responses.to_numpy()
            lo, hi = [100 * v for v in wilson01(k, n)]; p = 100 * k / n
            ax.errorbar(g.index, p, yerr=[p - lo, hi - p], fmt="o-" if dname in ("0×n", "n×0") else "s--",
                        capsize=3, markersize=6, lw=2, color=cols[dname], label=labs[dname])
        ax.set_xlabel(xl, fontsize=FONT_BIG); ax.set_ylim(0, None); sns.despine(ax=ax)
    axes[0].set_ylabel("% of all responses", fontsize=FONT_BIG)
    axes[0].legend(frameon=False, fontsize=FONT_MED - 3)
    axes[1].text(.97, .95, "Ages 9-15 pooled", transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=FONT_MED - 2)
    fig.tight_layout(); paint_it_black(axes); save(fig, stem)

error_type_figure("error_types_zero_one_facts_by_age_ES_2025")
copy_figure("operand_copy_error_share_zero_one_ES_2025")


# %% [markdown]
# ## 4. RT distribution by operand order: 1×n vs n×1 next to 0×n vs n×0
# Same population and test as `rt_zero_direction.ipynb` (DECISIONS #18-#21): correct responses only,
# one median RT per student x age x direction, Wilcoxon signed-rank on the within-student difference
# (rule operand second minus rule operand first), Holm over the ages. Ages 9-15 for both tables (age 8
# has no 0-facts and no n×1). The new line is the **0 vs 1 contrast inside the same students**: for every
# student who has both orders of both tables at an age, the difference of their two order effects.
# Source: queries/rt_by_fact_correct.sql (already cached; a superset of rt_zero_facts_correct.sql).

# %%
def load_rt_pairs(year):
    cells = cq("rt_error_by_fact_age.sql", year=year)
    d = cq("rt_by_fact_correct.sql", year=year)
    d = d[pd.MultiIndex.from_arrays([d.course_age, d.operation]).isin(set(zip(cells.course_age, cells.operation)))]
    a, b = d.left_factor, d.right_factor
    d = d.assign(direction=np.select([a.eq(0), b.eq(0), a.eq(1), b.eq(1)], ["0×n", "n×0", "1×n", "n×1"], "-"))
    d = d[d.direction.ne("-") & d.course_age.ge(9)].copy()
    d["table"] = np.where(d.direction.str.contains("0"), "0", "1")
    d["second"] = d.direction.isin(["n×0", "n×1"]).astype(int)
    return d

def order_tests(d):
    w = d.groupby(["table", "course_age", "student_uuid", "second"]).rt_seconds.median().unstack("second").dropna()
    w["diff"] = w[1] - w[0]
    rows = []
    for (tab, age), g in w.groupby(level=["table", "course_age"]):
        rows.append({"table": tab, "course_age": age, **signed_rank(g["diff"].to_numpy()),
                     "med_first": g[0].median(), "med_second": g[1].median()})
    out = pd.DataFrame(rows).set_index(["table", "course_age"])
    out["p_holm"] = np.nan
    for tab in ("0", "1"):
        out.loc[tab, "p_holm"] = holm(out.loc[tab, "p"].to_numpy())
    out["sig"] = [stars(p) for p in out.p_holm]
    # 0 vs 1 inside the same students
    dd = w["diff"].unstack("table").dropna()
    rows = []
    for age, g in dd.groupby(level="course_age"):
        r = signed_rank((g["0"] - g["1"]).to_numpy())
        rows.append({"course_age": age, **r})
    allr = signed_rank((dd["0"] - dd["1"]).to_numpy())
    vs = pd.DataFrame(rows).set_index("course_age")
    vs["p_holm"] = holm(vs.p.to_numpy()); vs["sig"] = [stars(p) for p in vs.p_holm]
    return out, vs, allr, w

rtp, rt_order, rt_vs, rt_vs_all = {}, {}, {}, {}
for y in YEARS:
    rtp[y] = load_rt_pairs(y)
    rt_order[y], rt_vs[y], rt_vs_all[y], _ = order_tests(rtp[y])
    print(y, "| order effect = Mdn(rule second - rule first), s")
    display(rt_order[y][["students", "med_first", "med_second", "med_diff", "ci_lo", "ci_hi", "pct_slower",
                         "r_rb", "p_holm", "sig"]].round(3))
    print(y, "| 0 vs 1 inside the same students: Mdn(order effect on 0 - order effect on 1)")
    display(rt_vs[y][["students", "med_diff", "ci_lo", "ci_hi", "r_rb", "p", "p_holm", "sig"]].round(3))
    r = rt_vs_all[y]
    print(f"  all ages: n = {r['students']:,} student x age, Mdn {1000 * r['med_diff']:+.0f} ms "
          f"[{1000 * r['ci_lo']:+.0f}, {1000 * r['ci_hi']:+.0f}], r_rb = {r['r_rb']:+.3f}, p {fmt(r['p'])}")

# %%
def rt_one_figure(stem, year=YEAR):
    d, o = rtp[year], rt_order[year]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [2.3, 1]})
    one = d[d.table.eq("1")]
    sns.violinplot(data=one, x="course_age", y="rt_seconds", hue="direction", hue_order=["1×n", "n×1"],
                   split=True, gap=.06, inner="quartile", cut=0, bw_adjust=.55, density_norm="width",
                   gridsize=1000, linewidth=1, palette={"1×n": C_DIR_ONE[0], "n×1": C_DIR_ONE[1]}, ax=ax)
    ax.set_ylim(0, 4)
    n_resp = one.groupby("course_age").size()
    ax.set_xticks(range(len(n_resp))); ax.set_xticklabels([f"{a}\n{compact(n)}" for a, n in n_resp.items()])
    ax.tick_params(axis="x", labelsize=FONT_MED - 1)
    ax.set_xlabel("Age / correct responses", fontsize=FONT_BIG); ax.set_ylabel("Response time (s)", fontsize=FONT_BIG)
    ax.legend(frameon=False, fontsize=FONT_MED, loc="upper left", title=None, ncol=2)
    ax2.axhline(0, color="#B8B8B8", lw=1, zorder=1)
    for tab, dx, col, lab in [("0", -.15, C_ZERO, "n×0 − 0×n"), ("1", .15, C_ONE, "n×1 − 1×n")]:
        t = o.loc[tab]; ages = t.index.to_numpy() + dx
        ax2.errorbar(ages, 1000 * t.med_diff, yerr=1000 * np.vstack([t.med_diff - t.ci_lo, t.ci_hi - t.med_diff]),
                     fmt="o", capsize=3, markersize=6, color=col, ecolor="black", elinewidth=1, label=lab,
                     markeredgecolor="black", markeredgewidth=.5)
    ax2.set_xticks(range(9, 16)); ax2.set_xlabel("Age", fontsize=FONT_BIG)
    ax2.set_ylabel("Median within-student\norder effect (ms)", fontsize=FONT_MED)
    ax2.legend(frameon=False, fontsize=FONT_MED - 3, loc="upper right")
    ax2.axvspan(8.5, 9.5, color="#F2F2F2", zorder=-1)
    ax2.text(9, .5, "n×1 only\nin wave T3", transform=ax2.get_xaxis_transform(), ha="center", fontsize=FONT_MED - 7)
    for a_, letter in [(ax, "A"), (ax2, "B")]:
        a_.text(-.02, 1.06, letter, transform=a_.transAxes, fontweight="bold", fontsize=FONT_BIG, va="top",
                ha="right", color="black"); sns.despine(ax=a_)
    fig.tight_layout(); paint_it_black([ax, ax2]); save(fig, stem)

rt_one_figure(f"rt_violin_one_direction_by_age_{YEAR}")


# %% [markdown]
# ## 5. The order effect at trial level: 0 and 1 in one model
# The design of `rt_order_robustness.ipynb` (DECISIONS #29-#33), extended: every kept trial of both rule
# tables (0×n / n×0 with n = 1-9 and 1×n / n×1 with n = 2-9), log RT on the correct ones and accuracy on
# all, student and PAIR fixed effects (a pair = table + non-rule operand, so the order contrast is always
# within the same two operands), SEs clustered by student. `second` = the rule operand comes second.
# The 0 vs 1 difference is the `second:one` interaction, estimated inside the same students.
# Age 9 is excluded from the main model: at age 9 the n×1 direction (and all 0-facts) appear only in the
# third wave while 1×n is tested all year, so the 1-table order contrast at age 9 is confounded with
# the time of year (section 2). It is shown separately.
# Query for the 1-facts: queries/rt_one_facts_trials.sql (new, same filters as rt_zero_facts_trials.sql).

# %%
from pyfixest.estimation import feols

def load_trials(year=YEAR):
    cells = cq("rt_error_by_fact_age.sql", year=year)
    keep = set(zip(cells.course_age, cells.operation))
    z = cq("rt_zero_facts_trials.sql", year=year); o = cq("rt_one_facts_trials.sql", year=year)
    t = pd.concat([z.assign(one=0), o.assign(one=1)], ignore_index=True)
    t = t[pd.MultiIndex.from_arrays([t.course_age, t.operation]).isin(keep)]
    t = t[~(t.left_factor.eq(0) & t.right_factor.eq(0))].copy()              # 0×0 out (#15)
    rule_op = np.where(t.one.eq(1), 1, 0)
    t["second"] = (t.right_factor.values == rule_op).astype(float)
    t["n"] = np.where(t.left_factor.values == rule_op, t.right_factor, t.left_factor)
    t["pair"] = t.one.astype(str) + "_" + t.n.astype(str)
    t["correct"] = t.statement_result.eq("Correct").astype(float)
    t["age_c"] = t.course_age - 12.0
    t["one"] = t.one.astype(float)
    return t[t.course_age.ge(9)]

tr = load_trials()
trc = tr[tr.correct.eq(1)].assign(log_rt=lambda x: np.log(x.rt_seconds))
print(tr.groupby(["one", "course_age"]).agg(trials=("correct", "size"), students=("student_uuid", "nunique"),
                                            accuracy=("correct", "mean")).round(4).to_string())

def to_ms(beta, base):
    return 1000 * base * (np.exp(beta) - 1)

main_rt = trc[trc.course_age.ge(10)]; main_acc = tr[tr.course_age.ge(10)]
m_rt = feols("log_rt ~ second + second:one + second:age_c + second:one:age_c | student_uuid + pair",
             data=main_rt, vcov={"CRV1": "student_uuid"})
m_acc = feols("correct ~ second + second:one + second:age_c + second:one:age_c | student_uuid + pair",
              data=main_acc, vcov={"CRV1": "student_uuid"})
print(f"\n[log RT, ages 10-15, age centred at 12] n = {m_rt._N:,}")
print(m_rt.tidy().round(5).to_string())
print(f"\n[accuracy LPM, ages 10-15] n = {m_acc._N:,}")
print(m_acc.tidy().round(5).to_string())

# replication: the same two models in ES_2024 (ages 10-15; ES_2024 has no wave T2, ages 12-15 only T3)
tr24 = load_trials("ES_2024"); tr24c = tr24[tr24.correct.eq(1)].assign(log_rt=lambda x: np.log(x.rt_seconds))
rep = {}
for name, f, d in [("log RT", "log_rt ~ second + second:one + second:age_c + second:one:age_c", tr24c),
                   ("accuracy", "correct ~ second + second:one + second:age_c + second:one:age_c", tr24)]:
    r = feols(f + " | student_uuid + pair", data=d[d.course_age.ge(10)], vcov={"CRV1": "student_uuid"})
    print(f"\n[ES_2024 replication, {name}, ages 10-15] n = {r._N:,}")
    print(r.tidy().round(5).to_string()); rep[name] = r

# %%
def by_age(table):
    rows = {}
    for age in range(9, 16):
        c = trc[trc.one.eq(table) & trc.course_age.eq(age)]
        a = tr[tr.one.eq(table) & tr.course_age.eq(age)]
        r = feols("log_rt ~ second | student_uuid + pair", data=c, vcov={"CRV1": "student_uuid"}).tidy().loc["second"]
        q = feols("correct ~ second | student_uuid + pair", data=a, vcov={"CRV1": "student_uuid"}).tidy().loc["second"]
        base = c.loc[c.second.eq(0), "rt_seconds"].median()
        rows[age] = {"students": c.student_uuid.nunique(), "ms": to_ms(r["Estimate"], base),
                     "ms_lo": to_ms(r["2.5%"], base), "ms_hi": to_ms(r["97.5%"], base), "p_rt": r["Pr(>|t|)"],
                     "acc_pp": 100 * q["Estimate"], "acc_lo": 100 * q["2.5%"], "acc_hi": 100 * q["97.5%"],
                     "p_acc": q["Pr(>|t|)"]}
    return pd.DataFrame(rows).T.rename_axis("course_age")

fe_age = {"0": by_age(0), "1": by_age(1)}
for k, v in fe_age.items():
    print(f"{k}-table, trial FE per age (rule operand second minus first)")
    display(v.round({"ms": 1, "ms_lo": 1, "ms_hi": 1, "p_rt": 5, "acc_pp": 2, "acc_lo": 2, "acc_hi": 2, "p_acc": 5}))

# %%
def fe_figure(stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for tab, dx, col, lab in [("0", -.13, C_ZERO, "Facts with 0 (n×0 − 0×n)"), ("1", .13, C_ONE, "Facts with 1 (n×1 − 1×n)")]:
        t = fe_age[tab]; x = t.index.to_numpy() + dx
        for ax, v, lo, hi in [(axes[0], "ms", "ms_lo", "ms_hi"), (axes[1], "acc_pp", "acc_lo", "acc_hi")]:
            ax.errorbar(x, t[v], yerr=np.vstack([t[v] - t[lo], t[hi] - t[v]]), fmt="o", capsize=3,
                        markersize=6, color=col, ecolor="black", elinewidth=1, markeredgecolor="black",
                        markeredgewidth=.5, label=lab)
    for ax, yl in [(axes[0], "Order effect on RT (ms)"), (axes[1], "Order effect on accuracy (points)")]:
        ax.axhline(0, color="#B8B8B8", lw=1, zorder=0); ax.set_xticks(range(9, 16))
        ax.set_xlabel("Age", fontsize=FONT_BIG); ax.set_ylabel(yl, fontsize=FONT_MED)
        ax.axvspan(8.5, 9.5, color="#F2F2F2", zorder=-1); sns.despine(ax=ax)
    axes[1].text(9, .03, "n×1 only\nin wave T3", transform=axes[1].get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=FONT_MED - 6)
    axes[0].legend(frameon=False, fontsize=FONT_MED - 4, loc="upper right")
    for ax, letter in zip(axes, "AB"):
        ax.text(-.02, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=FONT_BIG, va="top",
                ha="right", color="black")
    fig.tight_layout(); paint_it_black(axes); save(fig, stem)

fe_figure(f"order_effect_fe_zero_one_by_age_{YEAR}")


# %% [markdown]
# ## 6. Sequential consistency: ×1 next to ×0, by age and by transfer condition
# Design of `sequential_zero_consistency.ipynb` (DECISIONS #34-#41): each student's first two kept trials
# of a problem type at a given age, one pair per student. That notebook already compared the types pooled
# over ages (tetrachoric .70 ×0, .42 ×1, .41 retrieval). Added here: the comparison **at every age**, the
# four transfer conditions for ×1, and the robustness variant pending in #36 (×0 without 0×1 / 1×0).
# The cached pull of queries/sequential_trials.sql is read directly: the .sql was re-commented after
# that pull, so its text hash no longer matches, but the population is the one of #34-#37 (3,271,108 rows).

# %%
seq = pd.read_parquet(ROOT / "cache" / "sequential_trials__d107039f1884987b.parquet")
cells = cq("rt_error_by_fact_age.sql", year=YEAR)
seq["operation"] = seq.left_factor.astype(str) + "×" + seq.right_factor.astype(str)
seq = seq[pd.MultiIndex.from_arrays([seq.course_age, seq.operation]).isin(set(zip(cells.course_age, cells.operation)))]
seq = seq[seq.ptype.isin(["zero", "one", "retrieval"]) & ~seq.operation.eq("1×1")].copy()
seq["correct"] = seq.statement_result.eq("Correct").astype(int)

def first_pairs(t, ptype):
    t = t[t.ptype.eq(ptype)].sort_values(["student_uuid", "course_age", "pos_all"])
    t = t.assign(k=t.groupby(["student_uuid", "course_age"]).cumcount())
    a = t[t.k.eq(0)].set_index(["student_uuid", "course_age"]); b = t[t.k.eq(1)].set_index(["student_uuid", "course_age"])
    p = a.join(b, lsuffix="1", rsuffix="2", how="inner")
    if ptype != "retrieval":
        r = 0 if ptype == "zero" else 1
        for i in "12":
            p[f"n{i}"] = np.where(p[f"left_factor{i}"] == r, p[f"right_factor{i}"], p[f"left_factor{i}"])
            p[f"ord{i}"] = (p[f"right_factor{i}"] == r).astype(int)
        p["cond"] = np.where(p.n1 == p.n2, np.where(p.ord1 == p.ord2, "A same n, same order", "B same n, switched"),
                             np.where(p.ord1 == p.ord2, "C diff n, same order", "D diff n, switched"))
    return p.reset_index()

pairs = {pt: first_pairs(seq, pt) for pt in ["zero", "one", "retrieval"]}
seq_no01 = seq[~seq.operation.isin(["0×1", "1×0"])]
pairs["zero (no 0×1, 1×0)"] = first_pairs(seq_no01, "zero")

rows = []
for pt, p in pairs.items():
    for age, g in [("all", p)] + list(p.groupby("course_age")):
        r, lo, hi = tetrachoric_ci(g.correct1, g.correct2, rng, n_boot=100)
        lo = np.nan if lo < -.9 else lo          # a degenerate resample (an empty error cell) is not a bound
        rows.append({"type": pt, "course_age": age, **contrast_2x2(g.correct1, g.correct2),
                     "tetrachoric": r, "tc_lo": lo, "tc_hi": hi})
seq_tab = pd.DataFrame(rows).set_index(["type", "course_age"])
display(seq_tab[["students", "pC2C1", "pC2E1", "diff", "OR", "tetrachoric", "tc_lo", "tc_hi"]].round(3))

# %%
cond_rows = []
for pt in ["zero", "one"]:
    for cond, g in pairs[pt].groupby("cond"):
        cond_rows.append({"type": pt, "cond": cond, **contrast_2x2(g.correct1, g.correct2),
                          "tetrachoric": tetrachoric(g.correct1, g.correct2)})
cond_tab = pd.DataFrame(cond_rows).set_index(["type", "cond"])
display(cond_tab[["students", "pC2C1", "pC2E1", "diff", "OR", "tetrachoric"]].round(3))

# %%
def seq_figure(stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})
    cols = {"zero": C_ZERO, "one": C_ONE, "retrieval": "#8A8A8A"}
    labs = {"zero": "×0", "one": "×1", "retrieval": "Retrieval (2-9)"}
    for k, (pt, dx) in enumerate([("zero", -.18), ("one", 0), ("retrieval", .18)]):
        t = seq_tab.loc[pt].drop("all"); x = t.index.astype(int).to_numpy() + dx
        axes[0].errorbar(x, t.tetrachoric, yerr=np.vstack([t.tetrachoric - t.tc_lo, t.tc_hi - t.tetrachoric]),
                         fmt="o", capsize=3, markersize=6, color=cols[pt], ecolor="black", elinewidth=1,
                         markeredgecolor="black", markeredgewidth=.5, label=labs[pt])
    axes[0].set_xticks(range(8, 16)); axes[0].set_xlabel("Age", fontsize=FONT_BIG)
    axes[0].set_ylabel("Tetrachoric r, trial 1 → trial 2", fontsize=FONT_MED); axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=FONT_MED - 3, loc="lower left")
    conds = sorted(cond_tab.index.get_level_values(1).unique())
    y = np.arange(len(conds))[::-1]
    for pt, dy in [("zero", .15), ("one", -.15)]:
        t = cond_tab.loc[pt].reindex(conds)
        axes[1].plot(t.tetrachoric, y + dy, "o", color=cols[pt], markersize=8, markeredgecolor="black",
                     markeredgewidth=.5, label=labs[pt])
    axes[1].set_yticks(y); axes[1].set_yticklabels(conds, fontsize=FONT_MED - 4)
    axes[1].set_xlabel("Tetrachoric r", fontsize=FONT_MED); axes[1].set_xlim(0, 1)
    for ax, letter in zip(axes, "AB"):
        ax.text(-.02, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=FONT_BIG, va="top",
                ha="right", color="black"); sns.despine(ax=ax)
    fig.tight_layout(); paint_it_black(axes); save(fig, stem)

seq_figure(f"seq_consistency_zero_one_by_age_{YEAR}")



# %% [markdown]
# ## 7. Data checks behind reviewer questions 1-3 (DECISIONS #45-#46)
# **Q1, multiplication signs.** Every operation string in A998, digits replaced by `d`
# (queries/check_operation_glyphs.sql, all years and countries).

# %%
glyphs = cached_query(Q / "check_operation_glyphs.sql", cache_dir=ROOT / "cache")
display(glyphs.groupby("pattern").n_statements.sum().sort_values(ascending=False).to_frame())
assert set(glyphs.pattern) <= {"d×d", "d·d", "d:d", "dd:d", "d÷d", "dd÷d"}, "an unexpected sign"
print("A998 uses only × and · for multiplication and : or ÷ for division; nothing else exists, so the "
      "'^[0-9][×·][0-9]$' filter drops no multiplication.")

# %% [markdown]
# **Q2-Q3, Help and mis-recorded answers** (queries/check_user_answer_quality.sql, ES_2025, after the
# filters of #1-#7).

# %%
qa = cached_query(Q / "check_user_answer_quality.sql", cache_dir=ROOT / "cache")
display(qa.set_index(["statement_result", "grp"]))
tot = qa.groupby("statement_result").n.sum()
print(f"Help = {tot['Help']:,} of {tot.sum():,} responses ({100 * tot['Help'] / tot.sum():.2f}%); "
      f"0-facts {qa.query('statement_result==\"Help\" and grp==\"0-facts\"').n.sum()}, "
      f"1-facts {qa.query('statement_result==\"Help\" and grp==\"1-facts\"').n.sum()}")
print(f"repeated-digit answers (4444, 555 ...): {qa.repeated_digit_3plus.sum()} in total")

# %%
# How much the Help-as-error choice (#8) moves the error rates of the three groups, per age
fx = facts[YEAR]
g = fx.groupby(["course_age", "group"])
help_tab = pd.DataFrame({"responses": g.n_responses.sum(), "help": g.n_help.sum(),
                         "error_%": 100 * g.n_errors.sum() / g.n_responses.sum(),
                         "error_%_help_excluded": 100 * (g.n_errors.sum() - g.n_help.sum()) /
                                                  (g.n_responses.sum() - g.n_help.sum())})
help_tab["shift_pp"] = help_tab["error_%"] - help_tab["error_%_help_excluded"]
display(help_tab.round(3).unstack("group")[["error_%", "error_%_help_excluded", "shift_pp"]])

# the accuracy order model of section 5 with Help responses dropped
m_acc_nohelp = feols("correct ~ second + second:one + second:age_c + second:one:age_c | student_uuid + pair",
                     data=main_acc[main_acc.statement_result.ne("Help")], vcov={"CRV1": "student_uuid"})
print(m_acc_nohelp.tidy().round(5).to_string())


# %% [markdown]
# ## 8. Sanity checks

# %%
assert set(facts[YEAR].group) == set(GROUPS)
assert not facts[YEAR][facts[YEAR].course_age.eq(8)].group.eq("0-facts").any()       # #12
assert (facts[YEAR][facts[YEAR].course_age.eq(8) & facts[YEAR].group.eq("1-facts")].left_factor == 1).all()
for y in YEARS:   # the wave table adds up to the pooled table
    w_ = waves[y].groupby(["course_age", "operation"]).n_responses.sum()
    f_ = facts[y].set_index(["course_age", "operation"]).n_responses
    common = w_.index.intersection(f_.index)
    assert (w_[common] <= f_[common]).all()            # cells under 30 in a wave are dropped, never added
assert rule.direction.isin(["0×n", "n×0", "1×n", "n×1"]).all() and rule.course_age.min() == 9
assert np.isclose(share(rule, ["group"]).copy_other.sum(),
                  rule[rule.is_error & rule.error_type.str.contains("other operand")].n_responses.sum())
assert tr.second.isin([0, 1]).all() and set(tr.n) <= set(range(1, 10))
assert not tr[tr.one.eq(1)].n.eq(1).any()                                            # 1×1 never pulled
assert (pairs["zero"].student_uuid.value_counts().max() <= 8)                        # one pair per student x age
print("all checks passed")
