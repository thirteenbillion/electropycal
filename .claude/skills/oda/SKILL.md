---
name: oda
description: Run a data analysis or make a plot against a lab dataset held in cloud storage, in chat, using the project's own published analysis library — or decide the request is too heavy for chat and launch it as an unattended cloud job instead. Use this whenever someone asks for a quick stat, trend, figure, or extraction from their lab or study data; attaches a featureset, parquet, CSV or instrument export and asks what it shows; asks for a publication-ready or paper-ready figure; asks to re-extract or refresh a dataset; asks whether an analysis should run here or be launched as a sweep/job/workflow; or pastes a cloud-storage link and asks to analyse what is behind it. Also use it when someone asks for a plot the analysis library has no preset for — combining the library with general tools is the intended route, not a workaround.
---

# Running lab analyses in chat

Someone wants a number or a figure out of their data, now, in the conversation.
This skill covers getting the data in, using the project's own library rather than
reimplementing it, keeping the statistics honest, making output publication-ready,
and recognising when a request belongs in an unattended job instead.

## 1. Route the request first

Ask one question before anything else: **is this a chat-sized request?**

**Chat-sized** — a stat, a trend, a figure, a filter, a comparison, a single
session's raw traces, a re-extraction of a handful of sessions. Seconds to a couple
of minutes. Do it here.

**Not chat-sized** — a full model sweep, cross-validated selection over many
conditions, anything the project's own cost model puts in the tens of minutes or
beyond. Chat sandboxes are single-core and time-limited, so this fails slowly and
confusingly rather than quickly.

For the second kind, **say so immediately** and offer the unattended route rather
than starting and timing out. Name the actual cost if the project records one.
Then describe the launch: usually a manually-triggered CI workflow, with the
profile or parameters to pick, and where results will land.

A useful third case: the person wants a *preview* of an expensive analysis. Run a
cheap slice in chat — one target, one seed, a subsample — label it plainly as a
preview, and offer the full run as a job.

## 2. Get the data in

Routes, cheapest first:

**Attached file.** Best route in chat. Files attached to the conversation land on
the sandbox filesystem directly and cost nothing in context. A few megabytes is
fine. This is the primary route on mobile.

**Already on disk.** In an agentic coding session, the repo checkout may already
hold the dataset or a demo tree. Look before asking. Pull first if the data is
committed and refreshed by a job.

**A published or shared file store.** If the project syncs a current dataset to a
cloud folder, the person downloads from there and attaches. Say which file.

**Never pull bulk data through a storage connector.** Connector download paths
return file contents through the model's context. Fine for one small file,
unworkable for a batch, and it silently consumes an enormous amount of context.
Ask for an attachment instead.

**A storage URL is not a filesystem path.** If someone gives a share link, say so
plainly and give them the two options: download and attach, or mount the folder if
they are in a notebook environment that supports it. Do not pass a URL to code
expecting a path.

## 3. Use the project's library

Install the published package rather than reimplementing its logic:

```
pip install <package>
```

Verified to work in chat code execution. Prefer the published release over a
working tree unless the person asks otherwise, and **record which you used** —
results from an unreleased tree and a release must never be confusable.

The library exists because ingestion, quality gating, feature extraction and
cross-validation discipline are easy to get subtly wrong. Use its functions for
those. Do not hand-roll a parser for an instrument export the library already
reads.

**But do not treat a missing preset as a limitation.** These libraries return
dataframes and arrays. A plot with no preset is a few lines of matplotlib, and
combining the library with seaborn, statsmodels, scipy or anything else is the
intended use. Say what you are combining and why, rather than implying the
library cannot do it.

## 4. Two things to protect

**The statistics.** Ad-hoc analysis is where carefully-built discipline quietly
breaks. Before reporting a number, check:

- Any statistic used by a fitted model — an imputation value, a scaling factor, a
  threshold — must come from training rows only, never from the whole set.
- Global parameters derived from the data as a whole (a reference grid, a
  baseline origin, a band, a normalisation) make results depend on which subset is
  present. If the project persists these, load them rather than recomputing. If it
  does not, say the result is subset-dependent.
- Time-series data usually needs forward-chained splits, not random ones.
- A descriptive plot needs none of this. A predictive claim needs all of it. Be
  explicit about which you produced.

**The provenance.** Say which dataset version, which release, and — when it
matters — which commit produced a number. A figure with no provenance is a figure
nobody can defend later.

## 5. Make output publication-ready by default

Most scientific libraries ship a style helper and a colour-blind-safe palette that
nothing actually invokes. Look for one and call it. If none exists, apply sane
defaults: a readable font size, no default colour cycle for categorical data, axis
labels with units, and a colour-blind-safe palette.

**Always write figures to a file**, not only to a display call. In a headless
sandbox `plt.show()` is a silent no-op — the code completes and produces nothing.
Save, then present the file. Close figures after saving so a multi-figure pass
does not accumulate.

Offer a vector format alongside a raster one when the figure is headed for a
paper.

## 6. Ask, don't guess — but only about what matters

Ask when the answer changes the result and cannot be inferred. Typical:

- **Which subset.** Device type, cohort, condition, arm. The commonest silent
  error is including a group the person did not mean, especially when the dataset
  labels individuals but not their type.
- **Which baseline or reference.** Relative to what?
- **Pooled or per-unit.** Pooling across channels, subjects or replicates changes
  the answer and often the conclusion.
- **Which dataset version**, if the project has both a frozen and a refreshed one.

Do not ask about formatting, file names, or anything you can state an assumption
about and proceed. One round of questions, not an interrogation. When you assume,
say so inline.

## 7. Report like a colleague, not a tool

- Lead with the answer, then the caveats.
- Mark measured versus inferred, and give ranges rather than false precision.
- Say when a result is weaker than expected, or when a negative finding is the
  actual finding rather than a bug. Do not quietly present the more flattering
  reading.
- Say what you could not check.
- If the person's framing contains a wrong premise — a column that is actually
  derived, a "raw data" question answerable from a processed file, a remembered
  duration contradicted by the cost model — say so before answering.

## Project-specific facts

This section is the only part that is project-specific. A project using this skill
should carry its own dataset caveats alongside its data — in a README beside the
published file, or in the repo's own context file — so that whoever holds the data
holds the caveats too. Read those before analysing.

Look for, and read if present:
- the repo's context file (`CLAUDE.md` or equivalent)
- a cost model, before making any claim about how long something will take
- a README beside the published dataset, for the version, counts, and the
  identifier-to-group mapping
- the run record for a pinned analysis, before re-extracting anything
