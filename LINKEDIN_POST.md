# LinkedIn — post + carousel

Post this as a **document (carousel) post**: attach `ThermoLogic_LinkedIn_Slides.pdf`
(10 square slides) and paste the caption below. Claims are matched to the measured
results — safe to post as-is.

---

## Caption (paste this)

AI models will break your rules with total confidence — and never tell you.
So I built a way to catch it, and fix it. 👇

Over a few evenings I put together **ThermoLogic** — a small neuro-symbolic
proof-of-concept that borrows an idea from physics to keep AI outputs logically
valid.

The idea: give every answer an **“energy.”** Obeying your rules costs zero energy;
breaking one costs a lot. Then you let the answer **roll downhill** to the nearest
valid version — no training data, just your rules.

Three things I found (swipe through the deck 👉):

→ A plain neural net can *match* on accuracy — but can’t tell when it’s wrong. The
energy score is a **label-free error detector**.
→ **Repair** works at test time: hand it a rule-breaking answer, it fixes it to the
nearest valid one.
→ **Deeper reasoning costs proportionally more compute** — truth propagates one
step at a time (a small original finding).

Honest framing: it’s a proof-of-concept that builds on prior work (Logic Tensor
Networks, DeepProbLog, Semantic Loss) — not an LLM upgrade or a “hallucination
solved” claim. It works today on **structured outputs with clear rules**: configs,
forms, tabular decisions.

🕹️ Try the live demo (no install): joperjoker.github.io/Project-ThermoLogic
💻 Code + technical report: github.com/joperjoker/Project-ThermoLogic

If your team wrestles with constraint-satisfaction on AI outputs, I’d love to
compare notes.

#AI #MachineLearning #NeuroSymbolic #PyTorch #AIReliability

---

## Slide deck contents (`ThermoLogic_LinkedIn_Slides.pdf`)

1. Hook — “AI breaks rules with total confidence.”
2. The problem — trained to sound right, not to be right.
3. The idea — reasoning as thermodynamics (energy landscape).
4. How it works — Score → Explain → Repair.
5. Live demo — the interactive console (screenshot).
6. Example — an invalid signup, auto-repaired.
7. Finding 1 — reliability, not accuracy (baselines chart).
8. Finding 2 — deeper reasoning = more compute (wave chart).
9. What this is (and isn’t) — honest scope.
10. CTA — demo / paper / code / `pip install`.

Rebuild the deck after edits: open `assets/carousel.html` in a browser and
Print → Save as PDF (paper size 1080×1080 px / “square”), or re-run the headless
Chromium `page.pdf()` step.

## Posting checklist

- [ ] Repo public + Pages live at joperjoker.github.io/Project-ThermoLogic/
- [ ] Create a **Document** post (not image) → upload `ThermoLogic_LinkedIn_Slides.pdf`
- [ ] Paste the caption above; confirm the first two lines land before “…more”
- [ ] First comment: drop the demo link again (links in comments help reach)
- [ ] Post weekday morning for your timezone
