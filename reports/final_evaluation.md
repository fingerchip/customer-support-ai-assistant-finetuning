# Final Evaluation: Base vs SFT vs DPO-Aligned Model

Comparing all three stages on the same 10 customer-support questions: the untuned base model,
the Stage 2 instruction-tuned (SFT) model, and the Stage 3 DPO-aligned final assistant. Full
raw generations are reproducible via `notebooks/generate_final_eval.ipynb`.

## Evaluation Criteria

Correctness, helpfulness, domain accuracy, safety, tone, clarity, hallucination reduction,
professional response quality.

| # | Question | Base Model Answer (excerpt) | SFT Model Answer (excerpt) | DPO Model Answer (excerpt) | Best Answer | Reason |
|---|---|---|---|---|---|---|
| 1 | How can I cancel my order? | Hallucinated API error + C# code snippet. | Calls out 1-800-555-0199 / Live Chat, gives numbered steps. | Opens with an empathetic acknowledgment, then gives numbered steps referencing the order section. | DPO | Same procedural correctness as SFT, but noticeably warmer, more professional opening tone. |
| 2 | What are your customer support hours? | "24/7" (wrong) + garbled timezone list. | Correct hours, but trails into a confusing fake "### Customer:" role-play fragment. | Correct hours + phone + website + email, ends cleanly with an offer to help further. | DPO | Same correct fact as SFT, but doesn't degrade into an incoherent role-play tangent. |
| 3 | How do I track my order? | Hallucinated fake company's contact info + repeated garbage. | Asks for order number, offers email updates, ends somewhat abruptly. | Gives a concrete "check the order page" answer, ends with a clean closing line. | DPO | More complete and actionable without an abrupt cutoff. |
| 4 | Can I get a refund if I'm not satisfied with the product? | Vague apology, no real answer. | Confirms refunds possible, asks for details, but leaks a stray "### Prompt 2" template artifact. | Confirms refunds possible, asks for details, and starts a concrete numbered "How to Receive a Refund" list. | DPO | Begins giving actual step-by-step guidance instead of only asking a clarifying question. |
| 5 | My product arrived damaged, what should I do? | Describes an outdated stamped-envelope postal process. | Empathizes but deflects the user to contact the *shipping company/vendor*. | Empathizes, asks for damage details/evidence, and offers to guide the user through replacement/claim directly. | DPO | Doesn't deflect responsibility elsewhere - this is the clearest example of DPO correcting a real quality regression introduced by SFT. |
| 6 | How can I request a replacement for a defective item? | Generic "email us within 7 days." | Asks for defect/order details, promises support. | Cites the real support hours/phone (1-800-555-0199) and anticipates likely follow-up questions (timing, availability, cost). | DPO | More domain-consistent (matches the company's actual documented contact info) and proactively helpful. |
| 7 | My payment failed but the amount was deducted. What should I do? | Hallucinated authentication-error JSON. | Apologizes, asks for order/transaction details, but cuts off mid-word ("I sincer"). | Explains the likely cause (transaction/payment method error) and asks a clear follow-up question, without cutting off. | DPO | More complete and informative; doesn't trail off mid-sentence like the SFT answer. |
| 8 | How long does a refund take to process? | Hallucinated Express.js code snippet. | Asks for purchase details to estimate; doesn't state a timeframe. | Asks for order details; frames the answer around when funds "arrive in your bank account." | DPO (slight) | Neither model states a hard number of days (a shared dataset limitation), but DPO's framing is marginally more concrete. |
| 9 | Can I change the delivery address after placing an order? | Short "yes, email us" answer. | Detailed step-by-step (login, Order History, edit, save), trails off slightly. | Nearly identical step-by-step, more consistent labeling, ends with a complete invitation to reach out. | DPO (slight) | Marginal refinement over an already-good SFT answer - polish rather than a fix. |
| 10 | How do I contact customer support? | Breaks character, talks about the company's own limited dev resources. | Lists contact channels, but includes an odd generic-looking Gmail address. | Lists the real phone number and Live Chat, consistent with previously stated hours, professional tone throughout. | DPO | More consistent and professional; avoids the slightly off-brand Gmail address suggestion. |

## Summary

Across all 10 questions, the DPO-aligned model was judged the best answer, or tied for best,
every time. The improvement from base -> SFT is dramatic and qualitative (fixes hallucinated,
off-topic, or simply wrong answers). The improvement from SFT -> DPO is smaller but consistent
and measurable: warmer/more professional tone, fewer abrupt or incoherent cutoffs, and -
notably in question 5 - a direct correction of a real quality regression (deflecting
responsibility to a third party) that instruction fine-tuning alone had introduced. This
matches the quantitative DPO training signal: the chosen-vs-rejected log-probability margin
improved by +87.6 (from -147.69 to -60.10) over the course of Stage 3 training (see
`reports/fine_tuning_explanation.md` and `discussions.md`).

One shared limitation across SFT and DPO: neither model reliably states a concrete refund
processing timeframe (question 8) - the underlying Bitext dataset doesn't consistently include
that fact, so it wasn't something either fine-tuning stage could have learned. This is a good
candidate for a future data-quality improvement (see README, Section 14).
