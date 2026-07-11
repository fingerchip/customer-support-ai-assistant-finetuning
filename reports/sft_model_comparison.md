# Base Model vs Instruction Fine-Tuned (SFT) Model Comparison

Same 10 questions, comparing the untuned base model against the model after Stage 1
(non-instruction) + Stage 2 (instruction fine-tuning). Full raw generations are reproducible
via `notebooks/generate_final_eval.ipynb`.

## Evaluation Criteria

Correctness, domain accuracy, clarity, safety, helpfulness, less generic response, better
domain-specific behavior.

| # | Question | Base Model Answer (excerpt) | SFT Model Answer (excerpt) | Which is Better? | Reason |
|---|---|---|---|---|---|
| 1 | How can I cancel my order? | Hallucinated API error + C# code, totally off-topic. | "To cancel your order, please follow these steps: 1. Get in touch with us by calling 1-800-555-0199 or through the Live Chat... 2. ...select the option to cancel. 3. ...locate the order..." | SFT | On-topic, procedural, references real (learned) contact channel; base never addresses the question at all. |
| 2 | What are your customer support hours? | "We are open 24/7" + nonsensical timezone list. | "Our customer support hours are Monday to Friday, 9 AM to 6 PM." | SFT | Factually correct and matches the company's actual documented hours; base is simply wrong. |
| 3 | How do I track my order? | Fake unrelated company's contact info + repeated garbage block. | Asks for the order number/details and offers real-time email updates. | SFT | Coherent, on-topic, doesn't invent a fake company's support line. |
| 4 | Can I get a refund if I'm not satisfied with the product? | "We are sorry that you feel that way... let us know." (vague) | Confirms refunds are possible, asks for product name/order number to proceed. | SFT | Actually answers the question (yes, here's how) instead of a non-committal apology. |
| 5 | My product arrived damaged, what should I do? | Describes a stamped-envelope postal return process. | Empathizes, but suggests contacting the *shipping company/vendor* rather than handling it directly. | Base (mixed) | SFT's answer mildly deflects responsibility instead of offering a direct replacement path; base is outdated but at least stays within the company's own process. Flagged as a case DPO later improves. |
| 6 | How can I request a replacement for a defective item? | Generic "email us within 7 days" instruction. | Asks for defect details/order number, promises a resolution, more conversational and reassuring. | SFT | More helpful and interactive; still domain-appropriate. |
| 7 | My payment failed but the amount was deducted. What should I do? | Hallucinated authentication-error JSON, totally off-topic. | Apologizes, asks for order/transaction details to investigate. | SFT | Directly addresses the actual scenario; base doesn't engage with the question at all. |
| 8 | How long does a refund take to process? | Hallucinated Express.js code snippet, totally off-topic. | Asks for purchase/transaction details to give an estimate (doesn't state a fixed number of days). | SFT | At least engages with the real question, even though it doesn't give a concrete timeframe - a shared limitation noted for future improvement. |
| 9 | Can I change the delivery address after placing an order? | Short "yes, email us" answer. | Detailed step-by-step: log in, go to Order History, locate order, edit address, save. | SFT | More actionable and specific; matches how a real account-based order flow would work. |
| 10 | How do I contact customer support? | Breaks character, talks about the company's limited dev resources and slow personal response time. | Lists email, phone (1-800-555-0199), and chat as contact channels. | SFT | Base fails to answer at all and breaks persona; SFT gives concrete, usable contact information. |

## Summary

Instruction fine-tuning (on top of Stage 1's domain-adapted base) fixed the base model's two
biggest failure modes: (1) hallucinating completely unrelated content (API errors, code
snippets, a different company's contact info) and (2) failing to state this company's actual
policies correctly (support hours). The one case where the SFT model's answer was arguably
weaker (#5, deflecting a damaged-product issue to the shipping company) is exactly the kind of
"less helpful, less domain-appropriate" response the Stage 3 DPO preference data was built to
push the model away from - see `reports/final_evaluation.md`.
