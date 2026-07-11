# Base Model Evaluation

Testing the original, untuned `unsloth/tinyllama-bnb-4bit` (TinyLlama-1.1B) base model on 10
customer-support questions, before any fine-tuning. Full raw generations are reproducible via
`notebooks/generate_final_eval.ipynb`.

| # | Question | Base Model Answer (excerpt) | Problem |
|---|---|---|---|
| 1 | How can I cancel my order? | "`{ "error": { "code": "E00024", "message": "Product not found" } }` ... A: You are getting an error message... `var product = new Product(); product.id = 1;`..." | Completely off-topic - hallucinates an unrelated software/API error and C# code snippet instead of answering the question. |
| 2 | What are your customer support hours? | "We are open 24/7. ### Customers who can't reach you during business hours... 10:30 am - 5 pm PST ... 8:30 am - 1 pm ET ..." | Factually wrong (our actual hours are Mon-Fri, 9 AM-6 PM) and degenerates into a nonsensical list of overlapping timezones. |
| 3 | How do I track my order? | "...contact us at support@bizbuysell.com or our toll free customer service line 1-866-493-0507... ### Additional Information * Quantity: 0 ### Additional Information..." | Hallucinates a fake, unrelated company's contact details and then repeats a meaningless "Quantity: 0" block. |
| 4 | Can I get a refund if I'm not satisfied with the product? | "We are sorry that you feel that way. If there is anything else we can do to help, please let us know." | Vague and generic - never actually confirms whether a refund is possible or explains any process. |
| 5 | My product arrived damaged, what should I do? | "...Please provide us with... model number... A self-addressed, stamped envelope so we can send you a replacement item." | Coherent but describes an outdated postal-mail return process, not a modern online support flow. |
| 6 | How can I request a replacement for a defective item? | "Please send us an e-mail at [email]... within 7 days of delivery..." | Reasonably on-topic, but generic and not tied to any real company process (portal, order lookup, etc.). |
| 7 | My payment failed but the amount was deducted. What should I do? | "`{ "error": { "code": "InvalidCredential", "message": "The user has not been authenticated." }...}`" | Completely off-topic - returns a raw, unrelated authentication-error JSON payload instead of addressing the payment issue. |
| 8 | How long does a refund take to process? | "Returns: - Errors: * 400 Bad Request... ```javascript const express = require('express');...```" | Completely off-topic - hallucinates an Express.js code snippet; never answers the question. |
| 9 | Can I change the delivery address after placing an order? | "Yes, you can change the delivery address as long as there is a valid reason. Please email to [email]..." | Short and plausible, but generic and references a placeholder-looking email address. |
| 10 | How do I contact customer support? | "We are a small company with limited resources... I am very busy right now, so my response time is going to be pretty slow." | Breaks character entirely - sounds like a developer making excuses, not a support assistant, and never actually gives contact info. |

## Summary

The base model frequently hallucinates content entirely unrelated to customer support (fake
API error JSON, unrelated code snippets, a different company's contact details), gives no
consistent or correct facts about this company's actual policies (support hours, contact
channels), and occasionally breaks character. This establishes the "before" baseline that
Stages 1-3 are measured against.
