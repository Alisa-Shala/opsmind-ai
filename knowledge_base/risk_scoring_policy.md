# Invoice Risk Scoring Policy

OpsMind AI classifies invoice risk using machine learning and business rules.

Important risk indicators include:
- Potential duplicate invoice.
- Missing purchase order.
- High invoice amount.
- Vendor delay history.
- SLA breach.
- Approval delay.
- Failed three-way matching.

A low-risk invoice usually has a valid purchase order, no duplicate flag, no SLA breach, normal invoice amount, and no vendor delay history.

A medium-risk invoice may have some uncertainty, such as vendor delay history, higher amount, or additional validation requirement.

A high-risk invoice usually has one or more strong risk signals such as duplicate invoice flag, missing purchase order, high invoice amount, SLA breach, approval delay, or failed three-way matching.

High-risk invoices should be reviewed manually before payment.