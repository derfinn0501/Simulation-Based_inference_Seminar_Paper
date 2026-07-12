# Feature Quality Diagnostics

Timestamped diagnostics for synthetic-real feature mismatch.

Each run compares simulator-generated observed features against held-out
Mendeley features for each configured feature set. The diagnostics include
per-feature mismatch metrics, correlation mismatch, classifier two-sample tests,
PCA/optional UMAP domain-overlap plots, saved recovery metrics when available,
and a markdown report.
