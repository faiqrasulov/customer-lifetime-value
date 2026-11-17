# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: return netting uses cancellations dated after the split; listed as a limitation.

## 2026-11-15

- First version: BG/NBD and Gamma-Gamma by maximum likelihood, holdout validation against baselines, CLV segments.
- Fixed: tied RFM customers now get the lowest score of their block.
- README: states up front that BG/NBD reduces to NBD on this dataset.
