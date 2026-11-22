# Convenience wrappers around reproduce.py (works with GNU make; on Windows use the python commands directly).
PY ?= python

.PHONY: all quick legacy test install dashboard

all:            ## full reproduction of results/
	$(PY) reproduce.py

quick:          ## smoke run (~3 min)
	$(PY) reproduce.py --quick

legacy:         ## full run plus the original 3-class pipeline
	$(PY) reproduce.py --legacy

test:
	$(PY) -m pytest -q

install:        ## exact versions used for the published results
	$(PY) -m pip install -r requirements-lock.txt

dashboard:
	$(PY) run_dashboard.py
