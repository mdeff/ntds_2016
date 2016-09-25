NB  = $(wildcard toolkit/*.ipynb)
NB += $(wildcard algorithms/*.ipynb)

all: test run

run:
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=120 $(NB)

test:
	python3 check_install.py

install:
	pip3 install --upgrade pip
	pip3 install -r requirements.txt

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)
	rm -fr data

.PHONY: all run test install clean
