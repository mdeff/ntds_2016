NB  = $(sort $(wildcard toolkit/*.ipynb))
NB += $(sort $(wildcard algorithms/*.ipynb))

all: test run

test:
	python check_install.py

run: $(NB)
	grip README.md --export README.html

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)
	rm -f README.html
	rm -f toolkit/subset.html

cleanall: clean
	rm -fr data

install:
	pip install --upgrade pip
	pip install -r requirements.txt

readme:
	grip README.md

.PHONY: all test run $(NB) clean install readme
