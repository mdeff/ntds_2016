NB  = $(sort $(wildcard toolkit/*.ipynb))
NB += $(sort $(wildcard algorithms/*.ipynb))

all: test run

run:
	grip README.md --export README.html
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $(NB)

test:
	python check_install.py

install:
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)
	rm -f README.html
	rm -f toolkit/subset.html
	rm -fr data

readme:
	grip README.md
	#pandoc -f markdown_github README.md -s -o README.html

.PHONY: all run test install clean readme
