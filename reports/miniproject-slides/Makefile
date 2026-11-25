.PHONY: all watch todo clean

all: ## build main.pdf
	latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

watch: ## rebuild on every save
	latexmk -xelatex -pvc -interaction=nonstopmode main.tex

todo: ## list placeholders left, per part
	@grep -c '\\todo{' main.tex parts/*.tex

clean: ## remove build files
	latexmk -C
