.PHONY: all watch todo clean

all: ## build main.pdf (XeLaTeX: Vietnamese in code listings needs Unicode fonts)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

watch: ## rebuild on every save
	latexmk -xelatex -pvc -interaction=nonstopmode main.tex

todo: ## \dy placeholders left, per chapter file
	@for f in sections/*.tex; do printf '%-32s %s\n' $$f $$(grep -o '\\dy{' $$f | wc -l); done

clean: ## remove build files
	latexmk -C
