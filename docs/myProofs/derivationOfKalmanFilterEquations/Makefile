all: derivationKalmanFilterEqs.pdf

derivationKalmanFilterEqs.pdf: derivationKalmanFilterEqs.tex figLDS.tex conditionalDistForGaussianRVs.tex marginalOfLinearGaussianModel.tex
	pdflatex derivationKalmanFilterEqs.tex
	bibtex derivationKalmanFilterEqs.aux
	pdflatex derivationKalmanFilterEqs.tex
	pdflatex derivationKalmanFilterEqs.tex


