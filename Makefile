.PHONY: install data eda train evaluate sensitivity app test all
install:
	pip install -r requirements.txt && pip install -e .
data:
	python -m rental.data
eda:
	python -m rental.eda
train:
	python -m rental.train
evaluate:
	python -m rental.evaluate
sensitivity:
	python -m rental.sensitivity
app:
	streamlit run app/streamlit_app.py
test:
	pytest -q
all: data eda train evaluate sensitivity test
