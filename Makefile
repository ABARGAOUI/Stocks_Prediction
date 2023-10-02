VENV=venv
PYTHON=python
BASE_URL=http://localhost:8000

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

clear:
	rm -rf $(VENV)

run: install
	$(VENV)/bin/uvicorn main:app --host localhost --port 8002 --reload
