# Makefile for Image Processing Interactive Session

PYTHON = python3
PIP = pip

# Target for installing dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

# Target for running the script
run:
	@echo "Running the script..."
	$(PYTHON) image_processing_interactive.py

# Target for checking dependencies
check_dependencies:
	@echo "Checking dependencies..."
	$(PIP) check || (echo "Some dependencies are not satisfied. Please run 'make install' to install them."; exit 1)

# Target for updating dependencies
update_dependencies:
	@echo "Updating dependencies..."
	$(PIP) install --upgrade -r requirements.txt

# Target for cleaning up
clean:
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete

# Target for help
help:
	@echo "Usage:"
	@echo "  make install             Install dependencies listed in requirements.txt"
	@echo "  make run                 Run the Image Processing Interactive Session script"
	@echo "  make check_dependencies  Check if dependencies are satisfied"
	@echo "  make update_dependencies Update dependencies to the latest versions"
	@echo "  make clean               Clean up the workspace"
	@echo "  make help                Show this help message"

.PHONY: install run check_dependencies update_dependencies clean help