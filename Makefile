.PHONY: clean test
clean:
	rm -f Log_*.log
	rm -rf __pycache__

test:
	python -m unittest
	
