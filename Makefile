source := $(wildcard *.cpp)
submit_files := $(wildcard submi*.cpp)
source := $(filter-out $(submit_files), $(source))
headers := $(wildcard *.h)

targets := $(source) $(headers)

debug: $(targets)
	g++ -o main -Wall -g -DLOCAL=1 -lm -std=c++11 $(source)

release: $(targets)
	g++ -o main -Wall -lm -O2 -std=c++11 -pipe $(source)

submit: $(targets)
	echo "[*] Removing old submit files..."
	rm -rf $(submit_files)
	echo "[*] Merging all files..."
	codingame-merge -m main.cpp -o submit.cpp -e venv
	echo "[*] Removing remaining includes..."
	sed 's/#include\(.*\)$$/include\1/' submit.cpp > submit2.cpp
	echo "[*] Expanding macros..."
	/usr/local/bin/g++-8 -E -P submit2.cpp > submit3.cpp
	echo "[*] Adding back includes..."
	sed 's/include\(.*\)$$/#include\1/' submit3.cpp > submit4.cpp
	#/usr/local/bin/sunifdef --define DEBUG submit.cpp
	echo "[*] Sizes:"
	ls -la submit*.cpp
	echo "[*] Trying to build submit4.cpp..."
	g++ -o submit -Wall -lm -O2 -std=c++11 -pipe submit4.cpp