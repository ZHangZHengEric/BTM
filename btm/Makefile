CC=g++
CFLAGS=-c -pipe -O3 -I/usr/include/python3.8/ -fPIC
MODEL_OBJ=model.o infer.o
MODEL_LIB=btm.so
LDFLAGS=-lpython3 -lboost_python38 -lpython3.8 -lboost_numpy38
all:$(MODEL_LIB) 

$(MODEL_LIB):$(MODEL_OBJ)
	$(CC) -shared $(MODEL_OBJ) $(LDFLAGS) -o $@

model.o:model.h biterm.h doc.h
infer.o:infer.h doc.h

%.o:%.cpp
	$(CC) $(INCLUDE) $(CFLAGS) $< -o $@ 

clean:
	rm -rf $(MODEL_OBJ) $(MODEL_LIB)
