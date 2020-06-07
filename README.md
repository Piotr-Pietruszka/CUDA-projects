## CUDA-projects

# 1) Matrices
Zainicjalizowanie macierzy A i B o podanej wielkości losowymi liczbami typu zmiennoprzecinkowego, a następnie obliczenie macierzy: 
$C = AB$ 
$D = AA^T + BB^T + CC^T$ 

Obliczenia wykonywane są przy użyciu CPU oraz GPU, a następnie porównywane są różnice w wynikach i prędkość.

# 2) Pi
Obliczanie liczby $\pi$ przy pomocy wzoru Leibniza - rozwinięcie w szereg:
$\frac{\pi}{4}  = \frac{1}{1} - \frac{1}{3}  +\frac{1}{5} - \frac{1}{7} + \cdots$ 
Biorąc dużą liczbę elementów można zauważyć przyspieszenie wynikające z zrównoleglenia obliczeń na karcie graficznej. Wynik z CPU i GPU jest porównywany, podobnie jak czas obliczeń.
