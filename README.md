Tema 2 ASC - Glodariu Ana

Implementarea in CUDA a logicii din cpu_miner
--
Paralelizarea cautarii nonce-ului, folosind CUDA Threads l-am facut astfel:
- mi-am creat atatea CUDA threads cate nonce-uri posibile aveam de verificat (de la 1 la MAX_NONCE inclusiv)
    - mi-am definit 256 ca dimensiunea unui bloc de thread-uri, iar numarul de blocuri din grid l-am calculat
    astfel incat sa am >= MAX_NONCE Thread-uri: (MAX_NONCE + block_size - 1) / block_size
    - astfel ca fiecare thread va avea un index global: uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; si va verifica
    daca acel index poate fi un nonce valid ca in implementarea pe cpu
- parametrii dati kernelului i-am alocat cu cudaMallocManaged folosind memoria unificata pentru a nu mai face copieri de pe host->device si invers
- in kernel mi-am creeat variabile locale pentru block_content si block_hash, pentru ca fiecare thread sa lucreze cu copia lui
si sa faca operatii pe ele fara a fi vizibile celorlalte thread-uri
- mi-am declarat o variabila globala pentru nonce pentru a fi vizibila in toate blocurile, iar atunci cand este gasita o valoare valida, restul thread-urilor nu mai cauta un nonce valid
- deoarece nonce e o variabila globala, pentru a ne asigura ca nu au loc race conditions -> ma folosesc de functia atomica atomicCAS care face operatia de compare si swap intr-un mod atomic; daca valoarea variabilei globale global_nonce este 0 atunci va fi actualizata cu valoarea index-ului thread-ului : atomicCAS((unsigned long long*)&global_nonce, (unsigned long long)0, (unsigned long long)index); astfel ca primul thread care gaseste un nonce valid va fi si cel care actualizeaza valoarea hash-ului calculat (necesar la printare)
- folosind operatii atomice, nu va fi o problema daca se intampla ca 2 thread-uri sa gaseasca simultan un nonce valid
- in plus, deoarece doar thread-ul care gaseste nonce-ul valid este si cel care actualizeaza hash-ul, nu se poate intampla ca hash-ul sa nu fie cel calculat cu nonce-ul respectiv
- restul threadurilor nu vor mai putea actualiza global_nonce (pentru ca valoarea variabilei nu mai este 0, deci functia atomicCAS va pastra mereu prima valoarea diferita de 0 atribuita variabilei globale)
- odata ce nonce-ul este gasit, restul thread-urilor nu vor mai face calculele pentru gasirea altui nonce valid
- folosesc cudaDeviceSynchronize() pentru a ma asigura ca toate operatiile de pe gpu s-au terminat inainte de a accesa nonce-ul si hash-ul pe host
- folosesc cudaMemcpyFromSymbol() pentru a copia memoria din variabila globala de pe gpu pe host

Rezultate rulare locala:
- primul nonce valid gasit este: 515800
- block_hash: 00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee; cu dificultate mai mica sau egala decat
0000099999999999999999999999999999999999999999999999999999999999
Rezultatele sunt identice cu cele de la rularea pe cpu, deoarece ambele cauta doar pana ce gasesc primul nonce valid, difera
doar timpul care e redus semnificativ.

