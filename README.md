L’objectiu d'aquest treball és desenvolupar un programa que compti repeticions correctes i vàlides d’exercicis utilitzant tècniques de visió per computador, és a dir: anàlisi d’imatge amb intel·ligència artificial. El tipus d’exercici es determinarà automàticament a través d’una xarxa  RNN (Recurrent Neural Networks), determinant el tipus de seqüència mitjançant DTW (Dynamic Type Warping). 
Una vegada determinat el tipus d’exercici, es faran servir algorismes de Human Pose Estimation per determinar quan cada repetició comença i finalitza. Això es podrà analitzar mitjançant la posició dels keypoints del Human Pose Estimation (colzes, espatlles, etc.) respecte a una referència. 

L’eina s’integrarà en una API accessible des d’una aplicació web desenvolupada amb un framework d’alt nivell (Vue, React, Angular, etc.). Els usuaris podran enviar vídeos per analitzar, i l’eina retornarà el tipus d’exercici detectat i els moments en què es comptabilitzen les repeticions. La web inclourà mesures bàsiques de seguretat, com l'autenticació d’usuaris.

-Sapiens (construcció de dataset LSTM): 'docker attach docker_repcount' -> 'conda activate sapiens_lite' -> 'predict_pose.sh'

-RepCount: 'docker attach docker_repcount_web