# Rețele neuronale de la zero

Codul laboratorului [Rețele neuronale de la zero](https://inteligentartificiala.ro/laboratoare/retele-neuronale-de-la-zero/prezentare/)
de pe InteligentArtificiala.ro, scris pas cu pas în videourile canalului
[InteligentArtificiala.ro](https://www.youtube.com/@inteligentArtificiala): un sistem de diferențiere automată pentru
numere, grafurile lui computaționale desenate cu Graphviz, o rețea neuronală antrenată pe două semilune de puncte și o
rețea care recunoaște cifrele MNIST, scrisă cu NumPy și cu PyTorch.

## Rulare

Pentru a instala pachetele, rulați comenzile de mai jos. Programul `dot` al Graphviz se instalează separat, cu
managerul de pachete al sistemului (`brew install graphviz`, `sudo apt install graphviz`), iar PyTorch, pentru ultimele
scripturi, cu `pip install torch`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pașii din videouri

Fiecare pas al unui episod are o etichetă git, ca să puteți urmări codul pas cu pas: `git checkout ep001/step-2`.

Sistemul de diferențiere automată urmează ideea și structura din [micrograd](https://github.com/karpathy/micrograd),
scris de Andrej Karpathy (licență MIT).

## Licență

Codul este publicat sub licența MIT: fișierul [`LICENSE`](LICENSE).
