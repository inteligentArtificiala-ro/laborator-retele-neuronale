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

- Episodul 1 (`ep001/step-1` ... `ep001/end`): clasa `Value` și metoda `backward()`, derivatele locale, verificarea
  cu diferențe finite, grafurile desenate cu Graphviz și aceeași expresie în PyTorch.
- Episodul 2 (`ep002/step-1` ... `ep002/end`): neuroni, straturi și o rețea cu mai multe straturi, antrenarea pe două
  semilune, apoi rețeaua pentru MNIST cu NumPy și cu PyTorch.

```bash
python diagrams.py      # diagramele grafurilor computaționale, în directorul imagini/
python moons.py         # rețeaua mică, antrenată pe două semilune de puncte
python mnist_numpy.py   # rețeaua pentru MNIST, scrisă cu NumPy
python mnist_torch.py   # aceeași rețea în PyTorch (după pip install torch)
```

Sistemul de diferențiere automată urmează ideea și structura din [micrograd](https://github.com/karpathy/micrograd),
scris de Andrej Karpathy (licență MIT).

## Licență

Codul este publicat sub licența MIT: fișierul [`LICENSE`](LICENSE).
