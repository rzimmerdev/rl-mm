# rl-mm
Optimization of a Market-Making strategy using Reinforcement Learning techniques

To replicate the virtual-environment, run:

```bash
conda env create -f environment.yml
```

or 

```bash
pip install -r requirements.txt
```

To use JAX with GPU, run:

```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

To run the code, run:

```bash
python main.py
```