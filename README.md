# Spatial Network

**A geometric framework for causal inference via spacetime curvature.**

No loss function. No gradient descent. No training.  
Samples curve causal spacetime. Predictions are field readings.

📄 **Paper**: [spatial_network_paper_en.md](spatial_network_paper_en.md)  

---

## Quick Start

```bash
git clone https://github.com/MTDTonu/SpatialNetwork.git
cd SpatialNetwork
pip install -r requirements.txt
python main.py
```

Output:
```
=======================================================
  空间网络 — Iris (鸢尾花)
=======================================================
  训练: 105样本  测试: 45样本  特征: 4维  类别: 3
  准确率: 84.44%
  ε (背景涨落): 1.42  (自动从数据离散度涌现)
  ...
```

---

## Core API

```python
from spatial_network import SpatialNetwork

sn = SpatialNetwork()                 # zero parameters — everything auto-emerges
sn.place(X_train, y_train)            # place samples → curve spacetime
y_pred = sn.read(X_test)              # read field values → predict
sn.place_one(x_new, y_new)            # incremental placement
bs = sn.boundary_score(X_test)        # boundary detection (0=center, 1=edge)
```

---

## File Structure

| File | Description |
|------|-------------|
| `spatial_network.py` | Core implementation: `SpatialNetwork` |
| `grid_sn.py` | Grid variant: `GridSpatialNetwork` (2D field visualization + fast lookup) |
| `main.py` | Demo: Iris dataset |
| `example.py` | 4 usage scenarios (classification, grid, high-D, online) |
| `benchmark_full.py` | Full comparison: Spatial Network vs Logistic Regression vs Decision Tree |
| `requirements.txt` | Dependencies: numpy, scipy, scikit-learn |

---

## Key Concepts

- **Causal charge** `q_i` — each sample's influence weight, emerges from local same-class density
- **Field equation** `Φ(x) = q / (1 + (r/λ)²)` — inverse-quadratic decay, never reaches zero
- **Field boundary** `r_i = λ√(q/ε−1)` — where field drops below background fluctuation
- **Cross-class interference** — same-class +w, different-class −w/(K−1); boundaries emerge at nulls
- **Zero randomness** — 100% deterministic; same data always produces the same field

---

## Citation

```bibtex
@article{chen2025spatial,
  title={Spatial Network: A Geometric Framework for Causal Inference via Spacetime Curvature},
  author={Chen, Jiahe},
  year={2025},
  url={https://github.com/MTDTonu/SpatialNetwork}
}
```

---

## License

MIT
