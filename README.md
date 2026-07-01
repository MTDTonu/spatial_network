Spatial Predictor

A field-geometric machine learning tool. No loss function. No gradient descent. No iteration.

Samples carry causal charge — they generate fields that bend the space around them. Fields superpose and interfere. Class boundaries emerge where opposing charges cancel. place() deposits samples. read() probes the field.

---

Quick Start

    pip install numpy scipy scikit-learn

    from spatial_network import SpatialNetwork
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    sn = SpatialNetwork()                    # zero parameters — everything auto-emerges
    sn.place(X_train, y_train)               # place samples → space bends
    y_pred = sn.read(X_test)                 # read the field → predict
    proba = sn.predict_proba(X_test)         # probability output (softmax)
    
    print(f"Accuracy: {(y_pred == y_test).mean():.1%}")   # 93.3%

---

Core API

    # Classification
    sn = SpatialNetwork()
    sn.place(X_train, y_train)
    y_pred = sn.read(X_test)
    proba = sn.predict_proba(X_test)
    
    # Regression
    sn = SpatialNetwork(regression_method='local_linear')
    sn.place(X_train, y_train)
    y_pred = sn.read(X_test)                 # R² up to 0.99+
    
    # Incremental learning
    sn.place_one(x_new, y_new)               # plug and play — no retraining
    
    # Inspection
    sn.causal_strength_                       # causal charge q_i per sample
    sn.dim_weights_                           # dimension weights w_k
    sn.field_radius_                          # field radius r_i

---

How It Works

Each sample s_i produces a field everywhere in space:

    Φ_i(x) = q_i / (1 + ||x − x_i||² / λ_i²)

- q_i — causal charge. Emerges from local same-class density. A confident "this is a cat" produces a stronger field than a hesitant "this might be a cat."
- λ_i — characteristic length. The distance at which field strength decays to half. Sparsely distributed samples get larger λ; dense clusters get smaller λ.
- w_k — dimension weights. Computed from the F-statistic (between-class variance / within-class variance). Weak dimensions are suppressed; strong dimensions are amplified up to 100×. Causal dimensions emerge automatically.

The total field at any point is the superposition of all sample fields. Same-class fields amplify each other (causal resonance). Different-class fields cancel out (causal null zones). Class boundaries are not drawn — they emerge naturally at interference nulls.

The field never reaches zero — inverse-quadratic decay guarantees every sample influences the entire space, no matter how far. Generalization is not trained. It is built into the geometry.

---

When to Use

  ✅ Strengths                            	⚠️ Limitations                          
  Small-sample high-dimensional (n < d)  	Large datasets (n > 5000)               
  Ultra-high dimensions (d > 1000)       	Non-linear topology (concentric circles)
  Zero hyperparameter tuning             	Feature interactions (x₁ · x₂)          
  Online / incremental learning          	Image / NLP tasks                       
  Regression with n ≫ d                  	                                        
  Interpretability (q, λ, w all readable)	                                        

---

License

MIT
