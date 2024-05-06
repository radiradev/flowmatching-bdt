from flowmatching_bdt import FlowMatchingBDT

def test_model():
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
    model = FlowMatchingBDT()

    # train the model
    model.fit(data)

    # get new samples
    num_samples = 1000
    samples = model.predict(num_samples=num_samples)
    print(samples.shape)

def test_condtional():
    import numpy as np
    from sklearn.datasets import make_moons
    data, labels = make_moons(n_samples=1000, noise=0.1, random_state=42)
    model = FlowMatchingBDT()

    # train the model
    model.fit(data, conditions=labels)

    # get new samples
    num_samples = 1000
    conditions = np.ones((num_samples, 1))
    samples = model.predict(num_samples=num_samples, conditions=conditions)


if __name__ == "__main__":
    test_model()
    test_condtional()
