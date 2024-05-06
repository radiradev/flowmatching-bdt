from flow_matching_bdt.model import FlowMatchingBDT

def test_model():
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=1000, noise=0.1, random_state=0)
    model = FlowMatchingBDT()

    # train the model
    model.fit(data)

    # get new samples
    num_samples = 1000
    samples = model.predict(num_samples=num_samples)
    samples = samples.reshape(num_samples, 2)
    