from seq.transformer.synth import main


def test_transformer():
    # This might take a few mins, it's more of an integration test.
    decoded = main()
    assert (decoded.to('cpu').numpy().ravel() == [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1,  2,  3,  4,  5]).all()

