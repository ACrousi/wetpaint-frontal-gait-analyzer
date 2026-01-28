
class MockFeeder:
    def __init__(self, ldl_sigma):
        self.ldl_sigma = ldl_sigma

    def get_sigma(self, label_value):
        # Logic copied from coco.py for verification
        sigma = 1.0  # Default
        if isinstance(self.ldl_sigma, list):
            found_sigma = False
            for config in self.ldl_sigma:
                if isinstance(config, dict) and 'start' in config and 'end' in config:
                    if config['start'] <= label_value < config['end']:
                        sigma = float(config['sigma'])
                        found_sigma = True
                        break
            
            if not found_sigma:
                 # Check if it equals the max end of the last config (inclusive support for max)
                 last_config = self.ldl_sigma[-1]
                 if label_value == last_config.get('end'):
                     sigma = float(last_config['sigma'])
                 else:
                     pass
        else:
            sigma = float(self.ldl_sigma)
        return sigma

def test_sigma_logic():
    config = [
      {'start': 12, 'end': 18, 'sigma': 1.0},
      {'start': 18, 'end': 26, 'sigma': 2.0},
      {'start': 26, 'end': 35, 'sigma': 3.0}
    ]
    
    feeder = MockFeeder(config)
    
    # Test cases
    test_labels = [12, 15, 17.9, 18, 20, 25.9, 26, 30, 35]
    expected_sigmas = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    
    all_passed = True
    for label, expected in zip(test_labels, expected_sigmas):
        sigma = feeder.get_sigma(label)
        if sigma != expected:
            print(f"FAIL: Label {label} got sigma {sigma}, expected {expected}")
            all_passed = False
        else:
            print(f"PASS: Label {label} got sigma {sigma}")
    
    if all_passed:
        print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_sigma_logic()
