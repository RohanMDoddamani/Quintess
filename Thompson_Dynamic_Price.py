import numpy as np

prices = [10, 20, 30, 40]
n_prices = len(prices)

# Initialize alpha and beta for each price
alpha = np.ones(n_prices)
beta = np.ones(n_prices)

for t in range(1000):  # Simulate 1000 users
    sampled_rates = np.random.beta(alpha, beta)
    expected_revenues = [p * r for p, r in zip(prices, sampled_rates)]
    
    chosen_index = np.argmax(expected_revenues)
    chosen_price = prices[chosen_index]
    
    # Simulate user behavior (true conversion rates are unknown)
    # Let's assume true conv rates: [0.5, 0.3, 0.2, 0.1]
    true_rates = [0.5, 0.3, 0.2, 0.1]
    purchase = np.random.rand() < true_rates[chosen_index]
    
    # Update alpha/beta
    if purchase:
        alpha[chosen_index] += 1
    else:
        beta[chosen_index] += 1
