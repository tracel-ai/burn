import torch
from torch.nn import Linear
from torch.optim import Adam

# Define a Linear layer with given weights and biases
def given_linear_layer(weights: torch.Tensor, biases: torch.Tensor) -> Linear:
	out_features, in_features = weights.shape
	layer = Linear(in_features, out_features, bias=True)

	layer.weight.data = torch.FloatTensor(weights)
	layer.bias.data = torch.FloatTensor(biases)

	return layer


def test_adamw():
	torch.set_printoptions(precision=6)

	LEARNING_RATE = 0.01

	prior_weights = torch.tensor([
			[-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
			[0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
			[-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
			[-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
			[0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
			[-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
		])
	prior_bias = torch.tensor([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130])

	linear = given_linear_layer(
		torch.clone(prior_weights),
		torch.clone(prior_bias)
	)

	x_1 = torch.tensor([
		[0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
		[0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
	], requires_grad=True)

	x_2 = torch.tensor([
		[0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
		[0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
	], requires_grad=True)

	optimizer = Adam(
		linear.parameters(),
		lr=LEARNING_RATE,
		eps=1e-6,
		betas=(0.9, 0.999),
		weight_decay=0.5
	)

	# First forward-backward pass and optimization step
	optimizer.zero_grad()
	y = linear(x_1)
	y.backward(torch.ones_like(y))
	optimizer.step()

	# Second forward-backward pass and optimization step
	optimizer.zero_grad()
	y = linear(x_2)
	y.backward(torch.ones_like(y))
	optimizer.step()

	# Print updated state
	print(f"\nUpdated layer weights: \n{linear.weight.data}")
	print(f"\nUpdated layer bias: \n{linear.bias.data}")

	# Print difference between prior and updated weights
	print(f"\nWeight difference: \n{linear.weight.data - prior_weights}")
	print(f"\nBias difference: \n{linear.bias.data - prior_bias}")
