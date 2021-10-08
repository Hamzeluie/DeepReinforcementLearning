from keras_models.a2c import A2C_one_net, A2C_multi_net
# from keras_models.a3c import MasterAgent
# from torch_models.a3c import MasterAgent
# from torch_models.ddpg import DeepDeterministicPolicyGradient as ddpg
# from torch_models.a2c import A2C_one_net, A2C_multi_net
from keras_models.ddpg import DeepDeterministicPolicyGradient as ddpg
from utils.utils import Training, Evaluation, AsyncExperiment


if __name__ == "__main__":
    env_name = "CartPole-v0"
    agent = A2C_multi_net
    # Training(env_name, agent).train()
    Evaluation(env_name, agent).eval()